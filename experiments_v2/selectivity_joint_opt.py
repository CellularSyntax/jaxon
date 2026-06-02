"""Joint waveform amplitude + electrode position optimisation (Phase 5B).

Demonstrates that gradient descent through the biophysical cable equation
can simultaneously discover optimal electrode placements and stimulation
amplitudes, with no engineering prior.

Pipeline
--------
1. Build synthetic nerve (N_FIBERS MRG fibers, ring cuff).
2. Rect-only optimisation (warm-start baseline).
3. Joint optimisation: amps [K] + contact_xyz_um [K, 3] jointly via Adam.
4. Compare SI: baseline → rect → joint.
5. Save figure showing contact position displacement.

Run locally (GPU recommended):
    python experiments_v2/selectivity_joint_opt.py
"""
from __future__ import annotations

import sys
import pathlib
import json
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jaxfibers.nerve.geometry import make_hussain_style_nerve
from jaxfibers.stim.multichannel_field import (
    make_ring_cuff_positions, precompute_ve_unit, build_fiber_arrays,
)
from jaxfibers.stim.batch_solve import stack_fiber_statics, initial_states_batch
from jaxfibers.optim.losses import activation_proxy_batch, selectivity_index
from jaxfibers.optim.optimizer import run_rect_optimization, run_joint_optimization
from jaxfibers.fibers.mrg import section_centers_um
from experiments_v2.utils import ensure_dir, save_json, plot_seed_summary

OUT = ensure_dir(ROOT / "outputs" / "selectivity_joint_opt")

# ─────────────────────────────────────────────── experiment parameters ────────
SEED            = 0
N_FIBERS        = 200          # FD joint pass: 25×200=5000 effective fibers per step
FIBER_DIAMETER_UM = 5.7        # Hussain 2024 methodology: single representative
                                # MRG diameter per fascicle.  Mixed diameters
                                # confuse LBFGS and aren't needed for the
                                # selectivity optimisation (Hussain assumed all
                                # fibres of a given diameter in a fascicle share
                                # the same threshold).
N_NODES         = 21           # n_comp = 20×11 + 1 = 221 (~23 mm fiber for D=10 µm)
N_CONTACTS      = 6
NERVE_RADIUS_UM = 500.0
CUFF_RADIUS_UM  = 1500.0
TARGET_FRACTION = 0.30
DT              = 0.005        # ms
T_STOP          = 3.0          # ms; PW=0.1 + DELAY=1.0 + slowest-MRG
                                # propagation ≈ 2.1 ms, so 3 ms is enough.
DELAY_MS        = 1.0
PW_MS           = 0.1
N_OPT_RECT      = 100          # Adam plateaus well before 200 iters; 100
N_OPT_JOINT     = 100          # is plenty for the 200-fiber joint problem.
SIGMA_S_M       = 0.3


def main():
    print(f"[joint-opt] seed={SEED}, N_FIBERS={N_FIBERS}, N_NODES={N_NODES}", flush=True)

    # ── nerve + cuff (Hussain-style multi-fascicle anatomy) ───────────────────
    N_FASCICLES_JOINT = 8
    n_per_fasc = max(1, N_FIBERS // N_FASCICLES_JOINT)
    nerve = make_hussain_style_nerve(
        n_fascicles=N_FASCICLES_JOINT,
        n_fibers_per_fascicle=n_per_fasc,
        nerve_radius_um=NERVE_RADIUS_UM,
        divider_angle_deg=0.0,
        diameters=[FIBER_DIAMETER_UM],
        seed=SEED,
    )
    n_tgt = int(nerve.target_mask.sum())
    n_total = nerve.n_fibers
    n_tgt_fasc = sum(1 for f in nerve.fascicles if f.is_target)
    n_off_fasc = len(nerve.fascicles) - n_tgt_fasc
    print(f"[joint-opt] Nerve: {n_total} fibres in {len(nerve.fascicles)} "
          f"fascicles ({n_tgt_fasc} target / {n_off_fasc} off-target, "
          f"{n_per_fasc} fibres/fasc)  targets={n_tgt}/{n_total}  "
          f"D={FIBER_DIAMETER_UM} µm", flush=True)

    N_STEPS = int(T_STOP / DT)
    t_grid  = (np.arange(N_STEPS) + 1) * DT

    contact_xyz = make_ring_cuff_positions(
        n_contacts=N_CONTACTS,
        cuff_radius_um=CUFF_RADIUS_UM,
        cuff_z_um=0.0,
    )

    # ── field precomputation (numpy, for rect warm-start) ─────────────────────
    print("[joint-opt] Precomputing fields ...", flush=True)
    Ve_unit, node_indices, geoms = precompute_ve_unit(
        nerve_geom=nerve, n_nodes=N_NODES, contact_xyz_um=contact_xyz,
        sigma_S_m=SIGMA_S_M,
    )
    # Centre cuff axially at fiber midpoint
    mid_comp = node_indices[0, len(node_indices[0]) // 2]
    mid_z    = float(np.array(section_centers_um(geoms[0]))[mid_comp])
    contact_xyz[:, 2] = mid_z
    Ve_unit, node_indices, geoms = precompute_ve_unit(
        nerve_geom=nerve, n_nodes=N_NODES, contact_xyz_um=contact_xyz,
        sigma_S_m=SIGMA_S_M,
    )

    # ── solver statics ────────────────────────────────────────────────────────
    print("[joint-opt] Building solver statics ...", flush=True)
    fs_batch = stack_fiber_statics(geoms, DT)
    s0_batch = initial_states_batch(geoms)

    pulse_mask = np.where(
        (t_grid >= DELAY_MS) & (t_grid < DELAY_MS + PW_MS), 1.0, 0.0
    ).astype(np.float64)
    pulse_j    = jnp.asarray(pulse_mask)
    Ve_unit_j  = jnp.asarray(Ve_unit, dtype=jnp.float64)

    # ── baseline SI ───────────────────────────────────────────────────────────
    m_max_zero = activation_proxy_batch(
        jnp.zeros((N_FIBERS, geoms[0].n_comp), dtype=jnp.float64),
        jnp.asarray(node_indices, dtype=jnp.int32),
    )
    si_baseline = selectivity_index(np.array(m_max_zero), nerve.target_mask)
    print(f"[joint-opt] Baseline SI = {si_baseline:+.3f}", flush=True)

    # ── rect optimisation (warm-start) ────────────────────────────────────────
    print(f"[joint-opt] Rect optimisation ({N_OPT_RECT} iters) ...", flush=True)
    t0 = time.time()
    rect_res = run_rect_optimization(
        fiber_statics_batch=fs_batch,
        state0_batch=s0_batch,
        Ve_unit=Ve_unit_j,
        pulse_mask=pulse_j,
        node_indices=node_indices,
        target_mask=nerve.target_mask,
        dt=DT,
        n_steps=N_OPT_RECT,
        # Default amp_init_mA=-0.4 was tuned for the mixed-diameter regime
        # (5.7..16 µm) where large fibers fire at low amps and the gradient
        # is non-zero at init.  With single D=5.7 µm, -0.4 mA is well below
        # threshold (~-1.5 mA at this cuff distance) and the activation
        # proxy is flat → SI stays pinned at 0.000.  Init at -1.5 mA puts
        # the closest contact near threshold so gradients are informative
        # from iter 0.  Widen the clip to (-3, 3) for headroom in case
        # LBFGS / Adam want to push further during steering.
        amp_init_mA=-1.5,
        amp_clip=(-3.0, 3.0),
        verbose=True,
    )
    t_rect = time.time() - t0
    si_rect = rect_res["history"]["si"][-1]
    print(f"[joint-opt] Rect done: SI {si_baseline:+.3f} → {si_rect:+.3f}  "
          f"amps={[f'{a:+.2f}' for a in rect_res['amps']]} mA  "
          f"({t_rect:.0f}s)", flush=True)

    # ── JAX arrays for differentiable field ───────────────────────────────────
    fiber_xy_j, all_centers_j = build_fiber_arrays(nerve, geoms)

    # ── joint optimisation ────────────────────────────────────────────────────
    print(f"[joint-opt] Joint optimisation ({N_OPT_JOINT} iters, amps + positions) ...", flush=True)
    t0 = time.time()
    joint_res = run_joint_optimization(
        fiber_statics_batch=fs_batch,
        state0_batch=s0_batch,
        fiber_xy_um=fiber_xy_j,
        all_centers_um=all_centers_j,
        pulse_mask=pulse_j,
        node_indices=node_indices,
        target_mask=nerve.target_mask,
        dt=DT,
        contact_xyz_init=contact_xyz,
        amps_init=rect_res["amps"],
        n_steps=N_OPT_JOINT,
        # lr_amp default (8e-2) matches selectivity_demo's working regime;
        # lr_pos kept explicit because position scale (µm) is unrelated to lr_amp scale (mA).
        lr_pos=10.0,
        # Match the rect step's widened clip — otherwise the joint loop pulls
        # supra-threshold amps back to (-2.5, 2.5) and undoes the rect warm-start.
        amp_clip=(-3.0, 3.0),
        verbose=True,
        sigma_S_m=SIGMA_S_M,
    )
    t_joint = time.time() - t0
    si_joint = joint_res["history"]["si"][-1]
    xyz_disp = joint_res["contact_xyz_um"] - contact_xyz
    disp_um  = np.linalg.norm(xyz_disp, axis=1)
    amps_joint_str = "  ".join(f"{a:+.2f}" for a in joint_res["amps"])
    print(f"[joint-opt] Joint done: SI {si_rect:+.3f} → {si_joint:+.3f}  "
          f"disp mean={disp_um.mean():.1f} max={disp_um.max():.1f} µm  "
          f"amps=[{amps_joint_str}] mA  ({t_joint:.0f}s)", flush=True)

    # ── save results ──────────────────────────────────────────────────────────
    result = {
        "seed": SEED,
        "si_baseline": si_baseline,
        "si_rect": si_rect,
        "si_joint": si_joint,
        "si_joint_best": float(max(joint_res["history"]["si"])),
        "contact_xyz_init_um": contact_xyz.tolist(),
        "contact_xyz_final_um": joint_res["contact_xyz_um"].tolist(),
        "contact_displacement_um": disp_um.tolist(),
        "amps_rect_mA": rect_res["amps"].tolist(),
        "amps_joint_mA": joint_res["amps"].tolist(),
        "rect_si_history": rect_res["history"]["si"],
        "joint_si_history": joint_res["history"]["si"],
        "t_rect_s": t_rect,
        "t_joint_s": t_joint,
    }
    save_json(result, OUT / "data_joint_opt.json")

    # ── per-seed cross-section + activation summaries ─────────────────────────
    # Show the rect endpoint and the joint endpoint separately so the
    # user can see how moving contacts on top of amplitude tuning changed
    # which fibres ended up activated.
    plot_seed_summary(
        out_path=OUT / "fig_rect_summary.png",
        nerve=nerve,
        contact_xyz_final=contact_xyz,
        amps_mA=np.array(rect_res["amps"]),
        acts=np.array(rect_res["history"]["acts"][-1]),
        loss_hist=list(rect_res["history"]["loss"]),
        si_hist=list(rect_res["history"]["si"]),
        si_baseline=si_baseline,
        title=f"Rect (Adam-FD, {N_OPT_RECT} iters) — single seed {SEED}",
        nerve_radius_um=NERVE_RADIUS_UM,
        cuff_radius_um=CUFF_RADIUS_UM,
    )
    plot_seed_summary(
        out_path=OUT / "fig_joint_summary.png",
        nerve=nerve,
        contact_xyz_final=joint_res["contact_xyz_um"],
        amps_mA=np.array(joint_res["amps"]),
        acts=np.array(joint_res["history"]["acts"][-1]),
        loss_hist=list(joint_res["history"]["loss"]),
        si_hist=list(joint_res["history"]["si"]),
        si_baseline=si_baseline,
        title=f"Joint amp + position (Adam, {N_OPT_JOINT} iters) — seed {SEED}",
        contact_xyz_init=contact_xyz,
        nerve_radius_um=NERVE_RADIUS_UM,
        cuff_radius_um=CUFF_RADIUS_UM,
    )

    # ── legacy SI-comparison + displacement figure (kept for continuity) ──────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # SI comparison bar
    ax = axes[0]
    si_vals = [si_baseline, si_rect, si_joint]
    colors  = ["C7", "C0", "C3"]
    bars = ax.bar(["Baseline", "Rect", "Joint"], si_vals, color=colors, alpha=0.8)
    ax.set_ylabel("Selectivity Index (SI)")
    ax.set_title("SI comparison")
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    for bar, v in zip(bars, si_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02 * np.sign(v),
                f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)

    # SI history curves
    ax = axes[1]
    ax.plot(rect_res["history"]["si"],  label="Rect",  color="C0")
    ax.plot(joint_res["history"]["si"], label="Joint", color="C3")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("SI")
    ax.set_title("SI over training")
    ax.legend()
    ax.axhline(0, color="k", ls="--", lw=0.8)

    # Contact displacement in x-y plane (top view)
    ax = axes[2]
    ax.set_aspect("equal")
    circle = plt.Circle((0, 0), NERVE_RADIUS_UM, fill=False, color="k", lw=0.8, ls=":")
    ax.add_patch(circle)
    for k in range(N_CONTACTS):
        xi, yi = contact_xyz[k, 0], contact_xyz[k, 1]
        xf, yf = joint_res["contact_xyz_um"][k, 0], joint_res["contact_xyz_um"][k, 1]
        ax.plot([xi, xf], [yi, yf], "C3-", lw=1.5, alpha=0.7)
        ax.plot(xi, yi, "o", color="C0", ms=7)
        ax.plot(xf, yf, "s", color="C3", ms=7)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title("Contact displacement\n(blue=init, red=final)")
    lim = CUFF_RADIUS_UM * 1.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    fig.tight_layout()
    fig.savefig(OUT / "fig_joint_opt.png", dpi=150)
    plt.close(fig)
    print(f"  → {OUT / 'fig_joint_opt.png'}", flush=True)
    print(f"\n[joint-opt] All outputs in {OUT}", flush=True)


if __name__ == "__main__":
    main()
