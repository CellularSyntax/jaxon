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

from jaxfibers.nerve.geometry import make_synthetic_nerve
from jaxfibers.stim.multichannel_field import (
    make_ring_cuff_positions, precompute_ve_unit, build_fiber_arrays,
)
from jaxfibers.stim.batch_solve import stack_fiber_statics, initial_states_batch
from jaxfibers.optim.losses import activation_proxy_batch, selectivity_index
from jaxfibers.optim.optimizer import run_rect_optimization, run_joint_optimization
from jaxfibers.fibers.mrg import section_centers_um
from experiments_v2.utils import ensure_dir, save_json

OUT = ensure_dir(ROOT / "outputs" / "selectivity_joint_opt")

# ─────────────────────────────────────────────── experiment parameters ────────
SEED            = 0
N_FIBERS        = 20
N_NODES         = 51          # smaller than sweep for faster local run
N_CONTACTS      = 6
NERVE_RADIUS_UM = 500.0
CUFF_RADIUS_UM  = 1500.0
TARGET_FRACTION = 0.30
DT              = 0.005       # ms
T_STOP          = 8.0         # ms
DELAY_MS        = 1.0
PW_MS           = 0.1
N_OPT_RECT      = 150
N_OPT_JOINT     = 150
SIGMA_S_M       = 0.3


def main():
    print(f"[joint-opt] seed={SEED}, N_FIBERS={N_FIBERS}, N_NODES={N_NODES}", flush=True)

    # ── nerve + cuff ──────────────────────────────────────────────────────────
    nerve = make_synthetic_nerve(
        n_fibers=N_FIBERS,
        nerve_radius_um=NERVE_RADIUS_UM,
        target_fraction=TARGET_FRACTION,
        seed=SEED,
    )
    n_tgt = int(nerve.target_mask.sum())
    d_min, d_max = float(nerve.fiber_diam.min()), float(nerve.fiber_diam.max())
    print(f"[joint-opt] Nerve: {N_FIBERS} fibers  targets={n_tgt}/{N_FIBERS} ({n_tgt/N_FIBERS*100:.0f}%)  "
          f"D=[{d_min:.1f},{d_max:.1f}] µm", flush=True)

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
        lr_amp=5e-2,
        lr_pos=10.0,
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

    # ── figures ───────────────────────────────────────────────────────────────
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
