"""Mixed-diameter loss-landscape experiment.

Demonstrates the bistability that prevents single-diameter spatial
selectivity optimisation from generalising to mixed-diameter
type selectivity.  Setup:

  - 2-fascicle mixed-diameter nerve:
      target fascicle    (right half)  = small fibres,  D = 5.7  µm
      off-target fascicle(left  half)  = large fibres,  D = 14.0 µm
  - 6-contact 1.5 mm cuff
  - Bipolar amplitude template: contact 0 = +1, contact 3 = -1, rest 0
  - Sweep scaling factor alpha over a wide range, compute the WQ
    selectivity loss with energy regularisation lambda for several
    lambdas.

The point: as alpha increases from 0, the *larger* off-target fibres
fire first (their threshold is lower), giving anti-selective SI <= 0
in the intermediate-alpha regime; once alpha is large enough, the
small targets also fire and SI returns to 0 (all-fire).  No
alpha gives positive SI.  The loss landscape under regularisation
has two attractive minima — alpha = 0 (silent / trivial) and
alpha = large (saturated all-fire) — never the selective interior.

Run:
    python -m experiments_v2.mixed_diameter_landscape
"""
from __future__ import annotations
import json
import os
import pathlib
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jaxfibers.nerve.geometry import NerveGeometry, FascicleOutline
from jaxfibers.stim.multichannel_field import (
    make_ring_cuff_positions, precompute_ve_unit,
)
from jaxfibers.stim.batch_solve import (
    batch_integrate_m_max, stack_fiber_statics, initial_states_batch,
)
from jaxfibers.optim.losses import activation_proxy_batch, selectivity_index
from jaxfibers.fibers.mrg import section_centers_um

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT  = ROOT / "outputs" / "mixed_diameter_landscape"
OUT.mkdir(parents=True, exist_ok=True)

# ─── geometry knobs (scaled down for CPU run) ────────────────────────────────
N_PER_FASC      = int(os.environ.get("N_PER_FASC", 25))
N_NODES         = int(os.environ.get("N_NODES", 11))
N_CONTACTS      = 6
CUFF_RADIUS_UM  = 1500.0
NERVE_RADIUS_UM = 500.0
TARGET_D_UM     = 5.7
OFFTARGET_D_UM  = 14.0
DT              = 0.005
T_STOP          = float(os.environ.get("T_STOP", 2.5))
DELAY_MS        = 1.0
PW_MS           = 0.1
ALPHA_MAX       = float(os.environ.get("ALPHA_MAX", 4.0))
N_ALPHA         = int(os.environ.get("N_ALPHA", 31))
SEED            = int(os.environ.get("SEED", 7))


def build_mixed_diameter_nerve(seed: int) -> NerveGeometry:
    """Two-fascicle nerve: target (D=5.7 µm) right, off-target (D=14 µm) left."""
    rng = np.random.default_rng(seed)
    r_fasc = 200.0
    centres = [(+250.0, 0.0, True), (-250.0, 0.0, False)]  # (cx, cy, is_target)
    xs, ys, ds, tmask = [], [], [], []
    fascicles = []
    for (cx, cy, is_tgt) in centres:
        D = TARGET_D_UM if is_tgt else OFFTARGET_D_UM
        # uniform-in-disk sample
        pts = []
        while len(pts) < N_PER_FASC:
            x = rng.uniform(-r_fasc, r_fasc)
            y = rng.uniform(-r_fasc, r_fasc)
            if x*x + y*y < (0.92 * r_fasc)**2:
                pts.append((cx + x, cy + y))
        xs.extend(p[0] for p in pts)
        ys.extend(p[1] for p in pts)
        ds.extend([D] * N_PER_FASC)
        tmask.extend([is_tgt] * N_PER_FASC)
        fascicles.append(FascicleOutline(
            cx_um=cx, cy_um=cy, r_um=r_fasc, is_target=is_tgt,
            n_fibers=N_PER_FASC,
        ))
    return NerveGeometry(
        n_fibers=len(xs),
        fiber_x_um=np.asarray(xs, dtype=np.float64),
        fiber_y_um=np.asarray(ys, dtype=np.float64),
        fiber_diam=np.asarray(ds, dtype=np.float64),
        target_mask=np.asarray(tmask, dtype=bool),
        fascicles=fascicles,
        divider_angle_deg=0.0,
    )


def main():
    print(f"[mixed-D landscape] N_PER_FASC={N_PER_FASC}  N_NODES={N_NODES}  "
          f"T_STOP={T_STOP}ms  N_ALPHA={N_ALPHA}")
    nerve = build_mixed_diameter_nerve(SEED)
    print(f"Nerve: {nerve.n_fibers} fibres "
          f"({int(nerve.target_mask.sum())} target / "
          f"{int((~nerve.target_mask).sum())} off-target), "
          f"D_target={TARGET_D_UM} µm, D_off={OFFTARGET_D_UM} µm")

    contact_xyz = make_ring_cuff_positions(
        n_contacts=N_CONTACTS, cuff_radius_um=CUFF_RADIUS_UM, cuff_z_um=0.0,
    )
    Ve_unit, node_indices, geoms = precompute_ve_unit(
        nerve_geom=nerve, n_nodes=N_NODES, contact_xyz_um=contact_xyz,
        model="mrg",
    )
    mid_comp = node_indices[0, len(node_indices[0]) // 2]
    mid_z = float(np.array(section_centers_um(geoms[0]))[mid_comp])
    contact_xyz[:, 2] = mid_z
    Ve_unit, node_indices, geoms = precompute_ve_unit(
        nerve_geom=nerve, n_nodes=N_NODES, contact_xyz_um=contact_xyz,
        model="mrg",
    )

    fs_batch = stack_fiber_statics(geoms, DT)
    s0_batch = initial_states_batch(geoms)

    N_STEPS = int(T_STOP / DT)
    t_grid = (np.arange(N_STEPS) + 1) * DT
    pulse_mask = np.where(
        (t_grid >= DELAY_MS) & (t_grid < DELAY_MS + PW_MS), 1.0, 0.0,
    ).astype(np.float64)
    print(f"N_STEPS={N_STEPS}  Ve_unit shape={Ve_unit.shape}")

    # Bipolar amplitude template: contact 0 (right, near target) = -1 mA
    # cathodic, contact 3 (left, near off-target) = +1 mA anodic.  Other
    # contacts at 0.  We sweep a scaling factor alpha.
    I_base = np.zeros(N_CONTACTS, dtype=np.float64)
    I_base[0] = -1.0
    I_base[3] = +1.0

    alphas = np.linspace(0.0, ALPHA_MAX, N_ALPHA)

    Ve_unit_j = jnp.asarray(Ve_unit, dtype=jnp.float64)
    pulse_j   = jnp.asarray(pulse_mask, dtype=jnp.float64)
    node_idx_j = jnp.asarray(node_indices, dtype=jnp.int32)

    def forward(I_vec: np.ndarray) -> tuple[float, np.ndarray]:
        """Return (SI, acts) for an amplitude vector I_vec ∈ ℝ^K (mA)."""
        # Ve_seq[f, t, n] = sum_k I_vec[k] * pulse[t] * Ve_unit[k, f, n]
        I_j = jnp.asarray(I_vec, dtype=jnp.float64)
        Ve_static = jnp.einsum("k,kfn->fn", I_j, Ve_unit_j)  # [n_fibers, n_comp]
        Ve_seq = pulse_j[None, :, None] * Ve_static[:, None, :]  # [n_fib, T, n_comp]
        m_max = batch_integrate_m_max(fs_batch, s0_batch, Ve_seq, DT)
        acts = activation_proxy_batch(
            m_max, node_idx_j,
            soft_threshold=0.0, soft_temperature=0.0,
        )
        acts_np = np.asarray(acts)
        si = float(selectivity_index(acts, nerve.target_mask))
        return si, acts_np

    print(f"Sweeping {N_ALPHA} alpha values in [0, {ALPHA_MAX}] mA scale ...")
    sis    = []
    acts_all = []
    tgt_act_rate = []
    off_act_rate = []
    t_start = time.time()
    for i, a in enumerate(alphas):
        I_vec = a * I_base
        si, acts = forward(I_vec)
        sis.append(si)
        acts_all.append(acts)
        fired = acts >= 0.5
        tmask = nerve.target_mask
        tgt_act_rate.append(float(fired[tmask].mean()))
        off_act_rate.append(float(fired[~tmask].mean()))
        elapsed = time.time() - t_start
        print(f"  [{i+1:2d}/{N_ALPHA}] alpha={a:5.2f}  "
              f"tgt_fired={tgt_act_rate[-1]:.2f}  "
              f"off_fired={off_act_rate[-1]:.2f}  "
              f"SI={si:+.3f}  ({elapsed:.0f}s)")

    sis = np.array(sis)
    acts_all = np.stack(acts_all, axis=0)  # [n_alpha, n_fibers]
    tgt_act_rate = np.array(tgt_act_rate)
    off_act_rate = np.array(off_act_rate)

    # WQ loss = mean over fibres of (target * (1 - acts) + (1 - target) * acts)
    target_arr = nerve.target_mask.astype(np.float64)
    wq = (target_arr[None, :] * (1 - acts_all) +
          (1 - target_arr[None, :]) * acts_all).mean(axis=1)

    # Regularised loss at several lambdas (energy = mean(I_k^2) = α^2 * mean(I_base^2))
    energy = (alphas ** 2) * float(np.mean(I_base ** 2))
    lambdas = [0.0, 1e-3, 1e-2, 1e-1]
    losses = {f"{l:g}": (wq + l * energy).tolist() for l in lambdas}

    out = {
        "config": {
            "n_per_fasc": N_PER_FASC,
            "n_nodes": N_NODES,
            "T_stop_ms": T_STOP,
            "n_alpha": N_ALPHA,
            "alpha_max": ALPHA_MAX,
            "seed": SEED,
            "target_D_um": TARGET_D_UM,
            "offtarget_D_um": OFFTARGET_D_UM,
            "I_base_mA": I_base.tolist(),
            "lambdas": lambdas,
        },
        "alpha_mA": alphas.tolist(),
        "wq_loss": wq.tolist(),
        "losses_per_lambda": losses,
        "energy": energy.tolist(),
        "si": sis.tolist(),
        "target_act_rate": tgt_act_rate.tolist(),
        "offtarget_act_rate": off_act_rate.tolist(),
        "fiber_x_um": nerve.fiber_x_um.tolist(),
        "fiber_y_um": nerve.fiber_y_um.tolist(),
        "target_mask": nerve.target_mask.tolist(),
        "fiber_diam_um": nerve.fiber_diam.tolist(),
    }
    out_path = OUT / "data_landscape.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  -> {out_path}")
    print(f"Total wall time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
