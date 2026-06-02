"""Spatial selectivity optimisation — cluster sweep (paper-quality).

Runs N_NERVES synthetic nerve realisations, each with both rect and waveform
optimisation.  Saves one JSON per seed and produces aggregate figures.

Setup (per seed)
----------------
- 20 MRG fibers, mixed diameters, 500 µm nerve radius
- 6-contact ring cuff, 1 mm from nerve centre
- Target: central 30% area
- 8 ms simulation window (accommodates biphasic and slow dynamics)
- 200 Adam iterations per mode

Outputs
-------
outputs/selectivity_sweep/
  data_seed_<N>.json          — per-seed results
  fig_sweep_si_violin.png     — SI distributions: init vs rect vs waveform
  fig_sweep_loss_curves.png   — mean ± 1σ loss curves for both modes
  fig_sweep_activation_maps.png — example seed activation maps

Run from project root (GPU recommended):
    python experiments_v2/selectivity_sweep.py
"""
from __future__ import annotations

import os
import sys
import pathlib
import json
import time
import argparse

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
from jaxfibers.stim.multichannel_field import make_ring_cuff_positions, precompute_ve_unit
from jaxfibers.stim.batch_solve import stack_fiber_statics, initial_states_batch
from jaxfibers.optim.losses import activation_proxy_batch, selectivity_index
from jaxfibers.optim.optimizer import run_rect_optimization, run_waveform_optimization
from jaxfibers.fibers.mrg import section_centers_um
from experiments_v2.utils import ensure_dir

OUT = ensure_dir(ROOT / "outputs" / "selectivity_sweep")

# ─────────────────────────────────────────────────── sweep parameters ─────────
N_NERVES        = 100   # total seeds across all array tasks
N_FIBERS        = 100          # 5× more fibers → better population statistics (FD rect is cheap)
N_NODES         = 21           # n_comp = 20×11 + 1 = 221 (~23 mm fiber for D=10 µm)
N_CONTACTS      = 6
NERVE_RADIUS_UM = 500.0
CUFF_RADIUS_UM  = 1500.0
TARGET_FRACTION = 0.30
DT              = 0.005        # ms
T_STOP          = 3.0          # ms; PW=0.1 ms + DELAY=1.0 ms + slowest MRG
                                # propagation (24 mm fiber, 26 m/s at D=5.7 µm)
                                # = ~2.1 ms, so 3 ms covers AP arrival at both
                                # ends with 0.9 ms margin.  Each ms saved cuts
                                # ~25 % off per-iter FD pass cost.
DELAY_MS        = 1.0
PW_MS           = 0.1
N_OPT_RECT      = 100          # 200 was overkill: Adam plateaus by ~80 iters
N_OPT_WAVE      = 100          # for these problems.  selectivity_demo.py hits
                                # SI=1.0 in 50 iters on a 6-fiber problem;
                                # 100 leaves headroom for the 100-fiber sweep.
EXAMPLE_SEED    = 0            # seed for the example activation-map figure


def _run_one_seed(seed: int, verbose: bool = True) -> dict:
    label = f"[seed={seed}]"
    print(f"\n{'='*60}\n{label} Building nerve ...")

    nerve = make_synthetic_nerve(
        n_fibers=N_FIBERS,
        nerve_radius_um=NERVE_RADIUS_UM,
        target_fraction=TARGET_FRACTION,
        seed=seed,
    )
    n_tgt = int(nerve.target_mask.sum())
    d_min, d_max = float(nerve.fiber_diam.min()), float(nerve.fiber_diam.max())
    print(f"{label} Nerve: {N_FIBERS} fibers  targets={n_tgt}/{N_FIBERS} ({n_tgt/N_FIBERS*100:.0f}%)  "
          f"D=[{d_min:.1f},{d_max:.1f}] µm", flush=True)

    N_STEPS = int(T_STOP / DT)
    t_grid  = (np.arange(N_STEPS) + 1) * DT

    contact_xyz = make_ring_cuff_positions(
        n_contacts=N_CONTACTS,
        cuff_radius_um=CUFF_RADIUS_UM,
        cuff_z_um=0.0,
    )

    print(f"{label} Precomputing fields ...", flush=True)
    Ve_unit, node_indices, geoms = precompute_ve_unit(
        nerve_geom=nerve, n_nodes=N_NODES, contact_xyz_um=contact_xyz
    )
    # Center cuff z at fiber midpoint
    mid_comp = node_indices[0, len(node_indices[0]) // 2]
    mid_z    = float(np.array(section_centers_um(geoms[0]))[mid_comp])
    contact_xyz[:, 2] = mid_z
    Ve_unit, node_indices, geoms = precompute_ve_unit(
        nerve_geom=nerve, n_nodes=N_NODES, contact_xyz_um=contact_xyz
    )

    print(f"{label} Building solver statics ...", flush=True)
    fs_batch = stack_fiber_statics(geoms, DT)
    s0_batch = initial_states_batch(geoms)

    pulse_mask = np.where(
        (t_grid >= DELAY_MS) & (t_grid < DELAY_MS + PW_MS), 1.0, 0.0
    ).astype(np.float64)
    pulse_j   = jnp.asarray(pulse_mask)
    Ve_unit_j = jnp.asarray(Ve_unit, dtype=jnp.float64)

    # ── baseline (zero stimulation) ──
    m_max_zero = activation_proxy_batch(
        jnp.zeros((N_FIBERS, geoms[0].n_comp), dtype=jnp.float64),
        jnp.asarray(node_indices, dtype=jnp.int32),
    )
    si_baseline = selectivity_index(np.array(m_max_zero), nerve.target_mask)

    # ── rect ──
    print(f"{label} Baseline SI = {si_baseline:+.3f}", flush=True)
    print(f"{label} Rect optimisation ({N_OPT_RECT} iters) ...", flush=True)
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
        verbose=verbose,
    )
    t_rect = time.time() - t0
    si_rect = rect_res["history"]["si"][-1]
    print(f"{label} Rect done: SI {si_baseline:+.3f} → {si_rect:+.3f}  "
          f"amps={[f'{a:+.2f}' for a in rect_res['amps']]} mA  "
          f"({t_rect:.0f}s)", flush=True)

    # ── waveform (warm-start from rect) ──
    u_init = np.zeros((N_CONTACTS, N_STEPS), dtype=np.float64)
    for k in range(N_CONTACTS):
        u_init[k] = float(rect_res["amps"][k]) * pulse_mask

    print(f"{label} Waveform optimisation ({N_OPT_WAVE} iters) ...", flush=True)
    t0 = time.time()
    wave_res = run_waveform_optimization(
        fiber_statics_batch=fs_batch,
        state0_batch=s0_batch,
        Ve_unit=Ve_unit_j,
        node_indices=node_indices,
        target_mask=nerve.target_mask,
        dt=DT,
        T=N_STEPS,
        n_steps=N_OPT_WAVE,
        u_init=u_init,
        verbose=verbose,
    )
    t_wave = time.time() - t0
    si_wave = wave_res["history"]["si"][-1]
    print(f"{label} Wave done: SI {si_rect:+.3f} → {si_wave:+.3f}  ({t_wave:.0f}s)", flush=True)

    result = {
        "seed": seed,
        "si_baseline": si_baseline,
        "rect": {
            "final_si": rect_res["history"]["si"][-1],
            "loss_history": rect_res["history"]["loss"],
            "si_history": rect_res["history"]["si"],
            "amps_mA": rect_res["amps"].tolist(),
            "final_acts": rect_res["history"]["acts"][-1].tolist(),
            "time_s": t_rect,
        },
        "waveform": {
            "final_si": wave_res["history"]["si"][-1],
            "loss_history": wave_res["history"]["loss"],
            "si_history": wave_res["history"]["si"],
            "final_acts": wave_res["history"]["acts"][-1].tolist(),
            "time_s": t_wave,
        },
        "nerve": {
            "fiber_diam": nerve.fiber_diam.tolist(),
            "target_mask": nerve.target_mask.tolist(),
            "fiber_x_um": nerve.fiber_x_um.tolist(),
            "fiber_y_um": nerve.fiber_y_um.tolist(),
        },
    }
    return result


def main():
    # SLURM array support: set SEED_START / SEED_END env vars per task.
    # Falls back to running all N_NERVES seeds locally.
    seed_start = int(os.environ.get("SEED_START", 0))
    seed_end   = int(os.environ.get("SEED_END", N_NERVES))

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(range(seed_start, seed_end)),
                        help="Which seeds to run (default: SEED_START..SEED_END-1)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(f"[sweep] Running seeds {args.seeds[0]}..{args.seeds[-1]} "
          f"({len(args.seeds)} seeds)")

    for seed in args.seeds:
        out_path = OUT / f"data_seed_{seed:04d}.json"
        if out_path.exists():
            print(f"Skipping seed={seed} (output exists)")
            continue
        res = _run_one_seed(seed, verbose=not args.quiet)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[seed={seed:04d}]  rect SI={res['rect']['final_si']:+.3f}  "
              f"wave SI={res['waveform']['final_si']:+.3f}")

    print(f"\n[sweep] Done. Run analyze_selectivity_sweep.py for aggregate figures.")


if __name__ == "__main__":
    main()
