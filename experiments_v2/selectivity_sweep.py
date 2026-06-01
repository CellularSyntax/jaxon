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
N_NERVES        = 12
N_FIBERS        = 20
N_NODES         = 101          # n_comp = 100×11 + 1 = 1101
N_CONTACTS      = 6
NERVE_RADIUS_UM = 500.0
CUFF_RADIUS_UM  = 1500.0
TARGET_FRACTION = 0.30
DT              = 0.005        # ms
T_STOP          = 8.0          # ms
DELAY_MS        = 1.0
PW_MS           = 0.1
N_OPT_RECT      = 200
N_OPT_WAVE      = 200
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

    N_STEPS = int(T_STOP / DT)
    t_grid  = (np.arange(N_STEPS) + 1) * DT

    contact_xyz = make_ring_cuff_positions(
        n_contacts=N_CONTACTS,
        cuff_radius_um=CUFF_RADIUS_UM,
        cuff_z_um=0.0,
    )

    print(f"{label} Precomputing fields ...")
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

    print(f"{label} Building solver statics ...")
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
    print(f"{label} Rect optimisation ...")
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

    # ── waveform (warm-start from rect) ──
    u_init = np.zeros((N_CONTACTS, N_STEPS), dtype=np.float64)
    for k in range(N_CONTACTS):
        u_init[k] = float(rect_res["amps"][k]) * pulse_mask

    print(f"{label} Waveform optimisation ...")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(range(N_NERVES)),
                        help="Which seeds to run (default: 0..N_NERVES-1)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    all_results = []
    for seed in args.seeds:
        out_path = OUT / f"data_seed_{seed}.json"
        if out_path.exists():
            print(f"Skipping seed={seed} (output exists)")
            with open(out_path) as f:
                all_results.append(json.load(f))
            continue
        res = _run_one_seed(seed, verbose=not args.quiet)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        all_results.append(res)
        print(f"[seed={seed}]  rect SI={res['rect']['final_si']:+.3f}  "
              f"wave SI={res['waveform']['final_si']:+.3f}")

    if len(all_results) < 2:
        print("Not enough results for aggregate figures.")
        return

    # ── aggregate figures ──────────────────────────────────────────────────────
    si_base = [r["si_baseline"] for r in all_results]
    si_rect = [r["rect"]["final_si"] for r in all_results]
    si_wave = [r["waveform"]["final_si"] for r in all_results]

    # Violin plot
    fig, ax = plt.subplots(figsize=(5, 4))
    vp = ax.violinplot([si_base, si_rect, si_wave], showmedians=True)
    for pc, c in zip(vp["bodies"], ["C7", "C0", "C2"]):
        pc.set_facecolor(c); pc.set_alpha(0.7)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["Baseline", "Rect", "Waveform"])
    ax.set_ylabel("Selectivity Index (SI)")
    ax.set_title(f"SI across {len(all_results)} nerve realisations")
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sweep_si_violin.png", dpi=150)
    plt.close(fig)

    # Loss curves
    n_iter_rect = min(len(r["rect"]["loss_history"]) for r in all_results)
    n_iter_wave = min(len(r["waveform"]["loss_history"]) for r in all_results)
    loss_rect = np.array([r["rect"]["loss_history"][:n_iter_rect] for r in all_results])
    loss_wave = np.array([r["waveform"]["loss_history"][:n_iter_wave] for r in all_results])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for data, ax, title in [
        (loss_rect, axes[0], "Rectangular"),
        (loss_wave, axes[1], "Arbitrary waveform"),
    ]:
        mu = data.mean(0); sig = data.std(0)
        x = np.arange(len(mu))
        ax.plot(x, mu, "C0"); ax.fill_between(x, mu - sig, mu + sig, alpha=0.2, color="C0")
        ax.set_xlabel("Iteration"); ax.set_ylabel("WQ loss")
        ax.set_title(f"{title} (mean ± 1σ,  n={len(all_results)})")
    fig.tight_layout()
    fig.savefig(OUT / "fig_sweep_loss_curves.png", dpi=150)
    plt.close(fig)

    print(f"\nAll outputs saved to {OUT}")
    print(f"\nAggregate summary (n={len(all_results)} seeds):")
    print(f"  Baseline SI: {np.mean(si_base):.3f} ± {np.std(si_base):.3f}")
    print(f"  Rect     SI: {np.mean(si_rect):.3f} ± {np.std(si_rect):.3f}")
    print(f"  Waveform SI: {np.mean(si_wave):.3f} ± {np.std(si_wave):.3f}")


if __name__ == "__main__":
    main()
