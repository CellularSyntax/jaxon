"""Backfill rich per-fiber data into existing sparse_sampling_seed_*.json.

The original cluster runs stored only scalar SI / SI_transfer / amps for each
sparsity level.  Figure 3 now needs the same per-fiber data we keep for dense
runs: the sparse-optimised amps re-evaluated on the FULL dense nerve
(``transfer.final_acts`` for activation maps, ``transfer.firing`` for
on-/off-target recruitment), plus the in-sample sparse-nerve
``final_acts``/``firing``.

This script reuses the *exact* canonical helpers from selectivity_sweep_duke
(single source of truth) and only does forward passes — it never re-optimises,
so the stored ``amps_mA``/``si_transfer`` are untouched and every figure value
stays identical.  Each result's recomputed SI is checked against the stored
``si_transfer`` as a sanity gate.

Run:
  conda activate jaxon
  python -m experiments_v2.backfill_sparse_transfer            # all samples
  python -m experiments_v2.backfill_sparse_transfer sub-10_sam-1 human_sub-56_sam-1
"""
from __future__ import annotations

import os
# selectivity_sweep_duke raises at import unless DUKE_SAMPLE_DIR is set; it is
# only used to derive an output path in __main__, which we don't invoke here.
os.environ.setdefault("DUKE_SAMPLE_DIR", "duke_Ves/_backfill_placeholder")

import json
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jaxon.stim.batch_solve import stack_fiber_statics, initial_states_batch
from experiments_v2.utils import save_json
from experiments_v2.duke_loader import load_duke_sample, cluster_target_mask
from experiments_v2.selectivity_sweep_duke import (
    _build_pulse_mask,
    _sparse_subsample_duke,
    _firing_summary,
    _eval_amps_on_dense,
)

ROOT  = Path(__file__).resolve().parent.parent
SWEEP = ROOT / "outputs" / "duke_sweeps"
DUKE  = ROOT / "duke_Ves"

# Match selectivity_sweep_duke defaults (the values used for the cluster runs).
DT                    = 0.005
DELAY_MS              = 1.0
PW_MS                 = 1.0
FIBER_DIAMETER_UM     = 5.7
N_NODES               = 21
SPARSE_SWEEP_RNG_SEED = 42
SI_TOL                = 5e-3   # recomputed vs stored si_transfer
SETTLE_MS             = 2.0    # window past pulse end (proxy = max over window)


def _t_grid_for_pulse(shape: str, delay_ms: float, pw_ms: float,
                      asym_ratio: float) -> np.ndarray:
    """Time grid sized to comfortably cover the pulse + AP settle.  The exact
    T_STOP used on the cluster isn't stored, but the activation proxy is a max
    over the window, so any window past the pulse + a settle margin reproduces
    the stored result (verified against si_transfer)."""
    shape = shape.strip().lower()
    if shape == "biphasic_asym":
        pulse_end = delay_ms + pw_ms + pw_ms * asym_ratio
    else:  # monophasic / biphasic_sym
        pulse_end = delay_ms + pw_ms
    t_stop = max(3.0, pulse_end + SETTLE_MS)
    n_steps = int(round(t_stop / DT))
    return (np.arange(n_steps) + 1) * DT


def _seed_in_for_nerve(nd: dict, pulse_mask: np.ndarray, target_mask: np.ndarray):
    """Assemble the minimal seed_in dict that _eval_amps_on_dense consumes."""
    return {
        "Ve_unit":      nd["Ve_unit"],
        "node_indices": nd["node_indices"],
        "pulse_mask":   pulse_mask,
        "target_mask":  target_mask,
        "fs_batch":     stack_fiber_statics(nd["geoms"], DT),
        "s0_batch":     initial_states_batch(nd["geoms"]),
    }


def _backfill_sample(sample: str) -> tuple[int, float]:
    """Return (n_results_backfilled, max_abs_si_diff) for one sample."""
    sample_dir = SWEEP / sample
    sparse_files = sorted(sample_dir.glob("sparse_sampling_seed_*.json"))
    if not sparse_files:
        return 0, 0.0
    duke_path = DUKE / sample
    if not duke_path.exists():
        print(f"[{sample}] no duke_Ves bundle — skip", flush=True)
        return 0, 0.0

    t0 = time.time()
    duke = load_duke_sample(duke_path, fiber_diam_um=FIBER_DIAMETER_UM,
                            n_nodes=N_NODES, verbose=False)
    dense_seed = _seed_in_for_nerve(duke, np.zeros(1), np.zeros(1, bool))  # placeholders
    print(f"[{sample}] loaded {duke['nerve_geom'].n_fibers} fibers "
          f"({time.time() - t0:.0f}s)", flush=True)

    n_done = 0
    max_diff = 0.0
    for sp in sparse_files:
        sd = json.loads(sp.read_text())
        if not sd.get("enabled", True) or not sd.get("results"):
            continue
        target_ids = list(sd.get("target_ids", []))
        if not target_ids:
            # fall back to dense data_seed
            seed = int(sp.stem.split("_")[-1])
            dpath = sample_dir / f"data_seed_{seed:04d}.json"
            if dpath.exists():
                target_ids = list(json.loads(dpath.read_text())
                                  .get("cluster", {}).get("target_ids", []))
        dense_tgt = cluster_target_mask(duke["nerve_geom"], duke["fasc_id"],
                                        duke["fasc_meta"], target_ids)
        shape      = sd.get("pulse_shape", "biphasic_sym")
        delay_ms   = float(sd.get("delay_ms", DELAY_MS))
        pw_ms      = float(sd.get("pulse_pw_ms", PW_MS))
        asym_ratio = float(sd.get("pulse_asym_ratio") or 4.0)
        t_grid = _t_grid_for_pulse(shape, delay_ms, pw_ms, asym_ratio)
        pulse  = _build_pulse_mask(t_grid, delay_ms, pw_ms, shape, asym_ratio=asym_ratio)
        dense_seed["pulse_mask"]  = pulse
        dense_seed["target_mask"] = dense_tgt

        changed = False
        for res in sd["results"]:
            if res.get("skipped") or res.get("amps_mA") is None:
                continue
            amps = np.asarray(res["amps_mA"], dtype=float)

            # Transfer: sparse amps on the FULL dense nerve.
            si_tr, acts_tr = _eval_amps_on_dense(amps, dense_seed, DT)
            res["transfer"] = {
                "si":         si_tr,
                "final_acts": np.asarray(acts_tr, dtype=float).tolist(),
                "firing":     _firing_summary(acts_tr, dense_tgt),
            }
            if res.get("si_transfer") is not None:
                max_diff = max(max_diff, abs(si_tr - float(res["si_transfer"])))

            # In-sample: sparse amps on the sparse nerve (same data dense saves).
            n_pf = "centroid" if res["strategy"] == "centroid" else int(res["n_per_fascicle"])
            try:
                sub = _sparse_subsample_duke(duke, n_pf, rng_seed=SPARSE_SWEEP_RNG_SEED)
                sub_tgt = cluster_target_mask(sub["nerve_geom"], sub["fasc_id"],
                                              sub["fasc_meta"], target_ids)
                sub_seed = _seed_in_for_nerve(sub, pulse, sub_tgt)
                si_sp, acts_sp = _eval_amps_on_dense(amps, sub_seed, DT)
                res["final_acts"] = np.asarray(acts_sp, dtype=float).tolist()
                res["firing"]     = _firing_summary(acts_sp, sub_tgt)
            except Exception as e:
                print(f"[{sample} {sp.name}] sparse-side recompute failed "
                      f"({res['strategy']}/{res['n_per_fascicle']}): {e}", flush=True)

            n_done += 1
            changed = True

        if changed:
            save_json(sd, sp)

    print(f"[{sample}] backfilled {n_done} results  max|dSI_transfer|={max_diff:.2e}",
          flush=True)
    return n_done, max_diff


def _sharded(samples: list[str]) -> list[str]:
    """Interleaved sharding for SLURM arrays.

    Count  = GLOBAL_N_SHARDS  (fallback SLURM_ARRAY_TASK_COUNT, default 1).
    Index  = SHARD_INDEX      (fallback SLURM_ARRAY_TASK_ID,    default 0).
    The explicit GLOBAL_N_SHARDS/SHARD_INDEX are set by the sbatch so this works
    even inside a container where SLURM_ARRAY_* may not propagate.  Sample i is
    processed iff i % count == index.
    """
    n_shards  = int(os.environ.get("GLOBAL_N_SHARDS")
                    or os.environ.get("SLURM_ARRAY_TASK_COUNT") or 1)
    shard_idx = int(os.environ.get("SHARD_INDEX")
                    or os.environ.get("SLURM_ARRAY_TASK_ID") or 0)
    if n_shards <= 1:
        return samples
    mine = [s for i, s in enumerate(samples) if i % n_shards == shard_idx]
    print(f"[backfill] shard {shard_idx}/{n_shards}: {len(mine)}/{len(samples)} samples",
          flush=True)
    return mine


def main(argv: list[str]) -> int:
    if argv:
        samples = argv
    else:
        samples = sorted(d.name for d in SWEEP.iterdir()
                         if d.is_dir() and list(d.glob("sparse_sampling_seed_*.json")))
        samples = _sharded(samples)
    print(f"[backfill] {len(samples)} samples", flush=True)
    total, worst, failed = 0, 0.0, []
    for s in samples:
        try:
            n, d = _backfill_sample(s)
        except Exception as e:  # one bad sample must not abort the shard
            print(f"[{s}] FAILED: {e}", flush=True)
            failed.append(s)
            continue
        total += n
        worst = max(worst, d)
    if failed:
        print(f"[backfill] {len(failed)} sample(s) failed: {', '.join(failed)}",
              flush=True)
    print(f"[backfill] DONE: {total} results across {len(samples)} samples; "
          f"max|dSI_transfer|={worst:.2e} (tol {SI_TOL:g})", flush=True)
    if worst > SI_TOL:
        print("[backfill] WARNING: recomputed SI deviates beyond tolerance — "
              "check DT/T_STOP/pulse assumptions.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
