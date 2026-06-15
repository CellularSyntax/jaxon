"""Frozen-tripole vs relaxed-all-contacts sparsity, with the deployment penalty.

The cohort headline optimizer ("Adam-FD-smart") warm-starts from a probe-
selected tripole and then FREEZES the contacts the probe left at zero
(freeze_zero_mask in run_rect_optimization), so 91% of seeds end on a
3-contact pattern.  The freeze was added because, under the old binary
proxy, unfrozen Adam drifted the zero contacts and flooded off-target
fibers.  The soft proxy + energy regularization now counter that failure
mode, so the question re-opens: if we let ALL K contacts move from the same
tripolar warm start, do we get better selectivity -- or just overfit the
sampled fibers and widen the deployment penalty?

This script answers both, per nerve, for a quick subset:

  for variant in {frozen, relaxed}:
    dense_si   = optimize on the FULL population, score on it          (ceiling)
    transfer_si= optimize on a SPARSE subsample (1 fiber/fascicle),
                 then re-score those amps on the FULL population
    penalty    = dense_si - transfer_si        (the paper's deployment penalty)
    n_active   = non-zero contacts in the dense solution

Only the freeze differs between variants; optimizer, schedule, init, soft
proxy and energy reg are held identical.  A relaxed variant that lifts
dense_si but inflates penalty is OVERFITTING the sampled fibers -- which
would reframe frozen sparsity as protective regularization, the paper's
thesis.  One that lifts dense_si AND holds penalty is a genuinely better
optimizer.

Aggregate per-nerve JSONs:
    FROZEN_VS_RELAXED_AGGREGATE=1 python -m experiments_v2.frozen_vs_relaxed

Run one nerve (normally via the SLURM sharded sbatch):
    DUKE_SAMPLE_DIR=duke_Ves/sub-11_sam-3 \
    python -m experiments_v2.frozen_vs_relaxed
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "frozen_vs_relaxed"


def _species(name: str) -> str:
    return "human" if name.startswith("human") else "swine"


# ── aggregate mode: read per-nerve JSONs, print per-species summary (no JAX) ──
def aggregate() -> int:
    import numpy as np
    rows = [json.loads(jp.read_text())
            for jp in sorted(OUT_DIR.glob("*.json")) if jp.name != "summary.json"]
    rows = [r for r in rows if r.get("ok")]
    if not rows:
        print(f"[aggregate] no usable results in {OUT_DIR}", file=sys.stderr)
        return 1

    print(f"\n{'nerve':22s} {'sp':5s} {'N':>5} "
          f"{'denseF':>6} {'denseR':>6} {'dDense':>6}  "
          f"{'penF':>6} {'penR':>6} {'dPen':>6}  {'naF':>3} {'naR':>3}")
    for r in sorted(rows, key=lambda r: (r["species"], r["nerve"])):
        print(f"{r['nerve']:22s} {r['species']:5s} {r['n_fibers']:>5d} "
              f"{r['dense_si_frozen']:>+6.3f} {r['dense_si_relaxed']:>+6.3f} "
              f"{r['dense_si_relaxed']-r['dense_si_frozen']:>+6.3f}  "
              f"{r['penalty_frozen']:>+6.3f} {r['penalty_relaxed']:>+6.3f} "
              f"{r['penalty_relaxed']-r['penalty_frozen']:>+6.3f}  "
              f"{r['n_active_frozen']:>3d} {r['n_active_relaxed']:>3d}")

    summary = {"n_nerves": len(rows), "by_species": {}}
    for sp in ("swine", "human"):
        sr = [r for r in rows if r["species"] == sp]
        if not sr:
            continue
        def med(key):
            return float(np.median([r[key] for r in sr]))
        summary["by_species"][sp] = {
            "n": len(sr),
            "dense_si_frozen":  med("dense_si_frozen"),
            "dense_si_relaxed": med("dense_si_relaxed"),
            "penalty_frozen":   med("penalty_frozen"),
            "penalty_relaxed":  med("penalty_relaxed"),
            "n_active_relaxed": med("n_active_relaxed"),
        }
    print(f"\n[summary over {len(rows)} nerves]  (medians)")
    for sp, s in summary["by_species"].items():
        print(f"  {sp:5s} (n={s['n']:2d}): "
              f"dense SI  frozen {s['dense_si_frozen']:+.3f} -> relaxed "
              f"{s['dense_si_relaxed']:+.3f}   |   "
              f"penalty  frozen {s['penalty_frozen']:+.3f} -> relaxed "
              f"{s['penalty_relaxed']:+.3f}   |   "
              f"relaxed n_active {s['n_active_relaxed']:.0f}")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  -> {OUT_DIR / 'summary.json'}")
    return 0


if os.environ.get("FROZEN_VS_RELAXED_AGGREGATE", "").strip() in ("1", "true", "yes"):
    sys.exit(aggregate())


# ── per-nerve mode (needs DUKE_SAMPLE_DIR) ────────────────────────────────────
os.environ.setdefault("N_OPT_RECT", os.environ.get("FVR_STEPS", "15"))
# Soft proxy + energy reg: these are what make the relaxed (unfrozen) run
# stable.  Held identical for BOTH variants so the comparison isolates the
# freeze.  Read at import by jaxfibers.optim.optimizer.
os.environ.setdefault("JAXLEY_FIBERS_SOFT_TEMPERATURE", "0.15")
os.environ.setdefault("JAXLEY_FIBERS_ENERGY_LAMBDA", "1e-3")
# Sparse density for the transfer leg (paper's reduced-order = 1 fiber/fascicle).
_N_PER_FASC = int(os.environ.get("FVR_N_PER_FASC", "1"))

import numpy as np
import jax
import jax.numpy as jnp

from experiments_v2 import selectivity_sweep_duke as S
from jaxfibers.optim.optimizer import run_rect_optimization
from jaxfibers.optim.losses import selectivity_index


def _probe_init(seed_in):
    """Probe-selected warm start (bipolar + per-column tripolar magnitude probe)."""
    sb = S._smart_spatial_pattern(seed_in["Ve_unit"], seed_in["target_mask"],
                                  verbose=False, label="")
    tri = S._sparse_tripolar_patterns_per_column(
        sb, seed_in["contact_xyz_um"], verbose=False, label="")
    patterns = {"bipolar": sb}
    patterns.update(tri)
    amps, best_mag, _score, _acts, _hist, pattern = S._magnitude_probe(
        patterns, S.PROBE_MAGS_MA, seed_in["target_mask"],
        seed_in["fs_batch"], seed_in["s0_batch"],
        jnp.asarray(seed_in["Ve_unit"], jnp.float64),
        jnp.asarray(seed_in["pulse_mask"], jnp.float64),
        jnp.asarray(seed_in["node_indices"], jnp.int32),
        S.DT, label="", verbose=False,
    )
    return np.asarray(amps), str(pattern), float(best_mag)


def _optimize(seed_in, relaxed: bool):
    """Probe warm start, then Adam-FD with the zero contacts frozen (relaxed=False)
    or all K contacts free (relaxed=True).  Returns (amps, in-sample SI, n_active)."""
    amps_init, pattern, _mag = _probe_init(seed_in)
    freeze = None if relaxed else (np.abs(amps_init) < 1e-9)
    n_iters = max(int(os.environ.get("N_OPT_RECT", "15")) * 3, 30)
    res = run_rect_optimization(
        fiber_statics_batch=seed_in["fs_batch"], state0_batch=seed_in["s0_batch"],
        Ve_unit=seed_in["Ve_unit"], pulse_mask=seed_in["pulse_mask"],
        node_indices=seed_in["node_indices"], target_mask=seed_in["target_mask"],
        weights=seed_in["weights"], dt=S.DT, n_steps=n_iters,
        amps_init_vector=amps_init, amp_clip=S.AMP_CLIP,
        lr=S.ADAM_LR_MA, fd_eps=S.FD_EPS_SMART_MA,
        freeze_zero_mask=freeze, early_stop_si=S.EARLY_STOP_SI, verbose=False,
    )
    loss = np.asarray(res["history"]["loss"])
    i = int(np.argmin(loss))
    amps = np.asarray(res["history"]["amps"][i])
    acts = np.asarray(res["history"]["acts"][i])
    si = float(selectivity_index(acts, np.asarray(seed_in["target_mask"], bool)))
    n_active = int(np.sum(np.abs(amps) > 1e-6))
    return amps, si, n_active, pattern


def _sparse_seed(duke, dense_seed):
    """Build the 1-fiber/fascicle sparse seed sharing the dense target cluster."""
    ci = dense_seed.get("cluster_info")
    if ci is None:
        return None
    sd = S._sparse_subsample_duke(duke, _N_PER_FASC, rng_seed=S.SPARSE_SWEEP_RNG_SEED)
    tmask = S.cluster_target_mask(sd["nerve_geom"], sd["fasc_id"], sd["fasc_meta"],
                                  ci["target_ids"])
    if int(tmask.sum()) == 0:
        return None
    return {
        **dense_seed,
        "nerve":        sd["nerve_geom"],
        "geoms":        sd["geoms"],
        "target_mask":  tmask,
        "weights":      S._class_balanced_weights(tmask),
        "Ve_unit":      sd["Ve_unit"],
        "node_indices": sd["node_indices"],
        "fs_batch":     S.stack_fiber_statics(sd["geoms"], S.DT),
        "s0_batch":     S.initial_states_batch(sd["geoms"]),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = S.SAMPLE_NAME
    out_path = OUT_DIR / f"{name}.json"
    print(f"[fvr] {name} ({_species(name)})", flush=True)
    rec = {"nerve": name, "species": _species(name), "ok": False}
    try:
        duke = S.load_duke_sample(
            S.SAMPLE_PATH, fiber_diam_um=S.FIBER_DIAMETER_UM, n_nodes=S.N_NODES,
            max_fibers=(S.MAX_FIBERS if S.MAX_FIBERS > 0 else None),
            subsample_seed=S.SUBSAMPLE_SEED, verbose=False,
        )
        dense = S._build_seed(duke, seed=0, verbose=False)
        if dense is None:
            rec["skip"] = "no valid target at seed 0"
            out_path.write_text(json.dumps(rec, indent=2))
            print(f"  SKIP: {rec['skip']}", flush=True)
            return 0
        N = int(np.asarray(dense["target_mask"]).size)

        # Dense ceilings for both variants.
        t0 = time.time()
        _af, dsi_f, na_f, patt = _optimize(dense, relaxed=False)
        _ar, dsi_r, na_r, _    = _optimize(dense, relaxed=True)
        t_dense = time.time() - t0

        # Sparse-optimized transfer for both variants -> deployment penalty.
        sp = _sparse_seed(duke, dense)
        if sp is None:
            rec["skip"] = "sparse subsample had no target fibers"
            out_path.write_text(json.dumps(rec, indent=2))
            print(f"  SKIP: {rec['skip']}", flush=True)
            return 0
        amps_sf, _, _, _ = _optimize(sp, relaxed=False)
        amps_sr, _, _, _ = _optimize(sp, relaxed=True)
        tsi_f, _ = S._eval_amps_on_dense(amps_sf, dense, S.DT)
        tsi_r, _ = S._eval_amps_on_dense(amps_sr, dense, S.DT)
        tsi_f, tsi_r = float(tsi_f), float(tsi_r)

        rec.update(
            ok=True, n_fibers=N, K=int(dense["Ve_unit"].shape[0]),
            n_per_fascicle=_N_PER_FASC, pattern=patt,
            soft_temperature=float(os.environ["JAXLEY_FIBERS_SOFT_TEMPERATURE"]),
            energy_lambda=float(os.environ["JAXLEY_FIBERS_ENERGY_LAMBDA"]),
            dense_si_frozen=dsi_f, dense_si_relaxed=dsi_r,
            transfer_si_frozen=tsi_f, transfer_si_relaxed=tsi_r,
            penalty_frozen=dsi_f - tsi_f, penalty_relaxed=dsi_r - tsi_r,
            n_active_frozen=na_f, n_active_relaxed=na_r,
            t_dense_s=t_dense,
        )
        out_path.write_text(json.dumps(rec, indent=2))
        print(f"  dense SI  frozen {dsi_f:+.3f} | relaxed {dsi_r:+.3f}   "
              f"penalty  frozen {dsi_f-tsi_f:+.3f} | relaxed {dsi_r-tsi_r:+.3f}   "
              f"n_active {na_f}->{na_r}", flush=True)
        return 0
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out_path.write_text(json.dumps(rec, indent=2))
        print(f"  ERROR: {rec['error']}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
