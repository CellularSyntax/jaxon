"""Adam-FD vs Adam-autodiff head-to-head on the IDENTICAL amplitude problem,
across a cohort of nerves.

This is the *clean* A/B test.  Both arms use the SAME optimizer (Adam), the
SAME cosine-LR schedule, the SAME global-norm gradient clip, the SAME init and
early-stop, on the SAME (soft) loss surface -- only the gradient source
differs (batched finite differences vs exact reverse-mode autodiff).  Any
SI difference is therefore attributable to the gradient estimator, not to
Adam-vs-L-BFGS (the confound in the earlier comparison, which made autodiff
collapse to SI=0 via line-search overshoot on the near-flat binary proxy) and
not to the loss surface.

We deliberately run on the SOFT activation proxy
(JAXLEY_FIBERS_SOFT_TEMPERATURE, default 0.15 set below): with the default
binary proxy the autodiff gradient is ~0 except in a narrow threshold window
-- exactly the saturation Hussain et al. (Nat Commun 2024) avoid by
optimizing a smooth m-gate quotient, not a thresholded activation.  The
hard selectivity index reported is always measured at threshold 0.5,
independent of the proxy used for the gradient.

Set H2H_AUTODIFF_OPT=lbfgs to additionally reproduce the fragile L-BFGS arm
for the record.

For one nerve (DUKE_SAMPLE_DIR), seed 0:
  1. smart-init magnitude probe -> a focal FIRING start (cohort init);
  2. run Adam-FD and Adam-autodiff from that SAME start, same fixed budget,
     early-stop disabled -> compare final SI, loss and wall-clock;
  3. write outputs/headtohead_fd_vs_autodiff/<nerve>.json.

Aggregate the per-nerve JSONs into a table:
    HEADTOHEAD_AGGREGATE=1 python -m experiments_v2.headtohead_fd_vs_autodiff

Run one nerve (normally driven by the SLURM array sbatch):
    DUKE_SAMPLE_DIR=duke_Ves/sub-11_sam-3 \
    python -m experiments_v2.headtohead_fd_vs_autodiff
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "headtohead_fd_vs_autodiff"


def _species(name: str) -> str:
    return "human" if name.startswith("human") else "swine"


# ── aggregate mode: read per-nerve JSONs, print + save a table (no JAX) ──────
def aggregate() -> int:
    import numpy as np
    rows = []
    for jp in sorted(OUT_DIR.glob("*.json")):
        if jp.name == "summary.json":
            continue
        try:
            rows.append(json.loads(jp.read_text()))
        except Exception:
            continue
    rows = [r for r in rows if r.get("ok")]
    if not rows:
        print(f"[aggregate] no usable results in {OUT_DIR}", file=sys.stderr)
        return 1

    print(f"\n{'nerve':22s} {'sp':5s} {'N':>5} {'startSI':>7} "
          f"{'FD SI':>6} {'LB SI':>6} {'dSI':>6} "
          f"{'FD s':>7} {'LB s':>7} {'LB/FD':>6}")
    for r in sorted(rows, key=lambda r: (r["species"], r["nerve"])):
        sr = (r["lb_time"] / r["fd_time"]) if r["fd_time"] else float("nan")
        print(f"{r['nerve']:22s} {r['species']:5s} {r['n_fibers']:>5d} "
              f"{r['start_si']:>+7.3f} {r['fd_si']:>+6.3f} {r['lb_si']:>+6.3f} "
              f"{r['fd_si']-r['lb_si']:>+6.3f} {r['fd_time']:>7.0f} "
              f"{r['lb_time']:>7.0f} {sr:>6.1f}")

    fd = np.array([r["fd_si"] for r in rows]); lb = np.array([r["lb_si"] for r in rows])
    sr = np.array([r["lb_time"] / r["fd_time"] for r in rows if r["fd_time"]])
    n = len(rows)
    summary = {
        "n_nerves": n,
        "fd_si_median": float(np.median(fd)),
        "lb_si_median": float(np.median(lb)),
        "dSI_median": float(np.median(fd - lb)),
        "frac_fd_ge_lb": float(np.mean(fd >= lb - 1e-9)),
        "speed_ratio_median_lb_over_fd": float(np.median(sr)),
    }
    print(f"\n[summary over {n} nerves]")
    print(f"  median SI:  FD {summary['fd_si_median']:+.3f}  vs  "
          f"LBFGS {summary['lb_si_median']:+.3f}   (median dSI = "
          f"{summary['dSI_median']:+.3f})")
    print(f"  FD >= autodiff on {100*summary['frac_fd_ge_lb']:.0f}% of nerves")
    print(f"  autodiff-LBFGS is {summary['speed_ratio_median_lb_over_fd']:.1f}x "
          f"slower (median wall-clock)")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  -> {OUT_DIR / 'summary.json'}")
    return 0


if os.environ.get("HEADTOHEAD_AGGREGATE", "").strip() in ("1", "true", "yes"):
    sys.exit(aggregate())


# ── per-nerve mode (imports the sweep pipeline; needs DUKE_SAMPLE_DIR) ────────
os.environ.setdefault("N_OPT_RECT", os.environ.get("HEADTOHEAD_STEPS", "15"))
os.environ.setdefault("N_RESTARTS_RECT", "1")
# Soft proxy so the loss is genuinely differentiable for BOTH arms.  Without
# this the binary proxy's gradient is ~0 except at threshold and the autodiff
# arm is being asked to descend a flat surface (the real reason the old LBFGS
# arm collapsed).  Read at import time by jaxfibers.optim.optimizer.
os.environ.setdefault("JAXLEY_FIBERS_SOFT_TEMPERATURE", "0.15")
# Which autodiff arm: 'adam' (clean A/B, default) or 'lbfgs' (fragile, for record).
_AUTODIFF_OPT = os.environ.get("H2H_AUTODIFF_OPT", "adam").strip().lower()

import numpy as np
import jax
import jax.numpy as jnp

from experiments_v2 import selectivity_sweep_duke as S
from jaxfibers.optim.optimizer import (
    run_rect_optimization,
    run_rect_optimization_autodiff,
    run_rect_optimization_lbfgs,
)
from jaxfibers.optim.losses import selectivity_index


def gpu_peak_mb() -> float:
    try:
        st = jax.devices()[0].memory_stats() or {}
        return float(st.get("peak_bytes_in_use", st.get("bytes_in_use", 0))) / 1e6
    except Exception:
        return float("nan")


def firing_start(seed_in):
    """Cohort smart init: focal bipolar + per-column tripolar magnitude probe."""
    sb = S._smart_spatial_pattern(seed_in["Ve_unit"], seed_in["target_mask"],
                                  verbose=False, label="")
    tri = S._sparse_tripolar_patterns_per_column(
        sb, seed_in["contact_xyz_um"], verbose=False, label="")
    patterns = {"bipolar": sb}
    patterns.update(tri)
    amps, best_mag, _score, acts, _hist, pattern = S._magnitude_probe(
        patterns, S.PROBE_MAGS_MA, seed_in["target_mask"],
        seed_in["fs_batch"], seed_in["s0_batch"],
        jnp.asarray(seed_in["Ve_unit"], jnp.float64),
        jnp.asarray(seed_in["pulse_mask"], jnp.float64),
        jnp.asarray(seed_in["node_indices"], jnp.int32),
        S.DT, label="", verbose=False,
    )
    si = float(selectivity_index(np.asarray(acts),
                                 np.asarray(seed_in["target_mask"], bool)))
    return np.asarray(amps), si, float(best_mag), str(pattern)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = S.SAMPLE_NAME
    out_path = OUT_DIR / f"{name}.json"
    print(f"[h2h] {name} ({_species(name)})", flush=True)

    rec = {"nerve": name, "species": _species(name), "ok": False}
    try:
        duke = S.load_duke_sample(
            S.SAMPLE_PATH, fiber_diam_um=S.FIBER_DIAMETER_UM, n_nodes=S.N_NODES,
            max_fibers=(S.MAX_FIBERS if S.MAX_FIBERS > 0 else None),
            subsample_seed=S.SUBSAMPLE_SEED, verbose=False,
        )
        seed_in = S._build_seed(duke, seed=0, verbose=False)
        if seed_in is None:
            rec["skip"] = "no valid target at seed 0"
            out_path.write_text(json.dumps(rec, indent=2))
            print(f"  SKIP: {rec['skip']}", flush=True)
            return 0

        tgt = np.asarray(seed_in["target_mask"], bool)
        N = int(tgt.size); K = int(seed_in["Ve_unit"].shape[0])
        T = int(seed_in["N_STEPS"])
        amps_start, start_si, best_mag, pattern = firing_start(seed_in)
        print(f"  start: pattern={pattern} mag={best_mag:+.2f} mA SI={start_si:+.3f} "
              f"(N={N}, targets={tgt.sum()})", flush=True)

        budget = S.N_OPT_RECT
        NEVER = 10 ** 9
        common = dict(
            fiber_statics_batch=seed_in["fs_batch"], state0_batch=seed_in["s0_batch"],
            Ve_unit=seed_in["Ve_unit"], pulse_mask=seed_in["pulse_mask"],
            node_indices=seed_in["node_indices"], target_mask=seed_in["target_mask"],
            weights=seed_in["weights"], dt=S.DT, amp_clip=S.AMP_CLIP, verbose=False,
        )

        t0 = time.time()
        fd = run_rect_optimization(
            **common, n_steps=budget, amps_init_vector=amps_start,
            lr=S.ADAM_LR_MA, fd_eps=S.FD_EPS_MA,
            early_stop_si=2.0, early_stop_patience=NEVER, early_stop_si_patience=0,
        )
        jax.block_until_ready(fd["amps"]); fd_t = time.time() - t0
        hl = np.asarray(fd["history"]["loss"]); hs = np.asarray(fd["history"]["si"])
        i = int(np.argmin(hl)); fd_si = float(hs[i]); fd_loss = float(hl[i])

        t0 = time.time()
        if _AUTODIFF_OPT == "lbfgs":
            ad = run_rect_optimization_lbfgs(
                **common, n_restarts=1, n_steps=budget,
                amp_init_mA=S.AMP_INIT_MA, rng_seed=0, amps_init_vector=amps_start,
            )
            jax.block_until_ready(ad["final_acts"]); lb_t = time.time() - t0
            lb_si = float(selectivity_index(np.asarray(ad["final_acts"]), tgt))
            lb_loss = float(ad["final_loss"])
            ad_label = "LBFGS"
        else:
            ad = run_rect_optimization_autodiff(
                **common, n_steps=budget, amps_init_vector=amps_start,
                lr=S.ADAM_LR_MA,
                early_stop_si=2.0, early_stop_patience=NEVER, early_stop_si_patience=0,
            )
            jax.block_until_ready(ad["history"]["acts"][-1]); lb_t = time.time() - t0
            hl = np.asarray(ad["history"]["loss"]); hs = np.asarray(ad["history"]["si"])
            j = int(np.argmin(hl)); lb_si = float(hs[j]); lb_loss = float(hl[j])
            ad_label = "AD-Adam"

        rec.update(
            ok=True, n_fibers=N, K=K, T=T, n_target=int(tgt.sum()),
            start_si=start_si, best_mag=best_mag, pattern=pattern, budget=int(budget),
            autodiff_opt=_AUTODIFF_OPT, soft_temperature=float(
                os.environ.get("JAXLEY_FIBERS_SOFT_TEMPERATURE", "0.0")),
            fd_si=fd_si, fd_loss=fd_loss, fd_time=fd_t,
            lb_si=lb_si, lb_loss=lb_loss, lb_time=lb_t,
            dSI=fd_si - lb_si, peak_mb=gpu_peak_mb(),
        )
        out_path.write_text(json.dumps(rec, indent=2))
        print(f"  FD  SI={fd_si:+.3f} ({fd_t:.0f}s)   "
              f"{ad_label} SI={lb_si:+.3f} ({lb_t:.0f}s)   dSI={fd_si-lb_si:+.3f}",
              flush=True)
        return 0
    except Exception as e:                # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out_path.write_text(json.dumps(rec, indent=2))
        print(f"  ERROR: {rec['error']}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
