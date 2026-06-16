"""Frozen-tripole vs free-all-contacts sparsity + FD-vs-autodiff speedup.

Two questions, one sweep, per nerve (warm probe init, charge-balanced cuff):

  SCIENCE -- frozen vs free selectivity + deployment penalty
    for variant in {frozen, free}:
      dense_si    = optimize on the FULL population, score on it       (ceiling)
      transfer_si = optimize on a SPARSE subsample (1 fiber/fascicle),
                    then re-score those amps on the FULL population
      penalty     = dense_si - transfer_si       (the paper's deployment penalty)
    Only the freeze differs; optimizer/schedule/init/loss held identical.  A free
    variant that lifts dense_si but inflates penalty is OVERFITTING the sampled
    fibers -- which reframes frozen sparsity as protective regularization.

  METHODS -- gradient-source speedup (same workload, free arm, K contacts)
    The free arm is run with BOTH gradient sources:
      FD       (run_rect_optimization)          -- K+1 forward passes / step
      autodiff (run_rect_optimization_autodiff) -- 1 forward + 1 backward / step
    Both are gradient descent (Adam); only the gradient source differs.  We log
    wall-clock per nerve and confirm SI parity, so the speedup is attributable to
    the gradient source, not the optimizer.  (The frozen arm is FD-only: the
    autodiff path has no contact-freeze support, and the K=3 frozen case is the
    regime where FD's K+1 forwards is cheap anyway.)

Charge balance (balance_currents): the per-contact currents are projected onto
sum=0 each step -- the physical multipolar-cuff Kirchhoff constraint.  Essential
for the FREE arm: without it the unfrozen optimiser drifts into an unphysical
all-same-sign monopolar field that fires the whole nerve.  FVR_BALANCE=0 to off.

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
# Output root is configurable so a fresh full-dataset run (replacing the FD
# results in the manuscript) can land in e.g. outputs_new without touching the
# old outputs/.  Set FVR_OUT_ROOT=outputs_new.  The aggregator reads the same
# env, so pass it there too.
OUT_DIR = ROOT / os.environ.get("FVR_OUT_ROOT", "outputs") / "frozen_vs_relaxed"

# Degenerate-target threshold.  A position whose TARGET FRACTION exceeds this is
# ill-posed -- almost no off-target to spare, so selectivity is impossible for
# ANY method (e.g. the 96%-target nerve) -- and is excluded from the reported
# medians.  This is a property of the target DEFINITION, not the achieved SI:
# filtering on SI>=0.5 instead would also silently drop solver failures on
# SELECTABLE targets, biasing the result optimistically.  Defined before the
# aggregate early-exit so _summarize can use it.
_MAX_TGT_FRAC = float(os.environ.get("FVR_MAX_TGT_FRAC", "0.6"))


def _species(name: str) -> str:
    return "human" if name.startswith("human") else "swine"


# ── aggregate mode: read per-nerve JSONs, print per-species summary (no JAX) ──
def _summarize(rows, label) -> dict | None:
    """Print the science + methods tables and per-species medians for one
    config's rows; return its summary dict (or None if no usable rows)."""
    import numpy as np
    rows = [r for r in rows if r.get("ok")]
    if not rows:
        return None

    # Headline group (free-autodiff, position sweep): each per-nerve record
    # carries a per-position 'seeds' list -- no frozen/free-FD arms.  Print the
    # lean per-position free-AD selectivity table instead of frozen-vs-free.
    if all(r.get("seeds") is not None for r in rows):
        print(f"\n===== config: {label}   ({len(rows)} nerves, free-AD headline) =====")
        print(f"  excluding positions with target fraction > {100*_MAX_TGT_FRAC:.0f}% "
              f"(degenerate / ill-posed)")
        print(f"{'nerve':22s} {'sp':5s} {'pos':>3} {'N':>6} {'tgt%':>5} "
              f"{'freeAD_SI':>9} {'ms/iter':>8} {'n_act':>5}")
        flat = []          # (species, nerve, si) -- well-posed positions only
        n_degen = n_unknown = 0
        for r in sorted(rows, key=lambda r: (r["species"], r["nerve"])):
            for sd in sorted(r.get("seeds", []), key=lambda d: d.get("position_idx", 0)):
                si = sd.get("dense_si_relaxed_autodiff")
                ms = sd.get("ms_iter_relaxed_autodiff")
                na = sd.get("n_active_relaxed_autodiff")
                frac = sd.get("target_fraction")
                # degenerate flag, or recompute from fraction; None = pre-fix record
                if frac is None and sd.get("degenerate") is None:
                    degen, mark, n_unknown = None, "?", n_unknown + 1
                else:
                    degen = bool(sd.get("degenerate")
                                 if sd.get("degenerate") is not None
                                 else frac > _MAX_TGT_FRAC)
                    mark = "X" if degen else " "
                print(f"{r['nerve']:22s} {r['species']:5s} "
                      f"{sd.get('position_idx', -1):>3} {sd.get('n_fibers', 0):>6} "
                      f"{(100*frac if frac is not None else float('nan')):>4.0f}{mark} "
                      f"{(si if si is not None else float('nan')):>+9.3f} "
                      f"{(ms if ms is not None else float('nan')):>8.0f} "
                      f"{(na if na is not None else -1):>5d}")
                if si is None:
                    continue
                if degen:
                    n_degen += 1
                else:                      # well-posed (or unknown -> kept, flagged)
                    flat.append((r["species"], r["nerve"], float(si)))
        summary = {"n_nerves": len(rows), "max_tgt_frac": _MAX_TGT_FRAC,
                   "n_excluded_degenerate": n_degen,
                   "n_positions_missing_fraction": n_unknown, "by_species": {}}
        print(f"  excluded {n_degen} degenerate position(s)"
              + (f"; {n_unknown} pre-fix position(s) have no target_fraction "
                 f"(not excluded -- backfill/rerun to filter them)" if n_unknown else ""))
        print("  [medians, well-posed positions only]")
        for sp in ("swine", "human"):
            pts = [(nv, si) for (s, nv, si) in flat if s == sp]
            if not pts:
                continue
            pooled = float(np.median([si for _, si in pts]))
            frac_ok = float(np.mean([si >= 0.5 for _, si in pts]))   # success-rate stat
            nerves = sorted({nv for nv, _ in pts})
            per_nerve = [float(np.median([si for nv2, si in pts if nv2 == nv]))
                         for nv in nerves]
            mnn = float(np.median(per_nerve))
            summary["by_species"][sp] = {
                "n_nerves": len(nerves), "n_positions": len(pts),
                "free_ad_si_pooled_median": pooled,
                "free_ad_si_per_nerve_median": mnn,
                "frac_positions_si_ge_0p5": frac_ok,
            }
            print(f"  {sp:5s}: {len(nerves)} nerves, {len(pts)} positions  "
                  f"free-AD SI median {pooled:+.3f} (pooled) / {mnn:+.3f} (per-nerve)  "
                  f"[{100*frac_ok:.0f}% reach SI>=0.5]")
        return summary

    NAN = float("nan")
    # Coerce missing transfer/penalty/timing keys to NaN so formatting +
    # nan-aware medians still work (sparse leg or autodiff arm may be absent).
    soft_keys = ("transfer_si_frozen", "transfer_si_relaxed",
                 "penalty_frozen", "penalty_relaxed",
                 "dense_si_relaxed_autodiff", "t_dense_relaxed_fd",
                 "t_dense_relaxed_autodiff", "ms_iter_relaxed_fd",
                 "ms_iter_relaxed_autodiff", "speedup_fd_over_autodiff",
                 "si_parity_autodiff_minus_fd", "n_active_relaxed_autodiff")
    for r in rows:
        for k in soft_keys:
            if r.get(k) is None:
                r[k] = NAN

    def _xvar(r):  # dense_si_frozen - transfer_si_relaxed (cross-variant)
        return r["dense_si_frozen"] - r["transfer_si_relaxed"]

    # ---- SCIENCE: frozen vs free selectivity + deployment penalty ----
    print(f"\n===== config: {label}   ({len(rows)} nerves) =====")
    print("  [science] frozen vs free  (denseR_ad = free arm, autodiff grad)")
    print(f"{'nerve':22s} {'sp':5s} {'N':>5}  "
          f"{'denseF':>6} {'denseR':>6} {'dnsRad':>6}  "
          f"{'transF':>6} {'transR':>6}  {'penF':>6} {'penR':>6}  "
          f"{'Fd-Rt':>6}  {'naF':>3} {'naR':>3}")
    for r in sorted(rows, key=lambda r: (r["species"], r["nerve"])):
        print(f"{r['nerve']:22s} {r['species']:5s} {r['n_fibers']:>5d}  "
              f"{r['dense_si_frozen']:>+6.3f} {r['dense_si_relaxed']:>+6.3f} "
              f"{r['dense_si_relaxed_autodiff']:>+6.3f}  "
              f"{r['transfer_si_frozen']:>+6.3f} {r['transfer_si_relaxed']:>+6.3f}  "
              f"{r['penalty_frozen']:>+6.3f} {r['penalty_relaxed']:>+6.3f}  "
              f"{_xvar(r):>+6.3f}  "
              f"{r['n_active_frozen']:>3d} {r['n_active_relaxed']:>3d}")

    # ---- METHODS: FD vs autodiff wall-clock on the free arm ----
    has_timing = any(np.isfinite(r["ms_iter_relaxed_autodiff"]) for r in rows)
    if has_timing:
        print("\n  [methods] FD vs autodiff (free arm, per-iter ms, compile excluded)")
        print(f"{'nerve':22s} {'sp':5s} {'N':>5}  "
              f"{'msFD':>8} {'msAD':>8} {'speedup':>7}  "
              f"{'siFD':>6} {'siAD':>6} {'dSI':>6}")
        for r in sorted(rows, key=lambda r: (r["species"], r["nerve"])):
            print(f"{r['nerve']:22s} {r['species']:5s} {r['n_fibers']:>5d}  "
                  f"{r['ms_iter_relaxed_fd']:>8.0f} {r['ms_iter_relaxed_autodiff']:>8.0f} "
                  f"{r['speedup_fd_over_autodiff']:>6.2f}x  "
                  f"{r['dense_si_relaxed']:>+6.3f} {r['dense_si_relaxed_autodiff']:>+6.3f} "
                  f"{r['si_parity_autodiff_minus_fd']:>+6.3f}")

    summary = {"n_nerves": len(rows), "by_species": {}}
    for sp in ("swine", "human"):
        sr = [r for r in rows if r["species"] == sp]
        if not sr:
            continue
        def med(key, sr=sr):
            return float(np.nanmedian([r[key] for r in sr]))
        def med_expr(fn, sr=sr):
            return float(np.nanmedian([fn(r) for r in sr]))
        summary["by_species"][sp] = {
            "n": len(sr),
            "dense_si_frozen":          med("dense_si_frozen"),
            "dense_si_relaxed":         med("dense_si_relaxed"),
            "dense_si_relaxed_autodiff": med("dense_si_relaxed_autodiff"),
            "transfer_si_frozen":       med("transfer_si_frozen"),
            "transfer_si_relaxed":      med("transfer_si_relaxed"),
            "penalty_frozen":           med("penalty_frozen"),
            "penalty_relaxed":          med("penalty_relaxed"),
            "frozenDense_minus_relaxedTransfer": med_expr(_xvar),
            "n_active_relaxed":         med("n_active_relaxed"),
            "ms_iter_relaxed_fd":       med("ms_iter_relaxed_fd"),
            "ms_iter_relaxed_autodiff": med("ms_iter_relaxed_autodiff"),
            "speedup_fd_over_autodiff": med("speedup_fd_over_autodiff"),
            "si_parity_autodiff_minus_fd": med("si_parity_autodiff_minus_fd"),
        }
    print(f"  [medians]")
    for sp, s in summary["by_species"].items():
        print(f"  {sp:5s} (n={s['n']:2d}):")
        print(f"      dense SI   frozen {s['dense_si_frozen']:+.3f} "
              f"-> free {s['dense_si_relaxed']:+.3f}")
        print(f"      transfer   frozen {s['transfer_si_frozen']:+.3f} "
              f"-> free {s['transfer_si_relaxed']:+.3f}   "
              f"penalty  frozen {s['penalty_frozen']:+.3f} -> free {s['penalty_relaxed']:+.3f}")
        print(f"      cross-variant  frozen-dense - free-transfer = "
              f"{s['frozenDense_minus_relaxedTransfer']:+.3f}   "
              f"(>0: frozen headline beats free reduced-order on the dense nerve)")
        if np.isfinite(s["speedup_fd_over_autodiff"]):
            print(f"      per-iter (free)  FD {s['ms_iter_relaxed_fd']:.0f}ms / "
                  f"autodiff {s['ms_iter_relaxed_autodiff']:.0f}ms = "
                  f"{s['speedup_fd_over_autodiff']:.2f}x   "
                  f"(>1: autodiff faster per step)   "
                  f"SI parity (AD-FD) {s['si_parity_autodiff_minus_fd']:+.3f}")
    return summary


def aggregate() -> int:
    groups: dict[str, list] = {}
    for jp in sorted(OUT_DIR.rglob("*.json")):
        if jp.name == "summary.json":
            continue
        label = jp.parent.name if jp.parent != OUT_DIR else "(root)"
        try:
            groups.setdefault(label, []).append(json.loads(jp.read_text()))
        except Exception:
            continue
    if not groups:
        print(f"[aggregate] no usable results in {OUT_DIR}", file=sys.stderr)
        return 1
    all_summaries = {}
    for label in sorted(groups):
        s = _summarize(groups[label], label)
        if s is not None:
            all_summaries[label] = s
    (OUT_DIR / "summary.json").write_text(json.dumps(all_summaries, indent=2))
    print(f"\n  -> {OUT_DIR / 'summary.json'}  ({len(all_summaries)} config(s))")
    return 0


if os.environ.get("FROZEN_VS_RELAXED_AGGREGATE", "").strip() in ("1", "true", "yes"):
    sys.exit(aggregate())


# ── per-nerve mode (needs DUKE_SAMPLE_DIR) ────────────────────────────────────
os.environ.setdefault("N_OPT_RECT", os.environ.get("FVR_STEPS", "15"))
os.environ.setdefault("JAXLEY_FIBERS_SOFT_TEMPERATURE", "0.15")
os.environ.setdefault("JAXLEY_FIBERS_ENERGY_LAMBDA", "1e-3")
# Sparse density for the transfer leg (paper's reduced-order = 1 fiber/fascicle).
_N_PER_FASC = int(os.environ.get("FVR_N_PER_FASC", "1"))
# Init for the FREE arm: 'warm' (probe-selected start, default) or 'zero'
# (cold start, no probe).  The FROZEN arm is always warm.
_INIT_MODE  = os.environ.get("INIT_MODE", "warm").strip().lower()
_ZERO_LR    = float(os.environ.get("ZERO_LR_MA", "0.1"))
_ZERO_STEPS = int(os.environ.get("ZERO_STEPS", "200"))
_ZERO_LR_DECAY = float(os.environ.get("ZERO_LR_DECAY", "0.6"))
_ZERO_SI_FLOOR = float(os.environ.get("ZERO_SI_FLOOR", "0.5"))
_ZERO_PATIENCE = int(os.environ.get("ZERO_PATIENCE", "10"))
_VERBOSE = os.environ.get("FVR_VERBOSE", "0").strip() in ("1", "true", "yes")
# Charge-balance (Kirchhoff) projection: sum(contact currents)=0 each step.
# Essential for the FREE arm (no monopolar exploit); applied to both arms over
# the active/non-frozen contacts.  FVR_BALANCE=0 to disable.
_BALANCE = os.environ.get("FVR_BALANCE", "1").strip() not in ("0", "false", "")
# Fixed dense step count, early stop disabled, so FD-vs-autodiff wall-clock is a
# clean per-step comparison (both run the same number of iterations).
_DENSE_STEPS = int(os.environ.get("FVR_DENSE_STEPS", "45"))
# Run mode:
#   'speed'    -> all 3 dense arms (frozen-FD, free-FD, free-autodiff) + the
#                 transfer leg.  Gives frozen-vs-free science AND the FD-vs-
#                 autodiff per-step speedup/parity.  Use for the 6+6 perf study.
#   'headline' -> ONLY the free-autodiff dense arm (no FD arms, no transfer).
#                 ~1/3 the cost; the autodiff warm-start selectivity result that
#                 replaces the FD headline.  Use for the all-36 run.  Lands in a
#                 separate <cfg>_adheadline subdir so it never collides with a
#                 speed run of the same nerve.
_MODE = os.environ.get("FVR_MODE", "speed").strip().lower()
# Headline mode sweeps the cluster POSITIONS via seeds (pos_idx = seed %
# CLUSTER_N_POSITIONS).  The FD headline ran each of the 4 positions once, so the
# default is seeds 0..3.  Speed mode ignores this (single position, seed 0).
_SEED_START = int(os.environ.get("SEED_START", "0"))
_SEED_END   = int(os.environ.get("SEED_END", "4"))
# Stale-lock age (s): a claim left by a killed worker older than this is stolen
# so the nerve is not orphaned.  Set comfortably above the per-nerve runtime.
_LOCK_STALE_S = float(os.environ.get("FVR_LOCK_STALE_S", "7200"))


def _claim(lock: Path) -> bool:
    """Atomically claim a nerve via an exclusive lock file; steal it if it is a
    stale (> _LOCK_STALE_S) leftover from a killed worker.  Returns True iff this
    worker now owns the lock.  Lets any number of concurrent workers on any
    partition (a100, b200, ...) divide the work without coordination."""
    try:
        os.close(os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        try:
            age = time.time() - lock.stat().st_mtime
        except OSError:
            return False
        if age <= _LOCK_STALE_S:
            return False
        try:                      # steal the stale lock
            lock.unlink()
            os.close(os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return True
        except (OSError, FileExistsError):
            return False

import numpy as np
import jax
import jax.numpy as jnp

from experiments_v2 import selectivity_sweep_duke as S
from jaxfibers.optim.optimizer import (
    run_rect_optimization, run_rect_optimization_autodiff,
)
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


def _optimize(seed_in, relaxed: bool, grad_mode: str = "fd",
              n_steps: int | None = None, warm_init=None,
              early_stop: bool = False, verbose: bool = False):
    """Adam selectivity optimization, charge-balanced.

    relaxed=False (frozen): probe warm start, zero contacts frozen.  FD only.
    relaxed=True  (free):   all K contacts free.
        INIT_MODE=warm -> probe warm start (reused via warm_init if provided).
        INIT_MODE=zero -> cold start (all 0 mA, no probe), ReduceLROnPlateau ramp.
    grad_mode 'fd' -> run_rect_optimization (finite-difference gradient).
    grad_mode 'autodiff' -> run_rect_optimization_autodiff (exact reverse-mode);
        no freeze support, so the frozen arm must use 'fd'.

    Returns (amps, in-sample SI, n_active, pattern, t_opt_seconds,
    ms_per_iter) where ms_per_iter is the MEDIAN per-iteration wall time with
    the first (XLA-compile) iteration excluded -- the honest per-step cost for
    the FD-vs-autodiff comparison (total wall-clock is compile-contaminated,
    which penalises the autodiff graph)."""
    cold = relaxed and _INIT_MODE == "zero"
    if cold:
        amps_init = None; pattern = "zero"; freeze = None
        lr = _ZERO_LR
        n_iters = n_steps if n_steps is not None else _ZERO_STEPS
        amp_init_mA = 0.0
    else:
        if warm_init is not None:
            amps_init, pattern = warm_init
        else:
            amps_init, pattern, _mag = _probe_init(seed_in)
        freeze = None if relaxed else (np.abs(amps_init) < 1e-9)
        lr = S.ADAM_LR_MA
        n_iters = (n_steps if n_steps is not None
                   else max(int(os.environ.get("N_OPT_RECT", "15")) * 3, 30))
        amp_init_mA = S.AMP_INIT_MA

    if grad_mode == "autodiff" and freeze is not None:
        raise ValueError("autodiff path has no freeze support; the frozen arm "
                         "must use grad_mode='fd'")

    plateau = (dict(lr_mode="plateau", plateau_lr_decay=_ZERO_LR_DECAY,
                    plateau_si_floor=_ZERO_SI_FLOOR, plateau_patience=_ZERO_PATIENCE)
               if cold else dict())
    # early_stop=False (speed timing): fixed n_iters, never stop, so the FD/AD
    # wall-clock is a clean per-step comparison.  early_stop=True (headline SI
    # sweep): stop once selective / stagnant -- matches the FD headline's
    # EARLY_STOP_SI and keeps the 4-position sweep tractable.  Either way the
    # reported SI is hard-best over the trajectory, so it's unaffected.
    es = (dict(early_stop_si=S.EARLY_STOP_SI, early_stop_patience=20,
               early_stop_si_patience=10)
          if early_stop
          else dict(early_stop_si=2.0, early_stop_patience=10**9,
                    early_stop_si_patience=0))
    common = dict(
        fiber_statics_batch=seed_in["fs_batch"], state0_batch=seed_in["s0_batch"],
        Ve_unit=seed_in["Ve_unit"], pulse_mask=seed_in["pulse_mask"],
        node_indices=seed_in["node_indices"], target_mask=seed_in["target_mask"],
        weights=seed_in["weights"], dt=S.DT, n_steps=n_iters,
        amps_init_vector=amps_init, amp_init_mA=amp_init_mA, amp_clip=S.AMP_CLIP,
        lr=lr, balance_currents=_BALANCE, verbose=verbose, **es,
    )
    t0 = time.time()
    if grad_mode == "autodiff":
        res = run_rect_optimization_autodiff(**common, **plateau)
    else:
        res = run_rect_optimization(
            **common, fd_eps=S.FD_EPS_SMART_MA, freeze_zero_mask=freeze, **plateau)
    t_opt = time.time() - t0

    # Hard-best: highest SI, tiebroken by lowest loss (Hussain WBCE/WQ style).
    loss = np.asarray(res["history"]["loss"])
    si_hist = np.asarray(res["history"]["si"])
    i = int(np.lexsort((loss, -si_hist))[0])
    amps = np.asarray(res["history"]["amps"][i])
    acts = np.asarray(res["history"]["acts"][i])
    si = float(selectivity_index(acts, np.asarray(seed_in["target_mask"], bool)))
    n_active = int(np.sum(np.abs(amps) > 1e-6))
    dt_ms = list(res["history"].get("dt_ms", []))
    ms_per_iter = (float(np.median(dt_ms[1:])) if len(dt_ms) > 1
                   else (float(dt_ms[0]) if dt_ms else float("nan")))
    return amps, si, n_active, pattern, t_opt, ms_per_iter


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


def _backfill(out_path: Path) -> int:
    """Patch an existing headline JSON in place with per-position target_fraction
    (+ n_target/n_off + degenerate), WITHOUT re-optimising -- for records written
    before the fraction was recorded.  Loads geometry only; idempotent (a record
    already carrying target_fraction is skipped before the load).  Run with the
    SAME MAX_FIBERS as the original so the rebuilt seed matches what was optimised."""
    name = S.SAMPLE_NAME
    if not out_path.exists():
        print(f"[backfill] {name}: no JSON, skip", flush=True)
        return 0
    try:
        rec = json.loads(out_path.read_text())
    except Exception:
        print(f"[backfill] {name}: unreadable JSON, skip", flush=True)
        return 0
    seeds = rec.get("seeds")
    if not rec.get("ok") or not seeds:
        print(f"[backfill] {name}: nothing to backfill", flush=True)
        return 0
    if all(sd.get("target_fraction") is not None for sd in seeds):
        print(f"[backfill] {name}: already has target_fraction", flush=True)
        return 0
    duke = S.load_duke_sample(
        S.SAMPLE_PATH, fiber_diam_um=S.FIBER_DIAMETER_UM, n_nodes=S.N_NODES,
        max_fibers=(S.MAX_FIBERS if S.MAX_FIBERS > 0 else None),
        subsample_seed=S.SUBSAMPLE_SEED, verbose=False)
    n_patched = 0
    for sd in seeds:
        dense = S._build_seed(duke, seed=int(sd["seed"]), verbose=False)
        if dense is None:
            continue
        tm = np.asarray(dense["target_mask"], bool)
        frac = float(tm.sum()) / max(tm.size, 1)
        sd["target_fraction"] = frac
        sd["n_target"] = int(tm.sum()); sd["n_off"] = int((~tm).sum())
        sd["degenerate"] = bool(frac > _MAX_TGT_FRAC)
        n_patched += 1
    out_path.write_text(json.dumps(rec, indent=2))
    print(f"[backfill] {name}: patched {n_patched}/{len(seeds)} positions", flush=True)
    return 0


def _run_headline(duke, name: str, out_path: Path, loss_mode: str) -> int:
    """Headline mode: free-autodiff selectivity swept over the cluster positions
    (seeds _SEED_START.._SEED_END-1), one optimisation per position, early-stop
    on.  Writes one per-nerve JSON with a per-position 'seeds' list.  This is the
    autodiff warm-start, charge-balanced result that replaces the FD dense
    headline."""
    print(f"  headline free-AD sweep: seeds {_SEED_START}..{_SEED_END - 1} "
          f"({_SEED_END - _SEED_START} positions)", flush=True)
    seed_rows = []
    K = None
    for s in range(_SEED_START, _SEED_END):
        dense = S._build_seed(duke, seed=s, verbose=False)
        if dense is None:
            print(f"  seed {s}: no valid target at this position, skip", flush=True)
            continue
        tm = np.asarray(dense["target_mask"], bool)
        N = int(tm.size)
        frac = float(tm.sum()) / max(N, 1)
        degen = bool(frac > _MAX_TGT_FRAC)
        K = int(dense["Ve_unit"].shape[0])
        warm = None
        if _INIT_MODE != "zero":
            pa = _probe_init(dense)
            warm = (pa[0], pa[1])
        _ad, si, na, patt, t_ad, ms_ad = _optimize(
            dense, relaxed=True, grad_mode="autodiff", n_steps=_DENSE_STEPS,
            warm_init=warm, early_stop=True, verbose=_VERBOSE)
        pos = int(s % S.CLUSTER_N_POSITIONS)
        print(f"  seed {s} (pos {pos}): N={N}  tgt={100*frac:.0f}%"
              f"{' DEGENERATE' if degen else ''}  free-AD SI {si:+.3f}  "
              f"({ms_ad:.0f} ms/iter, n_active {na})", flush=True)
        seed_rows.append(dict(
            seed=s, position_idx=pos, n_fibers=N, pattern=patt,
            target_fraction=frac, n_target=int(tm.sum()), n_off=int((~tm).sum()),
            degenerate=degen,
            dense_si_relaxed_autodiff=si, ms_iter_relaxed_autodiff=ms_ad,
            n_active_relaxed_autodiff=na, t_dense_relaxed_autodiff=t_ad))

    rec = {"nerve": name, "species": _species(name)}
    if not seed_rows:
        rec.update(ok=False, skip="no valid target at any swept position")
        out_path.write_text(json.dumps(rec, indent=2))
        print(f"  SKIP: {rec['skip']}", flush=True)
        return 0
    si_vals = [r["dense_si_relaxed_autodiff"] for r in seed_rows]
    rec.update(
        ok=True, mode="headline", K=K, init_mode=_INIT_MODE, loss_mode=loss_mode,
        balance=bool(_BALANCE), dense_steps=_DENSE_STEPS,
        seed_start=_SEED_START, seed_end=_SEED_END, n_seeds=len(seed_rows),
        soft_temperature=float(os.environ["JAXLEY_FIBERS_SOFT_TEMPERATURE"]),
        energy_lambda=float(os.environ["JAXLEY_FIBERS_ENERGY_LAMBDA"]),
        seeds=seed_rows,
        dense_si_relaxed_autodiff_median=float(np.median(si_vals)),
    )
    out_path.write_text(json.dumps(rec, indent=2))
    print(f"  [headline] {len(seed_rows)} position(s), median free-AD SI "
          f"{float(np.median(si_vals)):+.3f}", flush=True)
    return 0


def main() -> int:
    name = S.SAMPLE_NAME
    # Per-config sub-directory so different configurations never overwrite each
    # other and the aggregator can group them.
    loss_mode = os.environ.get("JAXLEY_FIBERS_LOSS", "linear")
    cfg = f"{_INIT_MODE}_{loss_mode}_bal{int(_BALANCE)}"
    if _MODE in ("headline", "backfill"):
        cfg += "_adheadline"   # free-autodiff-only; never collide with a speed run
    # Keep downsampled checks (MAX_FIBERS>0) in their own subdir so they can
    # never shadow / collide with the full-population run via the claim logic.
    if S.MAX_FIBERS and S.MAX_FIBERS > 0:
        cfg += f"_n{S.MAX_FIBERS}"
    cfg_dir = OUT_DIR / cfg
    cfg_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg_dir / f"{name}.json"

    # Backfill mode: patch target_fraction into an existing headline JSON without
    # re-optimising (bypasses the claim/already-done logic on purpose).
    if _MODE == "backfill":
        return _backfill(out_path)

    # Claim-based work distribution (see _claim).  A finished or intentionally
    # skipped nerve is terminal; a previously errored one is left for retry.
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            if prev.get("ok") or prev.get("skip"):
                print(f"[fvr] {name}: already done, skip", flush=True)
                return 0
        except Exception:
            pass
    _lock = cfg_dir / f".{name}.lock"
    if not _claim(_lock):
        print(f"[fvr] {name}: claimed by another worker, skip", flush=True)
        return 0

    print(f"[fvr] {name} ({_species(name)})  config={cfg}  "
          f"balance={'sum0' if _BALANCE else 'off'}  dense_steps={_DENSE_STEPS}",
          flush=True)
    rec = {"nerve": name, "species": _species(name), "ok": False}
    try:
        duke = S.load_duke_sample(
            S.SAMPLE_PATH, fiber_diam_um=S.FIBER_DIAMETER_UM, n_nodes=S.N_NODES,
            max_fibers=(S.MAX_FIBERS if S.MAX_FIBERS > 0 else None),
            subsample_seed=S.SUBSAMPLE_SEED, verbose=False,
        )

        # Headline = free-autodiff swept over the cluster positions (its own
        # per-seed path); speed = single position (seed 0), 3-arm + transfer.
        if _MODE == "headline":
            return _run_headline(duke, name, out_path, loss_mode)

        dense = S._build_seed(duke, seed=0, verbose=False)
        if dense is None:
            rec["skip"] = "no valid target at seed 0"
            out_path.write_text(json.dumps(rec, indent=2))
            print(f"  SKIP: {rec['skip']}", flush=True)
            return 0
        N = int(np.asarray(dense["target_mask"]).size)
        K = int(dense["Ve_unit"].shape[0])
        n_arms = 1 if _MODE == "headline" else 3
        print(f"  loaded: N={N} fibers, K={K} contacts; mode={_MODE}; "
              f"{n_arms} dense arm(s) x {_DENSE_STEPS} steps "
              f"(verbose={'on' if _VERBOSE else 'off'})", flush=True)

        # Probe warm start once; reuse for all dense arms so the FD/autodiff free
        # arms start identically (SI parity meaningful, timing isolates grad).
        warm = None
        if not (_INIT_MODE == "zero"):
            print("  computing probe warm init ...", flush=True)
            _tp = time.time()
            pa = _probe_init(dense)
            warm = (pa[0], pa[1])
            print(f"    probe done ({time.time()-_tp:.0f}s)  pattern={pa[1]}", flush=True)

        # Defaults (None for any arm not run in this mode).
        dsi_f = na_f = t_f = ms_f = None
        dsi_r = na_r = t_r_fd = ms_r_fd = None
        dsi_r_ad = na_r_ad = t_r_ad = ms_r_ad = None
        patt = warm[1] if warm else "zero"

        if _MODE == "speed":
            #   frozen  : FD, zero contacts frozen   (science vs free)
            #   free FD : FD, all K free             (science + FD timing)
            #   free AD : autodiff, all K free       (autodiff timing)
            print(f"  [1/3] dense frozen (FD) ...", flush=True)
            _af, dsi_f, na_f, patt, t_f, ms_f = _optimize(
                dense, relaxed=False, grad_mode="fd",
                n_steps=_DENSE_STEPS, warm_init=warm, verbose=_VERBOSE)
            print(f"    frozen SI {dsi_f:+.3f}  ({t_f:.0f}s total, {ms_f:.0f} ms/iter)",
                  flush=True)
            print(f"  [2/3] dense free (FD) ...", flush=True)
            _ar, dsi_r, na_r, _, t_r_fd, ms_r_fd = _optimize(
                dense, relaxed=True, grad_mode="fd",
                n_steps=_DENSE_STEPS, warm_init=warm, verbose=_VERBOSE)
            print(f"    free SI {dsi_r:+.3f}  ({t_r_fd:.0f}s total, {ms_r_fd:.0f} ms/iter)",
                  flush=True)
            print(f"  [3/3] dense free (autodiff) ...", flush=True)
            try:
                _ad, dsi_r_ad, na_r_ad, _, t_r_ad, ms_r_ad = _optimize(
                    dense, relaxed=True, grad_mode="autodiff",
                    n_steps=_DENSE_STEPS, warm_init=warm, verbose=_VERBOSE)
                print(f"    free-AD SI {dsi_r_ad:+.3f}  ({t_r_ad:.0f}s total, "
                      f"{ms_r_ad:.0f} ms/iter)", flush=True)
            except Exception as e:  # noqa: BLE001 -- autodiff arm is non-fatal
                print(f"  autodiff arm failed: {type(e).__name__}: {str(e)[:120]}",
                      flush=True)
        else:  # headline: free-autodiff only (the result that replaces FD)
            print(f"  [1/1] dense free (autodiff) [headline] ...", flush=True)
            _ad, dsi_r_ad, na_r_ad, patt, t_r_ad, ms_r_ad = _optimize(
                dense, relaxed=True, grad_mode="autodiff",
                n_steps=_DENSE_STEPS, warm_init=warm, verbose=_VERBOSE)
            print(f"    free-AD SI {dsi_r_ad:+.3f}  ({t_r_ad:.0f}s total, "
                  f"{ms_r_ad:.0f} ms/iter)", flush=True)

        # Speedup from the COMPILE-EXCLUDED per-iter cost (speed mode only).
        speedup = (ms_r_fd / ms_r_ad) if (ms_r_fd and ms_r_ad and ms_r_ad > 0) else None
        si_parity = (dsi_r_ad - dsi_r) if (dsi_r_ad is not None and dsi_r is not None) else None

        # Record dense results first, so a failed sparse-transfer leg does NOT
        # discard the dense comparison.
        rec.update(
            ok=True, n_fibers=N, K=int(dense["Ve_unit"].shape[0]),
            n_per_fascicle=_N_PER_FASC, pattern=patt, mode=_MODE,
            init_mode=_INIT_MODE, loss_mode=loss_mode, balance=bool(_BALANCE),
            dense_steps=_DENSE_STEPS,
            soft_temperature=float(os.environ["JAXLEY_FIBERS_SOFT_TEMPERATURE"]),
            energy_lambda=float(os.environ["JAXLEY_FIBERS_ENERGY_LAMBDA"]),
            dense_si_frozen=dsi_f, dense_si_relaxed=dsi_r,
            dense_si_relaxed_autodiff=dsi_r_ad,
            transfer_si_frozen=None, transfer_si_relaxed=None,
            penalty_frozen=None, penalty_relaxed=None,
            n_active_frozen=na_f, n_active_relaxed=na_r,
            n_active_relaxed_autodiff=na_r_ad,
            t_dense_frozen_fd=t_f, t_dense_relaxed_fd=t_r_fd,
            t_dense_relaxed_autodiff=t_r_ad,
            ms_iter_frozen_fd=ms_f, ms_iter_relaxed_fd=ms_r_fd,
            ms_iter_relaxed_autodiff=ms_r_ad,
            speedup_fd_over_autodiff=speedup,
            si_parity_autodiff_minus_fd=si_parity,
        )
        # Checkpoint the (expensive) dense results NOW, before the transfer leg,
        # so a walltime kill during transfer does not discard them.  ok=True here
        # makes the nerve terminal; transfer/penalty fields are filled below.
        out_path.write_text(json.dumps(rec, indent=2))

        # Sparse-optimized transfer (FD, both variants) -> deployment penalty.
        # Speed mode only; the headline (free-AD) run reuses the existing FD
        # sparsity/transfer results, so it does no transfer leg.
        if _MODE != "speed":
            rec["transfer_skip"] = "headline mode (free-AD only)"
        else:
            print("  dense arms saved (transfer leg next) ...", flush=True)
            sp = _sparse_seed(duke, dense)
            if sp is None:
                rec["transfer_skip"] = "sparse subsample had no target fibers"
            else:
                amps_sf, *_ = _optimize(sp, relaxed=False, grad_mode="fd")
                amps_sr, *_ = _optimize(sp, relaxed=True, grad_mode="fd")
                tsi_f, _ = S._eval_amps_on_dense(amps_sf, dense, S.DT)
                tsi_r, _ = S._eval_amps_on_dense(amps_sr, dense, S.DT)
                tsi_f, tsi_r = float(tsi_f), float(tsi_r)
                rec.update(
                    transfer_si_frozen=tsi_f, transfer_si_relaxed=tsi_r,
                    penalty_frozen=dsi_f - tsi_f, penalty_relaxed=dsi_r - tsi_r,
                )

        out_path.write_text(json.dumps(rec, indent=2))
        if _MODE == "headline":
            print(f"  [headline] free-AD dense SI {dsi_r_ad:+.3f}  "
                  f"({ms_r_ad:.0f} ms/iter, n_active {na_r_ad})", flush=True)
        else:
            tline = (f"penalty F {rec['penalty_frozen']:+.3f} | R {rec['penalty_relaxed']:+.3f}"
                     if rec.get("penalty_frozen") is not None else "transfer skipped")
            sline = (f"per-iter FD {ms_r_fd:.0f}ms / AD {ms_r_ad:.0f}ms -> "
                     f"{speedup:.2f}x (SI {dsi_r:+.3f} vs {dsi_r_ad:+.3f})"
                     if speedup is not None else "autodiff arm n/a")
            print(f"  dense SI  frozen {dsi_f:+.3f} | free {dsi_r:+.3f}   {tline}   "
                  f"n_active {na_f}->{na_r}   {sline}", flush=True)
        return 0
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out_path.write_text(json.dumps(rec, indent=2))
        print(f"  ERROR: {rec['error']}", flush=True)
        return 1
    finally:
        # Release the claim.  out_path (ok/skip/error) is the durable record;
        # the lock is only mutual exclusion between concurrent workers.
        try:
            _lock.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
