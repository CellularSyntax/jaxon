"""Standalone cold-start (0 mA) GD smoke test -- a debug harness, not an experiment.

ONE nerve, downsampled, EVERY contact initialised to 0 mA, optimised with
everything we implemented (quotient loss + ReduceLROnPlateau LR schedule +
hard-best selection).  Prints the FULL per-iteration SI trajectory so we can
see whether gradient descent ever escapes SI=0 from a cold start -- the thing
we are debugging.

No frozen/relaxed, no sparse transfer, no aggregation.  Just: build one seed,
run the cold GD with verbose on, report whether and when SI lifts off zero.

Run (driven by slurm/run_smoke_zero_start.sbatch), or directly in-container:

    DUKE_SAMPLE_DIR=duke_Ves/human_sub-47_sam-2 MAX_FIBERS=50 \
    JAXLEY_FIBERS_LOSS=quotient CLUSTER_N_MIN_TARGET=8 \
    python -u -m experiments_v2.smoke_zero_start

Knobs (env): JAXLEY_FIBERS_LOSS (quotient|linear), ZERO_LR_MA, ZERO_STEPS,
ZERO_LR_DECAY, ZERO_SI_FLOOR, ZERO_PATIENCE, MAX_FIBERS, ZERO_GRAD_MODE
(fd|autodiff), ZERO_MAX_TGT_FRAC (reject near-whole-nerve degenerate targets).
"""
from __future__ import annotations

import os
import sys

# Loss mode is read at import time by jaxfibers.optim.optimizer -- set first.
os.environ.setdefault("JAXLEY_FIBERS_LOSS", "quotient")
os.environ.setdefault("JAXLEY_FIBERS_SOFT_TEMPERATURE", "0.15")
os.environ.setdefault("N_OPT_RECT", "1")  # unused here, keeps S import happy

import numpy as np
import jax  # noqa: F401  (ensures device init / X64 banner)

from experiments_v2 import selectivity_sweep_duke as S
from jaxfibers.optim.optimizer import (
    run_rect_optimization, run_rect_optimization_autodiff,
)

_LR       = float(os.environ.get("ZERO_LR_MA", "0.05"))   # 0.3 overshoots to fire-everything
_STEPS    = int(os.environ.get("ZERO_STEPS", "300"))
_DECAY    = float(os.environ.get("ZERO_LR_DECAY", "0.6"))
_FLOOR    = float(os.environ.get("ZERO_SI_FLOOR", "0.5"))
_PATIENCE = int(os.environ.get("ZERO_PATIENCE", "15"))
_WD       = float(os.environ.get("ZERO_WD", "0.02"))      # decoupled weight decay (AxonML-style)
# Symmetry-breaking jitter on the (otherwise exactly-zero) init: a tiny random
# per-contact amplitude so Adam can leave the symmetric all-zero saddle even
# when the gradient is near-symmetric.  0 = exact zero.
_JITTER   = float(os.environ.get("ZERO_JITTER", "0.01"))
# Gradient source for our GD.  BOTH are gradient descent: "fd" = the
# finite-difference-gradient flavour the production sweep uses
# (run_rect_optimization); "autodiff" = exact reverse-mode gradient
# (run_rect_optimization_autodiff).  "finite-diff" in the banner is HOW the
# gradient is estimated, not a different optimiser.
_GRAD_MODE = os.environ.get("ZERO_GRAD_MODE", "fd").strip().lower()
# Reject degenerate targets.  A 90 deg peripheral window can land on a dominant
# fascicle and grab ~all fibres (target fraction -> 1.0); with almost no
# off-target to spare, "fire everything" is optimal and SI=0 is the CORRECT
# answer, so such a target says nothing about cold-start selectivity.  Only
# accept minority targets.
_MAX_TGT_FRAC = float(os.environ.get("ZERO_MAX_TGT_FRAC", "0.5"))
# Charge-balance (Kirchhoff) constraint: project the per-contact currents onto
# sum=0 each step so the cuff has sources AND sinks.  Without it the optimiser
# escapes into an all-same-sign monopolar field that fires the whole nerve.  ON
# by default here -- this is the hypothesis under test.  Set ZERO_BALANCE=0 off.
_BALANCE = os.environ.get("ZERO_BALANCE", "1").strip() not in ("0", "false", "")


def main() -> int:
    name = S.SAMPLE_NAME
    loss_mode = os.environ["JAXLEY_FIBERS_LOSS"]
    print(f"[zero-smoke] {name}  loss={loss_mode}  "
          f"lr={_LR} wd={_WD} steps={_STEPS} decay={_DECAY} floor={_FLOOR} "
          f"patience={_PATIENCE}", flush=True)

    duke = S.load_duke_sample(
        S.SAMPLE_PATH, fiber_diam_um=S.FIBER_DIAMETER_UM, n_nodes=S.N_NODES,
        max_fibers=(S.MAX_FIBERS if S.MAX_FIBERS > 0 else None),
        subsample_seed=S.SUBSAMPLE_SEED, verbose=False,
    )
    # Survey every cluster position and pick a REPRESENTATIVE (minority) target.
    # seed s -> cluster position s % CLUSTER_N_POSITIONS.  A 90 deg window can
    # land on a dominant fascicle and grab ~all fibres; that target is
    # degenerate (no off-target to spare) and SI=0 is correct there, so reject
    # it -- see _MAX_TGT_FRAC.
    cands = []
    for s in range(max(S.CLUSTER_N_POSITIONS, 1)):
        sd = S._build_seed(duke, seed=s, verbose=False)
        if sd is None:
            continue
        tm = np.asarray(sd["target_mask"], bool)
        cands.append((s, sd, tm.sum() / max(tm.size, 1)))
    if not cands:
        print("  SKIP: no valid cluster target at any position "
              "(lower CLUSTER_N_MIN_TARGET or raise MAX_FIBERS)", flush=True)
        return 0

    print("  cluster positions:  "
          + "  ".join(f"pos{s}={100*f:.0f}%" for s, _, f in cands)
          + f"   (target fraction; reject > {100*_MAX_TGT_FRAC:.0f}%)", flush=True)

    valid = [c for c in cands if c[2] <= _MAX_TGT_FRAC]
    if valid:
        # Among the non-degenerate positions, the largest minority target is
        # the most informative selectivity problem (strong on-signal, real
        # off-target to spare).
        pos, seed, frac = max(valid, key=lambda c: c[2])
    else:
        pos, seed, frac = min(cands, key=lambda c: c[2])
        print(f"  *** WARNING: EVERY position is degenerate (smallest target "
              f"fraction {100*frac:.0f}% > {100*_MAX_TGT_FRAC:.0f}%).  At "
              f"MAX_FIBERS={S.MAX_FIBERS} this nerve's 90 deg window lands on a "
              f"dominant fascicle and grabs ~all fibres -- there is almost no "
              f"off-target to spare, so SI=0 is the CORRECT answer and this run "
              f"says NOTHING about cold-start.  Pick another SMOKE_SAMPLE or "
              f"shrink CLUSTER_WINDOW_DEG / raise CLUSTER_RADIUS_QUANTILE.",
              flush=True)

    tgt = np.asarray(seed["target_mask"], bool)
    K = int(seed["Ve_unit"].shape[0])
    init_desc = "ALL ZERO (0 mA)" if _JITTER <= 0 else f"~0 mA + N(0,{_JITTER}) jitter"
    print(f"  USING pos {pos}:  N={tgt.size}  targets={int(tgt.sum())} "
          f"({100*frac:.0f}%)  K={K} contacts  grad={_GRAD_MODE}  "
          f"balance={'sum0' if _BALANCE else 'off'}  "
          f"init = {init_desc}, no probe, no freeze", flush=True)

    amps_init = (None if _JITTER <= 0 else
                 np.asarray(np.random.default_rng(0).normal(0.0, _JITTER, K)))

    call = dict(
        fiber_statics_batch=seed["fs_batch"], state0_batch=seed["s0_batch"],
        Ve_unit=seed["Ve_unit"], pulse_mask=seed["pulse_mask"],
        node_indices=seed["node_indices"], target_mask=seed["target_mask"],
        weights=seed["weights"], dt=S.DT, n_steps=_STEPS,
        amps_init_vector=amps_init, amp_init_mA=0.0, amp_clip=S.AMP_CLIP,
        lr=_LR, lr_mode="plateau", plateau_lr_decay=_DECAY,
        plateau_si_floor=_FLOOR, plateau_patience=_PATIENCE, weight_decay=_WD,
        balance_currents=_BALANCE,
        # never early-stop: we want the whole trajectory.
        early_stop_si=2.0, early_stop_patience=10**9, early_stop_si_patience=0,
        verbose=True,
    )
    if _GRAD_MODE == "autodiff":
        res = run_rect_optimization_autodiff(**call)
    else:
        res = run_rect_optimization(
            **call, fd_eps=S.FD_EPS_SMART_MA, freeze_zero_mask=None,
        )

    si   = np.asarray(res["history"]["si"])
    loss = np.asarray(res["history"]["loss"])
    best_i  = int(np.lexsort((loss, -si))[0])       # hard-best: max SI, tiebreak loss
    best_si = float(si[best_i])
    first   = next((i for i, v in enumerate(si) if v > 1e-6), None)

    print("\n  === per-iter SI trajectory ===", flush=True)
    print("  " + "  ".join(f"{v:+.2f}" for v in si), flush=True)
    print(f"\n  best SI = {best_si:+.3f} @ iter {best_i} "
          f"(final loss {float(loss[best_i]):.4f})", flush=True)
    if first is None:
        print("  >>> NEVER escaped SI=0   (cold start FAILED)", flush=True)
    else:
        print(f"  >>> escaped SI=0 at iter {first} "
              f"(SI there = {float(si[first]):+.3f})   (cold start WORKS)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
