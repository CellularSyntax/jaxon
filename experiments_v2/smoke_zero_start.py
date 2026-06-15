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
ZERO_LR_DECAY, ZERO_SI_FLOOR, ZERO_PATIENCE, MAX_FIBERS.
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
from jaxfibers.optim.optimizer import run_rect_optimization

_LR       = float(os.environ.get("ZERO_LR_MA", "0.3"))
_STEPS    = int(os.environ.get("ZERO_STEPS", "300"))
_DECAY    = float(os.environ.get("ZERO_LR_DECAY", "0.6"))
_FLOOR    = float(os.environ.get("ZERO_SI_FLOOR", "0.5"))
_PATIENCE = int(os.environ.get("ZERO_PATIENCE", "15"))


def main() -> int:
    name = S.SAMPLE_NAME
    loss_mode = os.environ["JAXLEY_FIBERS_LOSS"]
    print(f"[zero-smoke] {name}  loss={loss_mode}  "
          f"lr={_LR} steps={_STEPS} decay={_DECAY} floor={_FLOOR} "
          f"patience={_PATIENCE}", flush=True)

    duke = S.load_duke_sample(
        S.SAMPLE_PATH, fiber_diam_um=S.FIBER_DIAMETER_UM, n_nodes=S.N_NODES,
        max_fibers=(S.MAX_FIBERS if S.MAX_FIBERS > 0 else None),
        subsample_seed=S.SUBSAMPLE_SEED, verbose=False,
    )
    seed = S._build_seed(duke, seed=0, verbose=False)
    if seed is None:
        print("  SKIP: no valid target at seed 0 "
              "(lower CLUSTER_N_MIN_TARGET or raise MAX_FIBERS)", flush=True)
        return 0

    tgt = np.asarray(seed["target_mask"], bool)
    K = int(seed["Ve_unit"].shape[0])
    print(f"  N={tgt.size}  targets={int(tgt.sum())}  K={K} contacts  "
          f"init = ALL ZERO (0 mA), no probe, no freeze", flush=True)

    res = run_rect_optimization(
        fiber_statics_batch=seed["fs_batch"], state0_batch=seed["s0_batch"],
        Ve_unit=seed["Ve_unit"], pulse_mask=seed["pulse_mask"],
        node_indices=seed["node_indices"], target_mask=seed["target_mask"],
        weights=seed["weights"], dt=S.DT, n_steps=_STEPS,
        amps_init_vector=None, amp_init_mA=0.0, amp_clip=S.AMP_CLIP,
        lr=_LR, fd_eps=S.FD_EPS_SMART_MA,
        lr_mode="plateau", plateau_lr_decay=_DECAY, plateau_si_floor=_FLOOR,
        plateau_patience=_PATIENCE, freeze_zero_mask=None,
        # never early-stop: we want the whole trajectory.
        early_stop_si=2.0, early_stop_patience=10**9, early_stop_si_patience=0,
        verbose=True,
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
