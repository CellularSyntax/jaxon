"""Duke FEM-nerve selectivity sweep.

Mirrors ``experiments_v2/selectivity_sweep.py`` but loads geometry +
extracellular potential from a Duke FEM bundle
(``duke_Ves/<sub>_<sam>/``) instead of generating a synthetic
Hussain-style nerve and analytic point-source Ve.

Differences from the synthetic sweep:

* **Geometry / Ve**: one Duke sample provides one nerve geometry
  (~1000 fibres across ~27 polygonal fascicles).  The cuff is the
  12-contact MultiContact array (3 axial rows × 4 azimuthal angles)
  pre-baked into the lead-field tensor ``Ve_VperA``.  The optimiser
  therefore searches over **12** amplitudes instead of 6.

* **Seeds**: each seed re-draws a random divider angle through (0, 0)
  and classifies the 27 fascicles into target / off-target by
  centroid side.  The Duke fascicle positions / shapes are fixed
  across seeds; only the target / off-target *labelling* changes.

* **Optimiser hyperparameters**: the FEM Ve magnitude is ~5× larger
  than the synthetic point-source Ve at the same nominal cuff radius
  (peak ~870 mV/mA vs ~175 mV/mA), so MRG activation thresholds are
  correspondingly ~5× smaller.  The default ``AMP_INIT_MA`` etc.
  reflect this.  Both can be overridden via env vars to retune.

Outputs land in ``outputs/selectivity_sweep_duke_<sub>_<sam>/``
with the same ``data_seed_NNNN.json`` schema as the synthetic sweep,
so the make_figures.py pipeline can render Duke figures with minimal
changes.

Run (smoketest, 1 seed):
    DUKE_SAMPLE_DIR=duke_Ves/sub-10_sam-1 \
    SEED_START=0 SEED_END=1 RECT_OPTIMIZER=adam_fd \
    python -m experiments_v2.selectivity_sweep_duke
"""
from __future__ import annotations
import json
import os
import pathlib
import sys
import time
from pathlib import Path

# Make sibling jaxfibers/ importable when this script is launched directly
# (e.g. on the cluster where PYTHONPATH may not include the project root).
# Mirrors the synthetic-sweep selectivity_sweep.py.
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jaxfibers.stim.batch_solve import (
    stack_fiber_statics, initial_states_batch,
)
from jaxfibers.optim.optimizer import (
    run_rect_optimization,
    run_rect_optimization_lbfgs,
    run_waveform_optimization,
)
from jaxfibers.optim.losses import activation_proxy_batch, selectivity_index
from experiments_v2.utils import ensure_dir, save_json
from experiments_v2.duke_loader import (
    load_duke_sample, divider_split_target_mask,
)

# ─────────────────────────────────────────────── sample location ──────────────
DUKE_SAMPLE_DIR = os.environ.get("DUKE_SAMPLE_DIR", "").strip()
if not DUKE_SAMPLE_DIR:
    raise RuntimeError(
        "selectivity_sweep_duke: set DUKE_SAMPLE_DIR (e.g. "
        "DUKE_SAMPLE_DIR=duke_Ves/sub-10_sam-1)"
    )
SAMPLE_PATH = Path(DUKE_SAMPLE_DIR)
if not SAMPLE_PATH.is_absolute():
    SAMPLE_PATH = ROOT / SAMPLE_PATH
SAMPLE_NAME = SAMPLE_PATH.name   # e.g. "sub-10_sam-1"

# Output dir scoped by sample so multiple samples land side-by-side
# under a single ``outputs/duke_sweeps/`` parent.  The sample directory
# itself keeps the bundle's name verbatim (e.g. ``sub-10_sam-1`` or
# ``human-sub-3_sam-1``) — the species can be inferred from the prefix
# downstream by the figure code.
_OUT_OVERRIDE = os.environ.get("JAXLEY_FIBERS_OUTPUT_DIR", "").strip()
if _OUT_OVERRIDE:
    _OUT_PATH = pathlib.Path(_OUT_OVERRIDE)
    if not _OUT_PATH.is_absolute():
        _OUT_PATH = ROOT / _OUT_PATH
    OUT = ensure_dir(_OUT_PATH)
else:
    OUT = ensure_dir(ROOT / "outputs" / "duke_sweeps" / SAMPLE_NAME)

# ────────────────────────────────────────── sweep parameters (env-overrideable)
def _env_int(name, default): return int(os.environ.get(name, default))
def _env_flt(name, default): return float(os.environ.get(name, default))

FIBER_DIAMETER_UM = _env_flt("FIBER_DIAMETER_UM", 5.7)
N_NODES         = _env_int("N_NODES", 21)
MAX_FIBERS      = _env_int("MAX_FIBERS", 0)  # 0 = no subsample (full ~1000 fibres)
SUBSAMPLE_SEED  = _env_int("SUBSAMPLE_SEED", 0)
DT              = _env_flt("DT", 0.005)
T_STOP          = _env_flt("T_STOP", 3.0)
DELAY_MS        = 1.0
PW_MS           = 0.1
SEED_START      = _env_int("SEED_START", 0)
SEED_END        = _env_int("SEED_END", 25)
N_OPT_RECT      = _env_int("N_OPT_RECT", 30)
N_RESTARTS_RECT = _env_int("N_RESTARTS_RECT", 2)
N_OPT_WAVE      = _env_int("N_OPT_WAVE", 100)
WAVE_LR         = _env_flt("WAVE_LR", 5e-4)
WAVE_PATIENCE   = _env_int("WAVE_PATIENCE", 20)
RECT_OPTIMIZER  = os.environ.get("RECT_OPTIMIZER", "adam_fd")  # adam_fd | lbfgs

# Per-Duke amplitude knobs.  Defaults below are tuned for MRG 5.7 µm
# against the FEM Ve scale of the sub-10_sam-1 bundle (peak Ve ≈ 870
# mV/mA, ~5× the synthetic point-source).  Iter-0 of the smoketest at
# ±0.30 mA already drove 72 % of off-target fibres → super-threshold;
# we back off to ±0.10 mA init / ±0.20 mA clip / smaller lr so the
# optimiser stays in the gradient-informative regime instead of
# landing at the all-fire attractor.  All env-overrideable.
AMP_INIT_MA = _env_flt("AMP_INIT_MA", -0.30)
_AMP_CLIP_LO = _env_flt("AMP_CLIP_LO", -1.50)
_AMP_CLIP_HI = _env_flt("AMP_CLIP_HI",  1.50)
AMP_CLIP    = (_AMP_CLIP_LO, _AMP_CLIP_HI)
ADAM_LR_MA  = _env_flt("ADAM_LR_MA",  0.015)
FD_EPS_MA   = _env_flt("FD_EPS_MA",   0.010)


def _build_seed(duke: dict, seed: int, verbose: bool = True) -> dict:
    """Per-seed pipeline: redraw divider angle, set target_mask, build
    solver statics, baseline SI.  All compute-light steps — the heavy
    Ve_unit + geometry come from ``duke`` and are seed-independent."""
    label = f"[{SAMPLE_NAME} | seed {seed:4d}]"
    # Random divider angle, reproducible per seed (uniform on [0, 180)°).
    divider_deg = float(
        np.random.default_rng(seed + 1_000_003).uniform(0.0, 180.0)
    )
    nerve = duke["nerve_geom"]
    target_mask = divider_split_target_mask(
        nerve, duke["fasc_id"], duke["fasc_meta"], divider_deg
    )
    n_tgt = int(target_mask.sum())
    n_tgt_fasc = sum(1 for f in nerve.fascicles if f.is_target)
    if verbose:
        print(f"{label} divider={divider_deg:5.1f}°  "
              f"targets={n_tgt}/{nerve.n_fibers} "
              f"({n_tgt_fasc}/{len(nerve.fascicles)} fascicles)", flush=True)

    geoms = duke["geoms"]
    fs_batch = stack_fiber_statics(geoms, DT)
    s0_batch = initial_states_batch(geoms)

    N_STEPS = int(T_STOP / DT)
    t_grid  = (np.arange(N_STEPS) + 1) * DT
    pulse_mask = np.where(
        (t_grid >= DELAY_MS) & (t_grid < DELAY_MS + PW_MS), 1.0, 0.0,
    ).astype(np.float64)

    # Baseline (zero stimulation)
    m_max_zero = activation_proxy_batch(
        jnp.zeros((nerve.n_fibers, geoms[0].n_comp), dtype=jnp.float64),
        jnp.asarray(duke["node_indices"], dtype=jnp.int32),
    )
    si_baseline = selectivity_index(np.array(m_max_zero), target_mask)

    return dict(
        seed=seed, label=label, divider_deg=divider_deg,
        nerve=nerve, geoms=geoms,
        target_mask=target_mask,
        Ve_unit=duke["Ve_unit"], node_indices=duke["node_indices"],
        fs_batch=fs_batch, s0_batch=s0_batch,
        pulse_mask=pulse_mask, N_STEPS=N_STEPS,
        si_baseline=si_baseline,
    )


def _run_one_seed(seed_in: dict, verbose: bool = True) -> dict:
    """Run rect + wave optimisation for a single Duke seed.  Returns the
    JSON-serialisable per-seed dict.  Mirrors the synthetic sweep's
    package_result so the figure code can read both."""
    label = seed_in["label"]
    K = seed_in["Ve_unit"].shape[0]
    if verbose:
        print(f"{label} K={K} contacts  init={AMP_INIT_MA:+.3f} mA  "
              f"clip=[{_AMP_CLIP_LO:+.2f}, {_AMP_CLIP_HI:+.2f}] mA  "
              f"lr={ADAM_LR_MA:.4f}  fd_eps={FD_EPS_MA:.4f}  "
              f"opt={RECT_OPTIMIZER}", flush=True)

    rect_t0 = time.time()
    if RECT_OPTIMIZER == "adam_fd":
        n_iters = max(N_OPT_RECT * 3, 30)
        if verbose:
            print(f"{label} Rect Adam-FD ({n_iters} iters) ...", flush=True)
        adam_res = run_rect_optimization(
            fiber_statics_batch=seed_in["fs_batch"],
            state0_batch=seed_in["s0_batch"],
            Ve_unit=seed_in["Ve_unit"],
            pulse_mask=seed_in["pulse_mask"],
            node_indices=seed_in["node_indices"],
            target_mask=seed_in["target_mask"],
            dt=DT, n_steps=n_iters,
            amp_init_mA=AMP_INIT_MA, amp_clip=AMP_CLIP,
            lr=ADAM_LR_MA, fd_eps=FD_EPS_MA,
            verbose=verbose,
        )
        loss_hist = np.asarray(adam_res["history"]["loss"])
        # Adam-FD can overshoot a good warm-start (mid-trajectory loss
        # 0.17, last-iter loss 0.55 was observed on the Duke smoketest).
        # Return the BEST-loss iter's amps + acts, not the last iter.
        best_iter = int(np.argmin(loss_hist))
        best_amps = np.asarray(adam_res["history"]["amps"][best_iter])
        best_acts = np.asarray(adam_res["history"]["acts"][best_iter])
        rect_res = {
            "optimizer":     "Adam-FD",
            "amps":          best_amps,
            "loss_history":  loss_hist,
            "final_loss":    float(loss_hist[best_iter]),
            "final_acts":    best_acts,
            "best_iter":     best_iter,
            "best_restart":  0,
            "n_restarts":    1,
            "n_steps":       n_iters,
            "all_final_losses": np.asarray([loss_hist[-1]]),
        }
    else:  # LBFGS
        if verbose:
            print(f"{label} Rect LBFGS ({N_RESTARTS_RECT} restarts × "
                  f"{N_OPT_RECT} steps) ...", flush=True)
        lbfgs_res = run_rect_optimization_lbfgs(
            fiber_statics_batch=seed_in["fs_batch"],
            state0_batch=seed_in["s0_batch"],
            Ve_unit=seed_in["Ve_unit"],
            pulse_mask=seed_in["pulse_mask"],
            node_indices=seed_in["node_indices"],
            target_mask=seed_in["target_mask"],
            dt=DT, n_restarts=N_RESTARTS_RECT, n_steps=N_OPT_RECT,
            amp_init_mA=AMP_INIT_MA, amp_clip=AMP_CLIP,
            rng_seed=seed_in["seed"], verbose=verbose,
        )
        rect_res = {
            "optimizer":     "LBFGS-multistart",
            "amps":          np.asarray(lbfgs_res["amps"]),
            "loss_history":  np.asarray(lbfgs_res["all_loss_traces"][
                                 lbfgs_res["best_restart"]]),
            "final_loss":    float(lbfgs_res["final_loss"]),
            "final_acts":    np.asarray(lbfgs_res["final_acts"]),
            "best_restart":  int(lbfgs_res["best_restart"]),
            "n_restarts":    N_RESTARTS_RECT,
            "n_steps":       N_OPT_RECT,
            "all_final_losses": np.asarray(lbfgs_res["all_final_losses"]),
        }
    rect_t = time.time() - rect_t0
    si_rect = selectivity_index(rect_res["final_acts"], seed_in["target_mask"])
    print(f"{label} Rect done: SI {seed_in['si_baseline']:+.3f} → "
          f"{si_rect:+.3f}  ({rect_t:.0f}s)", flush=True)

    # ── Waveform ──────────────────────────────────────────────────────────
    if N_OPT_WAVE <= 0:
        wave_res = {
            "best_si":   float(si_rect),
            "best_iter": 0,
            "best_acts": np.asarray(rect_res["final_acts"]),
            "history":   {"loss": [], "si": [], "acts": [np.asarray(rect_res["final_acts"])]},
        }
        wave_t = 0.0
    else:
        u_init = np.zeros((K, seed_in["N_STEPS"]), dtype=np.float64)
        for k in range(K):
            u_init[k] = float(rect_res["amps"][k]) * seed_in["pulse_mask"]
        if verbose:
            print(f"{label} Waveform Adam ({N_OPT_WAVE} iters) ...", flush=True)
        t0 = time.time()
        wave_res = run_waveform_optimization(
            fiber_statics_batch=seed_in["fs_batch"],
            state0_batch=seed_in["s0_batch"],
            Ve_unit=jnp.asarray(seed_in["Ve_unit"], dtype=jnp.float64),
            node_indices=seed_in["node_indices"],
            target_mask=seed_in["target_mask"],
            dt=DT, T=seed_in["N_STEPS"], n_steps=N_OPT_WAVE, u_init=u_init,
            lr=WAVE_LR, early_stop_patience=WAVE_PATIENCE, u_clip=AMP_CLIP,
            verbose=verbose,
        )
        wave_t = time.time() - t0
        si_wave = float(wave_res["best_si"])
        print(f"{label} Wave done: SI {si_rect:+.3f} → {si_wave:+.3f} "
              f"(best @ iter {wave_res['best_iter']})  ({wave_t:.0f}s)",
              flush=True)

    return _package_result(seed_in, rect_res, rect_t, wave_res, wave_t)


def _package_result(seed_in: dict, rect_res: dict, rect_t: float,
                    wave_res: dict, wave_t: float) -> dict:
    target_mask = seed_in["target_mask"]
    si_rect_final = selectivity_index(rect_res["final_acts"], target_mask)
    nerve = seed_in["nerve"]
    return {
        "sample":   SAMPLE_NAME,
        "seed":     seed_in["seed"],
        "divider_deg": seed_in["divider_deg"],
        "si_baseline": float(seed_in["si_baseline"]),
        "rect": {
            "optimizer":   rect_res["optimizer"],
            "n_restarts":  rect_res["n_restarts"],
            "n_steps":     rect_res["n_steps"],
            "best_restart": rect_res["best_restart"],
            "final_loss":  rect_res["final_loss"],
            "final_si":    float(si_rect_final),
            "loss_history": rect_res["loss_history"].tolist(),
            "all_final_losses": rect_res["all_final_losses"].tolist(),
            "amps_mA":     rect_res["amps"].tolist(),
            "final_acts":  rect_res["final_acts"].tolist(),
            "time_s":      rect_t,
        },
        "waveform": {
            "optimizer":  "Adam",
            "n_steps":    N_OPT_WAVE,
            "lr":         WAVE_LR,
            "patience":   WAVE_PATIENCE,
            "final_si":   float(wave_res["best_si"]),
            "best_iter":  int(wave_res["best_iter"]),
            "last_si":    (float(wave_res["history"]["si"][-1])
                           if wave_res["history"]["si"] else float(wave_res["best_si"])),
            "n_iters_run": len(wave_res["history"]["loss"]),
            "loss_history": wave_res["history"]["loss"],
            "si_history":  wave_res["history"]["si"],
            "final_acts":  np.asarray(wave_res["best_acts"]).tolist(),
            "time_s":      wave_t,
        },
        "nerve": {
            "fiber_diam":  nerve.fiber_diam.tolist(),
            "target_mask": target_mask.tolist(),
            "fiber_x_um":  nerve.fiber_x_um.tolist(),
            "fiber_y_um":  nerve.fiber_y_um.tolist(),
        },
    }


def main():
    t_load = time.time()
    duke = load_duke_sample(
        SAMPLE_PATH, fiber_diam_um=FIBER_DIAMETER_UM, n_nodes=N_NODES,
        max_fibers=(MAX_FIBERS if MAX_FIBERS > 0 else None),
        subsample_seed=SUBSAMPLE_SEED,
        verbose=True,
    )
    print(f"[duke sweep] Loaded {SAMPLE_NAME} in {time.time() - t_load:.1f}s. "
          f"Output dir: {OUT}", flush=True)

    seeds = list(range(SEED_START, SEED_END))
    for s in seeds:
        out_path = OUT / f"data_seed_{s:04d}.json"
        if out_path.exists():
            print(f"[seed {s}] already exists at {out_path}; skipping",
                  flush=True)
            continue
        seed_in = _build_seed(duke, s, verbose=True)
        result = _run_one_seed(seed_in, verbose=True)
        save_json(result, out_path)
        print(f"[seed {s}] -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
