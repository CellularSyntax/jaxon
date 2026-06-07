"""Spatial selectivity optimisation — cluster sweep (paper-quality).

Runs N_NERVES synthetic nerve realisations, each with both rect and waveform
optimisation.  Saves one JSON per seed and produces aggregate figures.

Setup (per seed)
----------------
- N_FIBERS MRG fibers, **single diameter D = FIBER_DIAMETER_UM** (Hussain 2024
  methodology: a representative 5.7 µm fiber per fascicle; we go further and
  vmap hundreds of fibers within the synthetic nerve to showcase the JAX
  parallel speed advantage).
- 6-contact ring cuff, 1.5 mm radius
- Target: eccentric fascicle ~30% area
- 3 ms simulation window (PW=0.1 + DELAY=1.0 + AP propagation ~2 ms)
- LBFGS with multi-restart for rect (default M=4)
- Adam for waveform optimisation (warm-started from rect amps)

Multi-seed batching
-------------------
Set SEEDS_PER_TASK > 1 in the environment to vmap the rect optimisation over
S seeds simultaneously.  The rect step batches; the waveform step still runs
per-seed (its autodiff backward through the T-step scan has large per-fiber
memory, so batching it requires care — left as future work).

Outputs
-------
outputs/selectivity_sweep/
  data_seed_<N>.json          — per-seed results
  fig_sweep_si_violin.png     — SI distributions: init vs rect vs waveform
  fig_sweep_loss_curves.png   — mean ± 1σ loss curves for both modes
  fig_sweep_activation_maps.png — example seed activation maps

Environment variables (set by the sbatch wrapper):
  SEED_START / SEED_END        Half-open range of seeds for this task.
  SEEDS_PER_TASK               How many seeds to vmap together for the
                                rect step.  Default 1 (single-seed path).

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

from jaxfibers.nerve.geometry import make_hussain_style_nerve
from jaxfibers.stim.multichannel_field import make_ring_cuff_positions, precompute_ve_unit
from jaxfibers.stim.batch_solve import stack_fiber_statics, initial_states_batch
from jaxfibers.optim.losses import activation_proxy_batch, selectivity_index
from jaxfibers.optim.optimizer import (
    run_rect_optimization,           # Adam-FD (pure forward, fast compile)
    run_rect_optimization_lbfgs,     # LBFGS autodiff (slow compile)
    run_rect_optimization_lbfgs_batched,
    run_waveform_optimization,
)

# Rect-step optimiser selection.  Default LBFGS is what the manuscript-scale
# job uses.  Smoketests / debug runs can set `OPTIMIZER=adam_fd` to swap in
# the legacy Adam-FD path — same loss, same forward, but FD gradient instead
# of autodiff backward, so no remat planner pathology and no autodiff
# memory blowup (compile drops from 25 min to ~2 min on A16).
RECT_OPTIMIZER = os.environ.get("OPTIMIZER", "lbfgs").lower()

# Fibre model dispatch.  Default 'mrg' uses the original (myelinated) pipeline;
# 'sundt' / 'sweeney' / 'rattay' route through the generic batched solver in
# jaxfibers.stim.batch_solve_generic.  When non-MRG, the optimiser's
# batch_integrate_m_max{,_fd} are monkey-patched at startup (per-seed) so the
# 7 call sites inside the optimiser don't need to know which model is running.
FIBER_MODEL = os.environ.get("FIBER_MODEL", "mrg").lower()

# Multichannel field's section_centers_um shim auto-dispatches on the geometry
# class, so we don't need a model-specific import here any more.
from jaxfibers.stim.multichannel_field import section_centers_um
from experiments_v2.utils import ensure_dir, plot_seed_summary, plot_seed_cross_section

# Output dir can be redirected per-phase by the manuscript sbatch via
# JAXLEY_FIBERS_OUTPUT_DIR; default is outputs/selectivity_sweep/.
_OUT_OVERRIDE = os.environ.get("JAXLEY_FIBERS_OUTPUT_DIR", "").strip()
if _OUT_OVERRIDE:
    _OUT_PATH = pathlib.Path(_OUT_OVERRIDE)
    if not _OUT_PATH.is_absolute():
        _OUT_PATH = ROOT / _OUT_PATH
    OUT = ensure_dir(_OUT_PATH)
else:
    OUT = ensure_dir(ROOT / "outputs" / "selectivity_sweep")

# ─────────────────────────────────────────────────── sweep parameters ─────────
# Every parameter below is env-overridable so a smoketest sbatch can shrink
# N_FIBERS / N_OPT_* / T_STOP to slash compile + run time without touching
# the code.  See slurm/run_selectivity_sweep_smoketest.sbatch.
def _env_int(name, default):  return int(os.environ.get(name, default))
def _env_flt(name, default):  return float(os.environ.get(name, default))

N_NERVES        = _env_int("N_NERVES", 100)
N_FIBERS        = _env_int("N_FIBERS", 200)
                                # Hussain placed 1 fiber per fascicle; we vmap
                                # hundreds per nerve realisation to showcase
                                # the parallel speed advantage of jaxfibers.
FIBER_DIAMETER_UM = _env_flt("FIBER_DIAMETER_UM", 5.7)
                                # Hussain's representative MRG fiber.  Single
                                # diameter only — mixed diameters slow LBFGS
                                # convergence and aren't needed for the
                                # selectivity optimisation (Hussain assumed
                                # all fibres of a given diameter in a fascicle
                                # share the same threshold).
N_NODES         = _env_int("N_NODES", 21)
                                # n_comp = 20×11 + 1 = 221 (~23 mm fiber for D=10 µm)
N_CONTACTS      = 6
NERVE_RADIUS_UM = 500.0
CUFF_RADIUS_UM  = 1500.0
TARGET_FRACTION = 0.30
N_FASCICLES        = _env_int("N_FASCICLES", 8)
DIVIDER_ANGLE_DEG  = _env_flt("DIVIDER_ANGLE_DEG", 0.0)
                                # Hussain-style nerve: N_FASCICLES non-
                                # overlapping fascicles packed inside the
                                # nerve outline, then split into target /
                                # off-target by a straight line at
                                # DIVIDER_ANGLE_DEG through the centre.
                                # 0° = horizontal line, target = top half.
                                # Only used when RANDOMIZE_DIVIDER=0.
RANDOMIZE_DIVIDER  = _env_int("RANDOMIZE_DIVIDER", 1)
                                # When 1 (default), the divider line angle
                                # is drawn uniformly from [0, 180)° per seed
                                # so the on/off-target split rotates randomly
                                # across the sweep — covering every possible
                                # cardiac-vs-non-cardiac fascicle layout
                                # rather than baking in the 0° (top vs
                                # bottom) split.  Set RANDOMIZE_DIVIDER=0 to
                                # pin the divider at DIVIDER_ANGLE_DEG.

# Mixed-diameter mode: when both are set (non-zero), target fascicles use
# TARGET_DIAMETER_UM, off-target fascicles use OFFTARGET_DIAMETER_UM.
# This is the regime where waveform optimisation can beat rectangular —
# different chronaxie between the two populations.  Typical pair:
# 5.7 µm sensory target vs 14 µm motor off-target.  When either is 0,
# fall back to single-diameter FIBER_DIAMETER_UM for the whole nerve.
TARGET_DIAMETER_UM    = _env_flt("TARGET_DIAMETER_UM",    0.0)
OFFTARGET_DIAMETER_UM = _env_flt("OFFTARGET_DIAMETER_UM", 0.0)
DT              = _env_flt("DT", 0.005)        # ms
T_STOP          = _env_flt("T_STOP", 3.0)      # ms; PW + DELAY + slowest-MRG propagation ≈ 2.1 ms.
DELAY_MS        = 1.0
PW_MS           = 0.1
N_OPT_RECT      = _env_int("N_OPT_RECT", 30)
                                # LBFGS converges much faster than Adam:
                                # ~20-30 iters of LBFGS ≈ ~100 iters of Adam-FD,
                                # because each LBFGS step uses curvature info
                                # via the Strong-Wolfe zoom line search.
N_RESTARTS_RECT = int(os.environ.get("N_RESTARTS_RECT", 2))
# M parallel restarts (vmapped over restart axis on the GPU).  Restart 0 is
# the bipolar Ve-weighted deterministic init; 1..M-1 are random in
# (-clip/4, +clip/4).  Best of M wins.
#
# Memory note: LBFGS autodiff backward through the T=600 scan with M
# restarts vmapped consumes ~3 GB per restart on the cluster (200 fibers ×
# 221 comp, with @jax.checkpoint).  M=4 needs ~12 GB and OOMs on A16's
# ~12 GB usable VRAM.  M=2 (the default below) is the verification-run
# sweet spot: still gets the multi-restart escape benefit but fits on A16.
# For a100/h100 with more VRAM, override with `N_RESTARTS_RECT=4` (or
# higher) in the sbatch.
N_OPT_WAVE      = _env_int("N_OPT_WAVE", 100)
                                # Waveform still uses Adam; 100 iters is plenty.
WAVE_LR         = _env_flt("WAVE_LR", 5e-4)
                                # Adam LR for waveform optimisation.  Default
                                # 5e-4 is tuned for the warm-start case where
                                # u_init = rect_amps × pulse_mask already sits
                                # at SI≈1; lr=5e-3 (the legacy default) was
                                # too aggressive and destroyed the warm-start
                                # in <20 iters (Phase 3 seed 0: SI 0.97 → 0.00).
                                # For cold-start runs (u_init=0) bump back to
                                # 5e-3 in the sbatch.
WAVE_PATIENCE   = _env_int("WAVE_PATIENCE", 20)
                                # Early-stop if best_loss hasn't improved for
                                # this many iters.  0 disables.  20 iters is
                                # plenty for Adam to escape a small local min;
                                # any longer means it's wandering.
EXAMPLE_SEED    = 0            # seed for the example activation-map figure


def _build_seed_inputs(seed: int, verbose: bool = True) -> dict:
    """Build the per-seed nerve + statics + Ve_unit + pulse_mask used by both
    the rect and waveform optimisers."""
    label = f"[seed={seed}]"
    if verbose:
        print(f"{label} Building nerve ...", flush=True)
    # Hussain-style multi-fascicle nerve: pack several non-overlapping
    # circular fascicles inside the nerve outline, then split into
    # target / off-target by a straight line through the centre — matches
    # the Hussain 2024 anatomies (P1-P6 pig: motor-vs-sensory split;
    # H1-H6 human: random semicircle split).
    n_per_fasc = max(1, N_FIBERS // N_FASCICLES)
    mixed_diam = TARGET_DIAMETER_UM > 0 and OFFTARGET_DIAMETER_UM > 0
    # Per-seed divider angle.  When RANDOMIZE_DIVIDER=1 (default), draw
    # uniformly from [0, 180)° using the seed itself as RNG so the
    # randomisation is reproducible.  Range stops at 180° because the
    # divider line is symmetric (angle and angle+180° produce the same
    # cut, swapping which half is "target").
    if RANDOMIZE_DIVIDER:
        divider_deg = float(np.random.default_rng(seed + 1_000_003).uniform(0.0, 180.0))
    else:
        divider_deg = DIVIDER_ANGLE_DEG
    nerve = make_hussain_style_nerve(
        n_fascicles=N_FASCICLES,
        n_fibers_per_fascicle=n_per_fasc,
        nerve_radius_um=NERVE_RADIUS_UM,
        divider_angle_deg=divider_deg,
        diameters=[FIBER_DIAMETER_UM],
        target_diameter_um=TARGET_DIAMETER_UM if mixed_diam else None,
        offtarget_diameter_um=OFFTARGET_DIAMETER_UM if mixed_diam else None,
        seed=seed,
    )
    n_tgt = int(nerve.target_mask.sum())
    n_total = nerve.n_fibers
    if verbose:
        n_tgt_fasc = sum(1 for f in nerve.fascicles if f.is_target)
        n_off_fasc = len(nerve.fascicles) - n_tgt_fasc
        if mixed_diam:
            diam_str = (f"D_target={TARGET_DIAMETER_UM} µm / "
                        f"D_offtarget={OFFTARGET_DIAMETER_UM} µm")
        else:
            diam_str = f"D={FIBER_DIAMETER_UM} µm"
        print(f"{label} Nerve: {n_total} fibres in {len(nerve.fascicles)} "
              f"fascicles ({n_tgt_fasc} target / {n_off_fasc} off-target, "
              f"{n_per_fasc} fibres/fasc)  targets={n_tgt}/{n_total}  "
              f"{diam_str}  divider={divider_deg:.1f}°"
              f"{' (rand)' if RANDOMIZE_DIVIDER else ' (fixed)'}",
              flush=True)

    contact_xyz = make_ring_cuff_positions(
        n_contacts=N_CONTACTS, cuff_radius_um=CUFF_RADIUS_UM, cuff_z_um=0.0,
    )

    if verbose:
        print(f"{label} Precomputing fields (model={FIBER_MODEL}) ...", flush=True)
    Ve_unit, node_indices, geoms = precompute_ve_unit(
        nerve_geom=nerve, n_nodes=N_NODES, contact_xyz_um=contact_xyz,
        model=FIBER_MODEL,
    )
    # Center cuff z at fiber midpoint
    mid_comp = node_indices[0, len(node_indices[0]) // 2]
    mid_z    = float(np.array(section_centers_um(geoms[0]))[mid_comp])
    contact_xyz[:, 2] = mid_z
    Ve_unit, node_indices, geoms = precompute_ve_unit(
        nerve_geom=nerve, n_nodes=N_NODES, contact_xyz_um=contact_xyz,
        model=FIBER_MODEL,
    )

    if verbose:
        print(f"{label} Building solver statics ...", flush=True)
    if FIBER_MODEL == "mrg":
        fs_batch = stack_fiber_statics(geoms, DT)
        s0_batch = initial_states_batch(geoms)
    else:
        # Non-MRG path: monkey-patch the optimiser's integrators so its 7
        # call sites get dispatched to the generic per-model solvers.  The
        # opaque fs_batch / s0_batch placeholders below satisfy the
        # optimiser's _tile_fiber_statics / _tile_states helpers; the
        # patched integrators ignore them and use the per-fibre closures
        # captured at registration time.
        from jaxfibers.stim import batch_solve_generic as _bsg
        from jaxfibers.stim.batch_solve import FiberStatics
        import jaxfibers.optim.optimizer as _opt
        _opt.batch_integrate_m_max = _bsg.make_batch_integrate_m_max_factory(
            FIBER_MODEL, geoms, DT,
        )
        _opt.batch_integrate_m_max_fd = _bsg.make_batch_integrate_m_max_fd_factory(
            FIBER_MODEL, geoms, DT,
        )
        n_comp = geoms[0].n_comp
        # Tilable-compatible dummy: same shape per field as the MRG one
        # would have, just filled with zeros (or False for is_node).
        _z   = jnp.zeros((N_FIBERS, n_comp), dtype=jnp.float64)
        _z_b = jnp.zeros((N_FIBERS, n_comp), dtype=jnp.bool_)
        _z2  = jnp.zeros((N_FIBERS, n_comp, 2, 2), dtype=jnp.float64)
        fs_batch = FiberStatics(
            Cm_dt=_z, Cmy_dt=_z, gmy=_z, A_in_cm2=_z,
            is_node=_z_b,
            Gi_diag=_z, Gp_diag=_z,
            Up=_z2, Low=_z2,
            g_pas=_z,
        )
        s0_batch = (_z, _z, _z, _z)   # (M, H, MP, S) shape — content unused

    N_STEPS = int(T_STOP / DT)
    t_grid  = (np.arange(N_STEPS) + 1) * DT
    pulse_mask = np.where(
        (t_grid >= DELAY_MS) & (t_grid < DELAY_MS + PW_MS), 1.0, 0.0,
    ).astype(np.float64)

    # Baseline (zero stimulation)
    m_max_zero = activation_proxy_batch(
        jnp.zeros((N_FIBERS, geoms[0].n_comp), dtype=jnp.float64),
        jnp.asarray(node_indices, dtype=jnp.int32),
    )
    si_baseline = selectivity_index(np.array(m_max_zero), nerve.target_mask)

    return dict(
        seed=seed, label=label,
        nerve=nerve, geoms=geoms,
        contact_xyz=contact_xyz,
        Ve_unit=Ve_unit, node_indices=node_indices,
        fs_batch=fs_batch, s0_batch=s0_batch,
        pulse_mask=pulse_mask, N_STEPS=N_STEPS,
        si_baseline=si_baseline,
    )


def _waveform_step_per_seed(seed_in: dict, rect_amps: np.ndarray, verbose: bool) -> dict:
    """Run waveform optimisation for a single seed, warm-started from rect amps."""
    label = seed_in["label"]
    pulse_mask = seed_in["pulse_mask"]
    N_STEPS = seed_in["N_STEPS"]

    u_init = np.zeros((N_CONTACTS, N_STEPS), dtype=np.float64)
    for k in range(N_CONTACTS):
        u_init[k] = float(rect_amps[k]) * pulse_mask

    if verbose:
        print(f"{label} Waveform optimisation ({N_OPT_WAVE} iters) ...", flush=True)
    t0 = time.time()
    wave_res = run_waveform_optimization(
        fiber_statics_batch=seed_in["fs_batch"],
        state0_batch=seed_in["s0_batch"],
        Ve_unit=jnp.asarray(seed_in["Ve_unit"], dtype=jnp.float64),
        node_indices=seed_in["node_indices"],
        target_mask=seed_in["nerve"].target_mask,
        dt=DT, T=N_STEPS, n_steps=N_OPT_WAVE, u_init=u_init,
        lr=WAVE_LR, early_stop_patience=WAVE_PATIENCE,
        verbose=verbose,
    )
    return wave_res, time.time() - t0


def _package_result(seed_in: dict, rect_lbfgs_res: dict, rect_t: float,
                    wave_res: dict, wave_t: float) -> dict:
    """Convert LBFGS rect output + Adam wave output into a per-seed JSON dict."""
    nerve = seed_in["nerve"]

    # rect_lbfgs_res may come from the batched or single-seed entry; both have
    # the same dict structure (see optimizer.run_rect_optimization_lbfgs).
    si_rect_final = selectivity_index(rect_lbfgs_res["final_acts"], nerve.target_mask)

    return {
        "seed": seed_in["seed"],
        "si_baseline": seed_in["si_baseline"],
        "rect": {
            "optimizer": "LBFGS-multistart",
            "n_restarts": N_RESTARTS_RECT,
            "n_steps": N_OPT_RECT,
            "best_restart": rect_lbfgs_res["best_restart"],
            "final_loss": rect_lbfgs_res["final_loss"],
            "final_si": si_rect_final,
            "loss_history": rect_lbfgs_res["all_loss_traces"][
                rect_lbfgs_res["best_restart"]
            ].tolist(),
            "all_final_losses": rect_lbfgs_res["all_final_losses"].tolist(),
            "amps_mA": rect_lbfgs_res["amps"].tolist(),
            "final_acts": rect_lbfgs_res["final_acts"].tolist(),
            "time_s": rect_t,
        },
        "waveform": {
            "optimizer": "Adam",
            "n_steps": N_OPT_WAVE,
            "lr": WAVE_LR,
            "patience": WAVE_PATIENCE,
            # final_si / final_acts now report the BEST-LOSS iter (not the
            # last iter), because Adam can wander off a good warm-start.
            # Phase 3 seed 0 hit SI=0.97 at iter 0 but reported 0.00 at
            # iter 99 under the old packaging — losing the actual result.
            "final_si":    float(wave_res["best_si"]),
            "best_iter":   int(wave_res["best_iter"]),
            "last_si":     (float(wave_res["history"]["si"][-1])
                            if wave_res["history"]["si"] else float(wave_res["best_si"])),
            "n_iters_run": len(wave_res["history"]["loss"]),
            "loss_history": wave_res["history"]["loss"],
            "si_history":  wave_res["history"]["si"],
            "final_acts":  wave_res["best_acts"].tolist(),
            "time_s":      wave_t,
        },
        "nerve": {
            "fiber_diam": nerve.fiber_diam.tolist(),
            "target_mask": nerve.target_mask.tolist(),
            "fiber_x_um": nerve.fiber_x_um.tolist(),
            "fiber_y_um": nerve.fiber_y_um.tolist(),
        },
    }


def _run_seed_chunk(seeds: list[int], verbose: bool = True) -> list[dict]:
    """Run a chunk of seeds together: vmap rect over the chunk, run waveform per-seed.

    A chunk size of 1 reduces to the legacy single-seed path with LBFGS.
    """
    print(f"\n{'='*60}\n[chunk] Running {len(seeds)} seeds: {seeds}", flush=True)
    seed_inputs = [_build_seed_inputs(s, verbose=verbose) for s in seeds]

    # ── Rect (LBFGS + multi-restart, batched over seeds when len > 1) ─────────
    pulse_mask = seed_inputs[0]["pulse_mask"]    # same across seeds
    rect_t0 = time.time()
    # amp_init_mA=-1.5, amp_clip=(-3, 3): default (-0.4, ±2.5) was tuned for
    # the mixed-diameter regime where 14-16 µm fibers fire at ~-0.2 mA.
    # With single D=5.7 µm, that init is well below threshold (≈ -1.5 mA
    # at this cuff distance), so the activation proxy is flat and SI stays
    # pinned at 0.000.  Initialising at the expected threshold magnitude
    # gives the gradient useful signal from iter 0.
    AMP_INIT_MA = -1.5
    AMP_CLIP    = (-3.0, 3.0)

    if len(seeds) == 1 and RECT_OPTIMIZER == "adam_fd":
        # Adam-FD smoketest path: pure forward + finite-difference gradient.
        # No autodiff backward, so no remat planner pathology and no
        # @jax.checkpoint required.  Compiles in 2-3 min on A16 vs LBFGS's
        # 25+ min.  Adapt run_rect_optimization's history into the same
        # dict shape that the rest of the script expects from LBFGS.
        s_in = seed_inputs[0]
        print(f"{s_in['label']} Rect Adam-FD ({N_OPT_RECT} iters input → "
              f"{max(N_OPT_RECT * 3, 30)} Adam iters) [OPTIMIZER=adam_fd]",
              flush=True)
        adam_res = run_rect_optimization(
            fiber_statics_batch=s_in["fs_batch"],
            state0_batch=s_in["s0_batch"],
            Ve_unit=s_in["Ve_unit"],
            pulse_mask=pulse_mask,
            node_indices=s_in["node_indices"],
            target_mask=s_in["nerve"].target_mask,
            dt=DT, n_steps=max(N_OPT_RECT * 3, 30),  # Adam needs more iters
            amp_init_mA=AMP_INIT_MA, amp_clip=AMP_CLIP,
            verbose=verbose,
        )
        # Convert Adam-FD result into the LBFGS-result dict shape that
        # _package_result and the figure code expect.
        loss_hist = np.asarray(adam_res["history"]["loss"])
        rect_lbfgs_res = {
            "amps":             adam_res["amps"],
            "final_loss":       float(loss_hist.min()),
            "final_acts":       np.asarray(adam_res["history"]["acts"][-1]),
            "best_restart":     0,
            "all_loss_traces":  loss_hist[None, :],
            "all_final_amps":   adam_res["amps"][None, :],
            "all_final_losses": np.asarray([loss_hist[-1]]),
            "wall_time_s":      0.0,
        }
        rect_lbfgs_results = [rect_lbfgs_res]
    elif len(seeds) == 1:
        s_in = seed_inputs[0]
        print(f"{s_in['label']} Rect LBFGS ({N_RESTARTS_RECT} restarts × "
              f"{N_OPT_RECT} steps) ...", flush=True)
        rect_lbfgs_res = run_rect_optimization_lbfgs(
            fiber_statics_batch=s_in["fs_batch"],
            state0_batch=s_in["s0_batch"],
            Ve_unit=s_in["Ve_unit"],
            pulse_mask=pulse_mask,
            node_indices=s_in["node_indices"],
            target_mask=s_in["nerve"].target_mask,
            dt=DT, n_restarts=N_RESTARTS_RECT, n_steps=N_OPT_RECT,
            amp_init_mA=AMP_INIT_MA, amp_clip=AMP_CLIP,
            rng_seed=seeds[0], verbose=verbose,
        )
        rect_lbfgs_results = [rect_lbfgs_res]
    else:
        print(f"[chunk] Rect LBFGS batched: {len(seeds)} seeds × "
              f"{N_RESTARTS_RECT} restarts × {N_OPT_RECT} steps ...", flush=True)
        rect_lbfgs_results = run_rect_optimization_lbfgs_batched(
            fiber_statics_batches=[s_in["fs_batch"] for s_in in seed_inputs],
            state0_batches=[s_in["s0_batch"] for s_in in seed_inputs],
            Ve_units=np.stack([s_in["Ve_unit"] for s_in in seed_inputs]),
            pulse_mask=pulse_mask,
            node_indices_batches=np.stack([s_in["node_indices"] for s_in in seed_inputs]),
            target_masks=np.stack([s_in["nerve"].target_mask for s_in in seed_inputs]),
            dt=DT, n_restarts=N_RESTARTS_RECT, n_steps=N_OPT_RECT,
            amp_init_mA=AMP_INIT_MA, amp_clip=AMP_CLIP,
            rng_seeds=seeds, verbose=verbose,
        )
    rect_t_total = time.time() - rect_t0
    rect_t_per_seed = rect_t_total / max(len(seeds), 1)

    for s_in, rect_res in zip(seed_inputs, rect_lbfgs_results):
        si_rect = selectivity_index(rect_res["final_acts"], s_in["nerve"].target_mask)
        print(f"{s_in['label']} Rect done: SI {s_in['si_baseline']:+.3f} → "
              f"{si_rect:+.3f}  best_restart={rect_res['best_restart']}", flush=True)

    # ── Waveform (Adam, per-seed) ─────────────────────────────────────────────
    # Skip the waveform step entirely when N_OPT_WAVE=0 (used by the LBFGS
    # verification phase of the manuscript sbatch, which only cares about
    # the rect result).
    results = []
    skip_wave = N_OPT_WAVE <= 0
    for s_in, rect_res in zip(seed_inputs, rect_lbfgs_results):
        si_rect = selectivity_index(rect_res["final_acts"], s_in["nerve"].target_mask)
        if skip_wave:
            print(f"{s_in['label']} Waveform step skipped (N_OPT_WAVE=0).",
                  flush=True)
            rect_acts_np = np.array(rect_res["final_acts"])
            wave_res = {
                "u":         None,
                "best_iter": 0,
                "best_acts": rect_acts_np,
                "best_si":   float(si_rect),
                "history":   {"loss": [], "bce": [], "si": [], "acts": [rect_acts_np]},
            }
            wave_t = 0.0
        else:
            wave_res, wave_t = _waveform_step_per_seed(s_in, rect_res["amps"], verbose)
            si_wave_best = float(wave_res["best_si"])
            si_wave_last = float(wave_res["history"]["si"][-1])
            print(f"{s_in['label']} Wave done: SI {si_rect:+.3f} → "
                  f"{si_wave_best:+.3f} (best @ iter {wave_res['best_iter']}, "
                  f"last={si_wave_last:+.3f})  ({wave_t:.0f}s)", flush=True)

        results.append(_package_result(s_in, rect_res, rect_t_per_seed, wave_res, wave_t))

        # ── Per-seed summary figures ─────────────────────────────────────────
        # Both panels reuse the same cross-section + activation-bar template,
        # but show the rect (LBFGS) and waveform (Adam) endpoints separately
        # so the user can compare what the two stages achieved on the same
        # nerve realisation.
        seed_id = s_in["seed"]
        # Rect figure: LBFGS only stores per-restart loss traces, not per-iter
        # SI, so we pass the winning restart's loss curve and a flat SI line
        # at the achieved value for visual reference.
        rect_loss = rect_res["all_loss_traces"][rect_res["best_restart"]]
        rect_loss_list = np.array(rect_loss).tolist()
        rect_title = (
            f"seed {seed_id} — Rect (Adam-FD, {len(rect_loss_list)} iters)"
            if RECT_OPTIMIZER == "adam_fd"
            else f"seed {seed_id} — Rect (LBFGS, "
                  f"best of {N_RESTARTS_RECT} restarts × {N_OPT_RECT} steps)"
        )
        plot_seed_summary(
            out_path=OUT / f"fig_seed_{seed_id:04d}_rect.png",
            nerve=s_in["nerve"],
            contact_xyz_final=s_in["contact_xyz"],
            amps_mA=np.array(rect_res["amps"]),
            acts=np.array(rect_res["final_acts"]),
            loss_hist=rect_loss_list,
            si_hist=[si_rect] * len(rect_loss_list),
            si_baseline=s_in["si_baseline"],
            title=rect_title,
            nerve_radius_um=NERVE_RADIUS_UM,
            cuff_radius_um=CUFF_RADIUS_UM,
        )
        plot_seed_cross_section(
            out_path=OUT / f"fig_seed_{seed_id:04d}_rect_xsection.png",
            nerve=s_in["nerve"],
            contact_xyz=s_in["contact_xyz"],
            amps_mA=np.array(rect_res["amps"]),
            acts=np.array(rect_res["final_acts"]),
            si=si_rect, si_baseline=s_in["si_baseline"],
            title=rect_title,
            nerve_radius_um=NERVE_RADIUS_UM,
            cuff_radius_um=CUFF_RADIUS_UM,
        )
        # Waveform figure: pick the BEST iter (by SI) rather than the last,
        # so the cross-section reflects the activation pattern at the
        # solution we're actually reporting — not whatever post-overshoot
        # state Adam happened to land in at the final iter.  Skipped if
        # the waveform step was bypassed (N_OPT_WAVE=0, LBFGS verify).
        if skip_wave:
            continue
        wave_si_hist  = list(wave_res["history"]["si"])
        wave_loss     = list(wave_res["history"]["loss"])
        wave_acts_all = wave_res["history"]["acts"]
        best_wave_idx = int(np.argmax(wave_si_hist)) if wave_si_hist else -1
        wave_acts_best = np.asarray(wave_acts_all[best_wave_idx])
        plot_seed_summary(
            out_path=OUT / f"fig_seed_{seed_id:04d}_wave.png",
            nerve=s_in["nerve"],
            contact_xyz_final=s_in["contact_xyz"],
            amps_mA=np.array(rect_res["amps"]),    # waveform doesn't change
                                                     # per-contact amplitudes
                                                     # directly; show rect amps
                                                     # as the steering pattern
            acts=wave_acts_best,
            loss_hist=wave_loss,
            si_hist=wave_si_hist,
            si_baseline=s_in["si_baseline"],
            title=(f"seed {seed_id} — Waveform (Adam, warm-started from rect, "
                    f"{N_OPT_WAVE} iters; cross-section shown at "
                    f"best iter {best_wave_idx})"),
            nerve_radius_um=NERVE_RADIUS_UM,
            cuff_radius_um=CUFF_RADIUS_UM,
        )
        wave_si_best = max(wave_si_hist) if wave_si_hist else float("nan")
        plot_seed_cross_section(
            out_path=OUT / f"fig_seed_{seed_id:04d}_wave_xsection.png",
            nerve=s_in["nerve"],
            contact_xyz=s_in["contact_xyz"],
            amps_mA=np.array(rect_res["amps"]),
            acts=wave_acts_best,
            si=wave_si_best, si_baseline=s_in["si_baseline"],
            title=(f"seed {seed_id} — Waveform (Adam, best iter {best_wave_idx})"),
            nerve_radius_um=NERVE_RADIUS_UM,
            cuff_radius_um=CUFF_RADIUS_UM,
        )
    return results


def main():
    seed_start = int(os.environ.get("SEED_START", 0))
    seed_end   = int(os.environ.get("SEED_END",   N_NERVES))
    seeds_per_task = max(1, int(os.environ.get("SEEDS_PER_TASK", 1)))

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(range(seed_start, seed_end)),
                        help="Which seeds to run (default: SEED_START..SEED_END-1)")
    parser.add_argument("--seeds-per-task", type=int, default=seeds_per_task,
                        help="Vmap the rect optimiser over this many seeds at once. "
                             "Default 1 (no batching). Use 2-8 on a16 GPUs, more on "
                             "a100/h100 if memory allows.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(f"[sweep] Running seeds {args.seeds[0]}..{args.seeds[-1]} "
          f"({len(args.seeds)} seeds), {args.seeds_per_task} per vmap chunk")

    # Pre-filter seeds that already have output, UNLESS FORCE_RERUN=1.
    # The skip is convenient for resuming interrupted multi-seed sweeps but
    # is a footgun for smoketest re-runs (silently produces no new output).
    # Smoketest sbatch always sets FORCE_RERUN=1.
    force_rerun = os.environ.get("FORCE_RERUN", "0") == "1"
    if force_rerun:
        print("[sweep] FORCE_RERUN=1 — re-running every seed and overwriting "
              "existing outputs.", flush=True)
        pending = list(args.seeds)
        skipped = []
    else:
        pending = [s for s in args.seeds
                    if not (OUT / f"data_seed_{s:04d}.json").exists()]
        skipped = [s for s in args.seeds if s not in pending]
    if skipped:
        print(f"[sweep] Skipping {len(skipped)} seeds with existing output: "
              f"{skipped[:5]}{'...' if len(skipped) > 5 else ''}  "
              f"(set FORCE_RERUN=1 to overwrite)", flush=True)

    # Chunk by SEEDS_PER_TASK
    for i in range(0, len(pending), args.seeds_per_task):
        chunk = pending[i:i + args.seeds_per_task]
        chunk_results = _run_seed_chunk(chunk, verbose=not args.quiet)
        for res in chunk_results:
            out_path = OUT / f"data_seed_{res['seed']:04d}.json"
            with open(out_path, "w") as f:
                json.dump(res, f, indent=2)
            print(f"[seed={res['seed']:04d}]  rect SI={res['rect']['final_si']:+.3f}  "
                  f"wave SI={res['waveform']['final_si']:+.3f}", flush=True)

    print(f"\n[sweep] Done. Run analyze_selectivity_sweep.py for aggregate figures.")


if __name__ == "__main__":
    main()
