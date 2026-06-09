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
    stack_fiber_statics, initial_states_batch, batch_integrate_m_max,
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
    select_peripheral_cluster_target, cluster_target_mask,
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
DELAY_MS        = _env_flt("DELAY_MS", 1.0)
PW_MS           = _env_flt("PW_MS",    1.0)
SEED_START      = _env_int("SEED_START", 0)
SEED_END        = _env_int("SEED_END", 25)
N_OPT_RECT      = _env_int("N_OPT_RECT", 30)
N_RESTARTS_RECT = _env_int("N_RESTARTS_RECT", 2)
N_OPT_WAVE      = _env_int("N_OPT_WAVE", 100)
WAVE_LR         = _env_flt("WAVE_LR", 5e-4)
WAVE_PATIENCE   = _env_int("WAVE_PATIENCE", 20)
RECT_OPTIMIZER  = os.environ.get("RECT_OPTIMIZER", "adam_fd")  # adam_fd | lbfgs

# Pulse shape (env-overrideable).  Charge-balanced biphasic is the
# clinical default for any chronic implant -- monophasic deposits net
# charge into tissue.  Options:
#   monophasic     - single rectangular cathodic pulse, PW_MS wide.
#                    Net charge ~ -|amp| * PW_MS.  Reference / legacy.
#   biphasic_sym   - symmetric charge-balanced: +1 for PW_MS/2, then
#                    -1 for PW_MS/2.  Net charge = 0.  Most common in
#                    PNS selectivity research.
#   biphasic_asym  - asymmetric charge-balanced: +1 for PW_MS (strong
#                    cathodic), then -1/ASYM_RATIO for PW_MS*ASYM_RATIO
#                    (weaker, longer anodic recharge).  Net charge = 0.
#                    Reduces anodic-block side effects.  Default
#                    ASYM_RATIO=4 matches typical clinical settings.
PULSE_SHAPE = os.environ.get("PULSE_SHAPE", "biphasic_sym").strip().lower()
ASYM_RATIO  = _env_flt("ASYM_RATIO", 4.0)

# Per-Duke amplitude knobs.  Defaults below are tuned for MRG 5.7 µm
# against the FEM Ve scale of the typical Duke bundle (peak Ve ≈ 800–
# 900 mV/mA per contact, ~5× the synthetic point-source).  The
# original smoketest tuning at AMP_INIT_MA=-0.30 / FD_EPS_MA=0.010
# was retuned after the initial full-cohort sweep showed that on
# unbalanced 2-fascicle anatomies (e.g. human_sub-46_sam-2 with
# 953/47 target split) the init amplitudes landed every fibre either
# deeply sub-threshold or deeply supra-threshold; the 0.010 mA FD
# probe was then too small to flip any fibre's firing state, so the
# FD gradient was exactly zero and Adam-FD stalled for 20+ iters
# before early-stopping at SI=0.  Smaller init + larger FD probe
# brings the init into the gradient-informative threshold regime.
# All env-overrideable.
AMP_INIT_MA = _env_flt("AMP_INIT_MA", -0.08)
_AMP_CLIP_LO = _env_flt("AMP_CLIP_LO", -2.00)
_AMP_CLIP_HI = _env_flt("AMP_CLIP_HI",  2.00)
AMP_CLIP    = (_AMP_CLIP_LO, _AMP_CLIP_HI)
ADAM_LR_MA  = _env_flt("ADAM_LR_MA",  0.005)
FD_EPS_MA   = _env_flt("FD_EPS_MA",   0.030)

# Adam-FD multi-start: a single AMP_INIT_MA cannot put every Duke
# anatomy into the gradient-informative threshold regime simultaneously
# (e.g. unbalanced 2-fascicle nerves with 953/47 splits land entirely
# sub- or supra-threshold at any fixed magnitude, collapsing the FD
# gradient to zero).  Instead we run Adam-FD from K different init
# magnitudes and keep the lowest-loss trajectory.  Comma-separated
# signed mA values; default spans 8x in magnitude.  Set to a single
# value (e.g. "-0.08") to revert to a single-start Adam-FD.
def _parse_restart_mags(env_val: str) -> list[float]:
    if not env_val.strip():
        return [AMP_INIT_MA]
    try:
        return [float(x.strip()) for x in env_val.split(",") if x.strip()]
    except ValueError as e:
        raise ValueError(
            f"ADAM_FD_RESTART_MAGS must be comma-separated signed floats "
            f"(mA), got {env_val!r}"
        ) from e

ADAM_FD_RESTART_MAGS = _parse_restart_mags(
    os.environ.get("ADAM_FD_RESTART_MAGS", "-0.10,-0.30,-0.80,-1.50")
)

# Smart init replaces the multi-restart strategy entirely on the
# default path.  Instead of running 4 full Adam-FD passes at different
# magnitudes (~50 min/sample), we:
#   1. Compute a per-contact target/off-target Ve-contrast pattern
#      (one numpy operation, no GPU).
#   2. Scale that pattern by each candidate magnitude in PROBE_MAGS_MA
#      and run a single forward pass per candidate (~30 s each on
#      A100 → ~3 min total for 6 candidates).
#   3. Pick the magnitude with the highest (target_fired - off_fired)
#      fraction and run a SINGLE Adam-FD from that init
#      (~10-20 min, often early-stops).
# Total ~15-25 min/sample vs ~50 min for 4-restart.  Set
# SMART_INIT_ENABLED=false to fall back to the multi-restart path.
SMART_INIT_ENABLED = os.environ.get("SMART_INIT_ENABLED", "true").strip().lower() in (
    "1", "true", "yes", "y", "on",
)
# Probe magnitudes (signed mA).  Sign convention: negative = cathodic
# for the target-preferring contacts.  Span ~75x to cover:
#   - tightly-coupled bipolar (threshold ~0.05-0.2 mA)
#   - weakly-coupled bipolar (threshold ~0.5-0.8 mA)
#   - tripolar with axial guards (threshold 1.5-2x bipolar, can reach
#     ~1.5-2.0 mA on hard anatomies like sub-53 where contacts are
#     800-1000 um from the target fascicle).
# The ceiling MUST exceed AMP_CLIP otherwise the probe never explores
# the actual operating range available to the optimizer.
PROBE_MAGS_MA = _parse_restart_mags(
    os.environ.get("PROBE_MAGS_MA",
                    "-0.02,-0.05,-0.10,-0.20,-0.40,-0.80,-1.20,-1.50")
)

# L1-discovery: an alternative optimization path that does NOT rely on
# hand-coded spatial patterns (bipolar / tripolar).  Instead it starts
# from a random init and uses L1 (proximal soft-thresholding) to drive
# uninformative contacts toward zero, so the optimizer DISCOVERS which
# contacts matter.  When enabled it runs ALONGSIDE the probe-based path
# (not in place of it), and its results are saved in JSON.rect_l1
# parallel to JSON.rect.  Reviewer-facing point: this validates the
# probe path isn't "cheating" -- the optimizer arrives at similar
# sparse configurations on its own, just more slowly.
L1_DISCOVERY_ENABLED = os.environ.get(
    "L1_DISCOVERY_ENABLED", "false"
).strip().lower() in ("1", "true", "yes", "y", "on")
L1_LAMBDA      = _env_flt("L1_LAMBDA",      0.10)   # sparsity penalty strength
L1_INIT_SCALE  = _env_flt("L1_INIT_SCALE",  0.10)   # random uniform [-S, +S] mA
L1_INIT_SEED   = _env_int("L1_INIT_SEED",   0)      # rng seed for random init
L1_N_ITERS     = _env_int("L1_N_ITERS",     150)    # more than probe path

# NOTE: PROBE_L1_LAMBDA was replaced by a hard freeze_zero_mask passed
# to run_rect_optimization (zeros stay at exactly zero throughout the
# probe-path Adam-FD).  L1 prox at sensible lambdas is too weak to pin
# given the per-step Adam update magnitude (~lr=0.005 mA per iter).


# ── target-fascicle selection paradigm ────────────────────────────────
# The original divider-line-through-centroid paradigm was replaced
# after sanity-figure review showed it produced topologically
# meaningless target/off-target splits.  See duke_loader:
# select_peripheral_cluster_target for the cluster-based paradigm.
#
# Set TARGET_PARADIGM=cluster (default) for the peripheral-cluster
# selection.  Set TARGET_PARADIGM=divider to fall back to the legacy
# random-divider behaviour (kept available for reproducing pre-cluster
# results).
TARGET_PARADIGM         = os.environ.get("TARGET_PARADIGM", "cluster").strip().lower()
CLUSTER_RADIUS_QUANTILE = _env_flt("CLUSTER_RADIUS_QUANTILE", 0.6)
CLUSTER_WINDOW_DEG      = _env_flt("CLUSTER_WINDOW_DEG", 90.0)
CLUSTER_N_MIN_TARGET    = _env_int("CLUSTER_N_MIN_TARGET", 50)
CLUSTER_MIN_FASCICLES   = _env_int("CLUSTER_MIN_FASCICLES", 3)


def _build_pulse_mask(t_grid: np.ndarray, delay_ms: float, pw_ms: float,
                      shape: str, asym_ratio: float) -> np.ndarray:
    """Construct the per-step pulse-mask waveform for one stim cycle.

    Returns a length-T array with values in [-1, +1], normalised so
    that the cathodic peak is +1 and the optimiser's ``amps[k]``
    multiplier scales the whole shape.  All shapes are charge-balanced
    *except* the monophasic reference.
    """
    shape = shape.strip().lower()
    # Verify the chosen pulse fits inside the t_grid window; otherwise
    # the integrator silently truncates the anodic phase and the
    # waveform is no longer charge-balanced.
    pulse_end_ms = delay_ms + pw_ms
    if shape == "biphasic_asym":
        pulse_end_ms = delay_ms + pw_ms * (1.0 + float(asym_ratio))
    t_max = float(t_grid[-1])
    if pulse_end_ms > t_max:
        raise ValueError(
            f"Pulse ends at t={pulse_end_ms:.3f} ms (shape={shape!r}, "
            f"delay={delay_ms:.3f}, PW={pw_ms:.3f}, asym_ratio={asym_ratio}) "
            f"but t_grid only extends to {t_max:.3f} ms.  Increase T_STOP "
            f"or shorten the pulse."
        )
    pulse = np.zeros_like(t_grid, dtype=np.float64)
    if shape == "monophasic":
        m = (t_grid >= delay_ms) & (t_grid < delay_ms + pw_ms)
        pulse[m] = 1.0
    elif shape == "biphasic_sym":
        half = pw_ms / 2.0
        cath = (t_grid >= delay_ms) & (t_grid < delay_ms + half)
        anod = (t_grid >= delay_ms + half) & (t_grid < delay_ms + pw_ms)
        pulse[cath] =  1.0
        pulse[anod] = -1.0
    elif shape == "biphasic_asym":
        # Strong cathodic phase, weaker long anodic recharge.  Charge
        # balanced: anodic_amp * anodic_dur == cathodic_amp * cathodic_dur
        cath_dur = pw_ms
        anod_dur = pw_ms * float(asym_ratio)
        anod_amp = -1.0 / float(asym_ratio)
        cath = (t_grid >= delay_ms) & (t_grid < delay_ms + cath_dur)
        anod = ((t_grid >= delay_ms + cath_dur)
                & (t_grid < delay_ms + cath_dur + anod_dur))
        pulse[cath] = 1.0
        pulse[anod] = anod_amp
    else:
        raise ValueError(
            f"Unknown PULSE_SHAPE={shape!r}; expected one of "
            f"'monophasic', 'biphasic_sym', 'biphasic_asym'"
        )
    return pulse


def _class_balanced_weights(target_mask: np.ndarray) -> np.ndarray:
    """Per-fibre weights that give equal *class* weight to target and
    off-target populations.  Without this, an imbalanced split (e.g.
    400 target / 600 off-target on a Duke nerve) makes the loss
    dominated by whichever class is larger and the optimiser has
    little incentive to spare the smaller class -- in extreme cases
    the global minimum of the uniform-weight loss is just "fire
    everything" rather than the selective config.

    Returns a length-n_fibres float64 vector that sums to 1.0.
    """
    target_mask = np.asarray(target_mask, dtype=bool)
    n_t = int(target_mask.sum())
    n_nt = int((~target_mask).sum())
    n_total = int(target_mask.size)
    if n_t == 0 or n_nt == 0:
        # Degenerate -- fall back to uniform.  This shouldn't happen
        # on a valid cluster-target split with n_min_target_fibers >= 50.
        return (np.ones(n_total, dtype=np.float64) / max(n_total, 1))
    return np.where(target_mask, 0.5 / n_t, 0.5 / n_nt).astype(np.float64)


def _phi_to_compass(phi_deg: float) -> str:
    """Convert a phi angle (atan2(y,x), degrees, [-180, 180]) to an
    8-point compass label (E, NE, N, NW, W, SW, S, SE).  Used to name
    the per-column tripolar patterns so log lines and JSON keys are
    self-describing (`tripolar_W`, `tripolar_S`, ...).
    """
    # Normalize to [0, 360) where 0 = +x (east), 90 = +y (north).
    phi = float(phi_deg) % 360.0
    # 8-point compass, centred on the nominal direction.
    compass = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    idx = int(np.round(phi / 45.0)) % 8
    return compass[idx]


def _angular_column_groups(contact_xyz_um: np.ndarray,
                            xy_tol_um: float = 50.0) -> list[list[int]]:
    """Cluster contacts into angular columns by (x, y) proximity.

    Returns a list of column groups (list of contact indices), one per
    distinct angular position.  Within a column, contacts differ only
    in z (axial position).
    """
    K = int(contact_xyz_um.shape[0])
    contact_xy = np.asarray(contact_xyz_um, dtype=np.float64)[:, :2]
    assigned = np.zeros(K, dtype=bool)
    groups: list[list[int]] = []
    for i in range(K):
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True
        for j in range(i + 1, K):
            if assigned[j]:
                continue
            d_xy = float(np.hypot(contact_xy[i, 0] - contact_xy[j, 0],
                                   contact_xy[i, 1] - contact_xy[j, 1]))
            if d_xy < xy_tol_um:
                group.append(j)
                assigned[j] = True
        groups.append(group)
    return groups


def _tripolar_pattern_for_column(column: list[int], n_contacts: int,
                                  contact_z_um: np.ndarray) -> np.ndarray:
    """Build a single-column tripolar (or bipolar / monopolar) spatial
    pattern of length ``n_contacts``.  All contacts not in ``column``
    are zero.  Within the column: outer anodic guards, middle cathodic,
    charge-balanced.  Caller scales by a signed magnitude.
    """
    column_sorted = sorted(column, key=lambda k: float(contact_z_um[k]))
    n_col = len(column_sorted)
    pattern = np.zeros(n_contacts, dtype=np.float64)
    if n_col == 1:
        pattern[column_sorted[0]] = -1.0
    elif n_col == 2:
        pattern[column_sorted[0]] =  1.0
        pattern[column_sorted[1]] = -1.0
    else:
        mid     = n_col // 2
        n_outer = n_col - 1
        outer_amp = 1.0 / float(n_outer)   # so sum(outer) = +1 = -middle
        for idx, k in enumerate(column_sorted):
            pattern[k] = -1.0 if idx == mid else outer_amp
    return pattern


def _sparse_tripolar_patterns_per_column(
        spatial_contrast: np.ndarray,
        contact_xyz_um: np.ndarray,
        verbose: bool = False,
        label: str = "") -> dict[str, np.ndarray]:
    """Build one tripolar pattern *per angular column* (N, E, S, W, ...).

    The target fascicle cluster may lie close to ANY angular position
    around the nerve cross-section, so probing only the "best contrast"
    column (the previous behaviour) was brittle -- the contrast metric
    uses |Ve| and doesn't always pick the right side.  By emitting one
    pattern per column we let the magnitude probe decide which spatial
    configuration is best for this anatomy, on the same footing as the
    bipolar pattern.

    Returns a dict mapping ``f"tripolar_{compass}"`` (e.g.
    ``"tripolar_W"``) to a length-K spatial pattern.  Caller adds these
    into ``spatial_patterns`` for the magnitude probe.
    """
    K = int(len(spatial_contrast))
    contact_xy = np.asarray(contact_xyz_um, dtype=np.float64)[:, :2]
    contact_z  = np.asarray(contact_xyz_um, dtype=np.float64)[:, 2]
    if contact_xy.shape[0] != K:
        return {}

    groups = _angular_column_groups(contact_xyz_um, xy_tol_um=50.0)
    if not groups:
        return {}

    patterns: dict[str, np.ndarray] = {}
    diagnostics: list[str] = []
    used_names: dict[str, int] = {}
    for grp in groups:
        # Mean phi of the column → compass name.
        mean_x = float(np.mean([contact_xy[k, 0] for k in grp]))
        mean_y = float(np.mean([contact_xy[k, 1] for k in grp]))
        phi_deg = float(np.degrees(np.arctan2(mean_y, mean_x)))
        compass = _phi_to_compass(phi_deg)
        # Disambiguate collisions (two columns rounding to the same compass
        # bin, rare but possible on irregular cuffs) by appending an index.
        name = f"tripolar_{compass}"
        if name in patterns:
            used_names[compass] = used_names.get(compass, 1) + 1
            name = f"tripolar_{compass}{used_names[compass]}"

        pattern = _tripolar_pattern_for_column(grp, K, contact_z)
        patterns[name] = pattern

        col_score = float(np.mean([spatial_contrast[k] for k in grp]))
        diagnostics.append(
            f"{name} (phi={phi_deg:+.0f}°, mean contrast={col_score:+.3f})"
        )

    if verbose:
        diag = "; ".join(diagnostics)
        print(f"{label} Tripolar patterns (one per column): {diag}",
              flush=True)
    return patterns


def _run_l1_discovery(seed_in: dict, n_iters: int, l1_lambda: float,
                       init_scale: float, init_seed: int,
                       verbose: bool = True) -> dict:
    """L1-regularised Adam-FD from a random init.

    This is the *principled-discovery* alternative to the probe-based
    smart init.  Instead of starting from a hand-coded spatial pattern
    (bipolar / tripolar) and finding amplitudes, this path:

    1. Initialises amps from uniform random ``[-init_scale, +init_scale]``
       on every contact -- NO prior knowledge of which contacts should
       be active.
    2. Runs Adam-FD with an L1 proximal soft-thresholding step after
       each Adam update.  Contacts whose amplitude drops below
       ``lr * l1_lambda`` are pushed exactly to zero -- the optimizer
       therefore DISCOVERS which subset of contacts contributes to
       selective activation.

    For a defensible methodology paper this is essential validation: if
    L1-discovery converges to similar selectivity as the probe-based
    path on the same anatomies, we have evidence that the hand-coded
    bipolar / tripolar patterns are not a shortcut -- they are a
    speedup of a sparsity-discovery process the optimizer can do on
    its own.
    """
    label = seed_in["label"]
    K = seed_in["Ve_unit"].shape[0]
    if verbose:
        print(f"{label} L1-discovery: random init [+/-{init_scale:.3f} mA, "
              f"seed={init_seed}] + Adam-FD + L1 prox (lambda={l1_lambda}, "
              f"{n_iters} iters)", flush=True)

    rng = np.random.default_rng(int(init_seed))
    amps_init = rng.uniform(-init_scale, init_scale, size=K).astype(np.float64)
    if verbose:
        init_str = "  ".join(f"{a:+.3f}" for a in amps_init)
        print(f"{label} L1 init amps [{init_str}] mA", flush=True)

    t0 = time.time()
    res = run_rect_optimization(
        fiber_statics_batch=seed_in["fs_batch"],
        state0_batch=seed_in["s0_batch"],
        Ve_unit=seed_in["Ve_unit"],
        pulse_mask=seed_in["pulse_mask"],
        node_indices=seed_in["node_indices"],
        target_mask=seed_in["target_mask"],
        weights=seed_in["weights"],
        dt=DT, n_steps=n_iters,
        amps_init_vector=amps_init,
        amp_clip=AMP_CLIP,
        lr=ADAM_LR_MA, fd_eps=FD_EPS_MA,
        l1_lambda=l1_lambda,
        verbose=verbose,
    )
    wall_s = time.time() - t0

    loss_hist = np.asarray(res["history"]["loss"])
    best_iter = int(np.argmin(loss_hist))
    best_amps = np.asarray(res["history"]["amps"][best_iter])
    best_acts = np.asarray(res["history"]["acts"][best_iter])
    n_active = int(np.sum(np.abs(best_amps) > 0.0))
    if verbose:
        print(f"{label} L1-discovery done: best_loss="
              f"{float(loss_hist[best_iter]):.4f} @ iter {best_iter}, "
              f"{n_active}/{K} contacts non-zero ({wall_s:.0f}s)",
              flush=True)
    return {
        "optimizer":        "Adam-FD-L1",
        "amps":             best_amps,
        "loss_history":     loss_hist,
        "final_loss":       float(loss_hist[best_iter]),
        "final_acts":       best_acts,
        "best_iter":        best_iter,
        "n_steps":          n_iters,
        "l1_lambda":        float(l1_lambda),
        "init_scale_mA":    float(init_scale),
        "init_seed":        int(init_seed),
        "amps_init_vector": amps_init.tolist(),
        "n_active_contacts": n_active,
        "wall_s":           float(wall_s),
        "all_final_losses": np.asarray([float(loss_hist[-1])]),
    }


def _smart_spatial_pattern(Ve_unit: np.ndarray,
                            target_mask: np.ndarray,
                            verbose: bool = False,
                            label: str = "") -> np.ndarray:
    """Compute the per-contact target/off-target Ve contrast pattern.

    For each contact k:
        contrast[k] = (mean|Ve[k, target]| - mean|Ve[k, off]|) /
                       (mean|Ve[k, target]| + mean|Ve[k, off]| + eps)

    contrast[k] > 0  : contact reaches target more than off-target
                       → should be cathodic (negative amps).
    contrast[k] < 0  : contact reaches off-target more than target
                       → should be anodic (positive amps; guard).

    Returns a length-K vector in [-1, +1].  Multiply by a NEGATIVE
    magnitude (mA) to get the actual amps init vector: amps = mag *
    contrast with mag < 0 puts cathodic current on target-preferring
    contacts and anodic current on off-target-preferring contacts —
    the optimal guard-pattern start.
    """
    Ve_unit = np.asarray(Ve_unit)
    target_mask = np.asarray(target_mask, dtype=bool)
    K = Ve_unit.shape[0]
    if int(target_mask.sum()) == 0 or int((~target_mask).sum()) == 0:
        return np.zeros(K, dtype=np.float64)
    # Per-fiber peak Ve magnitude across compartments.
    Ve_abs_max = np.max(np.abs(Ve_unit), axis=2)          # [K, n_fibers]
    mean_tgt = Ve_abs_max[:, target_mask].mean(axis=1)    # [K]
    mean_off = Ve_abs_max[:, ~target_mask].mean(axis=1)   # [K]
    raw_contrast = (mean_tgt - mean_off) / (mean_tgt + mean_off + 1e-10)
    # The raw contrast is mathematically in [-1, +1] but in practice it
    # is often much smaller (typical Duke geometry gives raw spreads of
    # ~±0.05-0.15 because each fascicle is reached by *all* contacts to
    # some degree).  Stretching to the full [-1, +1] range gives the
    # magnitude probe a meaningful per-contact scale: amps = mag * pattern
    # then spans [-|mag|, +|mag|] as intended.  Without this stretch the
    # probe at mag=-0.05 produces amps near ±0.005 mA, which is
    # sub-threshold for every fibre at every magnitude tried -- the
    # entire probe trace then reports score = 0 and the first (smallest)
    # magnitude wins by default.
    rng = float(raw_contrast.max() - raw_contrast.min())
    if rng > 1e-10:
        contrast = 2.0 * (raw_contrast - raw_contrast.min()) / rng - 1.0
    else:
        contrast = np.zeros_like(raw_contrast)
    if verbose:
        print(f"{label} Raw contrast: range "
              f"[{float(raw_contrast.min()):+.4f}, "
              f"{float(raw_contrast.max()):+.4f}]  "
              f"span={rng:.4f}  (normalised to [-1, +1])",
              flush=True)
    return contrast.astype(np.float64)


def _magnitude_probe(spatial_patterns: dict,
                      probe_mags_mA: list,
                      target_mask: np.ndarray,
                      fs_batch, s0_batch, Ve_unit_j, pulse_mask_j,
                      node_idx_j, dt: float,
                      label: str = "",
                      verbose: bool = True) -> tuple:
    """For each (pattern, mag) combination, evaluate one forward and
    pick the configuration with the highest selectivity score.

    ``spatial_patterns`` is a dict mapping a short pattern name (e.g.
    ``"bipolar"``, ``"tripolar"``) to a length-K spatial pattern.  All
    pattern × signed-magnitude combinations are tested; strict signed
    comparison on ``(target_frac - off_frac)`` picks the best.

    Returns
    -------
    (amps_vec, best_mag, best_score, best_acts, probe_history, best_pattern_name)
    """
    target_mask_bool = np.asarray(target_mask, dtype=bool)
    n_t = int(target_mask_bool.sum())
    n_off = int((~target_mask_bool).sum())

    @jax.jit
    def _fwd(amps):
        # amps [K]  *  pulse_mask [T]  →  u [K, T]
        u = amps[:, None] * pulse_mask_j[None, :]
        # Per-fibre Ve trajectory: -sum_k u_kt * Ve_unit_kfn  → [F, T, n_c]
        Ve_seq = jnp.einsum("kt,kfn->ftn", -u, Ve_unit_j)
        m_max = batch_integrate_m_max(fs_batch, s0_batch, Ve_seq, dt)
        return activation_proxy_batch(m_max, node_idx_j)

    # Dual-polarity probe: the contrast pattern's sign can be wrong for
    # some geometries (the |Ve| metric in _smart_spatial_pattern doesn't
    # distinguish "contact reaches target with positive Ve" from "contact
    # reaches target with negative Ve" -- but only the former activates
    # the target when cathodic).  Testing both +|mag| and -|mag| × pattern
    # always finds the right polarity for the anatomy.
    unique_abs = sorted({abs(float(m)) for m in probe_mags_mA})
    signed_mags = []
    for m_abs in unique_abs:
        signed_mags.extend([-m_abs, +m_abs])

    active_patterns = {name: p for name, p in spatial_patterns.items()
                        if np.any(np.asarray(p) != 0)}
    if verbose:
        names = ", ".join(active_patterns.keys()) or "<none>"
        n_total = len(active_patterns) * len(signed_mags)
        print(f"{label} Magnitude probe: {n_total} candidates "
              f"({len(active_patterns)} patterns × "
              f"{len(signed_mags)} signed mags) -- patterns: [{names}]",
              flush=True)

    best_score = -float("inf")
    best_mag = float(signed_mags[0]) if signed_mags else 0.0
    best_pattern_name = next(iter(active_patterns.keys())) if active_patterns else "none"
    best_amps: np.ndarray | None = None
    best_acts: np.ndarray | None = None
    history: list[dict] = []
    for pattern_name, spatial_pattern in active_patterns.items():
        if verbose:
            n_nz = int(np.sum(spatial_pattern != 0))
            print(f"{label}  -- pattern '{pattern_name}' "
                  f"({n_nz}/{len(spatial_pattern)} non-zero entries) --",
                  flush=True)
        for mag in signed_mags:
            amps_vec = float(mag) * np.asarray(spatial_pattern,
                                                dtype=np.float64)
            amps_j = jnp.asarray(amps_vec, dtype=jnp.float64)
            t0 = time.time()
            acts = _fwd(amps_j)
            acts_np = np.asarray(acts)
            dt_ms = (time.time() - t0) * 1000.0
            fired = acts_np > 0.5
            nft = int(np.sum(fired & target_mask_bool))
            nfn = int(np.sum(fired & ~target_mask_bool))
            target_frac = nft / max(n_t, 1)
            off_frac = nfn / max(n_off, 1)
            score = target_frac - off_frac
            history.append({
                "pattern_name":    pattern_name,
                "mag_mA":          float(mag),
                "amps_vec":        amps_vec.tolist(),
                "n_fired_target":  nft,
                "n_fired_off":     nfn,
                "target_frac":     float(target_frac),
                "off_frac":        float(off_frac),
                "score":           float(score),
                "wall_ms":         float(dt_ms),
            })
            if verbose:
                print(f"  probe '{pattern_name}' mag={float(mag):+.3f} mA  "
                      f"fired={nft}/{n_t}t+{nfn}/{n_off}nt "
                      f"({100*target_frac:.0f}%T, {100*off_frac:.0f}%NT)  "
                      f"score={score:+.3f}  dt={dt_ms:.0f}ms",
                      flush=True)
            if score > best_score:
                best_score = score
                best_mag = float(mag)
                best_pattern_name = pattern_name
                best_amps = amps_vec.copy()
                best_acts = acts_np
    if verbose:
        print(f"{label} Probe winner: pattern='{best_pattern_name}' "
              f"mag={best_mag:+.3f} mA  score={best_score:+.3f}",
              flush=True)
    return (best_amps, best_mag, float(best_score), best_acts,
            history, best_pattern_name)


def _firing_summary(acts, target_mask) -> dict:
    """Compute per-group firing counts from ``acts`` (sigmoid-like activation
    proxy in [0,1], > 0.5 ≈ fired).  Distinguishing oversaturated (everything
    fires) from undersaturated (nothing fires) is essential for diagnosing
    optimiser stalls — both give SI = 0 but call for opposite tuning fixes
    (smaller vs larger init amplitude)."""
    acts = np.asarray(acts).ravel()
    target_mask = np.asarray(target_mask, dtype=bool).ravel()
    fired = acts > 0.5
    n_target    = int(target_mask.sum())
    n_nontarget = int((~target_mask).sum())
    n_total     = int(target_mask.size)
    n_fired_target    = int(np.sum(fired & target_mask))
    n_fired_nontarget = int(np.sum(fired & ~target_mask))
    n_fired_total     = int(fired.sum())
    return {
        "n_target":             n_target,
        "n_nontarget":          n_nontarget,
        "n_total":              n_total,
        "n_fired_target":       n_fired_target,
        "n_fired_nontarget":    n_fired_nontarget,
        "n_fired_total":        n_fired_total,
        "frac_fired_total":     n_fired_total / max(n_total, 1),
        "frac_fired_target":    n_fired_target / max(n_target, 1),
        "frac_fired_nontarget": n_fired_nontarget / max(n_nontarget, 1),
    }


def _fmt_firing(fs: dict) -> str:
    """One-line human-readable firing summary suitable for the per-stage log."""
    return (f"fired {fs['n_fired_target']}/{fs['n_target']} tgt, "
            f"{fs['n_fired_nontarget']}/{fs['n_nontarget']} nt "
            f"({100.0 * fs['frac_fired_total']:.1f}% total)")


def _build_seed(duke: dict, seed: int, verbose: bool = True) -> dict:
    """Per-seed pipeline: redraw divider angle, set target_mask, build
    solver statics, baseline SI.  All compute-light steps — the heavy
    Ve_unit + geometry come from ``duke`` and are seed-independent."""
    label = f"[{SAMPLE_NAME} | seed {seed:4d}]"
    nerve = duke["nerve_geom"]

    # Target/off-target mask.  The cluster paradigm is the default;
    # divider is kept as a fallback for reproducing pre-cluster runs.
    cluster_info: dict | None = None
    divider_deg = 0.0
    if TARGET_PARADIGM == "cluster":
        cluster_info = select_peripheral_cluster_target(
            duke["fasc_meta"], duke["fasc_id"],
            duke["nerve_outline_xy_um"],
            radius_quantile=CLUSTER_RADIUS_QUANTILE,
            angular_window_deg=CLUSTER_WINDOW_DEG,
            n_min_target_fibers=CLUSTER_N_MIN_TARGET,
        )
        if not cluster_info["target_ids"]:
            if verbose:
                print(f"{label} SKIP: no peripheral cluster meets "
                      f"n_min_target_fibers={CLUSTER_N_MIN_TARGET} "
                      f"(best cluster had {cluster_info['n_target_fibers']} "
                      f"fibres)", flush=True)
            return None  # signal "skip this sample"
        target_mask = cluster_target_mask(
            nerve, duke["fasc_id"], duke["fasc_meta"],
            cluster_info["target_ids"],
        )
        n_tgt = int(target_mask.sum())
        n_tgt_fasc = len(cluster_info["target_ids"])
        if verbose:
            print(f"{label} cluster target: "
                  f"sector {cluster_info['window_start_deg']:.0f}-"
                  f"{cluster_info['window_end_deg']:.0f}°  "
                  f"target_fascs={cluster_info['target_ids']}  "
                  f"targets={n_tgt}/{nerve.n_fibers} "
                  f"({n_tgt_fasc}/{len(nerve.fascicles)} fascicles)",
                  flush=True)
    else:
        # Legacy random-divider paradigm.
        divider_deg = float(
            np.random.default_rng(seed + 1_000_003).uniform(0.0, 180.0)
        )
        target_mask = divider_split_target_mask(
            nerve, duke["fasc_id"], duke["fasc_meta"], divider_deg
        )
        n_tgt = int(target_mask.sum())
        n_tgt_fasc = sum(1 for f in nerve.fascicles if f.is_target)
        if verbose:
            print(f"{label} divider={divider_deg:5.1f}°  "
                  f"targets={n_tgt}/{nerve.n_fibers} "
                  f"({n_tgt_fasc}/{len(nerve.fascicles)} fascicles)",
                  flush=True)

    geoms = duke["geoms"]
    fs_batch = stack_fiber_statics(geoms, DT)
    s0_batch = initial_states_batch(geoms)

    N_STEPS = int(T_STOP / DT)
    t_grid  = (np.arange(N_STEPS) + 1) * DT
    pulse_mask = _build_pulse_mask(
        t_grid, DELAY_MS, PW_MS, PULSE_SHAPE, ASYM_RATIO
    )
    # Class-balanced per-fibre weights so the loss isn't dominated by
    # whichever of target / off-target is larger.  See _class_balanced_weights
    # docstring for the motivation.
    weights = _class_balanced_weights(target_mask)

    # Baseline (zero stimulation)
    m_max_zero = activation_proxy_batch(
        jnp.zeros((nerve.n_fibers, geoms[0].n_comp), dtype=jnp.float64),
        jnp.asarray(duke["node_indices"], dtype=jnp.int32),
    )
    si_baseline = selectivity_index(np.array(m_max_zero), target_mask)

    return dict(
        seed=seed, label=label, divider_deg=divider_deg,
        cluster_info=cluster_info,
        nerve=nerve, geoms=geoms,
        target_mask=target_mask, weights=weights,
        Ve_unit=duke["Ve_unit"], node_indices=duke["node_indices"],
        contact_xyz_um=duke["contact_xyz_um"],
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
    if RECT_OPTIMIZER == "adam_fd" and SMART_INIT_ENABLED:
        # ── Smart init path ─────────────────────────────────────────
        # (1) per-contact spatial pattern from target/off-target Ve contrast,
        # (2) magnitude probe over PROBE_MAGS_MA (single forwards),
        # (3) ONE Adam-FD run from amps_init = best_mag × spatial_pattern.
        n_iters = max(N_OPT_RECT * 3, 30)
        if verbose:
            print(f"{label} Smart init path: spatial-contrast pattern + "
                  f"{len(PROBE_MAGS_MA)}-magnitude probe + 1×Adam-FD "
                  f"({n_iters} iters)", flush=True)

        spatial_bipolar = _smart_spatial_pattern(
            seed_in["Ve_unit"], seed_in["target_mask"],
            verbose=verbose, label=label,
        )
        tripolar_patterns_by_col = _sparse_tripolar_patterns_per_column(
            spatial_bipolar, seed_in["contact_xyz_um"],
            verbose=verbose, label=label,
        )
        # All columns become first-class probe candidates: the target
        # fascicle cluster may lie close to ANY angular position around
        # the nerve cross-section, so the probe needs to test each one.
        spatial_patterns = {"bipolar": spatial_bipolar}
        spatial_patterns.update(tripolar_patterns_by_col)
        # For backward-compat with downstream JSON code that expects a
        # single canonical "spatial pattern" we keep the dense bipolar
        # as the headline for plotting; the actual init used is captured
        # in best_pattern_name + amps_init_vec.
        spatial = spatial_bipolar
        if verbose:
            patt_str = "  ".join(f"{c:+.2f}" for c in spatial)
            print(f"{label} Spatial contrast [{spatial.min():+.3f},"
                  f"{spatial.max():+.3f}] mean={spatial.mean():+.3f}: "
                  f"[{patt_str}]", flush=True)

        (amps_init_vec, best_mag, best_score, _probe_acts,
         probe_hist, best_pattern_name) = _magnitude_probe(
            spatial_patterns, PROBE_MAGS_MA, seed_in["target_mask"],
            seed_in["fs_batch"], seed_in["s0_batch"],
            jnp.asarray(seed_in["Ve_unit"], dtype=jnp.float64),
            jnp.asarray(seed_in["pulse_mask"], dtype=jnp.float64),
            jnp.asarray(seed_in["node_indices"], dtype=jnp.int32),
            DT, label=label, verbose=verbose,
        )

        if verbose:
            init_str = "  ".join(f"{a:+.2f}" for a in amps_init_vec)
            print(f"{label} Rect Adam-FD ({n_iters} iters) from smart "
                  f"init  [{init_str}] mA ...", flush=True)

        # Hard freeze on contacts the probe winner left at exactly zero.
        # Without this the Adam-FD step drifts the zeros by ~lr=0.005 mA
        # per iter in a coherent direction (since the FD gradient sees
        # tiny apparent improvements from adding global drive) -- after
        # ~5 iters the accumulated unbalanced current floods every nt
        # fibre and the probe winner's selectivity is destroyed.  L1
        # prox at default lambda is too weak to pin (would need
        # lambda~3 to overcome the Adam step).  The hard freeze is
        # cleaner and matches the user's intent: the probe winner
        # SELECTED a sparse pattern; the optimiser should polish the
        # non-zero amps without re-activating the zeros.
        probe_freeze_mask = np.abs(amps_init_vec) < 1e-9
        adam_res = run_rect_optimization(
            fiber_statics_batch=seed_in["fs_batch"],
            state0_batch=seed_in["s0_batch"],
            Ve_unit=seed_in["Ve_unit"],
            pulse_mask=seed_in["pulse_mask"],
            node_indices=seed_in["node_indices"],
            target_mask=seed_in["target_mask"],
            weights=seed_in["weights"],
            dt=DT, n_steps=n_iters,
            amps_init_vector=amps_init_vec,
            amp_clip=AMP_CLIP,
            lr=ADAM_LR_MA, fd_eps=FD_EPS_MA,
            freeze_zero_mask=probe_freeze_mask,
            verbose=verbose,
        )
        loss_hist = np.asarray(adam_res["history"]["loss"])
        best_iter = int(np.argmin(loss_hist))
        best_amps = np.asarray(adam_res["history"]["amps"][best_iter])
        best_acts = np.asarray(adam_res["history"]["acts"][best_iter])
        rect_res = {
            "optimizer":         "Adam-FD-smart",
            "amps":              best_amps,
            "loss_history":      loss_hist,
            "final_loss":        float(loss_hist[best_iter]),
            "final_acts":        best_acts,
            "best_iter":         best_iter,
            "best_restart":      0,
            "restart_mags":      [best_mag],
            "n_restarts":        1,
            "n_steps":           n_iters,
            "smart_init": {
                "enabled":           True,
                "best_pattern_name": best_pattern_name,
                "best_mag_mA":       best_mag,
                "best_score":        best_score,
                "n_frozen_contacts": int(np.sum(probe_freeze_mask)),
                "spatial_patterns":  {
                    name: pat.tolist()
                    for name, pat in spatial_patterns.items()
                },
                "amps_init_vector":  amps_init_vec.tolist(),
                "probe_history":     probe_hist,
            },
            "all_final_losses": np.asarray([float(loss_hist[-1])]),
        }
        if verbose:
            print(f"{label} Smart-init Adam-FD done: best_loss="
                  f"{float(loss_hist[best_iter]):.4f} @ iter {best_iter}",
                  flush=True)
    elif RECT_OPTIMIZER == "adam_fd":
        n_iters = max(N_OPT_RECT * 3, 30)
        mags = ADAM_FD_RESTART_MAGS
        if verbose:
            mag_str = ", ".join(f"{m:+.3f}" for m in mags)
            print(
                f"{label} Rect Adam-FD multi-start: {len(mags)} restarts "
                f"× {n_iters} iters, init magnitudes (mA) = [{mag_str}]",
                flush=True,
            )

        best_restart = -1
        best_overall_loss = float("inf")
        best_adam_res = None
        all_final_losses: list[float] = []
        for r_idx, init_mag in enumerate(mags):
            if verbose:
                print(f"{label} --- restart {r_idx + 1}/{len(mags)}  "
                      f"init={init_mag:+.3f} mA ---", flush=True)
            adam_res = run_rect_optimization(
                fiber_statics_batch=seed_in["fs_batch"],
                state0_batch=seed_in["s0_batch"],
                Ve_unit=seed_in["Ve_unit"],
                pulse_mask=seed_in["pulse_mask"],
                node_indices=seed_in["node_indices"],
                target_mask=seed_in["target_mask"],
                weights=seed_in["weights"],
                dt=DT, n_steps=n_iters,
                amp_init_mA=init_mag, amp_clip=AMP_CLIP,
                lr=ADAM_LR_MA, fd_eps=FD_EPS_MA,
                verbose=verbose,
            )
            run_loss_hist = np.asarray(adam_res["history"]["loss"])
            run_best = float(np.min(run_loss_hist))
            all_final_losses.append(run_best)
            if verbose:
                print(f"{label} --- restart {r_idx + 1} done: "
                      f"best_loss={run_best:.4f} "
                      f"(global best so far {min(run_best, best_overall_loss):.4f})",
                      flush=True)
            if run_best < best_overall_loss:
                best_overall_loss = run_best
                best_restart = r_idx
                best_adam_res = adam_res

        # Extract best-iter from the winning restart.  Adam-FD can
        # overshoot a good warm-start (mid-trajectory loss 0.17, last-iter
        # loss 0.55 was observed on the Duke smoketest); we return the
        # BEST-loss iter's amps + acts, not the last iter.
        loss_hist = np.asarray(best_adam_res["history"]["loss"])
        best_iter = int(np.argmin(loss_hist))
        best_amps = np.asarray(best_adam_res["history"]["amps"][best_iter])
        best_acts = np.asarray(best_adam_res["history"]["acts"][best_iter])
        rect_res = {
            "optimizer":     ("Adam-FD-multistart" if len(mags) > 1
                              else "Adam-FD"),
            "amps":          best_amps,
            "loss_history":  loss_hist,
            "final_loss":    float(loss_hist[best_iter]),
            "final_acts":    best_acts,
            "best_iter":     best_iter,
            "best_restart":  best_restart,
            "restart_mags":  list(mags),
            "n_restarts":    len(mags),
            "n_steps":       n_iters,
            "all_final_losses": np.asarray(all_final_losses),
        }
        if verbose:
            print(f"{label} Rect Adam-FD multi-start winner: "
                  f"restart {best_restart + 1}/{len(mags)} "
                  f"(init={mags[best_restart]:+.3f} mA, "
                  f"best_loss={loss_hist[best_iter]:.4f} @ iter {best_iter})",
                  flush=True)
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
            weights=seed_in["weights"],
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
    rect_fire = _firing_summary(rect_res["final_acts"], seed_in["target_mask"])
    rect_res["firing"] = rect_fire
    print(f"{label} Rect done: SI {seed_in['si_baseline']:+.3f} → "
          f"{si_rect:+.3f}  ({rect_t:.0f}s) | {_fmt_firing(rect_fire)}",
          flush=True)

    # ── L1-discovery (parallel validation path) ──────────────────────────
    # Independent Adam-FD run with random init + L1 sparsity reg.  Saved
    # in JSON under rect_l1; main rect path above is unchanged.  This
    # demonstrates the optimiser can discover sparse configurations
    # WITHOUT relying on the hand-coded bipolar/tripolar probe init --
    # validating that our probe init is a speedup, not a shortcut.
    rect_l1_res = None
    if L1_DISCOVERY_ENABLED:
        rect_l1_res = _run_l1_discovery(
            seed_in, n_iters=L1_N_ITERS, l1_lambda=L1_LAMBDA,
            init_scale=L1_INIT_SCALE, init_seed=L1_INIT_SEED + seed_in["seed"],
            verbose=verbose,
        )
        rect_l1_fire = _firing_summary(
            rect_l1_res["final_acts"], seed_in["target_mask"],
        )
        rect_l1_res["firing"] = rect_l1_fire
        si_rect_l1 = selectivity_index(
            rect_l1_res["final_acts"], seed_in["target_mask"]
        )
        print(f"{label} L1-discovery summary: SI={si_rect_l1:+.3f} | "
              f"{_fmt_firing(rect_l1_fire)} | "
              f"{rect_l1_res['n_active_contacts']}/"
              f"{seed_in['Ve_unit'].shape[0]} contacts active",
              flush=True)

    # ── Waveform ──────────────────────────────────────────────────────────
    if N_OPT_WAVE <= 0:
        wave_res = {
            "best_si":   float(si_rect),
            "best_iter": 0,
            "best_acts": np.asarray(rect_res["final_acts"]),
            "history":   {"loss": [], "si": [], "acts": [np.asarray(rect_res["final_acts"])]},
            "firing":    rect_fire,
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
            weights=seed_in["weights"],
            dt=DT, T=seed_in["N_STEPS"], n_steps=N_OPT_WAVE, u_init=u_init,
            lr=WAVE_LR, early_stop_patience=WAVE_PATIENCE, u_clip=AMP_CLIP,
            verbose=verbose,
        )
        wave_t = time.time() - t0
        si_wave = float(wave_res["best_si"])
        wave_fire = _firing_summary(wave_res["best_acts"],
                                    seed_in["target_mask"])
        wave_res["firing"] = wave_fire
        print(f"{label} Wave done: SI {si_rect:+.3f} → {si_wave:+.3f} "
              f"(best @ iter {wave_res['best_iter']})  ({wave_t:.0f}s) | "
              f"{_fmt_firing(wave_fire)}", flush=True)

    return _package_result(seed_in, rect_res, rect_t, wave_res, wave_t,
                            rect_l1_res=rect_l1_res)


def _package_result(seed_in: dict, rect_res: dict, rect_t: float,
                    wave_res: dict, wave_t: float,
                    rect_l1_res: dict | None = None) -> dict:
    target_mask = seed_in["target_mask"]
    # Signed SI as the optimiser reports it.  A negative value means the
    # optimiser found a strongly anti-selective config — i.e. it can fire
    # the *non-target* group selectively, which by the symmetry of the
    # divider labelling is equivalent to a positive-SI solution with the
    # target/non-target labels swapped.  We record both:
    #
    #   final_si        — signed (kept for backward compatibility)
    #   achievable_si   — |final_si|  (the magnitude the optimiser found)
    #   target_flipped  — True iff signed SI was negative (i.e. would
    #                     require swapping target↔non-target to land
    #                     positive)
    si_rect_signed = float(selectivity_index(rect_res["final_acts"], target_mask))
    si_wave_signed = float(wave_res["best_si"])
    nerve = seed_in["nerve"]
    # L1-discovery snapshot (None if L1_DISCOVERY_ENABLED=false).
    rect_l1_snapshot = None
    if rect_l1_res is not None:
        si_l1_signed = float(selectivity_index(
            rect_l1_res["final_acts"], target_mask,
        ))
        rect_l1_snapshot = {
            "optimizer":         rect_l1_res["optimizer"],
            "n_steps":           rect_l1_res["n_steps"],
            "l1_lambda":         rect_l1_res["l1_lambda"],
            "init_scale_mA":     rect_l1_res["init_scale_mA"],
            "init_seed":         rect_l1_res["init_seed"],
            "amps_init_vector":  rect_l1_res["amps_init_vector"],
            "final_loss":        rect_l1_res["final_loss"],
            "final_si":          si_l1_signed,
            "achievable_si":     abs(si_l1_signed),
            "target_flipped":    si_l1_signed < 0,
            "firing":            rect_l1_res.get("firing", {}),
            "n_active_contacts": rect_l1_res["n_active_contacts"],
            "best_iter":         rect_l1_res["best_iter"],
            "loss_history":      rect_l1_res["loss_history"].tolist(),
            "amps_mA":           rect_l1_res["amps"].tolist(),
            "final_acts":        rect_l1_res["final_acts"].tolist(),
            "time_s":            rect_l1_res["wall_s"],
        }
    cluster_info = seed_in.get("cluster_info")
    # Sanitised cluster snapshot (centroid tuple → list for JSON).
    cluster_snapshot = None
    if cluster_info is not None:
        cluster_snapshot = {
            "target_ids":            list(cluster_info["target_ids"]),
            "peripheral_ids":        list(cluster_info["peripheral_ids"]),
            "window_start_deg":      float(cluster_info["window_start_deg"]),
            "window_end_deg":        float(cluster_info["window_end_deg"]),
            "nerve_centroid_xy_um":  list(cluster_info["nerve_centroid_xy_um"]),
            "r_threshold_um":        float(cluster_info["r_threshold_um"]),
            "n_target_fibers":       int(cluster_info["n_target_fibers"]),
            "radius_quantile":       float(CLUSTER_RADIUS_QUANTILE),
            "angular_window_deg":    float(CLUSTER_WINDOW_DEG),
        }
    return {
        "sample":   SAMPLE_NAME,
        "seed":     seed_in["seed"],
        "target_paradigm": TARGET_PARADIGM,
        "cluster":         cluster_snapshot,
        "divider_deg":     seed_in["divider_deg"],
        "pulse_shape":     PULSE_SHAPE,
        "pulse_pw_ms":     float(PW_MS),
        "pulse_asym_ratio": float(ASYM_RATIO) if PULSE_SHAPE == "biphasic_asym" else None,
        "si_baseline":     float(seed_in["si_baseline"]),
        "rect": {
            "optimizer":   rect_res["optimizer"],
            "n_restarts":  rect_res["n_restarts"],
            "n_steps":     rect_res["n_steps"],
            "best_restart": rect_res["best_restart"],
            "restart_mags": rect_res.get("restart_mags", []),
            "final_loss":  rect_res["final_loss"],
            "final_si":      si_rect_signed,
            "achievable_si": abs(si_rect_signed),
            "target_flipped": si_rect_signed < 0,
            "firing":      rect_res.get("firing", {}),
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
            "final_si":      si_wave_signed,
            "achievable_si": abs(si_wave_signed),
            "target_flipped": si_wave_signed < 0,
            "firing":     wave_res.get("firing", {}),
            "best_iter":  int(wave_res["best_iter"]),
            "last_si":    (float(wave_res["history"]["si"][-1])
                           if wave_res["history"]["si"] else si_wave_signed),
            "n_iters_run": len(wave_res["history"]["loss"]),
            "loss_history": wave_res["history"]["loss"],
            "si_history":  wave_res["history"]["si"],
            "final_acts":  np.asarray(wave_res["best_acts"]).tolist(),
            "time_s":      wave_t,
        },
        "rect_l1": rect_l1_snapshot,
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
    print(f"[duke sweep] target_paradigm={TARGET_PARADIGM}  "
          f"pulse_shape={PULSE_SHAPE}  PW={PW_MS:.3f} ms", flush=True)

    # Cluster-paradigm pre-flight: skip the whole sample cleanly if it
    # has fewer than CLUSTER_MIN_FASCICLES fascicles -- the cluster
    # selector is degenerate there and we don't want to burn 90 minutes
    # of GPU time on something we'd throw out anyway.
    if TARGET_PARADIGM == "cluster":
        n_fasc = len(duke["fasc_meta"])
        if n_fasc < CLUSTER_MIN_FASCICLES:
            print(f"[duke sweep] SKIP {SAMPLE_NAME}: n_fasc={n_fasc} < "
                  f"CLUSTER_MIN_FASCICLES={CLUSTER_MIN_FASCICLES}",
                  flush=True)
            return

    seeds = list(range(SEED_START, SEED_END))
    for s in seeds:
        out_path = OUT / f"data_seed_{s:04d}.json"
        if out_path.exists():
            print(f"[seed {s}] already exists at {out_path}; skipping",
                  flush=True)
            continue
        seed_in = _build_seed(duke, s, verbose=True)
        if seed_in is None:
            # _build_seed returns None when the cluster selector finds
            # no acceptable target; nothing to optimise.  Skip cleanly.
            print(f"[seed {s}] no target -> no JSON written", flush=True)
            continue
        result = _run_one_seed(seed_in, verbose=True)
        save_json(result, out_path)
        print(f"[seed {s}] -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
