#!/bin/bash
#SBATCH -J jaxfibers-duke-sweep
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -q a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 23:59:00
#SBATCH --array=0-2
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

# GPU partition default: A100.  The scaling benchmark + the
# manuscript headline numbers are reported on A100 (see Section 3.3),
# so the selectivity sweep should match.  Override on submission with
# `sbatch --qos=a16 --gres=gpu:a16:1 ...` if A100 is saturated and
# you need to fall back.  a16 partition was the original default but
# is ~3-5x slower per Adam-FD iter on the 1000-fibre Duke geometries.

# SLURM array: by default the sweep is split into 3 parallel shards
# (--array=0-2) so each shard processes ~1/3 of the Duke sample
# folders.  Override with `sbatch --array=0-N ...` for finer / coarser
# splits, or `--array=0` to fall back to a single-job sequential run.
# Shard assignment is interleaved (sample i goes to shard i % N) so
# the wall-time stays balanced even if individual samples differ in
# fibre count.  All shards write into the SAME outputs/duke_sweeps/
# tree, so there are no per-shard output files to merge afterwards.

# Duke FEM nerve selectivity sweep.
#
# Iterates over every Duke FEM bundle in DUKE_VES_ROOT (default:
# duke_Ves/) and runs the same SEED_START..SEED_END random-divider
# selectivity sweep that the synthetic phase-3 sbatch runs, but with
# the geometry + Ve_unit loaded from the FEM bundle via
# experiments_v2/selectivity_sweep_duke.py.  Each sample's outputs
# land in outputs/selectivity_sweep_duke_<sub>_<sam>/, parallel to
# the synthetic outputs/selectivity_sweep_phase3_*/ directories, so
# the figure code can find them.
#
# Memory: 997 fibres × 12 contacts × 221 comp × 600 t-steps is ~5×
# the synthetic per-seed footprint.  --mem=96G + a100 (40 GB VRAM)
# fits comfortably under both Adam-FD and LBFGS.  On the a16 fallback
# (12 GB VRAM) Adam-FD still fits but LBFGS will OOM.
#
# Resume-friendly: per-seed JSONs are written incrementally; re-submit
# after walltime expiry picks up the missing seeds.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
DEFAULT_SQSH="${HOME}/containers/jaxfibers.sqsh"
DEFAULT_IMAGE="nvcr.io#nvidia/pytorch:25.03-py3"
if [ -z "${CONTAINER_IMAGE:-}" ]; then
  if [ -f "${DEFAULT_SQSH}" ]; then
    CONTAINER_IMAGE="${DEFAULT_SQSH}"
  else
    CONTAINER_IMAGE="${DEFAULT_IMAGE}"
  fi
fi

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/outputs"

export SKIP_PYFIBERS_COMPILE=1
export SLURM_STEP_LAUNCH_TIMEOUT=600

# Sweep knobs (env-overrideable on sbatch submission).
#
# SEED_END=1 (default): 1 random divider angle per nerve.  The
# manuscript story across the 38 swine + human Duke samples is the
# cross-anatomy spread, not the per-anatomy robustness; 25 seeds ×
# 38 nerves would be ~950 runs and dilute the per-nerve signal.  Set
# SEED_END=25 if you really want the 25-seed per-anatomy distribution.
export SEED_START="${SEED_START:-0}"
export SEED_END="${SEED_END:-1}"
export FIBER_DIAMETER_UM="${FIBER_DIAMETER_UM:-5.7}"
export N_NODES="${N_NODES:-21}"
export RECT_OPTIMIZER="${RECT_OPTIMIZER:-adam_fd}"
export N_OPT_RECT="${N_OPT_RECT:-30}"
export N_OPT_WAVE="${N_OPT_WAVE:-100}"
export WAVE_LR="${WAVE_LR:-5e-4}"
export WAVE_PATIENCE="${WAVE_PATIENCE:-20}"
# Duke-specific amp scaling.  FEM Ve peaks at ~870 mV/mA on this
# bundle (~5× the synthetic point-source); single-contact thresholds
# are correspondingly ~5× smaller.  Defaults below were re-tuned after
# the first full-cohort run: the smoketest-tuned ±0.30 mA / 0.010 mA
# combination put many nerves (e.g. 2-fascicle imbalanced anatomies)
# either entirely sub-threshold or entirely supra-threshold at init,
# in which case the ±0.010 mA FD probe was too small to flip any fibre
# across threshold and the FD gradient collapsed to exactly zero —
# Adam-FD then made no progress for 20+ iters before the early-stop
# fired (see logs from initial run on human_sub-46_sam-2 etc.).  The
# new defaults put the init amplitude near per-contact threshold for
# the median Duke anatomy and widen the FD probe enough to register
# firing-state flips:
#
#   AMP_INIT_MA: -0.30 → -0.08 (~4× smaller; near-threshold init)
#   FD_EPS_MA  :  0.010 →  0.030 (3× larger; enough to flip near-threshold fibres)
#   ADAM_LR_MA :  0.015 → 0.005 (smaller; avoids overshoot back into saturation)
#
# All env-overrideable on submission for further tuning.
export AMP_INIT_MA="${AMP_INIT_MA:--0.08}"
export AMP_CLIP_LO="${AMP_CLIP_LO:--1.50}"
export AMP_CLIP_HI="${AMP_CLIP_HI:-1.50}"
export ADAM_LR_MA="${ADAM_LR_MA:-0.005}"
export FD_EPS_MA="${FD_EPS_MA:-0.030}"

# Adam-FD multi-start over init magnitude.  A single fixed init cannot
# put every Duke anatomy in the gradient-informative threshold regime
# (sub-46 / sub-47 stalled at SI=0 because the 953/47 fascicle split
# left every fibre either fully sub- or fully supra-threshold at init,
# so the FD gradient was exactly zero).  Grid retuned after the first
# multi-start run showed -0.05 was below threshold on every Duke
# anatomy (wasted restart) and -0.40 was still insufficient on poorly-
# coupled fascicles like sub-46 where even -0.30 mA gave 0% firing.
# New default spans 15x:
#   -0.10 mA  for well-coupled anatomies (e.g. sub-50, fires here)
#   -0.30 mA  middle ground, the old single-start default
#   -0.80 mA  for poorly-coupled target fascicles
#   -1.50 mA  the amp_clip limit, last-resort for stuck geometries
# Comma-separated, signed mA.  Set to a single value (e.g. "-0.30") to
# revert to single-start Adam-FD.  Cost: rect time scales linearly in
# the number of restarts; with the default 4 entries each sample takes
# ~4x the rect wall-time of a single-start run.
export ADAM_FD_RESTART_MAGS="${ADAM_FD_RESTART_MAGS:--0.10,-0.30,-0.80,-1.50}"

# Smart init: instead of running 4 full Adam-FD restarts at different
# magnitudes (~50 min/sample), compute a per-contact Ve-contrast pattern
# from the target/off-target geometry, probe magnitudes with single
# forwards (~3 min total), and run ONE Adam-FD from that informed
# starting point.  Roughly 3x faster than multi-restart, with a
# spatial pattern that is already target-aligned at iter 0.  Set
# SMART_INIT_ENABLED=false to fall back to the multi-restart path.
export SMART_INIT_ENABLED="${SMART_INIT_ENABLED:-true}"
export PROBE_MAGS_MA="${PROBE_MAGS_MA:--0.02,-0.05,-0.10,-0.20,-0.40,-0.80}"

# Loss / activation-proxy regularisation read by jaxfibers.optim.optimizer
# at import time (defaults preserve legacy 0 / 0 behaviour if unset).
#
# JAXLEY_FIBERS_SOFT_TEMPERATURE > 0 switches the activation proxy from
# binary-ish min(m_first, m_last) to a sigmoid surrogate that is
# gradient-informative even when no fibre has crossed binary threshold.
# 0.1 gives the FD optimiser a smooth landscape on Duke nerves where
# the all-sub-threshold init regime would otherwise zero the gradient.
export JAXLEY_FIBERS_SOFT_TEMPERATURE="${JAXLEY_FIBERS_SOFT_TEMPERATURE:-0.1}"
# JAXLEY_FIBERS_ENERGY_LAMBDA > 0 adds an amplitude-energy penalty to
# wq_loss.  A small value (1e-3 mA²) discourages the all-fire trivial
# minimum without preventing the optimiser from exploring high-amp
# solutions when they are necessary.
export JAXLEY_FIBERS_ENERGY_LAMBDA="${JAXLEY_FIBERS_ENERGY_LAMBDA:-1e-3}"

# Pulse waveform: clinical implants are charge-balanced biphasic
# (any net DC charge corrodes electrodes and damages tissue), so the
# Duke sweep defaults to symmetric biphasic.  Options:
#   monophasic, biphasic_sym (default), biphasic_asym
export PULSE_SHAPE="${PULSE_SHAPE:-biphasic_sym}"
export PW_MS="${PW_MS:-1.0}"           # cathodic pulse width
export ASYM_RATIO="${ASYM_RATIO:-4.0}" # anodic/cathodic duration ratio (asym only)

# Target-fascicle selection: cluster paradigm replaces the legacy
# random divider-through-centroid.  See duke_loader and
# duke_target_sanity for the geometry rationale.  Override
# TARGET_PARADIGM=divider to reproduce pre-cluster behaviour.
export TARGET_PARADIGM="${TARGET_PARADIGM:-cluster}"
export CLUSTER_RADIUS_QUANTILE="${CLUSTER_RADIUS_QUANTILE:-0.6}"
export CLUSTER_WINDOW_DEG="${CLUSTER_WINDOW_DEG:-90.0}"
export CLUSTER_N_MIN_TARGET="${CLUSTER_N_MIN_TARGET:-50}"
export CLUSTER_MIN_FASCICLES="${CLUSTER_MIN_FASCICLES:-3}"

# Where the Duke FEM bundles live.  Default = duke_meshes/ (the
# 18-swine + 20-human bundle from the FEM pipeline).  Earlier bundles
# lived under duke_Ves/ and either name is accepted on disk; override
# DUKE_VES_ROOT on resubmit to point at a different staging directory.
if [ -d "${PROJECT_ROOT}/duke_meshes" ]; then
  DUKE_VES_ROOT="${DUKE_VES_ROOT:-${PROJECT_ROOT}/duke_meshes}"
else
  DUKE_VES_ROOT="${DUKE_VES_ROOT:-${PROJECT_ROOT}/duke_Ves}"
fi

# Shard bookkeeping for SLURM array jobs (--array=0-N).  When the
# sbatch is run as an array, each task processes only the samples
# whose interleaved index matches the task id; outside an array we
# fall back to single-shard (all samples).
N_SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
SHARD_IDX="${SLURM_ARRAY_TASK_ID:-0}"

echo "Duke sweep: seeds ${SEED_START}..$((SEED_END-1))  D=${FIBER_DIAMETER_UM} µm"
echo "DUKE_VES_ROOT=${DUKE_VES_ROOT}"
echo "Shard ${SHARD_IDX} of ${N_SHARDS}"
echo "Optimiser: ${RECT_OPTIMIZER}  (init=${AMP_INIT_MA}, clip=[${AMP_CLIP_LO}, ${AMP_CLIP_HI}], lr=${ADAM_LR_MA}, fd_eps=${FD_EPS_MA})"
echo "Samples discovered:"
ls -d "${DUKE_VES_ROOT}"/*/ 2>/dev/null | xargs -n1 basename | sed "s/^/  /"

# Iterate over every sample directory under DUKE_VES_ROOT.  A sample
# directory is recognised by the presence of paths_Ve.npz + nerve_xsec.json.
# In array mode, each shard handles samples i where i mod N_SHARDS ==
# SHARD_IDX (interleaved to balance per-sample wall time).
SAMPLE_INDEX=-1
for SAMPLE_DIR in "${DUKE_VES_ROOT}"/*/; do
  if [ ! -f "${SAMPLE_DIR}/paths_Ve.npz" ] || [ ! -f "${SAMPLE_DIR}/nerve_xsec.json" ]; then
    echo "[skip] ${SAMPLE_DIR} — missing paths_Ve.npz or nerve_xsec.json"
    continue
  fi
  SAMPLE_INDEX=$((SAMPLE_INDEX+1))
  if [ $((SAMPLE_INDEX % N_SHARDS)) -ne ${SHARD_IDX} ]; then
    continue
  fi
  SAMPLE_NAME=$(basename "${SAMPLE_DIR%/}")
  echo
  echo "============================================================"
  echo "Sample: ${SAMPLE_NAME}  (index ${SAMPLE_INDEX}, shard ${SHARD_IDX}/${N_SHARDS})"
  echo "============================================================"
  # Per-sample failures (e.g. a corrupt nerve_xsec.json) must NOT kill the
  # remaining samples in this shard — wrap srun in an if-guard so set -e
  # doesn't propagate.
  if srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
    --container-workdir="${PROJECT_ROOT}" \
    bash -lc "
set -euo pipefail
export PROJECT_ROOT=\"${PROJECT_ROOT}\"
export JAX_ENABLE_X64=1
export SKIP_PYFIBERS_COMPILE=1
export DUKE_SAMPLE_DIR=\"${SAMPLE_DIR%/}\"
export SEED_START=\"${SEED_START}\"
export SEED_END=\"${SEED_END}\"
export FIBER_DIAMETER_UM=\"${FIBER_DIAMETER_UM}\"
export N_NODES=\"${N_NODES}\"
export RECT_OPTIMIZER=\"${RECT_OPTIMIZER}\"
export N_OPT_RECT=\"${N_OPT_RECT}\" N_OPT_WAVE=\"${N_OPT_WAVE}\"
export WAVE_LR=\"${WAVE_LR}\" WAVE_PATIENCE=\"${WAVE_PATIENCE}\"
export AMP_INIT_MA=\"${AMP_INIT_MA}\"
export AMP_CLIP_LO=\"${AMP_CLIP_LO}\" AMP_CLIP_HI=\"${AMP_CLIP_HI}\"
export ADAM_LR_MA=\"${ADAM_LR_MA}\" FD_EPS_MA=\"${FD_EPS_MA}\"
export ADAM_FD_RESTART_MAGS=\"${ADAM_FD_RESTART_MAGS}\"
export SMART_INIT_ENABLED=\"${SMART_INIT_ENABLED}\"
export PROBE_MAGS_MA=\"${PROBE_MAGS_MA}\"
export JAXLEY_FIBERS_SOFT_TEMPERATURE=\"${JAXLEY_FIBERS_SOFT_TEMPERATURE}\"
export JAXLEY_FIBERS_ENERGY_LAMBDA=\"${JAXLEY_FIBERS_ENERGY_LAMBDA}\"
export PULSE_SHAPE=\"${PULSE_SHAPE}\"
export PW_MS=\"${PW_MS}\"
export ASYM_RATIO=\"${ASYM_RATIO}\"
export T_STOP=\"${T_STOP:-3.0}\"
export DELAY_MS=\"${DELAY_MS:-1.0}\"
export DT=\"${DT:-0.005}\"
export TARGET_PARADIGM=\"${TARGET_PARADIGM}\"
export CLUSTER_RADIUS_QUANTILE=\"${CLUSTER_RADIUS_QUANTILE}\"
export CLUSTER_WINDOW_DEG=\"${CLUSTER_WINDOW_DEG}\"
export CLUSTER_N_MIN_TARGET=\"${CLUSTER_N_MIN_TARGET}\"
export CLUSTER_MIN_FASCICLES=\"${CLUSTER_MIN_FASCICLES}\"
bash \"${PROJECT_ROOT}/slurm/setup_env.sh\"
python -u experiments_v2/selectivity_sweep_duke.py
"; then
    echo "[shard ${SHARD_IDX}] ${SAMPLE_NAME}: OK"
  else
    rc=$?
    echo "[shard ${SHARD_IDX}] ${SAMPLE_NAME}: FAILED rc=${rc} — continuing to next sample"
  fi
done

echo
echo "All Duke samples processed."
