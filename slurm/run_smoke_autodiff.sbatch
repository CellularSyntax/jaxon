#!/bin/bash
#SBATCH -J jaxfibers-smoke-autodiff
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -q a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# ONE-CASE smoke test for the autodiff / gradient-based optimization path,
# to run BEFORE committing a full cohort sweep.  Verifies, on a single Duke
# nerve/seed, that (A) autodiff gradients are correct vs finite differences,
# (B) the exact-gradient LBFGS amplitude optimizer runs and improves SI, and
# (C) the high-dimensional autodiff WAVEFORM optimizer runs AND fits GPU
# memory at the real nerve scale (the documented risk: backprop through the
# ODE scan is memory-bound and OOMs the a16's 12 GB -- hence a100 here).
#
# Submit:
#   sbatch slurm/run_smoke_autodiff.sbatch
#   # pick a specific nerve / cap fibers / shorten window:
#   DUKE_SAMPLE_DIR=duke_Ves/sub-11_sam-3 MAX_FIBERS=300 T_STOP=1.5 \
#     sbatch slurm/run_smoke_autodiff.sbatch

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
DEFAULT_SQSH="${HOME}/containers/jaxfibers.sqsh"
DEFAULT_IMAGE="nvcr.io#nvidia/pytorch:25.03-py3"
if [ -z "${CONTAINER_IMAGE:-}" ]; then
  if [ -f "${DEFAULT_SQSH}" ]; then CONTAINER_IMAGE="${DEFAULT_SQSH}";
  else CONTAINER_IMAGE="${DEFAULT_IMAGE}"; fi
fi
mkdir -p "${PROJECT_ROOT}/logs"

# Data root (duke_meshes preferred, duke_Ves fallback), then pick a sample.
if [ -d "${PROJECT_ROOT}/duke_meshes" ]; then
  DUKE_VES_ROOT="${DUKE_VES_ROOT:-${PROJECT_ROOT}/duke_meshes}"
else
  DUKE_VES_ROOT="${DUKE_VES_ROOT:-${PROJECT_ROOT}/duke_Ves}"
fi
# Default to a known multi-fascicle swine nerve; override DUKE_SAMPLE_DIR.
DUKE_SAMPLE_DIR="${DUKE_SAMPLE_DIR:-${DUKE_VES_ROOT}/sub-11_sam-3}"
if [ ! -d "${DUKE_SAMPLE_DIR}" ]; then
  DUKE_SAMPLE_DIR="$(ls -d "${DUKE_VES_ROOT}"/*/ 2>/dev/null | head -n1)"
fi

echo "Smoke autodiff on: ${DUKE_SAMPLE_DIR}"
echo "MAX_FIBERS=${MAX_FIBERS:-0(full)}  T_STOP=${T_STOP:-3.0}  DT=${DT:-0.005}"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc "
set -euo pipefail
export PROJECT_ROOT=\"${PROJECT_ROOT}\"
export JAX_ENABLE_X64=1
export SKIP_PYFIBERS_COMPILE=1
export DUKE_SAMPLE_DIR=\"${DUKE_SAMPLE_DIR%/}\"

# Autodiff path + small smoke iteration counts.
export RECT_OPTIMIZER=lbfgs
export N_RESTARTS_RECT=\"${N_RESTARTS_RECT:-2}\"
export SMOKE_RECT_STEPS=\"${SMOKE_RECT_STEPS:-15}\"
export SMOKE_WAVE_ITERS=\"${SMOKE_WAVE_ITERS:-5}\"
export MAX_FIBERS=\"${MAX_FIBERS:-0}\"

# Match the cohort pipeline so _build_seed behaves identically.
export FIBER_DIAMETER_UM=\"${FIBER_DIAMETER_UM:-5.7}\"
export N_NODES=\"${N_NODES:-21}\"
export T_STOP=\"${T_STOP:-3.0}\"
export DT=\"${DT:-0.005}\"
export DELAY_MS=\"${DELAY_MS:-1.0}\"
export PULSE_SHAPE=\"${PULSE_SHAPE:-biphasic_sym}\"
export PW_MS=\"${PW_MS:-1.0}\"
export ASYM_RATIO=\"${ASYM_RATIO:-4.0}\"
export AMP_INIT_MA=\"${AMP_INIT_MA:--0.08}\"
export AMP_CLIP_LO=\"${AMP_CLIP_LO:--2.00}\"
export AMP_CLIP_HI=\"${AMP_CLIP_HI:-2.00}\"
export JAXLEY_FIBERS_SOFT_TEMPERATURE=\"${JAXLEY_FIBERS_SOFT_TEMPERATURE:-0.1}\"
export JAXLEY_FIBERS_ENERGY_LAMBDA=\"${JAXLEY_FIBERS_ENERGY_LAMBDA:-1e-3}\"
export TARGET_PARADIGM=cluster
export CLUSTER_RADIUS_QUANTILE=\"${CLUSTER_RADIUS_QUANTILE:-0.32}\"
export CLUSTER_WINDOW_DEG=\"${CLUSTER_WINDOW_DEG:-90.0}\"
export CLUSTER_N_MIN_TARGET=\"${CLUSTER_N_MIN_TARGET:-50}\"
export CLUSTER_MIN_FASCICLES=\"${CLUSTER_MIN_FASCICLES:-3}\"
export CLUSTER_N_POSITIONS=\"${CLUSTER_N_POSITIONS:-4}\"
export CLUSTER_RNG_SEED=\"${CLUSTER_RNG_SEED:-0}\"

bash \"${PROJECT_ROOT}/slurm/setup_env.sh\"
python -u -m experiments_v2.smoke_autodiff_opt
"
echo "Smoke autodiff done."
