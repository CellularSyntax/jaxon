#!/bin/bash
#SBATCH -J jaxfibers-zerosmoke
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -q a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Standalone cold-start (0 mA) GD smoke test on ONE downsampled nerve.
# Prints the per-iter SI trajectory so we can see whether GD escapes SI=0.
#
#   SMOKE_SAMPLE=human_sub-47_sam-2 MAX_FIBERS=50 JAXLEY_FIBERS_LOSS=quotient \
#     sbatch slurm/run_smoke_zero_start.sbatch
#
# Knobs (env): JAXLEY_FIBERS_LOSS, ZERO_LR_MA, ZERO_STEPS, ZERO_LR_DECAY,
#              ZERO_SI_FLOOR, ZERO_PATIENCE, MAX_FIBERS, CLUSTER_N_MIN_TARGET.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
DEFAULT_SQSH="${HOME}/containers/jaxfibers.sqsh"
DEFAULT_IMAGE="nvcr.io#nvidia/pytorch:25.03-py3"
if [ -z "${CONTAINER_IMAGE:-}" ]; then
  if [ -f "${DEFAULT_SQSH}" ]; then CONTAINER_IMAGE="${DEFAULT_SQSH}";
  else CONTAINER_IMAGE="${DEFAULT_IMAGE}"; fi
fi
mkdir -p "${PROJECT_ROOT}/logs"

if [ -d "${PROJECT_ROOT}/duke_meshes" ]; then
  DUKE_VES_ROOT="${DUKE_VES_ROOT:-${PROJECT_ROOT}/duke_meshes}"
else
  DUKE_VES_ROOT="${DUKE_VES_ROOT:-${PROJECT_ROOT}/duke_Ves}"
fi

SAMPLE="${SMOKE_SAMPLE:-human_sub-47_sam-2}"
SAMPLE_DIR="${DUKE_VES_ROOT}/${SAMPLE}"
echo "Zero-start smoke: ${SAMPLE}  MAX_FIBERS=${MAX_FIBERS:-50}  LOSS=${JAXLEY_FIBERS_LOSS:-quotient}"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc "
set -euo pipefail
export PROJECT_ROOT=\"${PROJECT_ROOT}\"
export JAX_ENABLE_X64=1
export SKIP_PYFIBERS_COMPILE=1
export DUKE_SAMPLE_DIR=\"${SAMPLE_DIR}\"
export MAX_FIBERS=\"${MAX_FIBERS:-50}\"
export JAXLEY_FIBERS_LOSS=\"${JAXLEY_FIBERS_LOSS:-quotient}\"
export JAXLEY_FIBERS_SOFT_TEMPERATURE=\"${JAXLEY_FIBERS_SOFT_TEMPERATURE:-0.15}\"
export JAXLEY_FIBERS_ENERGY_LAMBDA=\"${JAXLEY_FIBERS_ENERGY_LAMBDA:-1e-3}\"
export ZERO_LR_MA=\"${ZERO_LR_MA:-0.05}\"
export ZERO_STEPS=\"${ZERO_STEPS:-300}\"
export ZERO_LR_DECAY=\"${ZERO_LR_DECAY:-0.6}\"
export ZERO_SI_FLOOR=\"${ZERO_SI_FLOOR:-0.5}\"
export ZERO_PATIENCE=\"${ZERO_PATIENCE:-15}\"
export ZERO_WD=\"${ZERO_WD:-0.02}\"
export ZERO_JITTER=\"${ZERO_JITTER:-0.01}\"
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
export ADAM_LR_MA=\"${ADAM_LR_MA:-0.005}\"
export FD_EPS_MA=\"${FD_EPS_MA:-0.030}\"
export PROBE_MAGS_MA=\"${PROBE_MAGS_MA:--0.02,-0.05,-0.10,-0.20,-0.40,-0.80,-1.20,-1.50}\"
export TARGET_PARADIGM=\"${TARGET_PARADIGM:-cluster}\"
export CLUSTER_RADIUS_QUANTILE=\"${CLUSTER_RADIUS_QUANTILE:-0.32}\"
export CLUSTER_WINDOW_DEG=\"${CLUSTER_WINDOW_DEG:-90.0}\"
export CLUSTER_N_MIN_TARGET=\"${CLUSTER_N_MIN_TARGET:-8}\"
export CLUSTER_MIN_FASCICLES=\"${CLUSTER_MIN_FASCICLES:-3}\"
export CLUSTER_N_POSITIONS=\"${CLUSTER_N_POSITIONS:-4}\"
export CLUSTER_RNG_SEED=\"${CLUSTER_RNG_SEED:-0}\"
bash \"${PROJECT_ROOT}/slurm/setup_env.sh\"
python -u -m experiments_v2.smoke_zero_start
"
echo "done."
