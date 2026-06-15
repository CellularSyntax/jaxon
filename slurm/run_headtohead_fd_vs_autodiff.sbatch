#!/bin/bash
#SBATCH -J jaxfibers-h2h
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -q a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH --array=0-11%4
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

# Adam-FD vs autodiff-LBFGS head-to-head on the identical amplitude problem,
# across 6 swine + 6 human nerves (one nerve per array task, <=4 concurrent for
# the a100 QOS).  Each task writes outputs/headtohead_fd_vs_autodiff/<nerve>.json.
#
# After all tasks finish, build the table:
#   srun --container-image=... bash -lc \
#     'bash slurm/setup_env.sh; HEADTOHEAD_AGGREGATE=1 \
#      python -m experiments_v2.headtohead_fd_vs_autodiff'
# (or run that aggregate line anywhere the JSONs are visible).
#
# Submit:
#   sbatch slurm/run_headtohead_fd_vs_autodiff.sbatch
#   # full population instead of the 300-fiber default:
#   MAX_FIBERS=0 sbatch slurm/run_headtohead_fd_vs_autodiff.sbatch

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

# Build the nerve list: first 6 swine (no "human_" prefix) + first 6 human, each
# with a valid paths_Ve.npz.  Override with HEADTOHEAD_SAMPLES="name1,name2,...".
if [ -n "${HEADTOHEAD_SAMPLES:-}" ]; then
  IFS=',' read -r -a SAMPLES <<< "${HEADTOHEAD_SAMPLES}"
else
  mapfile -t SWINE < <(for d in "${DUKE_VES_ROOT}"/*/; do
    n=$(basename "${d%/}"); [[ "$n" == human_* ]] && continue
    [ -f "${d}/paths_Ve.npz" ] && echo "$n"; done | sort | head -n 6)
  mapfile -t HUMAN < <(for d in "${DUKE_VES_ROOT}"/*/; do
    n=$(basename "${d%/}"); [[ "$n" != human_* ]] && continue
    [ -f "${d}/paths_Ve.npz" ] && echo "$n"; done | sort | head -n 6)
  SAMPLES=("${SWINE[@]}" "${HUMAN[@]}")
fi

echo "Head-to-head nerve list (${#SAMPLES[@]}):"
for i in "${!SAMPLES[@]}"; do echo "  [$i] ${SAMPLES[$i]}"; done

TASK="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${TASK}" -ge "${#SAMPLES[@]}" ]; then
  echo "Task ${TASK} >= ${#SAMPLES[@]} nerves — nothing to do."; exit 0
fi
NERVE="${SAMPLES[$TASK]}"
SAMPLE_DIR="${DUKE_VES_ROOT}/${NERVE}"
echo "Task ${TASK}: ${NERVE}  (MAX_FIBERS=${MAX_FIBERS:-300})"

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

export RECT_OPTIMIZER=lbfgs
export N_RESTARTS_RECT=1
export HEADTOHEAD_STEPS=\"${HEADTOHEAD_STEPS:-15}\"
export MAX_FIBERS=\"${MAX_FIBERS:-300}\"

# match the cohort _build_seed pipeline
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
python -u -m experiments_v2.headtohead_fd_vs_autodiff
"
echo "Task ${TASK} (${NERVE}) done."
