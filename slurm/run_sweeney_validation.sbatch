#!/bin/bash
#SBATCH -J jaxfibers-sweeney-val
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -q a16
#SBATCH --gres=gpu:a16:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Sweeney myelinated-fiber validation: 6 diameters × 8 pulse shapes × 6 PWs + traces + CV.
# Outputs: outputs/sweeney_validation/
#
# Submit from the project root:
#   cd /path/to/jaxley_fibers
#   sbatch slurm/run_sweeney_validation.sbatch

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
# Resolve the container image once at submit time:
#   1. Honour an explicit CONTAINER_IMAGE env-var override.
#   2. Else prefer a user-local SquashFS cache at
#      $HOME/containers/pytorch_25.03.sqsh if present (gives sub-second
#      Pyxis startup vs ~5-10 min for a fresh nvcr.io pull).
#   3. Else fall back to the nvcr.io reference (Pyxis pulls + caches
#      it on the assigned node; this still works on a fresh clone).
DEFAULT_SQSH="${HOME}/containers/pytorch_25.03.sqsh"
DEFAULT_IMAGE="nvcr.io#nvidia/pytorch:25.03-py3"
if [ -z "${CONTAINER_IMAGE:-}" ]; then
  if [ -f "${DEFAULT_SQSH}" ]; then
    CONTAINER_IMAGE="${DEFAULT_SQSH}"
  else
    CONTAINER_IMAGE="${DEFAULT_IMAGE}"
  fi
fi

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/outputs"

echo "Project root : ${PROJECT_ROOT}"
echo "Container    : ${CONTAINER_IMAGE}"

# Pyxis container pulls can exceed SLURM's 32-second default step-launch
# timeout on a cold node cache (the nvcr.io PyTorch image is ~10 GB).
# Allow up to 10 minutes for srun to bring the container up.
export SLURM_STEP_LAUNCH_TIMEOUT=600

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc "
set -euo pipefail
export PROJECT_ROOT=\"${PROJECT_ROOT}\"
export JAX_ENABLE_X64=1
bash \"${PROJECT_ROOT}/slurm/setup_env.sh\"
echo '[job] Running experiments_v2/sweeney_validation.py ...'
python -u experiments_v2/sweeney_validation.py
echo '[job] Done. Outputs in outputs/sweeney_validation/'
"
