#!/bin/bash
#SBATCH -J jaxfibers-mrg-val
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

# MRG validation: 9 diameters × 8 pulse shapes × 6 PWs + traces + CV.
# Outputs: outputs/mrg_validation/
#
# Submit from the project root:
#   cd /path/to/jaxley_fibers
#   sbatch slurm/run_mrg_validation.sbatch

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
# Resolve the container image once at submit time:
#   1. Honour an explicit CONTAINER_IMAGE env-var override.
#   2. Else prefer the project SquashFS at $HOME/containers/jaxfibers.sqsh
#      (built once via slurm/build_container.sh; has pip deps baked in,
#      so setup_env.sh skips its pip install step).
#   3. Else fall back to nvcr.io#nvidia/pytorch:25.03-py3 (Pyxis pulls +
#      caches it on the assigned node; setup_env.sh installs deps).
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
echo '[job] Running experiments_v2/mrg_validation.py ...'
python experiments_v2/mrg_validation.py
echo '[job] Done. Outputs in outputs/mrg_validation/'
"
