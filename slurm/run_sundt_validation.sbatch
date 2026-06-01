#!/bin/bash
#SBATCH -J jaxfibers-sundt-val
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

# Sundt C-fiber validation: 5 diameters × 8 pulse shapes × 6 PWs + traces + CV.
# Outputs: outputs/sundt_validation/
#
# Submit from the project root:
#   cd /path/to/jaxley_fibers
#   sbatch slurm/run_sundt_validation.sbatch

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:25.03-py3}"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/outputs"

echo "Project root : ${PROJECT_ROOT}"
echo "Container    : ${CONTAINER_IMAGE}"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc "
set -euo pipefail
export PROJECT_ROOT=\"${PROJECT_ROOT}\"
export JAX_ENABLE_X64=1
bash \"${PROJECT_ROOT}/slurm/setup_env.sh\"
echo '[job] Running experiments_v2/sundt_validation.py ...'
python -u experiments_v2/sundt_validation.py
echo '[job] Done. Outputs in outputs/sundt_validation/'
"
