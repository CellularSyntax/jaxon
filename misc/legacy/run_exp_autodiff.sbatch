#!/bin/bash
#SBATCH -J jaxfibers-exp-autodiff
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -q a16
#SBATCH --gres=gpu:a16:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Autodiff validation: jax.grad vs central finite differences.
# Produces: outputs/fig_autodiff_validation.png
#
# Submit from the project root:
#   cd /path/to/jaxley_fibers
#   sbatch slurm/run_exp_autodiff.sbatch

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
bash \"${PROJECT_ROOT}/slurm/setup_env.sh\"
echo '[job] Running exp_autodiff_validation.py ...'
python experiments/exp_autodiff_validation.py
echo '[job] Output: outputs/fig_autodiff_validation.png'
"
