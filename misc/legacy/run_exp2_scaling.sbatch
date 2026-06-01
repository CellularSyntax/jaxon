#!/bin/bash
#SBATCH -J jaxfibers-exp2-scaling
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -q a16
#SBATCH --gres=gpu:a16:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# GPU scaling benchmark: PyFibers serial vs Jaxley CPU vmap vs Jaxley GPU vmap.
# N fibers from 1 to 1000 (extend JAXLEY_BATCH_NS in the script for N=10000 on A100).
# Produces: outputs/fig2_scaling.png
#
# Submit from the project root:
#   cd /path/to/jaxley_fibers
#   sbatch slurm/run_exp2_scaling.sbatch

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
echo '[job] Running exp_2_scaling.py ...'
# Note: experiments/exp_2_scaling.py has JAXLEY_BATCH_NS = [1,10,100,1000].
# On the A100 (40 GB) you can safely extend that list to include 10000.
python experiments/exp_2_scaling.py
echo '[job] Output: outputs/fig2_scaling.png'
"
