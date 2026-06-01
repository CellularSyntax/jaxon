#!/bin/bash
#SBATCH -J jaxfibers-exp-benchmark
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

# Wall-clock performance benchmark: MRG + Sundt; PyFibers vs JAX coupled solver.
# Produces: outputs/fig_performance_benchmark.png, benchmark JSON/CSV files.
#
# Submit from the project root:
#   cd /path/to/jaxley_fibers
#   sbatch slurm/run_exp_benchmark.sbatch

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
echo '[job] Running exp_performance_benchmark.py ...'
python experiments/exp_performance_benchmark.py
echo '[job] Output: outputs/fig_performance_benchmark.png'
"
