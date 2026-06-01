#!/bin/bash
#SBATCH -J jaxfibers-selectivity
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -q b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --array=0-9
#SBATCH -o logs/%x-%A-%a.out
#SBATCH -e logs/%x-%A-%a.err

# Spatial selectivity sweep: 100 synthetic nerve realisations split across
# 10 array tasks (10 seeds each).  Aggregate analysis: analyze_selectivity_sweep.py
#
# Submit from the project root:
#   cd /path/to/jaxley_fibers
#   sbatch slurm/run_selectivity_sweep.sbatch

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:25.03-py3}"

SEEDS_PER_TASK=10
SEED_START=$(( SLURM_ARRAY_TASK_ID * SEEDS_PER_TASK ))
SEED_END=$(( SEED_START + SEEDS_PER_TASK ))

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/outputs"

echo "Project root  : ${PROJECT_ROOT}"
echo "Container     : ${CONTAINER_IMAGE}"
echo "Array task    : ${SLURM_ARRAY_TASK_ID}  seeds ${SEED_START}..${SEED_END}"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc "
set -euo pipefail
export PROJECT_ROOT=\"${PROJECT_ROOT}\"
export JAX_ENABLE_X64=1
export SEED_START=\"${SEED_START}\"
export SEED_END=\"${SEED_END}\"
bash \"${PROJECT_ROOT}/slurm/setup_env.sh\"
echo '[job] Running experiments_v2/selectivity_sweep.py seeds ${SEED_START}..${SEED_END} ...'
python experiments_v2/selectivity_sweep.py
echo '[job] Done. Outputs in outputs/selectivity_sweep/'
"
