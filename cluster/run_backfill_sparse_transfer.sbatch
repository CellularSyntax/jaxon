#!/bin/bash
#SBATCH --job-name=backfill_sparse
#SBATCH -p gpu
#SBATCH -q a16
#SBATCH --gres=gpu:a16:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --array=0-3
#SBATCH -o logs/backfill_sparse_%A_%a.out
#SBATCH -e logs/backfill_sparse_%A_%a.err

# Backfill rich per-fiber transfer data into existing sparse_sampling_seed_*.json
# (forward-pass only; reuses stored amps_mA, NO re-optimization -> far faster than
# the sparse sweep).  Adds a `transfer` block (final_acts/firing/si on the FULL
# nerve) + sparse-side acts/firing; never touches amps_mA/si/si_transfer.
#
# Runs in the SAME container + setup_env.sh environment as run_duke_sweep.sbatch
# (the compute nodes have no `conda`).  Sharding: --array splits samples across
# tasks (sample i -> shard i % N); each shard writes disjoint sample folders.
#
# Submit:
#   sbatch cluster/run_backfill_sparse_transfer.sbatch
#   sbatch --array=0   cluster/run_backfill_sparse_transfer.sbatch   # single node
#   sbatch --qos=a100 --gres=gpu:a100:1 cluster/run_backfill_sparse_transfer.sbatch

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
DEFAULT_SQSH="${HOME}/containers/jaxfibers.sqsh"
DEFAULT_IMAGE="nvcr.io#nvidia/pytorch:25.03-py3"
if [ -z "${CONTAINER_IMAGE:-}" ]; then
  if [ -f "${DEFAULT_SQSH}" ]; then
    CONTAINER_IMAGE="${DEFAULT_SQSH}"
  else
    CONTAINER_IMAGE="${DEFAULT_IMAGE}"
  fi
fi

mkdir -p "${PROJECT_ROOT}/logs"
export SKIP_PYFIBERS_COMPILE=1
export SLURM_STEP_LAUNCH_TIMEOUT=600

# Shard bookkeeping (resolved in the outer script where SLURM_ARRAY_* are set,
# then passed explicitly so the in-container python sees them reliably).
N_SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
SHARD_IDX="${SLURM_ARRAY_TASK_ID:-0}"
echo "backfill shard ${SHARD_IDX} of ${N_SHARDS}  (image=${CONTAINER_IMAGE})"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc "
set -euo pipefail
export PROJECT_ROOT=\"${PROJECT_ROOT}\"
export JAX_ENABLE_X64=1
export SKIP_PYFIBERS_COMPILE=1
export GLOBAL_N_SHARDS=\"${N_SHARDS}\"
export SHARD_INDEX=\"${SHARD_IDX}\"
bash \"${PROJECT_ROOT}/slurm/setup_env.sh\"
python -u -m experiments_v2.backfill_sparse_transfer
"

echo "backfill shard ${SHARD_IDX} done"
