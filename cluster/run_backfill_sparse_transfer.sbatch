#!/bin/bash
#SBATCH --job-name=backfill_sparse
#SBATCH --partition=gpu
#SBATCH --qos=a16
#SBATCH --gres=gpu:a16:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/backfill_sparse_%A_%a.out
#SBATCH --error=logs/backfill_sparse_%A_%a.err

# Backfill rich per-fiber data into existing sparse_sampling_seed_*.json.
#
# The original sparse runs stored only scalar SI / SI_transfer / amps for each
# sparsity level.  Figure 3 now needs the same per-fiber data we keep for dense
# runs: the sparse-optimised amps re-evaluated on the FULL dense nerve
# (transfer.final_acts for activation maps, transfer.firing for on-/off-target
# recruitment), plus the in-sample sparse-nerve final_acts/firing.
#
# This is forward-pass-only — it reuses the stored amps_mA and never
# re-optimises, so amps_mA / si_transfer are untouched and every figure value
# stays identical.  Each result's recomputed SI is gated against the stored
# si_transfer (tol 5e-3); the job exits non-zero if any seed deviates.
#
# SHARDING (split across nodes, like run_duke_sweep.sbatch):
#   --array=0-3 (default) splits the samples into 4 interleaved shards, one per
#   array task.  The python script reads SLURM_ARRAY_TASK_ID / _COUNT and
#   processes sample i iff i % N_SHARDS == SHARD_IDX.  All shards write disjoint
#   sample folders into the SAME outputs/duke_sweeps/ tree — nothing to merge.
#   a16 QOS allows 4 concurrent jobs, so 4 shards run in parallel.  For more
#   shards than the concurrency limit, throttle with e.g. --array=0-7%4.
#
# Submit:
#   sbatch cluster/run_backfill_sparse_transfer.sbatch
#   # finer split / different QOS:
#   sbatch --array=0-7%4 cluster/run_backfill_sparse_transfer.sbatch
#   sbatch --qos=a100 --gres=gpu:a100:1 cluster/run_backfill_sparse_transfer.sbatch
#
# After all shards finish: commit the updated sparse_sampling_seed_*.json and
# pull locally for the figure.

set -euo pipefail

source ~/.bashrc
conda activate jaxley_fibers

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

export JAX_ENABLE_X64=1
export SKIP_PYFIBERS_COMPILE=1

echo "=== backfill sparse transfer ==="
echo "array task ${SLURM_ARRAY_TASK_ID:-0} of ${SLURM_ARRAY_TASK_COUNT:-1}"

# No argv -> the script discovers all samples under outputs/duke_sweeps/ and
# selects this task's interleaved shard from SLURM_ARRAY_TASK_ID / _COUNT.
python -u -m experiments_v2.backfill_sparse_transfer

echo "=== shard ${SLURM_ARRAY_TASK_ID:-0} done ==="
