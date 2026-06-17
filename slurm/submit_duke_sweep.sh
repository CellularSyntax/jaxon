#!/bin/bash
# Launch the Duke selectivity sweep across MULTIPLE partitions at once
# (a100 + h100), the same way submit_frozen_vs_relaxed.sh does.  The sweep
# shards the nerve set with the sbatch's GLOBAL_N_SHARDS/SHARD_OFFSET
# mechanism (run_duke_sweep.sbatch lines ~226-235): each array task owns
# global shard (SLURM_ARRAY_TASK_ID + SHARD_OFFSET) of GLOBAL_N_SHARDS, so
# the a100 pool and the h100 pool cover DISJOINT samples with no overlap.
#
# Total shards = DUKE_A100_N + DUKE_SEC_N (default 4 + 2 = 6).  Each pool is
# ONE sbatch array sized to that pool, so the a100 array (4 tasks) fills the
# a100 QOS=4 and the h100 array (2 tasks) fills the h100 QOS=2 exactly.
#
# Autodiff full-figure re-run (dense + sparse, 4 positions):
#   RECT_OPTIMIZER=autodiff SWEEP_BALANCE=1 \
#   DUKE_SWEEP_ROOT=outputs/duke_sweeps_autodiff \
#   SPARSE_SAMPLING_SWEEP=true SEED_END=4 \
#   JAXLEY_FIBERS_LOSS=quotient JAXLEY_FIBERS_SOFT_TEMPERATURE=0.15 \
#     bash slurm/submit_duke_sweep.sh
#
# a100-only / different counts / b200 secondary:
#   DUKE_A100_N=4 DUKE_SEC_N=0 bash slurm/submit_duke_sweep.sh
#   DUKE_SEC_Q=b200 DUKE_SEC_GRES=gpu:b200:1 bash slurm/submit_duke_sweep.sh
#
# Re-submit to mop up stragglers: per-seed JSONs are written incrementally and
# completed seeds are skipped, so re-running this picks up only missing work.
#
# Aggregate / render afterwards (reads the SAME tree):
#   DUKE_SWEEP_ROOT=outputs/duke_sweeps_autodiff \
#     python -m experiments_v2.figures_duke_main
set -euo pipefail

# Primary pool = a100 (QOS limit 4); secondary pool = h100 (QOS limit 2).
N_A100="${DUKE_A100_N:-4}"
N_SEC="${DUKE_SEC_N:-2}"
# Partition / QOS / GRES per pool (override if your cluster names differ).
A100_PART="${DUKE_A100_PART:-gpu}"; A100_Q="${DUKE_A100_Q:-a100}"; A100_GRES="${DUKE_A100_GRES:-gpu:a100:1}"
SEC_PART="${DUKE_SEC_PART:-gpu}";   SEC_Q="${DUKE_SEC_Q:-h100}";   SEC_GRES="${DUKE_SEC_GRES:-gpu:h100:1}"

N_TOTAL=$(( N_A100 + N_SEC ))
if [ "${N_TOTAL}" -le 0 ]; then
  echo "DUKE_A100_N + DUKE_SEC_N must be > 0" >&2; exit 1
fi

SBATCH="slurm/run_duke_sweep.sbatch"
echo "Duke sweep over ${N_TOTAL} disjoint shards:"
echo "  a100 = ${N_A100}  (shards 0..$((N_A100 - 1)))"
echo "  ${SEC_Q} = ${N_SEC}  (shards ${N_A100}..$((N_TOTAL - 1)))"
echo "  RECT_OPTIMIZER=${RECT_OPTIMIZER:-adam_fd}  DUKE_SWEEP_ROOT=${DUKE_SWEEP_ROOT:-outputs/duke_sweeps}"
echo "  SPARSE_SAMPLING_SWEEP=${SPARSE_SAMPLING_SWEEP:-false}  SEED_END=${SEED_END:-25}  LOSS=${JAXLEY_FIBERS_LOSS:-linear}"

# Each `VAR=.. sbatch` runs with --export=ALL (default), so the autodiff env
# the caller set (RECT_OPTIMIZER, DUKE_SWEEP_ROOT, SPARSE_SAMPLING_SWEEP, ...)
# propagates to the job; we only need to set the per-pool shard coordinates.
if [ "${N_A100}" -gt 0 ]; then
  GLOBAL_N_SHARDS="${N_TOTAL}" SHARD_OFFSET=0 \
    sbatch -p "${A100_PART}" -q "${A100_Q}" --gres="${A100_GRES}" \
           --array=0-$((N_A100 - 1)) "${SBATCH}"
fi
if [ "${N_SEC}" -gt 0 ]; then
  GLOBAL_N_SHARDS="${N_TOTAL}" SHARD_OFFSET="${N_A100}" \
    sbatch -p "${SEC_PART}" -q "${SEC_Q}" --gres="${SEC_GRES}" \
           --array=0-$((N_SEC - 1)) "${SBATCH}"
fi

echo "Submitted ${N_A100} a100 + ${N_SEC} ${SEC_Q} shard(s) of ${N_TOTAL}."
echo "Re-run this anytime to mop up missing seeds (completed seeds are skipped)."
