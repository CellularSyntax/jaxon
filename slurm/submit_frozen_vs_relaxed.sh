#!/bin/bash
# Launch frozen-vs-free + FD-vs-autodiff workers across MULTIPLE partitions at
# once (a100 + h100).  Every worker walks the full nerve list and atomically
# claims each nerve (see _claim in frozen_vs_relaxed.py), so the workers divide
# the work without duplication regardless of how many run on which partition --
# no shard-index coordination.  This is what lets you fill both the a100 and the
# h100 QOS submit limits simultaneously.  (h100 instead of b200 avoids the
# Blackwell/container risk; switch the secondary pool to anything via FVR_SEC_*.)
#
#   bash slurm/submit_frozen_vs_relaxed.sh                 # 4 a100 + 2 h100 -> outputs_new
#   FVR_A100_N=4 FVR_SEC_N=0 bash slurm/submit_frozen_vs_relaxed.sh    # a100 only
#   FVR_SEC_Q=b200 FVR_SEC_GRES=gpu:b200:1 bash slurm/submit_frozen_vs_relaxed.sh  # use b200
#   FVR_OUT_ROOT=outputs2 bash slurm/submit_frozen_vs_relaxed.sh       # different out root
#
# Aggregate when done (reads the SAME out root):
#   FVR_OUT_ROOT=outputs_new FROZEN_VS_RELAXED_AGGREGATE=1 \
#     python -m experiments_v2.frozen_vs_relaxed
set -euo pipefail

OUT_ROOT="${FVR_OUT_ROOT:-outputs_new}"
export FVR_OUT_ROOT="${OUT_ROOT}"          # inherited by the sbatch jobs

# Primary pool = a100 (QOS limit 4); secondary pool = h100 (QOS limit 2).
N_A100="${FVR_A100_N:-4}"
N_SEC="${FVR_SEC_N:-2}"
# Partition / QOS / GRES per pool (override if your cluster names differ).
A100_PART="${FVR_A100_PART:-gpu}"; A100_Q="${FVR_A100_Q:-a100}"; A100_GRES="${FVR_A100_GRES:-gpu:a100:1}"
SEC_PART="${FVR_SEC_PART:-gpu}";   SEC_Q="${FVR_SEC_Q:-h100}";   SEC_GRES="${FVR_SEC_GRES:-gpu:h100:1}"

SBATCH="slurm/run_frozen_vs_relaxed.sbatch"
echo "Shared claim pool -> ${OUT_ROOT}/frozen_vs_relaxed/   (a100=${N_A100}, ${SEC_Q}=${N_SEC})"

for ((i = 0; i < N_A100; i++)); do
  sbatch -p "${A100_PART}" -q "${A100_Q}" --gres="${A100_GRES}" "${SBATCH}"
done
for ((i = 0; i < N_SEC; i++)); do
  sbatch -p "${SEC_PART}" -q "${SEC_Q}" --gres="${SEC_GRES}" "${SBATCH}"
done

echo "Submitted ${N_A100} a100 + ${N_SEC} ${SEC_Q} worker(s).  They self-divide the"
echo "full nerve set via per-nerve claims; re-run this to add more workers anytime."
