#!/bin/bash
# Launch frozen-vs-free + FD-vs-autodiff workers across MULTIPLE partitions at
# once (a100 + b200).  Every worker walks the full nerve list and atomically
# claims each nerve (see _claim in frozen_vs_relaxed.py), so the workers divide
# the work without duplication regardless of how many run on which partition --
# no shard-index coordination.  This is what lets you fill both the a100 and the
# b200 QOS submit limits simultaneously.
#
#   bash slurm/submit_frozen_vs_relaxed.sh                 # 4 a100 + 2 b200 -> outputs_new
#   FVR_A100_N=4 FVR_B200_N=0 bash slurm/submit_frozen_vs_relaxed.sh   # a100 only
#   FVR_OUT_ROOT=outputs2 bash slurm/submit_frozen_vs_relaxed.sh       # different out root
#
# Aggregate when done (reads the SAME out root):
#   FVR_OUT_ROOT=outputs_new FROZEN_VS_RELAXED_AGGREGATE=1 \
#     python -m experiments_v2.frozen_vs_relaxed
set -euo pipefail

OUT_ROOT="${FVR_OUT_ROOT:-outputs_new}"
export FVR_OUT_ROOT="${OUT_ROOT}"          # inherited by the sbatch jobs

# Per-partition worker counts (default to the a100=4 / b200=2 QOS limits).
N_A100="${FVR_A100_N:-4}"
N_B200="${FVR_B200_N:-2}"
# Partition / QOS / GRES per node type (override if your cluster names differ).
A100_PART="${FVR_A100_PART:-gpu}"; A100_Q="${FVR_A100_Q:-a100}"; A100_GRES="${FVR_A100_GRES:-gpu:a100:1}"
B200_PART="${FVR_B200_PART:-gpu}"; B200_Q="${FVR_B200_Q:-b200}"; B200_GRES="${FVR_B200_GRES:-gpu:b200:1}"

SBATCH="slurm/run_frozen_vs_relaxed.sbatch"
echo "Shared claim pool -> ${OUT_ROOT}/frozen_vs_relaxed/   (a100=${N_A100}, b200=${N_B200})"

for ((i = 0; i < N_A100; i++)); do
  sbatch -p "${A100_PART}" -q "${A100_Q}" --gres="${A100_GRES}" "${SBATCH}"
done
for ((i = 0; i < N_B200; i++)); do
  sbatch -p "${B200_PART}" -q "${B200_Q}" --gres="${B200_GRES}" "${SBATCH}"
done

echo "Submitted ${N_A100} a100 + ${N_B200} b200 worker(s).  They self-divide the"
echo "full nerve set via per-nerve claims; re-run this to add more workers anytime."
