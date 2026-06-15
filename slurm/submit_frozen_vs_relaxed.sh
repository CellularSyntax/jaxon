#!/bin/bash
# Submit N_SHARDS (default 4) shard jobs of the frozen-vs-relaxed sparsity
# experiment, i.e. N_SHARDS a100 running in parallel -- one sbatch per shard,
# which stays within the a100 QOS submit limit.
#
#   bash slurm/submit_frozen_vs_relaxed.sh               # 4 shards (4 a100)
#   N_SHARDS=2 bash slurm/submit_frozen_vs_relaxed.sh    # 2 shards
#   MAX_FIBERS=300 bash slurm/submit_frozen_vs_relaxed.sh # quick downsampled check
#
# After all shards finish, build the summary:
#   FROZEN_VS_RELAXED_AGGREGATE=1 python -m experiments_v2.frozen_vs_relaxed
set -euo pipefail
N="${N_SHARDS:-4}"
for ((s = 0; s < N; s++)); do
  SHARD_INDEX="$s" N_SHARDS="$N" sbatch slurm/run_frozen_vs_relaxed.sbatch
done
echo "Submitted ${N} shard job(s) of the frozen-vs-relaxed sparsity experiment."
