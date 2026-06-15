#!/bin/bash
# Submit N_SHARDS (default 4) shard jobs of the FD-vs-autodiff head-to-head,
# i.e. N_SHARDS a100 running in parallel -- one sbatch per shard, which stays
# within the a100 QOS submit limit (an --array of 12 does not).
#
#   bash slurm/submit_headtohead.sh                 # 4 shards (4 a100)
#   N_SHARDS=2 bash slurm/submit_headtohead.sh      # 2 shards
#   MAX_FIBERS=0 bash slurm/submit_headtohead.sh    # full population
#
# After all shards finish, build the table:
#   HEADTOHEAD_AGGREGATE=1 python -m experiments_v2.headtohead_fd_vs_autodiff
set -euo pipefail
N="${N_SHARDS:-4}"
for ((s = 0; s < N; s++)); do
  SHARD_INDEX="$s" N_SHARDS="$N" sbatch slurm/run_headtohead_fd_vs_autodiff.sbatch
done
echo "Submitted ${N} shard job(s) of the FD-vs-autodiff head-to-head."
