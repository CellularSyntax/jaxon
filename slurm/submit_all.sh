#!/bin/bash
# Submit all experiments as independent Slurm jobs.
# They will run in parallel (subject to cluster availability).
#
# Run from the project root:
#   cd /path/to/jaxley_fibers
#   bash slurm/submit_all.sh

set -euo pipefail

mkdir -p logs

echo "Submitting experiments_v2 jobs ..."

sbatch slurm/run_scaling.sbatch
sbatch slurm/run_mrg_validation.sbatch
sbatch slurm/run_sundt_validation.sbatch

echo ""
echo "All jobs submitted. Monitor with:"
echo "  squeue -u \$(whoami)"
echo "  tail -f logs/<job-name>-<jobid>.out"
