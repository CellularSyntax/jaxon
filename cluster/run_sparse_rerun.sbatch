#!/bin/bash
#SBATCH --job-name=sparse_rerun
#SBATCH --partition=a16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/sparse_rerun_%j.out
#SBATCH --error=logs/sparse_rerun_%j.err

# Re-run sparse fiber-sampling sweep for one Duke sample.
# DUKE_SAMPLE_DIR must be passed via --export when submitting, e.g.:
#   sbatch --export=ALL,DUKE_SAMPLE_DIR=duke_Ves/sub-10_sam-1 cluster/run_sparse_rerun.sbatch
#
# Requires data_seed_NNNN.json to already exist (dense optimisation done).
# Overwrites existing sparse_sampling_seed_NNNN.json files.
# Output: outputs/duke_sweeps/<sample>/sparse_sampling_seed_NNNN.json
#   -- now extended with amps_mA, si_transfer, and pulse metadata.

source ~/.bashrc
conda activate jaxon

cd "$SLURM_SUBMIT_DIR"

export SPARSE_SAMPLING_SWEEP=true
export SPARSE_FORCE_RERUN=true
export SEED_START=${SEED_START:-0}
export SEED_END=${SEED_END:-25}

echo "=== sparse rerun: $DUKE_SAMPLE_DIR  seeds $SEED_START-$((SEED_END-1)) ==="

python -m experiments_v2.selectivity_sweep_duke
