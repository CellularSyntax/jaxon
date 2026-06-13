#!/bin/bash
# Submit sparse re-run jobs for all Duke samples that have completed dense data.
#
# Usage (from project root):
#   bash cluster/submit_sparse_rerun.sh
#
# One SLURM job per sample.  Max 4 concurrent (a16 QOS limit).
# Each job overwrites existing sparse_sampling_seed_NNNN.json files with the
# updated format that includes amps_mA, si_transfer, and pulse metadata.

SWEEP_DIR="outputs/duke_sweeps"
SBATCH_SCRIPT="cluster/run_sparse_rerun.sbatch"

submitted=0
for sample_dir in "$SWEEP_DIR"/*/; do
    sample=$(basename "$sample_dir")
    # Only submit if at least one dense JSON exists for this sample
    if ! ls "$sample_dir"/data_seed_*.json 1>/dev/null 2>&1; then
        echo "[skip] $sample — no dense data found"
        continue
    fi
    duke_path="duke_Ves/${sample}"
    if [ ! -d "$duke_path" ]; then
        echo "[skip] $sample — duke_Ves/$sample not found"
        continue
    fi
    echo "[submit] $sample"
    sbatch \
        --job-name="sp_${sample:0:12}" \
        --export=ALL,DUKE_SAMPLE_DIR="$duke_path" \
        "$SBATCH_SCRIPT"
    submitted=$((submitted + 1))
done

echo ""
echo "Submitted $submitted jobs."
