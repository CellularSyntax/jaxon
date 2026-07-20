#!/bin/bash
#SBATCH -J jaxon-all-experiments
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -q a16
#SBATCH --gres=gpu:a16:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Run all experiments_v2 scripts sequentially in a single job.
# Useful for a clean full-reproduction run.
#
# Expected wall time: ~48–72 h on A16 (scaling + MRG + Sundt + Rattay + Sweeney validation).
# Adjust -t above if needed (max for a16 QOS: 72 h).
#
# Submit from the project root:
#   cd /path/to/jaxon
#   sbatch slurm/run_all.sbatch

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
# Resolve the container image once at submit time:
#   1. Honour an explicit CONTAINER_IMAGE env-var override.
#   2. Else prefer the project SquashFS at $HOME/containers/jaxon.sqsh
#      (built once via slurm/build_container.sh; has pip deps baked in,
#      so setup_env.sh skips its pip install step).
#   3. Else fall back to nvcr.io#nvidia/pytorch:25.03-py3 (Pyxis pulls +
#      caches it on the assigned node; setup_env.sh installs deps).
DEFAULT_SQSH="${HOME}/containers/jaxon.sqsh"
DEFAULT_IMAGE="nvcr.io#nvidia/pytorch:25.03-py3"
if [ -z "${CONTAINER_IMAGE:-}" ]; then
  if [ -f "${DEFAULT_SQSH}" ]; then
    CONTAINER_IMAGE="${DEFAULT_SQSH}"
  else
    CONTAINER_IMAGE="${DEFAULT_IMAGE}"
  fi
fi

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/outputs"

echo "Project root : ${PROJECT_ROOT}"
echo "Container    : ${CONTAINER_IMAGE}"
echo "Start time   : $(date)"

# Pyxis container pulls can exceed SLURM's 32-second default step-launch
# timeout on a cold node cache (the nvcr.io PyTorch image is ~10 GB).
# Allow up to 10 minutes for srun to bring the container up.
export SLURM_STEP_LAUNCH_TIMEOUT=600

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc "
set -euo pipefail
export PROJECT_ROOT=\"${PROJECT_ROOT}\"
export JAX_ENABLE_X64=1

# ── Install once, reuse for all experiments ──────────────────────────────────
bash \"${PROJECT_ROOT}/slurm/setup_env.sh\"

run_exp() {
  local script=\"\$1\"
  echo \"\"
  echo '========================================================='
  echo \"[job] \$(date '+%H:%M:%S')  Starting: \$script\"
  echo '========================================================='
  python \"\$script\"
  echo \"[job] \$(date '+%H:%M:%S')  Finished: \$script\"
}

# ── Experiments v2 ───────────────────────────────────────────────────────────
run_exp experiments_v2/scaling.py
run_exp experiments_v2/mrg_validation.py
run_exp experiments_v2/mrg_interp_validation.py
run_exp experiments_v2/sundt_validation.py
run_exp experiments_v2/rattay_validation.py
run_exp experiments_v2/sweeney_validation.py
run_exp experiments_v2/schild94_validation.py
run_exp experiments_v2/schild97_validation.py

echo ''
echo '========================================================='
echo '[job] All experiments complete.'
echo '[job] Outputs are in: ${PROJECT_ROOT}/outputs/'
echo '========================================================='
"

echo "End time : $(date)"
