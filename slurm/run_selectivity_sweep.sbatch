#!/bin/bash
#SBATCH -J jaxfibers-selectivity
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -q a16
#SBATCH --gres=gpu:a16:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-99%8

# Spatial selectivity sweep: 100 synthetic nerve realisations, run as a
# SLURM array — one task per seed.  %8 limits concurrent tasks to 8
# (s0-n12 has 8 a16 GPUs).  Each task is ~40-60 min after the speed
# cuts in selectivity_sweep.py (T_STOP=3 ms, N_OPT_RECT/WAVE=100), so
# the 6 h wall is ample.
#
# Outputs: outputs/selectivity_sweep/data_seed_<id>.json (one per task).
#
# Submit from the project root:
#   cd /path/to/jaxley_fibers
#   sbatch slurm/run_selectivity_sweep.sbatch
#
# To rerun a single seed:
#   sbatch --array=42 slurm/run_selectivity_sweep.sbatch
# To rerun a subset:
#   sbatch --array=0-19 slurm/run_selectivity_sweep.sbatch

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
# Resolve the container image once at submit time:
#   1. Honour an explicit CONTAINER_IMAGE env-var override.
#   2. Else prefer the project SquashFS at $HOME/containers/jaxfibers.sqsh
#      (built once via slurm/build_container.sh; has pip deps baked in,
#      so setup_env.sh skips its pip install step).
#   3. Else fall back to nvcr.io#nvidia/pytorch:25.03-py3 (Pyxis pulls +
#      caches it on the assigned node; setup_env.sh installs deps).
DEFAULT_SQSH="${HOME}/containers/jaxfibers.sqsh"
DEFAULT_IMAGE="nvcr.io#nvidia/pytorch:25.03-py3"
if [ -z "${CONTAINER_IMAGE:-}" ]; then
  if [ -f "${DEFAULT_SQSH}" ]; then
    CONTAINER_IMAGE="${DEFAULT_SQSH}"
  else
    CONTAINER_IMAGE="${DEFAULT_IMAGE}"
  fi
fi

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/outputs"

echo "Project root  : ${PROJECT_ROOT}"
echo "Container     : ${CONTAINER_IMAGE}"
echo "Array task ID : ${SLURM_ARRAY_TASK_ID:-(none)}"

# This task runs exactly one seed (= the array task id).  selectivity_sweep.py
# honours SEED_START / SEED_END env vars (see its main()), running the
# half-open range [SEED_START, SEED_END).
export SEED_START="${SLURM_ARRAY_TASK_ID:-0}"
export SEED_END="$((SEED_START + 1))"

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
export SEED_START=\"${SEED_START}\"
export SEED_END=\"${SEED_END}\"
bash \"${PROJECT_ROOT}/slurm/setup_env.sh\"
echo \"[job] Running experiments_v2/selectivity_sweep.py for seed \${SEED_START} ...\"
python -u experiments_v2/selectivity_sweep.py
echo '[job] Done. Outputs in outputs/selectivity_sweep/'
"
