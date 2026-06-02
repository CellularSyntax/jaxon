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
#SBATCH --array=0-24%8

# Spatial selectivity sweep: 100 synthetic nerve realisations run as a SLURM
# array.  Each array task processes SEEDS_PER_TASK seeds (default 4) — the
# rect optimiser is vmapped over the chunk, so all M=4 restarts × 4 seeds
# = 16 LBFGS trajectories run in parallel on the GPU.  Waveform step is
# per-seed (still uses Adam autodiff).
#
# Sizing:
#   --array=0-24      25 tasks * 4 seeds/task = 100 seeds (= N_NERVES)
#   %8                 ≤ 8 concurrent on s0-n12's 8 a16 GPUs
#   SEEDS_PER_TASK=4   safe on a16 (16 GB); set 8 on a100, 16 on h100/b200
#   per-task wall ~30-60 min, so 6 h is ample
#
# Outputs: outputs/selectivity_sweep/data_seed_<id>.json (one per seed,
# saved by selectivity_sweep.py as it finishes each).
#
# Submit from the project root:
#   cd /path/to/jaxley_fibers
#   sbatch slurm/run_selectivity_sweep.sbatch
#
# Re-run with a larger seed-batch (e.g. on an a100):
#   SEEDS_PER_TASK=8 sbatch --array=0-12%8 slurm/run_selectivity_sweep.sbatch
# Re-run a single seed:
#   sbatch --array=42 --export=ALL,SEEDS_PER_TASK=1 slurm/run_selectivity_sweep.sbatch

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"

# How many seeds this task should run (vmapped together for rect).
SEEDS_PER_TASK="${SEEDS_PER_TASK:-4}"

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

# Map the array task id to a contiguous chunk of seeds:
#   task 0 -> seeds 0..SEEDS_PER_TASK
#   task 1 -> seeds SEEDS_PER_TASK..2*SEEDS_PER_TASK
#   ...
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
export SEED_START="$(( TASK_ID * SEEDS_PER_TASK ))"
export SEED_END="$(( SEED_START + SEEDS_PER_TASK ))"
export SEEDS_PER_TASK

echo "Project root    : ${PROJECT_ROOT}"
echo "Container       : ${CONTAINER_IMAGE}"
echo "Array task ID   : ${TASK_ID}"
echo "Seeds this task : ${SEED_START}..$((SEED_END - 1)) (${SEEDS_PER_TASK} seeds vmapped)"

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
export SEEDS_PER_TASK=\"${SEEDS_PER_TASK}\"
bash \"${PROJECT_ROOT}/slurm/setup_env.sh\"
echo \"[job] Running selectivity_sweep.py seeds \${SEED_START}..\$((SEED_END-1)) (\${SEEDS_PER_TASK} per vmap chunk) ...\"
python -u experiments_v2/selectivity_sweep.py
echo '[job] Done. Outputs in outputs/selectivity_sweep/'
"
