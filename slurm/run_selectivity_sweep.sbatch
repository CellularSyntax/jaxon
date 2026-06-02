#!/bin/bash
#SBATCH -J jaxfibers-selectivity
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -q a16
#SBATCH --gres=gpu:a16:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-9%4

# Spatial selectivity sweep: 100 synthetic nerve realisations run as a SLURM
# array.  Each array task processes SEEDS_PER_TASK seeds — the rect optimiser
# is vmapped over the chunk, so all M=4 restarts × N seeds = 4N LBFGS
# trajectories run in parallel on the GPU.  Waveform step is per-seed
# (still uses Adam autodiff).
#
# Default sizing (a16 on s0-n12, 16 GB VRAM):
#   --array=0-9%4         10 tasks × 10 seeds = 100 (= N_NERVES); %4 limits
#                         concurrent tasks to 4 (leaves headroom on the
#                         8-GPU node for other users; bump to %8 if you want
#                         to monopolise all a16s)
#   SEEDS_PER_TASK=10     ~600 MB working set per task — comfortable on a16
#   per-task wall ~30-90 min (LBFGS + waveform-Adam combined)
#   total sweep wall: ~3-4 h in ~3 array waves
#
# Outputs: outputs/selectivity_sweep/data_seed_<id>.json (one per seed,
# saved by selectivity_sweep.py as it finishes each).
#
# Submit from the project root:
#   cd /path/to/jaxley_fibers
#   sbatch slurm/run_selectivity_sweep.sbatch
#
# Override examples
# -----------------
# Run on h100 instead (s0-n15, 4 GPUs available; per-task ~3× faster):
#   sbatch --qos=h100 --gres=gpu:h100:1 -t 1:00:00 \
#          slurm/run_selectivity_sweep.sbatch
#
# Push h100 harder (25 seeds × 4 restarts ≈ 43 GB):
#   sbatch --qos=h100 --gres=gpu:h100:1 -t 1:00:00 \
#          --array=0-3%4 --export=ALL,SEEDS_PER_TASK=25 \
#          slurm/run_selectivity_sweep.sbatch
#
# Use all 8 a16s instead of throttling to 4:
#   sbatch --array=0-9%8 slurm/run_selectivity_sweep.sbatch
#
# Re-run a single seed:
#   sbatch --array=42 --export=ALL,SEEDS_PER_TASK=1 \
#          slurm/run_selectivity_sweep.sbatch

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"

# How many seeds this task should run (vmapped together for rect).
# Default matches --array=0-9: 10 tasks × 10 seeds = 100 = N_NERVES.
SEEDS_PER_TASK="${SEEDS_PER_TASK:-10}"

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
