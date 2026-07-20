#!/bin/bash
#SBATCH -J jaxfibers-selectivity
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -q a16
#SBATCH --gres=gpu:a16:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Spatial selectivity sweep — VERIFICATION RUN.
#
# Single job, single seed (seed 0), single nerve realisation.  Used to
# verify the optimisation pipeline (LBFGS rect + Adam waveform) works
# end-to-end before scaling to multiple geometries.
#
# Setup (matches Hussain 2024 methodology, scaled for jaxfibers parallel
# demonstration):
#   * N_FIBERS = 200, all at single diameter D = 5.7 µm
#     (Hussain placed 1 fiber per fascicle at 5.7 µm; we vmap 200 fibers
#     in a single synthetic nerve to showcase the parallel speed advantage)
#   * 6-contact ring cuff, 1.5 mm radius
#   * One eccentric target fascicle ~30 % area
#
# Once this run succeeds, scale to the full 100-seed sweep with the
# array override (see "Scaling examples" below).
#
# Cluster QOS limits (MaxJobsPerUser per QOS):
#   a16   : 4    a30 : unlimited    a100 : 4 (12/account)
#   h100  : 2    b200: 2            a100-sxm4-80gb : 4
#
# Outputs: outputs/selectivity_sweep/data_seed_0000.json
#
# Submit from the project root:
#   cd /path/to/jaxley_fibers
#   sbatch slurm/run_selectivity_sweep.sbatch
#
# LBFGS rect restarts
# -------------------
# The verification run defaults to N_RESTARTS_RECT=2 (set inside
# selectivity_sweep.py) so the autodiff-backward graph fits in A16's
# ~12 GB usable VRAM with N_FIBERS=200.  Override via env var on
# bigger GPUs:
#   sbatch --qos=a100 --gres=gpu:a100:1 -t 1:00:00 \
#          --export=ALL,N_RESTARTS_RECT=4 \
#          slurm/run_selectivity_sweep.sbatch
#
# Scaling examples (after verification succeeds)
# ----------------------------------------------
# Full 100-seed sweep on a16, 10 seeds vmapped per task, 4 concurrent
# (a16 MaxJobsPerUser=4); also override N_RESTARTS_RECT for h100/a100:
#   sbatch --array=0-9%4 --export=ALL,SEEDS_PER_TASK=10 -t 4:00:00 \
#          slurm/run_selectivity_sweep.sbatch
#
# Same on a100 (~3× faster per task, fits N_RESTARTS_RECT=4):
#   sbatch --qos=a100 --gres=gpu:a100:1 -t 1:00:00 \
#          --array=0-9%4 \
#          --export=ALL,SEEDS_PER_TASK=10,N_RESTARTS_RECT=4 \
#          slurm/run_selectivity_sweep.sbatch
#
# Re-run a single seed:
#   sbatch --export=ALL,SEED_START=42,SEED_END=43 \
#          slurm/run_selectivity_sweep.sbatch

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"

# How many seeds this task should run (vmapped together for rect).
# Verification-run default = 1 (single seed only).  Override to 10
# (or higher on a100/h100) when scaling the full sweep via --array.
SEEDS_PER_TASK="${SEEDS_PER_TASK:-1}"

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
# For non-array verification runs SLURM_ARRAY_TASK_ID is unset → TASK_ID=0
# → SEED_START=0, SEED_END=SEEDS_PER_TASK (default 1 → just seed 0).
# An explicit SEED_START / SEED_END override takes precedence.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
export SEED_START="${SEED_START:-$(( TASK_ID * SEEDS_PER_TASK ))}"
export SEED_END="${SEED_END:-$(( SEED_START + SEEDS_PER_TASK ))}"
export SEEDS_PER_TASK

# LBFGS rect-restart count.  Default 2 fits A16's ~12 GB usable VRAM with
# N_FIBERS=200; bump to 4 on a100/h100 (see header).
export N_RESTARTS_RECT="${N_RESTARTS_RECT:-2}"

# selectivity_sweep.py is pure JAX — no NEURON mechanisms needed.
# Skip the ~10 s pyfibers_compile pass in setup_env.sh.
export SKIP_PYFIBERS_COMPILE=1

echo "Project root    : ${PROJECT_ROOT}"
echo "Container       : ${CONTAINER_IMAGE}"
echo "Array task ID   : ${TASK_ID}"
echo "Seeds this task : ${SEED_START}..$((SEED_END - 1)) (${SEEDS_PER_TASK} seeds vmapped)"
echo "LBFGS restarts  : ${N_RESTARTS_RECT}"

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
export N_RESTARTS_RECT=\"${N_RESTARTS_RECT}\"
export SKIP_PYFIBERS_COMPILE=\"${SKIP_PYFIBERS_COMPILE}\"
bash \"${PROJECT_ROOT}/slurm/setup_env.sh\"
echo \"[job] Running selectivity_sweep.py seeds \${SEED_START}..\$((SEED_END-1)) (\${SEEDS_PER_TASK} per vmap chunk) ...\"
python -u experiments_v2/selectivity_sweep.py
echo '[job] Done. Outputs in outputs/selectivity_sweep/'
"
