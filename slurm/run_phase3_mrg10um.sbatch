#!/bin/bash
#SBATCH -J jaxfibers-phase3-mrg10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -q a16
#SBATCH --gres=gpu:a16:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 23:59:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Phase 3 (cross-diameter) — MRG @ D=10 µm.
#
# Same pipeline as run_phase3_manuscript_sweep.sbatch but with the
# fibre diameter bumped to 10 µm (Hussain 2024's other commonly-used
# MRG diameter).  Runs in parallel with the 5.7 µm sweep so the
# cross-diameter violin in Fig. 5 can be populated.
#
# Output:  outputs/selectivity_sweep_phase3_mrg10um/data_seed_*.json
# Resume:  FORCE_RERUN unset → re-submit picks up where it left off.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
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

export SKIP_PYFIBERS_COMPILE=1
export SLURM_STEP_LAUNCH_TIMEOUT=600

export SEED_START="${SEED_START:-0}"
export SEED_END="${SEED_END:-25}"
export SEEDS_PER_TASK=1
export OPTIMIZER=adam_fd
export N_OPT_RECT=30 N_OPT_WAVE=100
export WAVE_LR=5e-4
export WAVE_PATIENCE=20

# The only change vs the baseline phase-3 sweep:
export FIBER_DIAMETER_UM=10.0

# Distinct output dir so it doesn't collide with the 5.7 µm sweep.
export JAXLEY_FIBERS_OUTPUT_DIR=outputs/selectivity_sweep_phase3_mrg10um

echo "Phase 3 (MRG D=10 µm), seeds ${SEED_START}..$((SEED_END-1))"
echo "Output: ${JAXLEY_FIBERS_OUTPUT_DIR}"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc "
set -euo pipefail
export PROJECT_ROOT=\"${PROJECT_ROOT}\"
export JAX_ENABLE_X64=1
export SKIP_PYFIBERS_COMPILE=1
export SEED_START=\"${SEED_START}\"
export SEED_END=\"${SEED_END}\"
export SEEDS_PER_TASK=1
export OPTIMIZER=adam_fd
export N_OPT_RECT=30 N_OPT_WAVE=100
export WAVE_LR=5e-4
export WAVE_PATIENCE=20
export FIBER_DIAMETER_UM=10.0
export JAXLEY_FIBERS_OUTPUT_DIR=\"${JAXLEY_FIBERS_OUTPUT_DIR}\"
bash \"${PROJECT_ROOT}/slurm/setup_env.sh\"
python -u experiments_v2/selectivity_sweep.py
"
