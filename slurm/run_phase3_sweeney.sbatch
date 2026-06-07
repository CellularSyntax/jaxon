#!/bin/bash
#SBATCH -J jaxfibers-phase3-sweeney
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -q a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 23:59:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Phase 3 (cross-model) — Sweeney @ D=10 µm.
#
# Sweeney is a myelinated fibre with simpler dynamics than MRG (m^2 h
# Na + leak, no slow K, no persistent Na), so the activation proxy
# behaves like MRG's.  T_STOP can stay at the MRG-style 3 ms.

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
DEFAULT_SQSH="${HOME}/containers/jaxfibers.sqsh"
DEFAULT_IMAGE="nvcr.io#nvidia/pytorch:25.03-py3"
if [ -z "${CONTAINER_IMAGE:-}" ]; then
  CONTAINER_IMAGE="${DEFAULT_SQSH:-$DEFAULT_IMAGE}"
  [ -f "${DEFAULT_SQSH}" ] || CONTAINER_IMAGE="${DEFAULT_IMAGE}"
fi
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/outputs"

export SKIP_PYFIBERS_COMPILE=1
export SLURM_STEP_LAUNCH_TIMEOUT=600

export SEED_START="${SEED_START:-0}"
export SEED_END="${SEED_END:-100}"
export SEEDS_PER_TASK=1
export OPTIMIZER=adam_fd
export N_OPT_RECT=30 N_OPT_WAVE=100
export WAVE_LR=5e-4 WAVE_PATIENCE=20

export FIBER_MODEL=sweeney
export FIBER_DIAMETER_UM=10.0
export T_STOP=3.0
export JAXLEY_FIBERS_OUTPUT_DIR=outputs/selectivity_sweep_phase3_sweeney_10um

echo "Phase 3 (Sweeney D=10 µm), seeds ${SEED_START}..$((SEED_END-1))"
srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc "
set -euo pipefail
export PROJECT_ROOT=\"${PROJECT_ROOT}\"
export JAX_ENABLE_X64=1
export SKIP_PYFIBERS_COMPILE=1
export SEED_START=\"${SEED_START}\" SEED_END=\"${SEED_END}\" SEEDS_PER_TASK=1
export OPTIMIZER=adam_fd
export N_OPT_RECT=30 N_OPT_WAVE=100
export WAVE_LR=5e-4 WAVE_PATIENCE=20
export FIBER_MODEL=sweeney FIBER_DIAMETER_UM=10.0 T_STOP=3.0
export JAXLEY_FIBERS_OUTPUT_DIR=\"${JAXLEY_FIBERS_OUTPUT_DIR}\"
bash \"${PROJECT_ROOT}/slurm/setup_env.sh\"
python -u experiments_v2/selectivity_sweep.py
"
