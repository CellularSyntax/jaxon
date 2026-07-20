#!/bin/bash
#SBATCH -J jaxon-phase3-sundt
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

# Phase 3 (cross-model) — Sundt C-fibre @ D=1.0 µm.
#
# Routes through the new model-agnostic batched solver in
# jaxon.stim.batch_solve_generic, which dispatches the per-step
# membrane currents through SundtAxon (nahh + borgkdr + leak).
#
# Output: outputs/selectivity_sweep_phase3_sundt_1um/data_seed_*.json
#
# Sundt has slower CV (~0.5 m/s vs MRG ~60 m/s), so T_STOP is bumped
# from 2 ms to 12 ms to give the AP time to traverse the fibre.  Amp
# clip is bumped to ±50 mA to span the higher unmyelinated threshold
# (~22 mA at the 1.5 mm cuff-to-fibre distance vs ~0.2 mA for MRG).

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
DEFAULT_SQSH="${HOME}/containers/jaxon.sqsh"
DEFAULT_IMAGE="nvcr.io#nvidia/pytorch:25.03-py3"
if [ -z "${CONTAINER_IMAGE:-}" ]; then
  CONTAINER_IMAGE="${DEFAULT_SQSH:-$DEFAULT_IMAGE}"
  [ -f "${DEFAULT_SQSH}" ] || CONTAINER_IMAGE="${DEFAULT_IMAGE}"
fi
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/outputs"

export SKIP_PYFIBERS_COMPILE=1
export SLURM_STEP_LAUNCH_TIMEOUT=600

export SEED_START="${SEED_START:-0}"
export SEED_END="${SEED_END:-25}"
export SEEDS_PER_TASK=1
export OPTIMIZER=adam_fd
export N_OPT_RECT=30 N_OPT_WAVE=100
export WAVE_LR=5e-4 WAVE_PATIENCE=20

export FIBER_MODEL=sundt
export FIBER_DIAMETER_UM=1.0
export T_STOP=12.0
export JAXON_OUTPUT_DIR=outputs/selectivity_sweep_phase3_sundt_1um

echo "Phase 3 (Sundt D=1.0 µm), seeds ${SEED_START}..$((SEED_END-1))"
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
export FIBER_MODEL=sundt FIBER_DIAMETER_UM=1.0 T_STOP=12.0
export JAXON_OUTPUT_DIR=\"${JAXON_OUTPUT_DIR}\"
bash \"${PROJECT_ROOT}/slurm/setup_env.sh\"
python -u experiments_v2/selectivity_sweep.py
"
