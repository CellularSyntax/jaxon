#!/bin/bash
#SBATCH -J jaxfibers-phase2-lbfgs
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -q a16
#SBATCH --gres=gpu:a16:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 2:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Phase 2 — LBFGS rect at full N=200 (single-diameter, sequential
# restarts via lax.map).  Verifies the second-order optimiser at
# manuscript scale on a real problem.
#
# Skipped waveform (N_OPT_WAVE=0) — Phase 2 is about LBFGS rect only.

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

export SEED_START=0 SEED_END=1 SEEDS_PER_TASK=1
export FORCE_RERUN=1
export OPTIMIZER=lbfgs
export N_OPT_RECT=30 N_OPT_WAVE=0
export N_RESTARTS_RECT=2
export JAXLEY_FIBERS_OUTPUT_DIR=outputs/selectivity_sweep_phase2_lbfgs_verify

echo "Phase 2 — LBFGS rect at N=200, sequential restarts (lax.map)"
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
export SEED_START=0 SEED_END=1 SEEDS_PER_TASK=1
export FORCE_RERUN=1
export OPTIMIZER=lbfgs
export N_OPT_RECT=30 N_OPT_WAVE=0
export N_RESTARTS_RECT=2
export JAXLEY_FIBERS_OUTPUT_DIR=\"${JAXLEY_FIBERS_OUTPUT_DIR}\"
bash \"${PROJECT_ROOT}/slurm/setup_env.sh\"
python -u experiments_v2/selectivity_sweep.py
"
