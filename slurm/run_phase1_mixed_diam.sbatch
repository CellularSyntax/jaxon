#!/bin/bash
#SBATCH -J jaxfibers-phase1-mixeddiam
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -q a16
#SBATCH --gres=gpu:a16:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 2:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Phase 1 — Mixed-diameter rect + waveform.
#
# Target = 5.7 µm sensory fibres, off-target = 14 µm motor fibres.
# Demonstrates the regime where waveform optimisation actually beats
# rect (different chronaxie between the two populations), justifying
# the existence of the waveform step.  Single seed, Adam-FD.
#
# Output: outputs/selectivity_sweep_phase1_mixed_diam/{data,figs} for
# seed 0.

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

# Phase-specific config
export SEED_START=0 SEED_END=1 SEEDS_PER_TASK=1
export FORCE_RERUN=1
export OPTIMIZER=adam_fd
export TARGET_DIAMETER_UM=5.7
export OFFTARGET_DIAMETER_UM=14.0
export N_OPT_RECT=30 N_OPT_WAVE=50
export T_STOP=2.0
# Soft sigmoid activation proxy — required for mixed-diameter selectivity.
# Without this the binary min(m[first], m[last]) proxy has zero gradient
# at the saturated state both rect and waveform converge to, trapping
# them at SI=0 (verified empirically on this geometry).  0.1 is the
# default sweet spot: wide enough to give gradients in the relevant
# range, narrow enough that "fired" still means m_node ≈ 1.
export JAXLEY_FIBERS_SOFT_TEMPERATURE=0.1
export JAXLEY_FIBERS_OUTPUT_DIR=outputs/selectivity_sweep_phase1_mixed_diam

echo "Phase 1 — mixed-diameter rect+wave (target 5.7 µm / off 14 µm)"
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
export OPTIMIZER=adam_fd
export TARGET_DIAMETER_UM=5.7
export OFFTARGET_DIAMETER_UM=14.0
export N_OPT_RECT=30 N_OPT_WAVE=50
export T_STOP=2.0
export JAXLEY_FIBERS_SOFT_TEMPERATURE=0.1
export JAXLEY_FIBERS_OUTPUT_DIR=\"${JAXLEY_FIBERS_OUTPUT_DIR}\"
bash \"${PROJECT_ROOT}/slurm/setup_env.sh\"
python -u experiments_v2/selectivity_sweep.py
"
