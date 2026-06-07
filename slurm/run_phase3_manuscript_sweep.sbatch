#!/bin/bash
#SBATCH -J jaxfibers-phase3-sweep
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

# Phase 3 — Manuscript sweep.  100 seeds sequential, Adam-FD rect +
# waveform, single-diameter Hussain-style anatomy.  Produces the JSONs
# and figures the paper's violin / loss / activation-map plots
# consume.
#
# Resume-friendly: FORCE_RERUN unset, so a re-submit after walltime
# expires picks up at whichever seed comes next.

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
export SEED_END="${SEED_END:-100}"
export SEEDS_PER_TASK=1
export OPTIMIZER=adam_fd
export N_OPT_RECT=30 N_OPT_WAVE=100
# Wave Adam tuned for warm-start (u_init = rect_amps × pulse_mask).
# Previous run with lr=5e-3: SI=0.97 at iter 0 → SI=0.00 at iter 99 on
# 5/7 seeds (Adam overshoots, lands in "fire everything").  10× smaller
# lr + early-stop fixes the overshoot.  Wave packaging now reports
# best-iter SI, so even on a remaining bad seed we keep the warm-start.
export WAVE_LR=5e-4
export WAVE_PATIENCE=20
export JAXLEY_FIBERS_OUTPUT_DIR=outputs/selectivity_sweep_phase3_manuscript

echo "Phase 3 — manuscript sweep, seeds ${SEED_START}..$((SEED_END-1))"
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
export JAXLEY_FIBERS_OUTPUT_DIR=\"${JAXLEY_FIBERS_OUTPUT_DIR}\"
bash \"${PROJECT_ROOT}/slurm/setup_env.sh\"
python -u experiments_v2/selectivity_sweep.py
"
