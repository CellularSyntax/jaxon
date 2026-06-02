#!/bin/bash
#SBATCH -J jaxfibers-manuscript
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

# Manuscript-scale selectivity sweep — single sequential job.
#
# Runs three phases back-to-back on one GPU so concurrent JAX compiles
# can't fight for the same node's CPU/RAM (which is what hurt us with
# the earlier 4-job submission).
#
# Phase 1 — mixed-diameter verification (5 min, 1 seed)
#   Target = 5.7 µm sensory, off-target = 14 µm motor.  This is the
#   regime where the waveform optimisation actually beats rect (different
#   chronaxie between the two populations).  Verifies the per-side
#   diameter wiring and gives us a Fig 5-style demo for the paper.
#
# Phase 2 — LBFGS verification at full N=200 (15-20 min, 1 seed)
#   Single-diameter Hussain-style nerve; LBFGS rect via lax.map over
#   restarts (no more vmap×scan×autodiff compile pathology).  Verifies
#   the second-order optimiser hits manuscript-grade SI on the actual
#   problem size.
#
# Phase 3 — manuscript sweep (16-18 h, 100 seeds)
#   Adam-FD rect + waveform, sequentially over seeds 0..99.  Same
#   single-diameter geometry as the published headline.  Produces
#   100 × {data_seed_NNNN.json + 2 fig_seed figures} for the violin
#   plot + activation-map figures the paper will use.
#
# Outputs land in:
#   outputs/selectivity_sweep/mixed_diam/    (phase 1)
#   outputs/selectivity_sweep/lbfgs_verify/  (phase 2)
#   outputs/selectivity_sweep/manuscript/    (phase 3)
#
# Cancel anytime with `scancel <JOBID>`; phases 1-2 finish in the first
# ~25 min so even an early cancel buys you the demo figures.

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

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc "
set -euo pipefail
export PROJECT_ROOT=\"${PROJECT_ROOT}\"
export JAX_ENABLE_X64=1
export SKIP_PYFIBERS_COMPILE=1
bash \"${PROJECT_ROOT}/slurm/setup_env.sh\"

#─── Phase 1: mixed-diameter rect + waveform (Adam-FD) ──────────────────
echo
echo '================================================================'
echo '   Phase 1 — Mixed-diameter rect+wave (target 5.7 µm sensory,'
echo '   off-target 14 µm motor) — demonstrates Fig 5 chronaxie beat'
echo '================================================================'
mkdir -p outputs/selectivity_sweep_phase1_mixed_diam
export SEED_START=0 SEED_END=1 SEEDS_PER_TASK=1
export FORCE_RERUN=1
export OPTIMIZER=adam_fd
export TARGET_DIAMETER_UM=5.7
export OFFTARGET_DIAMETER_UM=14.0
export N_OPT_RECT=30 N_OPT_WAVE=50
export T_STOP=2.0
JAXLEY_FIBERS_OUTPUT_DIR=outputs/selectivity_sweep_phase1_mixed_diam \
  python -u experiments_v2/selectivity_sweep.py
unset TARGET_DIAMETER_UM OFFTARGET_DIAMETER_UM

#─── Phase 2: LBFGS rect verification at N=200 (single-diameter) ────────
echo
echo '================================================================'
echo '   Phase 2 — LBFGS rect at N=200 (sequential restarts via lax.map)'
echo '================================================================'
mkdir -p outputs/selectivity_sweep_phase2_lbfgs_verify
export OPTIMIZER=lbfgs
export N_OPT_RECT=30 N_OPT_WAVE=0
export N_RESTARTS_RECT=2
JAXLEY_FIBERS_OUTPUT_DIR=outputs/selectivity_sweep_phase2_lbfgs_verify \
  python -u experiments_v2/selectivity_sweep.py

#─── Phase 3: manuscript sweep (Adam-FD, 100 seeds sequential) ──────────
echo
echo '================================================================'
echo '   Phase 3 — Manuscript sweep: Adam-FD rect+wave, seeds 0..99'
echo '================================================================'
mkdir -p outputs/selectivity_sweep_phase3_manuscript
export OPTIMIZER=adam_fd
export N_OPT_RECT=30 N_OPT_WAVE=100
export SEED_START=0 SEED_END=100 SEEDS_PER_TASK=1
unset FORCE_RERUN   # resume-friendly: skip seeds already done
JAXLEY_FIBERS_OUTPUT_DIR=outputs/selectivity_sweep_phase3_manuscript \
  python -u experiments_v2/selectivity_sweep.py

echo
echo '================================================================'
echo '   All phases complete.  Inspect outputs/selectivity_sweep_phase*/'
echo '================================================================'
"
