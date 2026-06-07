#!/bin/bash
#SBATCH -J jaxfibers-phase3-${FIBER_MODEL:-mrg}
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

# ──────────────────────────────────────────────────────────────────────
# CROSS-MODEL SELECTIVITY SWEEP — TEMPLATE (requires refactor first)
# ──────────────────────────────────────────────────────────────────────
#
# Status: NOT YET RUNNABLE for Sundt / Sweeney / Rattay.
#
# Why this file exists:
#   The manuscript flags the across-model selectivity violin (MRG @
#   10 µm, Sweeney @ 10 µm, Sundt @ 1 µm, Rattay @ 1 µm in one
#   figure) as the natural next step.  This template documents the
#   intended SLURM submission shape.
#
# What works TODAY:
#   FIBER_MODEL=mrg with any FIBER_DIAMETER_UM.  Use
#   slurm/run_phase3_mrg10um.sbatch for the second MRG diameter
#   (D=10 µm) — that one is a known-good, runnable file.
#
# What is BLOCKED on a codebase refactor:
#   FIBER_MODEL ∈ {sundt, sweeney, rattay}.  experiments_v2/
#   selectivity_sweep.py is currently hardcoded to MRG via:
#     line 80:  from jaxfibers.fibers.mrg import section_centers_um
#     line 226: section_centers_um(geoms[0])
#   And jaxfibers/stim/multichannel_field.precompute_ve_unit constructs
#   MRG geometry internally (no model parameter exposed).
#
# Refactor scope (~1-2 day task):
#   1. Add `model` parameter to make_hussain_style_nerve OR pass model
#      name through a separate construction step.
#   2. Replace the hardcoded MRG import in selectivity_sweep.py with a
#      dispatch on FIBER_MODEL: {"mrg", "sundt", "sweeney", "rattay"}
#      that imports the right section_centers_um and right
#      build_geometry helper.
#   3. Extend precompute_ve_unit to accept a model parameter and
#      build the right per-fiber compartmental structure.
#   4. Adjust default T_STOP per model (unmyelinated CV is ~5 m/s vs
#      MRG ~60 m/s; the existing 3 ms default is too short for C-fibre
#      AP to traverse the fibre — use 8-10 ms for Sundt / Rattay).
#   5. Adjust default amp_init / amp_clip per model (C-fibre threshold
#      is ~20 mA at the 1 mm source distance, vs ~0.2 mA for MRG).
#   6. Adjust per-fascicle fibre count (currently 25; reasonable for
#      MRG but may be too many for unmyelinated given memory).
#
# Once the refactor lands, this template is meant to be invoked as:
#   FIBER_MODEL=sundt   FIBER_DIAMETER_UM=1.0  sbatch run_phase3_crossmodel_TEMPLATE.sbatch
#   FIBER_MODEL=sweeney FIBER_DIAMETER_UM=10.0 sbatch run_phase3_crossmodel_TEMPLATE.sbatch
#   FIBER_MODEL=rattay  FIBER_DIAMETER_UM=1.0  sbatch run_phase3_crossmodel_TEMPLATE.sbatch
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

if [ "${FIBER_MODEL:-mrg}" != "mrg" ]; then
  echo "ERROR: FIBER_MODEL=${FIBER_MODEL} not yet supported."
  echo "       selectivity_sweep.py is currently hardcoded to MRG."
  echo "       See the refactor scope at the top of this file."
  echo "       For MRG @ a different diameter, use run_phase3_mrg10um.sbatch."
  exit 2
fi

# (Body identical to run_phase3_manuscript_sweep.sbatch from here on.)
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
export WAVE_LR=5e-4
export WAVE_PATIENCE=20
export FIBER_MODEL="${FIBER_MODEL:-mrg}"
export FIBER_DIAMETER_UM="${FIBER_DIAMETER_UM:-5.7}"
export JAXLEY_FIBERS_OUTPUT_DIR="outputs/selectivity_sweep_phase3_${FIBER_MODEL}_${FIBER_DIAMETER_UM}um"

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
export FIBER_MODEL=\"${FIBER_MODEL}\" FIBER_DIAMETER_UM=\"${FIBER_DIAMETER_UM}\"
export JAXLEY_FIBERS_OUTPUT_DIR=\"${JAXLEY_FIBERS_OUTPUT_DIR}\"
bash \"${PROJECT_ROOT}/slurm/setup_env.sh\"
python -u experiments_v2/selectivity_sweep.py
"
