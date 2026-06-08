#!/bin/bash
#SBATCH -J jaxfibers-duke-sweep
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -q a16
#SBATCH --gres=gpu:a16:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 23:59:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Duke FEM nerve selectivity sweep.
#
# Iterates over every Duke FEM bundle in DUKE_VES_ROOT (default:
# duke_Ves/) and runs the same SEED_START..SEED_END random-divider
# selectivity sweep that the synthetic phase-3 sbatch runs, but with
# the geometry + Ve_unit loaded from the FEM bundle via
# experiments_v2/selectivity_sweep_duke.py.  Each sample's outputs
# land in outputs/selectivity_sweep_duke_<sub>_<sam>/, parallel to
# the synthetic outputs/selectivity_sweep_phase3_*/ directories, so
# the figure code can find them.
#
# Memory: 997 fibres × 12 contacts × 221 comp × 600 t-steps is ~5×
# the synthetic per-seed footprint.  --mem=96G + a16 (12 GB VRAM)
# fits comfortably under Adam-FD; LBFGS will OOM on a16, override
# --gres to a100/h100 if you really want LBFGS.
#
# Resume-friendly: per-seed JSONs are written incrementally; re-submit
# after walltime expiry picks up the missing seeds.

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

# Sweep knobs (env-overrideable on sbatch submission).
export SEED_START="${SEED_START:-0}"
export SEED_END="${SEED_END:-25}"
export FIBER_DIAMETER_UM="${FIBER_DIAMETER_UM:-5.7}"
export N_NODES="${N_NODES:-21}"
export RECT_OPTIMIZER="${RECT_OPTIMIZER:-adam_fd}"
export N_OPT_RECT="${N_OPT_RECT:-30}"
export N_OPT_WAVE="${N_OPT_WAVE:-100}"
export WAVE_LR="${WAVE_LR:-5e-4}"
export WAVE_PATIENCE="${WAVE_PATIENCE:-20}"
# Duke-specific amp scaling.  FEM Ve peaks at ~870 mV/mA on this
# bundle (~5× the synthetic point-source); single-contact thresholds
# are correspondingly ~5× smaller.  Defaults below were tuned on a
# 100-fibre CPU smoketest: at ±0.30 mA / lr 0.015 the optimiser
# overshot a good warm-start and landed at the all-fire attractor.
# All env-overrideable on submission.
export AMP_INIT_MA="${AMP_INIT_MA:--0.30}"
export AMP_CLIP_LO="${AMP_CLIP_LO:--1.50}"
export AMP_CLIP_HI="${AMP_CLIP_HI:-1.50}"
export ADAM_LR_MA="${ADAM_LR_MA:-0.015}"
export FD_EPS_MA="${FD_EPS_MA:-0.010}"

# Where the Duke FEM bundles live.  Override on resubmit to point at a
# different staging directory if the data moves.
DUKE_VES_ROOT="${DUKE_VES_ROOT:-${PROJECT_ROOT}/duke_Ves}"

echo "Duke sweep: seeds ${SEED_START}..$((SEED_END-1))  D=${FIBER_DIAMETER_UM} µm"
echo "DUKE_VES_ROOT=${DUKE_VES_ROOT}"
echo "Optimiser: ${RECT_OPTIMIZER}  (init=${AMP_INIT_MA}, clip=[${AMP_CLIP_LO}, ${AMP_CLIP_HI}], lr=${ADAM_LR_MA}, fd_eps=${FD_EPS_MA})"
echo "Samples discovered:"
ls -d "${DUKE_VES_ROOT}"/*/ 2>/dev/null | xargs -n1 basename | sed "s/^/  /"

# Iterate over every sample directory under DUKE_VES_ROOT.  A sample
# directory is recognised by the presence of paths_Ve.npz + nerve_xsec.json.
for SAMPLE_DIR in "${DUKE_VES_ROOT}"/*/; do
  if [ ! -f "${SAMPLE_DIR}/paths_Ve.npz" ] || [ ! -f "${SAMPLE_DIR}/nerve_xsec.json" ]; then
    echo "[skip] ${SAMPLE_DIR} — missing paths_Ve.npz or nerve_xsec.json"
    continue
  fi
  SAMPLE_NAME=$(basename "${SAMPLE_DIR%/}")
  echo
  echo "============================================================"
  echo "Sample: ${SAMPLE_NAME}"
  echo "============================================================"
  srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
    --container-workdir="${PROJECT_ROOT}" \
    bash -lc "
set -euo pipefail
export PROJECT_ROOT=\"${PROJECT_ROOT}\"
export JAX_ENABLE_X64=1
export SKIP_PYFIBERS_COMPILE=1
export DUKE_SAMPLE_DIR=\"${SAMPLE_DIR%/}\"
export SEED_START=\"${SEED_START}\"
export SEED_END=\"${SEED_END}\"
export FIBER_DIAMETER_UM=\"${FIBER_DIAMETER_UM}\"
export N_NODES=\"${N_NODES}\"
export RECT_OPTIMIZER=\"${RECT_OPTIMIZER}\"
export N_OPT_RECT=\"${N_OPT_RECT}\" N_OPT_WAVE=\"${N_OPT_WAVE}\"
export WAVE_LR=\"${WAVE_LR}\" WAVE_PATIENCE=\"${WAVE_PATIENCE}\"
export AMP_INIT_MA=\"${AMP_INIT_MA}\"
export AMP_CLIP_LO=\"${AMP_CLIP_LO}\" AMP_CLIP_HI=\"${AMP_CLIP_HI}\"
export ADAM_LR_MA=\"${ADAM_LR_MA}\" FD_EPS_MA=\"${FD_EPS_MA}\"
bash \"${PROJECT_ROOT}/slurm/setup_env.sh\"
python -u experiments_v2/selectivity_sweep_duke.py
"
done

echo
echo "All Duke samples processed."
