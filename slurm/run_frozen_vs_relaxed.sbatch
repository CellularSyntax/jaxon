#!/bin/bash
#SBATCH -J jaxfibers-fvr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -q a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Frozen-tripole vs free-all-contacts sparsity (deployment penalty) AND the
# FD-vs-autodiff gradient-source speedup, across the FULL nerve set on the full
# fibre population.  Each worker walks the whole nerve list and atomically CLAIMS
# each nerve (see _claim in frozen_vs_relaxed.py), so any number of jobs on any
# partitions (a100 + b200 at once) divide the work without duplicating -- no
# shard-index coordination needed.  This file's #SBATCH -q/--gres are just the
# a100 default; override per partition on the sbatch command line.
#
# Launch a mix of a100 + b200 workers into outputs_new/ (claim pool shared):
#   bash slurm/submit_frozen_vs_relaxed.sh
# or by hand, e.g. 4 a100 + 2 b200:
#   for i in 1 2 3 4; do FVR_OUT_ROOT=outputs_new \
#     sbatch -q a100 --gres=gpu:a100:1 slurm/run_frozen_vs_relaxed.sbatch; done
#   for i in 1 2;     do FVR_OUT_ROOT=outputs_new \
#     sbatch -q b200 --gres=gpu:b200:1 slurm/run_frozen_vs_relaxed.sbatch; done
#
# Each task writes ${FVR_OUT_ROOT}/frozen_vs_relaxed/<cfg>/<nerve>.json.  Then:
#   FVR_OUT_ROOT=outputs_new FROZEN_VS_RELAXED_AGGREGATE=1 \
#     python -m experiments_v2.frozen_vs_relaxed
#
# Quick downsampled check:  FVR_SAMPLES=sub-10_sam-1 MAX_FIBERS=300 \
#     sbatch slurm/run_frozen_vs_relaxed.sbatch

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
DEFAULT_SQSH="${HOME}/containers/jaxfibers.sqsh"
DEFAULT_IMAGE="nvcr.io#nvidia/pytorch:25.03-py3"
if [ -z "${CONTAINER_IMAGE:-}" ]; then
  if [ -f "${DEFAULT_SQSH}" ]; then CONTAINER_IMAGE="${DEFAULT_SQSH}";
  else CONTAINER_IMAGE="${DEFAULT_IMAGE}"; fi
fi
mkdir -p "${PROJECT_ROOT}/logs"

if [ -d "${PROJECT_ROOT}/duke_meshes" ]; then
  DUKE_VES_ROOT="${DUKE_VES_ROOT:-${PROJECT_ROOT}/duke_meshes}"
else
  DUKE_VES_ROOT="${DUKE_VES_ROOT:-${PROJECT_ROOT}/duke_Ves}"
fi

# Nerve list: ALL swine + ALL human with a valid paths_Ve.npz (cap per species
# with FVR_MAX_PER_SPECIES).  Override the list entirely with FVR_SAMPLES=...
if [ -n "${FVR_SAMPLES:-}" ]; then
  IFS=',' read -r -a SAMPLES <<< "${FVR_SAMPLES}"
else
  mapfile -t SWINE < <(for d in "${DUKE_VES_ROOT}"/*/; do
    n=$(basename "${d%/}"); [[ "$n" == human_* ]] && continue
    [ -f "${d}/paths_Ve.npz" ] && echo "$n"; done | sort | head -n "${FVR_MAX_PER_SPECIES:-100000}")
  mapfile -t HUMAN < <(for d in "${DUKE_VES_ROOT}"/*/; do
    n=$(basename "${d%/}"); [[ "$n" != human_* ]] && continue
    [ -f "${d}/paths_Ve.npz" ] && echo "$n"; done | sort | head -n "${FVR_MAX_PER_SPECIES:-100000}")
  SAMPLES=("${SWINE[@]}" "${HUMAN[@]}")
fi

N_SHARDS="${N_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
echo "Frozen-vs-free: ${#SAMPLES[@]} nerves (claim-distributed), shard ${SHARD_INDEX}/${N_SHARDS}"
echo "DUKE_VES_ROOT=${DUKE_VES_ROOT}  MAX_FIBERS=${MAX_FIBERS:-0 (full)}  OUT=${FVR_OUT_ROOT:-outputs}"

for i in "${!SAMPLES[@]}"; do
  if [ $(( i % N_SHARDS )) -ne "${SHARD_INDEX}" ]; then continue; fi
  NERVE="${SAMPLES[$i]}"
  SAMPLE_DIR="${DUKE_VES_ROOT}/${NERVE}"
  if [ ! -f "${SAMPLE_DIR}/paths_Ve.npz" ]; then
    echo "[shard ${SHARD_INDEX}] skip ${NERVE} (no paths_Ve.npz)"; continue
  fi
  echo
  echo "=== [shard ${SHARD_INDEX}] nerve ${i}: ${NERVE} ==="
  if srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
    --container-workdir="${PROJECT_ROOT}" \
    bash -lc "
set -euo pipefail
export PROJECT_ROOT=\"${PROJECT_ROOT}\"
export JAX_ENABLE_X64=1
export SKIP_PYFIBERS_COMPILE=1
export DUKE_SAMPLE_DIR=\"${SAMPLE_DIR}\"
export FVR_STEPS=\"${FVR_STEPS:-15}\"
export FVR_N_PER_FASC=\"${FVR_N_PER_FASC:-1}\"
export FVR_VERBOSE=\"${FVR_VERBOSE:-0}\"
export FVR_BALANCE=\"${FVR_BALANCE:-1}\"
export FVR_DENSE_STEPS=\"${FVR_DENSE_STEPS:-45}\"
export FVR_MODE=\"${FVR_MODE:-speed}\"
export FVR_OUT_ROOT=\"${FVR_OUT_ROOT:-outputs}\"
export FVR_LOCK_STALE_S=\"${FVR_LOCK_STALE_S:-7200}\"
export MAX_FIBERS=\"${MAX_FIBERS:-0}\"
export FIBER_DIAMETER_UM=\"${FIBER_DIAMETER_UM:-5.7}\"
export N_NODES=\"${N_NODES:-21}\"
export T_STOP=\"${T_STOP:-3.0}\"
export DT=\"${DT:-0.005}\"
export DELAY_MS=\"${DELAY_MS:-1.0}\"
export PULSE_SHAPE=\"${PULSE_SHAPE:-biphasic_sym}\"
export PW_MS=\"${PW_MS:-1.0}\"
export ASYM_RATIO=\"${ASYM_RATIO:-4.0}\"
export AMP_INIT_MA=\"${AMP_INIT_MA:--0.08}\"
export AMP_CLIP_LO=\"${AMP_CLIP_LO:--2.00}\"
export AMP_CLIP_HI=\"${AMP_CLIP_HI:-2.00}\"
export ADAM_LR_MA=\"${ADAM_LR_MA:-0.005}\"
export FD_EPS_MA=\"${FD_EPS_MA:-0.030}\"
export PROBE_MAGS_MA=\"${PROBE_MAGS_MA:--0.02,-0.05,-0.10,-0.20,-0.40,-0.80,-1.20,-1.50}\"
export JAXLEY_FIBERS_SOFT_TEMPERATURE=\"${JAXLEY_FIBERS_SOFT_TEMPERATURE:-0.15}\"
export JAXLEY_FIBERS_ENERGY_LAMBDA=\"${JAXLEY_FIBERS_ENERGY_LAMBDA:-1e-3}\"
export JAXLEY_FIBERS_LOSS=\"${JAXLEY_FIBERS_LOSS:-linear}\"
export JAXLEY_FIBERS_WQ_END_NODES=\"${JAXLEY_FIBERS_WQ_END_NODES:-3}\"
export INIT_MODE=\"${INIT_MODE:-warm}\"
export ZERO_LR_MA=\"${ZERO_LR_MA:-0.1}\"
export ZERO_STEPS=\"${ZERO_STEPS:-200}\"
export ZERO_LR_DECAY=\"${ZERO_LR_DECAY:-0.6}\"
export ZERO_SI_FLOOR=\"${ZERO_SI_FLOOR:-0.5}\"
export ZERO_PATIENCE=\"${ZERO_PATIENCE:-10}\"
export TARGET_PARADIGM=cluster
export CLUSTER_RADIUS_QUANTILE=\"${CLUSTER_RADIUS_QUANTILE:-0.32}\"
export CLUSTER_WINDOW_DEG=\"${CLUSTER_WINDOW_DEG:-90.0}\"
export CLUSTER_N_MIN_TARGET=\"${CLUSTER_N_MIN_TARGET:-50}\"
export CLUSTER_MIN_FASCICLES=\"${CLUSTER_MIN_FASCICLES:-3}\"
export CLUSTER_N_POSITIONS=\"${CLUSTER_N_POSITIONS:-4}\"
export CLUSTER_RNG_SEED=\"${CLUSTER_RNG_SEED:-0}\"
bash \"${PROJECT_ROOT}/slurm/setup_env.sh\"
python -u -m experiments_v2.frozen_vs_relaxed
"; then
    echo "[shard ${SHARD_INDEX}] ${NERVE}: OK"
  else
    echo "[shard ${SHARD_INDEX}] ${NERVE}: FAILED rc=$? — continuing"
  fi
done

echo
echo "Shard ${SHARD_INDEX}/${N_SHARDS} done."
