#!/bin/bash
#SBATCH -J jaxfibers-scaling-fill10k
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -q a16
#SBATCH --gres=gpu:a16:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 23:59:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Scaling benchmark — re-run with a larger pyfibers budget so the
# N=10,000 cells for the slow models (Sundt, Rattay, Schild94,
# Schild97) actually complete instead of being truncated by the
# 2-hour default budget in experiments_v2/scaling.py.
#
# What changed vs the original scaling run:
#   - JAXLEY_FIBERS_PF_BUDGET_S=86400  (24 h per pyfibers chunk)
#     The previous default (7200 = 2 h) truncated pyfibers serial
#     timing at N=1000 for the slow models, leaving the N=10,000
#     cell of Table 3 blank for Sundt / Rattay / Schild94 / Schild97.
#   - 16 CPUs allocated (vs the usual 8) so the pyfibers serial path
#     benefits from a faster CPU host on the GPU node.
#
# After this run completes, outputs/scaling/data_scaling.json will
# have the missing N=10k pyfibers timings filled in; re-running
# `python manuscript/make_figures.py scaling` will regenerate the
# scaling figure with the completed table.

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

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/outputs/scaling"

# NOTE: SKIP_PYFIBERS_COMPILE is intentionally NOT set here.  Unlike
# the Phase 3 selectivity sweeps (which only call jaxon and never
# touch NEURON), the scaling benchmark compares pyfibers vs jaxon
# wall times -- so the NMODL mechanisms (axnode_myel etc.) MUST be
# compiled inside the container before python runs, otherwise NEURON
# raises 'argument not a density mechanism name' the moment pyfibers
# tries to insert the MRG node channels.
export SLURM_STEP_LAUNCH_TIMEOUT=600

# 24 h budget per pyfibers chunk - plenty for N=10k on the C-fiber
# models which extrapolate to ~30 min based on the N=1000 timings
# (Sundt ~168s, Rattay ~149s, Schild ~340s).
export JAXLEY_FIBERS_PF_BUDGET_S=86400

echo "Scaling re-run with extended pyfibers budget (24 h/chunk)"
echo "Output: outputs/scaling/data_scaling.json (overwritten)"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc "
set -euo pipefail
export PROJECT_ROOT=\"${PROJECT_ROOT}\"
export JAX_ENABLE_X64=1
# Intentionally NOT setting SKIP_PYFIBERS_COMPILE: scaling.py calls
# pyfibers which needs NEURON's axnode_myel + friends compiled.
export JAXLEY_FIBERS_PF_BUDGET_S=${JAXLEY_FIBERS_PF_BUDGET_S}
bash \"${PROJECT_ROOT}/slurm/setup_env.sh\"
python -u experiments_v2/scaling.py
"
