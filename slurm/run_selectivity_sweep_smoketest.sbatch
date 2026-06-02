#!/bin/bash
#SBATCH -J jaxfibers-smoke
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -q a16
#SBATCH --gres=gpu:a16:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 0:45:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Selectivity sweep — SMOKE TEST.
#
# Shrinks N_FIBERS, N_OPT_*, T_STOP via env vars so the LBFGS autodiff +
# vmap + scan graph is small enough to compile in ~3-5 min on A16 instead
# of the ~25-40 min the manuscript-scale config (N_FIBERS=200,
# N_OPT_RECT=30, N_OPT_WAVE=100, T_STOP=3.0 ms) needs.
#
# Used to verify on the cluster that:
#   - container starts cleanly
#   - JAX picks up the GPU
#   - LBFGS rect path returns a non-degenerate result (rect SI > 0)
#   - waveform Adam runs and the per-seed figures render
#
# Total wall budget: ~10 min (5 min compile + a few minutes run + figures).
# After it passes, submit run_selectivity_sweep.sbatch for the real thing.
#
# Output: outputs/selectivity_sweep/data_seed_0000.json
#         outputs/selectivity_sweep/fig_seed_0000_rect.png
#         outputs/selectivity_sweep/fig_seed_0000_wave.png

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

# Single seed only.
export SEED_START=0
export SEED_END=1
export SEEDS_PER_TASK=1

# Smoketest dimensions — keep graph small.
# Compile cost scales with (N_FIBERS × T × N_OPT_RECT × N_RESTARTS_RECT) — shrink
# each by ~3-7×.
export N_FIBERS=30                # 200 → 30 fibers
export N_OPT_RECT=10              # 30  → 10 LBFGS outer iters
export N_OPT_WAVE=20              # 100 → 20 waveform Adam iters
export N_RESTARTS_RECT=2          # keep, restart 0 is bipolar Ve-weighted init
export T_STOP=2.0                 # 3.0 → 2.0 ms (still covers PW + delay + AP)

# Pure JAX job — skip the ~10 s NMODL compile.
export SKIP_PYFIBERS_COMPILE=1

echo "Project root    : ${PROJECT_ROOT}"
echo "Container       : ${CONTAINER_IMAGE}"
echo "Smoketest dims  : N_FIBERS=${N_FIBERS} N_OPT_RECT=${N_OPT_RECT} "\
"N_OPT_WAVE=${N_OPT_WAVE} N_RESTARTS_RECT=${N_RESTARTS_RECT} T_STOP=${T_STOP}"

export SLURM_STEP_LAUNCH_TIMEOUT=600

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc "
set -euo pipefail
export PROJECT_ROOT=\"${PROJECT_ROOT}\"
export JAX_ENABLE_X64=1
export SEED_START=\"${SEED_START}\"
export SEED_END=\"${SEED_END}\"
export SEEDS_PER_TASK=\"${SEEDS_PER_TASK}\"
export N_FIBERS=\"${N_FIBERS}\"
export N_OPT_RECT=\"${N_OPT_RECT}\"
export N_OPT_WAVE=\"${N_OPT_WAVE}\"
export N_RESTARTS_RECT=\"${N_RESTARTS_RECT}\"
export T_STOP=\"${T_STOP}\"
export SKIP_PYFIBERS_COMPILE=\"${SKIP_PYFIBERS_COMPILE}\"
bash \"${PROJECT_ROOT}/slurm/setup_env.sh\"
echo '[job] Running selectivity_sweep.py smoketest ...'
python -u experiments_v2/selectivity_sweep.py
echo '[job] Smoketest done. Inspect outputs/selectivity_sweep/data_seed_0000.json + figures.'
"
