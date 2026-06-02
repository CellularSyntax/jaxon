#!/bin/bash
#SBATCH -J jaxfibers-joint-opt
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -q a16
#SBATCH --gres=gpu:a16:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 3:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Joint waveform amplitude + electrode position optimisation — VERIFICATION RUN.
#
# Single job, single seed (SEED=0 is hardcoded in selectivity_joint_opt.py).
# Used alongside run_selectivity_sweep.sbatch to verify the full optimisation
# pipeline before scaling to multiple geometries.
#
# Setup (matches Hussain 2024 methodology, scaled for jaxfibers parallel
# demonstration):
#   * N_FIBERS = 200, all at single diameter D = 5.7 µm
#     (Hussain placed 1 fiber per fascicle at 5.7 µm; we vmap 200 fibers
#     in a single synthetic nerve to showcase the parallel speed advantage)
#   * 6-contact ring cuff, 1.5 mm radius
#   * Warm-starts from rect (Adam) solution, then jointly optimises amps +
#     contact_xyz via Adam through the cable equation
#
# Walltime 3 h (was 6 h) — single-diameter LBFGS+Adam converges much faster
# than the mixed-diameter case the original 6 h budget was sized for.
#
# Cluster QOS limits (MaxJobsPerUser per QOS):
#   a16   : 4    a30 : unlimited    a100 : 4 (12/account)
#   h100  : 2    b200: 2            a100-sxm4-80gb : 4
#
# Outputs: outputs/selectivity_joint_opt/
#
# Submit from the project root:
#   cd /path/to/jaxley_fibers
#   sbatch slurm/run_selectivity_joint_opt.sbatch
#
# Scaling examples (after verification succeeds)
# ----------------------------------------------
# On a100 (~3× faster per task; comfortable for 200 fibers):
#   sbatch --qos=a100 --gres=gpu:a100:1 -t 1:00:00 \
#          slurm/run_selectivity_joint_opt.sbatch
#
# Multiple seed geometries (once verification passes — bumps SEED via env):
#   for s in 0 1 2 3 4; do
#     sbatch --export=ALL,JOINT_OPT_SEED=$s slurm/run_selectivity_joint_opt.sbatch
#   done
# (note: selectivity_joint_opt.py currently hardcodes SEED=0; add an env
# override there before running multi-seed.)

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
# Resolve the container image once at submit time:
#   1. Honour an explicit CONTAINER_IMAGE env-var override.
#   2. Else prefer the project SquashFS at $HOME/containers/jaxfibers.sqsh
#      (built once via slurm/build_container.sh; has pip deps baked in,
#      so setup_env.sh skips its pip install step).
#   3. Else fall back to nvcr.io#nvidia/pytorch:25.03-py3 (Pyxis pulls +
#      caches it on the assigned node; setup_env.sh installs deps).
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

echo "Project root : ${PROJECT_ROOT}"
echo "Container    : ${CONTAINER_IMAGE}"

# Pyxis container pulls can exceed SLURM's 32-second default step-launch
# timeout on a cold node cache (the nvcr.io PyTorch image is ~10 GB).
# Allow up to 10 minutes for srun to bring the container up.
export SLURM_STEP_LAUNCH_TIMEOUT=600

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc "
set -euo pipefail
export PROJECT_ROOT=\"${PROJECT_ROOT}\"
export JAX_ENABLE_X64=1
bash \"${PROJECT_ROOT}/slurm/setup_env.sh\"
echo '[job] Running experiments_v2/selectivity_joint_opt.py ...'
python -u experiments_v2/selectivity_joint_opt.py
echo '[job] Done. Outputs in outputs/selectivity_joint_opt/'
"
