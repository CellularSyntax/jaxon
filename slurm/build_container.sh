#!/bin/bash
# One-time build of the project's custom Pyxis SquashFS container.
#
# Starts from nvcr.io#nvidia/pytorch:25.03-py3, runs the same pip install
# steps that setup_env.sh would have run on every job, and saves the
# result to $HOME/containers/jaxfibers.sqsh.  All sbatch files in slurm/
# auto-detect that path and prefer it over the nvcr.io reference, so
# subsequent jobs start in seconds with deps already in place.
#
# What is and is NOT baked in:
#   YES: jax[cuda12], jaxley, optax, numpy, scipy, matplotlib, pandas, tqdm,
#        neuron, pyfibers, pytest, ipykernel.
#   NO : pyfibers_compile (compiles NMODL mechanisms into $PROJECT_ROOT/x86_64/,
#        outside the container; still runs per-job via setup_env.sh — takes ~10 s).
#
# Memory note: --mem=128G is required.  mksquashfs parallel-compresses the
# extracted layers in RAM and can peak well above 60 GB; --mem=16G OOM-kills
# at the 'Creating squashfs filesystem...' step.
#
# Usage (run from project root):
#   bash slurm/build_container.sh
#
# Environment overrides (optional):
#   QOS, PARTITION, GRES   pass to srun (defaults: a16/gpu/gpu:a16:1).
#   OUT                    output path (default: $HOME/containers/jaxfibers.sqsh).
#   PROJECT_ROOT           project root mount (default: $PWD).
#   BASE_IMAGE             starting container (default: the nvcr PyTorch image).
#   TIME_LIMIT             srun -t value (default: 1:00:00).

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
OUT="${OUT:-${HOME}/containers/jaxfibers.sqsh}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io#nvidia/pytorch:25.03-py3}"
PARTITION="${PARTITION:-gpu}"
QOS="${QOS:-a16}"
GRES="${GRES:-gpu:a16:1}"
TIME_LIMIT="${TIME_LIMIT:-1:00:00}"

mkdir -p "$(dirname "${OUT}")"

if [[ -f "${OUT}" ]]; then
  echo "Already exists: ${OUT}"
  echo "Remove it first if you want to rebuild:"
  echo "  rm ${OUT}"
  exit 0
fi

if [[ ! -f "${PROJECT_ROOT}/requirements_gpu.txt" ]]; then
  echo "ERROR: requirements_gpu.txt not found in PROJECT_ROOT=${PROJECT_ROOT}" >&2
  exit 1
fi

echo "[build] Output            : ${OUT}"
echo "[build] Base image        : ${BASE_IMAGE}"
echo "[build] Project root mount: ${PROJECT_ROOT}"
echo "[build] Partition / QOS   : ${PARTITION} / ${QOS}"
echo "[build] GRES              : ${GRES}"
echo "[build] Time limit        : ${TIME_LIMIT}"
echo ""
echo "[build] Launching srun (this will pull the base image if uncached,"
echo "        run pip install -r requirements_gpu.txt inside it, and save"
echo "        the result as a SquashFS).  Expected wall time: 5-15 min."
echo ""

# Pyxis container pulls can exceed SLURM's 32-second default step-launch
# timeout on a cold node cache; allow up to 10 min for the build srun too.
export SLURM_STEP_LAUNCH_TIMEOUT=600

srun \
  --partition="${PARTITION}" \
  --qos="${QOS}" \
  --gres="${GRES}" \
  --cpus-per-task=4 \
  --mem=128G \
  -t "${TIME_LIMIT}" \
  --container-image="${BASE_IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${PROJECT_ROOT}" \
  --container-save="${OUT}" \
  bash -lc "
set -euo pipefail
echo '[build:in-container] Python: '\$(python -V)
echo '[build:in-container] Installing pip deps ...'
pip install --upgrade pip --quiet
# NVIDIA containers pin numpy to 1.26.4 via /etc/pip-constraints.txt;
# unset PIP_CONSTRAINT so requirements_gpu.txt's numpy>=2.2 wins.
unset PIP_CONSTRAINT
pip install --upgrade --ignore-installed numpy
pip install --upgrade -r requirements_gpu.txt
echo '[build:in-container] Installed packages:'
pip list 2>/dev/null | grep -iE 'jax|jaxley|neuron|pyfibers|optax|numpy|scipy|matplotlib|pandas' || true
echo '[build:in-container] Done.  Pyxis will now snapshot the layer to SquashFS ...'
"

echo ""
echo "[build] Verifying output ..."
ls -lh "${OUT}"
echo ""
echo "[build] Done.  Future sbatch jobs auto-detect and use ${OUT}."
echo "[build] To verify on the next submission, grep the .out file for the Container line:"
echo "        grep '^Container' logs/<new-job-id>.out"
