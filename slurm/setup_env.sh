#!/bin/bash
# Install project dependencies inside the NVIDIA container.
# Called at the start of every Slurm job from within the container.
#
# Requires PROJECT_ROOT to be set (done by each .sbatch script).
set -euo pipefail

echo "[setup] Python: $(python -V)"
echo "[setup] CUDA devices visible to JAX:"
python -c "import jax; print(jax.devices())" 2>/dev/null || echo "  (JAX not yet installed)"

echo "[setup] Installing pip dependencies from requirements_gpu.txt ..."
pip install --upgrade pip --quiet
# Clear container-level pip constraints (NVIDIA containers pin numpy to 1.26.4)
unset PIP_CONSTRAINT
pip install --upgrade --ignore-installed numpy
pip install --upgrade -r "${PROJECT_ROOT}/requirements_gpu.txt"

echo "[setup] Compiling PyFibers NMODL mechanisms (creates x86_64/ in project root) ..."
cd "${PROJECT_ROOT}"
pyfibers_compile

echo "[setup] JAX devices after install:"
python -c "import jax; print(jax.devices())"

# Enable 64-bit floats in JAX (required for double-precision membrane dynamics)
export JAX_ENABLE_X64=1
echo "[setup] JAX_ENABLE_X64=${JAX_ENABLE_X64}"
echo "[setup] Environment ready."
