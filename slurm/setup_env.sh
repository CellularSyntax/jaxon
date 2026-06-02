#!/bin/bash
# Set up the project's Python environment inside the NVIDIA container.
# Called at the start of every Slurm job from within the container.
#
# Behaviour:
#   * If the container was built via `slurm/build_container.sh`, all pip
#     dependencies are already baked in; we detect that and skip pip.
#   * Otherwise (fresh nvcr.io pull or basic SquashFS), we run the full
#     pip install -r requirements_gpu.txt cycle as before.
#   * pyfibers_compile is always run: it compiles NMODL mechanisms into
#     $PROJECT_ROOT/x86_64/ which lives on the user's filesystem outside
#     the container, so it must run per-job (cheap, ~10 s).
#
# Requires PROJECT_ROOT to be set (done by each .sbatch script).
set -euo pipefail

echo "[setup] Python: $(python -V)"

# ── Skip pip if the project's deps are already in the container ───────────────
# Probe-import the two project-specific packages that aren't in the base
# nvcr.io image; if both are present, the container is fully provisioned.
if python -c "import jaxley, pyfibers" 2>/dev/null; then
    echo "[setup] jaxley + pyfibers already installed in container — skipping pip."
else
    echo "[setup] Installing pip dependencies from requirements_gpu.txt ..."
    pip install --upgrade pip --quiet
    # NVIDIA containers pin numpy to 1.26.4 via PIP_CONSTRAINT; clear it
    # so requirements_gpu.txt's `numpy>=2.2` wins.
    unset PIP_CONSTRAINT
    pip install --upgrade --ignore-installed numpy
    pip install --upgrade -r "${PROJECT_ROOT}/requirements_gpu.txt"
fi

echo "[setup] Compiling PyFibers NMODL mechanisms (creates x86_64/ in project root) ..."
cd "${PROJECT_ROOT}"
pyfibers_compile

echo "[setup] JAX devices:"
python -c "import jax; print(jax.devices())"

# Enable 64-bit floats in JAX (required for double-precision membrane dynamics)
export JAX_ENABLE_X64=1
echo "[setup] JAX_ENABLE_X64=${JAX_ENABLE_X64}"
echo "[setup] Environment ready."
