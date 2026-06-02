# jaxley_fibers

Differentiable, GPU-batched JAX/[Jaxley](https://github.com/jaxleyverse/jaxley)
reimplementation of six published peripheral-nerve fiber models, paired with a
custom backward-Euler coupled (V_i, V_pax) solver that reproduces NEURON's
`extracellular` mechanism to <0.5 % threshold error and machine-precision
conduction velocity. The framework enables gradient-based optimisation of
extracellular stimulation parameters — selectivity, waveform shape, and
electrode position — directly through the cable equation.

> **State of the project:** see [AUDIT.md](AUDIT.md) for the full status report,
> known gaps, and roadmap toward a Nature Communications follow-up to
> Hussain et al. *Nat. Commun.* 15:7597 (2024).
> See [CHANGELOG.md](CHANGELOG.md) for chronological changes.

## Models

| Model      | Type                        | Compartments        | Validation status     |
|------------|-----------------------------|---------------------|-----------------------|
| MRG        | Myelinated A-fiber (2002)   | 21 nodes (double-cable) | ✓ <0.5 % thresholds, 1e-12 CV |
| MRG_Interp | MRG, polynomial geometry    | same                | scaling-only          |
| Sweeney    | Myelinated A-fiber (1987)   | 21 nodes            | ✓ <0.1 % thresholds, 1e-12 CV |
| Sundt      | Unmyelinated C-fiber (2015) | 51 uniform          | ✓ <0.5 % thresholds, CV fixed |
| Rattay     | Unmyelinated HH C-fiber     | 51 uniform          | ✓ <0.5 % thresholds, CV fixed |
| Schild94   | Unmyelinated C-fiber (full Ca dynamics) | 51 uniform | code complete; validation pending |
| Schild97   | Unmyelinated C-fiber (mean) | 51 uniform          | code complete; validation pending |

## Quick start

### Install

```bash
# 1. enter the project directory
cd jaxley_fibers

# 2. create the env (CPU by default; uncomment jax[cuda12] in environment.yml for GPU)
conda env create -f environment.yml

# 3. activate and compile NEURON .mod files
conda activate jaxley_fibers
pyfibers_compile
```

For CUDA hosts, edit `environment.yml` per its header comments before step 2,
then verify:

```bash
python -c "import jax; print(jax.devices())"   # should list CudaDevice
```

### Run the validation suite

```bash
conda activate jaxley_fibers

python experiments_v2/mrg_validation.py        # thresholds, CV, traces (MRG)
python experiments_v2/sundt_validation.py      # ditto, Sundt
python experiments_v2/rattay_validation.py     # ditto, Rattay
python experiments_v2/sweeney_validation.py    # ditto, Sweeney
python experiments_v2/scaling.py               # scaling benchmark, all models
python experiments_v2/selectivity_demo.py      # local selectivity demo
python experiments_v2/selectivity_sweep.py     # full population sweep
python experiments_v2/selectivity_joint_opt.py # joint amps + electrode positions
```

Outputs land in `outputs/<experiment>/`. Heavier sweeps are wired for SLURM
under `slurm/` — see `slurm/submit_all.sh`.

### One-time cluster setup: cache the container as SquashFS

The sbatch jobs run inside `nvcr.io#nvidia/pytorch:25.03-py3` via Pyxis.
A fresh pull of this ~10 GB image on a cold node cache takes 5–10 min and
counts against `SLURM_STEP_LAUNCH_TIMEOUT` (we already bump that to 600 s,
but pulls can still slow down every job).

Pull it once to a SquashFS in your home directory and every subsequent
job starts in seconds, regardless of which node SLURM assigns:

```bash
mkdir -p $HOME/containers

# One-time pull — adjust QOS / GRES to whatever you have access to.
# Note: --mem=128G is intentional. The fetch is small, but mksquashfs
# (which builds the .sqsh file from extracted layers) can peak at
# 60-100 GB of RAM with default parallel compression. Requesting 16 G
# causes OOM-kill at the "Creating squashfs filesystem..." step.
# Throttle with ENROOT_MAX_PROCESSORS=2 if you want to use less memory.
srun --partition=gpu --qos=a16 --gres=gpu:a16:1 \
     --cpus-per-task=4 --mem=128G -t 0:30:00 \
     --container-image=nvcr.io#nvidia/pytorch:25.03-py3 \
     --container-save=$HOME/containers/pytorch_25.03.sqsh \
     true
```

After the file exists at `$HOME/containers/pytorch_25.03.sqsh` all sbatch
files in `slurm/` auto-detect and use it (the resolution logic at the top
of each script: explicit `CONTAINER_IMAGE` env-var override > local
SquashFS > nvcr.io fallback). No code change needed; nothing breaks if
you skip this step.

## Package layout

```
jaxley_fibers/
├── AUDIT.md                       state-of-the-project + Nat Comms roadmap
├── CHANGELOG.md                   chronological changes
├── README.md                      this file
├── environment.yml                pinned conda env
├── jaxfibers/                     core package
│   ├── channels/                  Jaxley Channel translations (MRG, Sundt,
│   │                              Rattay, Sweeney, Schild94, Schild97)
│   ├── fibers/                    morphology builders, one per model
│   ├── stim/
│   │   ├── intracellular.py       rectangular intracellular pulse helper
│   │   ├── extracellular.py       point-source Ve profile (V/mA at each node)
│   │   ├── extracellular_coupled.py
│   │   │                          coupled (V_i, V_pax) backward-Euler solver
│   │   │                          with 2×2 block-Thomas sweep
│   │   ├── batch_solve.py         vmapped multi-fiber forward pass
│   │   └── multichannel_field.py  multi-contact ring-cuff field
│   ├── optim/
│   │   ├── losses.py              activation proxy, WQ loss, WBCE, SI
│   │   └── optimizer.py           rect / waveform / joint Adam loops
│   ├── nerve/geometry.py          synthetic nerve cross-section
│   ├── objectives.py              differentiable objective helpers
│   └── nrn_baseline.py            thin PyFibers / NEURON wrappers
├── experiments_v2/                paper-relevant runs (manifest)
│   ├── utils.py                   pulse registry, JAX bisection, AP detection
│   ├── <model>_validation.py      thresholds + CV + traces, one per fiber model
│   ├── scaling.py                 PyFibers vs Jaxley CPU vs Jaxley GPU
│   ├── selectivity_*.py           selectivity optimisation experiments
│   └── debug/                     dev-time investigation and verification scripts
├── outputs/                       validation JSONs + figures + scaling data
├── slurm/                         SLURM sbatch drivers for the cluster
└── reference_code/pyfibers/       upstream PyFibers, cloned for reference
```

## Core design

* **Channels.** Each NMODL mechanism is hand-translated into a pure-JAX
  `Channel` class with `solve_gate_exponential` updates. Rate constants are
  validated against the published `.mod` formulas to machine precision.
* **Fibers.** Each builder returns a `jaxley.Cell` and an `MrgGeometry`-style
  dataclass that fully describes the per-compartment static parameters.
* **Coupled solver (`extracellular_coupled.integrate`).** Custom
  backward-Euler integrator for the 2-state (V_i, V_pax) cable equation —
  the principled equivalent of NEURON's `extracellular` mechanism, not the
  single-cable activating-function approximation. Uses a 2×2 block-Thomas
  sweep verified to 5e-11 mV against dense LU; back-substitution is via
  `jax.lax.associative_scan` for O(log n) backward depth.
* **Batching.** `stack_fiber_statics` packs N fibers into vmapped arrays;
  `batch_integrate_m_max` runs them in parallel on the GPU for the optimisation
  loops.
* **Optimisation.** Three modes in `optim/optimizer.py`:
  * rectangular per-contact amplitudes via packed finite-difference gradient
    (K+1 configs in one forward pass);
  * arbitrary K×T waveforms via autodiff through the ODE scan (with gradient
    checkpointing);
  * joint amplitudes + electrode positions via FD over both, with the field
    recomputed differentiably each step.

## Headline numbers (post-audit, 2026-06-02)

* **MRG threshold error vs NEURON:** median +0.04 %, max 0.39 %, across
  432 cases (9 diameters × 8 pulse shapes × 6 pulse widths). 100-1000× better
  than Hussain et al.'s S-MF surrogate (2.5 % MAPE, range −11 % to +7.3 %).
* **Sweeney threshold error:** max 0.09 % across 240 cases.
* **MRG / Sweeney conduction velocity:** 1e-12 % (machine precision) across
  all diameters.
* **Scaling (vs single-core NEURON):** 30-40× CPU vmap, 60-120× GPU vmap at
  N = 1000 fibers. Headline speedup is smaller than the AxonML surrogate
  (~10⁴×); this project's value is *accuracy* and *differentiability*, not
  raw forward speed (see [AUDIT.md](AUDIT.md) §4.1).

## Caveats

1. **Extracellular field is point-source.** All current experiments use a
   point-source potential in a homogeneous medium (σ = 0.3 S/m). Real
   FEM-derived fields (e.g. from ASCENT, as used by Hussain et al.) are
   on the Phase B roadmap.
2. **Synthetic nerve geometry.** Fibers are randomly placed in a circular
   cross-section with an eccentric target fascicle. Histology-derived
   morphology (Pelot 2020 pig / 2021 human SPARC datasets) is on the
   roadmap.
3. **Schild94 / Schild97 are not yet validated.** Channel and fiber code
   exist and are wired into the scaling benchmark but the dedicated
   validation runs have not been executed on the cluster.
4. **Sundt at D = 1.2 µm gives a non-physiological CV (~19 m/s)** in the
   current validation script because the bisection lands on a
   suprathreshold pulse that fires multi-site. Reported by both JAX and
   PyFibers identically — a measurement artefact, not a model error.
   Lower the `amp = thr * 1.3` factor or use a centred intracellular
   pulse for CV measurement if this matters.

## Reproducibility

Versions pinned in `environment.yml` / `requirements.txt`:

```
python       3.11
jax          0.6.2  (jax[cpu] on local, jax[cuda12] on CUDA hosts)
jaxley       0.13.0
neuron       9.0.1
pyfibers     0.8.5
optax        0.2.8
numpy        2.2.6
matplotlib   3.10.9
pandas       2.3.3
```

Cluster validation hardware: MedUni Vienna HPC, NVIDIA A16 (16 GB VRAM),
SLURM. Cluster has no git — copy via scp from the dev host and re-run
`pyfibers_compile` after the conda env is created on the new host
(NEURON `.mod` artefacts are platform-specific and intentionally
gitignored).
