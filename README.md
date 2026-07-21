<p align="center">
  <img src="docs/jaxon_logo.png" alt="jaxon" width="320">
</p>

<p align="center">
  <b>A differentiable, GPU-native simulator for peripheral-nerve fiber models.</b>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/backend-JAX%20%C2%B7%20Jaxley-orange" alt="JAX · Jaxley">
  <img src="https://img.shields.io/badge/python-3.11-3776AB" alt="Python 3.11">
  <img src="https://img.shields.io/badge/validated%20vs-NEURON%20%2F%20PyFibers-success" alt="validated vs NEURON/PyFibers">
  <img src="https://img.shields.io/badge/platform-Linux%20%C2%B7%20macOS%20%C2%B7%20Windows%20(WSL2)-lightgrey" alt="Linux · macOS · Windows (WSL2)">
</p>

---

**jaxon** is a fully differentiable, GPU-native reimplementation of the canonical
peripheral-nerve fiber models in [JAX](https://github.com/jax-ml/jax) /
[Jaxley](https://github.com/jaxleyverse/jaxley). It reproduces NEURON's
`extracellular` mechanism to sub-0.5 % threshold error and machine-precision
conduction velocity, runs whole fiber populations in parallel on one GPU, and
exposes **gradients through the cable equation** — so extracellular stimulation
parameters (per-contact amplitudes, waveform shape, electrode position) can be
optimized directly rather than grid-searched. It slots into existing pipelines
as a gradient-enabled, vectorized replacement for the NEURON forward solver.

The cohort-scale selectivity experiments run on FEM-derived per-contact lead
fields and histology-segmented vagus-nerve geometries produced by the **golgi**
platform [1, 2] (29 nerves: 18 pig, 11 human), applied to a 12-contact ring cuff.

## ✨ Highlights

- **Differentiable through the cable equation.** Reverse-mode autodiff and packed
  finite-difference gradients drive Adam optimization of per-contact amplitudes,
  arbitrary K×T waveforms, and joint amplitude + electrode-position search — no
  surrogate model, gradients of the true biophysics.
- **GPU-native population scale.** `vmap`-batched forward passes simulate
  100,000 fibers on a single GPU, reaching a geometric-mean ~820× speedup over
  single-thread NEURON at that scale (~200–300× at the ~1000-fiber operating
  point used in the paper).
- **Validated against NEURON / PyFibers.** 99.6 % of 943 threshold configurations
  agree within 1 %; conduction velocity matches to machine precision. jaxon is
  checked against the reference solver, not a fit to it.
- **Principled coupled solver.** A custom backward-Euler integrator for the
  two-state (V_i, V_pax) double-cable equation — the equivalent of NEURON's
  `extracellular` mechanism, not the single-cable activating-function shortcut —
  solved by a 2×2 block-Thomas sweep verified to 5.6×10⁻¹¹ mV against dense LU.
- **Five model/parameterization variants.** Myelinated MRG (McIntyre–Richardson–Grill,
  original + interpolated tables) and Sweeney; unmyelinated Sundt and Rattay C-fibers,
  each hand-translated from NMODL and rate-checked to machine precision.
- **Realistic cohort fields.** Drives golgi-derived FEM lead fields over
  histology-segmented swine and human vagus nerves, so selectivity is scored on
  anatomically realistic populations, not synthetic cross-sections.

## Models

| Model      | Type                                    | Compartments            | Validation status                |
|------------|-----------------------------------------|-------------------------|----------------------------------|
| MRG        | Myelinated A-fiber (2002)               | 21 nodes (double-cable) | ✓ <0.5 % thresholds, 1e-12 CV    |
| MRG_Interp | MRG, polynomial geometry                | same                    | scaling-only                     |
| Sweeney    | Myelinated A-fiber (1987)               | 21 nodes                | ✓ <0.1 % thresholds, 1e-12 CV    |
| Sundt      | Unmyelinated C-fiber (2015)             | 51 uniform              | ✓ <0.5 % thresholds, CV fixed    |
| Rattay     | Unmyelinated HH C-fiber                 | 51 uniform              | ✓ <0.5 % thresholds, CV fixed    |
| Schild94   | Unmyelinated C-fiber (full Ca dynamics) | 51 uniform              | code complete; validation pending |
| Schild97   | Unmyelinated C-fiber (mean)             | 51 uniform              | code complete; validation pending |

## Installation

jaxon installs with conda (for the pinned scientific + NEURON stack) and runs on
CPU by default; a single edit switches it to CUDA.

```bash
# 1. enter the project directory
cd jaxley_fibers

# 2. create the env (CPU by default; uncomment jax[cuda12] in environment.yml for GPU)
conda env create -f environment.yml

# 3. activate and compile the NEURON .mod files used by the PyFibers reference
conda activate jaxley_fibers
pyfibers_compile
```

For CUDA hosts, edit `environment.yml` per its header comments before step 2, then verify:

```bash
python -c "import jax; print(jax.devices())"   # should list CudaDevice
```

## Quick start

**Run the validation suite** (CPU, minutes):

```bash
conda activate jaxley_fibers

python experiments_v2/mrg_validation.py        # thresholds, CV, traces (MRG)
python experiments_v2/sundt_validation.py      # ditto, Sundt
python experiments_v2/rattay_validation.py     # ditto, Rattay
python experiments_v2/sweeney_validation.py    # ditto, Sweeney
python experiments_v2/scaling.py               # scaling benchmark, all models
```

**Optimize stimulation for selectivity:**

```bash
python experiments_v2/selectivity_demo.py      # local single-nerve selectivity demo
python experiments_v2/selectivity_sweep.py     # full population sweep
python experiments_v2/selectivity_joint_opt.py # joint amplitudes + electrode positions
```

Outputs land in `outputs/<experiment>/`. Heavier sweeps are wired for SLURM under
`slurm/` — see `slurm/submit_all.sh`. To reproduce every figure and number in the
paper from processed data, see **[REPRODUCE.md](REPRODUCE.md)**.

> **State of the project:** see [AUDIT.md](AUDIT.md) for the full status report and
> known gaps, and [CHANGELOG.md](CHANGELOG.md) for chronological changes.

## Reproducing the paper

The processed data (golgi lead fields + all sweep / validation / phenomena
outputs) is archived on Zenodo; `REPRODUCE.md` documents two paths — regenerating
every figure from that processed data (fast, CPU-only), or re-running the full
simulation pipeline from scratch (slow, GPU).

The driver reads all inputs from `outputs/` (in particular it needs
`outputs/duke_sweeps/`, the corrected cohort sweep). The Zenodo archive
unpacks to `processed_outputs/` and `lead_fields/`, so map `processed_outputs/`
onto `outputs/` before running. From the repository root:

```bash
# 1. unpack the Zenodo data archive (jaxon_data_archive_v1.tar.gz)
tar -xzf jaxon_data_archive_v1.tar.gz

# 2. expose the archive's processed_outputs/ as the repo's outputs/
ln -s "$(pwd)/processed_outputs" outputs        # symlink (recommended)
#   - or -  mv processed_outputs outputs          # move it into place

# 3. regenerate all main + named-supplementary figures from processed data
bash reproduce_figures.sh
```

`lead_fields/` is only needed for the from-scratch path (re-running the cohort
sweeps); see `REPRODUCE.md`.

The raw golgi FEM meshes are **not** bundled — they are golgi outputs, available
via the golgi platform and the underlying SPARC datasets (swine
[doi:10.26275/MAQ2-EII4](https://doi.org/10.26275/MAQ2-EII4), human
[doi:10.26275/OFJA-GHOZ](https://doi.org/10.26275/OFJA-GHOZ)).

## Core design

- **Channels.** Each NMODL mechanism is hand-translated into a pure-JAX `Channel`
  class with `solve_gate_exponential` updates; rate constants are checked against
  the published `.mod` formulas to machine precision.
- **Fibers.** Each builder returns a `jaxley.Cell` plus a geometry dataclass fully
  describing the per-compartment static parameters.
- **Coupled solver** (`stim/extracellular_coupled.py`). Backward-Euler integrator
  for the two-state (V_i, V_pax) double-cable equation, solved by a 2×2
  block-Thomas sweep (verified to 5.6e-11 mV vs dense LU); back-substitution via
  `jax.lax.associative_scan` for O(log n) depth.
- **Batching.** `stack_fiber_statics` packs N fibers into vmapped arrays;
  `batch_integrate_m_max` runs them in parallel on the GPU for the optimization loops.
- **Optimization** (`optim/optimizer.py`), three modes: rectangular per-contact
  amplitudes via a packed finite-difference gradient (K+1 configs per forward
  pass); arbitrary K×T waveforms via autodiff through the ODE scan (with gradient
  checkpointing); joint amplitudes + electrode positions via FD over both, with
  the field recomputed differentiably each step.

## Repository layout

```text
jaxley_fibers/
├── README.md               this file
├── AUDIT.md                state-of-the-project + roadmap
├── CHANGELOG.md            chronological changes
├── REPRODUCE.md            step-by-step paper reproduction
├── reproduce_figures.sh    one-command figure driver (processed data)
├── environment.yml         pinned conda env
├── jaxfibers/              core package
│   ├── channels/           Jaxley Channel translations (MRG, Sundt, Rattay,
│   │                       Sweeney, Schild94, Schild97)
│   ├── fibers/             morphology builders, one per model
│   ├── stim/
│   │   ├── intracellular.py        rectangular intracellular pulse helper
│   │   ├── extracellular.py        point-source Ve profile (demo option)
│   │   ├── extracellular_coupled.py  coupled (V_i, V_pax) backward-Euler solver
│   │   ├── batch_solve.py          vmapped multi-fiber forward pass
│   │   └── multichannel_field.py   multi-contact ring-cuff field
│   ├── optim/
│   │   ├── losses.py       activation proxy, WQ loss, WBCE, selectivity index
│   │   └── optimizer.py    rect / waveform / joint Adam loops
│   ├── nerve/geometry.py   synthetic nerve cross-section (demos/tests)
│   └── nrn_baseline.py     thin PyFibers / NEURON wrappers
├── experiments_v2/         paper-relevant runs (see EXPERIMENTS.md, FIGURE_DATA_MAP.md)
│   ├── <model>_validation.py       thresholds + CV + traces, one per fiber model
│   ├── scaling.py                  PyFibers vs Jaxley CPU vs Jaxley GPU
│   ├── selectivity_sweep_duke.py   cohort selectivity sweep on golgi fields
│   ├── figures_*_main.py           paper figure generators
│   └── analyze_*.py                statistics + tables
├── duke_Ves/               per-nerve golgi FEM lead fields + geometry (gitignored)
├── outputs/                validation JSONs + figures + sweep data (gitignored)
├── slurm/                  SLURM sbatch drivers for the cluster
└── reference_code/pyfibers/  upstream PyFibers, cloned for reference
```

## Headline numbers

- **MRG threshold error vs NEURON:** median +0.04 %, max 0.39 % across 432 cases
  (9 diameters × 8 pulse shapes × 6 pulse widths). **Sweeney:** max 0.09 % across
  240 cases. Overall, **99.6 % of 943 configurations within 1 %**.
- **MRG / Sweeney conduction velocity:** machine precision (1e-12 %) across all diameters.
- **Scaling vs single-thread NEURON:** ~200–300× at N = 1000 fibers, geometric-mean
  ~820× at N = 100,000 on one A100. The value here is *accuracy* and
  *differentiability*, not raw forward speed against fitted surrogates (see
  [AUDIT.md](AUDIT.md) §4.1).

## Caveats

1. **Extracellular field.** The cohort selectivity experiments use FEM-derived
   per-contact lead fields (the pre-baked `Ve_VperA` tensor) computed by **golgi**
   [1, 2] for a multi-contact ring cuff on histology-segmented vagus nerves — not
   a point source. The analytic point-source potential in a homogeneous medium
   (σ = 0.3 S/m, `stim/extracellular.py`) is retained as a lightweight option for
   local single-nerve demos.
2. **Nerve geometry.** The cohort experiments use histology-derived vagus-nerve
   morphology segmented and meshed by **golgi** [1, 2] from the Pelot 2020 pig /
   2021 human SPARC datasets (29 nerves: 18 pig, 11 human). The synthetic circular
   cross-section (`nerve/geometry.py`) is retained for local demos and unit tests.
3. **Schild94 / Schild97 are not yet validated.** Channel and fiber code exist and
   are wired into the scaling benchmark, but the dedicated validation runs have
   not been executed.
4. **Sundt at D = 1.2 µm gives a non-physiological CV (~19 m/s)** in the current
   validation script because the bisection lands on a suprathreshold pulse that
   fires multi-site. Reported identically by JAX and PyFibers — a measurement
   artifact, not a model error.

## Reproducibility

Versions pinned in `environment.yml` / `requirements.txt`:

```
python       3.11        jaxley       0.13.0      optax        0.2.8
jax          0.6.2       neuron       9.0.1       numpy        2.2.6
                         pyfibers     0.8.5       matplotlib   3.10.9
```

Cluster validation hardware: MedUni Vienna HPC, NVIDIA A16 (16 GB VRAM), SLURM.
NEURON `.mod` artifacts are platform-specific and gitignored — re-run
`pyfibers_compile` after creating the conda env on a new host.

## License

jaxon is released under the **MIT License** — see [LICENSE](LICENSE). The golgi
FEM datasets and the manuscript reproduction archive are released separately on
Zenodo under CC-BY-4.0.

## Citation

If you use jaxon, please cite the accompanying manuscript (Lung & Haberbusch,
Medical University of Vienna; citation to be updated on publication) and the
golgi platform that supplies the cohort fields:

1. Lung D, Jia Y, Moro A, Fachino M, Haberbusch M. *golgi: open-source software
   for automated nerve model generation and recruitment simulation.* bioRxiv
   2026.07.10.737846. https://doi.org/10.64898/2026.07.10.737846
2. Lung D, Jia Y, Blumer R, Reissig L, Zopf LM, Heimel P, Kraus C, Moro A,
   Fachino M, Haberbusch M. *golgi: an open-source graphical platform for
   image-to-recruitment modeling of peripheral nerve stimulation.* bioRxiv
   2026.07.10.737529. https://doi.org/10.64898/2026.07.10.737529

## Built with

| Role | Built on |
|---|---|
| Differentiable cable simulation | [JAX](https://github.com/jax-ml/jax) · [Jaxley](https://github.com/jaxleyverse/jaxley) |
| Optimization | [Optax](https://github.com/google-deepmind/optax) |
| Reference solver | [NEURON](https://neuron.yale.edu/) via [PyFibers](https://github.com/wmglab-duke/pyfibers) |
| Cohort FEM fields & anatomy | [golgi](https://github.com/CellularSyntax/golgi) |
