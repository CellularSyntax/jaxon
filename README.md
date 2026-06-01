# jaxley_fibers

Migration of the **MRG** myelinated peripheral-nerve-fiber model from **PyFibers**
(NEURON / NMODL) into **Jaxley** (JAX / Python), with three experiments
demonstrating biophysical equivalence, scaling, and gradient-based
stimulation-waveform optimization.

See [migration_plan.md](migration_plan.md) for the full state tracker, design
decisions, and accuracy caveats.

## Quick start

### First-time install (any OS)

```bash
# 1. clone / copy this directory
cd jaxley_fibers

# 2. create the env (CPU-only by default; see environment.yml comments for CUDA)
conda env create -f environment.yml

# 3. activate and compile the NEURON .mod files for this host
conda activate jaxley_fibers
pyfibers_compile
```

For CUDA (Linux / Windows with NVIDIA GPU), edit `environment.yml` per its
header comments **before** step 2 — uncomment the `jax[cuda12]` line and
comment out the `jax[cpu]` line. Then:

```bash
python -c "import jax; print(jax.devices())"
# expect to see at least one CudaDevice
```

### Run the experiments

```bash
conda activate jaxley_fibers
cd jaxley_fibers

python experiments/exp_1_validation.py     # fig1 + table1
python experiments/exp_2_scaling.py        # fig2 (GPU line auto-included if CUDA visible)
python experiments/exp_3_optimization.py   # fig3
```

All outputs land in `outputs/`.

### Moving the project to another host

The repo is git-clean (`.gitignore` excludes compiled `.dylib` / `.so` / `.dll`
NEURON mechanisms and Python caches). To transfer:

```bash
# option A: git
git init && git add . && git commit -m "snapshot"
git push <your remote>

# option B: zip
cd ..; zip -r jaxley_fibers.zip jaxley_fibers/ -x '*/__pycache__/*' '*/arm64/*' '*/x86_64/*'
```

On the new host, after `conda env create -f environment.yml`, you **must**
re-run `pyfibers_compile` — the compiled `.mod` artefacts are
platform-specific and intentionally not transferred.

## Package layout

```
jaxley_fibers/
├── README.md                 (this file)
├── migration_plan.md         (state tracker + design notes + caveats)
├── jaxfibers/                (the new pure-Python/JAX package)
│   ├── channels/
│   │   └── mrg_axnode.py     (AXNODE_myel.mod → differentiable Jaxley Channel)
│   ├── fibers/
│   │   └── mrg.py            (MRG morphology builder, returns a jaxley.Cell)
│   ├── stim/
│   │   ├── intracellular.py  (rectangular intracellular pulse helper)
│   │   └── extracellular.py  (point-source Ve + activating-function injection)
│   └── nrn_baseline.py       (thin PyFibers wrapper for side-by-side runs)
├── experiments/
│   ├── exp_1_validation.py   → outputs/fig1_validation.png + table1_thresholds.csv
│   ├── exp_2_scaling.py      → outputs/fig2_scaling.png
│   ├── exp_3_optimization.py → outputs/fig3_optimization.png
│   ├── _smoke_channel.py     (verify Jaxley rate functions vs NEURON to 1e-7)
│   ├── _smoke_bisection.py   (sanity-check the JIT-cached threshold bisection)
│   ├── _smoke_data_stim.py   (sanity-check data_stimulate inside JIT)
│   └── _smoke_node_only_act.py (a failed hypothesis kept for the record)
└── reference_code/pyfibers/  (upstream PyFibers, cloned for reference; not modified)
```

## What lives where

* **Channel translation** (`jaxfibers/channels/mrg_axnode.py`) translates the
  NMODL mechanism `AXNODE_myel.mod` — Fast Na (m³h), Persistent Na (mp³),
  Slow K (s), and a leak — into one Jaxley `Channel` class with pure
  `jax.numpy` rate functions. The five `vtrap*` helpers preserve the
  small-denominator and ±150 mV asymptote guards via `jnp.where`.
  Cross-validated against NEURON to ~1 × 10⁻⁷ relative error
  (see `_smoke_channel.py`).

* **Fiber morphology** (`jaxfibers/fibers/mrg.py`) reconstructs the
  MRG_DISCRETE geometry: a repeating period of
  `node — MYSA — FLUT — STIN×6 — FLUT — MYSA` (11 sections per period).
  The myelin shell is **lumped** into the compartment-level capacitance and
  conductance via the series formulas
    `1/C_eff = 1/C_axon + 1/C_myelin`,
    `1/g_eff = 1/g_axon + 1/g_myelin`,
  with `C_myelin = mycm/(2·nl)` and `g_myelin = mygm/(2·nl)`.
  This is the standard single-cable approximation of NEURON's
  two-layer `extracellular` mechanism and is the dominant source of
  morphology-level error in the threshold table (see caveats).

* **Stimulation** (`jaxfibers/stim/`):
  * Intracellular pulses go directly through `cell.branch(0).comp(i).stimulate(...)`.
  * Extracellular stimulation is implemented as the classical Rattay
    *activating function*: compute the static spatial profile of V_ext at
    each compartment center (point-source formula) and inject the
    Kirchhoff sum of axial currents driven by V_ext as a time-varying
    intracellular current at each compartment.

* **PyFibers baseline** (`jaxfibers/nrn_baseline.py`) is a thin wrapper
  around `pyfibers.build_fiber` + `IntraStim` / `ScaledStim` for the
  side-by-side comparisons.

## Headline results

* **Channel rates match NEURON to ~1 × 10⁻⁷** across v ∈ [-120, +60] mV.
  See `experiments/_smoke_channel.py`.

* **Vm and gate traces** at the mid node overlay reasonably for both intra-
  and extracellular stimuli ([outputs/fig1_validation.png](outputs/fig1_validation.png)).
  Jaxley peaks are slightly higher than NEURON's due to single-cable lumping.

* **Extracellular threshold** ([outputs/table1_thresholds.csv](outputs/table1_thresholds.csv)):

  | d (µm) | PyFibers (mA) | Jaxley (mA) | |err| |
  | ------ | ------------- | ----------- | ---- |
  | 5.7    | -0.312        | -0.201      | 35.5 % |
  | 10.0   | -0.183        | -0.138      | 24.7 % |
  | 14.0   | -0.159        | -0.127      | 20.3 % |

  This 20-35 % gap is the *expected* cost of the single-cable myelin
  approximation. The brief asked for < 0.01 % which is not achievable across
  two different cable formulations; achieving it would require implementing
  NEURON's full two-layer `extracellular` mechanism in Jaxley.

* **Scaling** ([outputs/fig2_scaling.png](outputs/fig2_scaling.png)):
  Jaxley `vmap` on CPU is ~65× faster than PyFibers' serial loop at N = 100
  identical fibers, with the gap widening as N grows (Jaxley = 0.31 s at
  N = 1000 vs PyFibers extrapolated ~32 s).
  No CUDA GPU on this host (M1 Max); jax-metal is not installed, so the
  "GPU" line in the figure is annotated N/A.

* **Gradient-based waveform optimization** ([outputs/fig3_optimization.png](outputs/fig3_optimization.png)):
  Adam through a `jax.jit`-compiled `jx.integrate` call backpropagates
  through every NMODL rate function and every cable-solve step. The
  optimizer reduces injected energy until the AP-threshold constraint binds,
  then rides the energy/AP-margin Pareto front.

## Caveats

1. **Single-cable myelin lumping.** Jaxley does not have NEURON's two-layer
   `extracellular` mechanism, so the myelin sheath and periaxonal space are
   folded into the compartment-level Cm and g_pas via series formulas. This
   reduces extracellular threshold accuracy by ~20-35 % (see table above).
   The channel-level translation is exact.

2. **CPU only on this host (M1 Max).** No CUDA GPU is available, and
   `jax-metal` is not installed in the env — `fig2_scaling.png` annotates
   that line as N/A. On a CUDA host the Jaxley line is expected to drop
   another ~1-2 orders of magnitude at N = 10⁴.

3. **Tigerholm (unmyelinated C-fiber)** translation is *not* included.
   Tigerholm uses 11 NMODL mechanisms with shared Na/K ion-concentration
   pools — roughly 10× the MRG translation work — and was scoped out for
   this session. Future work.

## Reproducibility

The verified stack is pinned in [`environment.yml`](environment.yml) and
[`requirements.txt`](requirements.txt):

```
python       3.10
jax          0.6.2  (jax[cpu] on Apple silicon; jax[cuda12] on CUDA hosts)
jaxley       0.13.0
neuron       9.0.1
pyfibers     0.8.5
optax        0.2.8
numpy        2.2.6
matplotlib   3.10.9
pandas       2.3.3
```

Dev host where the v1 figures in `outputs/` were produced: Apple M1 Max,
32 GB RAM, macOS 24.6. From v2 onward (see [`migration_plan.md`](migration_plan.md))
the project moves to a CUDA host for the headline experiments.
