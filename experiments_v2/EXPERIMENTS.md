# Experiment strategy — jaxon v2

## Goal

Validate a differentiable, GPU-accelerated JAX reimplementation of two canonical
peripheral nerve fiber models against the reference PyFibers/NEURON implementation,
then demonstrate the computational scaling advantage on GPU.

---

## Models

| Model | Type | Diameter range | Compartments | dt |
|-------|------|---------------|--------------|-----|
| MRG (McIntyre-Richardson-Grill 2002) | Myelinated A-fiber | 5.7–16.0 µm | 21 nodes (double-cable) | 0.005 ms |
| Sundt (Sundt 2015) | Unmyelinated C-fiber | 0.3–1.2 µm | 51 uniform | 0.005 ms |

Both models are simulated with a point-source extracellular electrode (1 mm, σ=0.3 S/m, 37°C).

---

## Scripts

### `scaling.py` — Computational scaling benchmark

Measures wall-clock time for N independent fibers (N = 1, 10, 100, 1000, 10 000) via:

- **PyFibers** (NEURON, CPU, serial loop) — reference baseline
- **Jaxley** (CPU, `jax.vmap`) — batched JAX on CPU
- **Jaxley** (GPU, `jax.vmap`) — batched JAX on GPU (skipped if no GPU)

Outputs `outputs/scaling/`:
- `fig_scaling.png` — log-log scaling comparison per model
- `data_scaling.json` — all timing data

### `mrg_validation.py` — MRG comprehensive validation

Three tasks for the MRG double-cable coupled (Vi, Vpax) solver:

1. **Traces** (D=10 µm): intracellular (Jaxley bwd_euler) and extracellular (coupled solver) Vm + gating traces versus NEURON
2. **Strength-duration curves**: 9 diameters × 8 pulse shapes × 6 pulse widths; JAX bisection vs PyFibers threshold search
3. **Conduction velocity**: all 9 diameters, self-contained (threshold found inline)

Pulse shapes: mono_c, mono_a, bi_ca, bi_ac, sine, sawtooth, exp, gaussian

Outputs `outputs/mrg_validation/`:
- `data_mrg_traces.json`, `data_mrg_sd.json`, `data_mrg_cv.json`
- `fig_mrg_traces.png` — 2×2: intra Vm, gates, extra Vm, placeholder
- `fig_mrg_sd_curves.png` — 2×4: one subplot per pulse shape, all diameters
- `fig_mrg_analysis.png` — 2×3: correlation R², APE vs diameter, error vs stim type, violin by group, CV comparison, CV error

### `sundt_validation.py` — Sundt C-fiber comprehensive validation

Identical structure to `mrg_validation.py`, adapted for the Sundt unmyelinated model:

1. **Traces** (D=0.8 µm): intracellular and extracellular Vm + gating (m, h: Na; n, l: K)
2. **Strength-duration curves**: 5 diameters × 8 pulse shapes × 6 pulse widths  
   (longer PWs: 0.05–2.0 ms; C-fibers need longer pulses)
3. **Conduction velocity**: all 5 diameters

Outputs `outputs/sundt_validation/`:
- `data_sundt_traces.json`, `data_sundt_sd.json`, `data_sundt_cv.json`
- `fig_sundt_traces.png`, `fig_sundt_sd_curves.png`, `fig_sundt_analysis.png`

### `utils.py` — Shared utilities

- `PULSES` dict (8 pulse shapes) with bisection bounds
- `make_pulse_array(key, pw, n_steps, dt, delay)` — normalised pulse arrays
- `pulse_array_to_callable(arr, dt)` — PyFibers-compatible scipy interpolant
- `jax_bisect(run_fn, pulse_arr, lo, hi)` — JAX bisection for threshold
- `pf_find_threshold(...)` — PyFibers wrapper for arbitrary waveforms
- `save_json / ensure_dir` — I/O helpers

---

## Pulse-shape sign convention

All normalised pulse arrays use **+1 at the cathodic phase, −1 at the anodic phase**.
The stimulation amplitude (mA) carries the polarity:

| Pulse | lo bound | hi bound |
|-------|---------|---------|
| mono_c | −5.0 mA | −0.001 mA |
| mono_a | +5.0 mA | +0.001 mA |
| bi_ca, bi_ac, sine, sawtooth, exp, gaussian | −5.0 mA | −0.001 mA |

JAX bisection and PyFibers threshold search both respect this convention.

---

## JAX coupled solver

The extracellular simulations use `extracellular_coupled.integrate`, a custom
backward-Euler solver for the (Vi, Vpax) double-cable system. For Sundt, the same
solver is used with `is_node=True` everywhere, which collapses the periaxonal space
to the trivial constraint Vp = Ve (equivalent to a single-cable equation).

Key implementation detail: `cm_uF_cm2` must be set to `CM_AXON` (MRG) or `CM`
(Sundt) via `dataclasses.replace(geom, ...)` before calling `arrays_from_geometry`.

---

## Slurm cluster execution

Scripts run unchanged on the MedUni Vienna HPC cluster (A16 GPU, 16 GB VRAM).
Submit from the project root after copying files with scp:

```bash
sbatch slurm/run_mrg_validation.sbatch
sbatch slurm/run_sundt_validation.sbatch
sbatch slurm/run_scaling.sbatch
# or submit all at once:
bash slurm/submit_all.sh
```

Results are written to `outputs/` in the project root (mounted inside the container).

---

## Expected accuracy targets

| Metric | MRG target | Sundt target |
|--------|-----------|-------------|
| Threshold error (SD curves) | < 2% median | < 3% median |
| Correlation R² (all shapes/diameters) | > 0.999 | > 0.998 |
| Conduction velocity error | < 5% | < 5% |
| Intracellular peak Vm error | < 1 mV | < 1 mV |

These targets are based on the 0.0% error achieved for D=10 µm MRG in the coupled
solver unit test (`extracellular_coupled`).

---

## Duke FEM selectivity sweep — `selectivity_sweep_duke.py`

Runs probe + Adam-FD optimization on the Duke microCT vagus nerve cohort
(swine + human, ~1000 fibers/nerve, 12-contact cuff, MRG 5.7 µm).
Target fascicles selected by peripheral cluster paradigm (90° window).

### Standard run (h100, 2 shards, 3 swine nerves)

```bash
sbatch --array=0-1 \
  --qos=h100 --gres=gpu:h100:1 \
  --export=ALL,PULSE_SHAPE=biphasic_asym,PW_MS=0.5,ASYM_RATIO=4.0,T_STOP=5.0 \
  slurm/run_duke_sweep.sbatch
```

### Full cohort run (a100, 3 shards)

```bash
sbatch --array=0-2 \
  --qos=a100 --gres=gpu:a100:1 \
  --export=ALL,PULSE_SHAPE=biphasic_asym,PW_MS=0.5,ASYM_RATIO=4.0,T_STOP=5.0 \
  slurm/run_duke_sweep.sbatch
```

### Sparse fiber sampling sweep (manuscript Fig: SI vs sampling density)

Quantifies the SI penalty from centroid-only sampling (1 fiber/fascicle, as in
Pelot/Grill 2024 Nat Comms) vs the full ~1000-fiber population.  Runs optimization
at: centroid, 1/fasc, 3/fasc, 10/fasc, dense reference — all from the same FEM
data (no FEM re-run needed, pure row subsampling of Ve_unit).

```bash
sbatch --array=0-2 \
  --qos=a100 --gres=gpu:a100:1 \
  --export=ALL,PULSE_SHAPE=biphasic_asym,PW_MS=0.5,ASYM_RATIO=4.0,T_STOP=5.0,\
SPARSE_SAMPLING_SWEEP=true,SPARSE_N_PER_FASCICLE_LIST=1,3,10 \
  slurm/run_duke_sweep.sbatch
```

Outputs per seed: `outputs/duke_sweeps/<sample>/sparse_sampling_seed_NNNN.json`
with fields: `strategy`, `n_per_fascicle`, `n_fibers`, `n_target_fibers`, `si`, `wall_s`.

To process results into a figure-ready CSV:
```python
# pseudo-code — adapt to make_figures.py
import json, glob, pandas as pd
rows = []
for f in glob.glob("outputs/duke_sweeps/*/sparse_sampling_seed_0000.json"):
    d = json.load(open(f))
    for r in d["results"]:
        rows.append({"sample": d["sample"], **r, "dense_n_fibers": d["dense_n_fibers"]})
df = pd.DataFrame(rows)
df.to_csv("outputs/sparse_sampling_summary.csv", index=False)
```
