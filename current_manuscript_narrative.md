# Current Manuscript Narrative — JAXON (JNE target)

Last updated: 2026-06-12

---

## Central claim

Exact biophysical simulation of large peripheral nerve fiber populations on GPU is now
tractable.  JAXON reimplements five canonical PNS fiber models in JAX with no surrogate
approximation, matches PyFibers/NEURON to <0.5 % threshold error across all models, and
runs 1000× faster at 100 k fibers.  Applied to real vagus nerve microCT anatomies, it
enables population-scale selectivity optimization and — for the first time — direct
quantification of the approximation error introduced by centroid-only fiber sampling
(as used in Hussain/Pelot/Grill 2024 Nat Comms and prior work).

---

## What we are NOT showing (removed from scope)

- **Mixed-diameter fiber-type selectivity** — optimizer fails (SI=0 plateau); honest
  negative, relevant for Discussion only (one paragraph max), not a Results section.
- **Waveform shape optimization** — out of scope for this paper; amplitude-pattern
  (rectangular pulse, variable per-contact magnitude) is sufficient to make the point.
  Waveform opt can appear as a one-sentence future-work note.

---

## Paper structure

### Introduction
- VNS selectivity problem: fascicle-level targeting matters clinically
- Current computational bottleneck: PyFibers/NEURON too slow for population-scale optimization
- Prior GPU work (AxonML): surrogate + centroid-only fibers — introduces approximation errors
  that have never been systematically quantified
- JAXON: exact model, GPU-native, population scale → enables the quantification

### Methods
1. Fiber models: MRG double-cable, MRG interpolated, Sundt C-fiber, Sweeney, Rattay — all
   JAX reimplementations with batched vmap across fibers
2. Validation protocol: threshold SD curves, conduction velocity, gating variables,
   membrane potential — all vs PyFibers/NEURON reference
3. Scaling benchmark: wall-clock time, 1–100 k fibers, JAX GPU vs PyFibers CPU
4. Duke VNS cohort: swine + human microCT, ~1000 fibers/nerve, 12-contact cuff, MRG 5.7 µm
5. Selectivity optimization: smart-init (bipolar/tripolar probe sweep → Adam-FD),
   random-init baseline (same Adam-FD from random start)
6. Sparse fiber sampling: centroid (1/fascicle, AxonML-style), 1/3/10/fasc random,
   dense reference (~1000) — pure row subsampling of pre-computed Ve_unit, no FEM re-run
7. Transfer evaluation: take amplitudes optimized under sparse model, evaluate activation
   on dense (~1000-fiber) model without re-optimizing — quantifies the "deployment gap"
8. Statistical testing: Wilcoxon signed-rank (paired per nerve); FDR correction across
   pairwise sparsity comparisons

### Results

#### §3.1 — Validation (DONE ✅)
- All 5 models vs PyFibers: threshold error <0.5 %, R²≥0.99999 across all pulse shapes
- Conduction velocity error <5 % across all diameters
- Gating variable and membrane potential traces match visually and quantitatively
- Key message: JAXON is numerically equivalent to the NEURON reference

#### §3.2 — Computational scaling (DONE ✅)
- Wall-clock time vs N fibers (1 → 100 k): log-log plot, three lines (PyFibers, JAX CPU, JAX GPU)
- ~1000× speedup at 100 k fibers on GPU vs PyFibers CPU
- Key message: population-scale simulation is now tractable

#### §3.3 — Selectivity optimization on Duke cohort (IN FLIGHT 🔄)
- Full cohort: all available swine + human Duke nerves, 4 random peripheral cluster seeds each
- Metric: achievable SI per nerve per seed, smart-init vs random-init
- Statistical test: does smart-init consistently outperform random-init?
- Key message: the optimizer reliably finds high-SI configurations; smart-init improves
  speed/reliability over random starting points
- **Status:** 3 swine nerves done, full cohort job running on h100 cluster

#### §3.4 — Sparse sampling comparison (CODE DONE, AWAITING RESULTS 🔄)
- For each nerve × seed: optimize at centroid, 1/fasc, 3/fasc, 10/fasc, dense reference
- Primary metric: SI achieved at each sparsity level (reported as SI of the sparse model)
- Statistical test: paired Wilcoxon across sparsity levels; how much SI do you lose vs dense?
- Key message: centroid approximation systematically overestimates SI; the bias is
  consistent and statistically significant across the cohort
- **Status:** sbatch submitted with SPARSE_SAMPLING_SWEEP=true; results pending

#### §3.5 — Transfer evaluation / deployment gap (PENDING RESULTS ⏳)
- Take the amplitude vector found by sparse optimization (centroid / low-sparsity)
- Evaluate activation on the full dense model WITHOUT re-optimizing
- Report: SI_sparse_reported vs SI_dense_transfer (the true performance)
- Expected: SI_dense_transfer < SI_sparse_reported < SI_dense_optimized
- The gap SI_sparse_reported − SI_dense_transfer is the clinically relevant error
- Statistical test: is the transfer gap significant? Does it correlate with nerve anatomy?
- **Status:** waiting for §3.4 sparse sweep results first; amps_mA are saved in result JSONs
  so transfer evaluation can be done as a post-processing script without re-running cluster jobs.
  Implement only after §3.4 results confirm the SI overestimation story is worth extending.

### Discussion
- JAXON vs PyFibers/NEURON: exact equivalence + 1000× speed
- JAXON vs AxonML: exact vs surrogate; centroid vs dense; no retraining for new fiber models
- Quantified centroid approximation bias: what this means for VNS parameter programming
- Mixed-diameter selectivity: brief note — optimizer cannot exploit diameter contrast with
  current objective formulation; future work
- Limitations: single pulse width / shape; rectangular per-contact amplitudes only; FEM
  from simplified cylindrical cuff geometry
- Future work: waveform shape opt; heterogeneous populations (see followup_ideas/);
  in vivo validation of optimized configurations

---

## Experiment status summary

| Experiment | Data ready | Analysis ready | In manuscript |
|---|---|---|---|
| Model validation (all 5) | ✅ | ✅ | §3.1 |
| Scaling benchmark | ✅ | ✅ | §3.2 |
| Duke optimization (full cohort) | 🔄 running | ❌ | §3.3 |
| Sparse sampling sweep | 🔄 running | ❌ | §3.4 |
| Transfer evaluation | ⏳ after §3.4 results | ❌ | §3.5 |

---

## Manuscript files

- `manuscript/main.tex` — iopart skeleton
- `manuscript/sections/01_introduction.tex` — drafted
- `manuscript/sections/02_methods.tex` — drafted (includes waveform opt methods — **TO REMOVE**)
- `manuscript/sections/03_results.tex` — stubbed with `\needdata{}` macros
  - Mixed-diameter subsection — **TO REMOVE**
  - Waveform optimization subsection — **TO REMOVE**
- `manuscript/sections/04_discussion.tex` — drafted
- Results stubs will be filled once §3.3–§3.5 data is available

---

## Key design decisions (locked)

- **Single diameter (MRG 5.7 µm)** for all selectivity experiments — matches Duke FEM precomputed Ve_unit
- **Rectangular per-contact amplitudes** (not waveform shape) — simpler, sufficient for SI story
- **Peripheral cluster target paradigm** — 90° angular window, 4 random seeds per nerve
- **Adam-FD optimizer** — finite-difference gradient, K+1=13 parallel forward passes, cosine LR
- **Smart-init** — bipolar + tripolar probe sweep selects best starting point before Adam-FD
- **Sparse sampling** — pure row subsampling of precomputed Ve_unit; no FEM re-run needed
