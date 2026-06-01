# jaxley_fibers — Publication Roadmap v3

Project root: `c:\Users\MaxHaberbusch\DATA\Postdoc\2026\Projects\jaxley_fibers`
v1 created: 2026-05-31 (PyFibers → Jaxley migration sprint)
v2 reframed: 2026-05-31 (NComms selectivity demo focus)
v3 reframed: 2026-05-31 (methods paper: differentiable biophysical nerve simulation)
Target venue: Nature Communications or similar high-impact methods journal
Target submission: ~2026-09-30

Live state tracker — check off tasks in-place as they are completed.

---

## 0. Publication narrative

**Core claim the paper must support:**

> We provide a framework for fully differentiable, biophysically detailed nerve
> fiber simulations with extracellular stimulation, enabling gradient-based
> optimization and scalable population studies while preserving the original
> mechanistic models.

**What this paper is NOT:** another axon modeling package.

**What it IS:** a demonstration that porting established NEURON biophysics into
JAX unlocks four capabilities that are difficult or impossible in conventional
NEURON-based workflows:

1. **Full differentiability** — exact gradients through the cable solve and
   all channel dynamics via JAX autodiff.
2. **GPU acceleration** — vmap over fiber populations gives orders-of-magnitude
   speedup on commodity GPUs with no code changes.
3. **Large-scale population simulation** — thousands of fibers in seconds,
   enabling recruitment landscapes and uncertainty quantification.
4. **Gradient-based stimulation optimization** — optimize waveforms, electrode
   amplitudes, and selectivity objectives via gradient descent rather than
   brute-force search.

**Every design and priority decision must serve one or more of these four
claims.** Do not implement additional models or features that do not advance
at least one of them.

**Priority order:** validation → differentiability → performance → optimization
demos → realistic nerve case study. Do not move to the next priority until the
current one is solid enough to support a paper claim.

---

## 1. Completed work

### v1 sprint (2026-05-31)
- [x] Channel translation `AXNODE_myel.mod` → `jaxfibers.channels.AxnodeMyel`
      (gate dynamics verified to 1e-7 relative error vs NEURON)
- [x] MRG morphology builder `jaxfibers.fibers.mrg` (single-cable, N-node,
      all discrete MRG diameters 5.7–16.0 µm)
- [x] PyFibers baseline wrapper `jaxfibers.nrn_baseline`
- [x] Intracellular + extracellular stim helpers `jaxfibers.stim.*`
- [x] `exp_2_scaling.py` → `outputs/fig2_scaling.png`
      (~65× CPU speedup at N=100 fibers, intracellular stim, preliminary)
- [x] `exp_performance_benchmark.py` → `outputs/fig_performance_benchmark.png`
      (extracellular coupled solver, MRG + Sundt, CPU vmap vs PyFibers serial)
      MRG D=10µm:  PyFibers ~340 ms/fiber → JAX ~17 ms/fiber at N=200 → ~20× speedup
      Sundt D=0.8µm: PyFibers ~297 ms/fiber → JAX ~8 ms/fiber at N=1000 → ~40× speedup
      JIT compile ≤18s one-time cost (N=1000 MRG); inference runtime scales sub-linearly
- [x] `exp_3_optimization.py` → `outputs/fig3_optimization.png`
      (gradient-based waveform optimization, non-trivial head-and-tail shape;
      needs threshold starting point updated to coupled-solver value −0.183 mA)

### v2 sprint (2026-05-31)
- [x] **Double-cable (Vi, Vpax) coupled solver** — `mrg_extracellular_coupled.py`.
      Block-tridiagonal 2×2 backward-Euler, exact analog of NEURON
      `nlayer=1 extracellular`. Critical impl detail: patch `cm_uF_cm2 = CM_AXON`
      for all compartments before `arrays_from_geometry`.
- [x] **MRG threshold validation, all diameters** — < 0.25 % error vs
      PyFibers/NEURON across D = 5.7–16.0 µm (9 discrete diameters).
      Outputs: `outputs/fig_mrg_comparison.png`, `outputs/jax_thresholds.json`,
      `outputs/pyfibers_thresholds.json`.
- [x] **`exp_1_validation.py` regenerated** with coupled solver.
      Four-panel figure: intracellular Vm + gates (Jaxley vs NEURON),
      extracellular Vm + gates (coupled solver vs NEURON, near-perfect overlay).
      Threshold table for D = 5.7 / 10.0 / 14.0 µm from JSON.
      Outputs: `outputs/fig1_validation.png`, `outputs/table1_thresholds.csv`.
- [x] `integrate_recording()` added to `mrg_extracellular_coupled.py`
      (returns Vm + gate traces at center node; used by exp_1).
- [x] `nrn_baseline.py` updated: `passive_end_nodes=False` default, callable
      waveforms (FutureWarning silenced).

---

## 2. Phase 1 — Model migration and validation

**Goal:** demonstrate that JAX implementations faithfully reproduce established
NEURON models across stimulus conditions, fiber parameters, and model types.
This phase supports paper Claim 1 (biophysical fidelity) and produces the
data for paper Figures 1–2 and Table 1.

**Acceptance criterion for each model:** threshold and waveform agreement
comparable to published PyFibers/AxonML validation quality; CV within 5% of
published values.

### Task 1 — MRG validation suite (COMPLETE ✓ 2026-06-01)

Full reproducible benchmark comparing JAX coupled (Vi, Vpax) solver vs PyFibers/NEURON.
All runs: point source 1 mm, σ = 0.3 S/m, dt = 0.005 ms, N = 21 nodes, T = 37 °C.

- [x] **1.1** SD curves: D=5.7/10.0/14.0 µm, PW=0.02–1.0 ms — max error 0.22 %
- [x] **1.2** CV: all 9 MRG diameters (26.7–109.1 m/s) — error 0.0 % for all
- [x] **1.3** Biphasic threshold (D=10 µm, PW=0.05–0.2 ms) — max 0.07 %
- [x] **1.4** AP propagation check: 21/21 nodes fired at 1.3× threshold — PASS ✓
- [x] **1.5** `experiments/exp_mrg_validation_suite.py` →
      `outputs/fig_mrg_validation.png`, `outputs/table_mrg_validation.csv`

**Key implementation note (CV):** symmetric k1=N//4, k2=3N//4 placement gives
div/0 (simultaneous AP arrival). Fixed: measure centre node (fires first) vs
near-end node k=2 (same side).

### Task 2 — Sundt model (COMPLETE ✓ 2026-06-01)

Sundt (2015) unmyelinated C-fiber. MRG 2×2 coupled solver reused with all
`is_node=True` — collapses to exact single-cable equation.

- [x] **2.1** `jaxfibers/channels/sundt_channels.py` — `SundtAxon` (nahh + borgkdr + pas).
      Resting gate steady states match NEURON to machine precision (< 6e-17).
- [x] **2.2** `jaxfibers/fibers/sundt.py` — `SundtGeometry` + `build_sundt()`.
- [x] **2.3** MRG coupled solver verified to handle Sundt geometry (all-node case).
- [x] **2.4** Validation suite — all checks pass:
      - Resting state: 0.00e+00 error on m, h, n; 5.6e-17 on l
      - AP peak: |diff| = 0.001 mV
      - SD curves (PW=0.02–1.0 ms, ext. 1 mm): max error 0.38 %
      - CV: 0.504 vs 0.504 m/s — 0.0 % error
- [x] **2.5** `experiments/exp_sundt_validation.py` →
      `outputs/fig_sundt_traces.png` (A: intra Vm, B: intra gates, C: extra Vm, D: extra gates),
      `outputs/fig_sundt_validation.png` (A: SD curves, B: CV bar, C: propagation, D: error summary),
      `outputs/table_sundt_validation.csv`, `outputs/data_sundt_threshold.json`,
      `outputs/data_sundt_cv.json`
      Extra Vm peak: JAX=11.154 mV vs NEURON=11.264 mV (|diff|=0.110 mV)

**Parameters (PyFibers-verified):** d=0.8 µm, δz=8.333 µm, Ra=100 Ω·cm,
Cm=1 µF/cm², ena=50 mV, ek=-90 mV, e_pas=-60 mV, gnabar=0.04, gkdrbar=0.04,
g_pas=1e-4 S/cm². Threshold at PW=0.1 ms ≈ -41 mA (1 mm, σ=0.3 S/m; high
because thin fiber at 1 mm depth). CV ≈ 0.5 m/s at 37 °C.

### Task 3 — Tigerholm model

Port Tigerholm C-fiber model from PyFibers (11 NMODL mechanisms + Na⁺/K⁺
ion-concentration state pools).

- [ ] **3.1** Translate each mechanism to a Jaxley `Channel`. Ion-concentration
      pools must be threaded as per-compartment state variables (no in-place
      mutation; all state updates must be pure functions for autodiff).
- [ ] **3.2** Verify resting state (all currents sum to zero — match PyFibers
      `balance()` check).
- [ ] **3.3** Validation suite: AP waveform, CV, threshold, extracellular
      responses, ion concentration dynamics.
      Acceptance: same criteria as Tasks 1–2.
- [ ] **3.4** `experiments/exp_tigerholm_validation.py` →
      `outputs/fig_tigerholm_validation.png`,
      `outputs/table_tigerholm_validation.csv`

> **Note on Tigerholm complexity:** shared ion pools across 11 mechanisms
> make this the most challenging port. Do NOT start until Tasks 1 and 2 are
> complete — the patterns established there will carry over and the validation
> framework will be in place.

### Milestone A

Checklist before proceeding to Phase 2:
- [ ] MRG Task 1 fully complete (all sub-tasks 1.1–1.5)
- [ ] Sundt Task 2 fully complete
- [ ] Tigerholm Task 3 fully complete

**Paper claim enabled:** *"The framework supports myelinated (MRG, Sundt) and
unmyelinated (Tigerholm C-fiber) biophysical axon models, reproducing
established NEURON/PyFibers behavior within numerical tolerance."*

---

## 3. Phase 2 — Unified validation framework

**Goal:** consolidate Task 1–3 outputs into a single publication-quality
validation section. Produces paper Figure 2 and Table 1.

- [ ] **P2.1** Single script `experiments/exp_validation_figure.py` that reads
      pre-computed benchmark CSVs and generates:
      - Panel A: threshold comparison (all models, monophasic 0.1 ms)
      - Panel B: strength-duration curves (MRG D=10 µm, Sundt, Tigerholm)
      - Panel C: AP waveform overlays (center node, JAX vs NEURON)
      - Panel D: CV comparison (MRG diameter sweep)
      - Panel E: error distributions (threshold % error, CV % error)
      → `outputs/fig2_validation.png`

- [ ] **P2.2** Table 1 — validation metrics per model:
      columns: Model | Threshold error (%) | CV error (%) | AP peak error (mV)
      | Runtime per trial (ms) | NEURON reference
      → `outputs/table1_validation.csv`

> **Note:** `outputs/fig_mrg_comparison.png` and `outputs/fig1_validation.png`
> from v2 are precursors to this figure. When P2.1 is complete, rename/supersede.

---

## 4. Phase 3 — Differentiability infrastructure

**Goal:** demonstrate that exact gradients through the full biophysical
simulation are correct and useful. Supports paper Claim 1 (differentiability)
and produces paper Figure 3.

### Task 4 — Autodiff gradient validation (COMPLETE ✓ 2026-06-01)

- [x] **4.1** Amplitude gradient `dV_peak/d(amp)` via `jax.grad` vs central FD
      (h=1e-5 mA) across 20 sub-threshold + 12 suprathreshold amplitudes.
      Median |rel err| = 0.0000%, max = 0.0049%.
- [x] **4.2** Waveform bin gradient (K=20 bins, 0.1 ms pulse) via `jax.grad`
      vs FD (2K=40 forward passes). Median |rel err| = 0.0000%, max = 0.0001%.
      Autodiff 499 ms flat; FD 7589 ms (15× faster at K=20).
- [x] **4.2b** Runtime scaling K=[1,5,10,20,50,100,200]: autodiff flat ~500 ms;
      FD linear O(K) → 147× speedup at K=200. Autodiff breaks even at K≈2.
- [x] **4.3** Electrode height gradient `dV_peak/d(height)` at 6 heights
      (500–3000 µm). Max |rel err| = 0.0012%. JAX-native `Ve_from_height_jax`
      using `jnp.sqrt(dy²+dz²)` in point-source formula.
- [x] `experiments/exp_autodiff_validation.py` →
      `outputs/fig_autodiff_validation.png`, `outputs/data_autodiff_validation.json`

### Task 5 — Differentiable objectives (COMPLETE ✓ 2026-06-01)

- [x] **5.1** `jaxfibers/objectives.py` — five pure-JAX differentiable objectives:
      - `max_vm(trace)` — peak Vm at center node; gradient non-zero below/above threshold
      - `activation_prob(trace, v_thresh=-20, sigma=2)` — sigmoid of V_peak; gradient
        requires σ ≥ 10 mV (sub-threshold EPSPs at −60 mV saturate narrow sigmoid)
      - `recruitment_fraction(population_traces, v_thresh, sigma)` — mean activation_prob
        over N fibers; gradient = mean of per-fiber gradients
      - `selectivity(target_traces, off_traces, v_thresh, sigma)` — r_target − r_off
      - `energy(waveform, dt)` — ∫I²dt; gradient = 2·amp·n_active·dt (analytical match)
- [x] **5.2** Unit tests — all 9 pass:
      - All 5 objectives: all 20 gradients finite (no NaN/Inf)
      - act_prob (σ=10 mV): max |grad| = 1.237 >> 1e-3
      - recruitment (σ=10 mV): max |grad| = 0.713 >> 1e-4
      - selectivity (σ=10 mV): max |grad| = 0.697 >> 1e-4
      - energy: gradient matches 2·amp·n_active·dt to 3.56e-16 (machine precision)
- [x] `experiments/exp_objectives_validation.py` →
      `outputs/fig_objectives_validation.png`, `outputs/data_objectives_validation.json`

**Key design note:** σ < 5 mV causes vanishing gradients because sub-threshold EPSPs
(V_peak ≈ −60 mV) are far outside the sigmoid transition at V_thresh = −20 mV.
Use σ ≥ 10 mV for effective gradient-based optimization across the threshold region.
Population geometry: target 1 mm / off-target 2 mm (thr_on = −0.183, thr_off = −0.574 mA).

### Milestone B (COMPLETE ✓ 2026-06-01)

- [x] Task 4 complete
- [x] Task 5 complete

**Paper claim enabled:** *"Exact gradients can be computed through the full
biophysical simulation — including cable solve, channel dynamics, and
extracellular field coupling — with O(1) cost w.r.t. parameter count via
reverse-mode autodiff."*

---

## 5. Phase 4 — Performance and scaling

**Goal:** quantify speedup vs conventional NEURON workflows. Produces paper
Figure 4 and Table 2. Requires a CUDA host for the GPU lines.

### Task 6 — Single-fiber benchmarks

Compare PyFibers/NEURON CPU vs JAX CPU vs JAX GPU for one MRG fiber,
as a function of simulation duration and dt.

- [ ] **6.1** Runtime vs tstop (1, 5, 10, 50 ms) at fixed dt=0.005 ms
- [ ] **6.2** Runtime vs dt (0.001, 0.005, 0.01, 0.05 ms) at fixed tstop=5 ms
- [ ] **6.3** Report JIT compile time separately from amortized runtime

> Current state: `exp_2_scaling.py` has a preliminary intracellular benchmark
> (CPU only). That file needs to be updated to include extracellular stim and
> GPU lines when CUDA host is available.

### Task 7 — Population-scale benchmarks

Benchmark JAX vmap over N independent fibers (MRG D=10 µm, extracellular
stim, dt=0.005 ms, tstop=5 ms).

- [ ] **7.1** N = 1, 10, 100, 1 000, 10 000, 100 000 (if memory allows)
      on JAX CPU and JAX GPU
- [ ] **7.2** PyFibers serial baseline up to N where wall-clock < 120 s;
      extrapolate beyond that on log-log fit
- [ ] **7.3** Report: wall-clock time, throughput (fibers/s), GPU memory used
- [ ] **7.4** Repeat for Sundt and Tigerholm (after Phase 1 is complete)
- [ ] **7.5** Publication figure: log-log scaling plot, all three backends,
      annotated GPU memory line → `outputs/fig4_scaling.png`
      Publication table: Table 2, all benchmarks → `outputs/table2_benchmarks.csv`

### Milestone C

- [ ] Tasks 6 and 7 complete (GPU lines require CUDA host access)

**Paper claim enabled:** *"Population-scale simulations of thousands of
biophysically detailed fibers become practical in seconds on commodity GPUs,
enabling Monte Carlo recruitment studies and Bayesian inference at population
scale."*

> **Blocking dependency:** CUDA host access. Plan: request time on lab/institute
> GPU node or cloud (Lambda/RunPod). All CPU benchmarks can be done on current host.

---

## 6. Phase 5 — Optimization demonstrations

**Goal:** show capabilities that are difficult or impossible in NEURON-based
workflows. Highest publication impact after validation. Produces paper
Figures 5–7.

### Task 8 — Gradient-based threshold estimation

Demonstrate that gradient descent recovers the activation threshold in far
fewer forward evaluations than bisection or brute-force.

- [ ] **8.1** Continuous threshold objective: `loss = activation_prob(amp) - 0.5`
      (find amp where sigmoid crosses 0.5). Optimize with gradient descent.
      Compare to bisection in: evaluations to converge, final error.
- [ ] **8.2** Parameter sensitivity: `jax.grad(threshold, wrt=electrode_position)`
      — show gradient correctly predicts threshold change for ±Δy electrode offset.
      Validate against finite-difference ground truth.
- [ ] **8.3** Figure: convergence curves, gradient vs FD scatter
      → `outputs/fig5_threshold_optimization.png`

### Task 9 — Selective recruitment optimization

- [ ] **9.1** Population setup: N_target = 20 MRG fibers (D = 10 µm, target
      fascicle), N_off = 20 MRG fibers (D = 14 µm, off-target fascicle).
      Extracellular field: point source, position optimized.
      Objective: maximize recruitment of target, suppress off-target,
      penalize energy.
- [ ] **9.2** Optimize pulse amplitude via gradient descent. Compare against:
      grid search, Bayesian optimization (scikit-optimize or botorch).
      Metrics: convergence speed (evaluations), final objective value.
- [ ] **9.3** Mixed-model off-target: replace large MRG off-target with
      Tigerholm C-fibers (after Phase 1 complete). Show optimizer finds
      different strategy.
- [ ] **9.4** Figure: Pareto front (target recruitment % vs off-target %),
      optimizer trajectories → `outputs/fig6_recruitment_optimization.png`

### Task 10 — Waveform optimization

- [x] **10.1** `experiments/exp_optimization_waveform.py` — K=40 bins × 50 µs over 2 ms window.
      loss = energy + 1e4·hinge² + 8e-3·L1(θ).
      Adam (lr=5e-3), 600 epochs.  Initial: flat 1.2×thr across all 40 bins (E=0.097 mA²·ms).
      Result: 7.9× energy reduction — optimizer deactivates last 21/40 bins (only first 0.95 ms
      of the 2 ms window is needed for AP initiation); 19 active bins at ~−0.113 mA.
      V_peak = 30.4 mV throughout (hinge = 0, AP constraint satisfied).
      Compared to 0.1ms reference (E=0.003354 mA²·ms): optimized is 3.65× higher (longer pulse
      but lower amplitude; L1 + hinge did not fully converge to the 0.1ms optimum).
      `outputs/fig_optimization_waveform.png`, `outputs/data_optimization_waveform.json`
      **Note:** `exp_3_optimization.py` (Jaxley intracellular, old solver) is superseded by
      this script. Threshold discrepancy (−0.170 → −0.183 mA) resolved by using coupled solver.

- [x] **10.1b** `experiments/exp_optimization_fourier.py` — charge-balanced Fourier basis, K=10.
      No DC term → 20 parameters (cos_k + sin_k, k=1..10); ∫I dt = 0 enforced by construction.
      loss = energy + 1e4·hinge²; no L1.
      Adam (lr=2e-3), 800 epochs.  Initial: k=1 sine at 3×|thr_mA| = 0.549 mA (E=0.302 mA²·ms).
      **Best valid theta tracked during run (epoch 351):**
        - Pure k=1 sine at 0.5 kHz, amplitude = 0.0819 mA — all other 9 harmonics exactly zero
        - I(t) = 0.0819·sin(2π·0.5kHz·t): anodic first half (0–1ms), cathodic second half (1–2ms)
        - AP fires via cathodic second half; V_peak = 21.4 mV (just above threshold)
      **Result: 44.3× energy reduction → 0.00682 mA²·ms = 2.03× reference 0.1ms pulse**
      `outputs/fig_optimization_fourier.png`, `outputs/data_optimization_fourier.json`
      **Key Adam pitfall:** optimizer bounces around threshold boundary; best solution appears
      mid-run (epoch 351), not at end (epoch 799 energy = 11× reference). Always track best
      supra-threshold theta via `if vpeak > V_THRESH and e < best_energy_valid: save theta`.
      **Physical interpretation:** pure sinusoidal charge-balanced waveform rediscovered from
      gradient descent alone; 0.5 kHz fundamental is the longest period fitting the 2ms window.
- [ ] **10.2** Multi-fiber waveform optimization: same K-bin waveform applied
      to a mixed-diameter population. Optimize for selective recruitment.
- [ ] **10.3** Show how waveform shape changes as population changes
      (diameter distribution shift). Demonstrates the gradient's responsiveness.
- [ ] **10.4** Figure: initial vs optimized waveform, recruitment change,
      loss curve → `outputs/fig7_waveform_optimization.png`

### Milestone D

- [ ] Tasks 8, 9, 10 complete

**Paper claim enabled:** *"Gradient descent converges to neurostimulation
strategies (waveform shape, current amplitude, selectivity) in orders of
magnitude fewer forward evaluations than heuristic search, exploiting
differentiability through the full biophysical cable model."*

---

## 7. Phase 6 — Realistic nerve demonstration

**Goal:** end-to-end case study tying all capabilities together. Produces
paper Figure 8. This is the centerpiece figure; its scope depends on time
and available FEM data.

- [ ] **6.1** Define a realistic nerve cross-section geometry (e.g., 2–3
      fascicles, MRG + Tigerholm fiber populations per fascicle).
- [ ] **6.2** Extracellular field source: either point sources (fast) or
      imported from FEniCS/COMSOL Ve(x,y,z) grid (preferred for realism).
      The parallel VAGUSPEC project provides natural FEM data here.
- [ ] **6.3** Optimize a multi-contact cuff electrode (N_contacts ≥ 4)
      for fascicle-selective stimulation via gradient descent. Compare
      against monopolar and grid-search baselines.
- [ ] **6.4** Show non-intuitive multi-polar current pattern and selectivity
      metric improvement.
- [ ] **6.5** Figure: nerve anatomy schematic, optimized current pattern,
      recruitment maps, selectivity metrics.
      → `outputs/fig8_realistic_nerve.png`

> **Blocking dependencies:** Phase 1 (Tigerholm), Phase 5 (Task 9 scaffolding),
> FEM data or point-source approximation decision.

---

## 8. Figure and table plan

| Output | Content | Status |
|--------|---------|--------|
| Fig 1 | Framework overview + differentiable pipeline diagram | Not started |
| Fig 2 | MRG / Sundt / Tigerholm validation vs NEURON | Partial (MRG thresholds + traces done; SD curves, CV, Sundt, Tigerholm pending) |
| Fig 3 | Autodiff vs finite-difference gradient validation | Not started |
| Fig 4 | Performance scaling: CPU vs GPU, N = 1–100k fibers | Partial (CPU intracellular only; extracellular + GPU pending) |
| Fig 5 | Threshold optimization via gradient descent | Not started |
| Fig 6 | Selective recruitment optimization | Not started |
| Fig 7 | Waveform optimization | Partial (exp_3_optimization.py exists; threshold needs update) |
| Fig 8 | Realistic nerve / fascicle-selective case study | Not started |
| Table 1 | Validation metrics (all models) | Partial (MRG thresholds done; full table pending) |
| Table 2 | Performance benchmarks | Partial (CPU intracellular preliminary) |

**Existing output files and their eventual figure mapping:**

| File | Maps to |
|------|---------|
| `outputs/fig_mrg_comparison.png` | Fig 2 panel A (threshold vs diameter) |
| `outputs/fig1_validation.png` | Fig 2 panels B/C (Vm + gate traces) |
| `outputs/table1_thresholds.csv` | Table 1 rows (MRG thresholds) |
| `outputs/fig2_scaling.png` | Fig 4 (preliminary; will be superseded) |
| `outputs/fig3_optimization.png` | Fig 7 (needs threshold update) |

---

## 9. Caveats and known issues

- `exp_3_optimization.py` uses old extracellular approach; threshold starting
  point should be updated from ~−0.170 mA to −0.183 mA (coupled solver,
  D=10 µm) before regenerating Fig 7.
- Tigerholm ion-concentration pools must use pure-function state threading
  (no in-place mutation) to preserve autodiff compatibility. This is the
  hardest constraint in Phase 1, Task 3.
- CUDA host required for GPU scaling lines (Phase 4). All CPU benchmarks can
  be run on current Windows host.
- Jaxley quirks that still apply:
  (a) `comp.insert()` + `comp.set()` must be in two passes.
  (b) `data_stimuli` is a 3-tuple, built via chained `data_stimulate(...)`.
- `jax-metal` is incompatible with jax ≥ 0.5 — do not attempt GPU on this
  host without a CUDA device.

---

## 10. Roadblocks / deviations log (append-only)

- 2026-05-31 v1→v2: project re-scoped from migration to NComms selectivity
  demo. Three-month sprint timeline.
- 2026-05-31: `jax-metal` 0.1.1 incompatible with jax 0.6.2 / jaxley 0.13.0.
  Uninstalled. GPU benchmarks require CUDA host.
- 2026-05-31: **Coupled solver complete.** (Vi, Vpax) backward-Euler solver
  implemented and validated. Max threshold error vs NEURON/PyFibers: 0.25 %
  across D = 5.7–16.0 µm.
- 2026-05-31: `exp_1_validation.py` regenerated with coupled solver
  (near-perfect Vm and gate trace overlay, panels C/D).
- 2026-05-31 v2→v3: paper scope refocused from selectivity demo to
  differentiable methods framework (capabilities over story). Phase structure
  replaced with 6-phase publication roadmap: validation → differentiability →
  performance → optimization demos → realistic case study.
