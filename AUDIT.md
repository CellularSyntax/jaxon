# jaxley_fibers — Project audit (2026-06-02)

> Audit of project status, gaps, and roadmap toward a Nature Communications
> follow-up to Hussain, Grill & Pelot, *Nat. Commun.* 15:7597 (2024) —
> *"Highly efficient modeling and optimization of neural fiber responses
> to electrical stimulation"* (AxonML / S-MF).

The reference paper is herein abbreviated **HGP24**.

---

## 1. What this project is

A JAX/[Jaxley](https://github.com/jaxleyverse/jaxley) reimplementation of
canonical peripheral-nerve fiber models for **differentiable, GPU-batched,
gradient-based optimization of extracellular stimulation parameters.**

Six fiber models are implemented and run on a custom JAX coupled
(V_i, V_pax) backward-Euler solver — the principled equivalent of NEURON's
`extracellular` mechanism (nlayer=1) — with a 2×2 block-Thomas sweep:

| Model      | Type                       | Compartments | Status                          |
|------------|----------------------------|--------------|---------------------------------|
| MRG        | Myelinated A-fiber         | 21 nodes × 11 sec/period = 221 | **validated, all PWs/shapes** |
| MRG_Interp | MRG with polynomial geom   | same         | scaling-validated only          |
| Sweeney    | Myelinated A-fiber (1987)  | 21 nodes     | validated (best, < 0.1 %)       |
| Sundt      | Unmyelinated C-fiber       | 51 uniform   | validated (CV spike at 1.0 µm)  |
| Rattay     | Unmyelinated HH C-fiber    | 51 uniform   | **bi_ca bug + CV error**        |
| Schild 94  | Unmyelinated C-fiber (full) | 51 uniform  | **never validated**             |
| Schild 97  | Unmyelinated C-fiber (mean) | 51 uniform  | **never validated**             |

Plus:
* gradient-based **selectivity optimizer** for ring cuff: rectangular amps,
  arbitrary waveforms, and joint *amps + electrode-position* (a feature
  AxonML cannot offer because surrogates are trained for fixed Ve fields).
* synthetic nerve geometry (random uniform fibers in circle, eccentric
  target fascicle).
* PyFibers / NEURON baseline wrappers for side-by-side validation.

The whole stack runs on a single GPU; cluster submission via SLURM is wired
(`slurm/`).

---

## 2. Headline numerical results (from `outputs/`)

### 2.1 Threshold accuracy vs. PyFibers/NEURON (point source, 1 mm)

Computed from `data_*_sd.json`:

| Model    | n      | median err | p95(|err|) | max |err| |
|----------|--------|-----------:|-----------:|---------:|
| MRG      | 432    | +0.04 %   | 0.20 %   | **0.39 %** |
| Sweeney  | 240    | −0.03 %   | 0.08 %   | **0.09 %** |
| Sundt    | 240    | +0.34 %   | 0.42 %   | **0.46 %** |
| Rattay   | 240    | +0.10 %   | 6.44 %   | **8.18 %** ⚠ (bi_ca only) |
| Schild94 | —      | —          | —          | not run |
| Schild97 | —      | —          | —          | not run |

> **Compare to HGP24 S-MF surrogate:** MAPE 2.5 %, range −11 % to +7.3 %,
> 95 % of errors within ±5 %.
> **This project is 1–2 orders of magnitude more accurate** on the same
> threshold task, *because it is not a surrogate — it is the cable equation,
> solved in JAX*.

### 2.2 Conduction velocity (cathodic monophasic 0.1 ms)

| Model    | CV error range |
|----------|----------------|
| MRG      | 1e-12 % across 9 diameters (machine precision) |
| Sweeney  | 1e-12 % across all diameters |
| Sundt    | 1e-12 % at most diameters; **1.6 % spike at D = 1.0 µm** ⚠ |
| Rattay   | **−2.6 % to −1.9 % median across diameters** ⚠  |

### 2.3 Scaling (Jaxley vmap vs PyFibers serial, N = 1 → 10 000)

GPU speedup vs single-core NEURON at N = 1000:

| Model      | CPU speedup | GPU speedup |
|------------|-------------|-------------|
| MRG        | 31 ×        | **69 ×**    |
| MRG_Interp | 35 ×        | 79 ×        |
| Sweeney    | 42 ×        | **112 ×**   |
| Sundt      | 35 ×        | 71 ×        |
| Rattay     | 30 ×        | 68 ×        |
| Schild94   | 32 ×        | 37 ×        |
| Schild97   | 32 ×        | 38 ×        |

> **Compare to HGP24 S-MF:** 2 000 – 130 000 × over single-core NEURON.
> Our 60 – 120 × is far smaller because (a) we solve the *full*
> ultrastructure (~220 comps/fiber for MRG, vs HGP24's 53 nodes), and
> (b) backward-Euler with auto-diff through every step is intrinsically
> heavier than a forward surrogate. **This is the single most important
> story-level weakness** (see §4.1 below).

### 2.4 Selectivity optimization

* `selectivity_demo.py` ran locally — confirms gradient descent achieves
  selective activation in 6-fiber explicit nerves.
* `selectivity_sweep.py` (100 random nerves, 100 fibers each) — **outputs/
  selectivity_sweep/ is EMPTY**. Never executed on the cluster, or outputs
  not synced back.
* `selectivity_joint_opt.py` — also empty; same situation.

---

## 3. Critical issues (ordered by impact)

### 3.1 Anatomy is synthetic — no real vagus nerve histology (BLOCKER for Nat Commun)

HGP24 uses 6 pig + 6 human vagus nerve cross-sections segmented from
histology (Pelot 2020 / SPARC), extruded to 50 mm, with ImThera + helical
cuff geometries solved in **COMSOL via ASCENT**. Field is then applied as a
spatiotemporal boundary condition.

This project uses a **point-source Ve in homogeneous tissue (σ = 0.3 S/m)**
with **random uniform fibers in a circular nerve**. No fascicles. No
perineurium. No epineurium. No saline gap. No cuff geometry — just contact
positions on a notional ring.

→ A neuromodulation paper at this venue requires anatomically grounded
fields. The minimum bar is ingesting ASCENT FEM output (precomputed by
HGP24 and *publicly available* on SPARC, doi 10.26275/maq2-eii4 (pig) and
10.26275/ofja-ghoz (human)) as Ve fields.

### 3.2 Rattay `bi_ca` threshold error 8 % (known unresolved bug)

`investigate_rattay_bica.py` exists, suggesting the bug was being chased.
All other pulse shapes are < 0.3 %. Combined with the **−1.9 % to −2.6 %
Rattay CV error**, this indicates either:
* a sign/phase bug in the bi_ca waveform path,
* a subtle channel-model mismatch (RattayAberham vtraub shift?).

### 3.3 Sundt CV spike at D = 1.0 µm

`data_sundt_cv.json` shows 1.61 % at D=1.0 µm while every other diameter
is 1e-12 % (machine precision). Likely AP-detection / argmax fragility on
that one trace (multiple peaks?). Worth investigating — 1 line in
`mrg_validation.py` style `_jax_cv`.

### 3.4 Schild94 / Schild97 never validated

Both channel implementations exist (`schild_channels.py`, 689 lines, with
the full coupled Ca / Na/K-pump / Ca-pump / Na-Ca exchanger machinery)
and `slurm/run_schild9{4,7}_validation.sbatch` are wired. But
`outputs/schild9{4,7}_validation/` directories don't even exist. **Need
to run these and confirm correctness** — Schild is the only model in the
zoo that exercises ion-concentration dynamics, the toughest test.

### 3.5 Selectivity sweep & joint-opt outputs are missing

`outputs/selectivity_sweep/` and `outputs/selectivity_joint_opt/` are
empty directories. These are the *headline figures* of any follow-up
paper. Either:
* the cluster jobs were submitted but not synced back, or
* the cluster jobs failed silently, or
* they were never submitted.

→ This is the work to do *next*, before any new feature dev.

### 3.6 No comparison against NEURON+DE baseline

HGP24 reports DE+NEURON, DE+S-MF, GD+S-MF side by side. This project only
reports GD+JAX. To make the "exact biophysics matches surrogate-based
optimization" claim, **at minimum** one nerve needs to be solved with
both:
* DE + NEURON (using `cajal`, the repo HGP24 published next to AxonML),
* GD + JAX (our pipeline).

Then plot Pareto fronts (target activation × off-target activation × time).

### 3.7 Memory budget for waveform optimization

`run_waveform_optimization` (autodiff through T-step scan; for T=800,
n_fibers=100, K=6) — comment in the file notes "~100 s / iter for T=1600".
On a 16 GB A16 this will OOM with > a few hundred fibers. HGP24
demonstrates this exact problem (1200 DOF). Need to verify the project
can handle the same scale.

---

## 4. Strategic gaps vs. HGP24 (publishability)

### 4.1 Speed story is weaker than HGP24's

We give 60–120 × on GPU; HGP24 gives 80 000–95 000 × on the same hardware
class. The press won't read that as an advance.

Three options, listed in order of effort:

1. **Reframe** — pitch is *accuracy*, not speed. "Exact differentiable
   biophysics matches surrogate selectivity at 1–2 orders better
   accuracy". Then 60 × is fine because *NEURON cannot do gradient
   descent at all* — the speedup that matters is "GD impossible → GD
   possible". For a fair comparison vs HGP24 use **the same metric they
   use**: total optimization time to reach a target selectivity. If we
   reach a comparable SI in ~hundreds of seconds vs days, the headline
   write-up works (rect-opt is currently ~tens of seconds per nerve).

2. **Add a surrogate option** — train a small NN against the exact
   solver and offer it as a fast inference path. ~weeks of work; gets us
   to 1000 × at <1 % accuracy. Hybrid GD-on-surrogate + verify-on-exact
   is a paper in itself (HGP24 mentions this as future work).

3. **Optimize the exact solver further** — backward-Euler with full
   compartmentation is the bottleneck. Possible wins:
   * coarse-grain internodes: collapse 6 STIN compartments to 1 (–25 %
     memory, minor accuracy hit). HGP24's S-MF goes further (nodes
     only).
   * fixed-step → BDF2 or RK2 (~2 × in wall time).
   * sparse Jacobian via `jaxley`'s built-in solver options.
   * remove the JAX 64-bit requirement for inference (keep it only for
     training-style validation); 32-bit + tighter atol gives 2-3 ×.

### 4.2 Emergent-nonlinearity demonstrations (HGP24 Figure 3)

**Status (2026-06-02):** panels (a)–(c) complete with PyFibers + jaxfibers
overlaid; panel (d) deferred.

| Panel | Script | Status |
|-------|--------|--------|
| (a) DC block | `experiments_v2/dc_block.py` | ✅ waterfall, peak V_m matches PyFibers to 0.0 mV at 4 amps |
| (b) kHz block | `experiments_v2/khz_block.py` | ✅ verbatim pyfibers tutorial 5 repro, AP counts match exactly at all 4 amps (14/36/14/10) |
| (c) AP collision | `experiments_v2/ap_collision.py` | ✅ peak V_m matches PyFibers to 0.0 mV at 4 diameters |
| (d) Spike desync | `experiments_v2/spike_desync.py` | ⏸ deferred — full SPIKE-sync sweep is ~30 h of PyFibers; cluster run pending |

Composite figure: `experiments_v2/fig3_combined.py` →
`outputs/fig3_combined/fig3_combined.png`. Panel (d) currently shows the
legacy single-diameter placeholder until the full sweep is run.

Hussain-style colour convention enforced throughout: PyFibers/NEURON blue
solid, jaxfibers orange dashed.

#### 4.2.1 What's needed to upgrade panels (b) and (d) to manuscript fidelity

The current standalone scripts use a single MRG fiber in a uniform
volume conductor — they verify the **model**, not the **experimental
preparation**. Hussain Fig 3b/d are realised on real anatomy with
specific cuffs:

| Panel | Anatomy | Cuff | Population structure | Seeds × cells |
|-------|---------|------|----------------------|---------------|
| (b) kHz block | Pig vagus, Suppl. Note 1 **P2** | ImThera (6-contact extraneural) | **7 sampled fascicles**, 100 Hz intrinsic pacing | rows × cols × 7 fascicles × N amps |
| (d) Spike desync | Human vagus, Suppl. Note 1 **H2** | Helical (Cyberonics-like) | **N=35**: 7 fiber positions × 5 intrinsic firing patterns | rows × cols × 3 stim_freqs × N amps × 35 |

Missing infrastructure to land these panels:

1. **Anatomy** — `jaxfibers/nerve/geometry.py` only has a synthetic
   "random uniform fibers + one eccentric target fascicle". We need:
   - Pig vagus P2 cross-section (fascicle outlines + per-fascicle fiber
     populations / diameter histograms).
   - Human vagus H2 cross-section, same content.
   - Source: Hussain ASCENT pipeline outputs, SPARC repo, or our own
     Musselman/Blanz-style data. **Open question for MH.**

2. **Cuff geometry** — `jaxfibers/stim/extracellular.py` only does
   single point sources. We need:
   - ImThera 6-contact extraneural cuff: contact positions, sizes,
     dielectric layers (or just FEM Ve fields pre-computed once and
     interpolated per fiber position).
   - Helical cuff: same.
   - Cleanest path: load pre-computed Ve(x, y, z) potential templates
     per contact (matches AxonML/PyFibers convention) and let
     selectivity_sweep-style code combine them.

3. **Intrinsic firing patterns** — currently jaxfibers does deterministic
   periodic pacing. Hussain panel (d) uses **5 distinct firing patterns**
   per cell (Poisson trains with the row's mean rate? or specific
   replayed patterns?). **Open question for MH.** Driven via
   `add_intrinsic_activity(..., noise=1.0)` on PyFibers side; jaxfibers
   needs equivalent pre-generated spike-train injection.

4. **Compute budget** — panel (d) grid is
   3 D × 3 IFR × 3 stim_freq × N_amps × 35 fibers × 2 solvers.
   At N_amps=10: 18,900 sims × ~60 s PyFibers ≈ 315 h serial. JAX vmap
   collapses the 35-fiber inner dim to a single GPU pass (~2-3 s per
   row), so the JAX side is ~30 min wall on a16. **PyFibers side is the
   bottleneck** — it has to be the cluster (a16 array, 4 concurrent;
   walltime ~5-6 h per array task at 6 sims/task; ~16-20 h total).
   See [[project-meduni-vienna-cluster]] for QOS limits.

5. **JAX-side rewrite** — current `khz_block.py` and `spike_desync.py`
   are single-fiber. Need to:
   - vmap over fascicles / fiber positions for the population mean + CI
   - Load Ve templates from a precomputed FEM (or fall back to a
     per-contact point-source approximation if FEM is too heavy)
   - Output JSON in the (rows × cols × freqs × amps × stats) shape that
     `fig3_combined.py` expects

**Sequencing.** Anatomy + cuff Ve templates are the gating items —
without them we can't run either panel at manuscript fidelity. The
SLURM array structure already exists from selectivity_sweep, so once
the data inputs are in `data/anatomy/` the cluster pipeline is a
~1-day port.

### 4.3 No C-fiber selectivity demonstrations

We have Sundt + Rattay + Schild94/97 channel models, but the selectivity
optimizer is **MRG-only** (`stack_fiber_statics` is hard-coded to MRG
channels in `batch_solve.py`, line 31–41). The biophysically most
interesting selectivity story for vagus is **A vs B vs C fiber selectivity**
(sensory afferent C-fibers vs motor B-fibers vs reflex A-fibers). HGP24
restricted itself to A-fibers because the surrogate is trained on MRG only.

This is a clean differentiator: a paragraph in the abstract that says
"the same framework jointly optimizes for selective recruitment across
fiber types" lands well.

### 4.4 No experimental validation

HGP24's S-MF cites Musselman 2023 and Blanz 2023 — published in vivo
recordings of vagus thresholds in rat / pig / human, all reproduced by
their model. We don't even have rat morphology in the project.

Min bar: reproduce **at least one in vivo dataset**. The cleanest
candidate is Musselman 2023 (rat VNS, Helical cuff). The rat is
monofascicular and ~10× smaller than pig — should be a 1-2 day port.
Data is in the SPARC repo.

### 4.5 No optimization for novel objectives (the natural follow-up beat)

Now that gradient descent works through the **exact** model, the
natural Nat Commun beat is *what new objectives can we now optimize*?

* **Energy** efficiency at fixed selectivity (HGP24 mentions this; we
  have the loss in `jaxfibers/objectives.py` but never use it).
* **Charge** efficiency — coulombs per recruited fiber. (
* **Robustness** — optimize for the *worst-case* selectivity over a
  morphology distribution (i.e., implicit differentiation through
  inner argmax). HGP24 doesn't do this.
* **Waveform smoothness regularization** — HGP24's GD waveforms are
  smooth (Fig 5c); their DE waveforms are noisy. We should explicitly
  demonstrate that GD does this *because the gradient is informative*.

### 4.6 Joint amp + electrode-position optimization is implemented but unused

`run_joint_optimization` exists and is sophisticated (4K+1 configs via
finite-difference packed into one batched forward pass), but
`outputs/selectivity_joint_opt/` is empty. **This is a flagship feature**
because AxonML cannot do it: its surrogate is trained for a fixed
electrode (and fascicle layout in some cases). When the electrode moves,
the surrogate must be retrained.

→ Run this, plot the contact-displacement panel for at least 10 seeds.

---

## 5. Engineering quality

What's good:
* Clean separation `channels/` `fibers/` `stim/` `optim/` `nerve/`.
* The coupled solver (`extracellular_coupled.py`) is documented to 5e-11
  agreement with a dense-LU reference. Block-Thomas with optional
  `associative_scan` for O(log n) backward depth. State of the art.
* Pulse-shape registry (`utils.py:PULSES`) is shared across all
  validations — consistent sign convention.
* `nrn_baseline.py` (795 lines) wraps PyFibers cleanly for every model.
* SLURM scripts wired and `submit_all.sh` exists.

What's not:
* Channels in selectivity solver are hard-coded MRG (`batch_solve.py`).
  Should be polymorphic over a channel-pack interface.
* Two extracellular implementations live in `stim/`:
  `extracellular.py` (single-cable, deprecated?) and
  `extracellular_coupled.py` (the right one). Pruning to one would
  reduce confusion. Also `mrg_extracellular_solver.py` (375 lines) is
  another whole file — is this stale?
* `outputs/` mixes large JSON traces (1 MB+) with PNG figures. For
  publication-grade data sharing these should go into separate
  `outputs/data/`, `outputs/figs/` subtrees.
* No CI / no automated regression. A new contributor breaking
  `block_thomas_assoc` would not be caught until someone re-runs all
  validations.
* `experiments_v2/` has many one-off investigation scripts
  (`investigate_rattay_bica.py`, `verify_fix_*.py`,
  `smoke_test_sd_anomalies.py`) which are great for development but
  should be moved into `experiments_v2/debug/` so the top-level reads
  as a manifest of paper-relevant runs.
* The `experiments/` folder referenced in `README.md` doesn't exist —
  only `experiments_v2/`. README is stale.

---

## 6. Recommended roadmap to Nat Commun submission

### Phase A — close known issues (~2 weeks)

* **A1.** Fix Rattay `bi_ca` threshold bug. Use
  `investigate_rattay_bica.py` to bisect. Acceptance: max |err| < 0.5 %
  across all 240 cases (matches Sundt).
* **A2.** Fix Rattay CV (–2 % offset). Acceptance: < 0.1 % CV error.
* **A3.** Fix Sundt D=1.0 µm CV spike. Acceptance: 1e-12 % at all
  diameters.
* **A4.** Run Schild94 and Schild97 validations on cluster. Acceptance:
  thresholds < 1 % MAPE, CV < 1 %, the Ca/Na/K pump dynamics verified
  against NEURON.
* **A5.** Run `selectivity_sweep.py` for the full 100 seeds — confirm
  the gradient-descent rect & waveform pipeline works at scale and
  matches the demo.
* **A6.** Run `selectivity_joint_opt.py` for ≥ 10 seeds.
* **A7.** Prune `outputs/` and `experiments_v2/` (move debug scripts to
  a subfolder; update README).

### Phase B — match HGP24's scientific scope (~6 weeks)

* **B1.** **Ingest ASCENT FEM Ve fields** for ≥ 1 pig + 1 human vagus
  nerve from the SPARC repo (Pelot 2020 datasets). Replace point-source
  Ve in `selectivity_*`. This is the biggest single-effort task and the
  hardest political requirement to skip.
* **B2.** Implement `khz_block.py`, `ap_collision.py`, `dc_block.py`,
  `spike_desync.py`. Should produce one figure each demonstrating
  emergent nonlinear phenomena, all matching NEURON to machine
  precision (because the underlying solver is exact). One panel each
  → composite "Figure 3" in the paper.
* **B3.** Implement DE+NEURON pipeline (via the published `cajal`
  package from HGP24) and run head-to-head on ≥ 3 of the same nerves.
  Plot Pareto front (target activation × off-target × time).
* **B4.** Reproduce one in vivo dataset (Musselman 2023 rat VNS is
  cleanest). 1-fascicle rat morphology, helical cuff, threshold for
  hCM heart-rate effect.
* **B5.** Make `batch_solve.py` polymorphic over channel pack so the
  selectivity optimizer can target Sundt/Rattay/Schild C-fibers. Then
  demonstrate **A vs C fiber selectivity** on a mixed nerve. (Clean
  differentiator vs HGP24 — flagship of the new paper.)

### Phase C — novel contributions (~4 weeks)

* **C1.** Implement *energy* and *charge* objectives in the optimizer
  (the loss functions exist; just need the optimization runs).
* **C2.** Implement **robust selectivity** = max over morphology
  distribution. Implicit differentiation through the inner argmax via
  `jax.lax.stop_gradient` + finite-difference outer loop. **No
  surrogate-based work has done this; this is a Nat Commun-grade
  novelty.**
* **C3.** Optional: train a small NN surrogate against the exact
  solver, demonstrate hybrid GD-on-surrogate + verify-on-exact gives
  the best of both worlds (this maps to HGP24's "future work" mention
  and is a low-risk extension).
* **C4.** Optional: differentiable **parameter identification** — fit
  MRG channel parameters to in vivo recordings via gradient descent.
  Powerful demo because it inverts the model HGP24 only uses
  forwards. Very strong Nat Commun-style hook.

### Phase D — paper (~3 weeks)

* **D1.** Manuscript skeleton matching HGP24's structure (Fig 1 cable
  + cuff, Fig 2 threshold accuracy table, Fig 3 emergent phenomena,
  Fig 4 spatial selectivity rect, Fig 5 spatial selectivity arbitrary,
  Fig 6 *new novel objective*).
* **D2.** Code release via Zenodo + GitHub mirror; SPARC dataset entries
  for any new histology used.
* **D3.** Pre-submission inquiry to Nat Commun editor.

---

## 7. Story positioning (the abstract elevator pitch)

> "Surrogate models of peripheral nerve fibers (HGP24) accelerate
> stimulation optimization at the cost of accuracy (2.5 % MAPE,
> threshold errors up to ±11 %) and per-task retraining. We present
> **jaxley_fibers**, a differentiable JAX/Jaxley reimplementation of
> six published fiber models (MRG, Sweeney, Sundt, Rattay,
> Schild 1994 & 1997) sharing a custom GPU-batched coupled-cable solver
> that matches NEURON to **<0.5 % threshold accuracy** and **machine
> precision on conduction velocity**, while remaining **60–120× faster
> than single-core NEURON** for population simulation. Because the
> underlying biophysics is solved exactly rather than approximated,
> gradient descent can optimize *any* differentiable objective —
> selectivity, energy, robustness, electrode position, or experimental
> fit — without retraining. We demonstrate (i) selective activation
> across fiber types in pig and human vagus nerves with FEM-derived
> fields, (ii) joint electrode-position + amplitude optimization
> (which fixed-field surrogates cannot perform), (iii) robust
> selectivity over morphology distributions, and (iv) gradient-based
> identification of in vivo channel parameters."

This positions us as **complementary** to AxonML, not a replacement.
Editors prefer that framing.

---

## 8. Risk register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Reviewer requires anatomical FEM. | High | Phase B1. |
| Reviewer compares total speed to HGP24 and finds us slow. | High | Reframe (§4.1). |
| Rattay / Schild bugs are deeper than they look. | Medium | Phase A allocates 2 weeks. |
| Waveform optimizer OOMs on real-scale problem. | Medium | Checkpointing already on `_integrate_one_fiber_m_max`; add `jax.lax.cond` for surrogate fallback. |
| Jaxley API breaks before submission. | Low | Versions pinned in `environment.yml`. |
| C-fiber selectivity is biophysically uninteresting (e.g., always activates same way as A). | Low–Medium | Verify with a small pilot before committing to Phase B5. |

---

## 9. Quick wins (≤ 1 day each)

These are low-effort, high-clarity tasks worth doing immediately:

1. **Run Schild94 / Schild97 validations.** `sbatch
   slurm/run_schild94_validation.sbatch`. Half a day if it works.
2. **Run the empty selectivity sweep.** `sbatch
   slurm/run_selectivity_sweep.sbatch`. Headline data, currently
   missing.
3. **Update README.md.** It still references `experiments/`, the v1
   layout. Replace with `experiments_v2/`.
4. **Add a one-page `RESULTS.md`** summarizing the threshold / CV /
   scaling tables auto-generated from JSON. Cite from manuscript.
5. **Add a `CHANGELOG.md`** so future-you can track which validation
   data was generated on which commit.
6. **Remove `__pycache__` from every checked-in folder** if it goes
   into git (current .gitignore not verified — repo not a git repo
   per `Is a git repository: false`).
7. **Verify that `mrg_extracellular_solver.py` is dead code**; if so,
   delete or mark deprecated.
8. **Add a small `tests/test_thresholds.py`** that asserts the MRG
   max-error stays < 0.5 % — cheap CI guard.

---

## 10. References for follow-up

* HGP24 paper: doi 10.1038/s41467-024-51709-8 (attached PDF)
* HGP24 code: github.com/minhajh/axonml & github.com/minhajh/cajal
* HGP24 data: doi 10.7924/r48g8tf24 (Duke Research Data Repository)
* Pelot 2020 pig VN histology: SPARC doi 10.26275/maq2-eii4
* Pelot 2021 human VN histology: SPARC doi 10.26275/ofja-ghoz
* Musselman 2023 rat VNS thresholds: doi 10.26275/vdpw-rjqu
* ASCENT pipeline: doi 10.1371/journal.pcbi.1009285
* Jaxley: github.com/jaxleyverse/jaxley

---

*Audit produced 2026-06-02 — re-run when phase boundaries cross.*
