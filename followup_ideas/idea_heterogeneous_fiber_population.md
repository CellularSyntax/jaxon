# Follow-up Paper Idea: Heterogeneous Fiber Population Selectivity

**Target journal:** Journal of Neural Engineering (JNE)  
**Timeline estimate:** 6–12 months post jaxon paper  
**Data needed:** No new experiments — Duke microCT cohort + existing FEM bundles sufficient.  
**Key enabler:** 1000× JAX speedup makes 10,000-fiber heterogeneous populations tractable for the first time.

---

## Core Narrative

Current VNS selectivity research (Pelot/Grill 2024 Nat Comms and similar) uses homogeneous
single-diameter fiber populations for computational tractability. With the jaxon speedup we
can model realistic multi-type populations at clinically relevant scales (10,000 fibers).

**Central claim:** The homogeneous approximation introduces systematic SI error, fiber-type-specific
selectivity is achievable with appropriate waveform/pattern design, and C-fiber co-activation
in current clinical VNS parameters is a plausible mechanistic explanation for known autonomic
side effects (bradycardia, cough, hoarseness).

---

## VN Fiber Composition (literature)

| Type | Fraction (count) | Diameter | Model |
|------|-----------------|----------|-------|
| Aβ/Aδ myelinated | ~5–10% | 6–16 µm | MRG |
| B-fiber (preganglionic autonomic) | ~10–15% | 1–3 µm | MRG small-diameter |
| C-fiber (unmyelinated) | ~75–80% | 0.2–2 µm | Sundt |

Ratios vary by study and nerve level — treat as a sensitivity parameter.

---

## Main Research Questions

**RQ1:** How wrong is the homogeneous single-diameter approximation for predicting SI?

**RQ2:** What is achievable fiber-type-specific selectivity with the 12-contact cuff geometry?

**RQ3:** Does the Adam-FD optimizer discover fiber-type-aware stimulation patterns when given a
heterogeneous population (exploiting the large threshold gap between Aβ and C-fibers)?

**RQ4:** What does the true recruitment curve look like — stratified by fiber type — for
optimized vs clinical stimulation configurations?

---

## In Silico Experiments

### Exp 1 — Sensitivity of SI to population assumption

Run the same optimization on the full Duke cohort under 4 population assumptions:

| Population | N fibers | Models |
|---|---|---|
| Homogeneous Aβ 5.7 µm | 1,000 | MRG only — current baseline |
| Heterogeneous myelinated | 1,000 | MRG diameter distribution (6–16 µm) |
| Heterogeneous mixed | 1,000 | 20% MRG + 80% Sundt C-fiber |
| Dense heterogeneous | 10,000 | 20% MRG + 80% Sundt, full scale |

→ **Figure:** violin plot of SI across cohort for each assumption.  
Directly quantifies the "homogeneous overestimation" in the current literature.

---

### Exp 2 — Fiber-type-specific SI optimization

Three optimization objectives (requires updated `wq_loss` with per-fiber-type weighting):

- **Aβ-only SI:** maximize Aβ target activation, minimize Aβ off-target. Ignores C-fibers.
- **C-fiber-only SI:** same for C-fibers. Requires higher amplitude / longer pulse width.
- **Aβ-selective, C-sparing:** maximize Aβ target activation while jointly penalizing
  C-fiber activation everywhere. The clinically relevant "side-effect-aware" objective.

The third objective is the novel contribution. C-fiber co-activation is a plausible
mechanistic driver of VNS side effects — optimizing against it has direct clinical relevance.

→ **Figure:** per-objective: optimized amplitude patterns, SI, fiber-type co-activation
rates across cohort.

---

### Exp 3 — Fiber type recruitment curves

For 3–5 representative nerves (heterogeneous population), sweep stimulation amplitude 0→clip for:
- Optimized selective configuration (from Exp 2)
- Clinical monopolar (reference)
- Naive tripolar

Record % activated fibers stratified by type (Aβ, Aδ, C) at each amplitude.

→ **Figure:** sigmoidal recruitment curves per fiber type, per configuration.  
Shows the therapeutic window and where C-fibers gate on relative to Aβ.

---

### Exp 4 — Pulse width × amplitude phase diagram

Strength-duration relationships differ sharply between fiber types.  
Short PW (0.1 ms) strongly favors large myelinated; long PW (2–5 ms) narrows the
selectivity window by lowering C-fiber thresholds.

Run full optimization at 5 pulse widths (0.1, 0.25, 0.5, 1.0, 2.0 ms), record
fiber-type co-activation rates.

→ **Figure:** 2D phase diagram (amplitude × PW) with contours of Aβ activation %
and C-fiber co-activation %. Defines the "safe operating zone."

---

### Exp 5 — 10,000-fiber scaling (showpiece experiment)

Pick 3 nerves. Run Exp 1–3 at N = 10,000 fibers. Compare SI and recruitment curves
against N = 1,000.

Two purposes:
1. Convergence check: does SI stabilize at 1,000 or shift at 10,000?
2. Headline compute: wall time at 10,000 fibers with JAX vs projected PyFibers/NEURON.

→ **Figure:** convergence plot SI vs N_fibers (100 → 10,000) + timing bar chart.

---

## Implementation Gaps

### 1. Heterogeneous Ve_unit interpolation
`Ve_unit` is currently precomputed at MRG 5.7 µm compartment z-positions.
For a heterogeneous population:
- Each MRG diameter class needs re-interpolation at its own z-grid
  (MRG internode length scales linearly with diameter)
- Sundt C-fibers need interpolation at uniform compartment z-positions

FEM `paths_Ve.npz` is already saved — this is a re-interpolation step, NOT a FEM re-run.
The duke loader needs to accept a fiber type/diameter list and build a joint Ve_unit.

### 2. Multi-objective SI loss
The "Aβ-selective, C-sparing" objective requires a modified `wq_loss`:
- Target fibers (Aβ in target fascicle): reward activation
- Off-target fibers (Aβ in off-target fascicle): penalize activation
- C-fibers everywhere: penalize activation (weighted penalty term λ_C)

Maps naturally onto the existing class-balanced weight vector — C-fibers just get
a negative weight rather than a zero weight.

### 3. Diameter distribution sampling
Need a function to seed N fibers per fascicle following the known VN diameter
distribution (lognormal for myelinated, separate for C-fibers). Treat the
literature ratios as parameters in a sensitivity sweep (Exp 1).

---

## Related / competing work to track

- Pelot & Grill (2024) Nat Comms — highly efficient optimization, single-diameter homogeneous
- Schiefer et al. — fiber type recruitment in peripheral nerve
- Tarotin et al. — CNAP decomposition and fiber type inference
- de Lannoy et al. — fiber diameter distribution estimation

---

## Possible extension (separate paper)

**Inverse problem:** given measured CNAP recordings, infer fiber diameter distribution
and/or fascicular anatomy via gradient descent through the JAX forward model.
JAX autodiff through the biophysical solver is the key enabler that makes this tractable
with full biophysical models (prior work uses simplified cable approximations).
See `idea_cnap_inverse_optimization.md` if that note exists.
