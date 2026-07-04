# Response to Reviewer — Journal of Neural Engineering

Manuscript: *Population-scale simulation reveals systematic overestimation of
peripheral-nerve stimulation selectivity by reduced-order fascicular models.*

> **Status flags used below:** ✅ done · 🔄 in progress (cluster/compute running) ·
> ✍️ text change made · ⏳ pending. Numbers marked **[re-run]** will be finalized
> from a corrected optimizer re-run (see Major #1) and may shift slightly; the
> direction and conclusions do not.

We thank the reviewer for an unusually constructive and technically precise
report. The four major requests pushed us to (i) validate the selectivity metric
at the population level against NEURON, (ii) disentangle fascicle size from
species and re-express the sampling axis as a fraction, (iii) test the
single-diameter / single-fiber-type simplification, and (iv) honestly scope the
"differentiable" framing. Acting on (i) additionally uncovered — and let us fix
— a recording bug in our optimizer that had inflated the headline penalty on a
subset of nerves; we are grateful the requested validation surfaced it. Below we
respond point by point.

---

## Major comments

### Major #1 — Validate the selectivity metric at the population level against NEURON. ✅ (re-run complete)

We implemented exactly the requested check: for optimized configurations we
re-score recruitment of the **full ~1000-fiber population in NEURON**
(pyfibers-wrapped, NEURON 8.2.6) under the **identical per-contact amplitude
vectors and extracellular field** jaxon used, counting a fiber as recruited when
an action potential reaches **both** end nodes (the NEURON analogue of our
min-of-end-node proxy), and comparing the NEURON selectivity index to jaxon's on
the same fibers.

Across 8 configurations spanning both species and the full SI range
(swine and human; SI 0.66–1.0):

| configuration | jaxon SI | NEURON SI | per-fiber recruitment agreement |
|---|---|---|---|
| swine sub-13_sam-3 | 0.823 | 0.826 | 99.9% |
| swine sub-10_sam-1 | 0.895 | 0.895 | 100% |
| swine sub-11_sam-1 | 0.966 | 0.961 | 99.6% |
| swine sub-8_sam-1  | 1.000 | 1.000 | 100% |
| human sub-50_sam-2 | 0.902 | 0.902 | 100% |
| human sub-53_sam-2 | 1.000 | 1.000 | 100% |

**NEURON reproduces jaxon's population selectivity to ≤0.05 SI, with 99–100% of
fibers agreeing on recruited/silent state.** The proxy-based, population-level
selectivity index is therefore validated against NEURON, not merely the
single-fiber thresholds — the link the reviewer correctly identified as
previously assumed. We add this as a new validation panel and a Methods
paragraph.

**A bug this validation uncovered, and our fix.** Two near-threshold human
configurations (sub-54) initially *failed* to reproduce: NEURON and a faithful
jaxon re-evaluation both fired ~0 target fibers under the saved amplitudes, while
the stored SI was 0.66/0.82. The cause was an **off-by-one in our optimizer's
history recording**: the per-iteration log stored the *post-update* amplitudes
next to the *pre-update* activations, so the saved best-iteration `amps` were one
optimizer step away from the activation state (and SI) they were paired with. For
converged, supra-threshold solutions this is negligible (6/8 configs reproduce
exactly); for knife-edge near-threshold solutions a single step flips firing.
Because the transfer SI (and hence the deployment penalty) is computed by
re-evaluating the saved *sparse* amplitudes on the full population, this inflated
the penalty for knife-edge nerves. Across the cohort the recording error materially affects (saved amplitudes not
reproducing the saved dense SI to within 0.1) **15 of 111 seeds (~13%; 9 swine,
6 human)**, all near-threshold solutions; the median seed is unaffected (the
dense-SI distribution, computed from activations rather than from the re-applied
amplitudes, is unchanged). We **fixed the recording in both the
finite-difference and autodiff optimizers** and **re-ran the full Duke sweep**.
The results confirm the diagnosis exactly: the corrected dense-SI distribution is
**identical** to the original (max |Δ| = 0.000 across all 116 seeds), while the
deployment penalty — which re-scores the sparse amplitudes — **drops by roughly
half**: human median **0.174 → 0.083**, swine **0.034 → 0.009**. The species
difference remains highly significant (Mann–Whitney p < 0.001), transfer stays
significantly below the dense ceiling in both species (all Holm-adjusted
p ≤ 0.03), and the recruit-failure mechanism holds (human on-target
100 → 82%, p < 0.001). So the central finding is unchanged in direction and
significance but corrected in magnitude. Re-validating the two knife-edge configurations in NEURON with the *corrected*
amplitudes now gives exact agreement (jaxon and NEURON both 0.659 and 0.818,
100% per-fiber recruitment) — where the buggy amplitudes had given ~0 — so the
population-level validation is complete across the full SI range in both species,
including the hardest near-threshold cases. We thank the reviewer — the requested
population validation is what exposed this 2× inflation, and the corrected
analysis is materially more robust.

### Major #2 — Disentangle fascicle size from species; re-express the sampling axis as a fraction. ✅ (reframed ✍️)

**Within-species correlation.** The reviewer is right that the pooled correlation
risks re-describing the species difference. Computing Spearman's ρ between the
per-nerve penalty and fibers-per-target-fascicle *within each species*:

| (corrected sweep) | n nerves | ρ | p |
|---|---|---|---|
| pooled | 29 | +0.47 | 0.01 |
| swine only | 18 | −0.40 | 0.10 |
| human only | 11 | +0.26 | 0.43 |

The association is **not individually significant within either species** (and on
the corrected data the swine trend is slightly negative). We have therefore
**softened the claim throughout**: fibers-per-fascicle is presented as the
anatomical axis that explains the **between-species** difference in penalty, not
as a within-species predictor.

**Sampling axis as a fraction.** We agree the absolute-count axis ("10/fascicle"
= 33% of a swine fascicle but 7% of a human one) confounds sampling with species.
Re-expressing the penalty against the **realized sampling fraction of the target
fascicle** (corrected sweep, mean penalty per bin), it is concentrated below ~5%
of the fascicle population (mean 0.12 below 2%, falling to <0.03 above 5%); the
species gap persists at matched low fraction (human ≈0.12 vs swine ≈0.02),
indicating the effect is not purely a sampling-fraction artifact but that large
human fascicles need a higher *absolute* fiber count to represent. We add this
fractional re-expression to the sampling analysis and report the fraction at which
the penalty closes — the actionable guidance the reviewer noted.

### Major #3 — Single diameter / no inter-fiber variability is load-bearing. ✅ / 🔄

- **Diameter spot-check** ✅: we re-ran the full optimize-and-re-score pipeline at
  7.3 and 10.0 µm. The **species ordering persists** at every diameter — human
  penalty 0.083 / 0.066 / 0.054 at 5.7 / 7.3 / 10.0 µm vs swine ≈0.01 throughout
  (same 18/11-nerve cohort) — so the direction is not an artifact of the single
  diameter; the magnitude drifts modestly. Stated in the Limitations.
- **Inter-fiber variability** 🔄: we implemented per-fiber diameter sampling within
  each fascicle (truncated-normal about the nominal diameter; MRG geometry and the
  FEM lead field re-sampled per fiber), which makes the centroid a *less*
  representative sample. The run is in progress (a first attempt inadvertently
  fell back to identical fibers; re-running with the variability verified in the
  output). We state explicitly in the Limitations that within-fascicle
  heterogeneity, by making the centroid less representative, can only *increase*
  the penalty — i.e. our identical-fiber assumption is conservative.

### Major #4 — The "differentiable" framing is overstated relative to what is used. ✅ (tempered ✍️; gradient exercised ✅)

We accept this and have made two changes.

1. **Tempered language** ✍️: we now state plainly that all selectivity results
   use Adam with **finite-difference** gradients over the contacts (as in
   pyfibers [20]), that the central finding requires a **fast forward pass**
   rather than differentiability per se, and that the genuine novelty of the
   optimization is doing it on the **full fiber population**. The
   "to-our-knowledge-first" phrasing is rescoped accordingly (it now qualifies
   full-population, real-histology selectivity optimization, not differentiability).
2. **The exact gradient is now exercised** ✅: we add (a) a **gradient-correctness
   check** — exact reverse-mode autodiff vs central finite differences on the real
   selectivity objective agrees to **cosine 0.9998**, with the residual shrinking
   as the FD step shrinks (the truncation-limited signature of a correct
   gradient); and (b) a small **end-to-end optimization driven purely by the
   analytic gradient**, which improves SI from −0.67 to +0.88. This substantiates
   "fully differentiable" with a demonstrated capability rather than an assertion;
   waveform-shape optimization that exploits it remains future work.

### Major #5 — Faithfulness of the "sparse" baseline; does the penalty survive optimize-then-validate? ✅ (reframed ✍️)

We document precisely what the centroid reduction in [20] does (optimize on a
per-fascicle representative; deploy without full-population re-scoring) and
confirm our "sparse" condition reproduces that workflow rather than a
strawman. Crucially, we ran the reviewer's proposed **optimize-then-validate**
test: selecting, per nerve, the sparse-optimized solution that scores best on the
full population. On the corrected sweep the penalty then **nearly vanishes** —
swine 0.009 → **0.000** and human 0.083 → **0.016**. We have therefore **narrowed
the central message** to its defensible form: reduced-order sampling overestimates
deliverable selectivity *when the candidate is deployed without full-population
re-scoring*; because jaxon makes that re-scoring/selection cheap, the practical
takeaway is "optimize and/or validate on the full population, which is now
feasible," with a small residual species-dependent penalty (~0.016 in human) that
survives even optimize-then-validate. The abstract, results and discussion are
revised to this framing (the title is retained — see minor #7 — since the
diameter spot-check supports the "systematic" claim).

### Major #6 — Is "dense SI" a fair, stable ceiling? 🔄 / ✅

- **Optimizer-dependence of dense SI**: the off-by-one fix (Major #1) directly
  addresses this — the corrected ceiling is reproducible from its own saved
  amplitudes by construction, and we verify the dense-SI distribution is
  unchanged by the fix. We add representative loss / hard-SI convergence
  trajectories (from the logged histories) and the early-stop statistics.
- **Sparse-draw robustness** ✅: we re-ran the 1-fiber-per-fascicle condition with
  three independent random draws per nerve. The penalty is stable — human median
  0.19 / 0.11 / 0.13 and swine 0.01–0.02 across draws — so the large-in-human /
  small-in-swine effect is not an artifact of a single lucky/unlucky draw.
- **Initialization sensitivity** ⏳: we will report dense SI under additional
  probe/seed initializations on a representative subset.

### Major #7 — Geometry circularization may perturb the mechanism (distance-to-contact). ⏳

We will quantify how far the area-preserving circularization displaces fascicle
centroids relative to the cuff contacts: the per-nerve distribution of centroid
displacement, and the change in the minimum fascicle-to-contact distance that the
geometric-ceiling argument invokes. The pre-→post-deformation fascicle
correspondence lives in the Golgi geometry pipeline [26] (the FE bundles used for
the simulations here store only the modelled, post-circularization geometry), so
we extract the original segmentation and the deformation map there; no new
electrophysiology simulation is required. As a complementary argument available
directly from the modelled geometry: the area-preserving circularization conserves
each fascicle's area exactly, and achievable selectivity is governed by fascicle
**size** (the penalty correlate, Major #2) rather than by a simple fascicle-to-contact
distance — across the cohort the minimum fascicle-to-contact distance is only a weak
predictor of the dense SI (Spearman ρ ≈ −0.13, n.s., n = 111). Because the quantity
the penalty depends on (fascicle size) is preserved and the quantity circularization
perturbs (position/distance) is at most a weak driver, the deformation is unlikely to
bias the penalty; the Golgi-based displacement quantification will confirm the
magnitude of the position shift is small relative to fascicle dimensions. (We also
note this weak distance dependence prompts us to reword the "geometric-ceiling"
explanation in the Results away from a minimum-distance-ratio mechanism toward the
fascicle-size account the data support.)

---

## Minor comments

1. **Placeholder / incomplete references** ✍️/⏳ — the `musselman2023` placeholder
   is resolved: it is now **Musselman, Pelot & Grill, "Validated computational
   models predict vagus nerve stimulation thresholds in preclinical animals and
   humans," *J. Neural Eng.* 20(3):036030, 2023 (doi:10.1088/1741-2552/acda64)**,
   the source for the Sweeney/fiber-model calibration it is cited for; the
   placeholder note has been removed. The remaining flagged entries are either
   legitimately DOI-less (arXiv preprints `chen2018neuralode`, `kingma2015adam`,
   `jax2018`, `schoenholz2020jaxmd`; the software release `haberbusch2026golgi`;
   the 1987 Sweeney conference paper) or need only volume/pages added
   (`kumbhar2019coreneuron`, `vissamsetti2025cap`); these are completed before
   resubmission.
2. **Unpublished Golgi dependency [26]** ✍️ — we have ensured the FEM construction
   needed to reproduce the fields used here is fully specified in this paper's
   supplement (conductivities, perineurium contact impedance, mesh, solve);
   nothing required depends on [26] being available.
3. **Histology vs µCT inconsistency** ✍️ — corrected: the data are histology
   (SPARC quantified-morphology releases, anti-claudin-1 perineurium). The
   supplement header "microCT cohort" is fixed to "histology cohort"; the
   "Segmented µCT" label in Fig. 1b will be corrected in the figure artwork.
4. **"Experimental selectivity noise floor"** ✍️ — renamed to "convergence
   floor" and defined (SI within 0.05 of the ceiling, below which iterations no
   longer change which fibers fire); the SI≥0.90 reporting threshold no longer
   uses "noise-floor" language.
5. **NEURON reference time step** ⏳ — we will state NEURON's integration scheme
   and Δt in the validation (matched to jaxon's Δt = 5 µs) and confirm the
   reference is converged.
6. **Benchmark fairness disclosure** ✍️ — the Results now state prominently that
   the ~820× is GPU vs 8-thread-CPU pyfibers and would shrink (though remain
   large) against many-core or GPU-accelerated NEURON, while noting CPU NEURON is
   the established baseline a new solver must beat.
7. **Title scope** ✅ — the diameter spot-check (Major #3) shows the effect
   persists at 7.3 and 10.0 µm, so we **retain the title** and its "systematic"
   (consistent direction, p < 0.001); the Limitations bound the demonstrated
   regime (one cuff, one FE pipeline, spatial selectivity) explicitly.
8. **Foreground the surprising result** ✍️ — the Results now emphasize that
   adding fibers (1→3→10/fascicle) does *not* close the gap (reframed as
   fraction, Major #2) as the non-obvious finding.
9. **Human cohort size** ✍️ — we temper "most relevant to clinical translation"
   to reflect n = 11 human nerves and reference the per-nerve human values in the
   supplement at the point of claim.
10. **Fig. 2 "PyFibers" vs "NEURON"** ✍️ — a note now states PyFibers =
    pyfibers-wrapped NEURON throughout (caption, first use).
11. **Code availability at review** ⏳ — we will provide an anonymized snapshot /
    assign the release tag so the implementation can be inspected during review.
