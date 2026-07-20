[Author letterhead / institution]
Center for Medical Physics and Biomedical Engineering
Medical University of Vienna

[Date]

The Editors
Journal of Neural Engineering
IOP Publishing

Dear Editors,

We are pleased to submit our manuscript, "A GPU-native, differentiable fiber
simulator reveals a species-dependent selectivity penalty in reduced-order
peripheral-nerve models," for consideration in the Journal of Neural
Engineering.

Computational models are central to the design of selective peripheral nerve
stimulation, but to stay tractable they almost universally reduce each fascicle
to one or a few representative fibers. Whether that reduction biases the
selectivity these models predict has, until now, been effectively untestable:
answering it requires simulating entire fiber populations under optimized
extracellular fields, which is prohibitive with the standard NEURON forward
solver. Our work makes that test feasible and then carries it out.

We report two contributions. First, we present jaxon, a fully differentiable,
GPU-native reimplementation in JAX of the canonical peripheral-nerve fiber
models -- the myelinated McIntyre-Richardson-Grill and Sweeney axons and the
unmyelinated Sundt and Rattay C-fibers. jaxon is validated against
pyfibers-wrapped NEURON (99.6% of 943 configurations within 1% threshold error,
conduction velocity to machine precision) and reaches a geometric-mean ~820x
speedup at N = 100,000 fibers on a single NVIDIA A100 -- fast enough to optimize
and then re-score whole-nerve populations, and differentiable so that
stimulation parameters can be optimized through the cable equation rather than
grid-searched.

Second, using FEM-derived lead fields over a 12-contact cuff on 29
histology-segmented vagus nerves (18 swine, 11 human), we use jaxon to quantify
how much a common reduced-order shortcut -- optimizing on one representative
fiber per fascicle -- overestimates the selectivity actually deliverable to the
full population. The overestimate is negligible in swine (median ~0.01) but
substantial and heavy-tailed in human (median 0.14, up to 0.45 on the
largest-fascicle nerves; Mann-Whitney p = 0.01), because human fascicles are
large. It manifests as a failure to recruit the target fascicle, not as
off-target spill, and sampling more fibers helps but does not close it on the
biggest fascicles. Crucially, re-scoring the optimized candidate on the full
population -- cheap in jaxon -- recovers most of the gap (human residual ~0.02).
The actionable message for reduced-order pipelines is therefore
population-scale validation, now feasible, rather than a different optimizer.

We believe this work is well suited to the Journal of Neural Engineering's
readership: it provides an open-source, gradient-enabled tool that drops into
existing peripheral-nerve modeling pipelines as a replacement for the NEURON
forward solver, and it delivers a concrete, quantified caution about a modeling
practice in widespread use for neural-interface design. jaxon is released
open-source under the MIT license, and all processed data and scripts needed to
reproduce every figure and reported number are archived (see the Data & Code
Availability statement).

This manuscript has been substantially revised in response to detailed peer
review. Among other improvements, we now give an explicit equation for the
selectivity index; reconcile and consistently report multiple-comparison-corrected
statistics throughout; add a subject-level robustness analysis confirming the
species difference is not an artifact of repeated cross-sections; correct and
clarify the description of the coupled block-tridiagonal solver; validate the
activation proxy against full NEURON spike detection on a population subset; and
temper the framing of the benchmark and the clinical scope. A point-by-point
response to the reviewers accompanies this submission.

The manuscript is original, is not under consideration elsewhere, and all
authors have approved this submission. We have no competing interests to
declare. We would be glad to suggest qualified reviewers on request.

Thank you for your consideration.

Sincerely,

Max Haberbusch (corresponding author)
on behalf of David Lung and Max Haberbusch
Medical University of Vienna
max.haberbusch@meduniwien.ac.at
