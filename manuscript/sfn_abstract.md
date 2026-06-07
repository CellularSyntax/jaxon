# SfN Neuroscience 2026 — abstract draft

Drop-in for the SfN online abstract form. **Verify the current character
limit on the SfN portal** (Neuroscience meetings have historically used
2300 characters including spaces for the body; the field name is
"Abstract Body"). Trim per the live limit if it has changed.

---

## Title

Differentiable JAX-based simulation of peripheral nerve fibers enables
gradient-based design of fascicle-selective stimulation

## Authors

Max Haberbusch¹

¹ Center for Medical Physics and Biomedical Engineering, Medical
University of Vienna, Vienna, Austria.

> _Add co-authors / supervisor as appropriate before submission._

## Theme / topic (SfN taxonomy)

Primary:   **F.04 — Computational and Theoretical Methods**
  (or F.06 — Models of Neuron and Network)
Secondary: **F.01 — Hardware Tools and Devices** (bioelectronics /
            neural interfaces audience)

Alternative cross-listings worth considering depending on the year's
session structure:
  - "Techniques: Neural Simulation" if listed separately
  - "Motor Systems" if you want to foreground the VNS / autonomic angle
  - "Neural Coding, Computation, and Modeling"

## Keywords (3–5)

peripheral nerve stimulation; differentiable simulation; selectivity
optimisation; vagus nerve; computational modelling

---

## Abstract body (~2300-char target)

Peripheral nerve stimulation underpins clinical therapies for
epilepsy, depression, rheumatoid arthritis, heart failure and chronic
pain. Because peripheral nerves carry mixed motor, sensory and
autonomic fibers in distinct fascicles, therapeutic efficacy depends
on which fibers are recruited—a selectivity problem nonlinearly
coupled to electrode geometry, per-contact amplitudes, and waveform
shape. Existing computational design pipelines wrap the NEURON
simulator from Python and rely on gradient-free optimisation (grid
sweeps, finite-difference, evolutionary), which scales poorly and
becomes prohibitive once arbitrary waveforms enter the search space.
We present jaxon, a fully differentiable JAX reimplementation of the
canonical peripheral nerve fiber models—McIntyre–Richardson–Grill
myelinated axons plus the unmyelinated Sundt, Sweeney and Rattay
C-fibers—built on the jaxley differentiable Hodgkin–Huxley framework,
with a coupled intracellular/periaxonal backward-Euler integrator and
a vmap-batched parallel-fiber GPU forward pass. We validated jaxon
against pyfibers-wrapped NEURON across 1,152 strength-duration test
configurations (4 models, 26 diameters spanning 0.3–16 µm, 8 stimulus
waveforms, 6 pulse widths): 100% agreed to within 1% relative
threshold error of NEURON, median <0.1%. jaxon also reproduces
action-potential collision, kHz and DC conduction block, and
repetitive-stimulation desynchronisation within floating-point
precision. On a single NVIDIA A100, jaxon outperforms pyfibers by
30× at N=200 fibers and 122× at N=10,000—making 100-realisation
selectivity sweeps tractable in a single GPU-hour. Using end-to-end
gradients, we optimised fascicle-selective stimulation across
randomised eight-fascicle nerves: multi-start LBFGS over per-contact
amplitudes achieves a mean selectivity index of 0.965 (median 0.987,
43% of seeds at SI=1.0); warm-started Adam over arbitrary waveforms
preserves this optimum. By making the peripheral nerve forward solver
end-to-end differentiable while preserving NEURON-equivalent accuracy,
jaxon enables gradient-based co-design of electrode geometry, contact
placement, and stimulus waveforms—dimensions currently inaccessible
to gradient-free pipelines.

---

## Notes on framing (delete before submission)

- **Lead is the therapeutic problem, not the JAX implementation.** SfN
  reviewers are biological neuroscientists first; bioelectronics
  people second. The opening sentence has to land "this is about
  human therapies, not a software paper."
- **The 100×-validated-against-NEURON claim is the credibility
  anchor.** Without it, reviewers (rightly) discount any JAX
  reimplementation as a tool engineering exercise. With it, the
  speedup and gradient-based result become *enabling*, not
  *novelty-claiming*.
- **The selectivity number (SI=0.965) does most of the work** for the
  "and here's something you couldn't do before" beat. If the 100-seed
  sweep finishes before the late-breaking deadline, refresh this with
  the final stats — but the 7-seed number is honest enough to ship now.
- **Avoid SfN-misreading the speedup.** SfN reviewers don't care about
  "X times faster than pyfibers" in isolation. The reason to mention
  it is that it's what makes a 100-realisation sweep tractable on a
  single GPU-hour — i.e. what makes the *gradient-based selectivity
  optimisation result* possible at all. Phrase it accordingly.
- **Deadlines (verify against the SfN portal):**
  - Regular abstracts: typically early May. Likely past.
  - Late-breaking: typically opens early August, closes mid-August.
    Late-breaking is the realistic submission window for this work,
    and the 100-seed sweep will probably be done by then.
