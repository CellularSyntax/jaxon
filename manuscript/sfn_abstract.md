# SfN Neuroscience 2026 — abstract draft

Drop-in for the SfN online abstract form. **Verify the current character
limit on the SfN portal** (Neuroscience meetings have historically used
2,300 characters including spaces for the body field). Trim per the
live limit if it has changed.

---

## Title

Differentiable simulation of peripheral nerve fibers enables gradient-based
design of fascicle-selective neuromodulation

## Authors

Max Haberbusch¹

¹ Center for Medical Physics and Biomedical Engineering, Medical
University of Vienna, Vienna, Austria.

> _Add co-authors / supervisor as appropriate before submission._

## Theme / topic (SfN taxonomy)

Primary:   **F.04 — Computational and Theoretical Methods**
           (or F.06 — Models of Neuron and Network)
Secondary: **F.01 — Hardware Tools and Devices**
           (bioelectronics / neural-interfaces audience)

## Keywords

peripheral nerve stimulation; differentiable simulation; selectivity
optimisation; vagus nerve; computational modelling

---

## Abstract body (~2,300-char target)

Implanted peripheral-nerve stimulators treat epilepsy, depression,
rheumatoid arthritis, heart failure and chronic pain; their efficacy
hinges on engaging the right fibres within an anatomically mixed
nerve. Optimising electrode geometry and stimulus waveform for
fascicle selectivity is currently slow: standard pipelines wrap the
NEURON simulator from Python and search by grid sweeps,
finite-difference probes or evolutionary algorithms—none of which
scale to multi-contact cuffs or arbitrary waveforms. We built jaxon,
a fully differentiable JAX reimplementation of the canonical
peripheral-fibre models—McIntyre–Richardson–Grill myelinated axons
plus the unmyelinated Sundt, Sweeney and Rattay C-fibres—adding a
coupled intracellular/periaxonal backward-Euler solver and a batched
parallel-fibre forward pass to the jaxley differentiable Hodgkin–Huxley
framework. The forward pass runs in parallel on a single GPU and is
end-to-end differentiable with respect to per-contact amplitudes and
arbitrary stimulus waveforms. We verified jaxon against
pyfibers-wrapped NEURON across 1,152 strength–duration configurations
(4 models, 26 diameters from 0.3–16 µm, 8 waveforms, 6 pulse widths):
100% agreed to within 1% relative threshold error of NEURON, median
<0.1%. jaxon also reproduces action-potential collision, kHz and DC
conduction block, and repetitive-stimulation desynchronisation within
floating-point precision of NEURON. On a single NVIDIA A16 GPU, jaxon
outperforms pyfibers by ∼30× at N=200 fibres and 122× at N=10,000—
turning 100-realisation selectivity sweeps from a multi-day CPU job
into a single GPU-hour. Multi-start LBFGS over per-contact amplitudes
achieves a mean selectivity index of 0.965 (median 0.987, 43% of seeds
at SI=1.0) across randomised eight-fascicle nerves; warm-started Adam
over arbitrary waveforms preserves this optimum. Differentiability
turns PNS device design from search into optimisation: gradient-based
co-design of electrode geometry, contact placement and waveform shape
is now feasible on commodity hardware. The library is open-source,
calibrated to the same fibre models the field uses, and slots into
existing NEURON-based workflows as a drop-in solver—lowering the
barrier for any group working on selective peripheral neuromodulation.

---

## Notes on framing (delete before submission)

- **Lead is the therapeutic problem and the fascicle-selectivity
  question**, not the JAX implementation. SfN reviewers are
  biological neuroscientists first; bioelectronics people second.
- **The 100% / 1,152-config validation result is the credibility
  anchor.** Without it, reviewers (rightly) discount any JAX
  reimplementation as a tool-engineering exercise. With it, the
  speedup and gradient-based result become *enabling*, not
  *novelty-claiming*.
- **The SI=0.965 selectivity number does most of the work** for the
  "and here's something you couldn't do before" beat.
- **Frame the speedup as enabling 100-realisation sweeps in a single
  GPU-hour**, not as raw throughput. SfN reviewers care about
  *experimental* throughput, not benchmark numbers in isolation.
- **The "Why this matters" paragraph is the SfN-specific add.** It
  speaks directly to a broad SfN audience: any group studying
  peripheral neuromodulation gets a free 30× compute reduction and a
  gradient-based optimisation pathway by switching forward solvers.
  This is the community benefit you couldn't claim from the technical
  numbers alone.
- **Hardware honesty.** All headline numbers are on a single NVIDIA
  A16 (the default GPU on our HPC cluster); the absolute speedups
  understate what a researcher on an A100/H100 would see, but the
  A16-vs-pyfibers comparison is the actual measurement we ran.
- **Deadlines (verify against the SfN portal):**
  - Regular abstracts: typically early May. Likely past.
  - Late-breaking: typically opens early August, closes mid-August.
    Late-breaking is the realistic window, and the 100-seed sweep
    will probably be done by then.
