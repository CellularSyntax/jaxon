# Agent handoff prompt (paste this verbatim to the new Claude Code agent on Windows/CUDA)

> Copy the block below (between the `---` lines) into the new agent. Everything it needs to be productive is either in this prompt, in `README.md`, or in `migration_plan.md`. Don't paraphrase — paste it as-is.

---

You are taking over the `jaxley_fibers` project on a new host (Windows, NVIDIA GPU with CUDA). The previous agent built it on an Apple M1 Max (CPU-only). All the code, plans, and outputs are checked in. Your job is to (a) get the env running on this host, (b) verify what the previous agent built, and (c) keep executing the v2 plan toward a Nature Communications submission.

## Read these in order, then come back

1. [`migration_plan.md`](migration_plan.md) — the v2 plan. **This is your ground truth for what to do.** Section 0 has the paper framing (user-supplied, verbatim). Section 1 lists what's done. Section 2 lists what's still to do, in phases A → E. Section 4 logs roadblocks and gotchas.
2. [`README.md`](README.md) — package layout, install, run instructions, headline results, caveats.
3. Skim [`jaxfibers/channels/mrg_axnode.py`](jaxfibers/channels/mrg_axnode.py) and [`jaxfibers/fibers/mrg.py`](jaxfibers/fibers/mrg.py) so you understand the shape of a Jaxley channel and how the MRG fiber is built.

## What the previous agent already did (v1)

* Translated the MRG node mechanism (`AXNODE_myel.mod`) to a pure-JAX `Channel` class, verified to ~1e-7 against NEURON across v ∈ [-120, +60] mV.
* Built the MRG fiber morphology in Jaxley (single-cable lumping of the myelin shell).
* Wrote a PyFibers wrapper for side-by-side comparison.
* Produced [`outputs/fig1_validation.png`](outputs/fig1_validation.png), [`outputs/table1_thresholds.csv`](outputs/table1_thresholds.csv) (20–35 % threshold gap — the canonical single-cable cost; v2 closes this with the double-cable mechanism), [`outputs/fig2_scaling.png`](outputs/fig2_scaling.png) (CPU-only — your job to add the CUDA line), and [`outputs/fig3_optimization.png`](outputs/fig3_optimization.png) (gradient-based waveform optimization, non-trivial head-and-tail optimum).

## What's still to do (v2 — the path to NComms)

Phase A — biophysical foundation:
* **A.1 Double-cable extracellular mechanism** — implement NEURON's two-layer `extracellular` (xc, xg, xraxial) in Jaxley as either a stacked second cable for V_periaxonal or a custom Channel that maintains a per-compartment V_ext auxiliary state. Target: drop the 20–35 % threshold gap to < 5 %. **This is critical-path** for the "biophysical equivalence" claim.
* **A.2 Tigerholm C-fiber translation** — 11 NMODL mechanisms (see `jaxfibers/nrn_baseline.py` for the full list as PyFibers inserts them, or `reference_code/pyfibers/src/pyfibers/MOD/` for the source). Has shared Na⁺/K⁺ concentration pools across mechanisms — careful to thread the ion-concentration state through `update_states` so JAX gradients stay clean (no in-place mutation).
* **A.3 Conduction-velocity sweep** — reproduce McIntyre 2002 Fig 4 (CV vs diameter for MRG); same for Tigerholm. Adds a row to the validation suite.

Phase B — multi-electrode / multi-fascicle infrastructure (weeks 4–5).
Phase C — selectivity demonstration (weeks 6–7) — **the headline experiment**.
Phase D — robustness / supplementary (week 8).
Phase E — manuscript (weeks 9–12).

See `migration_plan.md` Section 2 for full sub-task lists.

## Host-specific things to do FIRST

1. **Verify the env**:
   ```powershell
   conda env create -f environment.yml    # remember to edit for CUDA per the file header
   conda activate jaxley_fibers
   pyfibers_compile
   python -c "import jax; print(jax.devices())"   # expect [CudaDevice(id=0), ...]
   python -c "import jaxley, neuron, pyfibers; print('ok')"
   ```
2. **Regenerate the existing outputs and check they still work end-to-end on this host**:
   ```powershell
   python experiments/exp_1_validation.py     # ~5 min; check fig1 + table1
   python experiments/exp_2_scaling.py        # ~2 min; check that fig2 NOW has a real CUDA GPU line
   python experiments/exp_3_optimization.py   # ~1 min; check fig3
   ```
   `exp_2_scaling.py` was reworked to auto-add the GPU line whenever JAX sees a non-CPU device, so on CUDA you should see a third curve in `outputs/fig2_scaling.png`.
3. **Push the CUDA scaling further**: `JAXLEY_BATCH_NS` in `experiments/exp_2_scaling.py` is currently capped at `[1, 10, 100, 1000]` for safety on the 32 GB CPU host. Try extending to `[..., 10000, 100000]` on CUDA — this is now the headline scaling number for the paper.

Once those three pass: pick up where the v2 plan leaves off. **A.1 (double-cable) or A.2 (Tigerholm) is the natural next step** — both can be started in parallel since they don't interact.

## Things the previous agent learned (don't relearn the hard way)

1. **Jaxley quirk #1**: `comp.insert(SomeChannel())` followed in the same view by `comp.set("SomeChannel_param", value)` raises `KeyError`. Workaround: do all inserts in one sweep, then all param sets in a second sweep (see `jaxfibers/fibers/mrg.py` for the pattern).
2. **Jaxley quirk #2**: the `data_stimuli` argument to `jx.integrate` is a 3-tuple `(state_name, current_array, comp_index_dataframe)` — the return value of `comp.data_stimulate(current, prior_ds)`. The 2-tuple `(array, df)` form is rejected with `IndexError`. The chained `data_stimulate(...)` pattern works cleanly inside JIT (see `experiments/exp_3_optimization.py` `forward` function for an example).
3. **Don't rebuild the cell inside a bisection loop** — it forces a fresh JIT compile each time. Build cell once, use `data_stimuli` (or a traced scalar amplitude) inside the jit'd forward function. The v1 bisection went from ~10 min/diameter to ~13 s/diameter with this fix.
4. **`jax-metal` is a dead end** for this stack. v0.1.1 (Apple's latest) pins jax ≤ 0.4.34 and breaks at the first op on jax 0.6.x with `UNIMPLEMENTED: default_memory_space`. CUDA is the real GPU path; this is why the previous agent stopped trying.
5. **Output buffering**: `print(...)` inside a `conda run -n env python ...` pipeline buffers until script exit. Pass `flush=True` to every important `print`, or set `PYTHONUNBUFFERED=1` in the env, or use `conda run --no-capture-output`.

## Six deliverables every design choice must support

Verbatim from the paper framing (also in `migration_plan.md` Section 0):
1. multi-polar stimulation protocols → multi-electrode current steering
2. isolate therapeutic target fascicles → multi-fascicle anatomy + selectivity metric
3. minimize battery energy expenditure → energy term in the loss
4. completely suppress off-target motor and pain fibers → MRG **and** Tigerholm populations
5. exact analytical gradients through the physical equations of the nerve under complex spatiotemporal extracellular fields → keep the differentiability — no ML surrogates anywhere on the forward path
6. requiring zero training data → physics + grad descent only

If a feature doesn't serve one of these six, scope it out.

## Communication / autonomy

The user (postdoc lead) prefers:
* Brief, concrete updates (one sentence per work step).
* Honest accounting: report the actual error, not the headline you wish you had.
* If a design choice is materially different from the brief, surface it with `AskUserQuestion` *before* burning hours.

Aggressive 3-month timeline → target submission ~2026-08-31. Foundation work (Phase A) should be wrapped by week 3 so there's time for the selectivity demo and the writing.

---

End of handoff prompt.
