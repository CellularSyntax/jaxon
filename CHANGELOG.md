# Changelog

All notable changes to `jaxley_fibers` are recorded here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
each entry dated and tied to git commits so the project can be rewound to any
prior validated state.

## [Unreleased]

### Added
- `AUDIT.md` (2026-06-02): full project audit against Hussain, Grill & Pelot
  *Nat. Commun.* 15:7597 (2024), with prioritised roadmap to Nat Commun
  submission.
- `CHANGELOG.md` (this file).
- `experiments_v2/utils.py::ap_arrival_time` — sub-step linearly-interpolated
  rising-edge AP arrival time. Used by Sundt and Rattay CV measurement to
  eliminate single-step argmax discretisation noise (~1–2 % at typical
  C-fiber Δt ≈ 0.3–0.5 ms).

### Fixed
- **Selectivity optimisation stalled at suboptimal SI in cluster runs**
  (`selectivity_sweep.py`, `selectivity_joint_opt.py`) while
  `selectivity_demo.py` reached SI = 1.0. Root cause: the cluster scripts
  inherited `run_rect_optimization` / `run_joint_optimization` defaults
  that pre-dated the audit:
  - `amp_clip=(-5.0, -0.05)` — **cathodic-only**, no field steering.
    Multi-contact selective stimulation needs anodic contacts to actively
    hyperpolarise off-target fascicles via virtual anode. Without anodic
    headroom the optimiser is stuck with whatever passive shape the
    geometry permits.
  - `amp_init_mA=-1.5` — ~8× the MRG D = 10 µm threshold, putting the
    initial state deep in the suprathreshold regime where every fiber
    fires and the loss surface is flat (acts ≈ 1 everywhere, no
    gradient signal).
  - `lr=5e-2` — slower than necessary.

  Defaults changed to match the working demo regime: `lr=8e-2`,
  `amp_init_mA=-0.4` (~2× threshold, gentle), `amp_clip=(-2.5, +2.5)`
  (symmetric, allows anodic steering). Same change applied to
  `run_joint_optimization` (`lr_amp=8e-2`, `amp_clip=(-2.5, +2.5)`).

  The Ve-weighted amplitude initialisation formula was also broken under
  the new symmetric clip: it used `amp_clip[1]` as the "off" anchor, which
  with a symmetric clip pins the farthest contacts at the **positive
  (anodic) maximum** — actively firing off-target fibers from the wrong
  side. Reformulated as `clip(ve_norm * amp_init_mA, ...)` with 0 as the
  off anchor (clip-invariant).

  `experiments_v2/selectivity_joint_opt.py` removed its explicit
  `lr_amp=5e-2` override so it inherits the new default.

  - Re-run required:
    - `sbatch slurm/run_selectivity_sweep.sbatch`
    - `sbatch slurm/run_selectivity_joint_opt.sbatch`
  - Expected: rect SI = 1.0 reachable on most seeds (the same regime
    that worked in selectivity_demo). Mixed-diameter populations may
    cap the achievable rect SI below 1.0 — that's a real geometric
    limit, separately from this fix.

- **Rattay bi_ca threshold over-estimate (+6 to +8.2 %)** at PW ≥ 0.1 ms,
  across all 5 diameters. Root cause: asymmetric exponential clip
  `jnp.exp(jnp.clip(-vsh / k, -50.0, 0.0))` in
  `jaxfibers/channels/rattay_channels.py::_alpha_beta` truncated β_m, α_h,
  β_n at v < V_rest = -70 mV (clipping exp argument > 0 to 1). At V = -100 mV
  (end nodes during the cathodic phase of bi_ca), β_m was 5.3× too small
  and α_h was 4.5× too small. The h gate failed to recover from prolonged
  hyperpolarisation, so the subsequent virtual-cathode anodal break in
  the anodic phase needed extra current to overcome residual Na
  inactivation — direction and magnitude consistent with the +6-8 % bug.
  Fix: clip changed to symmetric ±50; verified against unclipped NEURON
  formulas to 2.5e-16 (machine epsilon) across V ∈ [-120, +30] mV.
  Other pulse shapes (mono_c, mono_a, bi_ac) were unaffected because they
  do not depend on h-gate recovery from chronic hyperpolarisation at
  end nodes.
  - **Re-run required**: `sbatch slurm/run_rattay_validation.sbatch`.
  - Expected post-fix: bi_ca |err| ≤ 0.5 % matching the other pulse shapes.
  - Stale: `outputs/rattay_validation/data_rattay_sd.json` (pre-fix).

- **Sundt CV spike at D = 1.0 µm (1.6 %)** and **Rattay CV systematic
  −1.7 % to −2.6 %** across D ≥ 0.5 µm: replaced argmax-peak AP detection in
  `_jax_cv` / `_pf_cv` of `sundt_validation.py` and `rattay_validation.py`
  with `ap_arrival_time` rising-edge crossings at V_m = 0 mV. C-fiber AP
  peaks are broad and flat (~10+ samples at dt = 0.005 ms within 0.1 mV of
  the peak); argmax landed on slightly different samples between JAX and
  PyFibers, producing systematic 1–2 % CV error. Rising-edge linear
  interpolation locks to the depolarisation event, which agrees between the
  two implementations to many decimal places.
  - **Re-run required**: `sbatch slurm/run_sundt_validation.sbatch` and
    `sbatch slurm/run_rattay_validation.sbatch` to regenerate
    `outputs/{sundt,rattay}_validation/data_*_cv.json`.
  - Expected post-fix: CV error matches MRG/Sweeney pattern (1e-12 % at
    diameters where the propagation is single-site; NaN where the
    suprathreshold pulse breaks down propagation, e.g. the existing
    Sundt D = 1.2 µm artefact).
  - MRG and Sweeney CV scripts were intentionally **not** changed:
    they already match to machine precision because their AP peaks are
    sharp and unambiguous; touching them risks regression for no gain.

### Removed
- `jaxfibers/stim/mrg_extracellular_solver.py` (375 lines). Old
  quasi-static V_pax solver, superseded by
  `jaxfibers/stim/extracellular_coupled.py` (the full coupled (V_i, V_pax)
  backward-Euler with block-Thomas sweep). Verified to have zero
  references in the codebase before deletion.
- `jaxfibers/stim/extracellular_utils.py` (228 lines). Old
  activating-function-via-`.stimulate()` helper, superseded by direct
  coupled-solver integration. Zero references before deletion.

### Moved
- `experiments_v2/{investigate_rattay_bica,smoke_test_mrg,smoke_test_sd_anomalies,verify_fix_all_pulses,verify_fix_mono_a}.py`
  → `experiments_v2/debug/`. These are development-time investigation and
  verification scripts kept in the repo for provenance but moved off the
  top-level so the `experiments_v2/` listing reads as a manifest of
  paper-relevant runs.

### Changed
- **All 11 SLURM sbatch files now auto-detect a local SquashFS container**
  at `$HOME/containers/pytorch_25.03.sqsh` and fall back to the nvcr.io
  reference if it doesn't exist. Resolution order:
  1. Explicit `CONTAINER_IMAGE` env-var override (unchanged behaviour).
  2. `$HOME/containers/pytorch_25.03.sqsh` if present.
  3. `nvcr.io#nvidia/pytorch:25.03-py3` (the previous default).
  A one-time `srun … --container-save=$HOME/containers/pytorch_25.03.sqsh`
  pull then makes every subsequent job start in seconds instead of paying
  the 5-10 min Pyxis pull cost on cold node caches. Documented in README
  under "One-time cluster setup".

- **All 11 SLURM sbatch files set `SLURM_STEP_LAUNCH_TIMEOUT=600`**.
  Several cluster runs failed with `srun: error: timeout waiting for task
  launch, started 0 of 1 tasks` after the 32-second default step-launch
  timeout — the Pyxis pull of the ~10 GB `nvcr.io#nvidia/pytorch:25.03-py3`
  container exceeds 32 s on nodes with a cold cache. The 10-minute timeout
  covers a cold pull while leaving warm-cache behaviour unchanged.
  Only matters when the container is not already cached on the assigned
  node; otherwise idempotent.
- **`PYFIBERS_BUDGET_S` in `experiments_v2/scaling.py` raised from 120 s
  to 7200 s** (2 hours), and made overridable via the
  `JAXLEY_FIBERS_PF_BUDGET_S` environment variable. The 2 min cap was
  truncating PyFibers data at N ≤ 1000 for Sundt, Rattay, Schild94 and
  Schild97 (which run ~30 min - 1 hour for N = 10 000 PyFibers serial),
  forcing the scaling figure to fall back on linear extrapolation for
  those cells. With the new budget, the cluster pass yields real
  measured timings at N = 10 000 for every model in the registry.
  Set `JAXLEY_FIBERS_PF_BUDGET_S=300` for fast local smoke runs that
  still extrapolate past the budget.
  - Re-run required: `sbatch slurm/run_scaling.sbatch` to regenerate
    `outputs/scaling/data_scaling.json` with real N = 10 000 PyFibers
    timings.
- **`outputs/` is now gitignored** (reversing the earlier decision in this
  changelog entry to track it). Reasons: PNGs are binary and don't diff;
  large sweeps would balloon the repo; cluster re-runs would require
  GitLab write authentication for every job. The provenance is preserved
  by commit hash + pinned `environment.yml`; headline numbers live in
  `AUDIT.md` / this changelog as plain text. The previously-tracked
  validation snapshots remain in the git history at commit `c74fcfb`
  and can be recovered from there if needed.
- `README.md` rewritten. The old README described the v1 single-cable
  approximation (with 20-35 % threshold errors), defunct `experiments/`
  paths, the M1 Mac dev host, and `_smoke_channel.py` references to files
  that were removed in the earlier `misc/legacy/` cleanup. The new README
  reflects the v2 double-cable coupled solver, the current outputs/
  layout, the validated headline numbers from `AUDIT.md`, and the
  MedUni Vienna A16 cluster as the validation host.

### Notes
- Began Phase A of the audit roadmap: known-bug fixes and deprecated-code
  removal. See `AUDIT.md` §3 and §6 for the full list.

---

## Pre-changelog history (selected milestones)

These commits predate this changelog and are summarised from `git log` for
context. From this point forward all changes should land in a `[Unreleased]`
section above and be promoted to a dated release when ready.

- **29c6942** Fix SD curve legend: add JAX style entry, show all diameters.
- **a341b2e** Replace sequential back-substitution with `associative_scan` in
  block_thomas (O(log n) backward depth → faster autodiff).
- **7e2ae89** Replace autodiff with packed finite-difference gradient for
  rect/joint optimisation; scale up fiber counts.
- **c8d1f6c** Switch sbatch scripts to single b200 job (no array).
- **3544aef** Rich progress output in optimisers and experiment scripts.
- **[baseline commit, this PR]** Drop `misc/legacy/`; refine selectivity init
  (eccentric fascicle, Ve-weighted amp initialisation, tighter clip + larger
  FD epsilon).
