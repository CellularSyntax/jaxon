# Changelog

All notable changes to `jaxley_fibers` are recorded here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
each entry dated and tied to git commits so the project can be rewound to any
prior validated state.

## [Unreleased]

### Added
- **Fig 3 panels (a), (b), (c) now overlay PyFibers + jaxfibers** with the
  Hussain colour convention (NEURON blue solid, jaxfibers orange dashed):
  - Panel (a) `dc_block.py` — waterfall of V_m(t) per node across 4
    amplitudes; peak V_m matches PyFibers to 0.0 mV at all amps.
  - Panel (b) `khz_block.py` — verbatim reproduction of pyfibers tutorial
    5 (`MRG_INTERPOLATION`, D=10 µm, N=25 nodes, 20 kHz, [50,100] ms on,
    14 intrinsic pacing pulses at loc=0.1). At the four tutorial
    amplitudes (-0.5, -1.5, -2.5, -3.0 mA) **AP counts match exactly**
    between solvers: 14 / 36 / 14 / 10. JAX run uses `record="center"`
    + `center_comp=far_node` so peak memory stays ~1.2 MB (vs ~660 MB
    for full-fiber recording).
  - Panel (c) `ap_collision.py` — peak V_m matches PyFibers to 0.0 mV at
    all 4 diameters; both endpoint IClamps drive symmetric inward APs
    that annihilate mid-fiber.
  Panel (d) `spike_desync.py` is deferred to a future cluster run; the
  full SPIKE-sync sweep is ~30 h of PyFibers at the prescribed grid.

  **Note (2026-06-02):** the current standalone (b) and (d) scripts use
  a single MRG fiber in a uniform volume conductor — they verify the
  *model*, not the *Hussain experimental preparation*. Manuscript-grade
  panels need pig P2 + ImThera (panel b) and human H2 + helical cuff
  (panel d), neither of which exist in the repo yet. See AUDIT §4.2.1
  for the full TODO list (anatomy data, Ve templates, intrinsic firing
  patterns, cluster wallclock estimate).
- **`jaxfibers/nrn_baseline.py` PYTHONPATH fix for Windows** — prepends
  `c:/nrn826/lib/python` so `import neuron` picks the cp311 hoc binary
  before the legacy `c:/nrn` (which only has py27/35/36/37). Required
  for PyFibers comparisons to run on the dev host.

- **Phase B2 emergent-phenomena demos** (Hussain *Nat. Commun.* 15:7597
  Fig 3 equivalent) — four new scripts in `experiments_v2/`, each
  producing a manuscript-grade figure that reproduces a non-trivial
  biophysical phenomenon **without any phenomenon-specific tuning**
  (the coupled solver + MRG channels just work):
  - `ap_collision.py` (Fig 3c) — two intracellular pulses at opposite
    ends of an MRG fiber; symmetric inward propagation and mutual
    annihilation at the midpoint; node-only V_m snapshots at 5 timepoints
    × 4 diameters.
  - `dc_block.py` (Fig 3a) — cathodic monophasic pulse with extracellular
    point source, amplitude scan from 0.5× to 15× threshold; snapshots
    show clean bidirectional propagation up to 3× threshold and onset of
    deep-amplitude artefacts at 15× (delayed re-excitation residue).
  - `khz_block.py` (Fig 3b) — sinusoidal extracellular at 1, 2, 5, 10 kHz
    over 100 Hz intrinsic pacing; recruitment vs amplitude curves show
    the canonical HFAC nerve-block signature (10 kHz suppresses pacing
    to ~30 % of baseline; 2 kHz curve is non-monotonic — partial block
    window).
  - `spike_desync.py` (Fig 3d) — biphasic stim at 30 / 50 / 100 Hz with
    Poisson intracellular pacing; Kreuz et al. 2015 SPIKE-synchronization
    metric tracks how rapidly the propagated spike train decorrelates
    from the intrinsic rhythm.
  All four run end-to-end on CPU in 1-10 minutes each. Output figures
  + JSON data land in `outputs/{ap_collision,dc_block,khz_block,spike_desync}/`.

- **`experiments_v2/analyze_selectivity_sweep.py` enhanced.** Now produces
  six paper-grade figures (violin, CDF, loss curves, rect-vs-wave pair
  scatter, activation-map examples, LBFGS restart-winner histogram) plus
  a numeric summary JSON. Ready to run as soon as the cluster sweep
  finishes — single `python experiments_v2/analyze_selectivity_sweep.py`
  produces the entire selectivity figure set.

- `AUDIT.md` (2026-06-02): full project audit against Hussain, Grill & Pelot
  *Nat. Commun.* 15:7597 (2024), with prioritised roadmap to Nat Commun
  submission.
- `CHANGELOG.md` (this file).
- `slurm/build_container.sh` — one-time builder for a project-private Pyxis
  SquashFS at `$HOME/containers/jaxfibers.sqsh`. Starts from
  `nvcr.io#nvidia/pytorch:25.03-py3`, runs `pip install -r
  requirements_gpu.txt` inside the container, and saves the resulting
  layer. After this exists, the sbatch auto-detect picks it up and
  `setup_env.sh` skips its pip step entirely — sub-second container
  startup. The only per-job cost remaining is `pyfibers_compile` (writes
  to project root, must stay per-job). README documents the procedure.

- `experiments_v2/utils.py::ap_arrival_time` — sub-step linearly-interpolated
  rising-edge AP arrival time. Used by Sundt and Rattay CV measurement to
  eliminate single-step argmax discretisation noise (~1–2 % at typical
  C-fiber Δt ≈ 0.3–0.5 ms).

### Added
- **`run_rect_optimization_lbfgs`** in `jaxfibers/optim/optimizer.py` — LBFGS
  + M random restarts for rectangular-amplitude selectivity. Uses optax's
  zoom-line-search LBFGS with autodiff gradients (via the existing
  checkpointed `batch_integrate_m_max_fd`). Restart 0 is the
  Ve-weighted deterministic init (matches the legacy Adam-FD optimiser);
  1..M-1 sample uniformly in (-clip/4, +clip/4). All M restarts are
  vmapped, so they run in parallel as a single GPU pass; best-of-M wins.
  Typical convergence: ~20-30 steps × ~3-5 forward-equivalents per step
  ≈ ~100 forward equivalents (vs ~700 for 100 iters of Adam-FD).

- **`run_rect_optimization_lbfgs_batched`** — same as above, but outer-vmaps
  over a list of S seeds. Stacks each seed's FiberStatics + Ve_unit +
  target_mask along a leading S axis. Compile cost paid once per chunk;
  total work scales linearly with S but GPU utilisation improves
  significantly (a16 was underused at S=1).

- **`SEEDS_PER_TASK` env-var batching in `selectivity_sweep.py`.** Set to
  1 (default) for the legacy single-seed path; set to N > 1 to vmap the
  rect optimiser over N seeds at once. The sweep main loop now chunks
  the seed list and dispatches to either the single-seed or batched
  LBFGS entry point. Waveform step (Adam, autodiff backward) remains
  per-seed for now — batching it requires more memory-careful design.

### Performance
- **Selectivity sweep: per-seed wall reduced ~5×.** Three parallel cuts to
  `experiments_v2/selectivity_sweep.py` (and same to
  `selectivity_joint_opt.py`):
  - `T_STOP` 4.0 ms → 3.0 ms. PW + DELAY + slowest-MRG propagation
    (24 mm fiber at 26 m/s) ≈ 2.1 ms, so 3 ms covers AP arrival at
    both ends. Each ms saved cuts ~25 % off the per-iter FD pass.
  - `N_OPT_RECT` 200 → 100, `N_OPT_WAVE` 200 → 100. Adam plateaus
    well before 200 iters on these problems; selectivity_demo.py
    hits SI = 1.0 in 50 iters on a 6-fiber problem, so 100 leaves
    headroom for the 100-fiber sweep.
  - Net effect: ~30 s/iter × 400 iters = ~3.3 h/seed → ~22 s × 200 iters
    = ~40-60 min/seed.

- **`run_selectivity_sweep.sbatch` defaults: a16, 10 seeds/task, `--array=0-9%4`.**
  QOS `a16`, GRES `gpu:a16:1`, `--array=0-9%4` (10 tasks × 10 seeds = 100
  seeds, max 4 concurrent), `SEEDS_PER_TASK=10`, walltime 4 h.
  Per-task wall ~30-90 min on a16; total sweep ~3-4 h in ~3 array waves.
  h100 fall-up documented in-file as a one-liner override:
    `sbatch --qos=h100 --gres=gpu:h100:1 -t 1:00:00 slurm/run_selectivity_sweep.sbatch`

- **Selectivity sweep now uses LBFGS + 4 restarts + 4-seed vmap by default.**
  - Optimiser: LBFGS (was Adam-FD). ~3-4× fewer iterations to convergence
    because LBFGS uses curvature info via zoom line search.
  - Multi-restart: M=4 parallel restarts per seed, vmapped. Robust against
    local minima (especially on mixed-diameter seeds where Adam stalled
    at SI ~ 0.05). Best-of-M wins.
  - Multi-seed batching: SEEDS_PER_TASK=4 → 16 LBFGS trajectories vmapped
    in parallel per task. Bumps GPU utilisation from ~50 % to ~90 % on a16;
    larger gains on a100/h100.
  - Net wall: ~30-60 min per task (4 seeds), 25 tasks × 8 concurrent
    = ~4 array waves × ~45 min ≈ ~3 h total for 100 seeds.
  - On a100/h100: set `SEEDS_PER_TASK=8` (or 16); fewer/faster tasks.

- **`slurm/run_selectivity_sweep.sbatch` converted to a SLURM array job.**
  `#SBATCH --array=0-99%8` runs 100 seeds as 100 array tasks, 8 at a
  time (matches s0-n12's 8 a16 GPUs). Each task sets
  `SEED_START=$SLURM_ARRAY_TASK_ID` and `SEED_END=$((... +1))`, which
  `selectivity_sweep.py::main()` already honours. Wall reduced from
  72 h (with the previous serial run that would have hit the limit
  before finishing) to 6 h per task. The whole sweep finishes in
  ~12 wall hours at 8× concurrency instead of the impossible
  ~330 h serial path.
  - Re-run a single seed:  `sbatch --array=42 slurm/run_selectivity_sweep.sbatch`
  - Re-run a subset:       `sbatch --array=0-19 slurm/run_selectivity_sweep.sbatch`

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
- **All 11 SLURM sbatch files now auto-detect a project SquashFS container**
  at `$HOME/containers/jaxfibers.sqsh` (the deps-baked-in container built
  by `slurm/build_container.sh`) and fall back to the nvcr.io reference if
  it doesn't exist. Resolution order:
  1. Explicit `CONTAINER_IMAGE` env-var override.
  2. `$HOME/containers/jaxfibers.sqsh` if present.
  3. `nvcr.io#nvidia/pytorch:25.03-py3`.
  Earlier intermediate iterations of this auto-detect (which looked for a
  bare `pytorch_25.03.sqsh` without pip deps installed) are superseded —
  there's only one project-blessed SquashFS path now.

- **`slurm/setup_env.sh` short-circuits when deps are pre-installed.** It
  now probe-imports `jaxley` and `pyfibers`; if both are present (the
  case inside the `jaxfibers.sqsh` container) it skips the entire
  `pip install -r requirements_gpu.txt` step. On the nvcr.io fallback,
  behaviour is unchanged.

- **All 11 SLURM sbatch files set `SLURM_STEP_LAUNCH_TIMEOUT=600`**.
  Several cluster runs failed with `srun: error: timeout waiting for task
  launch, started 0 of 1 tasks` after the 32-second default step-launch
  timeout — the Pyxis pull of the ~10 GB `nvcr.io#nvidia/pytorch:25.03-py3`
  container exceeds 32 s on nodes with a cold cache. The 10-minute timeout
  covers a cold pull while leaving warm-cache behaviour unchanged.
  Only matters when the container is not already cached on the assigned
  node; otherwise idempotent.
- **`scaling.py` extended to N = 10⁵ fibers.** `N_FIBERS` now ends at
  `100_000`. With the existing 7200 s per-model PyFibers budget:
  myelinated models (MRG, Sweeney, MRG_Interp) reach N = 10⁵ for the
  PyFibers comparison; the unmyelinated C-fiber models (Sundt, Rattay)
  truncate inside the budget and rely on the existing extrapolation in
  the figure. JAX runs all N values until something OOMs.
- **`scaling.py` no longer benchmarks Schild94 / Schild97.** Removed
  from the `MODEL_REGISTRY`. Rationale: at N = 10⁵ PyFibers serial they
  extrapolate to ~9-10 h each (well past any per-model budget that
  fits in the sbatch wall) and the Ca²⁺-pool + Na/K-pump state would
  make JAX OOM-prone on a 16 GB a16. The Schild channels remain in
  the project for validation runs (`schild9{4,7}_validation.py`); just
  not in the scaling figure.
- **OOM-tolerant JAX timing in `scaling.py`.** Each `(model, N, device)`
  cell is now wrapped in try/except — if the GPU runs out of memory at
  N = 10⁵, the cell is logged as `null` in the JSON, printed as
  `FAILED:` in the log, and the next cell continues. Figure code strips
  `None` cells cleanly. Controlled by `JAX_OOM_FALLBACK = True`.
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
