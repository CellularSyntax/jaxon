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
