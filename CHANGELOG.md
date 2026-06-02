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
