# Repository cleanup log (submission preparation)

Performed while preparing the repository for journal submission. All removals
were cross-checked against `experiments_v2/FIGURE_DATA_MAP.md` to confirm that
nothing referenced by a paper figure, table, or headline number was affected.

## Removed

### Build cruft (gitignored)
- All `__pycache__/` directories (8) and `*.pyc` files (107) outside `reference_code/`.

### Stale scratch (gitignored)
- `logs/` — old SLURM job logs (frozen_vs_relaxed run, June 2026).
- `outputs_new/` — stray one-directory output tree (`frozen_vs_relaxed`).

### Superseded / exploratory sweep outputs (gitignored; ~250 MB)
Not referenced by any paper figure or number:
- The uncorrected off-by-one sweep was removed; the corrected sweep it was superseded by now occupies `outputs/duke_sweeps/` (renamed from the interim `duke_sweeps/`).
- `outputs/duke_sweeps_D14.0/` — 14 um diameter variant (paper spot-checks only 5.7/7.3/10 um).
- `outputs/duke_sweeps_rs1,rs2,rs3/` — reseed robustness variants of the corrected sweep, not in final paper.
- `outputs/duke_sweeps_var/`, `outputs/duke_sweeps_var2/` — exploratory variants.
- `outputs/duke_sweeps_autodiff/` — autodiff-vs-FD sweep; optimizer validation uses `reviewer_analyses/` instead.
- `outputs/spike_desync/` — exploratory, not in paper.
- `outputs/fig3_combined/` — empty intermediate figure-staging dir.

### Dev-time scratch (was git-tracked; recoverable from history)
- `experiments_v2/debug/` — two smoke-test/investigation scripts (`investigate_rattay_bica.py`, `smoke_test_sd_anomalies.py`).
- `followup_ideas/` — a single brainstorming note (`idea_heterogeneous_fiber_population.md`).
- `current_manuscript_narrative.md` — superseded narrative draft (moved to Trash).

## Kept (load-bearing or cited)
- `outputs/duke_sweeps/` — corrected cohort analysis (Fig 3, tab:duke-cohort, all penalty statistics).
- `outputs/{mrg,sweeney,sundt,rattay}_validation/` — Fig 1 + validation tables.
- `outputs/scaling/` — Fig 1e, tab:scaling.
- `outputs/{dc_block,depol_block,ap_collision}{,_rattay,_sundt,_sweeney}/` — Fig 2.
- `outputs/khz_block{,_rattay,_sundt,_sweeney}/` — figS_khz_block.
- `outputs/reviewer_analyses/` — optimizer-validation + NEURON-population proxy validation.
- `outputs/duke_sweeps_D7.3/`, `outputs/duke_sweeps_D10.0/` — diameter spot-check cited in Discussion (human penalty 0.066, 0.054).
- `outputs/khz_population/` (392 KB) — retained; small, referenced only by a non-paper staging script.
- `duke_Ves/` (13 GB) — raw golgi FEM meshes; feed the per-specimen anatomy figures. Retained in the working repo but EXCLUDED from the lean Zenodo data archive (available via golgi/SPARC).

## Note
The repository remains ~14 GB because `duke_Ves/` (raw FEM meshes) dominates.
`outputs/` was reduced from ~700 MB to ~510 MB. The `.gitignore` already
excludes `outputs/`, `duke_Ves/`, `logs/`, `__pycache__/`, and `x86_64/`, so
these do not enter version control regardless.
