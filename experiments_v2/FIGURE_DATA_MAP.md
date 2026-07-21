# Figure / Table / Number -> Script -> Data map (jaxon paper)

Authoritative provenance for every result in the manuscript. Sweep root for
the cohort analysis is `outputs/duke_sweeps/` (the corrected off-by-one
sweep), passed to the figure scripts via `DUKE_SWEEP_ROOT=outputs/duke_sweeps`.

## Main figures

| Fig | Script | Input data | Notes |
|-----|--------|-----------|-------|
| Fig 0 concept_overview | (hand-designed asset, no script) | — | raster art, edited manually |
| Fig 1 validation | `figures_validation_main.py` | `outputs/{mrg,sweeney,sundt,rattay}_validation/data_*_{sd,cv,traces}.json`, `outputs/scaling/data_scaling.json` | thresholds, CV, throughput panels |
| Fig 2 phenomena | `figures_phenomena_main.py` | `outputs/{dc_block,depol_block,ap_collision}{,_rattay,_sundt,_sweeney}/data_*.json` | 4 models x 3 columns (AP prop / DC block / collision) |
| Fig 3 duke | `figures_duke_main.py` | `outputs/duke_sweeps/` (DUKE_SWEEP_ROOT) | selectivity, deployment penalty, recruitment |

## Named supplementary figures

| Fig | Script | Input data |
|-----|--------|-----------|
| figS_density | `figures_duke_main.py` (secondary output) | `outputs/duke_sweeps/` |
| figS_khz_block | `figures_phenomena_supp_khz.py` | `outputs/khz_block{,_rattay,_sundt,_sweeney}/data_*.json` |
| figS_optimizer_validation | `fig_optimizer_validation.py` | `outputs/duke_sweeps/` + `outputs/reviewer_analyses/reanalysis_summary.json` |

## Per-specimen supplementary figures (29 nerves)

| Family | Script | Input data |
|--------|--------|-----------|
| figA_* (anatomy/FEM, 29) | `figures_supp_anatomy.py` | `duke_Ves/<nerve>/` (raw golgi FEM meshes) |
| figS_* (selectivity/activation, 29) | `figures_supp_activation.py` | `outputs/duke_sweeps/<nerve>/` |

## Tables

| Table | Source |
|-------|--------|
| tab:scaling | `outputs/scaling/data_scaling.json` |
| tab:validation-sd, -sd-per-wf, -cv | `outputs/{model}_validation/data_*.json` |
| tab:duke-cohort | `outputs/duke_sweeps/` (analyze_sparse_sampling.py) |
| tab:app-mrg/-sundt/-rattay/-sweeney/-mrg-geom, tab:opt-hparams | static (model parameter tables) |

## Headline numbers -> source

| Number | Source |
|--------|--------|
| 99.6% of 943 configs within 1% | `outputs/{model}_validation/data_*_sd.json` |
| ~820x geomean speedup @ N=1e5; 209-304x @ N=1000 | `outputs/scaling/data_scaling.json` |
| centroid penalty swine 0.011 / human 0.136; MW p=0.01, U=157, r=0.59 | `outputs/duke_sweeps/` via analyze_sparse_sampling.py |
| subject-level p=0.011 | `outputs/duke_sweeps/` (7 human / 11 swine subjects) |
| Fig3b Holm recruitment p-values | `outputs/duke_sweeps/` via figures_duke_main.py |
| Friedman chi2=11.4 p=0.003 (human density) | `outputs/duke_sweeps/` sparse sweep |
| activation-proxy vs NEURON: max |dSI|=0.0045, 99.9% agreement | `outputs/reviewer_analyses/neuron_pop/` (8 nerves) |
| block-Thomas vs dense-LU 5.6e-11 mV | `jaxfibers/stim/extracellular_coupled.py` (verified in code) |

## LOAD-BEARING output dirs (keep)
- outputs/duke_sweeps/          <- cohort analysis (Fig 3, tab:duke-cohort, all penalty stats)
- outputs/{mrg,sweeney,sundt,rattay}_validation/  <- Fig 1, validation tables
- outputs/scaling/                    <- Fig 1e, tab:scaling
- outputs/{dc_block,depol_block,ap_collision}{,_rattay,_sundt,_sweeney}/  <- Fig 2
- outputs/khz_block{,_rattay,_sundt,_sweeney}/  <- figS_khz_block
- outputs/reviewer_analyses/          <- optimizer-validation, neuron_pop proxy validation
- duke_Ves/                           <- raw golgi FEM (feeds figA_* anatomy; EXCLUDED from Zenodo lean bundle)

## KEEP (cited but secondary)
- outputs/duke_sweeps_D7.3, _D10.0    <- diameter spot-check; Discussion quotes human penalty 0.066 (7.3um), 0.054 (10um)

## SUPERSEDED / redundant output dirs (safe to remove; not referenced by any paper figure/number)
- outputs/duke_sweeps/ (31M)          <- UNCORRECTED off-by-one sweep, superseded by _fixed
- outputs/duke_sweeps_D14.0 (11M)     <- 14um diameter variant, NOT quoted in paper (only 5.7/7.3/10)
- outputs/duke_sweeps_rs1/rs2/rs3 (58M)  <- reseed robustness variants, not in final paper
- outputs/duke_sweeps_var, _var2 (64M)  <- exploratory variants
- outputs/duke_sweeps_autodiff (31M)  <- autodiff-vs-FD sweep (optimizer-validation uses reviewer_analyses instead)
- outputs/fig3_combined (empty)       <- intermediate combined-figure staging
- outputs/spike_desync (192K)         <- exploratory, not in paper
- outputs_new/ (180K)                 <- stray output tree (frozen_vs_relaxed)

Total reclaimable from outputs/: ~255 MB. Plus code cruft: __pycache__, x86_64/ NMODL, stale logs/, experiments_v2/debug/ smoke tests.
