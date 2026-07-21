# jaxon — data archive for manuscript reproduction

This archive contains the processed data required to reproduce every figure,
table, and reported number in:

  "A GPU-native, differentiable fiber simulator reveals a species-dependent
   selectivity penalty in reduced-order peripheral-nerve models"
  D. Lung and M. Haberbusch, Medical University of Vienna.

It is a companion to the jaxon source-code release (see the paper's Data &
Code Availability statement). The analysis/figure scripts live in the code
repository; this archive supplies their inputs and the intermediate outputs.

## What is (and is not) here

INCLUDED (lean, processed data):
- `lead_fields/<nerve>/` — the golgi-derived FEM inputs actually consumed by
  the selectivity pipeline, three files per nerve (36 nerves):
    * `paths_Ve.npz`         per-fibre arc-length grid + per-contact lead-field
                             tensor (potential per amp, V/A == mV/mA)
    * `nerve_xsec.json`      nerve outline + polygonal fascicles + per-fibre xy (um)
    * `electrode_config.json` 12-contact ring cuff (3 axial x 4 azimuthal) metadata
- `processed_outputs/` — the simulation/sweep outputs the figure scripts read
  (25 directories; see table below).

NOT INCLUDED (available elsewhere):
- The raw golgi FEM meshes (E_rest/Ve_rest/Ve_endo .h5, surface .vtp, renders;
  ~13 GB). These are golgi outputs; regenerate them with the golgi platform or
  obtain the underlying histology from the SPARC datasets:
    * swine  SPARC doi:10.26275/MAQ2-EII4
    * human  SPARC doi:10.26275/OFJA-GHOZ
  golgi:  Lung et al., bioRxiv 2026.07.10.737846 (software) and
          bioRxiv 2026.07.10.737529 (platform).

## Reproduction
See `REPRODUCE.md` in the code repository. In brief: install the environment
(environment.yml), map `processed_outputs/` onto the repo's `outputs/` (the
scripts read `outputs/duke_sweeps/` by default), and run `reproduce_figures.sh`. To
re-run the cohort sweeps from the lead fields, point the sweep driver at
`lead_fields/`.

## processed_outputs/ contents

| directory | feeds |
|-----------|-------|
| duke_sweeps/ | Fig 3, Table (cohort), all deployment-penalty statistics |
| duke_sweeps_D7.3/, duke_sweeps_D10.0/ | Discussion diameter spot-check (human penalty 0.066, 0.054) |
| scaling/ | Fig 1e, scaling table |
| mrg_validation/, sweeney_validation/, sundt_validation/, rattay_validation/ | Fig 1, validation tables (99.6% of 943 configs) |
| dc_block*/, depol_block*/, ap_collision*/ | Fig 2 (4 models x 3 columns) |
| khz_block*/ | Supplementary kHz-block figure |
| reviewer_analyses/ | optimizer/gradient validation; NEURON-population proxy validation |

## Nerves in lead_fields/ (36)
human_sub-46_sam-2, human_sub-47_sam-2, human_sub-50_sam-2, human_sub-53_sam-2, human_sub-54_sam-2, human_sub-54_sam-3, human_sub-55_sam-3, human_sub-56_sam-1, human_sub-56_sam-3, human_sub-57_sam-1, human_sub-57_sam-3, human_sub-58_sam-1, human_sub-58_sam-3, human_sub-63_sam-1, human_sub-64_sam-1, human_sub-65_sam-1, human_sub-69_sam-1, human_sub-70_sam-1, sub-10_sam-1, sub-11_sam-1, sub-11_sam-3, sub-12_sam-1, sub-12_sam-3, sub-13_sam-1, sub-13_sam-3, sub-14_sam-2, sub-14_sam-3, sub-15_sam-2, sub-15_sam-3, sub-4_sam-3, sub-5_sam-1, sub-5_sam-3, sub-6_sam-7, sub-8_sam-1, sub-8_sam-7, sub-9_sam-3

Generated for the manuscript submission bundle.
