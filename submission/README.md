# Submission package — jaxon (Journal of Neural Engineering)

**Manuscript:** "A GPU-native, differentiable fiber simulator reveals a
species-dependent selectivity penalty in reduced-order peripheral-nerve models"
**Authors:** David Lung, Max Haberbusch (corresponding) — Medical University of Vienna
**Target journal:** Journal of Neural Engineering (IOP Publishing)

## Contents

| File | What it is |
|------|-----------|
| `jaxon_manuscript.pdf` | Compiled manuscript (main text + supplementary material + appendix), current revision. |
| `cover_letter.md` | Draft cover letter to the editors. Fill the letterhead/date brackets before sending. |
| `response_to_reviewers.md` | Point-by-point response to the previous review round. |
| `REPRODUCE.md` | Step-by-step guide to regenerate every figure, table, and headline number from the archived data. |
| `DATA_MANIFEST.md` | Contents of the Zenodo data archive and what each file feeds. |

## Companion artifacts (not in this folder)

- **Code:** jaxon source, released open-source on GitHub under the MIT license
  (living repository), with a frozen release archived on Zenodo for reproduction.
- **Data archive:** `jaxon_data_archive_v1.tar.gz` (~457 MB) — the lean processed
  data (golgi lead fields + all sweep/validation/phenomena outputs) that
  `REPRODUCE.md` consumes. To be deposited as the companion Zenodo record. The
  raw ~13 GB golgi FEM meshes are intentionally excluded; they are available via
  the golgi platform and the SPARC datasets (swine doi:10.26275/MAQ2-EII4, human
  doi:10.26275/OFJA-GHOZ).

## Before submitting — author to-do

Placeholders in the manuscript `\data{}` block and elsewhere still need real values:

- [ ] GitHub organization / repository URL (`\needdata{ORG}` in main.tex)
- [ ] Zenodo DOI for the frozen software release
- [ ] Zenodo DOI for the companion data archive (this bundle)
- [ ] Git commit hash of the tagged submission state
- [ ] Xeon physical core count (CPU-baseline footnote)
- [ ] Fig 0 (concept overview) raster-art fixes — see note below

### Fig 0 raster-art fixes (author-owned)

`figures/main/fig0_concept_overview.png` is a hand-designed asset with three
label errors that must be corrected in the source graphic (the caption text is
already fixed):
1. "Segmented µCT" → the data are **histology**, not micro-CT.
2. "Sweeny" → **Sweeney** (misspelling in the Available Models box).
3. Panel (d) "Human (sub-15_sam-3)" with SI = 1.000 → **sub-15 is a swine
   specimen**; relabel or replace with a genuine human example.
