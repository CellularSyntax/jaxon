# REPRODUCE.md — regenerating every figure, table, and headline number

This guide reproduces all results in the **jaxon** manuscript (*Journal of Neural
Engineering*, under revision) from the archived data. It is the authoritative,
tested companion to `experiments_v2/FIGURE_DATA_MAP.md` (the figure -> script ->
data provenance map) and `experiments_v2/EXPERIMENTS.md` (the experiment design
notes).

There are **two reproduction paths**:

- **Path A — from processed data (fast, CPU-only).** The `outputs/` tree (or the
  Zenodo bundle) already contains every simulation/sweep result as JSON. The
  figure scripts just read that JSON and render. This regenerates every main
  figure and the named supplementary figures in minutes on a laptop; **no GPU,
  no NEURON/pyfibers compile required.** Run `./reproduce_figures.sh`.
- **Path B — from scratch (slow, needs a GPU).** Re-run the upstream
  simulation, validation, scaling, and FEM-sweep scripts that *produce*
  `outputs/`, then run the Path-A figure scripts. This re-derives the JSON and
  needs a CUDA GPU and a compiled NEURON/pyfibers baseline.

> **The corrected sweep tree.** The whole-nerve cohort analysis (Fig 3,
> `tab:duke-cohort`, every deployment-penalty statistic) reads the corrected
> off-by-one sweep in **`outputs/duke_sweeps_fixed/`**, passed to the figure
> scripts via the environment variable **`DUKE_SWEEP_ROOT=outputs/duke_sweeps_fixed`**.
> All cohort scripts now honor `DUKE_SWEEP_ROOT` and default to
> `outputs/duke_sweeps_fixed`; the earlier uncorrected `outputs/duke_sweeps/`
> tree has been removed. Exporting `DUKE_SWEEP_ROOT` is optional but recommended
> so the sweep root is explicit.

---

## 1. Environment setup

The pinned environment is `jaxon` (conda, Python 3.11, JAX 0.6.2, Jaxley
0.13.0, pyfibers 0.8.5, optax 0.2.8). Recipe: `environment.yml`.

### 1a. Create the conda env

```bash
cd <repo root>

# CPU-only (any platform; sufficient for Path A):
conda env create -f environment.yml
conda activate jaxon

# GPU (Linux/Windows + NVIDIA; required for Path B):
#   1. edit environment.yml: comment out  "jax[cpu]==0.6.2"
#                            uncomment     "jax[cuda12]==0.6.2"
#   2. conda env create -f environment.yml
#   3. conda activate jaxon
#   4. verify the GPU is visible:
python -c "import jax; print(jax.devices())"   # should list a CudaDevice
```

A pip-only fallback is in `requirements.txt` (CPU) / `requirements_gpu.txt`
(CUDA).

### 1b. Compile the NEURON / pyfibers baseline (Path B only)

The NEURON reference (wrapped by pyfibers) needs its `.mod` files compiled once
for the local platform:

```bash
pyfibers_compile        # ~10 s; needs a C compiler (MSVC on Windows)
```

**Path A does not need this step** — none of the figure scripts import
NEURON/pyfibers; they only read JSON and use JAX for a couple of lightweight
re-optimizations. If you only want the figures, skip `pyfibers_compile`.

### 1c. GPU note

- **Path A** runs entirely on CPU in minutes.
- **Path B** requires a CUDA GPU. The manuscript numbers were produced on an
  NVIDIA A100 (cohort sweep, scaling benchmark). The validation/phenomena
  scripts pin their JAX work to CPU internally (`jax.default_device(cpu)`) for
  bit-reproducibility against NEURON, so they run on any host but are slow
  (tens of minutes each). The scaling benchmark and the Duke FEM sweep are the
  GPU-bound steps.
- All commands are run **from the repo root** with the `jaxon` env
  active. Figure scripts are invoked as modules (`python -m experiments_v2.<name>`);
  the data-producing phenomena/validation/scaling scripts are invoked as files
  (`python experiments_v2/<name>.py`) — both forms are taken verbatim from each
  script's docstring and `__main__` block.

---

## 2. Path A — regenerate all figures from processed data (fast)

With `outputs/` present (from the repo or the Zenodo bundle) and the env active:

```bash
./reproduce_figures.sh
```

That driver (authored alongside this file) sets
`DUKE_SWEEP_ROOT=outputs/duke_sweeps_fixed` and runs every main + named-supplementary
figure script in order. It regenerates:

- `manuscript/figures/main/fig1_validation.{png,svg}`
- `manuscript/figures/main/fig2_phenomena.{png,svg}`
- `manuscript/figures/main/fig3_duke.{png,svg}`
- `manuscript/figures/supp/figS_khz_block.{png,svg}`
- `manuscript/figures/supp/figS_optimizer_validation.{png,svg}`
- the 29 per-specimen `manuscript/figures/supp/figA_*.png` (anatomy/FEM) and
  `figS_<nerve>.{png,svg}` (selectivity/activation) pages.

Total wall time ≈ 5–15 min on a laptop (the per-specimen anatomy/activation
sets dominate). See the table in §3 for per-figure commands, outputs, and
runtimes, and for the numbers and the two figures the driver deliberately
leaves out (`figS_density`, cohort-stat CSV).

---

## 3. Path B — regenerate `outputs/` from scratch, then the figures (slow, GPU)

Run the upstream producers first (they write into `outputs/`), then run the
Path-A figure scripts. The producers below are grouped by which figure they
feed. **Only do this if you want to re-derive the JSON**; otherwise use Path A.

### 3.1 Validation JSON  (feeds Fig 1, validation tables, the 99.6%/943 number)

```bash
python experiments_v2/mrg_validation.py       # -> outputs/mrg_validation/data_mrg_{sd,cv,traces}.json
python experiments_v2/sundt_validation.py     # -> outputs/sundt_validation/data_sundt_{sd,cv,traces}.json
python experiments_v2/rattay_validation.py    # -> outputs/rattay_validation/data_rattay_{sd,cv,traces}.json
python experiments_v2/sweeney_validation.py   # -> outputs/sweeney_validation/data_sweeney_{sd,cv,traces}.json
```

Each runs strength-duration (9/5 diameters × 8 pulse shapes × 6 pulse widths),
conduction-velocity, and trace tasks against NEURON. ~20–40 min each on CPU
(needs a compiled pyfibers baseline).

### 3.2 Scaling JSON  (feeds Fig 1e, `tab:scaling`, the speedup number)

```bash
python experiments_v2/scaling.py              # -> outputs/scaling/data_scaling.json
# fast smoke: JAXON_PF_BUDGET_S=300 python experiments_v2/scaling.py
```

Benchmarks PyFibers (NEURON CPU serial), Jaxley CPU vmap, and Jaxley GPU vmap at
N = 1…1e5. GPU strongly recommended (the ~820× headline is measured on A100 at
N=1e5). Hours at full N on CPU; the GPU row is skipped if no CUDA device.

### 3.3 Propagation-phenomena JSON  (feeds Fig 2 and figS_khz_block)

Four models × three phenomena, plus the kHz-block supplement (4 models):

```bash
# AP propagation (column 1 of Fig 2 uses the dc_block dir):
python experiments_v2/dc_block.py            python experiments_v2/dc_block_sweeney.py
python experiments_v2/dc_block_sundt.py      python experiments_v2/dc_block_rattay.py
# DC depolarization block (column 2 of Fig 2):
python experiments_v2/depol_block.py         python experiments_v2/depol_block_sweeney.py
python experiments_v2/depol_block_sundt.py   python experiments_v2/depol_block_rattay.py
# AP collision (column 3 of Fig 2):
python experiments_v2/ap_collision.py        python experiments_v2/ap_collision_sweeney.py
python experiments_v2/ap_collision_sundt.py  python experiments_v2/ap_collision_rattay.py
# kHz block (figS_khz_block):
python experiments_v2/khz_block.py           python experiments_v2/khz_block_sweeney.py
python experiments_v2/khz_block_sundt.py     python experiments_v2/khz_block_rattay.py
```

Each writes `outputs/<phenomenon><suffix>/data_<phenomenon><suffix>.json`
(suffix ∈ {``, `_sweeney`, `_sundt`, `_rattay`}). A few minutes each on CPU.

### 3.4 Duke FEM cohort sweep JSON  (feeds Fig 3, figS_* per-specimen, all penalty stats)

This is the GPU-heavy step. The sweep runs a probe + Adam-FD optimization per
nerve, and (with `SPARSE_SAMPLING_SWEEP=true`) also the centroid / N-per-fascicle
sparse-sampling densities used for the deployment-penalty analysis. It processes
one nerve directory per invocation (`DUKE_SAMPLE_DIR`), so on a single host you
loop over the 29 cohort nerves; on the cluster it shards via SLURM.

**Single-host loop (writes the corrected tree):**

```bash
export DUKE_SWEEP_ROOT=outputs/duke_sweeps_fixed
export PULSE_SHAPE=biphasic_asym PW_MS=0.5 ASYM_RATIO=4.0 T_STOP=5.0
export SPARSE_SAMPLING_SWEEP=true SPARSE_N_PER_FASCICLE_LIST=1,3,10
export SEED_START=0 SEED_END=4          # 4 random target-divider seeds per nerve
for d in duke_Ves/*/; do
  DUKE_SAMPLE_DIR="${d%/}" python -m experiments_v2.selectivity_sweep_duke
done
# -> outputs/duke_sweeps_fixed/<nerve>/data_seed_NNNN.json
#    outputs/duke_sweeps_fixed/<nerve>/sparse_sampling_seed_NNNN.json
```

**Cluster (SLURM, A100), the way the manuscript run was launched:**

```bash
DUKE_SWEEP_ROOT=outputs/duke_sweeps_fixed \
SPARSE_SAMPLING_SWEEP=true SEED_END=4 \
PULSE_SHAPE=biphasic_asym PW_MS=0.5 ASYM_RATIO=4.0 T_STOP=5.0 \
  bash slurm/submit_duke_sweep.sh          # shards nerves across a100 (+h100) pools
```

Per-seed JSONs are written incrementally and completed seeds are skipped, so a
re-submit only mops up missing work. Order-of-hours to a day for the full
cohort on one A100; the SLURM path parallelizes across GPUs. Key env vars are
documented in `slurm/run_duke_sweep.sbatch` (optimizer, amp init/clip, probe
magnitudes, sparse density ladder).

### 3.5 Activation-proxy vs NEURON validation JSON  (feeds the proxy number)

For the reviewer proxy-validation number (max |dSI| = 0.0045, 99.9% per-fiber
agreement over 8 nerves), re-score the optimized amplitudes with the full NEURON
population per nerve (needs a compiled pyfibers baseline + GPU for the jaxon side):

```bash
for s in human_sub-50_sam-2 human_sub-53_sam-2 human_sub-54_sam-2 human_sub-54_sam-3 \
         sub-8_sam-1 sub-10_sam-1 sub-11_sam-1 sub-13_sam-3; do
  python -m experiments_v2.neuron_pop_validate --sample "$s" --seed 0 \
      --sweep_root outputs/duke_sweeps_fixed
done
# -> outputs/reviewer_analyses/neuron_pop/<sample>_seed0.json
#    (fields: json_si, jaxon_si_full, neuron_si, agreement, ...)
```

### 3.6 Then run all the figures

Once `outputs/` is populated, produce every figure with the Path-A driver:

```bash
./reproduce_figures.sh
```

---

## 4. Figure / table / number -> command -> output -> runtime

All figure scripts are run **from the repo root** with the `jaxon` env
active. `${SW}` below is `outputs/duke_sweeps_fixed`; export it first:

```bash
export DUKE_SWEEP_ROOT=outputs/duke_sweeps_fixed
```

| Result | Command (Path A) | Output file(s) | Runtime |
|--------|------------------|----------------|---------|
| **Fig 1** validation | `python -m experiments_v2.figures_validation_main` | `manuscript/figures/main/fig1_validation.{png,svg}` | ~20 s |
| **Fig 2** phenomena | `python -m experiments_v2.figures_phenomena_main` | `manuscript/figures/main/fig2_phenomena.{png,svg}` | ~20 s |
| **Fig 3** duke | `DUKE_SWEEP_ROOT=$SW python -m experiments_v2.figures_duke_main` | `manuscript/figures/main/fig3_duke.{png,svg}` | ~1–2 min |
| **figS_khz_block** | `python -m experiments_v2.figures_phenomena_supp_khz` | `manuscript/figures/supp/figS_khz_block.{png,svg}` | ~15 s |
| **figS_optimizer_validation** | `python -m experiments_v2.fig_optimizer_validation` | `manuscript/figures/supp/figS_optimizer_validation.{png,svg}` | ~1–2 min (does a small autodiff/FD gradient check + a 30-step re-optimization on CPU) |
| **figA_\*** (29 anatomy/FEM pages) | `DUKE_SWEEP_ROOT=$SW python -m experiments_v2.figures_supp_anatomy` | `manuscript/figures/supp/figA_<nerve>.png` (×29) | ~2–5 min |
| **figS_\*** (29 selectivity/activation pages) | `DUKE_SWEEP_ROOT=$SW python -m experiments_v2.figures_supp_activation` | `manuscript/figures/supp/figS_<nerve>.{png,svg}` | ~2–5 min |
| **figS_density** | *see caveat box below* | `manuscript/figures/supp/figS_density.{png,svg}` | — |
| `tab:scaling` | (from `outputs/scaling/data_scaling.json`; rendered in Fig 1e panel) | Fig 1e + JSON | with Fig 1 |
| `tab:validation-sd / -sd-per-wf / -cv` | (from `outputs/{model}_validation/data_*_{sd,cv}.json`) | JSON, summarized in Fig 1 stdout | with Fig 1 |
| `tab:duke-cohort` + cohort penalty CSV | `python -m experiments_v2.analyze_sparse_sampling --csv outputs/duke_cohort.csv` | `outputs/duke_cohort.csv` + stdout table | ~30 s |

### Headline numbers -> command -> where it appears

| Number | How to reproduce |
|--------|------------------|
| **99.6% of 943 configs within 1% threshold error** | Config count (943) is printed by `python -m experiments_v2.figures_validation_main` (`… threshold rows`); the 99.6%-within-1% fraction is derived from the same strength-duration rows in `outputs/{mrg,sweeney,sundt,rattay}_validation/data_*_sd.json`. |
| **~820× geomean speedup @ N=1e5; 209–304× @ N=1000** | From `outputs/scaling/data_scaling.json` (PyFibers/JAXON wall-time ratios), rendered in Fig 1e by `figures_validation_main.py`. |
| **centroid penalty swine ~0.011 / human ~0.136; MW p=0.01, U=157, r=0.59; subject-level p=0.011; Friedman χ²=11.4, p=0.003** | `python -m experiments_v2.analyze_sparse_sampling` prints the per-species deployment-penalty table (dense SI, per-strategy SI, gap); the Mann–Whitney / Friedman statistics are also computed and printed by `DUKE_SWEEP_ROOT=$SW python -m experiments_v2.figures_duke_main` while rendering Fig 3. (`analyze_sparse_sampling.py` honors `DUKE_SWEEP_ROOT`, default `duke_sweeps_fixed`.) |
| **Fig3b Holm-adjusted recruitment p-values** | Printed by `DUKE_SWEEP_ROOT=$SW python -m experiments_v2.figures_duke_main` (per-nerve Wilcoxon, Holm-corrected across the 8 metric×species cells). |
| **activation-proxy vs NEURON: max \|dSI\|=0.0045, 99.9% agreement (8 nerves)** | Per-nerve JSONs in `outputs/reviewer_analyses/neuron_pop/*.json` (`agreement`, `neuron_si`, `jaxon_si_full` fields), produced by `neuron_pop_validate.py` (§3.5). |
| **block-Thomas vs dense-LU 5.6e-11 mV** | Numerical identity of the two extracellular solvers, verified directly in `jaxon/stim/extracellular_coupled.py` (no separate figure/data step). |

> ### Sweep-root handling
>
> Both cohort helper scripts read `DUKE_SWEEP_ROOT` (default
> `outputs/duke_sweeps_fixed`, the corrected tree):
>
> - **`analyze_sparse_sampling.py`** — prints the cohort penalty table for
>   `tab:duke-cohort`. Run `DUKE_SWEEP_ROOT=$SW python -m experiments_v2.analyze_sparse_sampling`
>   (or rely on the `duke_sweeps_fixed` default). `figures_duke_main.py` prints
>   the same penalty statistics while rendering Fig 3.
> - **`fig_optimizer_validation.py`** — panel (c) reads representative loss
>   histories from the sweep root; the gradient check loads
>   `duke_Ves/human_sub-50_sam-2` directly. Valid as long as the sweep tree and
>   `duke_Ves/` are present.
>
> ### ⚠️ `figS_density` has no in-repo scripted regenerator
>
> `manuscript/figures/supp/figS_density.{png,svg}` (deployment penalty vs
> fiber-sampling density, per nerve) was committed as a rendered asset; no script
> in `experiments_v2/` writes a file named `figS_density`. The closest scripted
> equivalents that render the same sparse-sampling-density data (and *do* honor
> `DUKE_SWEEP_ROOT`) are:
>
> ```bash
> DUKE_SWEEP_ROOT=$SW python -m experiments_v2.figures_sparse_main       # -> manuscript/figures/main/fig5_sparse_main.{png,svg}
> DUKE_SWEEP_ROOT=$SW python -m experiments_v2.figures_sparse_sampling   # -> manuscript/figures/duke/sparse_sampling/fig_sparse_{si_strips,gap}.png
> ```
>
> The driver (`reproduce_figures.sh`) therefore does **not** attempt to write
> `figS_density`; regenerate it from the sparse-sampling figures above if the
> committed asset is lost.

---

## 5. Load-bearing input directories (do not delete)

These feed a paper figure/number and must be present for reproduction:

- `outputs/duke_sweeps_fixed/` — cohort analysis (Fig 3, `tab:duke-cohort`, all penalty stats)
- `outputs/{mrg,sweeney,sundt,rattay}_validation/` — Fig 1, validation tables, 99.6%/943
- `outputs/scaling/` — Fig 1e, `tab:scaling`, speedup
- `outputs/{dc_block,depol_block,ap_collision}{,_rattay,_sundt,_sweeney}/` — Fig 2
- `outputs/khz_block{,_rattay,_sundt,_sweeney}/` — figS_khz_block
- `outputs/reviewer_analyses/` — optimizer validation inputs, neuron_pop proxy validation
- `duke_Ves/` — raw golgi FEM meshes (feeds figA_* anatomy and the fig_optimizer_validation gradient check; excluded from the lean Zenodo bundle)

Diameter spot-check trees `outputs/duke_sweeps_D7.3` and `_D10.0` are cited in the
Discussion (human penalty 0.066 at 7.3 µm, 0.054 at 10 µm) but are secondary.
