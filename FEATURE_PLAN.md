# Feature Plan — JAX Fiber Simulator

**Goal:** Elevate to Nature Communications quality — population-level statistics, additional fiber models, mixed A+C selectivity, and joint electrode+waveform optimization.

---

## Dependency Graph

```
Phase 1 (checkpointing) ──► Phase 1C (100+ seeds)
                                   │
Phase 2 (Schild validation)        │
                                   │
Phase 3 (Tigerholm port) ──────────┤
                                   ▼
                          Phase 4 (Mixed A+C selectivity)
                                   │
Phase 5 (JAX-native field) ────────▼
              Joint waveform + electrode optimization
```

**Suggested work order:** 1A → 1B → 1C → 2A/2B (parallel) → 3 → 4B → 4C → 5A → 5B

---

## Phase 1 — Gradient Checkpointing + Scale

**Why:** Current selectivity sweep (N=20 fibers, T=8 ms) peaks at ~36 GB on A16 (OOM). XLA's minimum with full rematerialization is ~10.4 GB (measured). Checkpointing brings runtime within 16 GB A16 budget and enables larger sweeps.

### 1A — `batch_solve.py`: checkpoint inner scan step

File: `jaxfibers/stim/batch_solve.py`  
Function: `_integrate_one_fiber_m_max`

Wrap the inner `lax.scan` body with `jax.checkpoint`:

```python
@functools.partial(jax.checkpoint, prevent_cse=False)
def _step(carry, t_idx):
    ...  # existing step body unchanged
```

This rematerialises activations on the backward pass instead of storing them — O(√T) peak memory instead of O(T).

### 1B — `optimizer.py`: checkpoint inside rect + waveform loops

File: `jaxfibers/optim/optimizer.py`  
Functions: `run_rect_optimization`, `run_waveform_optimization`

Apply the same `jax.checkpoint` decorator to the per-fiber scan step used inside each optimizer's gradient computation.

### 1C — `selectivity_sweep.py`: 100+ seeds + SLURM array

File: `experiments_v2/selectivity_sweep.py`  
File: `slurm/run_selectivity_sweep.sbatch`

- Change outer loop from 12 seeds to accept `SEED_START` / `SEED_END` env vars (or `SLURM_ARRAY_TASK_ID`)
- Each array task handles 10 seeds; 10 tasks × 10 seeds = 100 total
- Outputs: `outputs/selectivity_sweep/seed_{i:04d}.json`
- Add population-level analysis script: `experiments_v2/analyze_selectivity_sweep.py` — loads all JSONs, plots SI distribution (violin/CDF), reports median ± IQR

SLURM array sbatch:
```bash
#SBATCH --array=0-9
SEED_START=$((SLURM_ARRAY_TASK_ID * 10))
SEED_END=$((SEED_START + 10))
```

---

## Phase 2 — Schild Full Validation

**Why:** Schild94/97 are the most complete unmyelinated autonomic fiber models (11 NMODL mechanisms, Ca²⁺ dynamics, 3 pumps). Validation against PyFibers/NEURON across all diameters and pulse shapes is essential for credibility.

### 2A — `schild94_validation.py`

File: `experiments_v2/schild94_validation.py`  
Template: `experiments_v2/sweeney_validation.py` (same structure)

Tasks:
- SD curves: all available diameters × PW=[0.02, 0.05, 0.1, 0.2, 0.5, 1.0] ms × pulse shapes [mono_c, bi_ca]
- Conduction velocity: all diameters, 0.1 ms mono_c at 1.3× threshold
- Intracellular traces: mid-node Vm + Ca²⁺ gating variables at representative diameter
- Target max error: <1% threshold vs NEURON (relaxed from MRG 0.25% due to Ca²⁺ solver tolerance)

Ca²⁺ state variables must be carried through the scan as extra state beyond `(M, H, ...)`. The Schild solver already handles this via its extended state tuple.

### 2B — `schild97_validation.py`

Same as 2A but for the 1997 variant. Compare Schild94 vs Schild97 SD curves in a combined figure.

---

## Phase 3 — Tigerholm C-fiber Model

**Why:** Tigerholm (2014) is the state-of-the-art human C-nociceptor model. Including it alongside Sundt and Schild makes the C-fiber coverage comprehensive.

### 3A — `tigerholm_channels.py`

File: `jaxfibers/channels/tigerholm_channels.py`

Port 12 mechanisms from Tigerholm 2014 supplementary:
- Nav1.8 (m³h with slow inactivation s)
- Nav1.9 (m²h, very slow — time constants in seconds range)
- Nav1.7-TTX-s (fast, m³h)
- HCN (hyperpolarization-activated, single gate)
- KDR (delayed rectifier)
- KA (A-type, fast inactivation)
- KCa (Ca²⁺-dependent, requires [Ca²⁺]i)
- KD, KF, KS (three slow K conductances)
- Ca²⁺ L-type channel
- Ca²⁺ dynamics: `d[Ca]/dt = -alpha * I_Ca - [Ca]/tau_Ca`

**L'Hôpital note:** Several Tigerholm rate functions have `0/0` forms at specific voltages. Apply L'Hôpital analytically at each singularity (same pattern as MRG `alpha_m` at V=−47 mV).

### 3B — `tigerholm.py`

File: `jaxfibers/fibers/tigerholm.py`

Geometry: single-cable (all `is_node=True`), same structure as `sundt.py`.  
Parameters: d=0.8 µm, δz=1.7 µm (node spacing = diameter), Ra=100 Ω·cm, Cm=1 µF/cm².

### 3C — `tigerholm_validation.py`

File: `experiments_v2/tigerholm_validation.py`

- SD curves vs PyFibers/NEURON
- AP waveform comparison (Nav1.8 drives the characteristic slow repolarization)
- Ca²⁺ transient trace

---

## Phase 4 — Mixed A+C Fiber Selectivity

**Note:** Real nerve anatomy loading (`load_prevent_geometry()` from PREVENT IHC data) is **DEFERRED** to a later phase. Phase 4 uses synthetic nerve geometry with explicitly mixed fiber type assignment.

### 4B — Mixed batch solver

Files: `jaxfibers/stim/batch_solve.py`

Two strategies for handling different `n_comp` across fiber types:

**Option A (recommended):** Pad shorter fibers to the longest `n_comp` with zero-conductance ghost compartments. Single vmap over all fibers. Simpler; minimal memory overhead since ghost compartments contribute nothing.

**Option B:** Separate vmaps per fiber type (MRG A-fibers + Sundt/Tigerholm C-fibers), then concatenate `acts = jnp.concatenate([acts_A, acts_C])`. More complex but exact.

Implement Option A. Add `pad_fiber_statics(statics_list, target_n_comp)` helper.

### 4C — `selectivity_sweep_mixed.py`

File: `experiments_v2/selectivity_sweep_mixed.py`

- Nerve: 10 MRG A-fibers (D sampled from [7.3, 10.0, 12.8, 14.0] µm) + 10 Sundt/Tigerholm C-fibers (D=0.8 µm)
- Inner-circle targets: select target fibers by spatial position, not type
- Compute SI_A (A-fiber activation fraction) and SI_C (C-fiber activation fraction) separately
- Goal: maximize SI_A while minimizing SI_C (fiber-type selectivity)
- Metrics: `SI_type = (mean_acts_A_target - mean_acts_C_target) / (mean_acts_A_target + mean_acts_C_target + eps)`

---

## Phase 5 — Joint Waveform + Electrode Position Optimization

**Why:** Current optimization holds electrode positions fixed and optimizes waveform amplitudes only. Making the electrode geometry differentiable enables simultaneous optimization — a qualitatively stronger result.

### 5A — JAX-native field computation

File: `jaxfibers/stim/multichannel_field.py`  
Function: `point_source_potentials_mV` (or `precompute_ve_unit`)

Replace `numpy` array ops with `jnp` equivalents in the distance/potential computation:

```python
# OLD (numpy — not differentiable w.r.t. contact_xyz_um):
r = np.sqrt(np.sum((fiber_xyz - contact_xyz)**2, axis=-1))
Ve = 1e3 / (4 * np.pi * sigma_S_m) * (i0_mA * 1e-3) / (r * 1e-6)

# NEW (jnp — differentiable):
r = jnp.sqrt(jnp.sum((fiber_xyz - contact_xyz)**2, axis=-1) + 1e-12)
Ve = 1e3 / (4 * jnp.pi * sigma_S_m) * (i0_mA * 1e-3) / (r * 1e-6)
```

The `1e-12` regularizer prevents division-by-zero when a contact coincides with a node.

Expose `contact_xyz_um` as a JAX array parameter threaded through `batch_integrate_m_max`.

### 5B — `selectivity_joint_opt.py`

File: `experiments_v2/selectivity_joint_opt.py`

Optimization over: `(amps [K], contact_xyz_um [K × 3])` jointly.

- Warm-start: rect-only solution from `run_rect_optimization`
- Clip contact positions to valid anatomical range (e.g., cuff radius ± 200 µm, axial ± 2 mm)
- Learning rates: separate Adam groups — `lr_amp=1e-2`, `lr_pos=10 µm/step`
- Report: SI improvement vs rect-only, total electrode displacement from initial positions

---

## Success Metrics (Nature Communications targets)

| Metric | Target |
|--------|--------|
| Fiber models validated | 6+ (MRG, MRG-interp, Sweeney, Sundt, Schild94, Schild97, Tigerholm) |
| Schild max threshold error vs NEURON | <1% across all diameters |
| Tigerholm max threshold error vs NEURON | <1% |
| Population SI (100 seeds, rect) | median ± IQR reported |
| Mixed A+C SI_type improvement | >0.3 units over baseline |
| Joint opt SI improvement over rect | >0.1 units |
| Peak GPU memory (N=20 fibers, T=8ms) | <16 GB (A16-compatible after checkpointing) |

---

## Status

| Phase | Task | Status |
|-------|------|--------|
| 1 | 1A batch_solve checkpoint | TODO |
| 1 | 1B optimizer checkpoint | TODO |
| 1 | 1C 100+ seeds + array job | TODO |
| 2 | 2A Schild94 validation | TODO |
| 2 | 2B Schild97 validation | TODO |
| 3 | 3A Tigerholm channels | TODO |
| 3 | 3B Tigerholm geometry | TODO |
| 3 | 3C Tigerholm validation | TODO |
| 4 | 4A Real nerve geometry | **DEFERRED** |
| 4 | 4B Mixed batch solver | TODO |
| 4 | 4C Mixed A+C sweep | TODO |
| 5 | 5A JAX-native field | TODO |
| 5 | 5B Joint opt experiment | TODO |
