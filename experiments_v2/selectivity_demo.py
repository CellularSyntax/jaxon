"""Spatial selectivity optimisation — local demo (capped for ~2 min runtime).

Demonstrates gradient-based optimisation of extracellular stimulation parameters
for selective activation of a target fascicle in a synthetic vagus nerve, using
the JAX double-cable (Vi, Vpax) MRG solver as the differentiable forward model.

Setup
-----
- Synthetic nerve: 6 MRG fibers, all D=10 µm, explicitly placed
    - Target (blue):     fibers 0, 1, 5  — right/upper sector
    - Off-target (red):  fibers 2, 3, 4  — left/lower sector
- 6-contact ring cuff, 1 mm radius, aligned with fiber midpoint
- Two optimisation runs:
    A) Rectangular pulses  —  6 DOF (one amplitude per contact, shared 0.1 ms pulse)
    B) Arbitrary waveforms — 6 × T DOF (warm-started from rect solution)

Sign convention (matching validation suite)
-------------------------------------------
  amps < 0 = cathodic (depolarising)   amps > 0 = anodic (hyperpolarising)
  amp_clip = (-3, 3) allows steering waveforms.

Outputs
-------
outputs/selectivity_demo/
  fig_nerve_layout.png       — cross-section map (fiber positions + contacts)
  fig_rect_results.png       — rect optimisation: loss, activation bar, waveforms
  fig_waveform_results.png   — waveform optimisation: loss, activation bar, waveforms
  data_selectivity_demo.json — numerical summary

Runtime: ~90 s on CPU (2.6 GHz, JAX jit; JIT compile ≈10 s, rest ≈80 s)

Run from project root:
    python experiments_v2/selectivity_demo.py
"""
from __future__ import annotations

import sys
import pathlib
import json
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jaxon.nerve.geometry import NerveGeometry
from jaxon.stim.multichannel_field import make_ring_cuff_positions, precompute_ve_unit
from jaxon.stim.batch_solve import stack_fiber_statics, initial_states_batch
from jaxon.optim.optimizer import run_rect_optimization, run_waveform_optimization
from jaxon.fibers.mrg import section_centers_um
from experiments_v2.utils import ensure_dir

OUT = ensure_dir(ROOT / "outputs" / "selectivity_demo")

# ─────────────────────────────────────────────────────── demo parameters ─────
N_CONTACTS      = 6
CUFF_RADIUS_UM  = 1500.0   # 1.5 mm cuff radius (wider amplitude window, avoids depol block)
DT              = 0.005    # ms
T_STOP          = 3.0      # ms
DELAY_MS        = 0.5      # ms pulse onset
PW_MS           = 0.1      # ms pulse width
N_OPT_RECT      = 50       # iterations for rectangular-pulse optimisation
N_OPT_WAVE      = 50       # iterations for arbitrary-waveform optimisation

# ─────────────────────────────────────────── explicit nerve cross-section ────
# 6 fibers: target (right/upper sector) and off-target (left/lower sector).
# All D=10 µm for equal thresholds — selectivity comes purely from geometry.
#
#         C1 (60°)
#  C2          C0 (0°)  ← right, close to targets 0,1
#   [T0]  [T1]
#   [OT2]  [OT3]
#  C3          C5
#         C4
#
nerve = NerveGeometry(
    n_fibers=6,
    fiber_x_um=np.array([350.,  200.,  -350.,  -200.,   -50.,  150.]),
    fiber_y_um=np.array([  0.,  250.,     0.,  -250.,  -200.,  200.]),
    fiber_diam=np.array([10.,   10.,    10.,    10.,    10.,   10.]),
    target_mask=np.array([True, True,  False,  False,  False,  True]),
)
N_FIBERS = nerve.n_fibers
print(f"Nerve: {N_FIBERS} fibers, {nerve.target_mask.sum()} target, "
      f"{(~nerve.target_mask).sum()} off-target")

# ─────────────────────────────────────────────────── build cuff & fields ─────
contact_xyz = make_ring_cuff_positions(
    n_contacts=N_CONTACTS, cuff_radius_um=CUFF_RADIUS_UM, cuff_z_um=0.0
)

# First pass to get fiber midpoint z, then recentre cuff
Ve_unit, node_indices, geoms = precompute_ve_unit(
    nerve_geom=nerve, n_nodes=21, contact_xyz_um=contact_xyz
)
mid_comp = node_indices[0, len(node_indices[0]) // 2]
mid_z_um = float(np.array(section_centers_um(geoms[0]))[mid_comp])
contact_xyz[:, 2] = mid_z_um
Ve_unit, node_indices, geoms = precompute_ve_unit(
    nerve_geom=nerve, n_nodes=21, contact_xyz_um=contact_xyz
)
print(f"Ve_unit shape: {Ve_unit.shape}   n_comp per fiber: {geoms[0].n_comp}")

# ─────────────────────────────────────────── build solver statics ────────────
print("Building solver statics and initial states ...")
fs_batch  = stack_fiber_statics(geoms, DT)
s0_batch  = initial_states_batch(geoms)

# ─────────────────────────────────────────── shared pulse mask ───────────────
N_STEPS = int(T_STOP / DT)
t_grid  = (np.arange(N_STEPS) + 1) * DT
pulse_mask = np.where(
    (t_grid >= DELAY_MS) & (t_grid < DELAY_MS + PW_MS), 1.0, 0.0
).astype(np.float64)

Ve_unit_j  = jnp.asarray(Ve_unit, dtype=jnp.float64)
pulse_j    = jnp.asarray(pulse_mask)
node_idx_j = jnp.asarray(node_indices, dtype=jnp.int32)

# ─────────────────────────────────────────── RECT optimisation ───────────────
print("\n=== A: Rectangular pulse amplitude optimisation ===")
t_rect_start = time.time()
rect_result  = run_rect_optimization(
    fiber_statics_batch=fs_batch,
    state0_batch=s0_batch,
    Ve_unit=Ve_unit_j,
    pulse_mask=pulse_j,
    node_indices=node_indices,
    target_mask=nerve.target_mask,
    dt=DT,
    n_steps=N_OPT_RECT,
    lr=8e-2,
    amp_init_mA=-0.4,
    amp_clip=(-2.5, 2.5),   # allow anodic for field steering; <-2.5 risks depol block
    verbose=True,
)
t_rect = time.time() - t_rect_start
print(f"\n  Rect: {t_rect:.1f}s | final SI = {rect_result['history']['si'][-1]:+.3f}")
print(f"  Final amplitudes (mA): {np.round(rect_result['amps'], 3)}")

# ─────────────────────────────────────────── WAVEFORM optimisation ───────────
print("\n=== B: Arbitrary waveform optimisation ===")
# Warm-start from rect solution: broadcast rect amps over the pulse window
u_init = np.zeros((N_CONTACTS, N_STEPS), dtype=np.float64)
for k in range(N_CONTACTS):
    u_init[k] = float(rect_result["amps"][k]) * pulse_mask

t_wave_start = time.time()
wave_result  = run_waveform_optimization(
    fiber_statics_batch=fs_batch,
    state0_batch=s0_batch,
    Ve_unit=Ve_unit_j,
    node_indices=node_indices,
    target_mask=nerve.target_mask,
    dt=DT,
    T=N_STEPS,
    n_steps=N_OPT_WAVE,
    lr=5e-3,
    u_init=u_init,
    u_clip=(-2.5, 2.5),
    verbose=True,
)
t_wave = time.time() - t_wave_start
print(f"\n  Waveform: {t_wave:.1f}s | final SI = {wave_result['history']['si'][-1]:+.3f}")

# ─────────────────────────────────────────────── save JSON ───────────────────
data = {
    "nerve": {
        "n_fibers": N_FIBERS,
        "fiber_x_um": nerve.fiber_x_um.tolist(),
        "fiber_y_um": nerve.fiber_y_um.tolist(),
        "fiber_diam": nerve.fiber_diam.tolist(),
        "target_mask": nerve.target_mask.tolist(),
    },
    "cuff": {"contact_xyz_um": contact_xyz.tolist()},
    "rect": {
        "amps_mA": rect_result["amps"].tolist(),
        "loss_history": rect_result["history"]["loss"],
        "si_history": rect_result["history"]["si"],
        "final_si": rect_result["history"]["si"][-1],
        "acts_history": [a.tolist() for a in rect_result["history"]["acts"]],
        "time_s": t_rect,
    },
    "waveform": {
        "loss_history": wave_result["history"]["loss"],
        "si_history": wave_result["history"]["si"],
        "final_si": wave_result["history"]["si"][-1],
        "acts_history": [a.tolist() for a in wave_result["history"]["acts"]],
        "time_s": t_wave,
    },
}
out_json = OUT / "data_selectivity_demo.json"
with open(out_json, "w") as f:
    json.dump(data, f, indent=2)

# ─────────────────────────────────────────────────────────── figures ──────────
fiber_colors = ["C0" if t else "C1" for t in nerve.target_mask]

# ── Fig 1: nerve cross-section layout ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_aspect("equal")
nerve_circle = plt.Circle((0, 0), 500, fill=False, edgecolor="k", lw=1.5)
ax.add_patch(nerve_circle)
for f in range(N_FIBERS):
    fc = plt.Circle(
        (nerve.fiber_x_um[f], nerve.fiber_y_um[f]),
        nerve.fiber_diam[f] / 2 * 15,
        facecolor=fiber_colors[f], edgecolor="k", lw=0.5, alpha=0.85,
    )
    ax.add_patch(fc)
    ax.text(nerve.fiber_x_um[f], nerve.fiber_y_um[f], f"F{f}",
            ha="center", va="center", fontsize=7, color="w", fontweight="bold")
for k, (cx, cy, _) in enumerate(contact_xyz):
    ax.plot(cx, cy, "k^", ms=9)
    ax.text(cx * 1.1, cy * 1.1, f"C{k}", fontsize=8, ha="center")
leg = [mpatches.Patch(facecolor="C0", label="Target"),
       mpatches.Patch(facecolor="C1", label="Off-target")]
ax.legend(handles=leg, loc="upper right", fontsize=9)
ax.set_xlim(-CUFF_RADIUS_UM * 1.25, CUFF_RADIUS_UM * 1.25)
ax.set_ylim(-CUFF_RADIUS_UM * 1.25, CUFF_RADIUS_UM * 1.25)
ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
ax.set_title("Nerve cross-section & ring cuff contacts")
fig.tight_layout(); fig.savefig(OUT / "fig_nerve_layout.png", dpi=150); plt.close(fig)

# ── Fig 2: rect results ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
ax = axes[0]
ax.plot(rect_result["history"]["loss"], "C0", lw=1.5)
ax2 = ax.twinx()
ax2.plot(rect_result["history"]["si"], "C3--", lw=1.5)
ax.set_xlabel("Iteration"); ax.set_ylabel("WQ loss", color="C0"); ax2.set_ylabel("SI", color="C3")
ax.set_title("(A) Training curves")

ax = axes[1]
acts_r = np.array(rect_result["history"]["acts"][-1])
ax.bar(np.arange(N_FIBERS), acts_r, color=fiber_colors, edgecolor="k", lw=0.6)
ax.axhline(0.5, color="gray", ls="--", lw=0.8)
ax.set_xticks(np.arange(N_FIBERS)); ax.set_xticklabels([f"F{f}" for f in range(N_FIBERS)])
ax.set_ylabel("Activation proxy"); ax.set_ylim(0, 1.05)
ax.set_title(f"(B) Activation — SI = {rect_result['history']['si'][-1]:+.3f}")

ax = axes[2]
ax.bar(np.arange(N_CONTACTS), rect_result["amps"],
       color=["C2" if a < 0 else "C3" for a in rect_result["amps"]],
       edgecolor="k", lw=0.6)
ax.axhline(0, color="k", lw=0.5)
ax.set_xticks(np.arange(N_CONTACTS)); ax.set_xticklabels([f"C{k}" for k in range(N_CONTACTS)])
ax.set_ylabel("Amplitude (mA)")
ax.set_title("(C) Optimised amplitudes  (blue=cathodic, red=anodic)")
fig.suptitle("Rectangular pulse amplitude optimisation", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "fig_rect_results.png", dpi=150); plt.close(fig)

# ── Fig 3: waveform results ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
ax = axes[0]
ax.plot(wave_result["history"]["loss"], "C0", lw=1.5)
ax2 = ax.twinx()
ax2.plot(wave_result["history"]["si"], "C3--", lw=1.5)
ax.set_xlabel("Iteration"); ax.set_ylabel("WQ loss", color="C0"); ax2.set_ylabel("SI", color="C3")
ax.set_title("(A) Training curves")

ax = axes[1]
acts_w = np.array(wave_result["history"]["acts"][-1])
ax.bar(np.arange(N_FIBERS), acts_w, color=fiber_colors, edgecolor="k", lw=0.6)
ax.axhline(0.5, color="gray", ls="--", lw=0.8)
ax.set_xticks(np.arange(N_FIBERS)); ax.set_xticklabels([f"F{f}" for f in range(N_FIBERS)])
ax.set_ylabel("Activation proxy"); ax.set_ylim(0, 1.05)
ax.set_title(f"(B) Activation — SI = {wave_result['history']['si'][-1]:+.3f}")

ax = axes[2]
u_opt = wave_result["u"]
cmap = plt.get_cmap("tab10")
for k in range(N_CONTACTS):
    ax.plot(t_grid, u_opt[k], color=cmap(k), lw=0.9, label=f"C{k}")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("Time (ms)"); ax.set_ylabel("Amplitude (mA)")
ax.set_title("(C) Optimised waveforms per contact")
ax.legend(fontsize=7, ncol=2, loc="upper right")
fig.suptitle("Arbitrary waveform optimisation (warm-started from rect)", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "fig_waveform_results.png", dpi=150); plt.close(fig)

# ─────────────────────────────────────────────── final summary ───────────────
print(f"\n{'='*55}")
print(f"Outputs saved to: {OUT}")
print(f"{'='*55}")
best_si_rect = max(rect_result['history']['si'])
best_si_wave = max(wave_result['history']['si'])
print(f"  Rect:     SI {rect_result['history']['si'][0]:+.3f} -> best {best_si_rect:+.3f}  ({t_rect:.1f}s)")
print(f"  Waveform: SI {wave_result['history']['si'][0]:+.3f} -> best {best_si_wave:+.3f}  ({t_wave:.1f}s)")
