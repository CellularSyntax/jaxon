"""Waveform optimization demo — Task 10.1 (Phase 5, migration_plan.md v3)

Gradient-based minimization of stimulation energy subject to an AP constraint,
using the MRG coupled (Vi, Vpax) solver and exact jax.grad (no FD).

Task formulation:
  minimize   energy(θ)  =  Σ_k θ_k² · dt_bin            [mA²·ms]
  s.t.       V_peak(θ)  ≥  V_thresh                       [AP fires]

  Encoded as  loss = energy + λ · relu(V_thresh − V_peak)²

Waveform parametrisation:
  K = 40 piecewise-constant bins, each 50 µs wide, spanning a 2 ms window.
  θ_k in mA (negative = cathodic). Optimizer may discover anodic phases.

Baseline comparison:
  A 0.1 ms rectangular pulse at threshold (minimum-energy standard waveform)
  is shown alongside the optimized waveform to quantify the energy gain.

Model: MRG D=10 µm, 21 nodes, extracellular point source 1 mm above centre,
       σ = 0.3 S/m, dt = 0.005 ms, tstop = 5 ms, T = 37 °C.

Outputs:
  outputs/fig_optimization_waveform.png   (3-panel)
  outputs/data_optimization_waveform.json

Run:
    conda run -n jaxley_fibers python experiments/exp_optimization_waveform.py
"""

from __future__ import annotations
import sys, pathlib, json, time, dataclasses
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax
from jaxley.solver_gate import solve_gate_exponential

from jaxfibers.fibers.mrg import (
    build_mrg, node_indices, section_centers_um,
    V_REST, CM_AXON, G_PAS_MYSA, G_PAS_FLUT, G_PAS_STIN,
)
from jaxfibers.channels.mrg_axnode import AxnodeMyel
from mrg_extracellular_coupled import arrays_from_geometry, integrate, find_threshold
from jaxfibers.objectives import max_vm, energy

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
CELSIUS  = 37.0
DT       = 0.005      # ms
TSTOP    = 5.0        # ms
DELAY    = 1.0        # ms  lead-in before pulse window
PULSE_MS = 2.0        # ms  pulse window width
K_BINS   = 40         # free amplitude bins (50 µs each)
BIN_DT   = PULSE_MS / K_BINS   # 0.05 ms = 50 µs per bin

SIGMA_SM = 0.3        # S/m  medium conductivity
DIAM     = 10.0       # µm   MRG fiber diameter
N_NODES  = 21
N_STEPS  = int(TSTOP / DT)   # 1000

DEPTH_MM  = 1.0       # mm  electrode depth
V_THRESH  = -20.0     # mV  AP detection threshold
LAMBDA    = 1e4       # hinge penalty weight
L1_LAMBDA = 8e-3      # sparsity penalty — forces late/redundant bins toward zero
                      # L1 favours shorter pulses: 4 bins × amp > 40 bins × (amp/10)
                      # in L1 cost → optimizer discovers concentrated waveforms

LR       = 5e-3
N_EPOCHS = 600

# MRG channel constants
GNABAR = 3.0; GNAPBAR = 0.01; GKBAR = 0.08; GL = 0.007
ENA = 50.0; EK = -90.0; EL = -90.0

# ── build fiber ────────────────────────────────────────────────────────────────
print("Building MRG D=10 µm, 21-node fiber ...")
_, geom  = build_mrg(diameter=DIAM, n_nodes=N_NODES)
centers  = np.array(section_centers_um(geom))
nidx     = node_indices(geom)
mid_comp = nidx[len(nidx) // 2]

geom_c = dataclasses.replace(geom, cm_uF_cm2=[CM_AXON] * geom.n_comp)
static = arrays_from_geometry(geom_c, DT)
is_node   = static["is_node"]
A_in      = static["A_in_cm2"]

stype_gp  = {"node": 0.0, "mysa": G_PAS_MYSA, "flut": G_PAS_FLUT, "stin": G_PAS_STIN}
g_pas_arr = jnp.asarray([stype_gp[s] for s in geom.section_type])

v0 = jnp.float64(V_REST)
(am, bm), (ah, bh), (amp0, bmp0), (as0, bs0) = AxnodeMyel._alpha_beta(v0, CELSIUS)
state0 = (
    jnp.where(is_node, float(am  / (am  + bm)),     0.0),
    jnp.where(is_node, float(ah  / (ah  + bh)),     0.0),
    jnp.where(is_node, float(amp0 / (amp0 + bmp0)), 0.0),
    jnp.where(is_node, float(as0 / (as0 + bs0)),    0.0),
)

def membrane_fn(Vm, state, dt_):
    M, H, MP, S = state
    (am, bm), (ah, bh), (amp_, bmp), (as_, bs_) = AxnodeMyel._alpha_beta(Vm, CELSIUS)
    M2  = solve_gate_exponential(M,  dt_, am,   bm)
    H2  = solve_gate_exponential(H,  dt_, ah,   bh)
    MP2 = solve_gate_exponential(MP, dt_, amp_, bmp)
    S2  = solve_gate_exponential(S,  dt_, as_,  bs_)
    gna  = GNABAR * M2**3 * H2; gnap = GNAPBAR * MP2**3; gk = GKBAR * S2
    g_node = (gna + gnap + gk + GL) * A_in * 1e6
    i_node = ((gna + gnap)*(Vm - ENA) + gk*(Vm - EK) + GL*(Vm - EL)) * A_in * 1e6
    g_pas  = g_pas_arr * A_in * 1e6
    i_pas  = g_pas * (Vm - V_REST)
    return (jnp.where(is_node, g_node, g_pas),
            jnp.where(is_node, i_node, i_pas),
            (M2, H2, MP2, S2))

def mfn_factory():
    return membrane_fn, state0

# ── extracellular profile ─────────────────────────────────────────────────────
dy = DEPTH_MM * 1e-3                               # m
dz = (centers[mid_comp] - centers) * 1e-6          # m
r  = np.sqrt(dy**2 + dz**2)
Ve_unit = jnp.array(-1.0 / (4.0 * np.pi * SIGMA_SM * r))  # mV at i0=−1 mA

# ── bin-assignment matrix B [n_steps, K_BINS] ─────────────────────────────────
# B[s, k] = 1 if timestep s belongs to bin k (zero outside pulse window).
pulse_start = round(DELAY / DT)           # step 200
pulse_end   = round((DELAY + PULSE_MS) / DT)  # step 600  (400 active steps)
n_active    = pulse_end - pulse_start     # = 400

B = np.zeros((N_STEPS, K_BINS), dtype=np.float64)
steps_per_bin = n_active // K_BINS        # = 10
for k in range(K_BINS):
    s0 = pulse_start + k * steps_per_bin
    s1 = s0 + steps_per_bin
    B[s0:s1, k] = 1.0
B_j = jnp.asarray(B)

# ── find threshold ─────────────────────────────────────────────────────────────
# Use a 0.1 ms square pulse (standard comparison) to find threshold.
pm_ref = np.zeros(N_STEPS, dtype=np.float64)
pm_ref[pulse_start : pulse_start + round(0.1 / DT)] = 1.0
pm_ref_j = jnp.array(pm_ref)

print("Finding threshold (0.1 ms square pulse) ...")
thr_mA = find_threshold(static, mfn_factory, Ve_unit, pm_ref_j, DT,
                         v_thresh=V_THRESH, v_rest=V_REST)
print(f"  threshold = {thr_mA:.5f} mA")

# ── baseline reference energy ──────────────────────────────────────────────────
# Energy of a 0.1 ms rectangular pulse at threshold (the canonical comparison).
E_ref_mA2ms = thr_mA**2 * 0.1     # I²·PW  (mA²·ms)
print(f"  baseline energy (0.1 ms square @ thr) = {E_ref_mA2ms:.6f} mA²·ms")

# ── loss function ─────────────────────────────────────────────────────────────
def forward(theta):
    """scalar loss for waveform parameters θ [K_BINS] (mA per bin)."""
    waveform = B_j @ theta              # (N_STEPS,) amplitude in mA
    mask     = waveform / (-1.0)        # signed multiplier on Ve_unit
    trace, _ = integrate(static, membrane_fn, state0, Ve_unit, mask, DT,
                         v_rest=V_REST, center_comp=mid_comp)
    vpeak = max_vm(trace)
    e     = energy(waveform, DT)
    hinge = jax.nn.relu(V_THRESH - vpeak)
    l1    = jnp.sum(jnp.abs(theta))    # sparsity: penalise total active duration
    loss  = e + LAMBDA * hinge ** 2 + L1_LAMBDA * l1
    return loss, (vpeak, e, l1)

value_and_grad = jax.jit(jax.value_and_grad(forward, has_aux=True))

# Convenience: run forward without grad (for recording traces)
@jax.jit
def get_trace(theta):
    waveform = B_j @ theta
    mask     = waveform / (-1.0)
    trace, _ = integrate(static, membrane_fn, state0, Ve_unit, mask, DT,
                         v_rest=V_REST, center_comp=mid_comp)
    return trace    # (N_STEPS,) center-node Vm

# ── initial waveform ──────────────────────────────────────────────────────────
# Flat cathodic pulse at 1.2× threshold across all 40 bins.
# The L1 sparsity penalty drives redundant (late-window) bins toward zero, so the
# optimizer discovers that a shorter, concentrated waveform achieves the same AP
# with lower combined energy + L1 cost.
theta0 = jnp.full((K_BINS,), thr_mA * 1.2, dtype=jnp.float64)
print(f"\nInitial energy (flat 2ms @ 1.2×thr): {float(energy(B_j @ theta0, DT)):.5f} mA²·ms")
print(f"Initial V_peak: {float(max_vm(get_trace(theta0))):.1f} mV")

# ── optimization ──────────────────────────────────────────────────────────────
theta = theta0
opt   = optax.adam(LR)
state = opt.init(theta)

losses, peaks, energies_hist, l1_hist = [], [], [], []
print(f"\nOptimizing K={K_BINS} bins ({BIN_DT*1e3:.0f} µs each), {N_EPOCHS} epochs, lr={LR} ...")
print(f"  Loss = energy + {LAMBDA:.0e}×hinge² + {L1_LAMBDA:.0e}×L1(θ)")
t0 = time.perf_counter()

for epoch in range(N_EPOCHS):
    (loss, (vpeak, e, l1)), grads = value_and_grad(theta)
    updates, state = opt.update(grads, state, theta)
    theta = optax.apply_updates(theta, updates)
    losses.append(float(loss))
    peaks.append(float(vpeak))
    energies_hist.append(float(e))
    l1_hist.append(float(l1))
    if epoch % 60 == 0 or epoch == N_EPOCHS - 1:
        n_active = int(np.sum(np.abs(np.array(theta)) > 5e-3))
        print(f"  epoch {epoch:3d}  loss={float(loss):.5f}  "
              f"V_peak={float(vpeak):+.1f} mV  energy={float(e):.5f}  "
              f"L1={float(l1):.4f}  bins|>5mA|={n_active}")

wall = time.perf_counter() - t0
print(f"Optimization wall-time: {wall:.1f} s")

theta_init  = np.array(theta0)
theta_final = np.array(theta)

E_init  = energies_hist[0]
E_final = energies_hist[-1]
print(f"\nEnergy:  initial={E_init:.5f}  final={E_final:.5f}  "
      f"reduction={E_init/E_final:.1f}×  vs baseline={E_final/E_ref_mA2ms:.2f}×baseline")

# ── traces ────────────────────────────────────────────────────────────────────
trace_init  = np.array(get_trace(theta0))
trace_final = np.array(get_trace(theta))

t_vec = (np.arange(N_STEPS) + 1) * DT   # ms, steps 1..N_STEPS

# Waveform time axis (bin centres in ms)
bin_edges = DELAY + np.arange(K_BINS + 1) * BIN_DT

# ── figure ────────────────────────────────────────────────────────────────────
print("\n=== Generating figure ===")
fig = plt.figure(figsize=(14, 5.5), constrained_layout=True)
gs  = fig.add_gridspec(1, 3)
ax_w = fig.add_subplot(gs[0, 0])
ax_v = fig.add_subplot(gs[0, 1])
ax_l = fig.add_subplot(gs[0, 2])

# ── Panel A: Waveforms ────────────────────────────────────────────────────────
ax_w.stairs(theta_init,  bin_edges, color="C0", lw=2.0, label="initial (flat 1.2×thr)")
ax_w.stairs(theta_final, bin_edges, color="C3", lw=2.0, label="optimized")
ax_w.axhline(0, color="k", lw=0.5, alpha=0.5)
ax_w.axhline(thr_mA, color="C0", ls=":", lw=1.0, alpha=0.6, label=f"thr={thr_mA:.3f} mA")
ax_w.set_xlim(0.8, DELAY + PULSE_MS + 0.3)
ax_w.set_xlabel("time (ms)")
ax_w.set_ylabel("amplitude (mA)")
ax_w.legend(fontsize=8, loc="lower right")
ax_w.grid(alpha=0.25)
ax_w.set_title(f"A  Waveform  (K={K_BINS} bins, {BIN_DT*1e3:.0f} µs each)",
               fontsize=10, fontweight="bold")

# Energy annotations
ax_w.annotate(f"E_init = {E_init:.4f} mA²·ms",
              xy=(0.04, 0.96), xycoords="axes fraction", color="C0",
              fontsize=8, va="top")
ax_w.annotate(f"E_final = {E_final:.4f} mA²·ms  ({E_init/E_final:.1f}× reduction)",
              xy=(0.04, 0.89), xycoords="axes fraction", color="C3",
              fontsize=8, va="top")
ax_w.annotate(f"E_ref (0.1ms sq) = {E_ref_mA2ms:.4f} mA²·ms",
              xy=(0.04, 0.82), xycoords="axes fraction", color="k",
              fontsize=8, va="top")

# ── Panel B: Vm traces ───────────────────────────────────────────────────────
ax_v.plot(t_vec, trace_init,  color="C0", lw=1.5, label=f"initial  Vpeak={trace_init.max():.1f} mV")
ax_v.plot(t_vec, trace_final, color="C3", lw=1.5, label=f"optimized  Vpeak={trace_final.max():.1f} mV")
ax_v.axhline(V_THRESH, color="k", ls="--", lw=0.8, alpha=0.6, label=f"V_thresh={V_THRESH} mV")
ax_v.set_xlim(0, TSTOP)
ax_v.set_xlabel("time (ms)")
ax_v.set_ylabel("V_m center node (mV)")
ax_v.legend(fontsize=8)
ax_v.grid(alpha=0.25)
ax_v.set_title("B  Center-node Vm trace", fontsize=10, fontweight="bold")

# ── Panel C: Loss / energy / peak Vm ─────────────────────────────────────────
ep = np.arange(N_EPOCHS)
ax_l.plot(ep, losses,        color="k",  lw=1.5, label="total loss")
ax_l.plot(ep, energies_hist, color="C2", lw=1.5, label="energy (∫I²dt)")
ax_l.plot(ep, [L1_LAMBDA * v for v in l1_hist], color="C1", lw=1.2, ls="--",
          label=f"L1 term (λ={L1_LAMBDA:.0e})")
ax_l.set_yscale("log"); ax_l.set_xlabel("epoch")
ax_l.set_ylabel("loss / energy (mA²·ms)", color="k")
ax_l.legend(fontsize=7, loc="upper right")
ax_l.grid(alpha=0.25)

ax_r = ax_l.twinx()
ax_r.plot(ep, peaks, color="C0", lw=1.5, alpha=0.8, label="V_peak")
ax_r.axhline(V_THRESH, color="C0", ls="--", lw=0.8, alpha=0.6)
ax_r.set_ylabel("V_peak center node (mV)", color="C0", fontsize=9)
ax_r.tick_params(axis="y", labelcolor="C0")
ax_r.legend(fontsize=7, loc="center right")
ax_l.set_title(f"C  Optimization curve  (Adam lr={LR}, λ_hinge={LAMBDA:.0e}, λ_L1={L1_LAMBDA:.0e})",
               fontsize=9, fontweight="bold")

fig.suptitle(
    f"Gradient-based waveform optimization through MRG coupled solver\n"
    f"(D={DIAM}µm, {N_NODES} nodes, {DEPTH_MM}mm depth, K={K_BINS} bins × {BIN_DT*1e3:.0f}µs)",
    fontsize=11,
)

fig_path = OUT / "fig_optimization_waveform.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {fig_path}")

# ── JSON ───────────────────────────────────────────────────────────────────────
data = {
    "model":        {"diam_um": DIAM, "n_nodes": N_NODES, "dt_ms": DT,
                     "tstop_ms": TSTOP, "celsius": CELSIUS},
    "setup":        {"depth_mm": DEPTH_MM, "delay_ms": DELAY, "pulse_ms": PULSE_MS,
                     "K_bins": K_BINS, "bin_dt_ms": BIN_DT},
    "optimizer":    {"lr": LR, "n_epochs": N_EPOCHS, "lambda": LAMBDA},
    "threshold_mA": float(thr_mA),
    "E_ref_mA2ms":  float(E_ref_mA2ms),
    "E_init":       float(E_init),
    "E_final":      float(E_final),
    "energy_reduction_x": float(E_init / E_final),
    "theta_init":   theta_init.tolist(),
    "theta_final":  theta_final.tolist(),
    "L1_lambda":    L1_LAMBDA,
    "losses":       losses,
    "peaks_mV":     peaks,
    "energies":     energies_hist,
    "l1_vals":      l1_hist,
    "wall_s":       float(wall),
}
json_path = OUT / "data_optimization_waveform.json"
with open(json_path, "w") as f:
    json.dump(data, f, indent=2)
print(f"  → {json_path}")

print(f"\nDone.  Energy reduction: {E_init/E_final:.1f}×  "
      f"({E_final/E_ref_mA2ms:.2f}× reference 0.1ms pulse)")
