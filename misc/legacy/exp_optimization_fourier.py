"""Waveform optimisation — charge-balanced Fourier basis (Task 10.1b)

Replaces the 40-bin piecewise-constant basis of exp_optimization_waveform.py
with a charge-balanced Fourier basis on the pulse window.  The DC term is
excluded so the waveform integrates to exactly zero (no net charge injection).

  I(t) = Σ_{k=1}^K [ θ_{2k-1} cos(2πk t/T) + θ_{2k} sin(2πk t/T) ]

K=10 frequency components → 20 free parameters spanning 0.5–5 kHz.
The charge-balance constraint (DC = 0) forces the optimizer to find an
oscillatory waveform where the cathodic half-cycles drive the AP while the
anodic half-cycles return charge.  This reveals which frequencies are most
energy-efficient for MRG AP initiation.

Task formulation:
  minimize   energy(θ) = ∫ I(t)² dt                [mA²·ms]
  s.t.       V_peak    ≥ −20 mV                    [AP fires]
  as:  loss = energy + λ · relu(V_thresh − V_peak)²

Initialisation:
  k=1 sine at amplitude A_INIT (cathodic first half-cycle expected to fire AP).
  The fundamental (0.5 kHz, T_half = 1 ms) is a natural starting waveform.

Model: MRG D=10 µm, 21 nodes, 1 mm depth, σ=0.3 S/m, dt=0.005 ms.

Outputs:
  outputs/fig_optimization_fourier.png
  outputs/data_optimization_fourier.json

Run:
    conda run -n jaxley_fibers python experiments/exp_optimization_fourier.py
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
PULSE_MS = 2.0        # ms  pulse window
K_FREQS  = 10         # Fourier harmonics → 2K = 20 charge-balanced parameters
N_PARAMS = 2 * K_FREQS   # no DC term → charge-balanced (∫I dt = 0)

SIGMA_SM = 0.3        # S/m
DIAM     = 10.0       # µm
N_NODES  = 21
N_STEPS  = int(TSTOP / DT)

DEPTH_MM  = 1.0
V_THRESH  = -20.0
LAMBDA    = 1e4       # hinge penalty

LR       = 2e-3
N_EPOCHS = 800

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
dy = DEPTH_MM * 1e-3
dz = (centers[mid_comp] - centers) * 1e-6
r  = np.sqrt(dy**2 + dz**2)
Ve_unit = jnp.array(-1.0 / (4.0 * np.pi * SIGMA_SM * r))

# ── Fourier basis matrix Φ [N_STEPS, 2K+1] ────────────────────────────────────
# Φ[s, j] = j-th Fourier basis function evaluated at timestep s.
# Only non-zero inside the pulse window [pulse_start, pulse_end).
# Columns: [1, cos(2πt/T), sin(2πt/T), cos(4πt/T), sin(4πt/T), …]
# where t is normalised to [0, 1] across the pulse window.
pulse_start = round(DELAY / DT)
pulse_end   = round((DELAY + PULSE_MS) / DT)
n_active    = pulse_end - pulse_start

t_n = np.arange(n_active) / n_active   # normalised time [0, 1)

cols = []   # no DC — all columns integrate to zero → charge-balanced
for k in range(1, K_FREQS + 1):
    cols.append(np.cos(2 * np.pi * k * t_n))
    cols.append(np.sin(2 * np.pi * k * t_n))
Phi_active = np.column_stack(cols)           # (n_active, 2K)

Phi = np.zeros((N_STEPS, N_PARAMS), dtype=np.float64)
Phi[pulse_start:pulse_end, :] = Phi_active
Phi_j = jnp.asarray(Phi)

# ── find threshold (0.1 ms square pulse) ─────────────────────────────────────
pm_ref = np.zeros(N_STEPS, dtype=np.float64)
pm_ref[pulse_start : pulse_start + round(0.1 / DT)] = 1.0
pm_ref_j = jnp.array(pm_ref)

print("Finding threshold (0.1 ms square pulse) ...")
thr_mA = find_threshold(static, mfn_factory, Ve_unit, pm_ref_j, DT,
                        v_thresh=V_THRESH, v_rest=V_REST)
print(f"  threshold = {thr_mA:.5f} mA")

E_ref_mA2ms = thr_mA**2 * 0.1
print(f"  baseline energy (0.1 ms square) = {E_ref_mA2ms:.6f} mA²·ms")

# ── loss function ─────────────────────────────────────────────────────────────
def forward(theta):
    """Scalar loss for Fourier coefficients θ [2K+1] (mA per basis function)."""
    waveform = Phi_j @ theta          # (N_STEPS,) signed current in mA
    mask     = waveform / (-1.0)      # Ve scaling convention
    trace, _ = integrate(static, membrane_fn, state0, Ve_unit, mask, DT,
                         v_rest=V_REST, center_comp=mid_comp)
    vpeak = max_vm(trace)
    e     = energy(waveform, DT)
    hinge = jax.nn.relu(V_THRESH - vpeak)
    return e + LAMBDA * hinge ** 2, (vpeak, e)

value_and_grad = jax.jit(jax.value_and_grad(forward, has_aux=True))

@jax.jit
def get_trace(theta):
    waveform = Phi_j @ theta
    mask     = waveform / (-1.0)
    trace, _ = integrate(static, membrane_fn, state0, Ve_unit, mask, DT,
                         v_rest=V_REST, center_comp=mid_comp)
    return trace

# ── initialisation ────────────────────────────────────────────────────────────
# Start with the fundamental sine (k=1, 0.5 kHz) at 3×|thr_mA| amplitude.
# Columns: [cos1, sin1, cos2, sin2, ...] → sin1 is column index 1.
# At 3× threshold, the cathodic half-cycle (first 1 ms) is expected to fire AP.
A_INIT   = abs(thr_mA) * 3.0
theta0   = jnp.zeros(N_PARAMS, dtype=jnp.float64)
theta0   = theta0.at[1].set(A_INIT)         # sin(2π·1·t) = cathodic first half-cycle

print(f"\nInitial energy (k=1 sine at 3×thr = {A_INIT:.4f} mA): "
      f"{float(energy(Phi_j @ theta0, DT)):.5f} mA²·ms")
print(f"Initial V_peak: {float(max_vm(get_trace(theta0))):.1f} mV")

# ── optimisation ──────────────────────────────────────────────────────────────
theta = theta0
opt   = optax.adam(LR)
opt_state = opt.init(theta)

losses, peaks, energies_hist = [], [], []
best_energy_valid = float("inf")
best_theta        = np.array(theta0)   # best supra-threshold theta seen so far
best_epoch        = 0

print(f"\nOptimising K={K_FREQS} frequencies ({N_PARAMS} params, charge-balanced), "
      f"{N_EPOCHS} epochs, lr={LR} ...")
print(f"  Loss = energy + {LAMBDA:.0e}×hinge²")
t0 = time.perf_counter()

for epoch in range(N_EPOCHS):
    (loss, (vpeak, e)), grads = value_and_grad(theta)
    updates, opt_state = opt.update(grads, opt_state, theta)
    theta = optax.apply_updates(theta, updates)
    vp_f = float(vpeak); e_f = float(e)
    losses.append(float(loss))
    peaks.append(vp_f)
    energies_hist.append(e_f)
    # track best supra-threshold solution (early-stopping style)
    if vp_f > V_THRESH and e_f < best_energy_valid:
        best_energy_valid = e_f
        best_theta        = np.array(theta)
        best_epoch        = epoch
    if epoch % 60 == 0 or epoch == N_EPOCHS - 1:
        print(f"  epoch {epoch:3d}  loss={float(loss):.5f}  "
              f"V_peak={vp_f:+.1f} mV  energy={e_f:.5f}  "
              f"best={best_energy_valid:.5f}@ep{best_epoch}")

wall = time.perf_counter() - t0
print(f"Optimisation wall-time: {wall:.1f} s")
print(f"Best valid energy: {best_energy_valid:.5f} mA²·ms at epoch {best_epoch} "
      f"({best_energy_valid/E_ref_mA2ms:.2f}× reference)")

theta_init  = np.array(theta0)
theta_final = best_theta     # use best supra-threshold solution for figure/JSON

E_init  = energies_hist[0]
E_final = best_energy_valid
print(f"\nEnergy:  initial={E_init:.5f}  best={E_final:.5f}  "
      f"reduction={E_init/E_final:.1f}×  vs baseline={E_final/E_ref_mA2ms:.2f}×baseline")

# ── traces ────────────────────────────────────────────────────────────────────
waveform_init  = Phi @ theta_init
waveform_final = Phi @ theta_final
trace_init  = np.array(get_trace(theta0))
trace_final = np.array(get_trace(theta))
t_vec = (np.arange(N_STEPS) + 1) * DT

# Fourier spectrum: power of each component (no DC — charge-balanced)
freqs_kHz = np.arange(1, K_FREQS + 1) / PULSE_MS    # kHz  (T in ms → 1/T in kHz)
cos_power = theta_final[0::2]**2    # indices 0,2,4,… = cos_k
sin_power = theta_final[1::2]**2    # indices 1,3,5,… = sin_k
harmonic_power = cos_power + sin_power   # total power per frequency (k=1..K)

# ── figure (4 panels) ─────────────────────────────────────────────────────────
print("\n=== Generating figure ===")
fig = plt.figure(figsize=(16, 5.5), constrained_layout=True)
gs  = fig.add_gridspec(1, 4)
ax_w  = fig.add_subplot(gs[0, 0])
ax_v  = fig.add_subplot(gs[0, 1])
ax_l  = fig.add_subplot(gs[0, 2])
ax_sp = fig.add_subplot(gs[0, 3])

t_pulse = t_vec[pulse_start:pulse_end]

# Panel A: Waveforms
ax_w.plot(t_vec, waveform_init,  color="C0", lw=1.5, alpha=0.7,
          label=f"initial (k=1 sine, A=3×thr)")
ax_w.plot(t_vec, waveform_final, color="C3", lw=2.0,
          label="optimized")
ax_w.fill_between(t_vec, waveform_final, 0, where=(waveform_final < 0),
                  color="C3", alpha=0.15, label="cathodic phase")
ax_w.fill_between(t_vec, waveform_final, 0, where=(waveform_final > 0),
                  color="C1", alpha=0.25, label="anodic phase")
ax_w.axhline(0,       color="k",  lw=0.5, alpha=0.5)
ax_w.axhline(thr_mA,  color="C0", ls=":", lw=1.0, alpha=0.5, label=f"thr={thr_mA:.3f} mA")
ax_w.set_xlim(0.8, DELAY + PULSE_MS + 0.3)
ax_w.set_xlabel("time (ms)")
ax_w.set_ylabel("amplitude (mA)")
ax_w.legend(fontsize=7, loc="lower right")
ax_w.grid(alpha=0.25)
ax_w.set_title(f"A  Waveform  (K={K_FREQS} harmonics, {N_PARAMS} params, DC=0)",
               fontsize=10, fontweight="bold")
ax_w.annotate(f"E_init  = {E_init:.4f} mA²·ms",
              xy=(0.04, 0.96), xycoords="axes fraction", color="C0", fontsize=7, va="top")
ax_w.annotate(f"E_final = {E_final:.4f} mA²·ms  ({E_init/E_final:.1f}× reduction)",
              xy=(0.04, 0.89), xycoords="axes fraction", color="C3", fontsize=7, va="top")
ax_w.annotate(f"E_ref (0.1 ms sq) = {E_ref_mA2ms:.4f} mA²·ms",
              xy=(0.04, 0.82), xycoords="axes fraction", color="k", fontsize=7, va="top")

# Panel B: Vm traces
ax_v.plot(t_vec, trace_init,  color="C0", lw=1.5, label=f"initial  Vpeak={trace_init.max():.1f} mV")
ax_v.plot(t_vec, trace_final, color="C3", lw=1.5, label=f"optimized  Vpeak={trace_final.max():.1f} mV")
ax_v.axhline(V_THRESH, color="k", ls="--", lw=0.8, alpha=0.6, label=f"V_thresh={V_THRESH} mV")
ax_v.set_xlim(0, TSTOP)
ax_v.set_xlabel("time (ms)")
ax_v.set_ylabel("V_m center node (mV)")
ax_v.legend(fontsize=8)
ax_v.grid(alpha=0.25)
ax_v.set_title("B  Center-node Vm trace", fontsize=10, fontweight="bold")

# Panel C: Loss / energy curve
ep = np.arange(N_EPOCHS)
ax_l.plot(ep, losses,        color="k",  lw=1.5, label="total loss")
ax_l.plot(ep, energies_hist, color="C2", lw=1.5, label="energy (∫I²dt)")
ax_l.set_yscale("log")
ax_l.set_xlabel("epoch")
ax_l.set_ylabel("energy (mA²·ms)")
ax_l.legend(fontsize=8, loc="upper right")
ax_l.grid(alpha=0.25)
ax_r = ax_l.twinx()
ax_r.plot(ep, peaks, color="C0", lw=1.5, alpha=0.8, label="V_peak")
ax_r.axhline(V_THRESH, color="C0", ls="--", lw=0.8, alpha=0.6)
ax_r.set_ylabel("V_peak (mV)", color="C0", fontsize=9)
ax_r.tick_params(axis="y", labelcolor="C0")
ax_r.legend(fontsize=7, loc="center right")
ax_l.set_title(f"C  Optimisation curve  (Adam lr={LR})", fontsize=9, fontweight="bold")

# Panel D: Fourier spectrum of optimised waveform (charge-balanced: no DC bar)
ax_sp.bar(np.arange(1, K_FREQS + 1), harmonic_power, color="C0", width=0.6,
          label="cos²+sin² (k=1..K)")
ax_sp.set_xlabel("harmonic index k")
ax_sp.set_ylabel("power  θ_cos² + θ_sin²  (mA²)")
ax_sp.set_xticks(np.arange(1, K_FREQS + 1))
ax_sp.set_xticklabels(
    [f"{k}\n({freqs_kHz[k-1]:.1f}kHz)" for k in range(1, K_FREQS + 1)],
    fontsize=7,
)
ax_sp.legend(fontsize=7)
ax_sp.grid(alpha=0.25, axis="y")
ax_sp.set_title("D  Fourier spectrum (optimised θ, charge-balanced)",
                fontsize=9, fontweight="bold")

fig.suptitle(
    f"Charge-balanced Fourier waveform optimisation through MRG coupled solver\n"
    f"(D={DIAM}µm, {N_NODES} nodes, {DEPTH_MM}mm depth, "
    f"K={K_FREQS} harmonics = {N_PARAMS} params, DC=0)",
    fontsize=11,
)

fig_path = OUT / "fig_optimization_fourier.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {fig_path}")

# ── JSON ───────────────────────────────────────────────────────────────────────
data = {
    "model":         {"diam_um": DIAM, "n_nodes": N_NODES, "dt_ms": DT,
                      "tstop_ms": TSTOP, "celsius": CELSIUS},
    "setup":         {"depth_mm": DEPTH_MM, "delay_ms": DELAY, "pulse_ms": PULSE_MS,
                      "K_freqs": K_FREQS, "n_params": N_PARAMS},
    "optimizer":     {"lr": LR, "n_epochs": N_EPOCHS, "lambda": LAMBDA},
    "threshold_mA":  float(thr_mA),
    "E_ref_mA2ms":   float(E_ref_mA2ms),
    "E_init":         float(E_init),
    "E_final":        float(E_final),
    "best_epoch":     int(best_epoch),
    "energy_reduction_x": float(E_init / E_final),
    "vs_baseline_x":  float(E_final / E_ref_mA2ms),
    "theta_init":    theta_init.tolist(),
    "theta_final":   theta_final.tolist(),
    "freqs_kHz":      freqs_kHz.tolist(),
    "harmonic_power": harmonic_power.tolist(),
    "losses":        losses,
    "peaks_mV":      peaks,
    "energies":      energies_hist,
    "wall_s":        float(wall),
}
json_path = OUT / "data_optimization_fourier.json"
with open(json_path, "w") as f:
    json.dump(data, f, indent=2)
print(f"  → {json_path}")

print(f"\nDone.  Energy: {E_init:.5f} → {E_final:.5f} mA²·ms  "
      f"({E_init/E_final:.1f}× reduction, {E_final/E_ref_mA2ms:.2f}× reference)")
k_dom = np.argmax(harmonic_power)    # 0-indexed into harmonic_power → k = k_dom+1
print(f"Dominant frequency: k={k_dom+1}  ({freqs_kHz[k_dom]:.1f} kHz)")
