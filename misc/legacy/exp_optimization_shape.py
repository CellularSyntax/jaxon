"""Pulse-shape optimisation within a fixed 0.1 ms window (Task 10.1c)

The 0.1 ms rectangular pulse at threshold is the minimum-energy RECTANGULAR
monophasic waveform for MRG D=10 µm (it sits at the chronaxie of the
strength-duration curve).  This experiment asks whether a SHAPED 0.1 ms pulse
can achieve the same AP with less energy by exploiting the nonlinear dynamics
of the Hodgkin-Huxley sodium channel.

Parameterisation:
  20 free amplitude values, one per DT=0.005 ms timestep → 0.1 ms total.
  No basis restriction: the optimizer controls the current at every step.
  No L1: we want the best SHAPE, not the sparsest activation.

Task formulation:
  minimize   energy(θ) = Σ_k θ_k² · DT        [mA²·ms]
  s.t.       V_peak    ≥ −20 mV               [AP fires]
  as:        loss = energy + λ · relu(V_thresh − V_peak)²

Reference:
  Rectangular pulse: θ_k = thr_mA for all k (starting point and baseline).
  E_ref = thr_mA² × 0.1 ms.  The optimized shape must beat this.

Model: MRG D=10 µm, 21 nodes, 1 mm depth, σ=0.3 S/m, dt=0.005 ms.

Outputs:
  outputs/fig_optimization_shape.png
  outputs/data_optimization_shape.json

Run:
    conda run -n jaxley_fibers python experiments/exp_optimization_shape.py
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
CELSIUS   = 37.0
DT        = 0.005     # ms  (= 5 µs per step)
DELAY     = 0.5       # ms  lead-in
PULSE_MS  = 0.1       # ms  fixed window — same duration as reference pulse
TSTOP     = 3.0       # ms
K         = round(PULSE_MS / DT)   # = 20 free amplitude steps

SIGMA_SM  = 0.3       # S/m
DIAM      = 10.0      # µm
N_NODES   = 21
N_STEPS   = round(TSTOP / DT)     # 600

DEPTH_MM  = 1.0
V_THRESH  = -20.0     # mV  AP detection threshold
LAMBDA    = 1e4       # hinge penalty weight

LR        = 5e-3
N_EPOCHS  = 1000

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

# ── embedding matrix B [N_STEPS, K] ──────────────────────────────────────────
# B[s, k] = 1 iff step s corresponds to bin k in the pulse window.
# Each bin is exactly one DT timestep wide within [pulse_start, pulse_end).
pulse_start = round(DELAY / DT)
pulse_end   = pulse_start + K

B = np.zeros((N_STEPS, K), dtype=np.float64)
for k in range(K):
    B[pulse_start + k, k] = 1.0
B_j = jnp.asarray(B)

# ── find threshold (rectangular 0.1 ms pulse) ─────────────────────────────────
pm_ref = np.zeros(N_STEPS, dtype=np.float64)
pm_ref[pulse_start:pulse_end] = 1.0
pm_ref_j = jnp.array(pm_ref)

print("Finding threshold (rectangular 0.1 ms pulse) ...")
thr_mA = find_threshold(static, mfn_factory, Ve_unit, pm_ref_j, DT,
                        v_thresh=V_THRESH, v_rest=V_REST)
print(f"  threshold = {thr_mA:.5f} mA")

E_ref = float(energy(jnp.array(pm_ref * thr_mA), DT))
print(f"  reference energy (rectangular 0.1 ms @ thr) = {E_ref:.6f} mA²·ms")

# ── loss function ─────────────────────────────────────────────────────────────
def forward(theta):
    waveform = B_j @ theta
    mask     = waveform / (-1.0)
    trace, _ = integrate(static, membrane_fn, state0, Ve_unit, mask, DT,
                         v_rest=V_REST, center_comp=mid_comp)
    vpeak = max_vm(trace)
    e     = energy(waveform, DT)
    hinge = jax.nn.relu(V_THRESH - vpeak)
    return e + LAMBDA * hinge ** 2, (vpeak, e)

value_and_grad = jax.jit(jax.value_and_grad(forward, has_aux=True))

@jax.jit
def get_trace(theta):
    waveform = B_j @ theta
    mask     = waveform / (-1.0)
    trace, _ = integrate(static, membrane_fn, state0, Ve_unit, mask, DT,
                         v_rest=V_REST, center_comp=mid_comp)
    return trace

# ── reference trace ───────────────────────────────────────────────────────────
theta_ref = jnp.full((K,), thr_mA, dtype=jnp.float64)   # flat rectangular @ thr
trace_ref  = np.array(get_trace(theta_ref))

# ── initialisation ────────────────────────────────────────────────────────────
# Flat rectangular pulse at 1.3× threshold across all 20 steps.
# This is the reference shape, scaled up so the optimizer has headroom to
# reshape while the AP constraint remains comfortably satisfied.
theta0 = jnp.full((K,), thr_mA * 1.3, dtype=jnp.float64)
print(f"\nInitial energy (flat @ 1.3×thr): {float(energy(B_j @ theta0, DT)):.6f} mA²·ms")
print(f"Initial V_peak: {float(max_vm(get_trace(theta0))):.1f} mV")
print(f"Reference energy: {E_ref:.6f} mA²·ms  (must beat this)\n")

# ── optimisation ──────────────────────────────────────────────────────────────
theta     = theta0
opt       = optax.adam(LR)
opt_state = opt.init(theta)

losses, peaks, energies_hist = [], [], []
best_energy_valid = float("inf")
best_theta        = np.array(theta0)
best_epoch        = 0

print(f"Optimising K={K} free steps × {DT*1e3:.0f} µs = {PULSE_MS} ms window, "
      f"{N_EPOCHS} epochs, lr={LR} ...")
print(f"  Loss = energy + {LAMBDA:.0e}×hinge²")
t0 = time.perf_counter()

for epoch in range(N_EPOCHS):
    (loss, (vpeak, e)), grads = value_and_grad(theta)
    updates, opt_state = opt.update(grads, opt_state, theta)
    theta = optax.apply_updates(theta, updates)
    vp_f = float(vpeak); e_f = float(e)
    losses.append(float(loss)); peaks.append(vp_f); energies_hist.append(e_f)
    if vp_f > V_THRESH and e_f < best_energy_valid:
        best_energy_valid = e_f
        best_theta        = np.array(theta)
        best_epoch        = epoch
    if epoch % 100 == 0 or epoch == N_EPOCHS - 1:
        improvement = (E_ref - best_energy_valid) / E_ref * 100 if best_energy_valid < E_ref else 0.0
        print(f"  epoch {epoch:4d}  loss={float(loss):.6f}  V_peak={vp_f:+.1f} mV  "
              f"energy={e_f:.6f}  best={best_energy_valid:.6f} ({improvement:+.1f}% vs ref)")

wall = time.perf_counter() - t0
print(f"Optimisation wall-time: {wall:.1f} s")

E_init  = energies_hist[0]
E_final = best_energy_valid
improvement_pct = (E_ref - E_final) / E_ref * 100

print(f"\nEnergy:   initial = {E_init:.6f} mA²·ms")
print(f"          reference (rect @ thr) = {E_ref:.6f} mA²·ms")
print(f"          best shaped = {E_final:.6f} mA²·ms  ({improvement_pct:+.1f}% vs reference)")
print(f"          best epoch = {best_epoch}")

# ── traces ────────────────────────────────────────────────────────────────────
theta_final   = best_theta
waveform_init  = B @ np.array(theta0)
waveform_ref   = B @ np.full(K, float(thr_mA))
waveform_final = B @ theta_final
trace_init     = np.array(get_trace(theta0))
trace_final    = np.array(get_trace(jnp.asarray(theta_final)))

t_vec      = (np.arange(N_STEPS) + 1) * DT
t_pulse    = (np.arange(K) + 0.5) * DT + DELAY   # bin centre times

# ── figure (3 panels) ─────────────────────────────────────────────────────────
print("\n=== Generating figure ===")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
ax_w, ax_v, ax_l = axes

# Panel A: waveform shapes
ax_w.step(t_pulse, np.full(K, float(thr_mA)),
          where="mid", color="k", lw=1.5, ls="--", label=f"rectangular @ thr ({float(thr_mA):.3f} mA)")
ax_w.step(t_pulse, waveform_init[pulse_start:pulse_end],
          where="mid", color="C0", lw=1.5, alpha=0.7, label=f"initial (1.3×thr)")
ax_w.step(t_pulse, waveform_final[pulse_start:pulse_end],
          where="mid", color="C3", lw=2.0, label="optimized shape")
ax_w.fill_between(t_pulse, waveform_final[pulse_start:pulse_end], 0,
                  step="mid", color="C3", alpha=0.15)
ax_w.axhline(0, color="k", lw=0.5, alpha=0.5)
ax_w.set_xlim(DELAY - 0.02, DELAY + PULSE_MS + 0.02)
ax_w.set_xlabel("time (ms)")
ax_w.set_ylabel("amplitude (mA)")
ax_w.legend(fontsize=8)
ax_w.grid(alpha=0.25)
ax_w.set_title(f"A  Pulse shape  (K={K} free steps, {DT*1e3:.0f} µs each)",
               fontsize=10, fontweight="bold")

E_init_val = float(energy(jnp.asarray(waveform_init), DT))
ax_w.annotate(f"E_ref  = {E_ref:.5f} mA²·ms",
              xy=(0.04, 0.96), xycoords="axes fraction", color="k", fontsize=8, va="top")
ax_w.annotate(f"E_init = {E_init_val:.5f} mA²·ms",
              xy=(0.04, 0.89), xycoords="axes fraction", color="C0", fontsize=8, va="top")
ax_w.annotate(f"E_best = {E_final:.5f} mA²·ms  ({improvement_pct:+.1f}% vs rect)",
              xy=(0.04, 0.82), xycoords="axes fraction", color="C3", fontsize=8, va="top")

# Panel B: Vm traces
ax_v.plot(t_vec, trace_ref,   color="k",  lw=1.5, ls="--",
          label=f"rectangular @ thr  Vpeak={trace_ref.max():.1f} mV")
ax_v.plot(t_vec, trace_init,  color="C0", lw=1.2, alpha=0.7,
          label=f"initial (1.3×thr)  Vpeak={trace_init.max():.1f} mV")
ax_v.plot(t_vec, trace_final, color="C3", lw=2.0,
          label=f"optimized  Vpeak={trace_final.max():.1f} mV")
ax_v.axhline(V_THRESH, color="k", ls=":", lw=0.8, alpha=0.6, label=f"V_thresh={V_THRESH} mV")
ax_v.set_xlim(0, TSTOP)
ax_v.set_xlabel("time (ms)")
ax_v.set_ylabel("V_m center node (mV)")
ax_v.legend(fontsize=8)
ax_v.grid(alpha=0.25)
ax_v.set_title("B  Center-node Vm trace", fontsize=10, fontweight="bold")

# Panel C: loss / energy curve
ep = np.arange(N_EPOCHS)
ax_l.plot(ep, losses,        color="k",  lw=1.5, label="total loss")
ax_l.plot(ep, energies_hist, color="C2", lw=1.5, label="energy (∫I²dt)")
ax_l.axhline(E_ref, color="k", ls="--", lw=1.0, alpha=0.7, label=f"E_ref = {E_ref:.5f}")
ax_l.set_yscale("log")
ax_l.set_xlabel("epoch")
ax_l.set_ylabel("energy (mA²·ms)")
ax_l.legend(fontsize=8, loc="upper right")
ax_l.grid(alpha=0.25)

ax_r = ax_l.twinx()
ax_r.plot(ep, peaks, color="C0", lw=1.2, alpha=0.8, label="V_peak")
ax_r.axhline(V_THRESH, color="C0", ls=":", lw=0.8, alpha=0.6)
ax_r.set_ylabel("V_peak (mV)", color="C0", fontsize=9)
ax_r.tick_params(axis="y", labelcolor="C0")
ax_r.legend(fontsize=7, loc="center right")
ax_l.set_title(f"C  Optimisation curve  (Adam lr={LR})", fontsize=9, fontweight="bold")

fig.suptitle(
    f"Pulse-shape optimisation: can a non-rectangular 0.1 ms pulse beat the reference?\n"
    f"MRG D={DIAM}µm, {N_NODES} nodes, {DEPTH_MM}mm depth  |  "
    f"best improvement: {improvement_pct:+.1f}% vs rectangular pulse",
    fontsize=11,
)

fig_path = OUT / "fig_optimization_shape.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {fig_path}")

# ── JSON ───────────────────────────────────────────────────────────────────────
data = {
    "model":          {"diam_um": DIAM, "n_nodes": N_NODES, "dt_ms": DT,
                       "tstop_ms": TSTOP, "celsius": CELSIUS},
    "setup":          {"depth_mm": DEPTH_MM, "delay_ms": DELAY, "pulse_ms": PULSE_MS,
                       "K_steps": K, "step_dt_ms": DT},
    "optimizer":      {"lr": LR, "n_epochs": N_EPOCHS, "lambda": LAMBDA},
    "threshold_mA":   float(thr_mA),
    "E_ref":          float(E_ref),
    "E_init":         float(E_init),
    "E_final":        float(E_final),
    "best_epoch":     int(best_epoch),
    "improvement_pct": float(improvement_pct),
    "theta_ref":      np.full(K, float(thr_mA)).tolist(),
    "theta_init":     np.array(theta0).tolist(),
    "theta_final":    theta_final.tolist(),
    "losses":         losses,
    "peaks_mV":       peaks,
    "energies":       energies_hist,
    "wall_s":         float(wall),
}
json_path = OUT / "data_optimization_shape.json"
with open(json_path, "w") as f:
    json.dump(data, f, indent=2)
print(f"  → {json_path}")

print(f"\nDone.  Best shaped pulse: {E_final:.6f} mA²·ms  ({improvement_pct:+.1f}% vs {E_ref:.6f} rect @ thr)")
