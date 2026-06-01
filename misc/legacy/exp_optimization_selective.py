"""Selective recruitment optimisation — Task 10.2

Optimise a 2 ms waveform to activate D=10 µm (target) while sparing D=14 µm
(off-target), both at 1 mm depth from a point source electrode.

Physical challenge:
  D=14 has a LOWER threshold (−0.159 mA) than D=10 (−0.183 mA).  Any monophasic
  cathodic pulse that fires D=10 has already fired D=14 → monophasic selectivity
  is physically impossible for this diameter pair.  The optimizer must discover
  a biphasic or multi-phase waveform that exploits differential timing of AP
  initiation between the two fiber types.

Selectivity objective (surrogate):
  sel = activation_prob(trace_on, σ) − activation_prob(trace_off, σ)
  σ = 10 mV  (required for non-vanishing gradients; see objectives.py)

Loss:
  loss = −sel + λ_e · energy    (maximize selectivity, penalize energy)

Waveform parameterisation:
  K=40 piecewise-constant bins × 50 µs each = 2 ms window.
  Positive (anodic) bins are allowed; the optimizer can freely create
  biphasic or multi-phase patterns.

Outputs:
  outputs/fig_optimization_selective.png
  outputs/data_optimization_selective.json

Run:
    conda run -n jaxley_fibers python experiments/exp_optimization_selective.py
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
from jaxfibers.objectives import activation_prob, energy

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
CELSIUS   = 37.0
DT        = 0.005      # ms
DELAY     = 1.0        # ms
PULSE_MS  = 2.0        # ms  (allow biphasic over full window)
K_BINS    = 40         # 50 µs per bin
BIN_DT    = PULSE_MS / K_BINS
TSTOP     = 5.0        # ms
N_STEPS   = round(TSTOP / DT)

DIAM_ON   = 10.0       # µm  target fiber
DIAM_OFF  = 14.0       # µm  off-target fiber
N_NODES   = 21
DEPTH_MM  = 1.0        # mm  both fibers at same depth
SIGMA_SM  = 0.3        # S/m

V_THRESH  = -20.0      # mV  AP detection
SIGMA_SIG = 10.0       # mV  sigmoid width for activation_prob
LAMBDA_E  = 0.01       # energy penalty relative to selectivity objective

LR        = 5e-3
N_EPOCHS  = 800

GNABAR = 3.0; GNAPBAR = 0.01; GKBAR = 0.08; GL = 0.007
ENA = 50.0; EK = -90.0; EL = -90.0

# ── build both fibers ─────────────────────────────────────────────────────────
print("Building MRG D=10 µm (target) and D=14 µm (off-target) ...")

def _setup_fiber(diam):
    _, geom = build_mrg(diameter=diam, n_nodes=N_NODES)
    geom_c  = dataclasses.replace(geom, cm_uF_cm2=[CM_AXON] * geom.n_comp)
    static  = arrays_from_geometry(geom_c, DT)
    is_node = static["is_node"]
    A_in    = static["A_in_cm2"]
    stype_gp = {"node": 0.0, "mysa": G_PAS_MYSA, "flut": G_PAS_FLUT, "stin": G_PAS_STIN}
    g_pas   = jnp.asarray([stype_gp[s] for s in geom.section_type])
    v0 = jnp.float64(V_REST)
    (am,bm),(ah,bh),(amp0,bmp0),(as0,bs0) = AxnodeMyel._alpha_beta(v0, CELSIUS)
    state0 = (
        jnp.where(is_node, float(am  / (am  + bm)),     0.0),
        jnp.where(is_node, float(ah  / (ah  + bh)),     0.0),
        jnp.where(is_node, float(amp0 / (amp0 + bmp0)), 0.0),
        jnp.where(is_node, float(as0 / (as0 + bs0)),    0.0),
    )
    centers = np.array(section_centers_um(geom))
    nidx    = node_indices(geom)
    mid     = nidx[len(nidx) // 2]
    dy      = DEPTH_MM * 1e-3
    dz      = (centers[mid] - centers) * 1e-6
    r       = np.sqrt(dy**2 + dz**2)
    Ve_unit = jnp.array(-1.0 / (4.0 * np.pi * SIGMA_SM * r))
    return static, is_node, A_in, g_pas, state0, mid, Ve_unit

(static_on, is_node_on, A_in_on, g_pas_on, state0_on, mid_on, Ve_unit_on) = _setup_fiber(DIAM_ON)
(static_off, is_node_off, A_in_off, g_pas_off, state0_off, mid_off, Ve_unit_off) = _setup_fiber(DIAM_OFF)

def membrane_fn_on(Vm, state, dt_):
    M, H, MP, S = state
    (am,bm),(ah,bh),(amp_,bmp),(as_,bs_) = AxnodeMyel._alpha_beta(Vm, CELSIUS)
    M2  = solve_gate_exponential(M,  dt_, am,   bm)
    H2  = solve_gate_exponential(H,  dt_, ah,   bh)
    MP2 = solve_gate_exponential(MP, dt_, amp_, bmp)
    S2  = solve_gate_exponential(S,  dt_, as_,  bs_)
    gna = GNABAR * M2**3 * H2; gnap = GNAPBAR * MP2**3; gk = GKBAR * S2
    g_n = (gna + gnap + gk + GL) * A_in_on * 1e6
    i_n = ((gna + gnap)*(Vm-ENA) + gk*(Vm-EK) + GL*(Vm-EL)) * A_in_on * 1e6
    g_p = g_pas_on * A_in_on * 1e6
    i_p = g_p * (Vm - V_REST)
    return (jnp.where(is_node_on, g_n, g_p), jnp.where(is_node_on, i_n, i_p), (M2,H2,MP2,S2))

def membrane_fn_off(Vm, state, dt_):
    M, H, MP, S = state
    (am,bm),(ah,bh),(amp_,bmp),(as_,bs_) = AxnodeMyel._alpha_beta(Vm, CELSIUS)
    M2  = solve_gate_exponential(M,  dt_, am,   bm)
    H2  = solve_gate_exponential(H,  dt_, ah,   bh)
    MP2 = solve_gate_exponential(MP, dt_, amp_, bmp)
    S2  = solve_gate_exponential(S,  dt_, as_,  bs_)
    gna = GNABAR * M2**3 * H2; gnap = GNAPBAR * MP2**3; gk = GKBAR * S2
    g_n = (gna + gnap + gk + GL) * A_in_off * 1e6
    i_n = ((gna + gnap)*(Vm-ENA) + gk*(Vm-EK) + GL*(Vm-EL)) * A_in_off * 1e6
    g_p = g_pas_off * A_in_off * 1e6
    i_p = g_p * (Vm - V_REST)
    return (jnp.where(is_node_off, g_n, g_p), jnp.where(is_node_off, i_n, i_p), (M2,H2,MP2,S2))

def mfn_factory_on():  return membrane_fn_on, state0_on
def mfn_factory_off(): return membrane_fn_off, state0_off

# ── find thresholds ────────────────────────────────────────────────────────────
pulse_start = round(DELAY / DT)
pulse_end   = round((DELAY + PULSE_MS) / DT)
n_active    = pulse_end - pulse_start

pm_ref = np.zeros(N_STEPS, dtype=np.float64)
pm_ref[pulse_start : pulse_start + round(0.1 / DT)] = 1.0   # 0.1 ms reference pulse
pm_ref_j = jnp.array(pm_ref)

print("Finding thresholds (0.1 ms cathodic pulse) ...")
thr_on  = find_threshold(static_on,  mfn_factory_on,  Ve_unit_on,  pm_ref_j, DT, V_THRESH, V_REST)
thr_off = find_threshold(static_off, mfn_factory_off, Ve_unit_off, pm_ref_j, DT, V_THRESH, V_REST)
print(f"  D={DIAM_ON}µm (target)    : {thr_on:.5f} mA")
print(f"  D={DIAM_OFF}µm (off-target): {thr_off:.5f} mA")
print(f"  Ratio thr_off/thr_on = {thr_off/thr_on:.3f}  "
      f"({'D='+str(DIAM_OFF)+' activates FIRST with monophasic' if abs(thr_off) < abs(thr_on) else 'D='+str(DIAM_ON)+' activates FIRST'})")

# ── bin-assignment matrix B ───────────────────────────────────────────────────
B = np.zeros((N_STEPS, K_BINS), dtype=np.float64)
steps_per_bin = n_active // K_BINS
for k in range(K_BINS):
    s0 = pulse_start + k * steps_per_bin
    B[s0 : s0 + steps_per_bin, k] = 1.0
B_j = jnp.asarray(B)

# ── loss: maximise selectivity, penalise energy ───────────────────────────────
def forward(theta):
    waveform  = B_j @ theta
    mask      = waveform / (-1.0)
    trace_on,  _ = integrate(static_on,  membrane_fn_on,  state0_on,  Ve_unit_on,
                             mask, DT, v_rest=V_REST, center_comp=mid_on)
    trace_off, _ = integrate(static_off, membrane_fn_off, state0_off, Ve_unit_off,
                             mask, DT, v_rest=V_REST, center_comp=mid_off)
    act_on  = activation_prob(trace_on,  V_THRESH, SIGMA_SIG)
    act_off = activation_prob(trace_off, V_THRESH, SIGMA_SIG)
    sel = act_on - act_off          # ∈ [−1, 1], want → +1
    e   = energy(waveform, DT)
    return -sel + LAMBDA_E * e, (sel, act_on, act_off, e)

value_and_grad = jax.jit(jax.value_and_grad(forward, has_aux=True))

@jax.jit
def get_traces(theta):
    waveform  = B_j @ theta
    mask      = waveform / (-1.0)
    trace_on,  _ = integrate(static_on,  membrane_fn_on,  state0_on,  Ve_unit_on,
                             mask, DT, v_rest=V_REST, center_comp=mid_on)
    trace_off, _ = integrate(static_off, membrane_fn_off, state0_off, Ve_unit_off,
                             mask, DT, v_rest=V_REST, center_comp=mid_off)
    return trace_on, trace_off

# ── initialisation ────────────────────────────────────────────────────────────
# Flat cathodic at 1.2×|thr_on| over 40 bins.  Both fibers fire initially
# (thr_on is higher, thr_off is lower; at 1.2×thr_on both are activated).
# The optimizer must reshape to stop D=14 while keeping D=10 active.
theta0 = jnp.full((K_BINS,), thr_on * 1.2, dtype=jnp.float64)

(loss0, (sel0, act_on0, act_off0, e0)), _ = value_and_grad(theta0)
print(f"\nInitial waveform (flat cathodic at 1.2×thr_on = {float(thr_on*1.2):.4f} mA):")
print(f"  act_on={float(act_on0):.3f}  act_off={float(act_off0):.3f}  "
      f"sel={float(sel0):.3f}  energy={float(e0):.5f}")

# ── optimisation ──────────────────────────────────────────────────────────────
theta     = theta0
opt       = optax.adam(LR)
opt_state = opt.init(theta)

sel_hist, act_on_hist, act_off_hist, energies_hist, losses_hist = [], [], [], [], []
best_sel   = float("-inf")
best_theta = np.array(theta0)
best_epoch = 0

print(f"\nOptimising K={K_BINS} bins × {BIN_DT*1e3:.0f} µs, {N_EPOCHS} epochs, lr={LR} ...")
print(f"  Loss = −sel + {LAMBDA_E}×energy   (sel = act_on − act_off, σ={SIGMA_SIG} mV)")
t0 = time.perf_counter()

for epoch in range(N_EPOCHS):
    (loss, (sel, act_on, act_off, e)), grads = value_and_grad(theta)
    updates, opt_state = opt.update(grads, opt_state, theta)
    theta = optax.apply_updates(theta, updates)
    s_f = float(sel); ao_f = float(act_on); off_f = float(act_off); e_f = float(e)
    sel_hist.append(s_f); act_on_hist.append(ao_f)
    act_off_hist.append(off_f); energies_hist.append(e_f); losses_hist.append(float(loss))
    if s_f > best_sel:
        best_sel   = s_f
        best_theta = np.array(theta)
        best_epoch = epoch
    if epoch % 80 == 0 or epoch == N_EPOCHS - 1:
        print(f"  epoch {epoch:4d}  sel={s_f:+.3f}  act_on={ao_f:.3f}  act_off={off_f:.3f}  "
              f"E={e_f:.4f}  best_sel={best_sel:+.3f}@ep{best_epoch}")

wall = time.perf_counter() - t0
print(f"Wall-time: {wall:.1f} s")
print(f"\nBest selectivity: {best_sel:+.4f} at epoch {best_epoch}")

theta_final = best_theta
waveform_init  = B @ np.array(theta0)
waveform_final = B @ theta_final
trace_on_init,  trace_off_init  = [np.array(t) for t in get_traces(theta0)]
trace_on_final, trace_off_final = [np.array(t) for t in get_traces(jnp.asarray(theta_final))]

t_vec      = (np.arange(N_STEPS) + 1) * DT
bin_edges  = DELAY + np.arange(K_BINS + 1) * BIN_DT

# ── figure (4 panels) ─────────────────────────────────────────────────────────
print("\n=== Generating figure ===")
fig = plt.figure(figsize=(16, 5.0), constrained_layout=True)
gs  = fig.add_gridspec(1, 4)
ax_w  = fig.add_subplot(gs[0, 0])
ax_vi = fig.add_subplot(gs[0, 1])
ax_vf = fig.add_subplot(gs[0, 2])
ax_s  = fig.add_subplot(gs[0, 3])

# Panel A: initial vs final waveform
ax_w.stairs(waveform_init[pulse_start:pulse_start+K_BINS*steps_per_bin:steps_per_bin],
            bin_edges, color="C0", lw=1.5, alpha=0.6, label="initial (flat)")
ax_w.stairs(waveform_final[pulse_start:pulse_start+K_BINS*steps_per_bin:steps_per_bin],
            bin_edges, color="C3", lw=2.0, label="optimized")
ax_w.axhline(0,       color="k", lw=0.5, alpha=0.5)
ax_w.axhline(float(thr_on),  color="C0", ls=":", lw=1.0,
             label=f"thr D={DIAM_ON}µm={thr_on:.3f} mA")
ax_w.axhline(float(thr_off), color="C1", ls=":", lw=1.0,
             label=f"thr D={DIAM_OFF}µm={thr_off:.3f} mA")
ax_w.set_xlim(DELAY - 0.1, DELAY + PULSE_MS + 0.3)
ax_w.set_xlabel("time (ms)"); ax_w.set_ylabel("amplitude (mA)")
ax_w.legend(fontsize=7, loc="lower right"); ax_w.grid(alpha=0.25)
ax_w.set_title(f"A  Waveform  (K={K_BINS} bins × {BIN_DT*1e3:.0f} µs)",
               fontsize=10, fontweight="bold")

# Panel B: initial Vm traces (both fibers)
ax_vi.plot(t_vec, trace_on_init,  color="C0", lw=1.5, label=f"D={DIAM_ON}µm (target)")
ax_vi.plot(t_vec, trace_off_init, color="C1", lw=1.5, label=f"D={DIAM_OFF}µm (off-target)")
ax_vi.axhline(V_THRESH, color="k", ls="--", lw=0.8, alpha=0.6)
ax_vi.set_xlim(0, TSTOP); ax_vi.set_xlabel("time (ms)"); ax_vi.set_ylabel("V_m (mV)")
ax_vi.legend(fontsize=8); ax_vi.grid(alpha=0.25)
ax_vi.set_title(f"B  Vm initial  (sel={float(sel0):+.2f})", fontsize=10, fontweight="bold")

# Panel C: optimised Vm traces
ax_vf.plot(t_vec, trace_on_final,  color="C0", lw=1.5, label=f"D={DIAM_ON}µm (target)")
ax_vf.plot(t_vec, trace_off_final, color="C1", lw=1.5, label=f"D={DIAM_OFF}µm (off-target)")
ax_vf.axhline(V_THRESH, color="k", ls="--", lw=0.8, alpha=0.6)
ax_vf.set_xlim(0, TSTOP); ax_vf.set_xlabel("time (ms)"); ax_vf.set_ylabel("V_m (mV)")
ax_vf.legend(fontsize=8); ax_vf.grid(alpha=0.25)
ax_vf.set_title(f"C  Vm optimized  (best sel={best_sel:+.3f})", fontsize=10, fontweight="bold")

# Panel D: selectivity and act_on/off over epochs
ep = np.arange(N_EPOCHS)
ax_s.plot(ep, sel_hist,      color="k",  lw=2.0, label="selectivity (on−off)")
ax_s.plot(ep, act_on_hist,   color="C0", lw=1.5, ls="--", label=f"act_on D={DIAM_ON}µm")
ax_s.plot(ep, act_off_hist,  color="C1", lw=1.5, ls="--", label=f"act_off D={DIAM_OFF}µm")
ax_s.axhline(0, color="k", lw=0.5, alpha=0.4)
ax_s.axhline(1, color="k", lw=0.5, ls=":", alpha=0.4)
ax_s.set_ylim(-1.1, 1.1); ax_s.set_xlabel("epoch")
ax_s.set_ylabel("activation / selectivity")
ax_s.legend(fontsize=8, loc="lower right"); ax_s.grid(alpha=0.25)
ax_s.set_title("D  Optimisation curve", fontsize=10, fontweight="bold")

fig.suptitle(
    f"Selective recruitment: target D={DIAM_ON}µm vs off-target D={DIAM_OFF}µm "
    f"(both {DEPTH_MM}mm depth)\n"
    f"Monophasic impossible (thr_off < thr_on) — optimizer finds biphasic solution   "
    f"best sel = {best_sel:+.3f}",
    fontsize=10,
)
fig_path = OUT / "fig_optimization_selective.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {fig_path}")

# ── JSON ───────────────────────────────────────────────────────────────────────
data = {
    "model":      {"n_nodes": N_NODES, "dt_ms": DT, "tstop_ms": TSTOP,
                   "celsius": CELSIUS, "depth_mm": DEPTH_MM},
    "fibers":     {"diam_on": DIAM_ON, "diam_off": DIAM_OFF,
                   "thr_on_mA": float(thr_on), "thr_off_mA": float(thr_off)},
    "setup":      {"K_bins": K_BINS, "bin_dt_ms": BIN_DT, "pulse_ms": PULSE_MS,
                   "sigma_sig_mV": SIGMA_SIG, "lambda_E": LAMBDA_E},
    "optimizer":  {"lr": LR, "n_epochs": N_EPOCHS},
    "initial":    {"sel": float(sel0), "act_on": float(act_on0),
                   "act_off": float(act_off0), "energy": float(e0)},
    "best":       {"sel": float(best_sel), "epoch": int(best_epoch),
                   "energy": float(energies_hist[best_epoch])},
    "theta_init":  np.array(theta0).tolist(),
    "theta_final": theta_final.tolist(),
    "sel_hist":    sel_hist,
    "act_on_hist": act_on_hist,
    "act_off_hist": act_off_hist,
    "energies":    energies_hist,
    "wall_s":      float(wall),
}
json_path = OUT / "data_optimization_selective.json"
with open(json_path, "w") as f:
    json.dump(data, f, indent=2)
print(f"  → {json_path}")

print(f"\nDone.  Initial sel = {float(sel0):+.3f}  →  best sel = {best_sel:+.3f}")
print(f"  act_on_final={float(act_on_hist[best_epoch]):.3f}  "
      f"act_off_final={float(act_off_hist[best_epoch]):.3f}")
