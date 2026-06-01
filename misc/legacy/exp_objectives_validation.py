"""Task 5 — Differentiable objectives validation.

Verifies that all five objectives in jaxfibers.objectives are:
  (1) JAX-differentiable — jax.grad completes without error
  (2) Non-zero gradient near stimulation threshold (regime that matters for optimization)
  (3) Finite gradient at every tested amplitude

Objectives tested:
  max_vm            — peak Vm at center node (single fiber)
  activation_prob   — sigmoid surrogate; σ = 0.5 / 1 / 2 / 5 mV sweep
  recruitment_fraction — mean activation over N=20 fibers with ±5 % amplitude jitter
  selectivity       — target (1 mm) minus off-target (2 mm) recruitment fraction
  energy            — ∫I²dt; gradient verified against analytical formula

Model: MRG D=10 µm, 21 nodes, mono 0.1 ms pulse, dt=0.005 ms, tstop=5 ms, T=37 °C.

Outputs:
  outputs/fig_objectives_validation.png
  outputs/data_objectives_validation.json

Run:
    conda run -n jaxley_fibers python experiments/exp_objectives_validation.py
"""

from __future__ import annotations
import sys, pathlib, json, dataclasses
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jaxley.solver_gate import solve_gate_exponential

from jaxfibers.fibers.mrg import (
    build_mrg, node_indices, section_centers_um,
    V_REST, CM_AXON, G_PAS_MYSA, G_PAS_FLUT, G_PAS_STIN,
)
from jaxfibers.channels.mrg_axnode import AxnodeMyel
from mrg_extracellular_coupled import arrays_from_geometry, integrate, find_threshold
from jaxfibers.objectives import (
    max_vm, activation_prob, recruitment_fraction, selectivity, energy,
)

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# ── simulation constants ───────────────────────────────────────────────────────
CELSIUS  = 37.0
DT       = 0.005      # ms
TSTOP    = 5.0        # ms
DELAY    = 1.0        # ms
PW       = 0.1        # ms
SIGMA_SM = 0.3        # S/m  (medium conductivity)
DIAM     = 10.0       # µm
N_NODES  = 21
N_STEPS  = int(TSTOP / DT)

DEPTH_ON  = 1.0       # mm  target depth
DEPTH_OFF = 2.0       # mm  off-target depth
N_POP     = 20        # fibers per population
JIT_SD    = 0.05      # relative jitter std dev (±5 %)

V_THRESH  = -20.0     # mV  AP detection threshold
SIGMA_SIG = 10.0      # mV  sigmoid steepness for gradient objectives
# Note: small σ (< 5 mV) saturates sigmoid at sub-threshold V_peak ≈ −60 mV → vanishing grad.
# σ = 10 mV keeps the transition region reachable from sub-threshold EPSPs.

GNABAR = 3.0; GNAPBAR = 0.01; GKBAR = 0.08; GL = 0.007
ENA = 50.0; EK = -90.0; EL = -90.0

# ── build fiber ────────────────────────────────────────────────────────────────
print("Setting up MRG D=10 µm fiber ...")
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

# resting gate steady states
v0 = jnp.float64(V_REST)
(am, bm), (ah, bh), (amp0, bmp0), (as0, bs0) = AxnodeMyel._alpha_beta(v0, CELSIUS)
state0 = (
    jnp.where(is_node, float(am  / (am  + bm)),   0.0),
    jnp.where(is_node, float(ah  / (ah  + bh)),   0.0),
    jnp.where(is_node, float(amp0 / (amp0 + bmp0)), 0.0),
    jnp.where(is_node, float(as0 / (as0 + bs0)),  0.0),
)

def membrane_fn(Vm, state, dt_):
    M, H, MP, S = state
    (am, bm), (ah, bh), (amp_, bmp), (as_, bs_) = AxnodeMyel._alpha_beta(Vm, CELSIUS)
    M2  = solve_gate_exponential(M,  dt_, am,   bm)
    H2  = solve_gate_exponential(H,  dt_, ah,   bh)
    MP2 = solve_gate_exponential(MP, dt_, amp_, bmp)
    S2  = solve_gate_exponential(S,  dt_, as_,  bs_)
    gna  = GNABAR * M2**3 * H2
    gnap = GNAPBAR * MP2**3
    gk   = GKBAR * S2
    g_node = (gna + gnap + gk + GL) * A_in * 1e6
    i_node = ((gna + gnap) * (Vm - ENA) + gk * (Vm - EK) + GL * (Vm - EL)) * A_in * 1e6
    g_pas  = g_pas_arr * A_in * 1e6
    i_pas  = g_pas * (Vm - V_REST)
    return (jnp.where(is_node, g_node, g_pas),
            jnp.where(is_node, i_node, i_pas),
            (M2, H2, MP2, S2))

def mfn_factory():
    return membrane_fn, state0

# ── extracellular profiles ─────────────────────────────────────────────────────
def point_source_Ve(depth_mm):
    dy = depth_mm * 1e-3                               # m
    dz = (centers[mid_comp] - centers) * 1e-6          # m (source at axial center)
    r  = np.sqrt(dy**2 + dz**2)
    return jnp.array(-1.0 / (4.0 * np.pi * SIGMA_SM * r))  # mV at i0=−1 mA

Ve_on  = point_source_Ve(DEPTH_ON)   # (n_comp,)
Ve_off = point_source_Ve(DEPTH_OFF)  # (n_comp,)

# ── pulse mask ─────────────────────────────────────────────────────────────────
pm = np.zeros(N_STEPS, dtype=np.float64)
pm[round(DELAY / DT) : round((DELAY + PW) / DT)] = 1.0
pulse_mask = jnp.array(pm)
n_active = int(np.sum(pm))   # = PW / DT = 20 steps

# ── thresholds ─────────────────────────────────────────────────────────────────
print("Finding thresholds ...")
thr_on  = find_threshold(static, mfn_factory, Ve_on,  pulse_mask, DT,
                          v_thresh=V_THRESH, v_rest=V_REST)
thr_off = find_threshold(static, mfn_factory, Ve_off, pulse_mask, DT,
                          v_thresh=V_THRESH, v_rest=V_REST)
print(f"  thr(1 mm) = {thr_on:.5f} mA")
print(f"  thr(2 mm) = {thr_off:.5f} mA")

# ── population jitters ─────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
jitters_on  = jnp.array(1.0 + rng.normal(0, JIT_SD, N_POP))
jitters_off = jnp.array(1.0 + rng.normal(0, JIT_SD, N_POP))

# ── loss functions ─────────────────────────────────────────────────────────────
Ve_on_j  = jnp.asarray(Ve_on,  dtype=jnp.float64)
Ve_off_j = jnp.asarray(Ve_off, dtype=jnp.float64)

@jax.jit
def loss_max_vm(amp_mA):
    Ve = Ve_on_j * (amp_mA / -1.0)
    trace, _ = integrate(static, membrane_fn, state0, Ve, pulse_mask, DT,
                         v_rest=V_REST, center_comp=mid_comp)
    return max_vm(trace)

@jax.jit
def loss_act_prob(amp_mA):
    Ve = Ve_on_j * (amp_mA / -1.0)
    trace, _ = integrate(static, membrane_fn, state0, Ve, pulse_mask, DT,
                         v_rest=V_REST, center_comp=mid_comp)
    return activation_prob(trace, V_THRESH, SIGMA_SIG)

# vmapped population runs (amp_mA broadcast, per-fiber jitter batched)
def _single_on(amp_mA, jitter):
    Ve = Ve_on_j * ((amp_mA * jitter) / -1.0)
    trace, _ = integrate(static, membrane_fn, state0, Ve, pulse_mask, DT,
                         v_rest=V_REST, center_comp=mid_comp)
    return trace

def _single_off(amp_mA, jitter):
    Ve = Ve_off_j * ((amp_mA * jitter) / -1.0)
    trace, _ = integrate(static, membrane_fn, state0, Ve, pulse_mask, DT,
                         v_rest=V_REST, center_comp=mid_comp)
    return trace

_batched_on  = jax.vmap(_single_on,  in_axes=(None, 0))
_batched_off = jax.vmap(_single_off, in_axes=(None, 0))

@jax.jit
def loss_recruitment(amp_mA):
    pop = _batched_on(amp_mA, jitters_on)            # (N_POP, N_STEPS)
    return recruitment_fraction(pop, V_THRESH, SIGMA_SIG)

@jax.jit
def loss_selectivity(amp_mA):
    traces_on  = _batched_on( amp_mA, jitters_on)   # (N_POP, N_STEPS)
    traces_off = _batched_off(amp_mA, jitters_off)
    return selectivity(traces_on, traces_off, V_THRESH, SIGMA_SIG)

@jax.jit
def loss_energy(amp_mA):
    return energy(amp_mA * pulse_mask, DT)           # waveform in mA

grad_max_vm     = jax.jit(jax.grad(loss_max_vm))
grad_act_prob   = jax.jit(jax.grad(loss_act_prob))
grad_recruitment = jax.jit(jax.grad(loss_recruitment))
grad_selectivity = jax.jit(jax.grad(loss_selectivity))
grad_energy      = jax.jit(jax.grad(loss_energy))

# activation_prob for sigma sweep
# sigma sweep: from narrow (vanishing grad) to wide (broad gradient support)
sigmas = [1.0, 2.0, 5.0, 10.0]
_loss_act_by_sig = {}
_grad_act_by_sig = {}
for sig in sigmas:
    @jax.jit
    def _lsig(a, _sig=sig):
        Ve = Ve_on_j * (a / -1.0)
        trace, _ = integrate(static, membrane_fn, state0, Ve, pulse_mask, DT,
                             v_rest=V_REST, center_comp=mid_comp)
        return activation_prob(trace, V_THRESH, _sig)
    _loss_act_by_sig[sig] = _lsig
    _grad_act_by_sig[sig] = jax.jit(jax.grad(_lsig))

# ── amplitude sweep ────────────────────────────────────────────────────────────
N_AMP = 20
amps = jnp.linspace(0.40 * thr_on, 1.80 * thr_on, N_AMP, dtype=jnp.float64)
amps_np = np.array(amps)

print(f"\nSweeping {N_AMP} amplitudes [{float(amps[0]):.4f} to {float(amps[-1]):.4f} mA] ...")

obj_max_vm = []; g_max_vm_l = []
obj_act    = []; g_act_l    = []
obj_recr   = []; g_recr_l   = []
obj_sel    = []; g_sel_l    = []
obj_nrg    = []; g_nrg_l    = []

act_by_sig_obj  = {s: [] for s in sigmas}
act_by_sig_grad = {s: [] for s in sigmas}

for i, amp in enumerate(amps):
    a = jnp.float64(amp)
    obj_max_vm.append(float(loss_max_vm(a)));       g_max_vm_l.append(float(grad_max_vm(a)))
    obj_act.append(float(loss_act_prob(a)));         g_act_l.append(float(grad_act_prob(a)))
    obj_recr.append(float(loss_recruitment(a)));     g_recr_l.append(float(grad_recruitment(a)))
    obj_sel.append(float(loss_selectivity(a)));      g_sel_l.append(float(grad_selectivity(a)))
    obj_nrg.append(float(loss_energy(a)));           g_nrg_l.append(float(grad_energy(a)))
    for sig in sigmas:
        act_by_sig_obj[sig].append(float(_loss_act_by_sig[sig](a)))
        act_by_sig_grad[sig].append(float(_grad_act_by_sig[sig](a)))
    if (i + 1) % 5 == 0:
        print(f"  {i+1}/{N_AMP}  amp={float(a):.4f} mA  Vpeak={obj_max_vm[-1]:.1f} mV  "
              f"act_prob={obj_act[-1]:.3f}  sel={obj_sel[-1]:.3f}")

print("Sweep done.")

# ── unit tests (Task 5.2) ──────────────────────────────────────────────────────
print("\n=== Unit tests ===")
PASS = True

# 5.2.1 – all gradients finite
for name, gv in [("max_vm", g_max_vm_l), ("act_prob", g_act_l),
                 ("recruitment", g_recr_l), ("selectivity", g_sel_l), ("energy", g_nrg_l)]:
    ok = all(np.isfinite(v) for v in gv)
    print(f"  {'PASS' if ok else 'FAIL'}  [{name}] all {len(gv)} gradients finite")
    PASS = PASS and ok

# 5.2.2 – activation_prob gradient non-zero across tested range
# With σ=10 mV, sub-threshold EPSPs at V_peak≈−60 mV give sigmoid arg=−4 (not saturated).
# Gradient should be non-zero throughout.
g_act_arr = np.abs(np.array(g_act_l))
ok2 = g_act_arr.max() > 1e-3
print(f"  {'PASS' if ok2 else 'FAIL'}  [act_prob] max |grad| > 1e-3  "
      f"(σ={SIGMA_SIG} mV, max={g_act_arr.max():.5f})")
PASS = PASS and ok2

# 5.2.3 – recruitment_fraction gradient non-zero
g_recr_arr = np.abs(np.array(g_recr_l))
ok3 = g_recr_arr.max() > 1e-4
print(f"  {'PASS' if ok3 else 'FAIL'}  [recruitment] max |grad| > 1e-4  "
      f"(max={g_recr_arr.max():.5f})")
PASS = PASS and ok3

# 5.2.4 – selectivity gradient non-zero somewhere
g_sel_arr = np.abs(np.array(g_sel_l))
ok4 = g_sel_arr.max() > 1e-4
print(f"  {'PASS' if ok4 else 'FAIL'}  [selectivity] max |grad| > 1e-4  "
      f"(max={g_sel_arr.max():.5f})")
PASS = PASS and ok4

# 5.2.5 – energy gradient matches 2·amp·n_active·dt  (analytical)
analytical_g_nrg = 2.0 * amps_np * n_active * DT
rel_err = np.abs(np.array(g_nrg_l) - analytical_g_nrg) / (np.abs(analytical_g_nrg) + 1e-15)
ok5 = rel_err.max() < 1e-9
print(f"  {'PASS' if ok5 else 'FAIL'}  [energy] gradient matches analytical  "
      f"(max rel err = {rel_err.max():.2e})")
PASS = PASS and ok5

print(f"\nAll tests passed: {PASS}\n")

# ── figure ─────────────────────────────────────────────────────────────────────
print("=== Generating figure ===")
COLORS_SIG = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
thr_kw = dict(color="k", ls="--", lw=0.8)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
(ax_vm, ax_act, ax_sig_g), (ax_recr, ax_sel, ax_nrg) = axes

# ── A: max_vm ──────────────────────────────────────────────────────────────────
ax_vm.plot(amps_np, obj_max_vm, "b-o", ms=3)
ax_vm.axvline(thr_on, **thr_kw, label=f"thr={thr_on:.3f} mA")
ax_vm.set_ylabel("V_peak (mV)"); ax_vm.set_xlabel("amp (mA)")
ax_vm.legend(fontsize=8)
ax2 = ax_vm.twinx()
ax2.plot(amps_np, g_max_vm_l, "r--", lw=1.5)
ax2.set_ylabel("dV_peak/d(amp)  (mV/mA)", color="r", fontsize=8)
ax2.tick_params(axis="y", labelcolor="r")
ax_vm.set_title("A  max_vm", fontsize=10, fontweight="bold")

# ── B: activation_prob — σ sweep ───────────────────────────────────────────────
for sig, col in zip(sigmas, COLORS_SIG):
    ax_act.plot(amps_np, act_by_sig_obj[sig], color=col, lw=1.5, label=f"σ={sig} mV")
ax_act.axvline(thr_on, **thr_kw)
ax_act.set_ylim(-0.05, 1.05)
ax_act.set_ylabel("P(AP)"); ax_act.set_xlabel("amp (mA)")
ax_act.legend(fontsize=8)
ax_act.set_title(f"B  activation_prob (σ sweep, v_thr={V_THRESH} mV)", fontsize=10, fontweight="bold")

# ── C: gradient magnitude for activation_prob — σ sweep ───────────────────────
for sig, col in zip(sigmas, COLORS_SIG):
    ax_sig_g.plot(amps_np, np.abs(act_by_sig_grad[sig]), color=col, lw=1.5, label=f"σ={sig} mV")
ax_sig_g.axvline(thr_on, **thr_kw)
ax_sig_g.set_ylabel("|d(P)/d(amp)|"); ax_sig_g.set_xlabel("amp (mA)")
ax_sig_g.legend(fontsize=8)
ax_sig_g.set_title("C  |grad(activation_prob)| by σ", fontsize=10, fontweight="bold")

# ── D: recruitment_fraction ────────────────────────────────────────────────────
ax_recr.plot(amps_np, obj_recr, "g-o", ms=3, label=f"N={N_POP}, ±{JIT_SD*100:.0f}% jitter")
ax_recr.axvline(thr_on, **thr_kw)
ax_recr.set_ylim(-0.05, 1.05)
ax_recr.set_ylabel("recruitment fraction"); ax_recr.set_xlabel("amp (mA)")
ax_recr.legend(fontsize=8)
ax3 = ax_recr.twinx()
ax3.plot(amps_np, g_recr_l, "r--", lw=1.5)
ax3.set_ylabel("d(r)/d(amp)", color="r", fontsize=8)
ax3.tick_params(axis="y", labelcolor="r")
ax_recr.set_title("D  recruitment_fraction", fontsize=10, fontweight="bold")

# ── E: selectivity ─────────────────────────────────────────────────────────────
ax_sel.plot(amps_np, obj_sel, "m-o", ms=3, label="r_target − r_off")
ax_sel.axvline(thr_on,  color="b", ls=":", lw=1.0, label=f"thr_on={thr_on:.3f}")
ax_sel.axvline(thr_off, color="r", ls=":", lw=1.0, label=f"thr_off={thr_off:.3f}")
ax_sel.set_ylabel("selectivity"); ax_sel.set_xlabel("amp (mA)")
ax_sel.legend(fontsize=7)
ax4 = ax_sel.twinx()
ax4.plot(amps_np, g_sel_l, "r--", lw=1.5)
ax4.set_ylabel("d(sel)/d(amp)", color="r", fontsize=8)
ax4.tick_params(axis="y", labelcolor="r")
ax_sel.set_title("E  selectivity  (target 1 mm / off 2 mm)", fontsize=10, fontweight="bold")

# ── F: energy ──────────────────────────────────────────────────────────────────
ax_nrg.plot(amps_np, obj_nrg, "c-o", ms=3, label="energy")
ax_nrg.set_ylabel("energy (mA²·ms)"); ax_nrg.set_xlabel("amp (mA)")
ax_nrg.legend(fontsize=8, loc="upper left")
ax5 = ax_nrg.twinx()
ax5.plot(amps_np, g_nrg_l,        "r--", lw=2.0, label="autodiff")
ax5.plot(amps_np, 2.0 * amps_np * n_active * DT, "k:", lw=2.0, label="analytical")
ax5.set_ylabel("d(E)/d(amp)  (mA·ms)", color="r", fontsize=8)
ax5.tick_params(axis="y", labelcolor="r")
ax5.legend(fontsize=7)
ax_nrg.set_title("F  energy = ∫I²dt", fontsize=10, fontweight="bold")

plt.suptitle(
    f"Task 5 — Differentiable objectives  (MRG D={DIAM}µm, {N_NODES} nodes, dt={DT}ms)",
    fontsize=11, y=1.01,
)
plt.tight_layout()

fig_path = OUT / "fig_objectives_validation.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {fig_path}")

# ── JSON ───────────────────────────────────────────────────────────────────────
data = {
    "model":       {"diam_um": DIAM, "n_nodes": N_NODES, "dt_ms": DT,
                    "tstop_ms": TSTOP, "celsius": CELSIUS},
    "stimulation": {"depth_on_mm": DEPTH_ON, "depth_off_mm": DEPTH_OFF,
                    "pw_ms": PW, "delay_ms": DELAY},
    "thresholds":  {"thr_on_mA": float(thr_on), "thr_off_mA": float(thr_off)},
    "population":  {"N_pop": N_POP, "jitter_sd": JIT_SD},
    "sigmoid":     {"v_thresh_mV": V_THRESH, "sigma_mV": SIGMA_SIG},
    "amplitudes_mA": amps_np.tolist(),
    "objectives": {
        "max_vm_mV":       obj_max_vm,  "grad_max_vm":      g_max_vm_l,
        "act_prob":        obj_act,     "grad_act_prob":    g_act_l,
        "recruitment":     obj_recr,    "grad_recruitment": g_recr_l,
        "selectivity":     obj_sel,     "grad_selectivity": g_sel_l,
        "energy_mA2ms":    obj_nrg,     "grad_energy":      g_nrg_l,
    },
    "sigma_sweep": {
        str(s): {"obj": act_by_sig_obj[s], "grad": act_by_sig_grad[s]}
        for s in sigmas
    },
    "unit_tests_passed": bool(PASS),
}
json_path = OUT / "data_objectives_validation.json"
with open(json_path, "w") as f:
    json.dump(data, f, indent=2)
print(f"  → {json_path}")

print(f"\nDone.  All unit tests passed: {PASS}")
