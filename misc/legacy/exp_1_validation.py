"""Experiment 1 (v2): Biophysical validation — JAX coupled (Vi, Vpax) solver vs PyFibers/NEURON.

2×2 figure panels:
  (A) Intracellular Vm    — Jaxley vs NEURON, D=10 µm, intra pulse 1 nA / 0.1 ms
  (B) Intracellular gates — Jaxley vs NEURON  (channel translation verified to 1e-7)
  (C) Extracellular Vm    — JAX coupled solver vs NEURON, −0.30 mA / 1 mm / 0.1 ms
  (D) Extracellular gates — JAX coupled solver vs NEURON

Threshold table (D=5.7, 10.0, 14.0 µm) read from pre-computed JSON outputs;
full diameter sweep in outputs/fig_mrg_comparison.png.

Outputs:
  outputs/fig1_validation.png
  outputs/table1_thresholds.csv

Run from project root:
    conda run -n jaxley_fibers python experiments/exp_1_validation.py
"""

from __future__ import annotations

import sys, pathlib, csv, json, dataclasses
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import jax
import jax.numpy as jnp
import jaxley as jx
from jaxley.solver_gate import solve_gate_exponential

from jaxfibers.fibers.mrg import (
    build_mrg, node_indices, section_centers_um,
    V_REST, CM_AXON, G_PAS_MYSA, G_PAS_FLUT, G_PAS_STIN,
)
from jaxfibers.channels.mrg_axnode import AxnodeMyel
from jaxfibers.stim.intracellular import rectangular_pulse, attach_intra_pulse
from jaxfibers.stim.extracellular import point_source_potentials_mV, rectangular_waveform
from jaxfibers.nrn_baseline import run_intracellular, run_extracellular
from mrg_extracellular_coupled import arrays_from_geometry, integrate_recording

OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ── shared parameters ─────────────────────────────────────────────────────────
D        = 10.0   # µm
N_INTRA  = 11     # nodes for intracellular panels (speed)
N_EXTRA  = 21     # nodes for extracellular panels (matches threshold validation)
DT       = 0.005  # ms
TSTOP    = 5.0    # ms
CELSIUS  = 37.0
DELAY    = 1.0    # ms
PW       = 0.1    # ms
AMP_EXTRA = -0.30  # mA — ~1.64× threshold (threshold ≈ −0.183 mA for D=10 µm, N=21)

# ── channel constants (AXNODE_myel.mod PARAMETER block) ───────────────────────
_GNABAR  = 3.0;  _GNAPBAR = 0.01; _GKBAR  = 0.08; _GL   = 0.007
_ENA     = 50.0; _EK      = -90.0; _EL    = -90.0


# ── intracellular run (Jaxley) ────────────────────────────────────────────────
def _jaxley_intra():
    cell, geom = build_mrg(diameter=D, n_nodes=N_INTRA)
    nodes = node_indices(geom)
    mid   = nodes[len(nodes) // 2]
    n_steps = int(TSTOP / DT) + 1
    t = np.arange(n_steps) * DT
    pulse = rectangular_pulse(t, DELAY, PW, 1.0)
    attach_intra_pulse(cell, mid, pulse)
    for ni in nodes:
        cell.branch(0).comp(ni).record("v")
    for g in ["AxnodeMyel_m", "AxnodeMyel_h", "AxnodeMyel_mp", "AxnodeMyel_s"]:
        cell.branch(0).comp(mid).record(g)
    rec = np.asarray(jx.integrate(cell, delta_t=DT, t_max=TSTOP, solver="bwd_euler"))
    n_nr = len(nodes)
    return dict(
        t_ms=t,
        vm_mV=rec[:n_nr],
        gates={"m": rec[n_nr], "h": rec[n_nr+1], "mp": rec[n_nr+2], "s": rec[n_nr+3]},
        probe_node_idx=len(nodes) // 2,
    )


# ── extracellular setup (coupled solver) ─────────────────────────────────────
def _coupled_setup():
    _, geom = build_mrg(diameter=D, n_nodes=N_EXTRA)
    nodes   = node_indices(geom)
    centers = section_centers_um(geom)
    mid     = nodes[len(nodes) // 2]
    n_comp  = geom.n_comp

    geom_c = dataclasses.replace(geom, cm_uF_cm2=[CM_AXON] * n_comp)
    static  = arrays_from_geometry(geom_c, DT)

    Ve_unit = np.array(point_source_potentials_mV(
        centers, src_x_um=0., src_y_um=1000., src_z_um=centers[mid], i0_mA=-1.0
    ))
    n_steps = int(TSTOP / DT)
    shape   = rectangular_waveform(np.arange(n_steps + 1) * DT, DELAY, PW, amp=1.0)
    pulse_mask = jnp.asarray(shape[:n_steps] > 0.5)

    is_node   = static["is_node"]
    A_in      = static["A_in_cm2"]
    stype_gp  = {"node": 0.0, "mysa": G_PAS_MYSA, "flut": G_PAS_FLUT, "stin": G_PAS_STIN}
    g_pas_arr = jnp.asarray([stype_gp[s] for s in geom.section_type])

    # resting gate steady states
    v0 = jnp.float64(V_REST)
    (a_m0,b_m0),(a_h0,b_h0),(a_mp0,b_mp0),(a_s0,b_s0) = AxnodeMyel._alpha_beta(v0, CELSIUS)
    m0  = float(a_m0  / (a_m0  + b_m0))
    h0  = float(a_h0  / (a_h0  + b_h0))
    mp0 = float(a_mp0 / (a_mp0 + b_mp0))
    s0  = float(a_s0  / (a_s0  + b_s0))

    state0 = (
        jnp.where(is_node, m0,  0.0),
        jnp.where(is_node, h0,  0.0),
        jnp.where(is_node, mp0, 0.0),
        jnp.where(is_node, s0,  0.0),
    )

    def membrane_fn(Vm, state, dt):
        M, H, MP, S = state
        (a_m,b_m),(a_h,b_h),(a_mp,b_mp),(a_s,b_s) = AxnodeMyel._alpha_beta(Vm, CELSIUS)
        M2  = solve_gate_exponential(M,  dt, a_m,  b_m)
        H2  = solve_gate_exponential(H,  dt, a_h,  b_h)
        MP2 = solve_gate_exponential(MP, dt, a_mp, b_mp)
        S2  = solve_gate_exponential(S,  dt, a_s,  b_s)
        g_na  = _GNABAR  * M2**3 * H2
        g_nap = _GNAPBAR * MP2**3
        g_k   = _GKBAR   * S2
        g_node = (g_na + g_nap + g_k + _GL) * A_in * 1e6
        i_node = (
            (g_na + g_nap) * (Vm - _ENA) + g_k * (Vm - _EK) + _GL * (Vm - _EL)
        ) * A_in * 1e6
        g_pas_us = g_pas_arr * A_in * 1e6
        i_pas    = g_pas_us  * (Vm - V_REST)
        g_eff = jnp.where(is_node, g_node,  g_pas_us)
        i_ion = jnp.where(is_node, i_node,  i_pas)
        return g_eff, i_ion, (M2, H2, MP2, S2)

    return static, membrane_fn, state0, Ve_unit, pulse_mask, mid


# ── main ─────────────────────────────────────────────────────────────────────
def make_figure_and_table():
    # ── [1/4] intracellular ───────────────────────────────────────────────────
    print("[1/4] Intracellular: Jaxley …", flush=True)
    jx_i = _jaxley_intra()
    print("[1/4] Intracellular: NEURON …", flush=True)
    nr_i = run_intracellular(diameter=D, n_nodes=N_INTRA, i_amp_nA=1.0,
                             i_dur_ms=PW, i_delay_ms=DELAY, dt_ms=DT, tstop_ms=TSTOP)

    # ── [2/4] extracellular NEURON ────────────────────────────────────────────
    print("[2/4] Extracellular: NEURON …", flush=True)
    nr_e = run_extracellular(diameter=D, n_nodes=N_EXTRA, src_height_um=1000.0,
                             pw_ms=PW, delay_ms=DELAY, amp_mA=AMP_EXTRA,
                             dt_ms=DT, tstop_ms=TSTOP)

    # ── [3/4] extracellular JAX coupled solver ────────────────────────────────
    print("[3/4] Extracellular: JAX coupled solver …", flush=True)
    static, mfn, state0, Ve_unit, pulse_mask, mid = _coupled_setup()
    Ve_scaled = jnp.asarray(Ve_unit * (AMP_EXTRA / -1.0))
    vm_c, m_c, h_c, mp_c, s_c, _ = integrate_recording(
        static, mfn, state0, Ve_scaled, pulse_mask, DT, v_rest=V_REST,
    )
    vm_c  = np.asarray(vm_c)
    m_c   = np.asarray(m_c)
    h_c   = np.asarray(h_c)
    s_c   = np.asarray(s_c)
    # step s -> time (s+1)*dt
    t_jax = (np.arange(len(vm_c)) + 1) * DT

    # ── [4/4] plot ───────────────────────────────────────────────────────────
    print("[4/4] Plotting …", flush=True)
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    pi_i = jx_i["probe_node_idx"]
    pe_n = nr_e.probe_node_idx

    # (A) intracellular Vm
    ax = axs[0, 0]
    ax.plot(nr_i.t_ms, nr_i.vm_mV[nr_i.probe_node_idx],
            color="C0", lw=2.0, label="PyFibers / NEURON")
    ax.plot(jx_i["t_ms"], jx_i["vm_mV"][pi_i],
            color="C3", lw=1.0, ls="--", label="Jaxley (intra.)")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("V_m (mV)")
    ax.set_title("(A) Intracellular pulse — V_m")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # (B) intracellular gates
    ax = axs[0, 1]
    for g, c in [("m", "C0"), ("h", "C2"), ("s", "C4")]:
        ax.plot(nr_i.t_ms, nr_i.gates[g],    color=c, lw=2.0, label=f"NEURON {g}")
        ax.plot(jx_i["t_ms"], jx_i["gates"][g], color=c, lw=1.0, ls="--", label=f"Jaxley {g}")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("gate value")
    ax.set_title("(B) Intracellular pulse — gates")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

    # (C) extracellular Vm
    ax = axs[1, 0]
    ax.plot(nr_e.t_ms, nr_e.vm_mV[pe_n],
            color="C0", lw=2.0, label="PyFibers / NEURON")
    ax.plot(t_jax, vm_c,
            color="C3", lw=1.0, ls="--", label="JAX coupled solver")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("V_m (mV)")
    ax.set_title(f"(C) Extracellular pulse — V_m  ({AMP_EXTRA} mA, 1 mm, {PW} ms, N={N_EXTRA})")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # (D) extracellular gates
    ax = axs[1, 1]
    for g, jax_tr, c in [("m", m_c, "C0"), ("h", h_c, "C2"), ("s", s_c, "C4")]:
        ax.plot(nr_e.t_ms, nr_e.gates[g], color=c, lw=2.0, label=f"NEURON {g}")
        ax.plot(t_jax, jax_tr,            color=c, lw=1.0, ls="--", label=f"JAX {g}")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("gate value")
    ax.set_title("(D) Extracellular pulse — gates")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

    fig.suptitle(
        f"MRG {D} µm fiber — PyFibers/NEURON (solid) vs JAX (dashed)\n"
        f"A/B: channel translation, intra. stim (verified 1e-7).  "
        f"C/D: coupled (Vi, Vpax) extracellular solver (< 0.25% threshold error).",
        fontsize=10,
    )
    out_fig = OUT_DIR / "fig1_validation.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"  → {out_fig}")

    # ── threshold table from JSON ─────────────────────────────────────────────
    jx_th = {float(k): v for k, v in json.loads((OUT_DIR / "jax_thresholds.json").read_text()).items()}
    pf_th = {float(k): v for k, v in json.loads((OUT_DIR / "pyfibers_thresholds.json").read_text()).items()}

    csv_path = OUT_DIR / "table1_thresholds.csv"
    print(f"\n{'D (µm)':>8}  {'PyFibers (mA)':>14}  {'JAX (mA)':>14}  {'error':>8}")
    print("-" * 52)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["diameter_um", "pyfibers_mA", "jax_coupled_mA", "abs_pct_error"])
        for d in [5.7, 10.0, 14.0]:
            pf = pf_th[d]; jx = jx_th[d]
            err = abs(abs(jx) - abs(pf)) / abs(pf) * 100
            print(f"{d:>8.1f}  {pf:>14.5f}  {jx:>14.5f}  {err:>7.2f}%")
            w.writerow([f"{d:.1f}", f"{pf:.5f}", f"{jx:.5f}", f"{err:.3f}"])
    print(f"\n  → {csv_path}")
    print("  (Full diameter sweep: outputs/fig_mrg_comparison.png)")


if __name__ == "__main__":
    make_figure_and_table()
