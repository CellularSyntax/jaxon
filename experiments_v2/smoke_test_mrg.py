"""Smoke test for mrg_validation fixes.

Checks two things in ~2-5 minutes:
  1. Intracellular traces: JAX (coupled solver) fires and agrees with NEURON.
  2. Anodic bisection: mono_a SD does not spike at PW=1 ms (expansion-direction fix).

Outputs:
  outputs/smoke_mrg/fig_smoke_mrg.png  — Vm + gate traces (A,B) and anodic SD (C)

Run from project root:
    python experiments_v2/smoke_test_mrg.py
"""

from __future__ import annotations

import sys, pathlib, dataclasses, time
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jaxley.solver_gate import solve_gate_exponential

jax.config.update("jax_enable_x64", True)

from jaxfibers.fibers.mrg import (
    build_mrg, node_indices, section_centers_um,
    V_REST, CM_AXON, G_PAS_MYSA, G_PAS_FLUT, G_PAS_STIN,
)
from jaxfibers.channels.mrg_axnode import AxnodeMyel
from jaxfibers.stim.extracellular import point_source_potentials_mV
from jaxfibers.nrn_baseline import run_intracellular
from jaxfibers.stim.extracellular_coupled import (
    arrays_from_geometry, integrate, integrate_recording,
)

from experiments_v2.utils import PULSES, make_pulse_array, jax_bisect, ensure_dir

CELSIUS = 37.0
DT      = 0.005
TSTOP   = 8.0
DELAY   = 1.0
N_NODES = 21
N_STEPS = int(TSTOP / DT)
SRC_H   = 1000.0

GNABAR = 3.0; GNAPBAR = 0.01; GKBAR = 0.08; GL = 0.007
ENA = 50.0;   EK = -90.0;     EL   = -90.0

PASS_THRESH_MV   = 5.0    # peak must be > V_REST + this to count as AP (both JAX and NEURON)
PASS_ANODIC_RATIO = 4.0   # mono_a threshold at PW=1ms must be < this × PW=0.1ms threshold


def _make_jax_setup(D, n_nodes=None):
    if n_nodes is None:
        n_nodes = N_NODES
    _, geom = build_mrg(diameter=D, n_nodes=n_nodes)
    nodes   = node_indices(geom)
    centers = np.array(section_centers_um(geom))
    mid     = nodes[len(nodes) // 2]
    n_comp  = geom.n_comp
    geom_c  = dataclasses.replace(geom, cm_uF_cm2=[CM_AXON] * n_comp)
    static  = arrays_from_geometry(geom_c, DT)
    Ve_unit = np.array(point_source_potentials_mV(
        list(centers), src_x_um=0., src_y_um=SRC_H,
        src_z_um=float(centers[mid]), i0_mA=-1.0,
    ))
    is_node   = static["is_node"]
    A_in      = static["A_in_cm2"]
    g_pas_arr = jnp.asarray([{"node": 0.0, "mysa": G_PAS_MYSA,
                               "flut": G_PAS_FLUT, "stin": G_PAS_STIN}[s]
                              for s in geom.section_type])
    v0 = jnp.float64(V_REST)
    (a_m0,b_m0),(a_h0,b_h0),(a_mp0,b_mp0),(a_s0,b_s0) = AxnodeMyel._alpha_beta(v0, CELSIUS)
    state0 = (
        jnp.where(is_node, float(a_m0/(a_m0+b_m0)),   0.0),
        jnp.where(is_node, float(a_h0/(a_h0+b_h0)),   0.0),
        jnp.where(is_node, float(a_mp0/(a_mp0+b_mp0)),0.0),
        jnp.where(is_node, float(a_s0/(a_s0+b_s0)),   0.0),
    )
    def membrane_fn(Vm, state, dt):
        M, H, MP, S = state
        (a_m,b_m),(a_h,b_h),(a_mp,b_mp),(a_s,b_s) = AxnodeMyel._alpha_beta(Vm, CELSIUS)
        M2  = solve_gate_exponential(M,  dt, a_m,  b_m)
        H2  = solve_gate_exponential(H,  dt, a_h,  b_h)
        MP2 = solve_gate_exponential(MP, dt, a_mp, b_mp)
        S2  = solve_gate_exponential(S,  dt, a_s,  b_s)
        g_na  = GNABAR  * M2**3 * H2
        g_nap = GNAPBAR * MP2**3
        g_k   = GKBAR   * S2
        g_node = (g_na + g_nap + g_k + GL) * A_in * 1e6
        i_node = ((g_na + g_nap)*(Vm-ENA) + g_k*(Vm-EK) + GL*(Vm-EL)) * A_in * 1e6
        g_pas_us = g_pas_arr * A_in * 1e6
        g_eff = jnp.where(is_node, g_node,  g_pas_us)
        i_ion = jnp.where(is_node, i_node,  g_pas_us * (Vm - V_REST))
        return g_eff, i_ion, (M2, H2, MP2, S2)

    return static, membrane_fn, state0, Ve_unit, mid, geom


def _make_runner(static, mfn, state0, Ve_unit):
    Ve_j = jnp.asarray(Ve_unit, dtype=jnp.float64)
    @jax.jit
    def run(amp_mA, pulse_shape):
        Ve = Ve_j * (amp_mA / -1.0)
        trace, _ = integrate(static, mfn, state0, Ve, pulse_shape, DT,
                             v_rest=V_REST, record="center")
        return jnp.max(trace)
    return run


OUT = ensure_dir(ROOT / "outputs" / "smoke_mrg")


# ── Test 1: intracellular traces ──────────────────────────────────────────────

N_NODES_INTRA = 11   # fewer nodes so 1 nA fires both JAX and NEURON (legacy: N_INTRA=11)
PASS_DELTA_MV = 5.0  # max |peak_JAX - peak_NEURON| for coupled-solver intracellular


def test_intra():
    print("\n── Test 1: intracellular traces (D=10 µm, coupled solver) ──")
    D = 10.0

    # Build coupled solver with N_NODES_INTRA nodes
    static_i, mfn_i, state0_i, _, mid_i, geom_i = _make_jax_setup(D, N_NODES_INTRA)
    n_comp_i = geom_i.n_comp

    # Intracellular stimulus: 1 nA at mid node for 0.1 ms after DELAY
    n_steps_i    = int(5.0 / DT)
    t_intra      = (np.arange(n_steps_i) + 1) * DT
    pulse_scalar = np.where(
        (t_intra >= DELAY) & (t_intra < DELAY + 0.1), 1.0, 0.0,
    ).astype(np.float64)
    i_intra_arr  = np.zeros((n_steps_i, n_comp_i))
    i_intra_arr[:, mid_i] = pulse_scalar

    Ve_zero = jnp.zeros(n_comp_i, dtype=jnp.float64)
    pm_zero = jnp.zeros(n_steps_i, dtype=jnp.float64)

    t0 = time.time()
    vm_i, m_i, h_i, mp_i, s_i, _ = integrate_recording(
        static_i, mfn_i, state0_i, Ve_zero, pm_zero, DT, v_rest=V_REST,
        center_comp=mid_i,
        i_intra=jnp.asarray(i_intra_arr, dtype=jnp.float64),
    )
    jax_vm    = np.asarray(vm_i)
    jax_gates = {"m": np.asarray(m_i), "h": np.asarray(h_i),
                 "mp": np.asarray(mp_i), "s": np.asarray(s_i)}
    jax_peak  = float(np.max(jax_vm))
    print(f"  JAX   : peak = {jax_peak:.1f} mV  ({time.time()-t0:.1f} s)")

    t0 = time.time()
    nr_i = run_intracellular(diameter=D, n_nodes=N_NODES_INTRA, i_amp_nA=1.0,
                             i_dur_ms=0.1, i_delay_ms=DELAY, dt_ms=DT, tstop_ms=5.0)
    nrn_peak = max(nr_i.vm_mV[nr_i.probe_node_idx])
    print(f"  NEURON: peak = {nrn_peak:.1f} mV  ({time.time()-t0:.1f} s)")

    delta     = abs(jax_peak - nrn_peak)
    jax_fired = jax_peak > V_REST + PASS_THRESH_MV
    nrn_fired = nrn_peak > V_REST + PASS_THRESH_MV
    ok        = jax_fired and nrn_fired and delta < PASS_DELTA_MV
    print(f"  JAX fired={jax_fired}  NRN fired={nrn_fired}  |Δpeak|={delta:.2f} mV  → {'PASS' if ok else 'FAIL'}")

    return ok, t_intra, jax_vm, jax_gates, nr_i


# ── Test 2: anodic SD bisection ───────────────────────────────────────────────

def test_anodic_sd():
    print("\n── Test 2: anodic SD bisection (D=10 µm, mono_a) ──")
    D = 10.0
    static, mfn, state0, Ve_unit, mid, geom = _make_jax_setup(D)
    runner = _make_runner(static, mfn, state0, Ve_unit)

    spec = PULSES["mono_a"]
    pws  = [0.05, 0.1, 0.2, 0.5, 1.0]
    thresholds = {}
    for pw in pws:
        arr = make_pulse_array("mono_a", pw, N_STEPS, DT, DELAY)
        t0  = time.time()
        try:
            thr = jax_bisect(runner, arr, lo=spec.lo, hi=spec.hi)
            thresholds[pw] = thr
            print(f"  PW={pw:.2f} ms → {thr:.4f} mA  ({time.time()-t0:.1f} s)")
        except RuntimeError as e:
            thresholds[pw] = float("nan")
            print(f"  PW={pw:.2f} ms → FAILED: {e}")

    thr_01 = thresholds.get(0.1, float("nan"))
    thr_10 = thresholds.get(1.0, float("nan"))
    if np.isnan(thr_01) or np.isnan(thr_10):
        ok = False
    else:
        ratio = thr_01 / thr_10
        ok    = ratio < PASS_ANODIC_RATIO
        print(f"  thr(0.1ms)/thr(1.0ms) = {ratio:.2f}  (expected < {PASS_ANODIC_RATIO})  → {'PASS' if ok else 'FAIL (spike likely)'}")

    return ok, pws, thresholds


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(intra_data, sd_data):
    t_jax, jax_vm, jax_gates, nr_i, pws, thresholds = intra_data + sd_data
    gate_colors = {"m": "C0", "h": "C2", "mp": "C4", "s": "C6"}

    fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    # (A) Vm
    ax = axs[0]
    ax.plot(nr_i.t_ms, nr_i.vm_mV[nr_i.probe_node_idx],
            color="C0", lw=2.0, label="NEURON")
    ax.plot(t_jax, jax_vm, color="C3", lw=1.2, ls="--", label="JAX (Jaxley)")
    ax.set_title(f"(A) Intracellular Vm  JAX={max(jax_vm):.1f} mV  NRN={max(nr_i.vm_mV[nr_i.probe_node_idx]):.1f} mV")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("Vm (mV)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (B) Gates
    ax = axs[1]
    for g, c in gate_colors.items():
        nrn_g = nr_i.gates.get(g, [])
        if len(nrn_g):
            ax.plot(nr_i.t_ms, nrn_g, color=c, lw=2.0, label=f"NEURON {g}")
        jg = jax_gates.get(g, [])
        if len(jg):
            ax.plot(t_jax, jg, color=c, lw=1.2, ls="--", label=f"JAX {g}")
    ax.set_title("(B) Gates  (m,h: fast Na;  mp: pers. Na;  s: K)")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("gate value")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    # (C) Anodic SD
    ax = axs[2]
    valid_pws = [pw for pw in pws if not np.isnan(thresholds.get(pw, float("nan")))]
    valid_thr = [thresholds[pw] for pw in valid_pws]
    ax.loglog(valid_pws, valid_thr, "o-", color="C1", lw=1.5, ms=6, label="JAX mono_a")
    ax.set_title("(C) Anodic SD  (D=10 µm)  — should be monotone ↓")
    ax.set_xlabel("PW (ms)"); ax.set_ylabel("|threshold| (mA)")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9)

    fig.suptitle("MRG smoke test — D=10 µm  (intra N=11, extra N=21)", fontsize=11)
    path = OUT / "fig_smoke_mrg.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  → {path}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")
    t_start = time.time()

    r1, t_jax, jax_vm, jax_gates, nr_i = test_intra()
    r2, pws, thresholds                 = test_anodic_sd()

    print("\n── Generating figure ──")
    make_figure(
        (t_jax, jax_vm, jax_gates, nr_i),
        (pws, thresholds),
    )

    print(f"\n{'='*40}")
    print(f"Test 1 (intra coupled)  : {'PASS' if r1 else 'FAIL'}")
    print(f"Test 2 (anodic bisect)  : {'PASS' if r2 else 'FAIL'}")
    print(f"Total time: {time.time()-t_start:.1f} s")
    sys.exit(0 if (r1 and r2) else 1)
