"""Sundt Validation Suite — Task 2 (Phase 1, migration_plan.md v3)

Directly parallel to the MRG validation outputs:

  fig_sundt_traces.png   ← analogue of fig1_validation.png
    (A) Intracellular Vm  — JAX (Jaxley bwd_euler) vs NEURON
    (B) Intracellular gates
    (C) Extracellular Vm  — JAX single-cable solver vs NEURON  ← key panel
    (D) Extracellular gates

  fig_sundt_validation.png  ← analogue of fig_mrg_validation.png
    (A) Strength-duration curves (log-log, D=0.8 µm)
    (B) CV comparison (JAX vs NEURON, D=0.8 µm)
    (C) AP propagation check (all N_NODES compartments at 1.3× threshold)
    (D) Error summary bar chart

  table_sundt_validation.csv

Parameters:
  diameter  = 0.8 µm,  δz = 8.333 µm,  N_NODES = 51
  point source 1 mm above centre, σ = 0.3 S/m, T = 37 °C, dt = 0.005 ms

Run from project root:
    conda run -n jaxley_fibers python experiments/exp_sundt_validation.py
"""

from __future__ import annotations

import sys, pathlib, json, csv, time, dataclasses
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import jax
import jax.numpy as jnp
from jaxley.solver_gate import solve_gate_exponential

from jaxfibers.fibers.sundt import (
    build_sundt, node_indices, section_centers_um,
    V_REST, CM, RA, DELTA_Z,
)
from jaxfibers.channels.sundt_channels import SundtAxon
from jaxfibers.stim.extracellular import point_source_potentials_mV
from jaxfibers.stim.intracellular import rectangular_pulse, attach_intra_pulse
from jaxfibers.nrn_baseline import (
    run_intracellular_sundt, run_extracellular_sundt,
    find_threshold_extracellular_sundt,
)
from mrg_extracellular_coupled import (
    arrays_from_geometry, integrate, integrate_recording,
)
import jaxley as jx

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# ── shared constants ──────────────────────────────────────────────────────────
CELSIUS   = 37.0
DT        = 0.005    # ms
TSTOP     = 10.0     # ms  (C-fiber is slower; needs longer window)
DELAY     = 1.0      # ms
SIGMA     = 0.3      # S/m
DIAM      = 0.8      # µm
N_NODES   = 51       # compartments
N_CV      = 101      # longer fiber for CV timing accuracy
SD_PWS    = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]   # ms
N_STEPS   = int(TSTOP / DT)                      # 2000

# SundtAxon channel constants
_GNA = 0.04; _GKDR = 0.04; _GPAS = 1e-4
_ENA = 50.0; _EK = -90.0;  _EPAS = -60.0
_MSHIFT = -6.0; _HSHIFT = 6.0; _ISHIFT = 0.0
_VHALFN = -32.0; _VHALFL = -61.0
_A0N = 0.03; _A0L = 0.001
_ZETAN = -5.0; _ZETAL = 2.0
_GMN = 0.4; _GML = 1.0


# ── helper: expM1 (from sundt_channels.py, copied inline for membrane_fn) ─────
def _expM1(x, y):
    ratio = x / y
    exp_term = jnp.exp(jnp.clip(ratio, -50.0, 50.0)) - 1.0
    safe = jnp.where(jnp.abs(exp_term) < 1e-30, 1e-30, exp_term)
    return jnp.where(jnp.abs(ratio) < 1e-6, y * (1.0 - 0.5 * ratio), x / safe)


# ── JAX setup (shared across all tasks) ───────────────────────────────────────
def _make_jax_setup(N: int = N_NODES):
    _, geom = build_sundt(diameter=DIAM, n_nodes=N, temperature=CELSIUS)
    centers = np.array(section_centers_um(geom))
    n_comp  = geom.n_comp
    mid     = n_comp // 2

    static = arrays_from_geometry(geom, DT)
    A_in   = static["A_in_cm2"]

    Ve_unit = np.array(point_source_potentials_mV(
        list(centers), src_x_um=0., src_y_um=1000.,
        src_z_um=float(centers[mid]), i0_mA=-1.0,
    ))

    def _nahh_ab(v):
        q10 = 3.0 ** ((CELSIUS - 30.0) / 10.0)
        vm = v + 65.0 + _MSHIFT
        a_m = q10 * 0.32 * _expM1(13.1 - vm, 4.0)
        b_m = q10 * 0.28 * _expM1(vm - 40.1, 5.0)
        vh = v + 65.0 + _HSHIFT
        a_h = q10 * 0.128 * jnp.exp((17.0 - vh + _ISHIFT) / 18.0)
        b_h = q10 * 4.0 / (jnp.exp((40.0 - vh) / 5.0) + 1.0)
        return (a_m, b_m), (a_h, b_h)

    def _borgkdr_ab(v):
        q10  = 3.0 ** ((CELSIUS - 30.0) / 10.0)
        FRT  = 9.648e4 / (8.315 * (273.16 + CELSIUS)) * 1e-3
        alpn = jnp.exp(_ZETAN * (v - _VHALFN) * FRT)
        betn = jnp.exp(_ZETAN * _GMN * (v - _VHALFN) * FRT)
        alpl = jnp.exp(_ZETAL * (v - _VHALFL) * FRT)
        betl = jnp.exp(_ZETAL * _GML * (v - _VHALFL) * FRT)
        a_n = q10 * _A0N / betn;  b_n = q10 * _A0N * alpn / betn
        a_l = q10 * _A0L / betl;  b_l = q10 * _A0L * alpl / betl
        return (a_n, b_n), (a_l, b_l)

    v0 = jnp.float64(V_REST)
    (a_m0,b_m0),(a_h0,b_h0) = _nahh_ab(v0)
    (a_n0,b_n0),(a_l0,b_l0) = _borgkdr_ab(v0)
    m0 = float(a_m0/(a_m0+b_m0)); h0 = float(a_h0/(a_h0+b_h0))
    n0 = float(a_n0/(a_n0+b_n0)); l0 = float(a_l0/(a_l0+b_l0))
    state0 = (
        jnp.full(n_comp, m0, dtype=jnp.float64),
        jnp.full(n_comp, h0, dtype=jnp.float64),
        jnp.full(n_comp, n0, dtype=jnp.float64),
        jnp.full(n_comp, l0, dtype=jnp.float64),
    )

    def membrane_fn(Vm, state, dt):
        M, H, N, L = state
        (a_m,b_m),(a_h,b_h) = _nahh_ab(Vm)
        (a_n,b_n),(a_l,b_l) = _borgkdr_ab(Vm)
        M2 = solve_gate_exponential(M, dt, a_m, b_m)
        H2 = solve_gate_exponential(H, dt, a_h, b_h)
        N2 = solve_gate_exponential(N, dt, a_n, b_n)
        L2 = solve_gate_exponential(L, dt, a_l, b_l)
        g_na  = _GNA  * M2**3 * H2
        g_k   = _GKDR * N2**3 * L2
        g_eff = (g_na + g_k + _GPAS) * A_in * 1e6
        i_ion = (g_na*(Vm-_ENA) + g_k*(Vm-_EK) + _GPAS*(Vm-_EPAS)) * A_in * 1e6
        return g_eff, i_ion, (M2, H2, N2, L2)

    return static, membrane_fn, state0, Ve_unit, mid, centers, geom


def _mono_shape(pw: float) -> np.ndarray:
    t = (np.arange(N_STEPS) + 1) * DT
    return ((t >= DELAY) & (t < DELAY + pw)).astype(np.float64)


def _make_jit_runner(static, mfn, state0, Ve_unit):
    Ve_j = jnp.asarray(Ve_unit, dtype=jnp.float64)
    @jax.jit
    def run(amp_mA, pulse_shape):
        Ve = Ve_j * (amp_mA / -1.0)
        trace, _ = integrate(static, mfn, state0, Ve, pulse_shape, DT,
                             v_rest=V_REST, record="center")
        return jnp.max(trace)
    return run


def _bisect(run_fn, pulse_shape, lo=-5.0, hi=-0.001, tol=1e-4, v_thresh=-30.0):
    pm = jnp.asarray(pulse_shape, dtype=jnp.float64)
    def fires(a): return bool(run_fn(jnp.float64(a), pm) > v_thresh)
    for _ in range(12):
        if fires(lo): break
        lo *= 2.0
    fire_hi = fires(hi)
    while abs(hi - lo) > tol:
        mid = 0.5 * (lo + hi)
        if fires(mid) == fire_hi: hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)


# ── [1] Intracellular traces (panels A/B of fig_sundt_traces.png) ─────────────
def run_intra_traces() -> dict:
    print("[1/5] Intracellular: JAX ...", flush=True)
    I_AMP_NA = 0.5; PW_INTRA = 0.5
    cell, geom2 = build_sundt(diameter=DIAM, n_nodes=N_NODES, temperature=CELSIUS)
    nodes2   = node_indices(geom2)
    mid_node = nodes2[len(nodes2) // 2]
    n_steps2 = int(TSTOP / DT) + 1
    t_arr    = np.arange(n_steps2) * DT
    pulse    = rectangular_pulse(t_arr, DELAY, PW_INTRA, I_AMP_NA)
    attach_intra_pulse(cell, mid_node, pulse)
    for ni in nodes2:
        cell.branch(0).comp(ni).record("v")
    for g in ["SundtAxon_m", "SundtAxon_h", "SundtAxon_n", "SundtAxon_l"]:
        cell.branch(0).comp(mid_node).record(g)
    rec = np.asarray(jx.integrate(cell, delta_t=DT, t_max=TSTOP, solver="bwd_euler"))
    jax_vm  = rec[mid_node]
    jax_gates = {"m": rec[N_NODES], "h": rec[N_NODES+1],
                 "n": rec[N_NODES+2], "l": rec[N_NODES+3]}

    print("[1/5] Intracellular: NEURON ...", flush=True)
    nr = run_intracellular_sundt(diameter=DIAM, n_nodes=N_NODES, temperature=CELSIUS,
                                 i_delay_ms=DELAY, i_dur_ms=PW_INTRA, i_amp_nA=I_AMP_NA,
                                 dt_ms=DT, tstop_ms=TSTOP)

    peak_jax = float(np.max(jax_vm)); peak_nrn = float(np.max(nr.vm_mV[nr.probe_node_idx]))
    print(f"  AP peak: JAX={peak_jax:.3f} mV  NEURON={peak_nrn:.3f} mV  |diff|={abs(peak_jax-peak_nrn):.3f} mV")
    return dict(
        t_jax=t_arr, vm_jax=jax_vm, jax_gates=jax_gates,
        t_nrn=nr.t_ms, vm_nrn=nr.vm_mV[nr.probe_node_idx], nrn_gates=nr.gates,
        peak_jax=peak_jax, peak_nrn=peak_nrn,
    )


# ── [2] Extracellular traces (panels C/D of fig_sundt_traces.png) ─────────────
def run_extra_traces(static, mfn, state0, Ve_unit, mid) -> dict:
    print("[2/5] Extracellular: computing threshold for trace amplitude ...", flush=True)
    runner = _make_jit_runner(static, mfn, state0, Ve_unit)
    pm01   = _mono_shape(0.1)
    thr    = _bisect(runner, pm01)
    amp    = thr * 1.3   # suprathreshold — same as MRG exp_1_validation

    print(f"[2/5] Extracellular: JAX (amp={amp:.3f} mA) ...", flush=True)
    Ve_scaled = jnp.asarray(Ve_unit * (amp / -1.0))
    pm = jnp.asarray(pm01)
    # integrate_recording records (Vm, M, H, MP, S) at centre — adapt for Sundt (M,H,N,L)
    # Use a local variant that records Vm + all 4 Sundt gates at centre
    n = static["is_node"].shape[0]
    shape      = jnp.asarray(pm, dtype=jnp.float64)
    shape_prev = jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), shape[:-1]])
    from mrg_extracellular_coupled import _be_step

    def step_rec(carry, s):
        Vi, Vp, st = carry
        ve      = Ve_scaled * shape[s]
        ve_prev = Ve_scaled * shape_prev[s]
        g_eff, i_ion, st2 = mfn(Vi - Vp, st, DT)
        Vi2, Vp2 = _be_step(Vi, Vp, ve, ve_prev, g_eff, i_ion,
                            static["Cm_dt"], static["Cmy_dt"], static["gmy"],
                            static["Gi_diag"], static["Gp_diag"],
                            static["Up"], static["Low"], static["is_node"])
        M2, H2, N2, L2 = st2
        out = (Vi2[mid] - Vp2[mid], M2[mid], H2[mid], N2[mid], L2[mid])
        return (Vi2, Vp2, st2), out

    Vi0 = jnp.full(n, V_REST, dtype=jnp.float64)
    Vp0 = jnp.zeros(n, dtype=jnp.float64)
    (_, _, _), (vm_t, m_t, h_t, n_t, l_t) = jax.lax.scan(
        step_rec, (Vi0, Vp0, state0), jnp.arange(N_STEPS)
    )
    t_jax = (np.arange(N_STEPS) + 1) * DT

    print(f"[2/5] Extracellular: NEURON (amp={amp:.3f} mA) ...", flush=True)
    nr = run_extracellular_sundt(DIAM, N_NODES, src_height_um=1000., pw_ms=0.1,
                                 delay_ms=DELAY, amp_mA=float(amp), dt_ms=DT, tstop_ms=TSTOP)
    t_nrn  = nr.t_ms
    vm_nrn = nr.vm_mV[nr.probe_node_idx]

    peak_jax = float(np.max(np.asarray(vm_t)))
    peak_nrn = float(np.max(vm_nrn))
    print(f"  AP peak: JAX={peak_jax:.3f} mV  NEURON={peak_nrn:.3f} mV  |diff|={abs(peak_jax-peak_nrn):.3f} mV")
    return dict(
        t_jax=t_jax, vm_jax=np.asarray(vm_t),
        jax_gates={"m": np.asarray(m_t), "h": np.asarray(h_t),
                   "n": np.asarray(n_t),  "l": np.asarray(l_t)},
        t_nrn=t_nrn, vm_nrn=vm_nrn, nrn_gates=nr.gates,
        amp_mA=float(amp), threshold_mA=float(thr),
    )


# ── [3] Strength-duration curves ──────────────────────────────────────────────
def run_sd_curves(runner) -> dict:
    print("[3/5] Strength-duration curves ...", flush=True)
    results = {}
    for pw in SD_PWS:
        t0 = time.time()
        pm = _mono_shape(pw)
        jax_thr = _bisect(runner, pm)
        t_jax = time.time() - t0

        t0 = time.time()
        nrn_thr = find_threshold_extracellular_sundt(
            DIAM, N_NODES, pw_ms=pw, delay_ms=DELAY, dt_ms=DT, tstop_ms=TSTOP,
        )
        t_nrn = time.time() - t0

        err = (abs(jax_thr) - abs(nrn_thr)) / abs(nrn_thr) * 100
        results[pw] = {"jax": float(jax_thr), "neuron": float(nrn_thr), "err_pct": float(err)}
        print(f"  PW={pw:.2f} ms | NEURON={nrn_thr:.5f} | JAX={jax_thr:.5f} | "
              f"err={err:+.2f}% | t={t_jax:.1f}/{t_nrn:.1f} s", flush=True)

    with open(OUT / "data_sundt_threshold.json", "w") as f:
        json.dump({str(pw): v for pw, v in results.items()}, f, indent=2)
    return results


# ── [4] Conduction velocity ────────────────────────────────────────────────────
def run_cv(runner) -> dict:
    print("[4/5] Conduction velocity ...", flush=True)
    pm01 = _mono_shape(0.1)
    thr  = _bisect(runner, pm01)
    amp  = thr * 1.3

    static_cv, mfn_cv, state0_cv, Ve_unit_cv, mid_cv, centers_cv, geom_cv = \
        _make_jax_setup(N=N_CV)
    nodes_cv = node_indices(geom_cv)
    k_ctr = len(nodes_cv) // 2
    k_end = 4

    pm = jnp.asarray(pm01)
    Ve_scaled = jnp.asarray(Ve_unit_cv * (amp / -1.0))
    trace, _ = integrate(static_cv, mfn_cv, state0_cv, Ve_scaled, pm, DT,
                         v_rest=V_REST, record="all")
    Vi_all, Vp_all = trace
    Vm_all  = np.asarray(Vi_all) - np.asarray(Vp_all)
    t_steps = (np.arange(N_STEPS) + 1) * DT
    onset   = int(DELAY / DT)
    t_ctr   = t_steps[onset + int(np.argmax(Vm_all[onset:, nodes_cv[k_ctr]]))]
    t_end_k = t_steps[onset + int(np.argmax(Vm_all[onset:, nodes_cv[k_end]]))]
    dist_um = abs(float(centers_cv[nodes_cv[k_ctr]]) - float(centers_cv[nodes_cv[k_end]]))
    jax_cv  = dist_um / abs(t_end_k - t_ctr) * 1e-3

    nr = run_extracellular_sundt(DIAM, N_CV, src_height_um=1000., pw_ms=0.1,
                                 delay_ms=DELAY, amp_mA=float(amp), dt_ms=DT, tstop_ms=TSTOP)
    onset_idx = int(np.searchsorted(nr.t_ms, DELAY))
    t_nrn_ctr = nr.t_ms[onset_idx + int(np.argmax(nr.vm_mV[k_ctr, onset_idx:]))]
    t_nrn_end = nr.t_ms[onset_idx + int(np.argmax(nr.vm_mV[k_end, onset_idx:]))]
    nrn_cv    = dist_um / abs(t_nrn_end - t_nrn_ctr) * 1e-3

    err = (jax_cv - nrn_cv) / nrn_cv * 100
    print(f"  JAX CV={jax_cv:.3f} m/s  NEURON={nrn_cv:.3f} m/s  err={err:+.1f}%")
    result = {"jax": float(jax_cv), "neuron": float(nrn_cv), "err_pct": float(err),
              "amp_mA": float(amp), "threshold_mA": float(thr)}
    with open(OUT / "data_sundt_cv.json", "w") as fh:
        json.dump(result, fh, indent=2)
    return result


# ── [5] AP propagation check ──────────────────────────────────────────────────
def run_propagation(static, mfn, state0, Ve_unit, runner) -> dict:
    print("[5/5] AP propagation check ...", flush=True)
    pm01 = _mono_shape(0.1)
    thr  = _bisect(runner, pm01)
    amp  = thr * 1.3
    pm   = jnp.asarray(pm01)
    Ve_scaled = jnp.asarray(Ve_unit * (amp / -1.0))
    trace, _ = integrate(static, mfn, state0, Ve_scaled, pm, DT,
                         v_rest=V_REST, record="all")
    Vi_all, Vp_all = trace
    Vm_all = np.asarray(Vi_all) - np.asarray(Vp_all)
    nodes  = list(range(N_NODES))
    fired  = [bool(np.max(Vm_all[:, k]) > -30.0) for k in nodes]
    n_fired = sum(fired)
    print(f"  Nodes fired: {n_fired}/{N_NODES}  {'PASS ✓' if n_fired == N_NODES else 'FAIL ✗'}")
    return {"n_nodes": N_NODES, "n_fired": n_fired, "all_fired": n_fired == N_NODES,
            "amp_mA": float(amp)}


# ── figures ───────────────────────────────────────────────────────────────────
def make_traces_figure(intra: dict, extra: dict):
    """fig_sundt_traces.png — parallel to fig1_validation.png."""
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    # (A) Intracellular Vm
    ax = axs[0, 0]
    ax.plot(intra["t_nrn"], intra["vm_nrn"], color="C0", lw=2.0, label="PyFibers / NEURON")
    ax.plot(intra["t_jax"], intra["vm_jax"], color="C3", lw=1.0, ls="--", label="JAX (Jaxley intra)")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("V_m (mV)")
    ax.set_title(f"(A) Intracellular pulse — V_m  (|Δpeak|={abs(intra['peak_jax']-intra['peak_nrn']):.3f} mV)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # (B) Intracellular gates
    ax = axs[0, 1]
    for g, c in [("m", "C0"), ("h", "C2"), ("n", "C4"), ("l", "C6")]:
        ax.plot(intra["t_nrn"], intra["nrn_gates"][g], color=c, lw=2.0, label=f"NEURON {g}")
        ax.plot(intra["t_jax"], intra["jax_gates"][g], color=c, lw=1.0, ls="--", label=f"JAX {g}")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("gate value")
    ax.set_title("(B) Intracellular pulse — gates")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    # (C) Extracellular Vm
    ax = axs[1, 0]
    ax.plot(extra["t_nrn"], extra["vm_nrn"], color="C0", lw=2.0, label="PyFibers / NEURON")
    ax.plot(extra["t_jax"], extra["vm_jax"], color="C3", lw=1.0, ls="--", label="JAX (single-cable)")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("V_m (mV)")
    ax.set_title(f"(C) Extracellular pulse — V_m  ({extra['amp_mA']:.2f} mA, 1 mm, 0.1 ms, N={N_NODES})")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # (D) Extracellular gates
    ax = axs[1, 1]
    for g, c in [("m", "C0"), ("h", "C2"), ("n", "C4"), ("l", "C6")]:
        nrn_g = extra["nrn_gates"].get(g, extra["nrn_gates"].get(f"{g}"))
        if nrn_g is not None:
            ax.plot(extra["t_nrn"], nrn_g, color=c, lw=2.0, label=f"NEURON {g}")
        ax.plot(extra["t_jax"], extra["jax_gates"][g], color=c, lw=1.0, ls="--", label=f"JAX {g}")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("gate value")
    ax.set_title("(D) Extracellular pulse — gates")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    fig.suptitle(
        f"Sundt C-fiber D=0.8 µm — PyFibers/NEURON (solid) vs JAX (dashed)\n"
        f"A/B: channel translation (intra stim).  C/D: single-cable extracellular solver.",
        fontsize=10,
    )
    fig.savefig(OUT / "fig_sundt_traces.png", dpi=150, bbox_inches="tight")
    print(f"  → {OUT / 'fig_sundt_traces.png'}")


def make_validation_figure(sd: dict, cv: dict, prop: dict):
    """fig_sundt_validation.png — parallel to fig_mrg_validation.png."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # (A) Strength-duration curves (log-log)
    ax = axs[0, 0]
    pws   = sorted(sd.keys())
    nrn_t = [abs(sd[pw]["neuron"]) for pw in pws]
    jax_t = [abs(sd[pw]["jax"])    for pw in pws]
    ax.plot(pws, nrn_t, "o-",  color="C0", lw=2.0, ms=5, label="PyFibers / NEURON")
    ax.plot(pws, jax_t, "s--", color="C3", lw=1.5, ms=4, label="JAX single-cable")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Pulse width (ms)"); ax.set_ylabel("|Threshold| (mA)")
    ax.set_title("(A) Strength-duration (D=0.8 µm, 1 mm, σ=0.3 S/m)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which="both")
    ax.xaxis.set_minor_locator(ticker.LogLocator(subs="all", numticks=10))

    # (B) Conduction velocity comparison
    ax = axs[0, 1]
    labels = ["PyFibers / NEURON", "JAX single-cable"]
    vals   = [cv["neuron"], cv["jax"]]
    bars   = ax.bar(labels, vals, color=["C0", "C3"], alpha=0.8, width=0.4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f"{v:.3f} m/s", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Conduction velocity (m/s)")
    ax.set_title(f"(B) Conduction velocity (D=0.8 µm, err={cv['err_pct']:+.1f}%)")
    ax.set_ylim(0, max(vals) * 1.2); ax.grid(axis="y", alpha=0.3)

    # (C) AP propagation check
    ax = axs[1, 0]
    fired_arr = [1 if i < prop["n_fired"] else 0 for i in range(N_NODES)]
    ax.bar(range(N_NODES), fired_arr, color="C2", alpha=0.8)
    ax.set_xlabel("Compartment index"); ax.set_ylabel("Fired (1=yes)")
    ax.set_title(f"(C) AP propagation: {prop['n_fired']}/{N_NODES} compartments fired at 1.3× thr  "
                 f"({'PASS ✓' if prop['all_fired'] else 'FAIL ✗'})")
    ax.set_ylim(0, 1.3); ax.grid(axis="y", alpha=0.3)

    # (D) Error summary
    ax = axs[1, 1]
    sd_errs = [abs(sd[pw]["err_pct"]) for pw in pws]
    sd_lbls = [f"SD\n{pw:.2f}ms" for pw in pws]
    all_lbls = sd_lbls + ["CV"]
    all_errs = sd_errs + [abs(cv["err_pct"])]
    ax.bar(np.arange(len(all_lbls)), all_errs,
           color=["C0"] * len(sd_lbls) + ["C1"], alpha=0.8)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="0.5 % guideline")
    ax.set_xticks(np.arange(len(all_lbls))); ax.set_xticklabels(all_lbls, fontsize=8)
    ax.set_ylabel("Mean |error| (%)"); ax.set_title("(D) Error summary")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Sundt C-fiber validation: JAX single-cable vs PyFibers/NEURON\n"
        f"D=0.8 µm, {N_NODES} compartments, δz={DELTA_Z} µm, T=37°C, σ=0.3 S/m",
        fontsize=10,
    )
    fig.savefig(OUT / "fig_sundt_validation.png", dpi=150, bbox_inches="tight")
    print(f"  → {OUT / 'fig_sundt_validation.png'}")


# ── CSV ────────────────────────────────────────────────────────────────────────
def write_csv(sd: dict, cv: dict):
    path = OUT / "table_sundt_validation.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "pw_ms", "neuron_val", "jax_val", "abs_err_pct"])
        for pw in sorted(sd.keys()):
            v = sd[pw]
            w.writerow(["SD", pw, f"{v['neuron']:.5f}", f"{v['jax']:.5f}",
                        f"{abs(v['err_pct']):.3f}"])
        w.writerow(["CV", 0.1, f"{cv['neuron']:.4f}", f"{cv['jax']:.4f}",
                    f"{abs(cv['err_pct']):.2f}"])
    print(f"  → {path}")


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t_total = time.time()
    static, mfn, state0, Ve_unit, mid, centers, geom = _make_jax_setup()
    runner = _make_jit_runner(static, mfn, state0, Ve_unit)

    intra_data = run_intra_traces()
    extra_data = run_extra_traces(static, mfn, state0, Ve_unit, mid)
    sd_data    = run_sd_curves(runner)
    cv_data    = run_cv(runner)
    prop_data  = run_propagation(static, mfn, state0, Ve_unit, runner)

    print("\n=== Generating figures ===", flush=True)
    make_traces_figure(intra_data, extra_data)
    make_validation_figure(sd_data, cv_data, prop_data)
    write_csv(sd_data, cv_data)
    print(f"\nDone. Total wall time: {(time.time()-t_total)/60:.1f} min")
