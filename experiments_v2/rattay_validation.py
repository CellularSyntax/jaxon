"""Rattay C-fiber validation suite — v2.

Validates the JAX coupled (Vi, Vpax) solver for the Rattay (1993) HH model
against PyFibers/NEURON across all fibre diameters, pulse widths, and pulse shapes.

Tasks
-----
1. Vm + gate traces (intracellular & extracellular) at D=0.8 µm
2. Strength-duration curves — all 5 diameters × 8 pulse shapes × 6 PWs
3. Conduction velocity — all 5 diameters (self-contained)
4. Figures: traces, SD curves, error analysis

Outputs (outputs/rattay_validation/)
---------------------------------------
  data_rattay_traces.json
  data_rattay_sd.json
  data_rattay_cv.json
  fig_rattay_traces.png
  fig_rattay_sd_curves.png
  fig_rattay_analysis.png

Run from project root:
    python experiments_v2/rattay_validation.py
"""

from __future__ import annotations

import sys, pathlib, time, dataclasses
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import jax
import jax.numpy as jnp
import jaxley as jx
from jaxley.solver_gate import solve_gate_exponential

jax.config.update("jax_enable_x64", True)

from jaxfibers.fibers.rattay import (
    build_rattay, node_indices, section_centers_um,
    V_REST, CM,
)
from jaxfibers.channels.rattay_channels import RattayHH
from jaxfibers.stim.extracellular import point_source_potentials_mV
from jaxfibers.stim.intracellular import rectangular_pulse, attach_intra_pulse
from jaxfibers.nrn_baseline import run_intracellular_rattay, run_extracellular_rattay
from jaxfibers.stim.extracellular_coupled import arrays_from_geometry, integrate, _be_step

from experiments_v2.utils import (
    PULSES, make_pulse_array, pf_find_threshold,
    jax_bisect, ensure_dir, save_json, GROUP_COLORS,
)

OUT = ensure_dir(ROOT / "outputs" / "rattay_validation")

# ── shared constants ──────────────────────────────────────────────────────────
CELSIUS  = 37.0
DT       = 0.005    # ms
TSTOP    = 12.0     # ms  (C-fiber APs slower; accommodates 2×pw_max=2 ms + buffer)
DELAY    = 1.0      # ms
N_NODES  = 51
N_STEPS  = int(TSTOP / DT)
SIGMA    = 0.3      # S/m
SRC_H    = 1000.0   # µm  (1 mm above fibre centre)

# RattayHH default channel parameters
GNABAR = 0.12     # S/cm²
GKBAR  = 0.036    # S/cm²
GL     = 3e-4     # S/cm²
EL     = -59.4    # mV
ENA    = 45.0     # mV
EK     = -82.0    # mV

DIAMETERS  = [0.3, 0.5, 0.8, 1.0, 1.5]           # µm
SD_PWS     = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]     # ms
PULSE_KEYS = list(PULSES.keys())

TRACE_DIAM   = 0.8   # µm
INTRA_AMP_NA = 0.5
INTRA_PW_MS  = 0.2
EXTRA_PW_MS  = 0.2


# ── JAX geometry + coupled-solver setup ───────────────────────────────────────

def _make_jax_setup(D: float):
    """Build coupled-solver static arrays and membrane function for diameter D."""
    _, geom = build_rattay(diameter=D, n_nodes=N_NODES)
    nodes   = node_indices(geom)
    centers = np.array(section_centers_um(geom))
    mid     = nodes[len(nodes) // 2]
    n_comp  = geom.n_comp

    geom_c = dataclasses.replace(geom, cm_uF_cm2=[CM] * n_comp)
    static  = arrays_from_geometry(geom_c, DT)

    Ve_unit = np.array(point_source_potentials_mV(
        list(centers), src_x_um=0., src_y_um=SRC_H,
        src_z_um=float(centers[mid]), i0_mA=-1.0,
    ))

    A_in = static["A_in_cm2"]

    # Steady-state gate values at V_REST
    v0 = jnp.float64(V_REST)
    (am0, bm0), (ah0, bh0), (an0, bn0) = RattayHH._alpha_beta(v0, CELSIUS)
    m0 = float(am0 / (am0 + bm0))
    h0 = float(ah0 / (ah0 + bh0))
    n0 = float(an0 / (an0 + bn0))
    state0 = (
        jnp.full(n_comp, m0, dtype=jnp.float64),
        jnp.full(n_comp, h0, dtype=jnp.float64),
        jnp.full(n_comp, n0, dtype=jnp.float64),
    )

    def membrane_fn(Vm, state, dt):
        M, H, N = state
        (am, bm), (ah, bh), (an, bn) = RattayHH._alpha_beta(Vm, CELSIUS)
        M2 = solve_gate_exponential(M, dt, am, bm)
        H2 = solve_gate_exponential(H, dt, ah, bh)
        N2 = solve_gate_exponential(N, dt, an, bn)
        g_na  = GNABAR * M2**3 * H2
        g_k   = GKBAR  * N2**4
        g_tot = (g_na + g_k + GL) * A_in * 1e6
        i_ion = (g_na * (Vm - ENA) + g_k * (Vm - EK) + GL * (Vm - EL)) * A_in * 1e6
        return g_tot, i_ion, (M2, H2, N2)

    return static, membrane_fn, state0, Ve_unit, mid, centers, nodes, geom


def _make_jit_runner(static, mfn, state0, Ve_unit):
    """Compile a JIT runner: (amp_mA, pulse_shape) -> peak Vm over all nodes.

    record='all' is required for correct anodic threshold detection: the AP
    originates at virtual-cathode peripheral nodes (not the centre), so
    recording only the centre gives wrong thresholds at short pulse widths.
    """
    Ve_j    = jnp.asarray(Ve_unit, dtype=jnp.float64)
    is_node = static["is_node"]

    @jax.jit
    def run(amp_mA, pulse_shape):
        Ve = Ve_j * (amp_mA / -1.0)
        (Vi_all, Vp_all), _ = integrate(static, mfn, state0, Ve, pulse_shape, DT,
                                        v_rest=V_REST, record="all")
        Vm_all   = Vi_all - Vp_all
        Vm_nodes = jnp.where(is_node[None, :], Vm_all, -jnp.inf)
        return jnp.max(Vm_nodes)

    return run


def _make_jit_all_runner(static, mfn, state0, Ve_unit):
    """Compile a JIT runner: (amp_mA, pulse_shape) -> (Vi[nsteps,n], Vp[nsteps,n])."""
    Ve_j = jnp.asarray(Ve_unit, dtype=jnp.float64)

    @jax.jit
    def run(amp_mA, pulse_shape):
        Ve = Ve_j * (amp_mA / -1.0)
        trace, _ = integrate(static, mfn, state0, Ve, pulse_shape, DT,
                             v_rest=V_REST, record="all")
        return trace

    return run


# ── Task 1: Vm + gate traces ──────────────────────────────────────────────────

def task_traces() -> dict:
    """Intracellular and extracellular Vm + gate traces at D=0.8 µm."""
    print("\n=== Task 1: Vm + gate traces (D=0.8 µm) ===")
    D = TRACE_DIAM

    # ── Intracellular (Jaxley bwd_euler) ─────────────────────────────────────
    print("  [intra] JAX ...", flush=True)
    cell_jax, geom_jax = build_rattay(diameter=D, n_nodes=N_NODES)
    nodes_jax = node_indices(geom_jax)
    mid_comp  = nodes_jax[len(nodes_jax) // 2]
    n_steps_i = int(10.0 / DT) + 1
    t_intra   = np.arange(n_steps_i) * DT
    pulse_i   = rectangular_pulse(t_intra, DELAY, INTRA_PW_MS, INTRA_AMP_NA)
    attach_intra_pulse(cell_jax, mid_comp, pulse_i)
    cell_jax.branch(0).comp(mid_comp).record("v")
    for g in ["RattayHH_m", "RattayHH_h", "RattayHH_n"]:
        cell_jax.branch(0).comp(mid_comp).record(g)
    rec_i = np.asarray(jx.integrate(cell_jax, delta_t=DT, t_max=10.0, solver="bwd_euler"))
    jax_vm_i    = rec_i[0]
    jax_gates_i = {"m": rec_i[1], "h": rec_i[2], "n": rec_i[3]}

    print("  [intra] NEURON ...", flush=True)
    nr_i = run_intracellular_rattay(
        diameter=D, n_nodes=N_NODES,
        i_amp_nA=INTRA_AMP_NA, i_dur_ms=INTRA_PW_MS, i_delay_ms=DELAY,
        dt_ms=DT, tstop_ms=10.0,
    )

    # ── Extracellular (coupled solver) ────────────────────────────────────────
    print("  [extra] building JAX setup ...", flush=True)
    static, mfn, state0, Ve_unit, mid, centers, nodes, geom = _make_jax_setup(D)
    runner = _make_jit_runner(static, mfn, state0, Ve_unit)
    pm_extra = jnp.asarray(make_pulse_array("mono_c", EXTRA_PW_MS, N_STEPS, DT, DELAY))
    thr_jax  = jax_bisect(runner, np.asarray(pm_extra),
                          lo=PULSES["mono_c"].lo, hi=PULSES["mono_c"].hi)
    amp_extra = thr_jax * 1.3

    print(f"  [extra] JAX (amp={amp_extra:.3f} mA) ...", flush=True)
    Ve_scaled  = jnp.asarray(Ve_unit * (amp_extra / -1.0))
    trace_e, _ = integrate(static, mfn, state0, Ve_scaled, pm_extra, DT,
                           v_rest=V_REST, record="center")
    t_extra_jax = (np.arange(N_STEPS) + 1) * DT
    jax_vm_e    = np.asarray(trace_e)

    print("  [extra] recording gate states (JAX scan) ...", flush=True)
    n_comp_e  = static["is_node"].shape[0]
    is_node_e = static["is_node"]
    shape_g      = jnp.asarray(pm_extra, dtype=jnp.float64)
    shape_prev_g = jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), shape_g[:-1]])

    @jax.jit
    def _extra_gates_scan(Vi0, Vp0, M0, H0, N0):
        def _step(carry, s):
            Vi, Vp, M, H, N = carry
            ve      = Ve_scaled * shape_g[s]
            ve_prev = Ve_scaled * shape_prev_g[s]
            g_eff, i_ion, (M2, H2, N2) = mfn(Vi - Vp, (M, H, N), DT)
            Vi2, Vp2 = _be_step(
                Vi, Vp, ve, ve_prev, g_eff, i_ion,
                static["Cm_dt"], static["Cmy_dt"], static["gmy"],
                static["Gi_diag"], static["Gp_diag"],
                static["Up"], static["Low"], is_node_e,
            )
            return (Vi2, Vp2, M2, H2, N2), (M2[mid], H2[mid], N2[mid])

        _, (m_t, h_t, n_t) = jax.lax.scan(
            _step,
            (Vi0, Vp0, M0, H0, N0),
            jnp.arange(N_STEPS),
        )
        return m_t, h_t, n_t

    M0, H0, N0 = state0
    Vi0_e = jnp.full(n_comp_e, V_REST, dtype=jnp.float64)
    Vp0_e = jnp.zeros(n_comp_e, dtype=jnp.float64)
    m_e, h_e, n_e = [np.array(x) for x in
                     _extra_gates_scan(Vi0_e, Vp0_e, M0, H0, N0)]
    jax_gates_e = {"m": m_e, "h": h_e, "n": n_e}

    print(f"  [extra] NEURON (amp={amp_extra:.3f} mA) ...", flush=True)
    nr_e     = run_extracellular_rattay(
        diameter=D, n_nodes=N_NODES, src_height_um=SRC_H,
        pw_ms=EXTRA_PW_MS, delay_ms=DELAY,
        amp_mA=float(amp_extra), dt_ms=DT, tstop_ms=TSTOP,
    )
    nrn_vm_e = nr_e.vm_mV[nr_e.probe_node_idx]

    data = dict(
        t_intra_jax=t_intra, vm_intra_jax=jax_vm_i,
        gates_intra_jax=jax_gates_i,
        t_intra_nrn=nr_i.t_ms, vm_intra_nrn=nr_i.vm_mV[nr_i.probe_node_idx],
        gates_intra_nrn=nr_i.gates,
        t_extra_jax=t_extra_jax, vm_extra_jax=jax_vm_e,
        gates_extra_jax=jax_gates_e,
        t_extra_nrn=nr_e.t_ms, vm_extra_nrn=nrn_vm_e,
        gates_extra_nrn=nr_e.gates,
        amp_extra_mA=float(amp_extra), threshold_extra_mA=float(thr_jax),
        diameter=D,
    )
    save_json(data, OUT / "data_rattay_traces.json")
    return data


# ── Task 2: Strength-duration curves ─────────────────────────────────────────

def task_sd_curves() -> dict:
    """Threshold vs pulse width for all diameters, all pulse shapes."""
    print("\n=== Task 2: Strength-duration curves ===")
    results: dict = {}

    for di, D in enumerate(DIAMETERS):
        print(f"\n  D = {D} µm", flush=True)
        static, mfn, state0, Ve_unit, *_ = _make_jax_setup(D)
        runner = _make_jit_runner(static, mfn, state0, Ve_unit)

        print(f"    [warmup] triggering JIT compile ...", flush=True)
        _arr_w = jnp.asarray(make_pulse_array("mono_c", SD_PWS[0], N_STEPS, DT, DELAY),
                              dtype=jnp.float64)
        t_warm = time.perf_counter()
        jax.block_until_ready(runner(jnp.float64(-1.0), _arr_w))
        print(f"    [warmup] done in {time.perf_counter() - t_warm:.2f} s", flush=True)

        results[D] = {}

        for pi, pk in enumerate(PULSE_KEYS):
            spec = PULSES[pk]
            results[D][pk] = {}
            for pwi, pw in enumerate(SD_PWS):
                _debug = (di == 0 and pi == 0 and pwi < 2)
                arr  = make_pulse_array(pk, pw, N_STEPS, DT, DELAY)
                t0   = time.time()
                try:
                    jax_thr = jax_bisect(runner, arr, lo=spec.lo, hi=spec.hi,
                                         debug=_debug)
                except RuntimeError as e:
                    print(f"    [{pk}] PW={pw} JAX bisect failed: {e}")
                    jax_thr = float("nan")
                t_jax = time.time() - t0

                tstop_pf = max(TSTOP, DELAY + (2 if "bi" in pk else 1) * pw + 4.0)
                t0 = time.time()
                try:
                    pf_thr = pf_find_threshold(
                        "rattay", D, N_NODES, arr, DT, tstop_pf,
                        lo=spec.lo, hi=spec.hi,
                        src_height_um=SRC_H, sigma_S_m=SIGMA,
                    )
                except RuntimeError as e:
                    print(f"    [{pk}] PW={pw} PF failed: {e}")
                    pf_thr = float("nan")
                t_pf = time.time() - t0

                err = (abs(jax_thr) - abs(pf_thr)) / abs(pf_thr) * 100 if not (
                    np.isnan(jax_thr) or np.isnan(pf_thr) or pf_thr == 0
                ) else float("nan")

                results[D][pk][pw] = {
                    "jax": float(jax_thr), "pyfibers": float(pf_thr),
                    "err_pct": float(err),
                }
                print(f"    [{pk}] PW={pw:.2f} ms | PF={pf_thr:.5f} | JAX={jax_thr:.5f} | "
                      f"err={err:+.2f}% | t_jax={t_jax:.1f} s  t_pf={t_pf:.1f} s", flush=True)

    save_json(results, OUT / "data_rattay_sd.json")
    return results


# ── Task 3: Conduction velocity ───────────────────────────────────────────────

def _jax_cv(run_all, amp_mA, nodes, centers) -> float:
    pm = jnp.asarray(make_pulse_array("mono_c", 0.2, N_STEPS, DT, DELAY))
    Vi_all, Vp_all = run_all(jnp.float64(amp_mA), pm)
    Vm_all  = np.asarray(Vi_all) - np.asarray(Vp_all)
    t_steps = (np.arange(N_STEPS) + 1) * DT
    onset   = int(DELAY / DT)
    k_ctr   = len(nodes) // 2
    k_end   = 2
    t_ctr   = t_steps[onset + int(np.argmax(Vm_all[onset:, nodes[k_ctr]]))]
    t_end   = t_steps[onset + int(np.argmax(Vm_all[onset:, nodes[k_end]]))]
    dist_um = abs(float(centers[nodes[k_ctr]]) - float(centers[nodes[k_end]]))
    return dist_um / abs(t_end - t_ctr) * 1e-3   # µm/ms → m/s


def _pf_cv(nr, n_nodes, centers, nodes) -> float:
    onset = int(np.searchsorted(nr.t_ms, DELAY))
    k_ctr, k_end = n_nodes // 2, 2
    t_ctr = nr.t_ms[onset + int(np.argmax(nr.vm_mV[k_ctr, onset:]))]
    t_end = nr.t_ms[onset + int(np.argmax(nr.vm_mV[k_end, onset:]))]
    dist  = abs(float(centers[nodes[k_ctr]]) - float(centers[nodes[k_end]]))
    return dist / abs(t_end - t_ctr) * 1e-3


def task_cv() -> dict:
    """Conduction velocity for all Rattay diameters (self-contained)."""
    print("\n=== Task 3: Conduction velocity ===")
    results: dict = {}

    for D in DIAMETERS:
        static, mfn, state0, Ve_unit, mid, centers, nodes, geom = _make_jax_setup(D)
        runner     = _make_jit_runner(static, mfn, state0, Ve_unit)
        runner_all = _make_jit_all_runner(static, mfn, state0, Ve_unit)

        arr = make_pulse_array("mono_c", 0.2, N_STEPS, DT, DELAY)
        thr = jax_bisect(runner, arr, lo=PULSES["mono_c"].lo, hi=PULSES["mono_c"].hi)
        amp = thr * 1.3

        t0     = time.time()
        jax_cv = _jax_cv(runner_all, amp, nodes, centers)
        t_jax  = time.time() - t0

        t0 = time.time()
        nr = run_extracellular_rattay(
            D, N_NODES, src_height_um=SRC_H, pw_ms=0.2,
            delay_ms=DELAY, amp_mA=float(amp), dt_ms=DT, tstop_ms=TSTOP,
        )
        pf_cv = _pf_cv(nr, N_NODES, centers, nodes)
        t_pf  = time.time() - t0

        err = (jax_cv - pf_cv) / pf_cv * 100
        results[D] = {"jax": float(jax_cv), "pyfibers": float(pf_cv),
                      "err_pct": float(err), "threshold_mA": float(thr)}
        print(f"  D={D:.1f} µm | PF={pf_cv:.2f} | JAX={jax_cv:.2f} m/s | "
              f"err={err:+.1f}% | t_jax={t_jax:.1f} s  t_pf={t_pf:.1f} s", flush=True)

    save_json(results, OUT / "data_rattay_cv.json")
    return results


# ── Figure 1: Traces ──────────────────────────────────────────────────────────

def fig_traces(data: dict) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    ax = axs[0, 0]
    ax.plot(data["t_intra_nrn"], data["vm_intra_nrn"], color="C0", lw=2.0, label="PyFibers/NEURON")
    ax.plot(data["t_intra_jax"], data["vm_intra_jax"], color="C3", lw=1.0, ls="--", label="JAX (Jaxley intra)")
    diff = abs(max(data["vm_intra_jax"]) - max(data["vm_intra_nrn"]))
    ax.set_title(f"(A) Intracellular V_m  |Δpeak|={diff:.3f} mV")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("V_m (mV)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axs[0, 1]
    gate_colors = {"m": "C0", "h": "C2", "n": "C4"}
    for g, c in gate_colors.items():
        nrn_g = data["gates_intra_nrn"].get(g, [])
        if len(nrn_g):
            ax.plot(data["t_intra_nrn"], nrn_g, color=c, lw=2.0, label=f"NEURON {g}")
        jg = data["gates_intra_jax"].get(g, [])
        if len(jg):
            ax.plot(data["t_intra_jax"], jg, color=c, lw=1.0, ls="--", label=f"JAX {g}")
    ax.set_title("(B) Intracellular gates  (m, h: Na;  n: K)")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("gate value")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    ax = axs[1, 0]
    ax.plot(data["t_extra_nrn"], data["vm_extra_nrn"], color="C0", lw=2.0, label="PyFibers/NEURON")
    ax.plot(data["t_extra_jax"], data["vm_extra_jax"], color="C3", lw=1.0, ls="--", label="JAX (coupled solver)")
    ax.set_title(f"(C) Extracellular V_m  ({data['amp_extra_mA']:.2f} mA, 1 mm)")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("V_m (mV)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axs[1, 1]
    gate_colors = {"m": "C0", "h": "C2", "n": "C4"}
    gates_e_nrn = data.get("gates_extra_nrn", {})
    gates_e_jax = data.get("gates_extra_jax", {})
    for g, c in gate_colors.items():
        nrn_g = gates_e_nrn.get(g, [])
        if len(nrn_g):
            ax.plot(data["t_extra_nrn"], nrn_g, color=c, lw=2.0, label=f"NEURON {g}")
        jax_g = gates_e_jax.get(g, [])
        if len(jax_g):
            ax.plot(data["t_extra_jax"], jax_g, color=c, lw=1.2, ls="--", label=f"JAX {g}")
    ax.set_title("(D) Extracellular gates  (NEURON solid, JAX dashed)")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("gate value")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

    fig.suptitle(f"Rattay D={data['diameter']} µm — PyFibers/NEURON vs JAX", fontsize=11)
    path = OUT / "fig_rattay_traces.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → {path}")


# ── Figure 2: SD curves ───────────────────────────────────────────────────────

def fig_sd_curves(sd: dict) -> None:
    n_cols = 4
    n_rows = (len(PULSE_KEYS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                             constrained_layout=True)
    axes = axes.flatten()
    cmap  = plt.get_cmap("tab10")
    diams = sorted(sd.keys())

    for idx, pk in enumerate(PULSE_KEYS):
        ax = axes[idx]
        for di, D in enumerate(diams):
            if pk not in sd[D]:
                continue
            pws   = sorted(sd[D][pk].keys())
            pf_t  = [abs(sd[D][pk][pw]["pyfibers"]) for pw in pws]
            jax_t = [abs(sd[D][pk][pw]["jax"])      for pw in pws]
            c = cmap(di / max(len(diams) - 1, 1))
            ax.loglog(pws, pf_t,  "o-",  color=c, lw=1.5, ms=4, label=f"{D} µm")
            ax.loglog(pws, jax_t, "s--", color=c, lw=1.0, ms=3)
        ax.set_title(PULSES[pk].name, fontsize=9)
        ax.set_xlabel("PW (ms)", fontsize=8); ax.set_ylabel("|Thr| (mA)", fontsize=8)
        ax.grid(True, alpha=0.3, which="both")
        if idx == 0:
            import matplotlib.lines as mlines
            diam_leg = ax.legend(fontsize=6, ncol=2, loc="upper right")
            ax.add_artist(diam_leg)
            h_pf  = mlines.Line2D([], [], color="k", ls="-",  marker="o", ms=4, lw=1.5, label="PyFibers")
            h_jax = mlines.Line2D([], [], color="k", ls="--", marker="s", ms=3, lw=1.0, label="JAX")
            ax.legend(handles=[h_pf, h_jax], fontsize=7, loc="lower left")

    for ax in axes[len(PULSE_KEYS):]:
        ax.set_axis_off()

    fig.suptitle("Rattay SD curves — PyFibers solid, JAX dashed (all 5 diameters)", fontsize=11)
    path = OUT / "fig_rattay_sd_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → {path}")


# ── Figure 3: Analysis ────────────────────────────────────────────────────────

def fig_analysis(sd: dict, cv: dict) -> None:
    jax_vals, pf_vals, errs, groups = [], [], [], []
    shape_errs = {pk: [] for pk in PULSE_KEYS}

    for D in sorted(sd.keys()):
        for pk in PULSE_KEYS:
            if pk not in sd[D]:
                continue
            for pw, v in sd[D][pk].items():
                if np.isnan(v["jax"]) or np.isnan(v["pyfibers"]) or v["pyfibers"] == 0:
                    continue
                jax_vals.append(abs(v["jax"]))
                pf_vals.append(abs(v["pyfibers"]))
                e = abs(v["err_pct"])
                errs.append(e)
                groups.append(PULSES[pk].group)
                shape_errs[pk].append(e)

    jax_vals = np.array(jax_vals)
    pf_vals  = np.array(pf_vals)
    errs     = np.array(errs)
    slope, intercept, r, *_ = stats.linregress(pf_vals, jax_vals)
    r2 = r ** 2

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    ax = axes[0, 0]
    group_list = np.array(groups)
    for grp, c in GROUP_COLORS.items():
        mask = group_list == grp
        ax.scatter(pf_vals[mask], jax_vals[mask], c=c, s=15, alpha=0.6, label=grp)
    x_line = np.linspace(pf_vals.min(), pf_vals.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "k--", lw=1.0, label=f"fit (R²={r2:.4f})")
    ax.plot(x_line, x_line, "gray", lw=0.8, ls=":", label="identity")
    ax.set_xlabel("|Threshold| PyFibers (mA)")
    ax.set_ylabel("|Threshold| JAX (mA)")
    ax.set_title(f"(A) Correlation  R²={r2:.4f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    diams_sorted = sorted(sd.keys())
    ape_per_diam = []
    for D in diams_sorted:
        vals = []
        if "mono_c" in sd[D]:
            for pw, v in sd[D]["mono_c"].items():
                if not np.isnan(v["err_pct"]):
                    vals.append(abs(v["err_pct"]))
        ape_per_diam.append(np.mean(vals) if vals else float("nan"))
    ax.plot(diams_sorted, ape_per_diam, "o-", color="C0", lw=2.0, ms=6)
    ax.set_xlabel("Fiber diameter (µm)")
    ax.set_ylabel("Mean |error| %  (mono_c)")
    ax.set_title("(B) APE vs diameter")
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    pk_means   = [np.mean(shape_errs[pk]) if shape_errs[pk] else float("nan") for pk in PULSE_KEYS]
    pk_labels  = [PULSES[pk].name for pk in PULSE_KEYS]
    bar_colors = [GROUP_COLORS[PULSES[pk].group] for pk in PULSE_KEYS]
    ax.bar(range(len(PULSE_KEYS)), pk_means, color=bar_colors, alpha=0.8)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(range(len(PULSE_KEYS)))
    ax.set_xticklabels(pk_labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Mean |error| %")
    ax.set_title("(C) Error vs stimulus type")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 0]
    group_data = {g: [] for g in GROUP_COLORS}
    for e, g in zip(errs, groups):
        group_data[g].append(e)
    vp = ax.violinplot([group_data[g] for g in GROUP_COLORS],
                       positions=range(len(GROUP_COLORS)), showmedians=True)
    for pc, c in zip(vp["bodies"], GROUP_COLORS.values()):
        pc.set_facecolor(c); pc.set_alpha(0.7)
    ax.set_xticks(range(len(GROUP_COLORS)))
    ax.set_xticklabels(list(GROUP_COLORS.keys()))
    ax.set_ylabel("|Error| %")
    ax.set_title("(D) Error distribution by pulse group")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    cv_diams = sorted(cv.keys())
    pf_cv    = [cv[d]["pyfibers"] for d in cv_diams]
    jax_cv   = [cv[d]["jax"]      for d in cv_diams]
    ax.plot(cv_diams, pf_cv,  "o-",  color="C0", lw=2.0, ms=5, label="PyFibers/NEURON")
    ax.plot(cv_diams, jax_cv, "s--", color="C3", lw=1.5, ms=4, label="JAX coupled solver")
    ax.set_xlabel("Fiber diameter (µm)"); ax.set_ylabel("CV (m/s)")
    ax.set_title("(E) Conduction velocity vs diameter")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    cv_errs = [cv[d]["err_pct"] for d in cv_diams]
    ax.plot(cv_diams, cv_errs, "o-", color="C3", lw=2.0, ms=6)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Fiber diameter (µm)"); ax.set_ylabel("CV error (%)")
    ax.set_title("(F) CV error vs diameter")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Rattay validation analysis — JAX coupled solver vs PyFibers/NEURON\n"
        f"N={N_NODES} nodes, 1 mm point source, σ={SIGMA} S/m, T={CELSIUS}°C",
        fontsize=11,
    )
    path = OUT / "fig_rattay_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → {path}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cpu_device = jax.devices("cpu")[0]
    print(f"JAX devices available : {jax.devices()}")
    print(f"Running JAX tasks on  : {cpu_device}")
    t0_total = time.time()

    with jax.default_device(cpu_device):
        traces_data = task_traces()
        sd_data     = task_sd_curves()
        cv_data     = task_cv()

    print("\n=== Generating figures ===", flush=True)
    fig_traces(traces_data)
    fig_sd_curves(sd_data)
    fig_analysis(sd_data, cv_data)

    print(f"\nDone. Total wall time: {(time.time() - t0_total) / 60:.1f} min")
