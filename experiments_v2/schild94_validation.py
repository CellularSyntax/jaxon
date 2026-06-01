"""Schild 1994 C-fiber validation suite — v2.

Validates the JAX coupled (Vi, Vpax) solver for the Schild 1994 C-fiber model
against PyFibers/NEURON across all fibre diameters, pulse widths, and pulse shapes.

Tasks
-----
1. Vm + key gate + Ca traces (intracellular & extracellular) at D=0.8 µm
2. Strength-duration curves — 3 diameters × 8 pulse shapes × 6 PWs
3. Conduction velocity — 3 diameters (JAX + PyFibers)
4. Figures: traces, SD curves, error analysis

Outputs (outputs/schild94_validation/)
---------------------------------------
  data_schild94_traces.json
  data_schild94_sd.json
  data_schild94_cv.json
  fig_schild94_traces.png
  fig_schild94_sd_curves.png
  fig_schild94_analysis.png

Run from project root:
    python experiments_v2/schild94_validation.py
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

jax.config.update("jax_enable_x64", True)

from jaxfibers.fibers.schild import (
    build_schild94, node_indices, section_centers_um,
    V_REST, CM, DELTA_Z, FHSPACE,
)
from jaxfibers.channels.schild_channels import (
    SchildCombined94,
    _kd_kinetics, _ka_kinetics, _kds_kinetics, _kca_kinetics,
    _can_kinetics, _cat_kinetics,
    _reversal_potentials,
    _nakpump_current, _capump_current, _nacapump_current,
    _ca_ode_step, _update_gate,
    ca_geometry,
)
from jaxfibers.stim.extracellular import point_source_potentials_mV
from jaxfibers.stim.intracellular import rectangular_pulse, attach_intra_pulse
from jaxfibers.nrn_baseline import (
    run_intracellular_schild94,
    run_extracellular_schild94,
)
from jaxfibers.stim.extracellular_coupled import arrays_from_geometry, integrate

from experiments_v2.utils import (
    PULSES, make_pulse_array, pf_find_threshold,
    jax_bisect, ensure_dir, save_json, GROUP_COLORS,
)

OUT = ensure_dir(ROOT / "outputs" / "schild94_validation")

# ── shared constants ──────────────────────────────────────────────────────────
CELSIUS  = 37.0
DT       = 0.005    # ms
TSTOP    = 15.0     # ms  (Schild C-fiber can have wide APs)
DELAY    = 1.0      # ms
N_NODES  = 51
N_STEPS  = int(TSTOP / DT)
SIGMA    = 0.3      # S/m
SRC_H    = 1000.0   # µm  (1 mm above fibre centre)

DIAMETERS  = [0.3, 0.5, 0.8]              # µm — Schild 1994 C-fiber range
SD_PWS     = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]  # ms
PULSE_KEYS = list(PULSES.keys())

TRACE_DIAM   = 0.8   # µm
INTRA_AMP_NA = 0.5
INTRA_PW_MS  = 0.5
EXTRA_PW_MS  = 0.2

# ── Schild94 channel parameters (SchildCombined94 defaults) ──────────────────
_P = "SchildCombined94"
_CH_PARAMS = SchildCombined94().channel_params
_CH_STATES = SchildCombined94().channel_states

GBAR_NAF    = _CH_PARAMS[f"{_P}_gbar_naf"]
GBAR_NAS    = _CH_PARAMS[f"{_P}_gbar_nas"]
GBAR_KD     = _CH_PARAMS[f"{_P}_gbar_kd"]
GBAR_KA     = _CH_PARAMS[f"{_P}_gbar_ka"]
GBAR_KDS    = _CH_PARAMS[f"{_P}_gbar_kds"]
GBAR_KCA    = _CH_PARAMS[f"{_P}_gbar_kca"]
GBAR_CAN    = _CH_PARAMS[f"{_P}_gbar_can"]
GBAR_CAT    = _CH_PARAMS[f"{_P}_gbar_cat"]
GBNA_LEAK   = _CH_PARAMS[f"{_P}_gbna_leak"]
GBCA_LEAK   = _CH_PARAMS[f"{_P}_gbca_leak"]
NAI         = _CH_PARAMS[f"{_P}_nai"]
NAO         = _CH_PARAMS[f"{_P}_nao"]
KI          = _CH_PARAMS[f"{_P}_ki"]
KO          = _CH_PARAMS[f"{_P}_ko"]
INAKMAX22   = _CH_PARAMS[f"{_P}_INaKmax22"]
KMNAI       = _CH_PARAMS[f"{_P}_Kmnai"]
KMKO        = _CH_PARAMS[f"{_P}_Kmko"]
ICAPMAX22   = _CH_PARAMS[f"{_P}_ICaPmax22"]
KMCA        = _CH_PARAMS[f"{_P}_KmCa"]
KNACA22     = _CH_PARAMS[f"{_P}_KNaCa22"]
DNACA       = _CH_PARAMS[f"{_P}_DNaCa"]
CABATH      = _CH_PARAMS[f"{_P}_cabath"]
TXFER       = _CH_PARAMS[f"{_P}_txfer"]

# initial gate values (approx SS at V_REST = -46.5 mV)
_GATE_INIT = {k.replace(f"{_P}_", ""): v for k, v in _CH_STATES.items()}


# ── JAX geometry + coupled-solver setup ───────────────────────────────────────

def _make_jax_setup(D: float):
    """Build coupled-solver static arrays and Schild94 membrane_fn for diameter D."""
    _, geom = build_schild94(diameter=D, n_nodes=N_NODES)
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

    A_in = static["A_in_cm2"]   # [n_comp] cm²

    # Ca geometry (same for all compartments since uniform diameter)
    SA, Vol, Vol_peri = ca_geometry(DELTA_Z, D, fhspace_um=FHSPACE)

    # Initial state (approx SS at V_REST)
    def _full(v): return jnp.full(n_comp, v, dtype=jnp.float64)

    state0 = (
        _full(_GATE_INIT["m_naf"]),
        _full(_GATE_INIT["h_naf"]),
        _full(_GATE_INIT["l_naf"]),
        _full(_GATE_INIT["m_nas"]),
        _full(_GATE_INIT["h_nas"]),
        _full(_GATE_INIT["n_kd"]),
        _full(_GATE_INIT["p_ka"]),
        _full(_GATE_INIT["q_ka"]),
        _full(_GATE_INIT["x_kds"]),
        _full(_GATE_INIT["y_kds"]),
        _full(_GATE_INIT["c_kca"]),
        _full(_GATE_INIT["d_can"]),
        _full(_GATE_INIT["f1_can"]),
        _full(_GATE_INIT["f2_can"]),
        _full(_GATE_INIT["d_cat"]),
        _full(_GATE_INIT["f_cat"]),
        _full(_GATE_INIT["cai"]),
        _full(_GATE_INIT["cao"]),
        _full(_GATE_INIT["Oc"]),
    )

    def membrane_fn(Vm, state, dt):
        (m_naf, h_naf, l_naf, m_nas, h_nas, n_kd, p_ka, q_ka,
         x_kds, y_kds, c_kca, d_can, f1_can, f2_can, d_cat, f_cat,
         cai, cao, Oc) = state

        # Reversal potentials from current Ca state
        ena, ek, eca = _reversal_potentials(CELSIUS, NAI, NAO, KI, KO, cai, cao)

        # Total Ca current density (mA/cm²) for Ca ODE update
        ica_can_now = GBAR_CAN * d_can * (0.55 * f1_can + 0.45 * f2_can) * (Vm - eca)
        ica_cat_now = GBAR_CAT * d_cat * f_cat * (Vm - eca)
        ica_leak_now = GBCA_LEAK * (Vm - eca)
        ica_cap_now  = _capump_current(cai, CELSIUS, ICAPMAX22, KMCA)
        inca_now     = _nacapump_current(Vm, cai, cao, NAI, NAO, CELSIUS, KNACA22, DNACA)
        ica_total    = (ica_can_now + ica_cat_now + ica_leak_now
                        + ica_cap_now + (-2.0 * inca_now))

        # Update Ca states (explicit Euler)
        new_cai, new_cao, new_Oc = _ca_ode_step(
            ica_total, cai, cao, Oc, dt, SA, Vol, Vol_peri, CABATH, TXFER,
        )

        # Update voltage-gated channels (exponential gate method, OLD cai for kca)
        (tau_m, minf), (tau_h, hinf), (tau_l, linf) = SchildCombined94._naf94_kinetics(Vm, CELSIUS)
        (tau_mn, mninf), (tau_hn, hninf) = SchildCombined94._nas94_kinetics(Vm, CELSIUS)
        tau_n,  ninf    = _kd_kinetics(Vm, CELSIUS)
        (tau_p, pinf),  (tau_q, qinf)  = _ka_kinetics(Vm, CELSIUS)
        (tau_x, xinf),  (tau_y, yinf)  = _kds_kinetics(Vm, CELSIUS)
        tau_c,  cinf    = _kca_kinetics(Vm, cai, CELSIUS)
        (tau_d, dinf),  (tau_f1, f1inf), (tau_f2, f2inf) = _can_kinetics(Vm, CELSIUS)
        (tau_dc, dcinf), (tau_fc, fcinf) = _cat_kinetics(Vm, CELSIUS)

        new_m_naf  = _update_gate(m_naf,  dt, tau_m,  minf)
        new_h_naf  = _update_gate(h_naf,  dt, tau_h,  hinf)
        new_l_naf  = _update_gate(l_naf,  dt, tau_l,  linf)
        new_m_nas  = _update_gate(m_nas,  dt, tau_mn, mninf)
        new_h_nas  = _update_gate(h_nas,  dt, tau_hn, hninf)
        new_n_kd   = _update_gate(n_kd,   dt, tau_n,  ninf)
        new_p_ka   = _update_gate(p_ka,   dt, tau_p,  pinf)
        new_q_ka   = _update_gate(q_ka,   dt, tau_q,  qinf)
        new_x_kds  = _update_gate(x_kds,  dt, tau_x,  xinf)
        new_y_kds  = _update_gate(y_kds,  dt, tau_y,  yinf)
        new_c_kca  = _update_gate(c_kca,  dt, tau_c,  cinf)
        new_d_can  = _update_gate(d_can,  dt, tau_d,  dinf)
        new_f1_can = _update_gate(f1_can, dt, tau_f1, f1inf)
        new_f2_can = _update_gate(f2_can, dt, tau_f2, f2inf)
        new_d_cat  = _update_gate(d_cat,  dt, tau_dc, dcinf)
        new_f_cat  = _update_gate(f_cat,  dt, tau_fc, fcinf)

        new_state = (new_m_naf, new_h_naf, new_l_naf, new_m_nas, new_h_nas,
                     new_n_kd, new_p_ka, new_q_ka, new_x_kds, new_y_kds, new_c_kca,
                     new_d_can, new_f1_can, new_f2_can, new_d_cat, new_f_cat,
                     new_cai, new_cao, new_Oc)

        # Linearized conductance with NEW gates (µS)
        g_naf = GBAR_NAF * new_m_naf**3 * new_h_naf * new_l_naf
        g_nas = GBAR_NAS * new_m_nas**3 * new_h_nas
        g_kd  = GBAR_KD  * new_n_kd
        g_ka  = GBAR_KA  * new_p_ka**3 * new_q_ka
        g_kds = GBAR_KDS * new_x_kds**3 * new_y_kds
        g_kca = GBAR_KCA * new_c_kca
        g_can = GBAR_CAN * new_d_can * (0.55 * new_f1_can + 0.45 * new_f2_can)
        g_cat = GBAR_CAT * new_d_cat * new_f_cat
        g_lin = (g_naf + g_nas + g_kd + g_ka + g_kds + g_kca
                 + g_can + g_cat + GBNA_LEAK + GBCA_LEAK)

        # Total ionic current density with NEW gates + OLD Vm (mA/cm²)
        i_naf = g_naf * (Vm - ena)
        i_nas = g_nas * (Vm - ena)
        i_kd  = g_kd  * (Vm - ek)
        i_ka  = g_ka  * (Vm - ek)
        i_kds = g_kds * (Vm - ek)
        i_kca = g_kca * (Vm - ek)
        i_can = g_can * (Vm - eca)
        i_cat = g_cat * (Vm - eca)
        i_leak_na = GBNA_LEAK * (Vm - ena)
        i_leak_ca = GBCA_LEAK * (Vm - eca)
        ink      = _nakpump_current(Vm, NAI, KO, CELSIUS, INAKMAX22, KMNAI, KMKO)
        ica_cap  = _capump_current(cai, CELSIUS, ICAPMAX22, KMCA)
        inca     = _nacapump_current(Vm, cai, cao, NAI, NAO, CELSIUS, KNACA22, DNACA)
        i_total  = (i_naf + i_nas + i_kd + i_ka + i_kds + i_kca + i_can + i_cat
                    + i_leak_na + i_leak_ca
                    + 3.0 * ink + (-2.0 * ink)
                    + ica_cap
                    + 3.0 * inca + (-2.0 * inca))

        g_eff = g_lin  * A_in * 1e6     # µS
        i_ion = i_total * A_in * 1e6    # nA  (mA/cm² × cm² × 1e6 = nA)

        return g_eff, i_ion, new_state

    return static, membrane_fn, state0, Ve_unit, mid, centers, nodes, geom


def _make_jit_runner(static, mfn, state0, Ve_unit):
    """JIT runner: (amp_mA, pulse_shape) -> peak Vm over all nodes."""
    Ve_j    = jnp.asarray(Ve_unit, dtype=jnp.float64)
    is_node = static["is_node"]   # all True for Schild

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
    """JIT runner: (amp_mA, pulse_shape) -> (Vi[nsteps,n], Vp[nsteps,n])."""
    Ve_j = jnp.asarray(Ve_unit, dtype=jnp.float64)

    @jax.jit
    def run(amp_mA, pulse_shape):
        Ve = Ve_j * (amp_mA / -1.0)
        trace, _ = integrate(static, mfn, state0, Ve, pulse_shape, DT,
                             v_rest=V_REST, record="all")
        return trace

    return run


# ── Task 1: Vm + gate + Ca traces ─────────────────────────────────────────────

# Jaxley state names for SchildCombined94
CH = "SchildCombined94"
JAX_STATES = {
    "m_naf":  f"{CH}_m_naf",
    "h_naf":  f"{CH}_h_naf",
    "l_naf":  f"{CH}_l_naf",
    "m_nas":  f"{CH}_m_nas",
    "h_nas":  f"{CH}_h_nas",
    "n_kd":   f"{CH}_n_kd",
    "cai":    f"{CH}_cai",
    "cao":    f"{CH}_cao",
}


def task_traces() -> dict:
    """Intracellular and extracellular Vm + gate + Ca traces at D=0.8 µm."""
    print("\n=== Task 1: Vm + gate + Ca traces (D=0.8 µm) ===")
    D = TRACE_DIAM

    # ── Intracellular (Jaxley bwd_euler) ─────────────────────────────────────
    print("  [intra JAX] building ...", flush=True)
    cell_jax, geom_jax = build_schild94(diameter=D, n_nodes=N_NODES)
    nodes_jax = node_indices(geom_jax)
    mid_comp  = nodes_jax[len(nodes_jax) // 2]
    n_steps_i = int(10.0 / DT) + 1
    t_intra   = np.arange(n_steps_i) * DT
    pulse_i   = rectangular_pulse(t_intra, DELAY, INTRA_PW_MS, INTRA_AMP_NA)
    attach_intra_pulse(cell_jax, mid_comp, pulse_i)
    cell_jax.branch(0).comp(mid_comp).record("v")
    for jax_key in JAX_STATES.values():
        cell_jax.branch(0).comp(mid_comp).record(jax_key)

    print("  [intra JAX] integrating ...", flush=True)
    t0_jax = time.time()
    rec    = np.asarray(jx.integrate(cell_jax, delta_t=DT, t_max=10.0, solver="bwd_euler"))
    print(f"  [intra JAX] done in {time.time() - t0_jax:.1f} s", flush=True)
    jax_vm_i    = rec[0]
    jax_gates_i = {k: rec[i + 1] for i, k in enumerate(JAX_STATES)}

    print("  [intra NEURON] ...", flush=True)
    t0_pf = time.time()
    nr_i  = run_intracellular_schild94(
        diameter=D, n_nodes=N_NODES, temperature=CELSIUS,
        i_delay_ms=DELAY, i_dur_ms=INTRA_PW_MS, i_amp_nA=INTRA_AMP_NA,
        dt_ms=DT, tstop_ms=10.0,
    )
    print(f"  [intra NEURON] done in {time.time() - t0_pf:.1f} s", flush=True)
    pf_vm_i  = nr_i.vm_mV[nr_i.probe_node_idx]
    pf_gates_i = nr_i.gates

    vm_jax_on_pf = np.interp(nr_i.t_ms, t_intra, jax_vm_i)
    rmse_i = float(np.sqrt(np.mean((vm_jax_on_pf - pf_vm_i)**2)))
    print(f"  Intra RMSE = {rmse_i:.3f} mV")

    # ── Extracellular (coupled solver) ────────────────────────────────────────
    print("  [extra JAX] building ...", flush=True)
    static, mfn, state0, Ve_unit, mid, centers, nodes, geom = _make_jax_setup(D)
    runner_all = _make_jit_all_runner(static, mfn, state0, Ve_unit)
    runner     = _make_jit_runner(static, mfn, state0, Ve_unit)

    pm_extra = jnp.asarray(make_pulse_array("mono_c", EXTRA_PW_MS, N_STEPS, DT, DELAY))
    print("  [extra JAX] finding threshold ...", flush=True)
    thr_jax = jax_bisect(runner, np.asarray(pm_extra),
                          lo=PULSES["mono_c"].lo, hi=PULSES["mono_c"].hi)
    amp_extra = float(thr_jax) * 1.3

    print(f"  [extra JAX] running at {amp_extra:.3f} mA ...", flush=True)
    t0 = time.time()
    Vi_e, Vp_e = runner_all(jnp.float64(amp_extra), pm_extra)
    print(f"  [extra JAX] done in {time.time() - t0:.1f} s", flush=True)
    t_extra   = (np.arange(N_STEPS) + 1) * DT
    vm_e_jax  = np.asarray(Vi_e - Vp_e)[:, mid]

    print("  [extra NEURON] ...", flush=True)
    t0 = time.time()
    nr_e = run_extracellular_schild94(
        diameter=D, n_nodes=N_NODES, src_height_um=SRC_H,
        pw_ms=EXTRA_PW_MS, delay_ms=DELAY, amp_mA=amp_extra,
        dt_ms=DT, tstop_ms=TSTOP,
    )
    print(f"  [extra NEURON] done in {time.time() - t0:.1f} s", flush=True)

    vm_e_pf = nr_e.vm_mV[nr_e.probe_node_idx]
    vm_e_jax_on_pf = np.interp(nr_e.t_ms, t_extra, vm_e_jax)
    rmse_e = float(np.sqrt(np.mean((vm_e_jax_on_pf - vm_e_pf)**2)))
    peak_jax_e = float(np.max(vm_e_jax))
    peak_pf_e  = float(np.max(vm_e_pf))
    print(f"  Extra RMSE = {rmse_e:.3f} mV | peak JAX={peak_jax_e:.1f} mV | PF={peak_pf_e:.1f} mV")

    data = dict(
        diameter=D, celsius=CELSIUS,
        # intracellular
        t_intra_jax=t_intra.tolist(), vm_intra_jax=jax_vm_i.tolist(),
        gates_intra_jax={k: v.tolist() for k, v in jax_gates_i.items()},
        t_intra_nrn=nr_i.t_ms.tolist(), vm_intra_nrn=pf_vm_i.tolist(),
        gates_intra_nrn={k: v.tolist() for k, v in pf_gates_i.items()},
        rmse_intra_mV=rmse_i,
        # extracellular
        t_extra_jax=t_extra.tolist(), vm_extra_jax=vm_e_jax.tolist(),
        t_extra_nrn=nr_e.t_ms.tolist(), vm_extra_nrn=vm_e_pf.tolist(),
        rmse_extra_mV=rmse_e,
        amp_extra_mA=amp_extra,
        peak_jax_mV=peak_jax_e, peak_pf_mV=peak_pf_e,
    )
    save_json(data, OUT / "data_schild94_traces.json")
    return data


# ── Task 2: Strength-duration curves ─────────────────────────────────────────

def task_sd_curves() -> dict:
    """SD curves — 3 diameters × 8 pulse shapes × 6 PWs (JAX + PyFibers)."""
    print("\n=== Task 2: Strength-duration curves ===")
    results: dict = {}

    for D in DIAMETERS:
        print(f"\n  Diameter = {D} µm", flush=True)
        static, mfn, state0, Ve_unit, mid, centers, nodes, geom = _make_jax_setup(D)
        runner = _make_jit_runner(static, mfn, state0, Ve_unit)

        # JIT warm-up
        pm_warm = make_pulse_array("mono_c", 0.2, N_STEPS, DT, DELAY)
        _ = jax_bisect(runner, pm_warm,
                       lo=PULSES["mono_c"].lo, hi=PULSES["mono_c"].hi)

        diam_res: dict = {}
        for pk in PULSE_KEYS:
            ps = PULSES[pk]
            diam_res[pk] = {}
            for pw in SD_PWS:
                pm = make_pulse_array(pk, pw, N_STEPS, DT, DELAY)

                # JAX bisect
                t0 = time.time()
                thr_jax = jax_bisect(runner, pm, lo=ps.lo, hi=ps.hi)
                t_jax = time.time() - t0

                # PyFibers reference
                t0 = time.time()
                try:
                    thr_pf = pf_find_threshold(
                        "schild94", D, N_NODES, pm, DT, TSTOP,
                        lo=ps.lo, hi=ps.hi,
                        temperature=CELSIUS,
                        src_height_um=SRC_H,
                        sigma_S_m=SIGMA,
                    )
                except Exception as e:
                    thr_pf = float("nan")
                    print(f"    PF failed {pk} pw={pw}: {e}", flush=True)
                t_pf = time.time() - t0

                err_pct = (float(thr_jax) - thr_pf) / abs(thr_pf) * 100 \
                          if not np.isnan(thr_pf) else float("nan")
                diam_res[pk][pw] = {
                    "jax_mA": float(thr_jax), "pf_mA": thr_pf,
                    "err_pct": err_pct,
                }
                print(f"    {pk:6s} pw={pw:.2f} ms | JAX={float(thr_jax):.4f} | "
                      f"PF={thr_pf:.4f} | err={err_pct:+.1f}% | "
                      f"t_jax={t_jax:.1f}s t_pf={t_pf:.1f}s", flush=True)

        results[str(D)] = diam_res

    save_json(results, OUT / "data_schild94_sd.json")
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


def _pf_cv(nr, centers, nodes) -> float:
    onset = int(np.searchsorted(nr.t_ms, DELAY))
    k_ctr, k_end = len(nodes) // 2, 2
    t_ctr = nr.t_ms[onset + int(np.argmax(nr.vm_mV[nodes[k_ctr], onset:]))]
    t_end = nr.t_ms[onset + int(np.argmax(nr.vm_mV[nodes[k_end], onset:]))]
    dist  = abs(float(centers[nodes[k_ctr]]) - float(centers[nodes[k_end]]))
    return dist / abs(t_end - t_ctr) * 1e-3


def task_cv() -> dict:
    """Conduction velocity for 3 Schild94 diameters (JAX + PyFibers)."""
    print("\n=== Task 3: Conduction velocity ===")
    results: dict = {}

    for D in DIAMETERS:
        static, mfn, state0, Ve_unit, mid, centers, nodes, geom = _make_jax_setup(D)
        runner     = _make_jit_runner(static, mfn, state0, Ve_unit)
        runner_all = _make_jit_all_runner(static, mfn, state0, Ve_unit)

        pm  = make_pulse_array("mono_c", 0.2, N_STEPS, DT, DELAY)
        thr = jax_bisect(runner, pm, lo=PULSES["mono_c"].lo, hi=PULSES["mono_c"].hi)
        amp = float(thr) * 1.3

        t0     = time.time()
        jax_cv = _jax_cv(runner_all, amp, nodes, centers)
        t_jax  = time.time() - t0

        t0 = time.time()
        nr = run_extracellular_schild94(
            D, N_NODES, src_height_um=SRC_H, pw_ms=0.2,
            delay_ms=DELAY, amp_mA=amp, dt_ms=DT, tstop_ms=TSTOP,
        )
        pf_cv = _pf_cv(nr, centers, nodes)
        t_pf  = time.time() - t0

        err = (jax_cv - pf_cv) / pf_cv * 100
        results[str(D)] = {"jax": float(jax_cv), "pyfibers": float(pf_cv),
                           "err_pct": float(err), "threshold_mA": float(thr)}
        print(f"  D={D:.1f} µm | PF={pf_cv:.3f} | JAX={jax_cv:.3f} m/s | "
              f"err={err:+.1f}% | t_jax={t_jax:.1f}s t_pf={t_pf:.1f}s", flush=True)

    save_json(results, OUT / "data_schild94_cv.json")
    return results


# ── Figure 1: Traces ──────────────────────────────────────────────────────────

def fig_traces(data: dict) -> None:
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    # (A) Intracellular Vm
    ax = axs[0, 0]
    ax.plot(data["t_intra_nrn"], data["vm_intra_nrn"], "k-", lw=1.8, label="NEURON")
    ax.plot(data["t_intra_jax"], data["vm_intra_jax"], "r--", lw=1.0,
            label=f"JAX (RMSE={data['rmse_intra_mV']:.2f} mV)")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("Vm (mV)")
    ax.set_title(f"(A) Intra Vm  D={data['diameter']} µm")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (B) naf gates
    ax = axs[0, 1]
    g_colors = {"m_naf": "C0", "h_naf": "C2", "l_naf": "C4"}
    jg = data["gates_intra_jax"]
    pg = data["gates_intra_nrn"]
    for gk, c in g_colors.items():
        full_k = f"SchildCombined94_{gk}"
        if full_k in jg:
            ax.plot(data["t_intra_jax"], jg[full_k], "--", color=c, lw=1.0, label=f"JAX {gk}")
    for pk, pv in pg.items():
        for gk, c in g_colors.items():
            if gk.split("_")[0] in pk.lower():
                ax.plot(data["t_intra_nrn"], pv, "-", color=c, lw=1.5, alpha=0.7, label=f"NRN {pk}")
                break
    ax.set_xlabel("time (ms)"); ax.set_ylabel("gate")
    ax.set_title("(B) naf gates (m, h, l)")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)

    # (C) Extracellular Vm
    ax = axs[0, 2]
    ax.plot(data["t_extra_nrn"], data["vm_extra_nrn"], "k-", lw=1.8, label="NEURON")
    ax.plot(data["t_extra_jax"], data["vm_extra_jax"], "r--", lw=1.0,
            label=f"JAX (RMSE={data['rmse_extra_mV']:.2f} mV)")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("Vm (mV)")
    ax.set_title(f"(C) Extra Vm  {data['amp_extra_mA']:.3f} mA")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (D) cai
    ax = axs[1, 0]
    full_cai = "SchildCombined94_cai"
    if full_cai in jg:
        ax.plot(data["t_intra_jax"], np.array(jg[full_cai]) * 1e3, "r--", lw=1.2, label="JAX cai (µM)")
    for pk, pv in pg.items():
        if "cai" in pk.lower():
            ax.plot(data["t_intra_nrn"], np.array(pv) * 1e3, "k-", lw=1.5, label=f"NRN {pk}")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("cai (µM)")
    ax.set_title("(D) Intracellular Ca²⁺")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (E) cao
    ax = axs[1, 1]
    full_cao = "SchildCombined94_cao"
    if full_cao in jg:
        ax.plot(data["t_intra_jax"], jg[full_cao], "r--", lw=1.2, label="JAX cao")
    for pk, pv in pg.items():
        if "cao" in pk.lower():
            ax.plot(data["t_intra_nrn"], pv, "k-", lw=1.5, label=f"NRN {pk}")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("cao (mM)")
    ax.set_title("(E) Periaxonal Ca²⁺")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (F) Summary
    ax = axs[1, 2]
    lines = [
        "Model: Schild 1994 C-fiber",
        f"D = {data['diameter']} µm,  T = {data['celsius']}°C",
        f"N_comp = {N_NODES},  dt = {DT} ms",
        f"Intra stim: {INTRA_AMP_NA} nA × {INTRA_PW_MS} ms",
        "",
        f"Intra RMSE: {data['rmse_intra_mV']:.3f} mV",
        f"Extra RMSE: {data['rmse_extra_mV']:.3f} mV",
        f"Extra amp:  {data['amp_extra_mA']:.3f} mA",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=8, fontfamily="monospace")
    ax.set_axis_off()
    ax.set_title("(F) Summary")

    fig.suptitle("Schild 1994 C-fiber — JAX coupled solver vs NEURON", fontsize=11)
    path = OUT / "fig_schild94_traces.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ── Figure 2: SD curves ───────────────────────────────────────────────────────

def fig_sd_curves(sd_data: dict) -> None:
    n_diams = len(DIAMETERS)
    fig, axes = plt.subplots(n_diams, 1, figsize=(9, 4 * n_diams), constrained_layout=True)
    if n_diams == 1:
        axes = [axes]

    for ax, D in zip(axes, DIAMETERS):
        diam_res = sd_data[str(D)]
        for pk in PULSE_KEYS:
            color = GROUP_COLORS.get(pk, "gray")
            pws   = sorted(diam_res[pk].keys(), key=float)
            jax_t = [abs(diam_res[pk][pw]["jax_mA"]) for pw in pws]
            pf_t  = [abs(diam_res[pk][pw]["pf_mA"])  for pw in pws]
            pws_f = [float(pw) for pw in pws]
            ax.plot(pws_f, pf_t,  "-",  color=color, lw=1.8, label=f"PF {pk}")
            ax.plot(pws_f, jax_t, "--", color=color, lw=1.0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("PW (ms)"); ax.set_ylabel("|threshold| (mA)")
        ax.set_title(f"SD curves  D={D} µm  (solid=PF, dashed=JAX)")
        ax.legend(fontsize=6, ncol=4); ax.grid(alpha=0.3, which="both")

    fig.suptitle("Schild 1994 C-fiber — Strength-Duration Curves", fontsize=12)
    path = OUT / "fig_schild94_sd_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ── Figure 3: Analysis ────────────────────────────────────────────────────────

def fig_analysis(sd_data: dict, cv_data: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    # (A) Correlation plot: JAX vs PF threshold (all combinations)
    ax = axes[0]
    all_jax, all_pf = [], []
    for D_str in sd_data:
        for pk in PULSE_KEYS:
            for pw, v in sd_data[D_str][pk].items():
                if not np.isnan(v["pf_mA"]):
                    all_jax.append(abs(v["jax_mA"]))
                    all_pf.append(abs(v["pf_mA"]))
    all_jax = np.array(all_jax)
    all_pf  = np.array(all_pf)
    slope, intercept, r, p, _ = stats.linregress(all_pf, all_jax)
    ax.scatter(all_pf, all_jax, s=15, alpha=0.6, color="steelblue", edgecolors="none")
    xlim = ax.get_xlim()
    ax.plot(xlim, [slope * x + intercept for x in xlim], "r-", lw=1.5,
            label=f"r={r:.4f}, slope={slope:.3f}")
    ax.set_xlabel("|threshold| PF (mA)"); ax.set_ylabel("|threshold| JAX (mA)")
    ax.set_title("(A) Threshold correlation")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (B) Error histogram
    ax = axes[1]
    errs = [v["err_pct"]
            for D_str in sd_data for pk in PULSE_KEYS
            for v in sd_data[D_str][pk].values()
            if not np.isnan(v["err_pct"])]
    errs = np.array(errs)
    ax.hist(errs, bins=30, color="steelblue", edgecolor="white", lw=0.5)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("% error (JAX vs PF)"); ax.set_ylabel("count")
    ax.set_title(f"(B) Error dist.  median={np.median(errs):.2f}%  "
                 f"IQR={np.percentile(errs,75)-np.percentile(errs,25):.2f}%")
    ax.grid(alpha=0.3)

    # (C) Conduction velocity
    ax = axes[2]
    d_vals = [float(k) for k in cv_data]
    jax_cv = [cv_data[k]["jax"] for k in cv_data]
    pf_cv  = [cv_data[k]["pyfibers"] for k in cv_data]
    ax.plot(d_vals, pf_cv,  "ko-",  lw=2, ms=6, label="PyFibers")
    ax.plot(d_vals, jax_cv, "rs--", lw=1, ms=5, label="JAX")
    ax.set_xlabel("Diameter (µm)"); ax.set_ylabel("CV (m/s)")
    ax.set_title("(C) Conduction velocity")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle("Schild 1994 C-fiber — Validation Analysis", fontsize=12)
    path = OUT / "fig_schild94_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")
    t0 = time.time()

    trace_data = task_traces()
    fig_traces(trace_data)

    sd_data = task_sd_curves()
    fig_sd_curves(sd_data)

    cv_data = task_cv()
    fig_analysis(sd_data, cv_data)

    print(f"\nDone. Total wall time: {(time.time() - t0) / 60:.1f} min")
