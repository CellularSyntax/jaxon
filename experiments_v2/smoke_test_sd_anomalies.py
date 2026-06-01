"""Smoke test for SD curve threshold anomalies.

Tests three cases where JAX and PyFibers disagree significantly:
  (A) MRG D=5.7, mono_a, PW=0.20 ms — JAX=0.51 vs PF~19 mA
  (B) MRG D=5.7, mono_a, PW=0.10 ms — control (both agree ~0.91 mA)
  (C) Rattay D=0.3, bi_ca, PW=0.05 ms — direction error flip

For each amplitude tested, outputs:
  - JAX spatiotemporal Vm map (heat map)
  - Which nodes exceed v_thresh=-30 mV (JAX "fires" criterion)
  - Whether the center node fires in JAX
  - Whether PF fires at center probe
  - Whether PF fires at end nodes
  - Any PyFibers warnings (stdout captured)

Run from project root:
    python experiments_v2/smoke_test_sd_anomalies.py
"""

from __future__ import annotations
import sys, pathlib, io, contextlib, time, dataclasses
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
    build_mrg, node_indices, section_centers_um, V_REST, CM_AXON,
    G_PAS_MYSA, G_PAS_FLUT, G_PAS_STIN,
)
from jaxfibers.channels.mrg_axnode import AxnodeMyel
from jaxfibers.stim.extracellular import point_source_potentials_mV
from jaxfibers.stim.extracellular_coupled import arrays_from_geometry, integrate
from jaxfibers.fibers.rattay import (
    build_rattay, node_indices as rattay_node_indices,
    section_centers_um as rattay_section_centers_um,
    V_REST as RATTAY_V_REST, CM as RATTAY_CM,
)
from jaxfibers.channels.rattay_channels import RattayHH
from jaxfibers.nrn_baseline import build_mrg_pyfibers, build_rattay_pyfibers
from pyfibers.stimulation import ScaledStim

from experiments_v2.utils import make_pulse_array, PULSES

OUT = pathlib.Path(ROOT / "outputs" / "smoke_test_sd")
OUT.mkdir(parents=True, exist_ok=True)

V_THRESH = -30.0   # mV — AP detection threshold (same as jax_bisect default)

# MRG constants (same as mrg_validation.py)
GNABAR = 3.0;  GNAPBAR = 0.01; GKBAR = 0.08; GL = 0.007
ENA = 50.0;    EK = -90.0;     EL   = -90.0
CELSIUS = 37.0
DT      = 0.005   # ms
TSTOP   = 10.0    # ms  (longer than MRG validation to catch post-pulse events)
DELAY   = 1.0     # ms
SIGMA   = 0.3     # S/m
SRC_H   = 1000.0  # µm

# Rattay constants
RATTAY_GNABAR = 0.12;  RATTAY_GKBAR = 0.036;  RATTAY_GL   = 3e-4
RATTAY_EL     = -59.4; RATTAY_ENA   = 45.0;   RATTAY_EK   = -82.0


# ── JAX setup helpers ─────────────────────────────────────────────────────────

def _make_mrg_setup(D: float, n_nodes: int = 21):
    _, geom = build_mrg(diameter=D, n_nodes=n_nodes)
    nodes   = node_indices(geom)
    centers = np.array(section_centers_um(geom))
    mid     = nodes[len(nodes) // 2]
    n_comp  = geom.n_comp

    geom_c = dataclasses.replace(geom, cm_uF_cm2=[CM_AXON] * n_comp)
    static  = arrays_from_geometry(geom_c, DT)

    Ve_unit = np.array(point_source_potentials_mV(
        list(centers), src_x_um=0., src_y_um=SRC_H,
        src_z_um=float(centers[mid]), i0_mA=-1.0,
    ))

    is_node   = static["is_node"]
    A_in      = static["A_in_cm2"]
    stype_gp  = {"node": 0.0, "mysa": G_PAS_MYSA, "flut": G_PAS_FLUT, "stin": G_PAS_STIN}
    g_pas_arr = jnp.asarray([stype_gp[s] for s in geom.section_type])

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
        g_na  = GNABAR  * M2**3 * H2
        g_nap = GNAPBAR * MP2**3
        g_k   = GKBAR   * S2
        g_node = (g_na + g_nap + g_k + GL) * A_in * 1e6
        i_node = ((g_na + g_nap)*(Vm-ENA) + g_k*(Vm-EK) + GL*(Vm-EL)) * A_in * 1e6
        g_pas_us = g_pas_arr * A_in * 1e6
        i_pas    = g_pas_us  * (Vm - V_REST)
        g_eff = jnp.where(is_node, g_node,  g_pas_us)
        i_ion = jnp.where(is_node, i_node,  i_pas)
        return g_eff, i_ion, (M2, H2, MP2, S2)

    return static, membrane_fn, state0, Ve_unit, nodes, centers, mid, geom


def _make_rattay_setup(D: float, n_nodes: int = 51):
    _, geom = build_rattay(diameter=D, n_nodes=n_nodes)
    nodes   = rattay_node_indices(geom)
    centers = np.array(rattay_section_centers_um(geom))
    mid     = nodes[len(nodes) // 2]
    n_comp  = geom.n_comp

    geom_c = dataclasses.replace(geom, cm_uF_cm2=[RATTAY_CM] * n_comp)
    static  = arrays_from_geometry(geom_c, DT)

    Ve_unit = np.array(point_source_potentials_mV(
        list(centers), src_x_um=0., src_y_um=SRC_H,
        src_z_um=float(centers[mid]), i0_mA=-1.0,
    ))

    A_in = static["A_in_cm2"]
    is_node = static["is_node"]

    v0 = jnp.float64(RATTAY_V_REST)
    (am0,bm0),(ah0,bh0),(an0,bn0) = RattayHH._alpha_beta(v0, CELSIUS)
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
        g_na  = RATTAY_GNABAR * M2**3 * H2
        g_k   = RATTAY_GKBAR  * N2**4
        g_tot = (g_na + g_k + RATTAY_GL) * A_in * 1e6
        i_ion = (g_na*(Vm-RATTAY_ENA) + g_k*(Vm-RATTAY_EK)
                 + RATTAY_GL*(Vm-RATTAY_EL)) * A_in * 1e6
        return g_tot, i_ion, (M2, H2, N2)

    return static, membrane_fn, state0, Ve_unit, nodes, centers, mid, geom


# ── JAX full simulation + diagnostics ────────────────────────────────────────

@jax.jit
def _jax_run_all_mrg(static, membrane_fn, state0, Ve_unit, amp_mA, pulse_shape):
    Ve = jnp.asarray(Ve_unit) * (amp_mA / -1.0)
    trace, _ = integrate(static, membrane_fn, state0, Ve, pulse_shape, DT,
                         v_rest=V_REST, record="all")
    return trace


def jax_diagnostics(static, membrane_fn, state0, Ve_unit, nodes, centers, mid,
                    is_node, amp_mA, pulse_arr, label="", v_rest=V_REST):
    """Run JAX at given amp and return diagnostics dict."""
    Ve_j = jnp.asarray(Ve_unit, dtype=jnp.float64)
    pm   = jnp.asarray(pulse_arr, dtype=jnp.float64)
    Ve   = Ve_j * (jnp.float64(amp_mA) / -1.0)

    trace, _ = integrate(static, membrane_fn, state0, Ve, pm, DT,
                         v_rest=v_rest, record="all")
    Vi_all, Vp_all = np.asarray(trace[0]), np.asarray(trace[1])
    Vm_all = Vi_all - Vp_all         # [n_steps, n_comp]

    n_steps = Vm_all.shape[0]
    t_arr   = (np.arange(n_steps) + 1) * DT

    # Metrics
    node_idx    = np.where(np.asarray(is_node))[0]
    Vm_nodes    = Vm_all[:, node_idx]          # [n_steps, n_nodes]
    peak_per_node = Vm_nodes.max(axis=0)       # [n_nodes]  max Vm over time

    center_node_global = mid                    # global compartment index
    center_node_local  = int(np.searchsorted(node_idx, center_node_global))
    center_peak        = float(Vm_nodes[:, center_node_local].max())

    any_node_fires  = bool(peak_per_node.max() > V_THRESH)
    center_fires    = bool(center_peak > V_THRESH)
    n_nodes_firing  = int((peak_per_node > V_THRESH).sum())

    # First node and last node (peripheral)
    end1_peak = float(peak_per_node[0])
    end2_peak = float(peak_per_node[-1])

    return {
        "label": label,
        "amp_mA": float(amp_mA),
        "any_node_fires": any_node_fires,
        "center_fires": center_fires,
        "n_nodes_firing": n_nodes_firing,
        "center_peak_mV": center_peak,
        "end1_peak_mV": end1_peak,
        "end2_peak_mV": end2_peak,
        "max_Vm_mV": float(peak_per_node.max()),
        "argmax_node": int(np.argmax(peak_per_node)),
        "Vm_all": Vm_all,             # [n_steps, n_comp]
        "Vm_nodes": Vm_nodes,         # [n_steps, n_nodes]
        "peak_per_node": peak_per_node,
        "t_arr": t_arr,
        "node_idx": node_idx,
    }


# ── PyFibers diagnostics ──────────────────────────────────────────────────────

def pf_diagnostics(fiber_builder, amp_mA, pw_ms, tstop_ms=TSTOP, label=""):
    """Run PyFibers at given amp and return diagnostics.

    Captures stdout (warnings), checks center and end-node AP count.
    fiber_builder: callable() -> fiber (pre-built, will be rebuilt each call).
    """
    fiber = fiber_builder()
    probe_idx = fiber.loc_index(0.5)
    fiber.record_vm()

    # Add APC at every node (not just center)
    from neuron import h  # noqa: F401
    apc_list = []
    try:
        for i in range(fiber.nodecount):
            apc = fiber.apc[i]   # may already exist from build
            apc_list.append(apc)
    except Exception:
        pass

    fiber.potentials = fiber.point_source_potentials(
        x=0.0, y=SRC_H, z=fiber.length / 2.0, i0=amp_mA, sigma=SIGMA,
    )
    waveform = lambda t: np.where((t >= DELAY) & (t < DELAY + pw_ms), 1.0, 0.0)
    stim = ScaledStim(waveform=waveform, dt=DT, tstop=tstop_ms)

    captured_stdout = io.StringIO()
    with contextlib.redirect_stdout(captured_stdout):
        stim.run_sim(stimamp=1.0, fiber=fiber, ap_detect_location=0.5,
                     fail_on_end_excitation=False)

    stdout_txt = captured_stdout.getvalue()
    warnings_found = [ln for ln in stdout_txt.splitlines()
                      if "warn" in ln.lower() or "else" in ln.lower()
                      or "not at detect" in ln.lower() or "AP" in ln]

    vm = np.array([np.array(v) for v in fiber.vm])  # [n_nodes, n_steps]
    peak_per_node = vm.max(axis=1)                    # [n_nodes]

    center_aps = fiber.apc[probe_idx].n
    end1_aps   = fiber.apc[0].n
    end2_aps   = fiber.apc[fiber.nodecount - 1].n

    any_fires   = bool(peak_per_node.max() > V_THRESH)
    center_fires = bool(center_aps > 0)

    return {
        "label": label,
        "amp_mA": float(amp_mA),
        "center_fires": center_fires,
        "center_aps": int(center_aps),
        "end1_aps": int(end1_aps),
        "end2_aps": int(end2_aps),
        "any_fires": any_fires,
        "center_peak_mV": float(peak_per_node[probe_idx]),
        "end1_peak_mV": float(peak_per_node[0]),
        "end2_peak_mV": float(peak_per_node[-1]),
        "peak_per_node": peak_per_node,
        "vm": vm,
        "warnings": warnings_found,
        "stdout": stdout_txt,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_spatiotemporal(cases, title, path, v_range=(-100, 60)):
    """Plot spatiotemporal Vm maps for a list of (jax_diag, pf_diag) tuples."""
    n_cases = len(cases)
    fig, axes = plt.subplots(n_cases, 2, figsize=(14, 4 * n_cases),
                             constrained_layout=True)
    if n_cases == 1:
        axes = axes[np.newaxis, :]

    for row, (jd, pd) in enumerate(cases):
        amp = jd["amp_mA"]

        # JAX heatmap (node-only compartments)
        ax = axes[row, 0]
        Vm_n = jd["Vm_nodes"]    # [n_steps, n_nodes]
        im = ax.imshow(Vm_n.T, aspect="auto", origin="lower",
                       extent=[jd["t_arr"][0], jd["t_arr"][-1],
                               0, Vm_n.shape[1]],
                       vmin=v_range[0], vmax=v_range[1], cmap="RdBu_r")
        plt.colorbar(im, ax=ax, label="Vm (mV)")
        # Mark nodes that exceed threshold (in time)
        for ni, pk in enumerate(jd["peak_per_node"]):
            if pk > V_THRESH:
                ax.axhline(ni, color="yellow", lw=0.5, alpha=0.5)
        # Mark center node
        ctr_local = jd["argmax_node"]  # use argmax for now, recompute below
        ax.set_title(f"JAX  {amp:+.3f} mA  "
                     f"any={jd['any_node_fires']}  center={jd['center_fires']}  "
                     f"firing={jd['n_nodes_firing']}")
        ax.set_xlabel("time (ms)"); ax.set_ylabel("node index")

        # PF heatmap (node Vm)
        ax = axes[row, 1]
        vm_pf = pd["vm"]   # [n_nodes, n_t_pf]
        if vm_pf is not None and vm_pf.shape[0] > 0:
            n_t = vm_pf.shape[1]
            t_pf = np.linspace(0, TSTOP, n_t)
            im2 = ax.imshow(vm_pf, aspect="auto", origin="lower",
                            extent=[t_pf[0], t_pf[-1], 0, vm_pf.shape[0]],
                            vmin=v_range[0], vmax=v_range[1], cmap="RdBu_r")
            plt.colorbar(im2, ax=ax, label="Vm (mV)")
        ax.set_title(f"PF  {amp:+.3f} mA  "
                     f"center={pd['center_fires']}  any={pd['any_fires']}  "
                     f"ctr_APs={pd['center_aps']}  end1={pd['end1_aps']}  end2={pd['end2_aps']}"
                     + (f"\n  ⚠ {pd['warnings'][0][:60]}" if pd["warnings"] else ""))
        ax.set_xlabel("time (ms)"); ax.set_ylabel("node index")

    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def print_table(cases, header=""):
    """Print comparison table."""
    if header:
        print(f"\n{'='*70}")
        print(header)
        print(f"{'='*70}")
    print(f"{'amp_mA':>10}  {'JAX_any':>8}  {'JAX_ctr':>8}  {'JAX_n_nodes':>12}  "
          f"{'JAX_maxVm':>10}  {'PF_ctr':>8}  {'PF_any':>8}  {'PF_warn':>8}")
    print("-" * 80)
    for jd, pd in cases:
        warn_flag = "YES" if pd["warnings"] else "no"
        print(f"  {jd['amp_mA']:+8.3f}  "
              f"  {str(jd['any_node_fires']):>7}  "
              f"  {str(jd['center_fires']):>7}  "
              f"  {jd['n_nodes_firing']:>11}  "
              f"  {jd['max_Vm_mV']:>9.1f}  "
              f"  {str(pd['center_fires']):>7}  "
              f"  {str(pd['any_fires']):>7}  "
              f"  {warn_flag:>8}")
    print("-" * 80)


# ── Case runner ───────────────────────────────────────────────────────────────

def run_case(label, jax_setup_fn, pf_fiber_fn, pulse_key, pw_ms,
             test_amps, n_steps_jax, v_rest=V_REST):
    """Run full diagnostics for one (model, D, pulse, PW) case."""
    print(f"\n{'#'*60}")
    print(f"  CASE: {label}")
    print(f"  pulse={pulse_key}  PW={pw_ms} ms")
    print(f"  test amps: {test_amps}")
    print(f"{'#'*60}")

    static, mfn, state0, Ve_unit, nodes, centers, mid, geom = jax_setup_fn()
    is_node = static["is_node"]

    pulse_arr = make_pulse_array(pulse_key, pw_ms, n_steps_jax, DT, DELAY)

    cases = []
    for amp in test_amps:
        print(f"\n  amp = {amp:+.4f} mA", flush=True)

        # JAX
        t0 = time.time()
        jd = jax_diagnostics(static, mfn, state0, Ve_unit, nodes, centers, mid,
                              is_node, amp, pulse_arr, label=f"JAX {amp:+.3f} mA",
                              v_rest=v_rest)
        t_jax = time.time() - t0
        print(f"    JAX: any={jd['any_node_fires']} center={jd['center_fires']} "
              f"n_firing={jd['n_nodes_firing']} maxVm={jd['max_Vm_mV']:.1f} mV "
              f"({t_jax:.1f}s)", flush=True)

        # PF
        t0 = time.time()
        pd = pf_diagnostics(pf_fiber_fn, amp, pw_ms, label=f"PF {amp:+.3f} mA")
        t_pf = time.time() - t0
        print(f"    PF : center={pd['center_fires']} (APc={pd['center_aps']}) "
              f"end1={pd['end1_aps']} end2={pd['end2_aps']} "
              f"any={pd['any_fires']} maxVm={pd['peak_per_node'].max():.1f} mV "
              f"({t_pf:.1f}s)", flush=True)
        if pd["warnings"]:
            for w in pd["warnings"]:
                print(f"    ⚠  PF warning: {w}", flush=True)

        cases.append((jd, pd))

    print_table(cases, header=label)
    return cases


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    n_steps_mrg    = int(TSTOP / DT)
    n_steps_rattay = int(TSTOP / DT)

    # ── CASE A: MRG mono_a PW=0.20 ms — BIG discrepancy ─────────────────────
    D_mrg = 5.7
    amps_a = [0.3, 0.51, 0.9, 2.0, 5.0, 10.0, 19.33, 25.0]  # anodic (positive)
    cases_a = run_case(
        label=f"MRG D={D_mrg} mono_a PW=0.20 ms",
        jax_setup_fn=lambda: _make_mrg_setup(D_mrg),
        pf_fiber_fn=lambda: build_mrg_pyfibers(diameter=D_mrg, n_nodes=21),
        pulse_key="mono_a",
        pw_ms=0.20,
        test_amps=amps_a,
        n_steps_jax=n_steps_mrg,
    )
    plot_spatiotemporal(
        cases_a[:4], f"MRG D={D_mrg} mono_a PW=0.20ms (low amps)",
        OUT / "spatio_mrg_mono_a_020_low.png",
    )
    plot_spatiotemporal(
        cases_a[4:], f"MRG D={D_mrg} mono_a PW=0.20ms (high amps)",
        OUT / "spatio_mrg_mono_a_020_high.png",
    )

    # ── CASE B: MRG mono_a PW=0.10 ms — control (both agree) ─────────────────
    amps_b = [0.5, 0.91, 1.2, 2.0]
    cases_b = run_case(
        label=f"MRG D={D_mrg} mono_a PW=0.10 ms (control)",
        jax_setup_fn=lambda: _make_mrg_setup(D_mrg),
        pf_fiber_fn=lambda: build_mrg_pyfibers(diameter=D_mrg, n_nodes=21),
        pulse_key="mono_a",
        pw_ms=0.10,
        test_amps=amps_b,
        n_steps_jax=n_steps_mrg,
    )
    plot_spatiotemporal(
        cases_b, f"MRG D={D_mrg} mono_a PW=0.10ms (control)",
        OUT / "spatio_mrg_mono_a_010_control.png",
    )

    # ── CASE C: Rattay bi_ca PW=0.05 ms — short biphasic ─────────────────────
    D_rat = 0.3
    amps_c = [-50.0, -85.8, -101.8, -120.0]  # cathodic (negative)
    cases_c = run_case(
        label=f"Rattay D={D_rat} bi_ca PW=0.05 ms",
        jax_setup_fn=lambda: _make_rattay_setup(D_rat),
        pf_fiber_fn=lambda: build_rattay_pyfibers(diameter=D_rat, n_nodes=51),
        pulse_key="bi_ca",
        pw_ms=0.05,
        test_amps=amps_c,
        n_steps_jax=n_steps_rattay,
        v_rest=RATTAY_V_REST,
    )
    plot_spatiotemporal(
        cases_c, f"Rattay D={D_rat} bi_ca PW=0.05ms",
        OUT / "spatio_rattay_bica_005.png",
    )

    # ── CASE D: MRG mono_a PW=0.50 ms — check if pattern persists ────────────
    amps_d = [0.2, 0.28, 0.5, 1.0, 3.0, 4.71]
    cases_d = run_case(
        label=f"MRG D={D_mrg} mono_a PW=0.50 ms",
        jax_setup_fn=lambda: _make_mrg_setup(D_mrg),
        pf_fiber_fn=lambda: build_mrg_pyfibers(diameter=D_mrg, n_nodes=21),
        pulse_key="mono_a",
        pw_ms=0.50,
        test_amps=amps_d,
        n_steps_jax=n_steps_mrg,
    )
    plot_spatiotemporal(
        cases_d[:3], f"MRG D={D_mrg} mono_a PW=0.50ms (low amps)",
        OUT / "spatio_mrg_mono_a_050_low.png",
    )
    plot_spatiotemporal(
        cases_d[3:], f"MRG D={D_mrg} mono_a PW=0.50ms (high amps)",
        OUT / "spatio_mrg_mono_a_050_high.png",
    )

    print(f"\n\nSmoke test complete. Figures in: {OUT}")


if __name__ == "__main__":
    main()
