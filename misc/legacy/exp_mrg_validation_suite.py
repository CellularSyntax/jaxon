"""MRG Validation Suite — Tasks 1.1–1.4 (Phase 1, migration_plan.md v3)

Four validation checks comparing JAX coupled (Vi, Vpax) solver vs PyFibers/NEURON:

  Task 1.1  Strength-duration curves — threshold vs pulse width
            D = {5.7, 10.0, 14.0} µm, PW = {0.02, 0.05, 0.1, 0.2, 0.5, 1.0} ms

  Task 1.2  Conduction velocity — AP propagation speed vs diameter
            All 9 MRG discrete diameters, 1.3× suprathreshold amplitude

  Task 1.3  Biphasic stimulation — cathodic-first symmetric biphasic vs monophasic
            D = 10.0 µm, PW = {0.05, 0.1, 0.2} ms per phase

  Task 1.4  AP propagation check — verify all nodes fire at 1.3× threshold
            D = 10.0 µm (qualitative, printed only)

Saves:
  outputs/data_mrg_sd_curves.json
  outputs/data_mrg_cv.json
  outputs/data_mrg_biphasic.json
  outputs/fig_mrg_validation.png
  outputs/table_mrg_validation.csv

Run from project root:
    conda run -n jaxley_fibers python experiments/exp_mrg_validation_suite.py
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

from jaxfibers.fibers.mrg import (
    build_mrg, node_indices, section_centers_um,
    V_REST, CM_AXON, G_PAS_MYSA, G_PAS_FLUT, G_PAS_STIN,
)
from jaxfibers.channels.mrg_axnode import AxnodeMyel
from jaxfibers.stim.extracellular import point_source_potentials_mV
from jaxfibers.nrn_baseline import run_extracellular, find_threshold_extracellular
from mrg_extracellular_coupled import arrays_from_geometry, integrate

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# ── shared constants ──────────────────────────────────────────────────────────
CELSIUS   = 37.0
DT        = 0.005   # ms
TSTOP     = 5.0     # ms
DELAY     = 1.0     # ms
N_NODES   = 21
N_STEPS   = int(TSTOP / DT)  # 1000
SIGMA     = 0.3     # S/m

GNABAR = 3.0;  GNAPBAR = 0.01; GKBAR = 0.08; GL   = 0.007
ENA    = 50.0; EK      = -90.0; EL   = -90.0

SD_DIAMETERS = [5.7, 10.0, 14.0]
SD_PWS       = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]  # ms
CV_DIAMETERS = [5.7, 7.3, 8.7, 10.0, 11.5, 12.8, 14.0, 15.0, 16.0]
BI_PWS       = [0.05, 0.1, 0.2]  # ms per phase

# Load pre-computed thresholds (need suprathreshold amplitude for CV)
with open(OUT / "pyfibers_thresholds.json") as f:
    PF_THR = {float(k): v for k, v in json.load(f).items()}


# ── JAX geometry + membrane setup (rebuilt once per diameter) ─────────────────
def _make_jax_setup(D: float, N: int = N_NODES):
    _, geom = build_mrg(diameter=D, n_nodes=N)
    nodes   = node_indices(geom)
    centers = np.array(section_centers_um(geom))
    mid     = nodes[len(nodes) // 2]
    n_comp  = geom.n_comp

    geom_c = dataclasses.replace(geom, cm_uF_cm2=[CM_AXON] * n_comp)
    static  = arrays_from_geometry(geom_c, DT)

    Ve_unit = np.array(point_source_potentials_mV(
        list(centers), src_x_um=0., src_y_um=1000.,
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
        i_node = (
            (g_na + g_nap) * (Vm - ENA) + g_k * (Vm - EK) + GL * (Vm - EL)
        ) * A_in * 1e6
        g_pas_us = g_pas_arr * A_in * 1e6
        i_pas    = g_pas_us  * (Vm - V_REST)
        g_eff = jnp.where(is_node, g_node,  g_pas_us)
        i_ion = jnp.where(is_node, i_node,  i_pas)
        return g_eff, i_ion, (M2, H2, MP2, S2)

    return static, membrane_fn, state0, Ve_unit, mid, centers, nodes, geom


def _make_jit_runner(static, mfn, state0, Ve_unit, v_thresh=-30.0):
    """Compile forward pass once per diameter. run(amp_mA, pulse_shape) -> peak Vm."""
    Ve_j = jnp.asarray(Ve_unit, dtype=jnp.float64)

    @jax.jit
    def run(amp_mA, pulse_shape):
        Ve = Ve_j * (amp_mA / -1.0)
        trace, _ = integrate(static, mfn, state0, Ve, pulse_shape, DT,
                             v_rest=V_REST, record="center")
        return jnp.max(trace)

    return run


def _bisect(run_fn, pulse_shape, lo=-1.0, hi=-0.001, tol=1e-4, v_thresh=-30.0):
    pm = jnp.asarray(pulse_shape, dtype=jnp.float64)
    def fires(a): return bool(run_fn(jnp.float64(a), pm) > v_thresh)
    # Auto-expand lo (short-pulse threshold can exceed 1 mA for small diameters)
    for _ in range(5):
        if fires(lo): break
        lo *= 2.0
    if not fires(lo):
        raise ValueError(f"Fiber did not fire at {lo} mA after expansion")
    if fires(hi):
        raise ValueError(f"Fiber fires at sub-threshold bound {hi} mA")
    fire_hi = fires(hi)
    while abs(hi - lo) > tol:
        mid = 0.5 * (lo + hi)
        if fires(mid) == fire_hi:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _mono_shape(pw: float) -> np.ndarray:
    t = (np.arange(N_STEPS) + 1) * DT
    return ((t >= DELAY) & (t < DELAY + pw)).astype(np.float64)


def _biphasic_shape(pw: float) -> np.ndarray:
    t = (np.arange(N_STEPS) + 1) * DT
    return (
        ((t >= DELAY) & (t < DELAY + pw)).astype(np.float64) -
        ((t >= DELAY + pw) & (t < DELAY + 2 * pw)).astype(np.float64)
    )


# ── Task 1.1: Strength-duration curves ───────────────────────────────────────
def run_sd_curves() -> dict:
    print("\n=== Task 1.1: Strength-duration curves ===")
    results = {}
    for D in SD_DIAMETERS:
        print(f"  D = {D} µm", flush=True)
        static, mfn, state0, Ve_unit, *_ = _make_jax_setup(D)
        runner = _make_jit_runner(static, mfn, state0, Ve_unit)
        results[D] = {}
        for pw in SD_PWS:
            t0 = time.time()
            pm = _mono_shape(pw)
            jax_thr = _bisect(runner, pm)
            t_jax = time.time() - t0

            t0 = time.time()
            nrn_thr = find_threshold_extracellular(
                D, N_NODES, pw_ms=pw, delay_ms=DELAY, dt_ms=DT, tstop_ms=TSTOP,
                bounds_mA=(-1.0, -0.001),
            )
            t_nrn = time.time() - t0

            err = (abs(jax_thr) - abs(nrn_thr)) / abs(nrn_thr) * 100
            results[D][pw] = {"jax": float(jax_thr), "neuron": float(nrn_thr), "err_pct": float(err)}
            print(f"    PW={pw:.2f} ms | NEURON={nrn_thr:.5f} | JAX={jax_thr:.5f} | "
                  f"err={err:+.2f}% | t={t_jax:.1f}/{t_nrn:.1f} s", flush=True)

    with open(OUT / "data_mrg_sd_curves.json", "w") as f:
        json.dump({str(d): {str(pw): v for pw, v in dv.items()} for d, dv in results.items()}, f, indent=2)
    return results


# ── Task 1.2: Conduction velocity ─────────────────────────────────────────────
def _cv_from_jax(static, mfn, state0, Ve_unit, amp_mA, nodes, centers) -> float:
    """Measure CV by timing the AP between the centre node and a near-end node.
    Both on the same 'side' avoids the div/0 that occurs when equidistant nodes
    fire simultaneously (symmetric cathodic stimulation from the centre)."""
    pm = jnp.asarray(_mono_shape(0.1))
    Ve_scaled = jnp.asarray(Ve_unit * (amp_mA / -1.0))
    trace, _ = integrate(static, mfn, state0, Ve_scaled, pm, DT, v_rest=V_REST, record="all")
    Vi_all, Vp_all = trace
    Vm_all = np.asarray(Vi_all) - np.asarray(Vp_all)  # (N_STEPS, n_comp)
    t_steps = (np.arange(N_STEPS) + 1) * DT

    k_ctr = len(nodes) // 2   # centre node — fires first
    k_end = 2                  # near the left end — fires later
    onset = int(DELAY / DT)
    t_ctr = t_steps[onset + int(np.argmax(Vm_all[onset:, nodes[k_ctr]]))]
    t_end = t_steps[onset + int(np.argmax(Vm_all[onset:, nodes[k_end]]))]
    dist_um = abs(float(centers[nodes[k_ctr]]) - float(centers[nodes[k_end]]))
    return dist_um / abs(t_end - t_ctr) * 1e-3  # µm/ms → m/s


def _cv_from_neuron(nr, n_nodes, centers, nodes) -> float:
    k_ctr = n_nodes // 2
    k_end = 2
    onset_idx = int(np.searchsorted(nr.t_ms, DELAY))
    t_ctr = nr.t_ms[onset_idx + int(np.argmax(nr.vm_mV[k_ctr, onset_idx:]))]
    t_end = nr.t_ms[onset_idx + int(np.argmax(nr.vm_mV[k_end, onset_idx:]))]
    dist_um = abs(float(centers[nodes[k_ctr]]) - float(centers[nodes[k_end]]))
    return dist_um / abs(t_end - t_ctr) * 1e-3


def run_cv() -> dict:
    print("\n=== Task 1.2: Conduction velocity ===")
    results = {}
    for D in CV_DIAMETERS:
        amp = PF_THR[D] * 1.3  # 30% suprathreshold
        static, mfn, state0, Ve_unit, mid, centers, nodes, geom = _make_jax_setup(D)

        t0 = time.time()
        jax_cv = _cv_from_jax(static, mfn, state0, Ve_unit, amp, nodes, centers)
        t_jax = time.time() - t0

        t0 = time.time()
        nr = run_extracellular(D, N_NODES, src_height_um=1000., pw_ms=0.1,
                               delay_ms=DELAY, amp_mA=amp, dt_ms=DT, tstop_ms=TSTOP)
        nrn_cv = _cv_from_neuron(nr, N_NODES, centers, nodes)
        t_nrn = time.time() - t0

        err = (jax_cv - nrn_cv) / nrn_cv * 100
        results[D] = {"jax": float(jax_cv), "neuron": float(nrn_cv), "err_pct": float(err)}
        print(f"  D={D:.1f} µm | NEURON={nrn_cv:.1f} | JAX={jax_cv:.1f} m/s | "
              f"err={err:+.1f}% | t={t_jax:.1f}/{t_nrn:.1f} s", flush=True)

    with open(OUT / "data_mrg_cv.json", "w") as f:
        json.dump({str(d): v for d, v in results.items()}, f, indent=2)
    return results


# ── Task 1.3: Biphasic threshold ──────────────────────────────────────────────
def run_biphasic(D: float = 10.0) -> dict:
    print(f"\n=== Task 1.3: Biphasic threshold (D={D} µm) ===")
    static, mfn, state0, Ve_unit, *_ = _make_jax_setup(D)
    runner = _make_jit_runner(static, mfn, state0, Ve_unit)
    results = {}
    for pw in BI_PWS:
        # monophasic
        pm_mono = _mono_shape(pw)
        jax_mono = _bisect(runner, pm_mono)
        nrn_mono = find_threshold_extracellular(
            D, N_NODES, pw_ms=pw, delay_ms=DELAY, dt_ms=DT, tstop_ms=TSTOP,
            bounds_mA=(-1.0, -0.001), biphasic=False,
        )
        # biphasic (cathodic-first symmetric)
        pm_bi = _biphasic_shape(pw)
        jax_bi = _bisect(runner, pm_bi)
        nrn_bi = find_threshold_extracellular(
            D, N_NODES, pw_ms=pw, delay_ms=DELAY, dt_ms=DT, tstop_ms=TSTOP,
            bounds_mA=(-1.0, -0.001), biphasic=True,
        )
        err_mono = (abs(jax_mono) - abs(nrn_mono)) / abs(nrn_mono) * 100
        err_bi   = (abs(jax_bi)   - abs(nrn_bi))   / abs(nrn_bi)   * 100
        results[pw] = {
            "jax_mono": float(jax_mono), "neuron_mono": float(nrn_mono),
            "jax_bi":   float(jax_bi),   "neuron_bi":   float(nrn_bi),
            "err_mono_pct": float(err_mono), "err_bi_pct": float(err_bi),
        }
        print(f"  PW={pw:.2f} ms | mono: NEURON={nrn_mono:.5f} JAX={jax_mono:.5f} err={err_mono:+.2f}%"
              f" | bi:   NEURON={nrn_bi:.5f}   JAX={jax_bi:.5f}   err={err_bi:+.2f}%", flush=True)

    with open(OUT / "data_mrg_biphasic.json", "w") as f:
        json.dump({str(pw): v for pw, v in results.items()}, f, indent=2)
    return results


# ── Task 1.4: AP propagation check ───────────────────────────────────────────
def run_propagation_check(D: float = 10.0):
    print(f"\n=== Task 1.4: AP propagation check (D={D} µm) ===")
    amp = PF_THR[D] * 1.3
    static, mfn, state0, Ve_unit, mid, centers, nodes, _ = _make_jax_setup(D)
    pm = jnp.asarray(_mono_shape(0.1))
    Ve_scaled = jnp.asarray(Ve_unit * (amp / -1.0))
    trace, _ = integrate(static, mfn, state0, Ve_scaled, pm, DT, v_rest=V_REST, record="all")
    Vi_all, Vp_all = trace
    Vm_all = np.asarray(Vi_all) - np.asarray(Vp_all)  # (N_STEPS, n_comp)

    v_thresh = -30.0
    fired = [bool(np.max(Vm_all[:, nodes[k]]) > v_thresh) for k in range(len(nodes))]
    n_fired = sum(fired)
    print(f"  Amplitude: {amp:.4f} mA (~1.3× threshold)")
    print(f"  Nodes fired: {n_fired} / {len(nodes)}  {'PASS ✓' if n_fired == len(nodes) else 'FAIL ✗'}")
    for k, f in enumerate(fired):
        print(f"    node {k:2d}: {'fire' if f else 'no fire'}", flush=True)
    return {"n_nodes": len(nodes), "n_fired": n_fired, "all_fired": n_fired == len(nodes)}


# ── Figure ────────────────────────────────────────────────────────────────────
def make_figure(sd: dict, cv: dict, bi: dict):
    print("\n=== Generating figure ===", flush=True)
    colors = {5.7: "C0", 10.0: "C1", 14.0: "C3"}
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # (A) Strength-duration curves
    ax = axs[0, 0]
    for D, dv in sd.items():
        pws = sorted(dv.keys())
        nrn_thr = [abs(dv[pw]["neuron"]) for pw in pws]
        jax_thr = [abs(dv[pw]["jax"])    for pw in pws]
        c = colors[D]
        ax.plot(pws, nrn_thr, "o-",  color=c, lw=2.0, ms=5, label=f"NEURON D={D}µm")
        ax.plot(pws, jax_thr, "s--", color=c, lw=1.5, ms=4, label=f"JAX D={D}µm")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Pulse width (ms)"); ax.set_ylabel("|Threshold| (mA)")
    ax.set_title("(A) Strength-duration curves")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3, which="both")

    # (B) Conduction velocity
    ax = axs[0, 1]
    diams = sorted(cv.keys())
    nrn_cv = [cv[d]["neuron"] for d in diams]
    jax_cv = [cv[d]["jax"]    for d in diams]
    ax.plot(diams, nrn_cv, "o-",  color="C0", lw=2.0, ms=5, label="PyFibers / NEURON")
    ax.plot(diams, jax_cv, "s--", color="C3", lw=1.5, ms=4, label="JAX coupled solver")
    ax.set_xlabel("Fiber diameter (µm)"); ax.set_ylabel("Conduction velocity (m/s)")
    ax.set_title("(B) Conduction velocity vs diameter")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # (C) Biphasic vs monophasic
    ax = axs[1, 0]
    pws_bi = sorted(bi.keys())
    nrn_mono = [abs(bi[pw]["neuron_mono"]) for pw in pws_bi]
    jax_mono = [abs(bi[pw]["jax_mono"])    for pw in pws_bi]
    nrn_bi   = [abs(bi[pw]["neuron_bi"])   for pw in pws_bi]
    jax_bi   = [abs(bi[pw]["jax_bi"])      for pw in pws_bi]
    ax.plot(pws_bi, nrn_mono, "o-",  color="C0", lw=2.0, ms=5, label="NEURON mono")
    ax.plot(pws_bi, jax_mono, "o--", color="C0", lw=1.5, ms=4, label="JAX mono")
    ax.plot(pws_bi, nrn_bi,   "s-",  color="C2", lw=2.0, ms=5, label="NEURON biphasic")
    ax.plot(pws_bi, jax_bi,   "s--", color="C2", lw=1.5, ms=4, label="JAX biphasic")
    ax.set_xlabel("Phase width (ms)"); ax.set_ylabel("|Threshold| (mA)")
    ax.set_title("(C) Biphasic vs monophasic (D=10 µm)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # (D) Error summary
    ax = axs[1, 1]
    # SD errors (mean abs % per diameter)
    sd_labels, sd_mean_err = [], []
    for D, dv in sorted(sd.items()):
        errs = [abs(v["err_pct"]) for v in dv.values()]
        sd_labels.append(f"SD\nD={D}µm")
        sd_mean_err.append(np.mean(errs))
    # CV errors per diameter
    cv_labels = [f"CV\n{d}µm" for d in sorted(cv.keys())]
    cv_errs   = [abs(cv[d]["err_pct"]) for d in sorted(cv.keys())]
    # Biphasic errors
    bi_labels = [f"Bi\n{pw:.2f}ms" for pw in sorted(bi.keys())]
    bi_errs   = [abs(bi[pw]["err_bi_pct"]) for pw in sorted(bi.keys())]

    all_labels = sd_labels + cv_labels + bi_labels
    all_errs   = sd_mean_err + cv_errs + bi_errs
    xpos = np.arange(len(all_labels))
    bars = ax.bar(xpos, all_errs, color=(
        ["C0"] * len(sd_labels) + ["C1"] * len(cv_labels) + ["C2"] * len(bi_labels)
    ), alpha=0.8)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="1% guideline")
    ax.set_xticks(xpos); ax.set_xticklabels(all_labels, fontsize=7)
    ax.set_ylabel("Mean |error| (%)"); ax.set_title("(D) Error summary")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "MRG validation: JAX coupled (Vi, Vpax) solver vs PyFibers/NEURON\n"
        "All stimuli: point source 1 mm above centre, σ=0.3 S/m, N=21 nodes, T=37°C",
        fontsize=10,
    )
    fig.savefig(OUT / "fig_mrg_validation.png", dpi=150, bbox_inches="tight")
    print(f"  → {OUT / 'fig_mrg_validation.png'}")


# ── CSV summary table ─────────────────────────────────────────────────────────
def write_csv(sd: dict, cv: dict, bi: dict):
    path = OUT / "table_mrg_validation.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "diameter_um", "pw_ms", "neuron_mA_or_ms",
                    "jax_mA_or_ms", "abs_err_pct"])
        for D, dv in sorted(sd.items()):
            for pw, v in sorted(dv.items()):
                w.writerow(["SD", D, pw, f"{v['neuron']:.5f}", f"{v['jax']:.5f}",
                             f"{abs(v['err_pct']):.3f}"])
        for D in sorted(cv.keys()):
            v = cv[D]
            w.writerow(["CV", D, "0.1", f"{v['neuron']:.2f}", f"{v['jax']:.2f}",
                        f"{abs(v['err_pct']):.2f}"])
        for pw in sorted(bi.keys()):
            v = bi[pw]
            w.writerow(["BI-mono", 10.0, pw, f"{v['neuron_mono']:.5f}",
                        f"{v['jax_mono']:.5f}", f"{abs(v['err_mono_pct']):.3f}"])
            w.writerow(["BI-bi",   10.0, pw, f"{v['neuron_bi']:.5f}",
                        f"{v['jax_bi']:.5f}",   f"{abs(v['err_bi_pct']):.3f}"])
    print(f"  → {path}")


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t_total = time.time()
    sd_data   = run_sd_curves()
    cv_data   = run_cv()
    bi_data   = run_biphasic()
    prop_data = run_propagation_check()
    make_figure(sd_data, cv_data, bi_data)
    write_csv(sd_data, cv_data, bi_data)
    print(f"\nDone. Total wall time: {(time.time()-t_total)/60:.1f} min")
