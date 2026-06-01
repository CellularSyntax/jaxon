"""Performance benchmark: PyFibers/NEURON vs JAX coupled solver.

Measures wall-clock time for N independent extracellular forward simulations
(rectangular pulse at 1.3× threshold, record=center) for:
  - MRG   D=10 µm,  21 nodes, 5 ms sim,  dt=0.005 ms
  - Sundt D=0.8 µm, 51 nodes, 10 ms sim, dt=0.005 ms
  (Tigerholm: add entry to MODELS once Task 3 is complete)

JAX timing
  Single `jax.jit(jax.vmap(single_run))` vmapped over N amplitudes
  (±5 % jitter around the benchmark amplitude — N genuinely distinct runs).
  For each N: (a) first call (JIT compile + execute), (b) mean of N_REPEATS
  subsequent calls (cached compilation / inference).

PyFibers timing
  Serial Python loop over run_extracellular / run_extracellular_sundt,
  measured up to PF_BUDGET seconds total per model; extrapolated beyond.

Outputs
  outputs/fig_performance_benchmark.png  — 2×2 figure
  outputs/data_performance_benchmark.json
  outputs/table_performance_benchmark.csv

Run from project root:
    conda run -n jaxley_fibers python experiments/exp_performance_benchmark.py
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

# ── MRG imports (exact same as exp_mrg_validation_suite) ─────────────────────
from jaxfibers.fibers.mrg import (
    build_mrg, node_indices as mrg_node_indices, section_centers_um as mrg_centers,
    V_REST as MRG_VREST, CM_AXON,
    G_PAS_MYSA, G_PAS_FLUT, G_PAS_STIN,
)
from jaxfibers.channels.mrg_axnode import AxnodeMyel

# ── Sundt imports (exact same as exp_sundt_validation) ────────────────────────
from jaxfibers.fibers.sundt import (
    build_sundt, node_indices as sundt_node_indices, section_centers_um as sundt_centers,
    V_REST as SUNDT_VREST,
)

from jaxfibers.stim.extracellular import point_source_potentials_mV
from jaxfibers.nrn_baseline import (
    run_extracellular, find_threshold_extracellular,
    run_extracellular_sundt, find_threshold_extracellular_sundt,
)
from mrg_extracellular_coupled import arrays_from_geometry, integrate

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# ── benchmark parameters ──────────────────────────────────────────────────────
JAX_NS    = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
PF_NS     = [1, 2, 5, 10, 20, 30, 50]          # measured; script halts on budget
PF_BUDGET = 120.0                               # seconds per model
N_REPEATS = 3                                   # JAX inference repeats
PW_MS     = 0.1
DELAY_MS  = 1.0
HEIGHT_UM = 1000.0
SIGMA     = 0.3

# MRG constants (from validation suite)
MRG_CELSIUS = 37.0
MRG_GNABAR  = 3.0;  MRG_GNAPBAR = 0.01; MRG_GKBAR = 0.08; MRG_GL = 0.007
MRG_ENA     = 50.0; MRG_EK      = -90.0; MRG_EL   = -90.0

# Sundt constants (from validation suite)
SUNDT_CELSIUS = 37.0
_SGNA=0.04; _SGKDR=0.04; _SGPAS=1e-4
_SENA=50.0; _SEK=-90.0; _SEPAS=-60.0
_MSHIFT=-6.0; _HSHIFT=6.0; _ISHIFT=0.0
_VHALFN=-32.0; _VHALFL=-61.0
_A0N=0.03; _A0L=0.001; _ZETAN=-5.0; _ZETAL=2.0; _GMN=0.4; _GML=1.0


# ── JAX setup helpers (verbatim from validation suites) ───────────────────────

def _make_jax_setup_mrg(diameter=10.0, n_nodes=21):
    """Exact copy of exp_mrg_validation_suite._make_jax_setup."""
    _, geom   = build_mrg(diameter=diameter, n_nodes=n_nodes)
    nodes     = mrg_node_indices(geom)
    centers   = np.array(mrg_centers(geom))
    n_comp    = geom.n_comp
    mid       = nodes[len(nodes) // 2]

    geom_c  = dataclasses.replace(geom, cm_uF_cm2=[CM_AXON] * n_comp)
    dt      = 0.005
    static  = arrays_from_geometry(geom_c, dt)

    Ve_unit = np.array(point_source_potentials_mV(
        list(centers), src_x_um=0., src_y_um=HEIGHT_UM,
        src_z_um=float(centers[mid]), i0_mA=-1.0,
    ))

    is_node   = static["is_node"]
    A_in      = static["A_in_cm2"]
    stype_gp  = {"node": 0.0, "mysa": G_PAS_MYSA, "flut": G_PAS_FLUT, "stin": G_PAS_STIN}
    g_pas_arr = jnp.asarray([stype_gp[s] for s in geom.section_type])

    v0 = jnp.float64(MRG_VREST)
    (a_m0,b_m0),(a_h0,b_h0),(a_mp0,b_mp0),(a_s0,b_s0) = AxnodeMyel._alpha_beta(v0, MRG_CELSIUS)
    m0  = float(a_m0  / (a_m0  + b_m0));  h0  = float(a_h0  / (a_h0  + b_h0))
    mp0 = float(a_mp0 / (a_mp0 + b_mp0)); s0  = float(a_s0  / (a_s0  + b_s0))
    state0 = (
        jnp.where(is_node, m0,  0.0),
        jnp.where(is_node, h0,  0.0),
        jnp.where(is_node, mp0, 0.0),
        jnp.where(is_node, s0,  0.0),
    )

    def membrane_fn(Vm, state, dt_):
        M, H, MP, S = state
        (a_m,b_m),(a_h,b_h),(a_mp,b_mp),(a_s,b_s) = AxnodeMyel._alpha_beta(Vm, MRG_CELSIUS)
        M2  = solve_gate_exponential(M,  dt_, a_m,  b_m)
        H2  = solve_gate_exponential(H,  dt_, a_h,  b_h)
        MP2 = solve_gate_exponential(MP, dt_, a_mp, b_mp)
        S2  = solve_gate_exponential(S,  dt_, a_s,  b_s)
        g_na  = MRG_GNABAR  * M2**3 * H2
        g_nap = MRG_GNAPBAR * MP2**3
        g_k   = MRG_GKBAR   * S2
        g_node = (g_na + g_nap + g_k + MRG_GL) * A_in * 1e6
        i_node = ((g_na + g_nap) * (Vm - MRG_ENA) + g_k * (Vm - MRG_EK)
                  + MRG_GL * (Vm - MRG_EL)) * A_in * 1e6
        g_pas_us = g_pas_arr * A_in * 1e6
        i_pas    = g_pas_us  * (Vm - MRG_VREST)
        return (jnp.where(is_node, g_node, g_pas_us),
                jnp.where(is_node, i_node, i_pas),
                (M2, H2, MP2, S2))

    return static, membrane_fn, state0, Ve_unit, mid, dt


def _make_jax_setup_sundt(diameter=0.8, n_nodes=51):
    """Exact copy of exp_sundt_validation._make_jax_setup."""
    _, geom  = build_sundt(diameter=diameter, n_nodes=n_nodes)
    centers  = np.array(sundt_centers(geom))
    n_comp   = geom.n_comp
    mid      = n_comp // 2

    dt = 0.005
    static = arrays_from_geometry(geom, dt)
    A_in   = static["A_in_cm2"]

    Ve_unit = np.array(point_source_potentials_mV(
        list(centers), src_x_um=0., src_y_um=HEIGHT_UM,
        src_z_um=float(centers[mid]), i0_mA=-1.0,
    ))

    def _expM1(x, y):
        ratio = x / y
        exp_term = jnp.exp(jnp.clip(ratio, -50.0, 50.0)) - 1.0
        safe = jnp.where(jnp.abs(exp_term) < 1e-30, 1e-30, exp_term)
        return jnp.where(jnp.abs(ratio) < 1e-6, y * (1.0 - 0.5 * ratio), x / safe)

    def _nahh(v):
        q10 = 3.0 ** ((SUNDT_CELSIUS - 30.0) / 10.0)
        vm = v + 65.0 + _MSHIFT
        a_m = q10 * 0.32 * _expM1(13.1 - vm, 4.0)
        b_m = q10 * 0.28 * _expM1(vm - 40.1, 5.0)
        vh  = v + 65.0 + _HSHIFT
        a_h = q10 * 0.128 * jnp.exp((17.0 - vh + _ISHIFT) / 18.0)
        b_h = q10 * 4.0 / (jnp.exp((40.0 - vh) / 5.0) + 1.0)
        return (a_m, b_m), (a_h, b_h)

    def _borgkdr(v):
        q10  = 3.0 ** ((SUNDT_CELSIUS - 30.0) / 10.0)
        FRT  = 9.648e4 / (8.315 * (273.16 + SUNDT_CELSIUS)) * 1e-3
        alpn = jnp.exp(_ZETAN * (v - _VHALFN) * FRT)
        betn = jnp.exp(_ZETAN * _GMN * (v - _VHALFN) * FRT)
        alpl = jnp.exp(_ZETAL * (v - _VHALFL) * FRT)
        betl = jnp.exp(_ZETAL * _GML * (v - _VHALFL) * FRT)
        return (q10 * _A0N / betn, q10 * _A0N * alpn / betn), \
               (q10 * _A0L / betl, q10 * _A0L * alpl / betl)

    def membrane_fn(Vm, state, dt_):
        M, H, N, L = state
        (am,bm),(ah,bh) = _nahh(Vm)
        (an,bn),(al,bl) = _borgkdr(Vm)
        M2 = solve_gate_exponential(M, dt_, am, bm)
        H2 = solve_gate_exponential(H, dt_, ah, bh)
        N2 = solve_gate_exponential(N, dt_, an, bn)
        L2 = solve_gate_exponential(L, dt_, al, bl)
        gna   = _SGNA * M2**3 * H2; gk = _SGKDR * N2**3 * L2
        g_eff = (gna + gk + _SGPAS) * A_in * 1e6
        i_ion = (gna * (Vm - _SENA) + gk * (Vm - _SEK) + _SGPAS * (Vm - _SEPAS)) * A_in * 1e6
        return g_eff, i_ion, (M2, H2, N2, L2)

    v0 = jnp.float64(SUNDT_VREST)
    (am0,bm0),(ah0,bh0) = _nahh(v0); (an0,bn0),(al0,bl0) = _borgkdr(v0)
    m0 = float(am0/(am0+bm0)); h0 = float(ah0/(ah0+bh0))
    n0 = float(an0/(an0+bn0)); l0 = float(al0/(al0+bl0))
    state0 = (jnp.full(n_comp, m0, dtype=jnp.float64),
              jnp.full(n_comp, h0, dtype=jnp.float64),
              jnp.full(n_comp, n0, dtype=jnp.float64),
              jnp.full(n_comp, l0, dtype=jnp.float64))

    return static, membrane_fn, state0, Ve_unit, mid, dt


# ── bisection ─────────────────────────────────────────────────────────────────

def _bisect(static, mfn, state0, Ve_unit, v_rest, dt, tstop_ms,
            lo=-5.0, hi=-0.001, tol=1e-4):
    n_steps = int(tstop_ms / dt)
    t_vec   = (np.arange(n_steps) + 1) * dt
    shape   = jnp.asarray(((t_vec >= DELAY_MS) & (t_vec < DELAY_MS + PW_MS)).astype(np.float64))
    Ve_j    = jnp.asarray(Ve_unit, dtype=jnp.float64)

    @jax.jit
    def run(a): return jnp.max(integrate(static, mfn, state0, Ve_j * (a / -1.0),
                                         shape, dt, v_rest=v_rest)[0])
    def fires(a): return bool(run(jnp.float64(a)) > -30.0)
    for _ in range(12):
        if fires(lo): break
        lo *= 2.0
    while abs(hi - lo) > tol:
        mid = 0.5 * (lo + hi)
        if fires(mid) == fires(hi): hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)


# ── JAX population timing ─────────────────────────────────────────────────────

def time_jax(static, mfn, state0, Ve_unit, v_rest, amp_ref, dt, tstop_ms,
             ns: list[int]) -> dict:
    n_steps = int(tstop_ms / dt)
    t_vec   = (np.arange(n_steps) + 1) * dt
    shape   = jnp.asarray(((t_vec >= DELAY_MS) & (t_vec < DELAY_MS + PW_MS)).astype(np.float64))
    Ve_j    = jnp.asarray(Ve_unit, dtype=jnp.float64)

    def single(amp_mA):
        Ve = Ve_j * (amp_mA / -1.0)
        trace, _ = integrate(static, mfn, state0, Ve, shape, dt, v_rest=v_rest)
        return jnp.max(trace)

    batched = jax.jit(jax.vmap(single))
    results = {}
    for N in ns:
        amps = jnp.linspace(0.95 * amp_ref, 1.05 * amp_ref, N, dtype=jnp.float64)
        t0 = time.perf_counter()
        _ = batched(amps).block_until_ready()
        t_first = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(N_REPEATS):
            _ = batched(amps).block_until_ready()
        t_run = (time.perf_counter() - t0) / N_REPEATS
        results[N] = {"compile_run_s": t_first, "run_s": t_run}
        print(f"    N={N:5d}: first={t_first:.3f}s  run={t_run:.3f}s  "
              f"per-fiber={t_run/N*1e3:.2f} ms", flush=True)
    return results


# ── PyFibers serial timing ────────────────────────────────────────────────────

def time_pyfibers(nrn_run_fn, amp_ref: float, ns: list[int]) -> dict:
    results   = {}
    cumulative = 0.0
    for N in ns:
        if cumulative >= PF_BUDGET:
            print(f"    PyFibers: budget ({PF_BUDGET}s) reached; stopping.", flush=True)
            break
        t0 = time.perf_counter()
        for _ in range(N):
            nrn_run_fn(amp_ref)
        t = time.perf_counter() - t0
        cumulative += t
        results[N] = {"total_s": t}
        print(f"    N={N:5d}: {t:.2f}s  per-fiber={t/N*1e3:.1f} ms", flush=True)
    return results


# ── figures ───────────────────────────────────────────────────────────────────

def make_figure(results: dict):
    mnames  = list(results.keys())
    n_mod   = len(mnames)
    fig, axs = plt.subplots(2, n_mod, figsize=(6.5 * n_mod, 10),
                            constrained_layout=True)
    if n_mod == 1:
        axs = axs[:, None]

    for col, mname in enumerate(mnames):
        res   = results[mname]
        jd    = res["jax"]
        pfd   = res["pyfibers"]
        thr   = res["threshold_jax_mA"]

        jax_ns    = sorted(jd.keys())
        jax_first = [jd[N]["compile_run_s"] for N in jax_ns]
        jax_run   = [jd[N]["run_s"]         for N in jax_ns]

        pf_ns  = sorted(pfd.keys())
        pf_tot = [pfd[N]["total_s"] for N in pf_ns]

        # extrapolate PyFibers linearly from last measured point
        pf_rate = (pfd[pf_ns[-1]]["total_s"] / pf_ns[-1]) if pf_ns else None
        pf_ext_ns = [N for N in jax_ns if N not in pfd]
        pf_ext_ts = [pf_rate * N for N in pf_ext_ns] if pf_rate else []

        # ── top: wall-clock time vs N
        ax = axs[0, col]
        ax.loglog(jax_ns, jax_first, "^--", color="C3", lw=1.5, ms=5,
                  label="JAX: 1st call (incl. JIT)")
        ax.loglog(jax_ns, jax_run,   "s-",  color="C2", lw=2.0, ms=6,
                  label="JAX: run (cached JIT)")
        if pf_ns:
            ax.loglog(pf_ns, pf_tot, "o-", color="C0", lw=2.0, ms=6,
                      label="PyFibers (measured)")
        if pf_ext_ns:
            link_ns = ([pf_ns[-1]] if pf_ns else []) + pf_ext_ns
            link_ts = ([pf_tot[-1]] if pf_ns else []) + pf_ext_ts
            ax.loglog(link_ns, link_ts, "o--", color="C0", lw=1.5,
                      ms=4, alpha=0.4, label="PyFibers (extrapolated)")
        ax.set_xlabel("N fibers")
        ax.set_ylabel("Wall-clock time (s)")
        ax.set_title(f"(A{col+1}) {mname}\nextracellular, PW=0.1 ms, 1.3× thr "
                     f"({abs(thr):.3f} mA)")
        ax.legend(fontsize=8.5); ax.grid(True, which="both", alpha=0.25)

        # ── bottom: per-fiber time + speedup
        ax2 = axs[1, col]
        ax2.loglog(jax_ns, [t/N for t, N in zip(jax_run, jax_ns)],
                   "s-", color="C2", lw=2.0, ms=6, label="JAX per-fiber (run)")
        if pf_ns:
            ax2.loglog(pf_ns, [t/N for t, N in zip(pf_tot, pf_ns)],
                       "o-", color="C0", lw=2.0, ms=6, label="PyFibers per-fiber")
            if pf_ext_ns:
                lnk2_ns = [pf_ns[-1]] + pf_ext_ns
                lnk2_ts = [pf_tot[-1]/pf_ns[-1]] + [t/N for t,N in zip(pf_ext_ts,pf_ext_ns)]
                ax2.loglog(lnk2_ns, lnk2_ts, "o--", color="C0", lw=1.5,
                           ms=4, alpha=0.4)
        ax2r = ax2.twinx()
        pf_all = {N: t for N, t in zip(pf_ns, pf_tot)}
        pf_all.update({N: t for N, t in zip(pf_ext_ns, pf_ext_ts)})
        spd_ns = [N for N in jax_ns if N in pf_all]
        spd    = [pf_all[N] / jd[N]["run_s"] for N in spd_ns]
        if spd:
            ax2r.semilogx(spd_ns, spd, "D-", color="C4", lw=1.5, ms=5,
                          alpha=0.8, label="Speedup (right)")
            ax2r.set_ylabel("Speedup  PyFibers / JAX", color="C4")
            ax2r.tick_params(axis="y", labelcolor="C4")
            ax2r.set_ylim(bottom=0)
        ax2.set_xlabel("N fibers")
        ax2.set_ylabel("Per-fiber time (s)")
        ax2.set_title(f"(B{col+1}) {mname} — per-fiber & speedup")
        h2, l2 = ax2.get_legend_handles_labels()
        h2r,l2r= ax2r.get_legend_handles_labels()
        ax2.legend(h2 + h2r, l2 + l2r, fontsize=8.5)
        ax2.grid(True, which="both", alpha=0.25)

    fig.suptitle(
        "Performance: PyFibers/NEURON (CPU serial) vs JAX vmap (coupled solver, CPU)\n"
        "JAX 1st-call includes JIT compilation.  PyFibers dashed = linear extrapolation.",
        fontsize=10,
    )
    out = OUT / "fig_performance_benchmark.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  → {out}")


def write_outputs(results: dict):
    with open(OUT / "data_performance_benchmark.json", "w") as f:
        json.dump({
            m: {
                "threshold_jax_mA":    r["threshold_jax_mA"],
                "threshold_nrn_mA":    r["threshold_nrn_mA"],
                "amp_benchmark_mA":    r["amp_benchmark_mA"],
                "jax":      {str(k): v for k, v in r["jax"].items()},
                "pyfibers": {str(k): v for k, v in r["pyfibers"].items()},
            }
            for m, r in results.items()
        }, f, indent=2)

    rows = [["model", "backend", "N", "total_s", "per_fiber_ms", "speedup_vs_pf"]]
    for mname, res in results.items():
        jd = res["jax"]; pfd = res["pyfibers"]
        pf_rate = (pfd[max(pfd)]["total_s"] / max(pfd)) if pfd else None
        for N, v in sorted(jd.items()):
            pf_est  = pf_rate * N if pf_rate else float("nan")
            spd     = pf_est / v["run_s"] if pf_rate else float("nan")
            rows.append([mname, "JAX_run",   N, f"{v['run_s']:.5f}",
                         f"{v['run_s']/N*1e3:.3f}", f"{spd:.1f}"])
            rows.append([mname, "JAX_first", N, f"{v['compile_run_s']:.5f}",
                         f"{v['compile_run_s']/N*1e3:.3f}", ""])
        for N, v in sorted(pfd.items()):
            rows.append([mname, "PyFibers", N, f"{v['total_s']:.5f}",
                         f"{v['total_s']/N*1e3:.3f}", "1.0 (baseline)"])
    with open(OUT / "table_performance_benchmark.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"  → {OUT / 'data_performance_benchmark.json'}")
    print(f"  → {OUT / 'table_performance_benchmark.csv'}")


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}", flush=True)
    t_wall = time.time()
    all_results = {}

    # ─── MRG ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\n  MRG D=10 µm\n{'='*60}", flush=True)
    mrg_static, mrg_mfn, mrg_s0, mrg_Ve, mrg_mid, mrg_dt = \
        _make_jax_setup_mrg(diameter=10.0, n_nodes=21)

    print("  JAX threshold ...", end=" ", flush=True)
    mrg_thr_jax = _bisect(mrg_static, mrg_mfn, mrg_s0, mrg_Ve,
                          MRG_VREST, mrg_dt, 5.0)
    print(f"{mrg_thr_jax:.5f} mA")

    print("  PyFibers threshold ...", end=" ", flush=True)
    mrg_thr_nrn = find_threshold_extracellular(
        diameter=10.0, n_nodes=21, pw_ms=PW_MS, delay_ms=DELAY_MS,
        dt_ms=0.005, tstop_ms=5.0, src_height_um=HEIGHT_UM, sigma_S_m=SIGMA,
    )
    print(f"{mrg_thr_nrn:.5f} mA")

    mrg_amp = mrg_thr_jax * 1.3
    print(f"  Benchmark amp: {mrg_amp:.4f} mA\n")

    print("  [JAX vmap]", flush=True)
    mrg_jax = time_jax(mrg_static, mrg_mfn, mrg_s0, mrg_Ve,
                       MRG_VREST, mrg_amp, mrg_dt, 5.0, JAX_NS)

    print("\n  [PyFibers serial]", flush=True)
    def mrg_nrn_run(amp):
        return run_extracellular(diameter=10.0, n_nodes=21, pw_ms=PW_MS,
                                 delay_ms=DELAY_MS, amp_mA=float(amp),
                                 dt_ms=0.005, tstop_ms=5.0,
                                 src_height_um=HEIGHT_UM, sigma_S_m=SIGMA)
    mrg_pf = time_pyfibers(mrg_nrn_run, float(mrg_amp), PF_NS)

    all_results["MRG (D=10 µm)"] = {
        "threshold_jax_mA": float(mrg_thr_jax),
        "threshold_nrn_mA": float(mrg_thr_nrn),
        "amp_benchmark_mA": float(mrg_amp),
        "jax":     mrg_jax,
        "pyfibers": mrg_pf,
    }

    # ─── Sundt ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\n  Sundt D=0.8 µm\n{'='*60}", flush=True)
    sdt_static, sdt_mfn, sdt_s0, sdt_Ve, sdt_mid, sdt_dt = \
        _make_jax_setup_sundt(diameter=0.8, n_nodes=51)

    print("  JAX threshold ...", end=" ", flush=True)
    sdt_thr_jax = _bisect(sdt_static, sdt_mfn, sdt_s0, sdt_Ve,
                          SUNDT_VREST, sdt_dt, 10.0, lo=-5.0)
    print(f"{sdt_thr_jax:.5f} mA")

    print("  PyFibers threshold ...", end=" ", flush=True)
    sdt_thr_nrn = find_threshold_extracellular_sundt(
        diameter=0.8, n_nodes=51, pw_ms=PW_MS, delay_ms=DELAY_MS,
        dt_ms=0.005, tstop_ms=10.0, src_height_um=HEIGHT_UM, sigma_S_m=SIGMA,
    )
    print(f"{sdt_thr_nrn:.5f} mA")

    sdt_amp = sdt_thr_jax * 1.3
    print(f"  Benchmark amp: {sdt_amp:.4f} mA\n")

    print("  [JAX vmap]", flush=True)
    sdt_jax = time_jax(sdt_static, sdt_mfn, sdt_s0, sdt_Ve,
                       SUNDT_VREST, sdt_amp, sdt_dt, 10.0, JAX_NS)

    print("\n  [PyFibers serial]", flush=True)
    def sdt_nrn_run(amp):
        return run_extracellular_sundt(diameter=0.8, n_nodes=51, pw_ms=PW_MS,
                                       delay_ms=DELAY_MS, amp_mA=float(amp),
                                       dt_ms=0.005, tstop_ms=10.0,
                                       src_height_um=HEIGHT_UM, sigma_S_m=SIGMA)
    sdt_pf = time_pyfibers(sdt_nrn_run, float(sdt_amp), PF_NS)

    all_results["Sundt (D=0.8 µm)"] = {
        "threshold_jax_mA": float(sdt_thr_jax),
        "threshold_nrn_mA": float(sdt_thr_nrn),
        "amp_benchmark_mA": float(sdt_amp),
        "jax":     sdt_jax,
        "pyfibers": sdt_pf,
    }

    # ─── outputs ────────────────────────────────────────────────────────────
    print("\n=== Writing outputs ===", flush=True)
    write_outputs(all_results)
    make_figure(all_results)
    print(f"\nDone. Total wall time: {(time.time()-t_wall)/60:.1f} min")
