"""State-dependent effects of stimulation (Hussain 2024 Fig 3d equivalent).

Biphasic extracellular pulses interact with ongoing periodic intracellular
firing to change the *timing* of propagated APs.  SPIKE-synchronization
(Kreuz et al. 2015) quantifies how rapidly the propagated spike train
decorrelates from the intrinsic rhythm as stim amplitude rises.

Layout matches Hussain Fig 3d:
  * 3 fiber diameters (5.7, 8.7, 14 µm)  — columns
  * 3 intrinsic firing rates (10, 50, 100 Hz)  — rows
  * 3 stim frequencies (30, 50, 100 Hz)  — line styles (solid / dashed / dotted)
  * jaxon (orange) vs PyFibers (blue) — line colors
  * Per cell: mean ± 95% CI across N=5 fiber locations
  * Amplitude axis bound per diameter

Pacing is deterministic periodic (avg_interval=1000/IFR, noise=0) so that
both solvers see identical input — different seeds would conflate Poisson
randomness with the stim effect.

Outputs (outputs/spike_desync/)
-------------------------------
  data_spike_desync.json   — per-(diameter, IFR, stim_freq, amp) stats
  fig_spike_desync.png     — 3×3 grid

Run from project root:
    python experiments_v2/spike_desync.py
"""

from __future__ import annotations

import sys
import pathlib
import dataclasses
import time

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

from jaxon.fibers.mrg import (
    build_mrg, node_indices, section_centers_um,
    V_REST, CM_AXON, G_PAS_MYSA, G_PAS_FLUT, G_PAS_STIN,
)
from jaxon.channels.mrg_axnode import AxnodeMyel
from jaxon.stim.extracellular import point_source_potentials_mV
from jaxon.stim.extracellular_coupled import arrays_from_geometry, integrate
from jaxon.nrn_baseline import build_mrg_pyfibers

from neuron import h
from pyfibers import ScaledStim

from experiments_v2.utils import ensure_dir, save_json

OUT = ensure_dir(ROOT / "outputs" / "spike_desync")

# ── shared constants ──────────────────────────────────────────────────────────
CELSIUS    = 37.0
DT         = 0.005
TSTOP      = 50.0
N_NODES    = 21
N_STEPS    = int(TSTOP / DT)
SIGMA      = 0.3
PACE_DELAY = 1.0
PW_BIPHASIC = 0.1

# 5 fiber locations at different transverse offsets — provides CI bands.
SRC_H_FIBERS = [300., 600., 900., 1200., 1500.]
N_FIBERS_POP = len(SRC_H_FIBERS)

# Hussain Fig 3d structure: 3 × 3 × 3 with line style for stim freq.
DIAMETERS    = [5.7, 8.7, 14.0]
IFR_HZ       = [10.0, 50.0, 100.0]
F_STIM_HZ    = [30, 50, 100]
N_AMPS       = 6
AMP_RANGE_MA = {
    5.7:  (0.5, 2.5),
    8.7:  (0.25, 1.25),
    14.0: (0.15, 0.75),
}

# MRG channel constants
GNABAR = 3.0;  GNAPBAR = 0.01; GKBAR = 0.08; GL = 0.007
ENA    = 50.0; EK      = -90.0; EL    = -90.0

# Intracellular pacing pulse parameters (for the JAX path; pyfibers uses
# add_intrinsic_activity which gives equivalent timing).
PACE_PW    = 0.1
PACE_AMP   = 2.0


def _build_membrane_fn(static, geom):
    is_node   = static["is_node"]
    A_in      = static["A_in_cm2"]
    stype_gp  = {"node": 0.0, "mysa": G_PAS_MYSA, "flut": G_PAS_FLUT, "stin": G_PAS_STIN}
    g_pas_arr = jnp.asarray([stype_gp[s] for s in geom.section_type])

    v0 = jnp.float64(V_REST)
    (a_m0, b_m0), (a_h0, b_h0), (a_mp0, b_mp0), (a_s0, b_s0) = AxnodeMyel._alpha_beta(v0, CELSIUS)
    state0 = (
        jnp.where(is_node, float(a_m0  / (a_m0  + b_m0)),  0.0),
        jnp.where(is_node, float(a_h0  / (a_h0  + b_h0)),  0.0),
        jnp.where(is_node, float(a_mp0 / (a_mp0 + b_mp0)), 0.0),
        jnp.where(is_node, float(a_s0  / (a_s0  + b_s0)),  0.0),
    )

    def membrane_fn(Vm, state, dt):
        M, H, MP, S = state
        (a_m, b_m), (a_h, b_h), (a_mp, b_mp), (a_s, b_s) = AxnodeMyel._alpha_beta(Vm, CELSIUS)
        M2  = solve_gate_exponential(M,  dt, a_m,  b_m)
        H2  = solve_gate_exponential(H,  dt, a_h,  b_h)
        MP2 = solve_gate_exponential(MP, dt, a_mp, b_mp)
        S2  = solve_gate_exponential(S,  dt, a_s,  b_s)
        g_na  = GNABAR  * M2**3 * H2
        g_nap = GNAPBAR * MP2**3
        g_k   = GKBAR   * S2
        g_node = (g_na + g_nap + g_k + GL) * A_in * 1e6
        i_node = ((g_na + g_nap) * (Vm - ENA) + g_k * (Vm - EK) + GL * (Vm - EL)) * A_in * 1e6
        g_pas_us = g_pas_arr * A_in * 1e6
        i_pas    = g_pas_us  * (Vm - V_REST)
        g_eff = jnp.where(is_node, g_node, g_pas_us)
        i_ion = jnp.where(is_node, i_node, i_pas)
        return g_eff, i_ion, (M2, H2, MP2, S2)

    return membrane_fn, state0


def _periodic_pulse_times(rate_hz: float) -> np.ndarray:
    period = 1000.0 / rate_hz
    n = int(np.floor((TSTOP - PACE_DELAY) / period))
    return PACE_DELAY + np.arange(n) * period


def _pacing_intra(pulse_t: np.ndarray, node_pace: int, n_comp: int) -> np.ndarray:
    t_step = (np.arange(N_STEPS) + 1) * DT
    i_intra = np.zeros((N_STEPS, n_comp), dtype=np.float64)
    for pt in pulse_t:
        mask = (t_step >= pt) & (t_step < pt + PACE_PW)
        i_intra[mask, node_pace] = PACE_AMP
    return i_intra


def _spike_times(vm: np.ndarray, t_axis: np.ndarray, v_thresh: float = -20.0) -> np.ndarray:
    above = vm > v_thresh
    if not above.any():
        return np.array([])
    rises = np.where((vm[:-1] <= v_thresh) & (vm[1:] > v_thresh))[0]
    times = []
    for i in rises:
        v0, v1 = vm[i], vm[i + 1]
        t0, t1 = t_axis[i], t_axis[i + 1]
        frac = (v_thresh - v0) / (v1 - v0) if v1 != v0 else 0.0
        times.append(t0 + frac * (t1 - t0))
    return np.array(times)


def spike_sync(t1: np.ndarray, t2: np.ndarray) -> float:
    """Kreuz et al. 2015 SPIKE-synchronization S_C in [0, 1]."""
    if len(t1) == 0 and len(t2) == 0:
        return 1.0
    if len(t1) == 0 or len(t2) == 0:
        return 0.0
    def half_isi(t):
        if len(t) < 2:
            return np.array([np.inf])
        d = np.diff(t)
        return np.concatenate([[d[0]], np.minimum(d[:-1], d[1:]), [d[-1]]]) / 2
    h1, h2 = half_isi(t1), half_isi(t2)
    def coinc(ts, ts_other, h_self, h_other):
        out = np.zeros(len(ts), dtype=np.float64)
        for i, ti in enumerate(ts):
            j = int(np.argmin(np.abs(ts_other - ti)))
            delta = abs(ts_other[j] - ti)
            tau = min(h_self[i], h_other[j])
            out[i] = 1.0 if delta < tau else 0.0
        return out
    c1 = coinc(t1, t2, h1, h2)
    c2 = coinc(t2, t1, h2, h1)
    return float(np.concatenate([c1, c2]).mean())


def _run_pyfibers_vm(diameter: float, n_nodes: int, src_y_um: float,
                      ifr_hz: float, f_stim_hz: int, amp_mA: float) -> np.ndarray:
    """Return V_m trace at far node from PyFibers with periodic pacing +
    biphasic extracellular train (or pacing-only if amp_mA == 0)."""
    fiber = build_mrg_pyfibers(diameter=diameter, n_nodes=n_nodes, temperature=CELSIUS)
    fiber.record_vm()
    period_ms = 1000.0 / ifr_hz
    n_pulses = int(np.floor((TSTOP - PACE_DELAY) / period_ms))
    fiber.add_intrinsic_activity(
        loc=0.0, start_time=PACE_DELAY, avg_interval=period_ms,
        num_stims=n_pulses, noise=0.0,
    )
    if amp_mA == 0.0:
        h.celsius = CELSIUS
        h.dt = DT
        h.finitialize(fiber.v_rest)
        h.continuerun(TSTOP)
        return np.array(fiber.vm[-1])

    fiber.potentials = fiber.point_source_potentials(
        x=0.0, y=src_y_um, z=fiber.length / 2.0, i0=amp_mA, sigma=SIGMA,
    )
    # Biphasic train: ±1 for each phase, 0 between pulses.
    pulse_period_ms = 1000.0 / f_stim_hz
    def wav(t, P=pulse_period_ms):
        in_p = t - PACE_DELAY
        if in_p < 0:
            return 0.0
        cyc = in_p % P
        if cyc < PW_BIPHASIC:
            return 1.0
        if cyc < 2 * PW_BIPHASIC:
            return -1.0
        return 0.0
    stim = ScaledStim(waveform=wav, dt=DT, tstop=TSTOP)
    stim.run_sim(stimamp=1.0, fiber=fiber, ap_detect_location=1.0,
                  fail_on_end_excitation=False)
    return np.array(fiber.vm[-1])


def _biphasic_train_array(freq_hz: float) -> np.ndarray:
    period_ms = 1000.0 / freq_hz
    t_step = (np.arange(N_STEPS) + 1) * DT
    in_p = t_step - PACE_DELAY
    cyc = np.where(in_p >= 0, in_p % period_ms, -1.0)
    shape = np.zeros_like(cyc)
    shape[(cyc >= 0) & (cyc < PW_BIPHASIC)]                     = +1.0
    shape[(cyc >= PW_BIPHASIC) & (cyc < 2 * PW_BIPHASIC)]       = -1.0
    return shape


def _run_one_cell(D: float, ifr_hz: float, f_stim_hz: int) -> dict:
    """Sweep amplitudes × 5 fiber locations, compute SPIKE-sync per fiber."""
    _, geom = build_mrg(diameter=D, n_nodes=N_NODES)
    nodes   = node_indices(geom)
    centers = np.array(section_centers_um(geom))
    n_comp  = geom.n_comp
    node_pace = nodes[0]; node_far = nodes[-1]
    mid_node  = nodes[len(nodes) // 2]

    geom_c = dataclasses.replace(geom, cm_uF_cm2=[CM_AXON] * n_comp)
    static = arrays_from_geometry(geom_c, DT)
    membrane_fn, state0 = _build_membrane_fn(static, geom)

    pulse_t = _periodic_pulse_times(ifr_hz)
    i_intra = _pacing_intra(pulse_t, node_pace, n_comp)
    i_intra_j = jnp.asarray(i_intra)

    Ve_units = np.stack([
        np.asarray(point_source_potentials_mV(
            list(centers), src_x_um=0., src_y_um=h_um,
            src_z_um=float(centers[mid_node]), i0_mA=1.0,
        ))
        for h_um in SRC_H_FIBERS
    ])
    Ve_units_j = jnp.asarray(Ve_units)

    shape_j = jnp.asarray(_biphasic_train_array(f_stim_hz))
    t_axis = (np.arange(N_STEPS) + 1) * DT

    @jax.jit
    def _one_jax(amp_mA, pulse_mask, Ve_unit_one):
        Ve = Ve_unit_one * amp_mA
        (Vi, Vp), _ = integrate(static, membrane_fn, state0,
                                  Ve, pulse_mask, DT, v_rest=V_REST, record="all",
                                  i_intra=i_intra_j)
        return (Vi - Vp)[:, node_far]
    batched = jax.vmap(_one_jax, in_axes=(None, None, 0))

    # Reference (no stim) — vmapped over fibers
    vm_ref_jax = np.asarray(batched(jnp.float64(0.0),
                                      jnp.zeros(N_STEPS, dtype=jnp.float64),
                                      Ve_units_j))
    t_ref_jax = [_spike_times(vm_ref_jax[i], t_axis) for i in range(N_FIBERS_POP)]
    vm_ref_pf = [
        _run_pyfibers_vm(D, N_NODES, h_um, ifr_hz, f_stim_hz, 0.0)
        for h_um in SRC_H_FIBERS
    ]
    t_ref_pf = [_spike_times(vm, t_axis) for vm in vm_ref_pf]

    amp_lo, amp_hi = AMP_RANGE_MA[D]
    amps = np.linspace(amp_lo, amp_hi, N_AMPS)
    sync_jax = np.zeros((N_AMPS, N_FIBERS_POP), dtype=np.float64)
    sync_pf  = np.zeros((N_AMPS, N_FIBERS_POP), dtype=np.float64)

    for ai, amp in enumerate(amps):
        vm_jax = np.asarray(batched(jnp.float64(amp), shape_j, Ve_units_j))
        for fi in range(N_FIBERS_POP):
            t_test = _spike_times(vm_jax[fi], t_axis)
            sync_jax[ai, fi] = spike_sync(t_ref_jax[fi], t_test)

        for fi, h_um in enumerate(SRC_H_FIBERS):
            vm = _run_pyfibers_vm(D, N_NODES, h_um, ifr_hz, f_stim_hz, amp)
            t_test = _spike_times(vm, t_axis)
            sync_pf[ai, fi] = spike_sync(t_ref_pf[fi], t_test)

    # 95% CI half-width: t_{0.975, n-1}/sqrt(n).  For n=5, t = 2.776.
    t_crit = 2.776
    return {
        "amps_mA":      amps.tolist(),
        "mean_jax":     sync_jax.mean(axis=1).tolist(),
        "ci95_half_jax": (t_crit * sync_jax.std(axis=1, ddof=1)
                           / np.sqrt(N_FIBERS_POP)).tolist(),
        "mean_pf":      sync_pf.mean(axis=1).tolist(),
        "ci95_half_pf": (t_crit * sync_pf.std(axis=1, ddof=1)
                           / np.sqrt(N_FIBERS_POP)).tolist(),
    }


def make_figure(results: dict, out_path: pathlib.Path) -> None:
    n_rows = len(IFR_HZ)
    n_cols = len(DIAMETERS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 2.6 * n_rows),
                              constrained_layout=True)
    linestyles = {30: "-", 50: "--", 100: ":"}
    for c, D in enumerate(DIAMETERS):
        for r, ifr in enumerate(IFR_HZ):
            ax = axes[r, c]
            for f in F_STIM_HZ:
                cell = results[(D, ifr, f)]
                amps = np.array(cell["amps_mA"])
                mp = np.array(cell["mean_pf"]);  cp = np.array(cell["ci95_half_pf"])
                mj = np.array(cell["mean_jax"]); cj = np.array(cell["ci95_half_jax"])
                ax.plot(amps, mp, color="C0", lw=1.2, ls=linestyles[f],
                         label=f"PF {f} Hz" if (r == 0 and c == 0) else None)
                ax.fill_between(amps, mp - cp, mp + cp, color="C0", alpha=0.12)
                ax.plot(amps, mj, color="C1", lw=0.9, ls=linestyles[f],
                         label=f"jax {f} Hz" if (r == 0 and c == 0) else None)
                ax.fill_between(amps, mj - cj, mj + cj, color="C1", alpha=0.12)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(*AMP_RANGE_MA[D])
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3, lw=0.4)
            if r == 0:
                ax.set_title(f"{D} µm", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{int(ifr)} Hz\nSPIKE-sync", fontsize=8)
            if r == n_rows - 1:
                ax.set_xlabel("stim amp (mA)", fontsize=8)
            if r == 0 and c == 0:
                ax.legend(fontsize=6, loc="lower left", ncol=2)
    fig.suptitle(f"State-dependent effects of stimulation  "
                 f"(N = {N_FIBERS_POP} fiber locations, mean +/- 95% CI; "
                 f"line style = stim freq, color = solver)",
                 fontsize=11)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  -> {out_path}")


def main():
    print(f"=== Spike desync (Hussain 2024 Fig 3d equivalent) - jax vs PyFibers ===")
    print(f"D: {DIAMETERS}  IFR: {IFR_HZ}  stim freqs: {F_STIM_HZ} Hz")
    print(f"N fiber positions: {N_FIBERS_POP}  N amps: {N_AMPS}  TSTOP: {TSTOP} ms")
    total_cells = len(DIAMETERS) * len(IFR_HZ) * len(F_STIM_HZ)
    print(f"Total cells: {total_cells}  total sims: ~{total_cells * (N_AMPS + 1) * N_FIBERS_POP * 2}")
    results = {}
    for D in DIAMETERS:
        for ifr in IFR_HZ:
            for f in F_STIM_HZ:
                t0 = time.time()
                print(f"\n  D={D}, IFR={ifr} Hz, stim={f} Hz ...", flush=True)
                results[(D, ifr, f)] = _run_one_cell(D, ifr, f)
                print(f"    done in {time.time()-t0:.0f}s; "
                      f"final sync_jax = {results[(D,ifr,f)]['mean_jax'][-1]:.3f}, "
                      f"sync_pf = {results[(D,ifr,f)]['mean_pf'][-1]:.3f}", flush=True)
    save_json({
        "diameters_um":  DIAMETERS,
        "ifr_hz":        IFR_HZ,
        "f_stim_hz":     F_STIM_HZ,
        "n_amps":        N_AMPS,
        "tstop_ms":      TSTOP,
        "amp_range_mA":  AMP_RANGE_MA,
        "src_h_fibers":  SRC_H_FIBERS,
        "results": {
            f"D{D}_ifr{ifr}_f{f}": cell
            for (D, ifr, f), cell in results.items()
        },
    }, OUT / "data_spike_desync.json")
    make_figure(results, OUT / "fig_spike_desync.png")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
