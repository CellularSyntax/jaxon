"""State-dependent effects of stimulation (Hussain 2024 Fig 3d equivalent).

Biphasic extracellular pulses interact with ongoing Poisson intracellular
firing to change the *timing* of propagated APs.  SPIKE-synchronization
(Kreuz et al. 2015) quantifies how rapidly the propagated spike train
decorrelates from the intrinsic rhythm as stim amplitude rises.

Layout matches Hussain Fig 3d:
  * 3 fiber diameters (5.7, 8.7, 14 µm)  — columns
  * 3 intrinsic firing rates (10, 50, 100 Hz)  — rows
  * 3 stim frequencies (30, 50, 100 Hz)  — line styles (solid / dashed / dotted)
  * jaxfibers (orange) vs PyFibers (blue) — line colors
  * Per cell: mean ± 95% CI across N_TRIALS random Poisson seeds
  * Amplitude axis bound per diameter (matches Hussain Fig 3d)

Outputs (outputs/spike_desync/)
-------------------------------
  data_spike_desync.json   — per-(diameter, IFR, stim_freq, amp) summary stats
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

from jaxfibers.fibers.mrg import (
    build_mrg, node_indices, section_centers_um,
    V_REST, CM_AXON, G_PAS_MYSA, G_PAS_FLUT, G_PAS_STIN,
)
from jaxfibers.channels.mrg_axnode import AxnodeMyel
from jaxfibers.stim.extracellular import point_source_potentials_mV
from jaxfibers.stim.extracellular_coupled import arrays_from_geometry, integrate
from jaxfibers.nrn_baseline import build_mrg_pyfibers

from neuron import h
from pyfibers import ScaledStim
from scipy.interpolate import interp1d

from experiments_v2.utils import ensure_dir, save_json

OUT = ensure_dir(ROOT / "outputs" / "spike_desync")

# ── shared constants ──────────────────────────────────────────────────────────
CELSIUS    = 37.0
DT         = 0.005
TSTOP      = 50.0     # ms
N_NODES    = 21
N_STEPS    = int(TSTOP / DT)
SIGMA      = 0.3
SRC_H      = 1000.0
PACE_PW    = 0.1
PACE_AMP   = 2.0
PACE_DELAY = 1.0
PW_BIPHASIC = 0.1

# Hussain Fig 3d structure: 3 × 3 × 3 grid with stim-freq as line style.
DIAMETERS    = [5.7, 8.7, 14.0]
IFR_HZ       = [10.0, 50.0, 100.0]                # intrinsic firing rates
F_STIM_HZ    = [30, 50, 100]                       # extracellular stim freqs
N_AMPS       = 6
AMP_RANGE_MA = {                                   # per-diameter amp range
    5.7:  (0.5, 2.5),
    8.7:  (0.25, 1.25),
    14.0: (0.15, 0.75),
}
N_TRIALS     = 3                                   # random Poisson seeds per cell

# MRG channel constants
GNABAR = 3.0;  GNAPBAR = 0.01; GKBAR = 0.08; GL = 0.007
ENA    = 50.0; EK      = -90.0; EL    = -90.0


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


def _poisson_pulse_times(rate_hz: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    isi = 1000.0 / rate_hz
    times, t = [], PACE_DELAY
    while t < TSTOP - PACE_PW:
        times.append(t)
        t += rng.exponential(scale=isi)
    return np.array(times)


def _pacing_intra(pulse_t: np.ndarray, node_pace: int, n_comp: int) -> np.ndarray:
    t_step = (np.arange(N_STEPS) + 1) * DT
    i_intra = np.zeros((N_STEPS, n_comp), dtype=np.float64)
    for pt in pulse_t:
        mask = (t_step >= pt) & (t_step < pt + PACE_PW)
        i_intra[mask, node_pace] = PACE_AMP
    return i_intra


def _biphasic_train(freq_hz: float) -> np.ndarray:
    period_ms = 1000.0 / freq_hz
    n_pulses = int(np.floor((TSTOP - PACE_DELAY) / period_ms))
    pulse_starts = PACE_DELAY + np.arange(n_pulses) * period_ms
    t_step = (np.arange(N_STEPS) + 1) * DT
    shape = np.zeros(N_STEPS, dtype=np.float64)
    for ps in pulse_starts:
        cat = (t_step >= ps) & (t_step < ps + PW_BIPHASIC)
        ano = (t_step >= ps + PW_BIPHASIC) & (t_step < ps + 2 * PW_BIPHASIC)
        shape[cat] = +1.0
        shape[ano] = -1.0
    return shape


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
    """Kreuz et al. 2015 SPIKE-synchronization S_C ∈ [0, 1]."""
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
    pooled = np.concatenate([c1, c2])
    return float(pooled.mean()) if len(pooled) else 0.0


def _run_pyfibers(diameter: float, n_nodes: int, pulse_t: np.ndarray,
                  shape: np.ndarray, amp_mA: float) -> np.ndarray:
    """Run PyFibers with extracellular shape × amp + Poisson IClamp pacing.
    Return Vm trace at far node.

    If amp_mA == 0 we run NEURON manually (no ScaledStim, no extracellular
    potentials), since pyfibers rejects all-zero potential fields.
    """
    fiber = build_mrg_pyfibers(diameter=diameter, n_nodes=n_nodes, temperature=CELSIUS)
    fiber.record_vm()
    iclamps = []
    for pt in pulse_t:
        ic = h.IClamp(fiber[0](0.5))
        ic.delay = pt; ic.dur = PACE_PW; ic.amp = PACE_AMP
        iclamps.append(ic)

    if amp_mA == 0.0:
        h.celsius = CELSIUS
        h.dt = DT
        h.finitialize(fiber.v_rest)
        h.continuerun(TSTOP)
    else:
        fiber.potentials = fiber.point_source_potentials(
            x=0.0, y=SRC_H, z=fiber.length / 2.0, i0=amp_mA, sigma=SIGMA,
        )
        t_pts = np.concatenate([[0.0], (np.arange(len(shape)) + 1) * DT])
        v_pts = np.concatenate([[0.0], shape])
        f_wav = interp1d(t_pts, v_pts, bounds_error=False, fill_value=0.0)
        wav = lambda t: float(f_wav(t))
        stim = ScaledStim(waveform=wav, dt=DT, tstop=TSTOP)
        stim.run_sim(stimamp=1.0, fiber=fiber, ap_detect_location=0.5,
                      fail_on_end_excitation=False)

    return np.array(fiber.vm[-1])


def _run_one_cell(D: float, ifr_hz: float, f_stim_hz: int) -> dict:
    """Run all amplitudes × N_TRIALS seeds for one (D, IFR, f_stim) cell."""
    _, geom = build_mrg(diameter=D, n_nodes=N_NODES)
    nodes   = node_indices(geom)
    centers = np.array(section_centers_um(geom))
    n_comp  = geom.n_comp
    node_pace = nodes[0]
    node_far  = nodes[-1]
    mid_node  = nodes[len(nodes) // 2]

    geom_c = dataclasses.replace(geom, cm_uF_cm2=[CM_AXON] * n_comp)
    static = arrays_from_geometry(geom_c, DT)
    membrane_fn, state0 = _build_membrane_fn(static, geom)

    Ve_unit = np.asarray(point_source_potentials_mV(
        list(centers), src_x_um=0., src_y_um=SRC_H,
        src_z_um=float(centers[mid_node]), i0_mA=1.0,
    ))
    Ve_unit_j = jnp.asarray(Ve_unit)
    shape = _biphasic_train(f_stim_hz)
    shape_j = jnp.asarray(shape)
    t_axis = (np.arange(N_STEPS) + 1) * DT

    amp_lo, amp_hi = AMP_RANGE_MA[D]
    amps = np.linspace(amp_lo, amp_hi, N_AMPS)

    # JAX runner: closes over i_intra (per trial); we rebuild it per trial.
    @jax.jit
    def _run_jax(amp_mA, pulse_mask, i_intra_j):
        Ve = Ve_unit_j * amp_mA
        (Vi_all, Vp_all), _ = integrate(
            static, membrane_fn, state0,
            Ve, pulse_mask, DT, v_rest=V_REST, record="all",
            i_intra=i_intra_j,
        )
        return (Vi_all - Vp_all)[:, node_far]

    sync_jax = np.zeros((N_AMPS, N_TRIALS), dtype=np.float64)
    sync_pf  = np.zeros((N_AMPS, N_TRIALS), dtype=np.float64)

    for trial in range(N_TRIALS):
        pulse_t = _poisson_pulse_times(ifr_hz, seed=(int(D * 100) * 100 + int(ifr_hz) * 10
                                                       + f_stim_hz + trial * 7919) % (2**31))
        i_intra = _pacing_intra(pulse_t, node_pace, n_comp)
        i_intra_j = jnp.asarray(i_intra)

        # Reference (no stim) — for SPIKE-sync
        vm_ref_jax = np.asarray(_run_jax(jnp.float64(0.0),
                                          jnp.zeros(N_STEPS, dtype=jnp.float64),
                                          i_intra_j))
        t_ref_jax = _spike_times(vm_ref_jax, t_axis)
        vm_ref_pf  = _run_pyfibers(D, N_NODES, pulse_t,
                                     np.zeros_like(shape), 0.0)
        t_ref_pf  = _spike_times(vm_ref_pf, t_axis)

        for ai, amp in enumerate(amps):
            vm_jax = np.asarray(_run_jax(jnp.float64(amp), shape_j, i_intra_j))
            t_test_jax = _spike_times(vm_jax, t_axis)
            sync_jax[ai, trial] = spike_sync(t_ref_jax, t_test_jax)

            vm_pf = _run_pyfibers(D, N_NODES, pulse_t, shape, amp)
            t_test_pf = _spike_times(vm_pf, t_axis)
            sync_pf[ai, trial] = spike_sync(t_ref_pf, t_test_pf)

    t_crit = 4.303 if N_TRIALS == 3 else (3.182 if N_TRIALS == 4 else 2.776)   # t_0.975, n-1
    return {
        "amps_mA":         amps.tolist(),
        "mean_jax":        sync_jax.mean(axis=1).tolist(),
        "ci95_half_jax":   (t_crit * sync_jax.std(axis=1, ddof=1)
                              / np.sqrt(N_TRIALS)).tolist(),
        "mean_pf":         sync_pf.mean(axis=1).tolist(),
        "ci95_half_pf":    (t_crit * sync_pf.std(axis=1, ddof=1)
                              / np.sqrt(N_TRIALS)).tolist(),
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
                mp = np.array(cell["mean_pf"]);   cp = np.array(cell["ci95_half_pf"])
                mj = np.array(cell["mean_jax"]);  cj = np.array(cell["ci95_half_jax"])
                ax.plot(amps, mp, color="C0", lw=1.4, ls=linestyles[f],
                         label=f"PF {f} Hz" if (r == 0 and c == 0) else None)
                ax.fill_between(amps, mp - cp, mp + cp, color="C0", alpha=0.15)
                ax.plot(amps, mj, color="C1", lw=1.0, ls=linestyles[f],
                         label=f"jax {f} Hz" if (r == 0 and c == 0) else None)
                ax.fill_between(amps, mj - cj, mj + cj, color="C1", alpha=0.15)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(*AMP_RANGE_MA[D])
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.3, lw=0.4)
            if r == 0:
                ax.set_title(f"{D} µm", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{int(ifr)} Hz\nSPIKE-sync", fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel("stim amplitude (mA)", fontsize=9)
            if r == 0 and c == 0:
                ax.legend(fontsize=6, loc="lower left", ncol=2)

    fig.suptitle(
        f"State-dependent effects of stimulation  "
        f"(N = {N_TRIALS} Poisson seeds per cell, mean ± 95% CI; "
        f"line style = stim freq, color = solver)",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  -> {out_path}", flush=True)


def main():
    print(f"=== Spike desync (Hussain 2024 Fig 3d equivalent) — jaxfibers vs PyFibers ===")
    print(f"Diameters: {DIAMETERS}  IFR: {IFR_HZ} Hz  stim freqs: {F_STIM_HZ} Hz")
    print(f"N_TRIALS = {N_TRIALS} per cell, N_AMPS = {N_AMPS}, TSTOP = {TSTOP} ms")
    print(f"Total cells: {len(DIAMETERS) * len(IFR_HZ) * len(F_STIM_HZ)} "
          f"× {N_AMPS} × {N_TRIALS} × 2 models = "
          f"{len(DIAMETERS) * len(IFR_HZ) * len(F_STIM_HZ) * N_AMPS * N_TRIALS * 2} sims")

    results = {}
    for D in DIAMETERS:
        for ifr in IFR_HZ:
            for f in F_STIM_HZ:
                t0 = time.time()
                print(f"\n  D={D}, IFR={ifr} Hz, stim={f} Hz ...", flush=True)
                results[(D, ifr, f)] = _run_one_cell(D, ifr, f)
                print(f"  done in {time.time()-t0:.0f}s; "
                      f"sync_jax (n=last amp) = {results[(D,ifr,f)]['mean_jax'][-1]:.3f}, "
                      f"sync_pf = {results[(D,ifr,f)]['mean_pf'][-1]:.3f}", flush=True)

    # Save flat keys for JSON
    save_json({
        "diameters_um":  DIAMETERS,
        "ifr_hz":        IFR_HZ,
        "f_stim_hz":     F_STIM_HZ,
        "n_amps":        N_AMPS,
        "n_trials":      N_TRIALS,
        "tstop_ms":      TSTOP,
        "amp_range_mA":  AMP_RANGE_MA,
        "results": {
            f"D{D}_ifr{ifr}_f{f}": cell
            for (D, ifr, f), cell in results.items()
        },
    }, OUT / "data_spike_desync.json")

    make_figure(results, OUT / "fig_spike_desync.png")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
