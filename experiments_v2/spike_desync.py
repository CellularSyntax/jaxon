"""Spike desynchronization — emergent phenomenon demo (Hussain 2024 Fig 3d).

Extracellular biphasic stimulation interacts with ongoing intrinsic spiking
to change the *timing* of the propagated APs without necessarily changing
their count.  As stimulation amplitude rises through threshold, individual
spikes get advanced or retarded relative to where they would have arrived
without stim, and the population becomes desynchronized from its intrinsic
rhythm.  The Kreuz et al. 2015 SPIKE-synchronization metric (S_C in [0, 1])
quantifies how many of the intrinsic spikes survive at the same time after
stimulation.

This is a "state-dependent" effect: the same stim parameters produce
different outcomes depending on the membrane's gating state when the pulse
arrives.  An exact biophysical solver gets this for free; a surrogate
trained from rest cannot.

Setup (smaller than Hussain Fig 3d due to local CPU budget)
-----------------------------------------------------------
* MRG fiber, D=8.7 µm, N=21 nodes.
* Poisson intracellular pacing at IFR_HZ mean rate, injected at the
  pacing-end node.  Single seed per amplitude (Hussain averages over 5).
* Extracellular biphasic train (PW=0.1 ms each phase, ca polarity) at
  stim frequencies F_STIM_HZ, at amplitudes from sub- to supra-threshold.
* Record spike times at the far end.
* Compute SPIKE-synchronization between (intrinsic-only) reference and
  (intrinsic + extracellular) test spike trains.

Outputs (outputs/spike_desync/)
-------------------------------
  data_spike_desync.json   — amp vs SPIKE-sync, per stim frequency
  fig_spike_desync.png     — SPIKE-sync vs amplitude, one line per stim freq

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

from experiments_v2.utils import ensure_dir, save_json, ap_arrival_time

OUT = ensure_dir(ROOT / "outputs" / "spike_desync")

# ── shared constants ──────────────────────────────────────────────────────────
CELSIUS    = 37.0
DT         = 0.005        # ms
TSTOP      = 50.0         # ms — ~5 intrinsic cycles at 100 Hz
N_NODES    = 21
N_STEPS    = int(TSTOP / DT)
SIGMA      = 0.3
SRC_H      = 1000.0

DIAMETER   = 8.7
IFR_HZ     = 100.0        # mean Poisson intrinsic firing rate
PACE_PW    = 0.1          # ms
PACE_AMP   = 2.0          # nA
PACE_DELAY = 1.0          # ms before first pulse
F_STIM_HZ  = [30, 50, 100]
PW_BIPHASIC = 0.1         # ms each phase
N_AMPS     = 6
RNG_SEED   = 42

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


def _poisson_pacing(node_pace: int, n_comp: int, mean_rate_hz: float,
                     seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate Poisson-distributed intracellular pulse times.

    Returns
    -------
    i_intra : [N_STEPS, n_comp]   — intracellular current per step
    pulse_t : np.ndarray          — pulse onset times (ms)
    """
    rng = np.random.default_rng(seed)
    mean_isi_ms = 1000.0 / mean_rate_hz
    # Inverse-CDF sample of exponential ISIs
    pulse_t = []
    t_cur = PACE_DELAY
    while t_cur < TSTOP - PACE_PW:
        pulse_t.append(t_cur)
        t_cur += rng.exponential(scale=mean_isi_ms)
    pulse_t = np.array(pulse_t)

    t_step = (np.arange(N_STEPS) + 1) * DT
    i_intra = np.zeros((N_STEPS, n_comp), dtype=np.float64)
    for pt in pulse_t:
        mask = (t_step >= pt) & (t_step < pt + PACE_PW)
        i_intra[mask, node_pace] = PACE_AMP
    return i_intra, pulse_t


def _biphasic_train(freq_hz: float) -> np.ndarray:
    """Return a [N_STEPS] float array: +1 cathodic phase, -1 anodic phase, 0 off.

    Frequency = number of pulses per second.  Each pulse is 2*PW_BIPHASIC ms wide.
    """
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


def _extract_spike_times(vm_far: np.ndarray, v_thresh: float = -20.0) -> np.ndarray:
    """Return spike times (ms) via rising-edge interpolation."""
    t_step = (np.arange(N_STEPS) + 1) * DT
    above = vm_far > v_thresh
    if not above.any():
        return np.array([])
    rises = np.where((vm_far[:-1] <= v_thresh) & (vm_far[1:] > v_thresh))[0]
    times = []
    for i in rises:
        v0, v1 = vm_far[i], vm_far[i + 1]
        t0, t1 = t_step[i], t_step[i + 1]
        frac = (v_thresh - v0) / (v1 - v0) if v1 != v0 else 0.0
        times.append(t0 + frac * (t1 - t0))
    return np.array(times)


def spike_synchronization(t1: np.ndarray, t2: np.ndarray) -> float:
    """Kreuz et al. 2015 SPIKE-synchronization metric S_C in [0, 1].

    Returns 0 if no coincidences, 1 if every spike in either train has a
    unique time-matched partner in the other train. Adaptive coincidence
    window τ_ij^{(1,2)} per spike (half of the minimum local ISI on either
    side, in either train).
    """
    if len(t1) == 0 and len(t2) == 0:
        return 1.0
    if len(t1) == 0 or len(t2) == 0:
        return 0.0

    # ISI half-distances for each spike (boundary uses one-sided).
    def half_isi(t):
        if len(t) < 2:
            return np.array([np.inf])
        d = np.diff(t)
        return np.concatenate([[d[0]], np.minimum(d[:-1], d[1:]), [d[-1]]]) / 2

    h1 = half_isi(t1)
    h2 = half_isi(t2)

    def coincidence_indicator(ts, ts_other, h_self, h_other):
        """For each spike in ts, find nearest in ts_other and check |delta| < tau."""
        out = np.zeros(len(ts), dtype=np.float64)
        for i, ti in enumerate(ts):
            j = int(np.argmin(np.abs(ts_other - ti)))
            delta = abs(ts_other[j] - ti)
            tau = min(h_self[i], h_other[j])
            out[i] = 1.0 if delta < tau else 0.0
        return out

    c1 = coincidence_indicator(t1, t2, h1, h2)
    c2 = coincidence_indicator(t2, t1, h2, h1)
    # Pool over both trains
    pooled = np.concatenate([c1, c2])
    return float(pooled.mean()) if len(pooled) else 0.0


def main():
    print(f"=== Spike desynchronization demo (Hussain 2024 Fig 3d equivalent) ===")
    print(f"D = {DIAMETER} µm, IFR = {IFR_HZ} Hz Poisson, TSTOP = {TSTOP} ms")
    print(f"Biphasic stim: f in {F_STIM_HZ} Hz, PW = {PW_BIPHASIC} ms each phase")

    # ── build fiber ───────────────────────────────────────────────────────────
    _, geom = build_mrg(diameter=DIAMETER, n_nodes=N_NODES)
    nodes   = node_indices(geom)
    centers = np.array(section_centers_um(geom))
    n_comp  = geom.n_comp
    node_pace = nodes[0]
    node_far  = nodes[-1]
    mid_node  = nodes[len(nodes) // 2]

    print(f"Fiber length: {centers[-1] / 1000:.1f} mm; pace at node {node_pace}, "
          f"record at node {node_far}", flush=True)

    geom_c = dataclasses.replace(geom, cm_uF_cm2=[CM_AXON] * n_comp)
    static = arrays_from_geometry(geom_c, DT)
    membrane_fn, state0 = _build_membrane_fn(static, geom)

    # Pacing
    i_intra, pace_t = _poisson_pacing(node_pace, n_comp, IFR_HZ, RNG_SEED)
    print(f"Generated {len(pace_t)} intrinsic Poisson pulses (mean ISI "
          f"{1000/IFR_HZ:.0f} ms, last at t={pace_t[-1]:.1f} ms)", flush=True)

    # Ve_unit at +1 mA
    Ve_unit = np.asarray(point_source_potentials_mV(
        list(centers), src_x_um=0., src_y_um=SRC_H,
        src_z_um=float(centers[mid_node]), i0_mA=1.0,
    ))

    @jax.jit
    def run_far_vm(amp_mA: float, pulse_mask: jnp.ndarray) -> jnp.ndarray:
        Ve = jnp.asarray(Ve_unit, dtype=jnp.float64) * amp_mA
        (Vi_all, Vp_all), _ = integrate(
            static, membrane_fn, state0,
            Ve, pulse_mask, DT, v_rest=V_REST, record="all",
            i_intra=jnp.asarray(i_intra),
        )
        Vm_all = Vi_all - Vp_all
        return Vm_all[:, node_far]

    # Reference spike train: intrinsic pacing alone (no extracellular stim)
    print("\nBaseline reference (pacing only) ...", flush=True)
    vm_ref = np.asarray(run_far_vm(0.0, jnp.zeros(N_STEPS, dtype=jnp.float64)))
    t_ref = _extract_spike_times(vm_ref)
    print(f"  reference far-end spikes: {len(t_ref)}", flush=True)

    # ── threshold of biphasic 0.1 ms pulse at 50 Hz ───────────────────────────
    # Use a low amplitude estimate.  Skip a full bisection here for speed —
    # the threshold for D=8.7 µm at 1 mm src + 0.1 ms biphasic is ~0.4 mA based
    # on our MRG validation data.  Sweep around that.
    thr_estimate_mA = 0.4
    amps_mA = np.linspace(0.5 * thr_estimate_mA, 1.5 * thr_estimate_mA, N_AMPS)

    # ── frequency × amplitude scan ────────────────────────────────────────────
    results = {}
    for f_hz in F_STIM_HZ:
        sync_values = []
        n_spikes_values = []
        shape = _biphasic_train(f_hz)
        shape_j = jnp.asarray(shape)
        print(f"\n  f_stim = {f_hz} Hz, amps {amps_mA[0]:.2f}..{amps_mA[-1]:.2f} mA "
              f"({N_AMPS} steps) ...", flush=True)
        for amp in amps_mA:
            t0 = time.time()
            vm_far = np.asarray(run_far_vm(float(amp), shape_j))
            t_test = _extract_spike_times(vm_far)
            sync = spike_synchronization(t_ref, t_test)
            dt_sim = time.time() - t0
            sync_values.append(sync)
            n_spikes_values.append(len(t_test))
            print(f"    amp = {amp:.3f} mA: {len(t_test)} spikes, "
                  f"SPIKE-sync = {sync:.3f}  ({dt_sim:.1f} s)", flush=True)
        results[f_hz] = {
            "amps_mA":       amps_mA.tolist(),
            "sync":          sync_values,
            "n_spikes":      n_spikes_values,
        }

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    cmap = plt.get_cmap("viridis")

    # Panel 1: SPIKE-sync vs amplitude
    ax = axes[0]
    for i, f in enumerate(F_STIM_HZ):
        c = cmap(i / max(len(F_STIM_HZ) - 1, 1))
        r = results[f]
        ax.plot(r["amps_mA"], r["sync"], "o-", color=c, lw=1.6, ms=6, label=f"{f} Hz")
    ax.set_xlabel("Stim amplitude (mA)", fontsize=10)
    ax.set_ylabel("SPIKE-synchronization with reference", fontsize=10)
    ax.set_title(f"State-dependent desync — D={DIAMETER} µm, IFR={IFR_HZ} Hz",
                 fontsize=10)
    ax.legend(fontsize=9, title="stim freq")
    ax.grid(alpha=0.3, lw=0.4)
    ax.set_ylim(-0.05, 1.05)

    # Panel 2: # spikes vs amplitude
    ax = axes[1]
    for i, f in enumerate(F_STIM_HZ):
        c = cmap(i / max(len(F_STIM_HZ) - 1, 1))
        r = results[f]
        ax.plot(r["amps_mA"], r["n_spikes"], "o-", color=c, lw=1.6, ms=6, label=f"{f} Hz")
    ax.axhline(len(t_ref), color="gray", lw=0.8, ls="--",
                label=f"reference = {len(t_ref)}")
    ax.set_xlabel("Stim amplitude (mA)", fontsize=10)
    ax.set_ylabel("# spikes at far end", fontsize=10)
    ax.set_title("Spike count (compare to reference)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, lw=0.4)

    path = OUT / "fig_spike_desync.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  -> {path}", flush=True)

    # ── save data ─────────────────────────────────────────────────────────────
    save_json({
        "diameter_um":   DIAMETER,
        "n_nodes":       N_NODES,
        "tstop_ms":      TSTOP,
        "dt_ms":         DT,
        "celsius":       CELSIUS,
        "src_height_um": SRC_H,
        "ifr_hz":        IFR_HZ,
        "pace_pw_ms":    PACE_PW,
        "pace_amp_nA":   PACE_AMP,
        "rng_seed":      RNG_SEED,
        "n_ref_spikes":  len(t_ref),
        "t_ref_ms":      t_ref.tolist(),
        "amps_mA":       amps_mA.tolist(),
        "results":       results,
    }, OUT / "data_spike_desync.json")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
