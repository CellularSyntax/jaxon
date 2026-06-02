"""Kilohertz frequency block — emergent phenomenon demo (Hussain 2024 Fig 3b).

A sinusoidal extracellular signal at kHz frequencies interacts with ongoing
intracellular pacing to produce a recruitment curve that is *non-monotonic*
in amplitude:

  * sub-threshold: pacing-driven APs propagate unimpeded.
  * low-suprathreshold ('asynchronous excitation'): the kHz signal adds
    extra APs from virtual-cathode firing.
  * mid amplitude ('block'): sustained depolarisation at the firing site
    pushes Na inactivation past its operating point — propagating APs are
    blocked.
  * higher amplitude: another firing regime may emerge.

The non-monotonicity (an actual "block window" as a function of amplitude)
is the canonical HFAC nerve-block signature.  Reproducing it from first
principles — without any kHz-specific training data — is one of the most
striking validations of an exact biophysical solver.

Setup (smaller than Hussain Fig 3b due to local CPU budget)
-----------------------------------------------------------
* MRG fiber, D=8.7 µm, N=21 nodes (~21 mm fiber).
* Point-source extracellular at y=1 mm.
* Intracellular pacing: 0.1 ms rectangular pulse at 100 Hz, injected at
  node 0 (one end of the fiber).
* Extracellular sinusoidal at f ∈ {1, 2, 5, 10} kHz, amplitude swept across
  N_AMPS values from sub-threshold to deep-block.
* Count APs arriving at the FAR end (last node) over the full simulation.

Outputs (outputs/khz_block/)
----------------------------
  data_khz_block.json   — amp vs # APs, per frequency
  fig_khz_block.png     — recruitment curves, one panel per frequency

Run from project root:
    python experiments_v2/khz_block.py
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

from experiments_v2.utils import ensure_dir, save_json

OUT = ensure_dir(ROOT / "outputs" / "khz_block")

# ── shared constants ──────────────────────────────────────────────────────────
CELSIUS    = 37.0
DT         = 0.002        # ms — 500 kHz sample rate, OK for up to 10 kHz signal
TSTOP      = 30.0         # ms — accommodates 3 pacing cycles at 100 Hz
DELAY_KHZ  = 1.0          # ms — kHz signal starts after first pacing pulse
DELAY_PACE = 0.5          # ms — first pacing pulse
PACE_HZ    = 100.0
PACE_PW_MS = 0.1
PACE_AMP_NA = 2.0         # well-suprathreshold for intracellular at end node
N_NODES    = 21
N_STEPS    = int(TSTOP / DT)
SIGMA      = 0.3
SRC_H      = 1000.0

DIAMETER   = 8.7
FREQS_KHZ  = [1.0, 2.0, 5.0, 10.0]
N_AMPS     = 7
AMP_MAX_MA = {1.0: 5.0, 2.0: 5.0, 5.0: 5.0, 10.0: 2.0}    # frequency-dependent

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


def _make_pacing_intra(node_pace: int, n_comp: int) -> np.ndarray:
    """Periodic intracellular pulses at PACE_HZ injected at node_pace."""
    t_step = (np.arange(N_STEPS) + 1) * DT
    period_ms = 1000.0 / PACE_HZ                            # ms
    n_cycles = int(np.floor((TSTOP - DELAY_PACE) / period_ms))
    pulse_starts = DELAY_PACE + np.arange(n_cycles) * period_ms
    pace_on = np.zeros_like(t_step, dtype=bool)
    for ps in pulse_starts:
        pace_on |= (t_step >= ps) & (t_step < ps + PACE_PW_MS)
    i_intra = np.zeros((N_STEPS, n_comp), dtype=np.float64)
    i_intra[pace_on, node_pace] = PACE_AMP_NA
    return i_intra


def _count_aps(vm_trace: np.ndarray, v_thresh: float = -20.0) -> int:
    """Count rising-edge crossings of v_thresh."""
    above = vm_trace > v_thresh
    if not above.any():
        return 0
    rises = (vm_trace[:-1] <= v_thresh) & (vm_trace[1:] > v_thresh)
    return int(rises.sum())


def main():
    print(f"=== kHz block demo (Hussain 2024 Fig 3b equivalent) ===")
    print(f"D = {DIAMETER} µm, N = {N_NODES} nodes, TSTOP = {TSTOP} ms, dt = {DT} ms")
    print(f"Pacing: {PACE_HZ} Hz × {PACE_PW_MS} ms × {PACE_AMP_NA} nA at node 0")
    print(f"kHz signal: {FREQS_KHZ} kHz, {N_AMPS} amplitudes per freq")

    # ── build fiber ──────────────────────────────────────────────────────────
    _, geom = build_mrg(diameter=DIAMETER, n_nodes=N_NODES)
    nodes   = node_indices(geom)
    centers = np.array(section_centers_um(geom))
    n_comp  = geom.n_comp
    node_pace = nodes[0]
    node_far  = nodes[-1]
    mid_node  = nodes[len(nodes) // 2]

    print(f"Fiber length: {centers[-1] / 1000:.1f} mm; pacing at node {node_pace}, "
          f"AP counting at node {node_far}", flush=True)

    geom_c = dataclasses.replace(geom, cm_uF_cm2=[CM_AXON] * n_comp)
    static = arrays_from_geometry(geom_c, DT)
    membrane_fn, state0 = _build_membrane_fn(static, geom)

    # Pacing current array
    i_intra = _make_pacing_intra(node_pace, n_comp)

    # Extracellular Ve_unit at +1 mA (sign convention used below)
    Ve_unit = np.asarray(point_source_potentials_mV(
        list(centers), src_x_um=0., src_y_um=SRC_H,
        src_z_um=float(centers[mid_node]), i0_mA=1.0,
    ))

    # JIT'd runner: (amp_mA, pulse_mask) -> Vm trace at node_far
    far_idx = node_far
    @jax.jit
    def run_far_vm(amp_mA: float, pulse_mask: jnp.ndarray) -> jnp.ndarray:
        Ve = jnp.asarray(Ve_unit, dtype=jnp.float64) * amp_mA
        (Vi_all, Vp_all), _ = integrate(
            static, membrane_fn, state0,
            Ve, pulse_mask, DT, v_rest=V_REST, record="all",
            i_intra=jnp.asarray(i_intra),
        )
        Vm_all = Vi_all - Vp_all
        return Vm_all[:, far_idx]

    # Baseline (no kHz stim): expected n_aps from pacing alone
    t_step = (np.arange(N_STEPS) + 1) * DT
    pulse_zero = jnp.zeros(N_STEPS, dtype=jnp.float64)
    print("\n  Baseline (no kHz stim) ...", flush=True)
    vm_far_baseline = np.asarray(run_far_vm(0.0, pulse_zero))
    n_aps_baseline = _count_aps(vm_far_baseline)
    print(f"    far-end APs without kHz: {n_aps_baseline}", flush=True)

    # ── frequency × amplitude scan ────────────────────────────────────────────
    results = {}
    for f_khz in FREQS_KHZ:
        amps = np.linspace(0.0, AMP_MAX_MA[f_khz], N_AMPS)
        n_aps_per_amp = []
        print(f"\n  f = {f_khz} kHz, amps {amps[0]:.2f}..{amps[-1]:.2f} mA "
              f"({N_AMPS} steps) ...", flush=True)
        for amp in amps:
            # Sinusoid starts at DELAY_KHZ; zero before that.
            phase = 2 * np.pi * f_khz * (t_step - DELAY_KHZ)
            pulse_mask = np.where(t_step >= DELAY_KHZ, np.sin(phase), 0.0).astype(np.float64)
            t0 = time.time()
            vm_far = np.asarray(run_far_vm(float(amp), jnp.asarray(pulse_mask)))
            n_aps = _count_aps(vm_far)
            dt_sim = time.time() - t0
            n_aps_per_amp.append(n_aps)
            print(f"    amp = {amp:.3f} mA: {n_aps} AP(s) at far end  "
                  f"({dt_sim:.1f} s)", flush=True)
        results[f_khz] = {
            "amps_mA": amps.tolist(),
            "n_aps":   n_aps_per_amp,
        }

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(FREQS_KHZ), figsize=(3.0 * len(FREQS_KHZ), 3.0),
                              sharey=True, constrained_layout=True)
    for ax, f_khz in zip(axes, FREQS_KHZ):
        r = results[f_khz]
        ax.plot(r["amps_mA"], r["n_aps"], "o-", color="C0", lw=1.5, ms=6)
        ax.axhline(n_aps_baseline, color="gray", lw=0.8, ls="--",
                   label=f"baseline = {n_aps_baseline}")
        ax.set_title(f"{f_khz} kHz", fontsize=10)
        ax.set_xlabel("Extracellular amplitude (mA)", fontsize=9)
        ax.set_ylabel("# APs at far end" if f_khz == FREQS_KHZ[0] else "")
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3, lw=0.4)
        ax.set_ylim(bottom=-0.5)

    fig.suptitle(
        f"kHz extracellular sinusoid vs intrinsic pacing — MRG D={DIAMETER} µm",
        fontsize=10,
    )
    path = OUT / "fig_khz_block.png"
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
        "pace_hz":       PACE_HZ,
        "pace_pw_ms":    PACE_PW_MS,
        "pace_amp_nA":   PACE_AMP_NA,
        "n_aps_baseline": n_aps_baseline,
        "results":       results,
    }, OUT / "data_khz_block.json")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
