"""Kilohertz frequency block — emergent phenomenon demo (Hussain 2024 Fig 3b).

A sinusoidal extracellular signal at kHz frequencies interacts with ongoing
intracellular pacing to produce a recruitment curve that is *non-monotonic*
in amplitude — the canonical HFAC nerve-block signature.

This reproduction mirrors Hussain Fig 3b:
  * 3 fiber diameters (5.7, 8.7, 14 µm) × 4 frequencies (1, 2, 5, 10 kHz)
  * N=7 fiber locations sampled at different transverse offsets from the
    point source (proxy for 7 different fascicle centroids in a real
    cuff-instrumented nerve)
  * Mean # of APs at the far end with 95% confidence intervals
  * Amplitude axis bound to physically-relevant range per diameter
    (smaller fibers need bigger currents)

Setup
-----
* MRG fibers, N=21 nodes each (~10-21 mm fiber depending on diameter).
* Intracellular pacing at 100 Hz × 0.1 ms × 2 nA at one end (node 0).
* 7 fiber locations: y-offsets [200, 400, 600, 800, 1000, 1200, 1400] µm
  above the source (point source at (0, 0, mid_z)).  Different offsets
  produce different field amplitudes at the fiber, mimicking different
  fascicle positions in a nerve.
* Extracellular sinusoidal at f ∈ {1, 2, 5, 10} kHz, swept in amplitude.
* Count APs arriving at the FAR end (last node) over the whole simulation.

Outputs (outputs/khz_block/)
----------------------------
  data_khz_block.json   — per-(diameter, freq, amp): mean/std/CI of # APs
  fig_khz_block.png     — 4×3 grid: rows = freq, cols = diameter

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
DT         = 0.002        # ms — 500 kHz sample rate (>50× Nyquist at 10 kHz)
TSTOP      = 30.0         # ms
DELAY_KHZ  = 1.0          # ms
DELAY_PACE = 0.5          # ms
PACE_HZ    = 100.0
PACE_PW    = 0.1
PACE_AMP   = 2.0
N_NODES    = 21
N_STEPS    = int(TSTOP / DT)
SIGMA      = 0.3

# Population: 7 fiber locations at different transverse offsets above the source.
# Mimics 7 fascicle centroids at different distances from the cuff.
SRC_H_FIBERS = [200., 400., 600., 800., 1000., 1200., 1400.]   # µm
N_FIBERS_POP = len(SRC_H_FIBERS)

DIAMETERS    = [5.7, 8.7, 14.0]
FREQS_KHZ    = [1.0, 2.0, 5.0, 10.0]
N_AMPS       = 10

# Hussain Fig 3b x-axis: smaller diameters need bigger currents.
AMP_MAX_MA   = {5.7: 10.0, 8.7: 5.0, 14.0: 2.0}

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
    t_step = (np.arange(N_STEPS) + 1) * DT
    period_ms = 1000.0 / PACE_HZ
    n_cycles = int(np.floor((TSTOP - DELAY_PACE) / period_ms))
    pulse_starts = DELAY_PACE + np.arange(n_cycles) * period_ms
    pace_on = np.zeros_like(t_step, dtype=bool)
    for ps in pulse_starts:
        pace_on |= (t_step >= ps) & (t_step < ps + PACE_PW)
    i_intra = np.zeros((N_STEPS, n_comp), dtype=np.float64)
    i_intra[pace_on, node_pace] = PACE_AMP
    return i_intra, n_cycles


def _count_aps(vm_trace: np.ndarray, v_thresh: float = -20.0) -> int:
    rises = (vm_trace[:-1] <= v_thresh) & (vm_trace[1:] > v_thresh)
    return int(rises.sum())


def _run_one_diameter(D: float) -> dict:
    """For one diameter, sweep freq × amplitude × N_FIBERS_POP."""
    print(f"\n=== D = {D} µm ===", flush=True)
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

    i_intra, n_pacing_cycles = _make_pacing_intra(node_pace, n_comp)

    # Ve_unit per fiber location (at +1 mA source).
    Ve_units = np.stack([
        np.asarray(point_source_potentials_mV(
            list(centers),
            src_x_um=0., src_y_um=h_um,
            src_z_um=float(centers[mid_node]), i0_mA=1.0,
        ))
        for h_um in SRC_H_FIBERS
    ])  # [N_FIBERS_POP, n_comp]

    far_idx = node_far
    Ve_units_j = jnp.asarray(Ve_units, dtype=jnp.float64)
    i_intra_j  = jnp.asarray(i_intra)

    # JIT'd single-fiber runner; we'll vmap over the population axis.
    @jax.jit
    def _one_fiber(amp_mA, pulse_mask, Ve_unit_one):
        Ve = Ve_unit_one * amp_mA
        (Vi_all, Vp_all), _ = integrate(
            static, membrane_fn, state0,
            Ve, pulse_mask, DT, v_rest=V_REST, record="all",
            i_intra=i_intra_j,
        )
        Vm_all = Vi_all - Vp_all
        return Vm_all[:, far_idx]

    batched = jax.vmap(_one_fiber, in_axes=(None, None, 0))

    print(f"  Baseline (no kHz) ...", flush=True)
    pulse_zero = jnp.zeros(N_STEPS, dtype=jnp.float64)
    vm_baseline = np.asarray(batched(jnp.float64(0.0), pulse_zero, Ve_units_j))  # [N_FIBERS, N_STEPS]
    n_aps_baseline = np.array([_count_aps(vm_baseline[i]) for i in range(N_FIBERS_POP)])
    print(f"    baseline APs per fiber: {n_aps_baseline.tolist()} "
          f"(mean {n_aps_baseline.mean():.1f})", flush=True)

    # ── freq × amp scan ──────────────────────────────────────────────────────
    t_step = (np.arange(N_STEPS) + 1) * DT
    results = {}
    for f_khz in FREQS_KHZ:
        amps = np.linspace(0.0, AMP_MAX_MA[D], N_AMPS)
        n_aps_array = np.zeros((N_AMPS, N_FIBERS_POP), dtype=np.int32)
        print(f"\n  f = {f_khz} kHz ...", flush=True)
        for ai, amp in enumerate(amps):
            phase = 2 * np.pi * f_khz * (t_step - DELAY_KHZ)
            pulse_mask = np.where(t_step >= DELAY_KHZ, np.sin(phase), 0.0).astype(np.float64)
            t0 = time.time()
            vm_far = np.asarray(batched(jnp.float64(amp), jnp.asarray(pulse_mask), Ve_units_j))
            n_aps = np.array([_count_aps(vm_far[i]) for i in range(N_FIBERS_POP)])
            n_aps_array[ai] = n_aps
            dt_sim = time.time() - t0
            print(f"    amp = {amp:6.2f} mA: {n_aps.tolist()} -> "
                  f"mean {n_aps.mean():.1f} ± {n_aps.std():.1f}  ({dt_sim:.1f} s)",
                  flush=True)
        results[float(f_khz)] = {
            "amps_mA":       amps.tolist(),
            "n_aps_per_fib": n_aps_array.tolist(),
            "mean":          n_aps_array.mean(axis=1).tolist(),
            "std":           n_aps_array.std(axis=1).tolist(),
            # 95% CI using t-distribution (n=7): t_{0.975, 6} ≈ 2.447
            "ci95_half":     (2.447 * n_aps_array.std(axis=1, ddof=1)
                              / np.sqrt(N_FIBERS_POP)).tolist(),
        }
    return {
        "diameter_um":       D,
        "n_aps_baseline":    n_aps_baseline.tolist(),
        "n_pacing_cycles":   n_pacing_cycles,
        "results_by_freq":   results,
    }


def make_figure(per_diam: list[dict], out_path: pathlib.Path) -> None:
    n_rows = len(FREQS_KHZ)
    n_cols = len(DIAMETERS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.5 * n_rows),
                              sharey=True, constrained_layout=True)

    for c, dia_data in enumerate(per_diam):
        D = dia_data["diameter_um"]
        baseline_mean = float(np.mean(dia_data["n_aps_baseline"]))
        for r, f_khz in enumerate(FREQS_KHZ):
            ax = axes[r, c]
            d = dia_data["results_by_freq"][float(f_khz)]
            amps = np.array(d["amps_mA"])
            mean = np.array(d["mean"])
            ci   = np.array(d["ci95_half"])
            ax.plot(amps, mean, color="C1", lw=1.6, label="jaxfibers")
            ax.fill_between(amps, mean - ci, mean + ci, color="C1", alpha=0.25)
            ax.axhline(baseline_mean, color="gray", lw=0.6, ls="--",
                       label=f"baseline {baseline_mean:.0f}")
            if r == 0:
                ax.set_title(f"D = {D} µm", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{int(f_khz)} kHz\n# APs", fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel("stimulus amplitude (mA)", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.3, lw=0.4)
            ax.set_ylim(bottom=-1)
            ax.set_xlim(0, AMP_MAX_MA[D])
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Kilohertz frequency population response  "
                 f"(N = {N_FIBERS_POP} fiber locations, mean ± 95% CI)",
                 fontsize=11)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  -> {out_path}", flush=True)


def main():
    print(f"=== kHz block demo (Hussain 2024 Fig 3b equivalent) ===")
    print(f"Population: N = {N_FIBERS_POP} fiber locations "
          f"(transverse offsets {SRC_H_FIBERS} µm)")
    print(f"Diameters: {DIAMETERS}")
    print(f"Frequencies: {FREQS_KHZ} kHz")
    print(f"Amplitudes per diameter: 0..{AMP_MAX_MA} mA ({N_AMPS} steps)")
    print(f"TSTOP = {TSTOP} ms; pacing = {PACE_HZ} Hz; dt = {DT} ms")

    per_diam = [_run_one_diameter(D) for D in DIAMETERS]

    save_json({
        "diameters_um":    DIAMETERS,
        "freqs_khz":       FREQS_KHZ,
        "fiber_y_um":      SRC_H_FIBERS,
        "n_fibers_pop":    N_FIBERS_POP,
        "n_amps":          N_AMPS,
        "amp_max_mA":      AMP_MAX_MA,
        "tstop_ms":        TSTOP,
        "dt_ms":           DT,
        "pace_hz":         PACE_HZ,
        "pace_pw_ms":      PACE_PW,
        "pace_amp_nA":     PACE_AMP,
        "per_diameter":    per_diam,
    }, OUT / "data_khz_block.json")

    make_figure(per_diam, OUT / "fig_khz_block.png")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
