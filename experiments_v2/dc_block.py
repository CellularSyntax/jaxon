"""Multiple conduction responses to cathodic stimulation
(Hussain 2024 Fig 3a equivalent) — waterfall visualisation.

For a single MRG fiber and a cathodic monophasic extracellular pulse,
we show V_m(t) at each node of Ranvier as a vertically-offset trace,
producing a "waterfall" / wave-propagation plot.  AP propagation is
visible as a diagonal sweep of depolarisation across the trace stack.

Reproducing this with the JAX coupled solver (orange dashed) vs.
PyFibers/NEURON (blue solid) shows the two implementations match at
every node simultaneously — not just at the soma/probe — even at high
stimulation amplitudes where Na inactivation dynamics matter.

Setup
-----
* MRG fiber, D=10 µm, N_NODES=21 (~22 mm fiber).
* Point-source extracellular at y=1 mm above the fiber centre.
* Cathodic monophasic pulse, PW=0.1 ms.
* 4 amplitudes (rows): 0.5×, 1.05×, 1.5×, 3.0× threshold.
* Each panel: waterfall plot, jaxon vs pyfibers.

Outputs (outputs/dc_block/)
---------------------------
  data_dc_block.json   — per-amplitude V_m traces for both models
  fig_dc_block.png     — 4-amplitude waterfall grid

Run from project root:
    python experiments_v2/dc_block.py
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
    build_mrg, build_mrg_interp, node_indices, section_centers_um,
    V_REST, CM_AXON, G_PAS_MYSA, G_PAS_FLUT, G_PAS_STIN,
)
from jaxon.channels.mrg_axnode import AxnodeMyel
from jaxon.stim.extracellular import point_source_potentials_mV
from jaxon.stim.extracellular_coupled import arrays_from_geometry, integrate
from neuron import h
from pyfibers import build_fiber, FiberModel, ScaledStim
from scipy.interpolate import interp1d

from experiments_v2.utils import (
    PULSES, make_pulse_array, jax_bisect, ensure_dir, save_json,
)

OUT = ensure_dir(ROOT / "outputs" / "dc_block")

# ── shared constants ──────────────────────────────────────────────────────────
CELSIUS  = 37.0
DT       = 0.005
TSTOP    = 7.5    # ms — Hussain Fig 3a; covers propagation across 101-node fiber
DELAY    = 0.5
N_NODES  = 101    # Hussain Fig 3a uses ~101 nodes; critical for visible spatial propagation
N_STEPS  = int(TSTOP / DT)
SIGMA    = 0.3
SRC_H    = 1000.0

DIAMETER     = 12.0               # µm — Hussain Fig 3a
PW_MS        = 0.75               # ms — Hussain Fig 3a: cathodic monophasic
AMP_FACTORS  = [0.5, 1.1, 1.5, 2.5]  # sub-thr → excitation → unidirectional → block/re-excitation

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


def _run_pyfibers(amp_mA: float, diameter: float, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """PyFibers MRG run with cathodic monophasic ScaledStim. Returns (t_ms, vm [n_nodes, n_t])."""
    fiber = build_fiber(FiberModel.MRG_INTERPOLATION, diameter=diameter,
                         n_nodes=n_nodes, temperature=CELSIUS)
    fiber.record_vm()
    fiber.potentials = fiber.point_source_potentials(
        x=0.0, y=SRC_H, z=fiber.length / 2.0, i0=amp_mA, sigma=SIGMA,
    )
    # Build a stepwise waveform: 0 before delay, +1 during [delay, delay+pw], else 0
    pulse_arr = make_pulse_array("mono_c", PW_MS, N_STEPS, DT, DELAY)
    t_pts = np.concatenate([[0.0], (np.arange(len(pulse_arr)) + 1) * DT])
    v_pts = np.concatenate([[0.0], pulse_arr])
    f = interp1d(t_pts, v_pts, bounds_error=False, fill_value=0.0)
    wav = lambda t: float(f(t))
    stim = ScaledStim(waveform=wav, dt=DT, tstop=TSTOP)
    stim.run_sim(stimamp=1.0, fiber=fiber, ap_detect_location=0.5, fail_on_end_excitation=False)
    vm = np.array([np.array(v) for v in fiber.vm])
    t_ms = np.array(fiber.time)
    return t_ms, vm


def main():
    print(f"=== Multiple conduction responses (Hussain 2024 Fig 3a, waterfall view) ===")
    print(f"D = {DIAMETER} µm, N = {N_NODES} nodes, PW = {PW_MS} ms cathodic\n")

    # ── build JAX fiber + bisect for threshold ───────────────────────────────
    _, geom = build_mrg_interp(diameter=DIAMETER, n_nodes=N_NODES)
    nodes   = node_indices(geom)
    centers = np.array(section_centers_um(geom))
    n_comp  = geom.n_comp
    mid_node = nodes[len(nodes) // 2]

    print(f"Fiber length: {centers[-1] / 1000:.1f} mm; n_comp = {n_comp}", flush=True)

    geom_c = dataclasses.replace(geom, cm_uF_cm2=[CM_AXON] * n_comp)
    static = arrays_from_geometry(geom_c, DT)
    membrane_fn, state0 = _build_membrane_fn(static, geom)

    Ve_unit = np.asarray(point_source_potentials_mV(
        list(centers), src_x_um=0., src_y_um=SRC_H,
        src_z_um=float(centers[mid_node]), i0_mA=-1.0,
    ))

    is_node_arr = static["is_node"]

    @jax.jit
    def run_peak(amp_mA, pulse_arr_j):
        Ve = jnp.asarray(Ve_unit, dtype=jnp.float64) * (amp_mA / -1.0)
        (Vi_all, Vp_all), _ = integrate(
            static, membrane_fn, state0, Ve, pulse_arr_j, DT,
            v_rest=V_REST, record="all",
        )
        Vm = Vi_all - Vp_all
        Vm_nodes = jnp.where(is_node_arr[None, :], Vm, -jnp.inf)
        return jnp.max(Vm_nodes)

    @jax.jit
    def run_full(amp_mA, pulse_arr_j):
        Ve = jnp.asarray(Ve_unit, dtype=jnp.float64) * (amp_mA / -1.0)
        (Vi_all, Vp_all), _ = integrate(
            static, membrane_fn, state0, Ve, pulse_arr_j, DT,
            v_rest=V_REST, record="all",
        )
        return Vi_all - Vp_all

    pulse_arr = make_pulse_array("mono_c", PW_MS, N_STEPS, DT, DELAY)
    pulse_j   = jnp.asarray(pulse_arr)
    spec = PULSES["mono_c"]

    print(f"Finding threshold (mono_c, PW = {PW_MS} ms) ...", flush=True)
    thr = jax_bisect(run_peak, pulse_arr, lo=spec.lo, hi=spec.hi)
    print(f"Threshold (JAX): {thr:.4f} mA\n", flush=True)

    # ── run JAX + PyFibers at each amplitude ─────────────────────────────────
    t_jax_axis = (np.arange(N_STEPS) + 1) * DT
    results = []
    for fac in AMP_FACTORS:
        amp = thr * fac
        print(f"  amp = {amp:.4f} mA  ({fac}x threshold)", flush=True)

        # JAX
        t0 = time.time()
        Vm_all = np.asarray(run_full(jnp.float64(amp), pulse_j))   # [N_STEPS, n_comp]
        t_jax = time.time() - t0
        jax_nodes = Vm_all[:, nodes]                                # [N_STEPS, n_nodes]
        print(f"    JAX done in {t_jax:.1f}s, peak {jax_nodes.max():.1f} mV", flush=True)

        # PyFibers
        t0 = time.time()
        t_pf, vm_pf = _run_pyfibers(amp, DIAMETER, N_NODES)         # [n_nodes, n_t]
        t_pf_wall = time.time() - t0
        print(f"    PyFibers done in {t_pf_wall:.1f}s, peak {vm_pf.max():.1f} mV", flush=True)

        results.append({
            "amp_mA":        amp,
            "amp_factor":    fac,
            "jax_t_ms":      t_jax_axis,
            "jax_vm_nodes":  jax_nodes,        # [N_STEPS, n_nodes]
            "pf_t_ms":       t_pf,
            "pf_vm_nodes":   vm_pf.T,           # [n_t, n_nodes] for plot consistency
            "node_pos_mm":   centers[nodes] / 1000.0,
        })

    # ── figure: 4-amplitude grid of waterfall plots ──────────────────────────
    fig, axes = plt.subplots(1, len(AMP_FACTORS),
                              figsize=(3.6 * len(AMP_FACTORS), 5.5),
                              sharey=True, constrained_layout=True)
    # Vertical scale: each Vm trace is rescaled to fit ~70% of the inter-node spacing.
    inter_node_mm = results[0]["node_pos_mm"][1] - results[0]["node_pos_mm"][0]
    v_range = 150.0     # mV (approx -100..+50)
    scale_mm_per_mV = (inter_node_mm * 0.7) / v_range

    for c, res in enumerate(results):
        ax = axes[c]
        npos = res["node_pos_mm"]
        for i, node_y in enumerate(npos):
            # JAX trace
            ax.plot(res["jax_t_ms"],
                    node_y + (res["jax_vm_nodes"][:, i] - V_REST) * scale_mm_per_mV,
                    color="C1", lw=0.9, alpha=0.9,
                    label="jaxon" if i == 0 else None)
            # PyFibers trace
            ax.plot(res["pf_t_ms"],
                    node_y + (res["pf_vm_nodes"][:, i] - V_REST) * scale_mm_per_mV,
                    color="C0", lw=0.9, ls="--", alpha=0.7,
                    label="PyFibers" if i == 0 else None)
        ax.set_xlim(0, TSTOP)
        ax.set_xlabel("time (ms)", fontsize=9)
        ax.set_title(f"{res['amp_mA']:.3f} mA  ({res['amp_factor']}× thr)",
                      fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3, lw=0.4)
        if c == 0:
            ax.set_ylabel("node position along fiber  (mm)", fontsize=9)
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"Multiple conduction responses to cathodic stimulation — "
        f"MRG D = {DIAMETER} µm, PW = {PW_MS} ms\n"
        f"Each trace = V$_m$(t) at one node, vertically offset by node position",
        fontsize=10,
    )
    path = OUT / "fig_dc_block.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  -> {path}", flush=True)

    # ── save data ────────────────────────────────────────────────────────────
    save_json({
        "diameter_um":   DIAMETER,
        "n_nodes":       N_NODES,
        "pw_ms":         PW_MS,
        "delay_ms":      DELAY,
        "dt_ms":         DT,
        "tstop_ms":      TSTOP,
        "src_height_um": SRC_H,
        "sigma_S_m":     SIGMA,
        "threshold_mA":  thr,
        "amp_factors":   AMP_FACTORS,
        "node_pos_mm":   results[0]["node_pos_mm"].tolist(),
        "results": [
            {
                "amp_mA":       r["amp_mA"],
                "amp_factor":   r["amp_factor"],
                "jax_t_ms":     r["jax_t_ms"].tolist(),
                "jax_vm_nodes": r["jax_vm_nodes"].tolist(),
                "pf_t_ms":      r["pf_t_ms"].tolist(),
                "pf_vm_nodes":  r["pf_vm_nodes"].tolist(),
            }
            for r in results
        ],
    }, OUT / "data_dc_block.json")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
