"""Action potential collision — Sundt C-fiber.

Analogous to ap_collision.py but for the Sundt (2015) unmyelinated C-fiber model.

Two intracellular pulses at each end of the fiber generate APs that propagate
inward and annihilate at the midpoint, demonstrating the same emergent
refractoriness-driven collision in the C-fiber model.

Setup
-----
* Sundt fiber, 4 diameters (0.3, 0.5, 0.8, 1.0 µm), 51 compartments each
  (51 × 8.333 µm ≈ 0.425 mm)
* Two intracellular pulses (0.5 nA, 0.2 ms) at the first and last compartment
* Snapshots at 5 time points scaled for C-fiber conduction velocities

Outputs (outputs/ap_collision_sundt/)
---------------------------------------
  data_ap_collision_sundt.json
  fig_ap_collision_sundt.png

Run from project root:
    python experiments_v2/ap_collision_sundt.py
"""

from __future__ import annotations

import sys
import pathlib
import dataclasses

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

from jaxfibers.fibers.sundt import (
    build_sundt, node_indices, section_centers_um,
    V_REST, CM,
)
from jaxfibers.channels.sundt_channels import SundtAxon
from jaxfibers.stim.extracellular_coupled import arrays_from_geometry, integrate
from jaxfibers.nrn_baseline import build_sundt_pyfibers
from neuron import h

from experiments_v2.utils import ensure_dir, save_json

OUT = ensure_dir(ROOT / "outputs" / "ap_collision_sundt")

# ── shared constants ──────────────────────────────────────────────────────────
CELSIUS  = 37.0
DT       = 0.005    # ms
# C-fibers: 51 × 8.333 µm = 425 µm fiber; CV 0.2–0.7 mm/ms → collision in 0.3–1.1 ms
# TSTOP covers the slowest diameter (D=0.3 µm, collision ~1.05 ms) plus AP tail
DELAY    = 0.1      # ms — small offset before pulse, matches MRG convention
TSTOP    = 5.0      # ms
N_NODES  = 51
N_STEPS  = int(TSTOP / DT)

# SundtAxon channel constants
GNABAR  = 0.04;  GKDRBAR = 0.04;  G_PAS  = 1e-4
ENA     = 50.0;  EK      = -90.0; E_PAS  = V_REST
MSHIFT  = -6.0;  HSHIFT  =  6.0;  ISHIFT = 0.0
VHALFN  = -32.0; VHALFL  = -61.0
A0N     =  0.03; A0L     =  0.001
ZETAN   = -5.0;  ZETAL   =  2.0
GMN     =  0.4;  GML     =  1.0

DIAMETERS    = [0.3, 0.5, 0.8, 1.0]   # µm
REP_DIAM_UM  = 1.0                     # representative diameter for the main-figure waterfall
PULSE_AMP_NA = 0.5                     # nA — reliable for Sundt
PULSE_PW_MS  = 0.2                     # ms
# Snapshot times from pulse onset; chosen so pre-/during-/post-collision
# phases are visible across all four diameters
SNAPSHOT_TS  = [0.05, 0.2, 0.5, 1.0, 2.0]   # ms after pulse onset


def _build_membrane_fn(static, n_comp: int):
    A_in = static["A_in_cm2"]
    v0   = jnp.float64(V_REST)
    (a_m0, b_m0), (a_h0, b_h0) = SundtAxon._nahh_alpha_beta(
        v0, CELSIUS, MSHIFT, HSHIFT, ISHIFT)
    (a_n0, b_n0), (a_l0, b_l0) = SundtAxon._borgkdr_alpha_beta(
        v0, CELSIUS, VHALFN, VHALFL, A0N, A0L, ZETAN, ZETAL, GMN, GML)
    state0 = (
        jnp.full(n_comp, float(a_m0 / (a_m0 + b_m0)), dtype=jnp.float64),
        jnp.full(n_comp, float(a_h0 / (a_h0 + b_h0)), dtype=jnp.float64),
        jnp.full(n_comp, float(a_n0 / (a_n0 + b_n0)), dtype=jnp.float64),
        jnp.full(n_comp, float(a_l0 / (a_l0 + b_l0)), dtype=jnp.float64),
    )

    def membrane_fn(Vm, state, dt):
        M, H, N, L = state
        (a_m, b_m), (a_h, b_h) = SundtAxon._nahh_alpha_beta(
            Vm, CELSIUS, MSHIFT, HSHIFT, ISHIFT)
        M2 = solve_gate_exponential(M, dt, a_m, b_m)
        H2 = solve_gate_exponential(H, dt, a_h, b_h)
        (a_n, b_n), (a_l, b_l) = SundtAxon._borgkdr_alpha_beta(
            Vm, CELSIUS, VHALFN, VHALFL, A0N, A0L, ZETAN, ZETAL, GMN, GML)
        N2 = solve_gate_exponential(N, dt, a_n, b_n)
        L2 = solve_gate_exponential(L, dt, a_l, b_l)
        g_na  = GNABAR  * M2**3 * H2
        g_k   = GKDRBAR * N2**3 * L2
        g_tot = (g_na + g_k + G_PAS) * A_in * 1e6
        i_ion = (g_na*(Vm-ENA) + g_k*(Vm-EK) + G_PAS*(Vm-E_PAS)) * A_in * 1e6
        return g_tot, i_ion, (M2, H2, N2, L2)

    return membrane_fn, state0


def _run_one_diameter(D: float) -> dict:
    print(f"\n=== D = {D} µm ===", flush=True)
    _, geom = build_sundt(diameter=D, n_nodes=N_NODES)
    nodes   = node_indices(geom)
    centers = np.array(section_centers_um(geom))
    n_comp  = geom.n_comp

    geom_c = dataclasses.replace(geom, cm_uF_cm2=[CM] * n_comp)
    static  = arrays_from_geometry(geom_c, DT)
    membrane_fn, state0 = _build_membrane_fn(static, n_comp)

    # Intracellular pulses at both ends
    t_steps  = (np.arange(N_STEPS) + 1) * DT
    pulse_on = (t_steps >= DELAY) & (t_steps < DELAY + PULSE_PW_MS)
    i_intra  = np.zeros((N_STEPS, n_comp), dtype=np.float64)
    i_intra[pulse_on, nodes[0]]  = PULSE_AMP_NA
    i_intra[pulse_on, nodes[-1]] = PULSE_AMP_NA

    Ve_zero    = jnp.zeros(n_comp, dtype=jnp.float64)
    pulse_mask = jnp.zeros(N_STEPS, dtype=jnp.float64)

    print(f"  Integrating ({N_STEPS} steps, n_comp={n_comp}) ...", flush=True)
    (Vi_all, Vp_all), _ = integrate(
        static, membrane_fn, state0,
        Ve_zero, pulse_mask, DT,
        v_rest=V_REST, record="all",
        i_intra=jnp.asarray(i_intra),
    )
    Vm_all = np.asarray(Vi_all) - np.asarray(Vp_all)   # [n_steps, n_comp]

    snap_idx       = [int(round((DELAY + s) / DT)) - 1 for s in SNAPSHOT_TS]
    snapshots      = np.stack([Vm_all[i] for i in snap_idx], axis=0)   # [n_snap, n_comp]
    snapshots_nodes = snapshots[:, nodes]                                 # [n_snap, n_nodes]

    # ── PyFibers comparison ───────────────────────────────────────────────────
    print(f"  PyFibers comparison ...", flush=True)
    fiber = build_sundt_pyfibers(diameter=D, n_nodes=N_NODES, temperature=CELSIUS)
    fiber.record_vm()
    iclamp0 = h.IClamp(fiber[0](0.5))
    iclamp0.delay = DELAY;   iclamp0.dur = PULSE_PW_MS;  iclamp0.amp = PULSE_AMP_NA
    iclampN = h.IClamp(fiber[-1](0.5))
    iclampN.delay = DELAY;   iclampN.dur = PULSE_PW_MS;  iclampN.amp = PULSE_AMP_NA
    t_vec = h.Vector().record(h._ref_t)
    h.celsius = CELSIUS
    h.dt      = DT
    h.finitialize(fiber.v_rest)
    h.continuerun(TSTOP)
    vm_pf  = np.array([np.array(v) for v in fiber.vm])    # [n_nodes, n_t]
    t_pf   = np.array(t_vec)
    pf_snap_idx        = [np.argmin(np.abs(t_pf - (DELAY + s))) for s in SNAPSHOT_TS]
    snapshots_nodes_pf = vm_pf[:, pf_snap_idx].T           # [n_snap, n_nodes]
    print(f"    JAX peak {Vm_all.max():.1f} mV  PyFibers peak {vm_pf.max():.1f} mV"
          f"  |diff|={abs(Vm_all.max()-vm_pf.max()):.2f} mV", flush=True)

    return {
        "diameter_um":        D,
        "n_comp":             n_comp,
        "n_nodes":            len(nodes),
        "node_indices":       nodes,
        "centers_um":         centers,
        "snapshots":          snapshots,
        "snapshots_nodes":    snapshots_nodes,
        "snapshots_nodes_pf": snapshots_nodes_pf,
        "snapshot_t_ms":      SNAPSHOT_TS,
        "vm_peak_mV":         float(Vm_all.max()),
        "vm_peak_pf_mV":      float(vm_pf.max()),
        # Full node-vs-time waterfall (for the main-figure collision panel).
        "node_pos_mm":        centers[nodes] / 1000.0,
        "jax_wf":             Vm_all[:, nodes],
        "pf_wf":              vm_pf.T,
        "t_jax":              (np.arange(N_STEPS) + 1) * DT,
        "t_pf":               t_pf,
    }


def make_figure(results: list[dict]) -> None:
    n_rows = len(results)
    n_cols = len(SNAPSHOT_TS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4*n_cols, 2.0*n_rows),
                              sharey=True, constrained_layout=True)
    if n_rows == 1:
        axes = axes[None, :]
    for r, res in enumerate(results):
        D = res["diameter_um"]
        node_x = np.arange(res["n_nodes"])
        for c, t in enumerate(SNAPSHOT_TS):
            ax = axes[r, c]
            ax.plot(node_x, res["snapshots_nodes_pf"][c], color="C0", lw=1.4,
                    label="PyFibers" if (r == 0 and c == 0) else None)
            ax.plot(node_x, res["snapshots_nodes"][c], color="C1", lw=1.1, ls="--",
                    label="JAXON" if (r == 0 and c == 0) else None)
            ax.axhline(V_REST, color="gray", lw=0.4, ls=":")
            ax.set_xlim(0, res["n_nodes"] - 1)
            ax.set_ylim(-70, 50)
            if r == 0:
                ax.set_title(f"t = {DELAY+t:.2f} ms", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"D={D} µm\nV$_m$ (mV)", fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel("node #", fontsize=9)
            ax.tick_params(labelsize=7)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"AP collision — Sundt C-fiber  ({PULSE_AMP_NA} nA × {PULSE_PW_MS} ms at both ends)",
                 fontsize=10)
    path = OUT / "fig_ap_collision_sundt.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  -> {path}", flush=True)


def main():
    print(f"=== AP collision — Sundt C-fiber ===")
    results = [_run_one_diameter(D) for D in DIAMETERS]

    _rep = next((r for r in results
                 if abs(r["diameter_um"] - REP_DIAM_UM) < 1e-6), results[-1])
    waterfall = {
        "diameter_um":  _rep["diameter_um"],
        "node_pos_mm":  _rep["node_pos_mm"].tolist(),
        "jax_t_ms":     _rep["t_jax"].tolist(),
        "jax_vm_nodes": _rep["jax_wf"].tolist(),
        "pf_t_ms":      _rep["t_pf"].tolist(),
        "pf_vm_nodes":  _rep["pf_wf"].tolist(),
    }

    save_json({
        "diameters_um":   DIAMETERS,
        "snapshot_t_ms":  SNAPSHOT_TS,
        "pulse_amp_nA":   PULSE_AMP_NA,
        "pulse_pw_ms":    PULSE_PW_MS,
        "delay_ms":       DELAY,
        "dt_ms":          DT,
        "tstop_ms":       TSTOP,
        "n_nodes":        N_NODES,
        "celsius":        CELSIUS,
        "waterfall":      waterfall,
        "results": [
            {
                "diameter_um":            r["diameter_um"],
                "n_nodes":                r["n_nodes"],
                "centers_um":             r["centers_um"].tolist(),
                "node_indices":           r["node_indices"],
                "snapshots_nodes_mV":     r["snapshots_nodes"].tolist(),
                "snapshots_nodes_pf_mV":  r["snapshots_nodes_pf"].tolist(),
                "snapshots_all_mV":       r["snapshots"].tolist(),
                "vm_peak_mV":             r["vm_peak_mV"],
                "vm_peak_pf_mV":          r["vm_peak_pf_mV"],
            }
            for r in results
        ],
    }, OUT / "data_ap_collision_sundt.json")
    print(f"  -> {OUT / 'data_ap_collision_sundt.json'}")

    make_figure(results)
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
