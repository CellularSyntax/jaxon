"""Action potential collision — Sweeney A-fiber.

Analogous to ap_collision.py (MRG) but for the Sweeney (1987) myelinated A-fiber model.

Two intracellular pulses at each end of the fiber generate APs that propagate
inward and annihilate at the midpoint, demonstrating refractoriness-driven
collision in the Sweeney A-fiber model.

Setup
-----
* Sweeney fiber, 4 diameters (5.7, 8.7, 10.0, 14.0 µm), 101 nodes each
* Two intracellular pulses (2.0 nA, 0.1 ms) at the first and last node
* Snapshots at 5 time points

Outputs (outputs/ap_collision_sweeney/)
---------------------------------------
  data_ap_collision_sweeney.json
  fig_ap_collision_sweeney.png

Run from project root:
    python experiments_v2/ap_collision_sweeney.py
"""

from __future__ import annotations

import sys
import pathlib

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

from jaxon.fibers.sweeney import (
    build_sweeney, node_indices, section_centers_um,
    V_REST,
)
from jaxon.channels.sweeney_channels import SweeneyNode
from jaxon.stim.extracellular_coupled import arrays_from_geometry, integrate
from jaxon.nrn_baseline import build_sweeney_pyfibers
from neuron import h

from experiments_v2.utils import ensure_dir, save_json

OUT = ensure_dir(ROOT / "outputs" / "ap_collision_sweeney")

# ── shared constants ──────────────────────────────────────────────────────────
CELSIUS  = 37.0
DT       = 0.005    # ms
DELAY    = 0.1      # ms — pulse onset
TSTOP    = 2.5      # ms — covers all snapshot timepoints
N_NODES  = 101
N_STEPS  = int(TSTOP / DT)

# SweeneyNode default channel parameters
GNABAR = 1.445    # S/cm²
GL     = 0.128    # S/cm²
EL     = -80.01   # mV
ENA    = 35.64    # mV

DIAMETERS    = [5.7, 8.7, 10.0, 14.0]   # µm — myelinated A-fiber range
REP_DIAM_UM  = 10.0                       # representative diameter for the main-figure waterfall
PULSE_AMP_NA = 2.0                       # nA
PULSE_PW_MS  = 0.1                       # ms
SNAPSHOT_TS  = [0.125, 0.5, 1.0, 1.5, 2.0]   # ms after pulse onset


def _build_membrane_fn(static, geom, n_comp: int):
    A_in = static["A_in_cm2"]
    has_channel = jnp.asarray(
        np.array([st == "node" for st in geom.section_type]), dtype=jnp.bool_)

    v0 = jnp.float64(V_REST)
    (am0, bm0), (ah0, bh0) = SweeneyNode._alpha_beta(v0)
    state0 = (
        jnp.where(has_channel, float(am0 / (am0 + bm0)), 0.0),
        jnp.where(has_channel, float(ah0 / (ah0 + bh0)), 0.0),
    )

    def membrane_fn(Vm, state, dt):
        M, H = state
        (am, bm), (ah, bh) = SweeneyNode._alpha_beta(Vm)
        M2 = solve_gate_exponential(M, dt, am, bm)
        H2 = solve_gate_exponential(H, dt, ah, bh)
        g_na   = GNABAR * M2**2 * H2
        g_node = (g_na + GL) * A_in * 1e6
        i_node = (g_na * (Vm - ENA) + GL * (Vm - EL)) * A_in * 1e6
        g_mye  = jnp.full_like(g_node, 1e-9)
        i_mye  = jnp.zeros_like(i_node)
        g_eff  = jnp.where(has_channel, g_node, g_mye)
        i_ion  = jnp.where(has_channel, i_node, i_mye)
        return g_eff, i_ion, (M2, H2)

    return membrane_fn, state0


def _run_one_diameter(D: float) -> dict:
    print(f"\n=== D = {D} µm ===", flush=True)
    _, geom = build_sweeney(diameter=D, n_nodes=N_NODES)
    nodes   = node_indices(geom)
    centers = np.array(section_centers_um(geom))
    n_comp  = geom.n_comp

    static = arrays_from_geometry(geom, DT)
    membrane_fn, state0 = _build_membrane_fn(static, geom, n_comp)

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
    fiber = build_sweeney_pyfibers(diameter=D, n_nodes=N_NODES, temperature=CELSIUS)
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
            ax.set_ylim(-100, 50)
            if r == 0:
                ax.set_title(f"t = {DELAY+t:.2f} ms", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"D={D} µm\nV$_m$ (mV)", fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel("node #", fontsize=9)
            ax.tick_params(labelsize=7)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"AP collision — Sweeney A-fiber  ({PULSE_AMP_NA} nA × {PULSE_PW_MS} ms at both ends)",
                 fontsize=10)
    path = OUT / "fig_ap_collision_sweeney.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  -> {path}", flush=True)


def main():
    print(f"=== AP collision — Sweeney A-fiber ===")
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
    }, OUT / "data_ap_collision_sweeney.json")
    print(f"  -> {OUT / 'data_ap_collision_sweeney.json'}")

    make_figure(results)
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
