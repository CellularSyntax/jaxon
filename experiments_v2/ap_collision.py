"""Action potential collision — emergent phenomenon demo (Hussain 2024 Fig 3c).

Two intracellular pulses, one at each end of an MRG fiber, generate APs that
propagate inward toward each other.  Inside the refractory wake there's no
substrate for forward propagation, so the two APs mutually annihilate when
they meet near the midpoint.

This panel is "emergent" in the sense that the coupled (Vi, Vpax) solver was
never tuned for AP collision — it just *works*, because the cable equation
and the MRG channel kinetics handle refractoriness correctly.  Reproducing
this with a learned surrogate would require it to have seen collision
examples during training; the exact biophysics gets it for free.

Setup
-----
* MRG fiber, 4 diameters (5.7, 8.7, 10.0, 14.0 µm), 21 nodes each.
* Two intracellular pulses (2 nA, 0.1 ms) injected simultaneously at the
  first and last node of Ranvier.
* No extracellular stimulation.
* Snapshots of V_m along the full compartment chain at 5 timepoints
  matching Hussain 2024 Fig 3c (0.125, 0.5, 1.0, 1.5, 2.0 ms after pulse).

Outputs (outputs/ap_collision/)
-------------------------------
  data_ap_collision.json    — V_m snapshots + metadata
  fig_ap_collision.png      — 4×5 panel: rows = diameter, cols = timepoint

Run from project root:
    python experiments_v2/ap_collision.py
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

from jaxfibers.fibers.mrg import (
    build_mrg, node_indices, section_centers_um,
    V_REST, CM_AXON, G_PAS_MYSA, G_PAS_FLUT, G_PAS_STIN,
)
from jaxfibers.channels.mrg_axnode import AxnodeMyel
from jaxfibers.stim.extracellular_coupled import arrays_from_geometry, integrate
from jaxfibers.nrn_baseline import build_mrg_pyfibers
from neuron import h

from experiments_v2.utils import ensure_dir, save_json

OUT = ensure_dir(ROOT / "outputs" / "ap_collision")

# ── shared constants ──────────────────────────────────────────────────────────
CELSIUS  = 37.0
DT       = 0.005      # ms
TSTOP    = 2.5        # ms — covers all snapshot timepoints
DELAY    = 0.1        # ms — pulse onset (small to capture early snapshots)
# N_NODES=101 matches Hussain Fig 3c (x-axis 0-100 nodes).  For D=14 µm with
# INL ~1.4 mm this is a ~140 mm fiber; APs collide near the midpoint around
# t ~ 0.9 ms, leaving snapshots at t = {0.125, 0.5, 1.0, 1.5, 2.0} ms in
# pre-collision, near-collision, and post-annihilation regimes respectively.
# Shorter fibers (N=21) make the APs annihilate before the first non-trivial
# snapshot is reached.
N_NODES  = 101
N_STEPS  = int(TSTOP / DT)

# MRG channel constants (identical to mrg_validation.py)
GNABAR = 3.0;  GNAPBAR = 0.01; GKBAR = 0.08; GL = 0.007
ENA    = 50.0; EK      = -90.0; EL    = -90.0

DIAMETERS    = [5.7, 8.7, 10.0, 14.0]
REP_DIAM_UM  = 10.0   # representative diameter for the main-figure waterfall
PULSE_AMP_NA = 2.0
PULSE_PW_MS  = 0.1
SNAPSHOT_TS  = [0.125, 0.5, 1.0, 1.5, 2.0]   # ms after pulse onset (== Hussain Fig 3c)


def _build_membrane_fn(static, geom):
    """Membrane function compatible with extracellular_coupled.integrate."""
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


def _run_one_diameter(D: float) -> dict:
    """Run the AP-collision simulation for one fiber diameter."""
    print(f"\n=== D = {D} µm ===", flush=True)
    _, geom = build_mrg(diameter=D, n_nodes=N_NODES)
    nodes   = node_indices(geom)
    centers = np.array(section_centers_um(geom))
    n_comp  = geom.n_comp

    # Coupled-solver setup (patch cm to CM_AXON consistently with the validation suite).
    geom_c = dataclasses.replace(geom, cm_uF_cm2=[CM_AXON] * n_comp)
    static = arrays_from_geometry(geom_c, DT)

    membrane_fn, state0 = _build_membrane_fn(static, geom)

    # Two intracellular pulses at the FIRST and LAST nodes of Ranvier,
    # simultaneous, both 2 nA, 0.1 ms.
    t_steps = (np.arange(N_STEPS) + 1) * DT
    pulse_on = (t_steps >= DELAY) & (t_steps < DELAY + PULSE_PW_MS)
    i_intra = np.zeros((N_STEPS, n_comp), dtype=np.float64)
    i_intra[pulse_on, nodes[0]]  = PULSE_AMP_NA
    i_intra[pulse_on, nodes[-1]] = PULSE_AMP_NA

    # No extracellular field
    Ve_zero    = jnp.zeros(n_comp, dtype=jnp.float64)
    pulse_mask = jnp.zeros(N_STEPS, dtype=jnp.float64)

    print(f"  Integrating ({N_STEPS} steps, n_comp={n_comp}) ...", flush=True)
    (Vi_all, Vp_all), _ = integrate(
        static, membrane_fn, state0,
        Ve_zero, pulse_mask, DT,
        v_rest=V_REST, record="all",
        i_intra=jnp.asarray(i_intra),
    )
    Vm_all = np.asarray(Vi_all) - np.asarray(Vp_all)   # [nsteps, n_comp]

    # Find the timestep indices for each requested snapshot time.
    # SNAPSHOT_TS are measured *from pulse onset*, so the absolute time is
    # DELAY + snapshot.  output[s] corresponds to t = (s+1)*dt (see integrate).
    snap_idx = [int(round((DELAY + s) / DT)) - 1 for s in SNAPSHOT_TS]
    snapshots = np.stack([Vm_all[i] for i in snap_idx], axis=0)   # [n_snap, n_comp]

    # Filter to node compartments only — matches Hussain Fig 3c (x = node #).
    # Internode compartments stay near rest, dominating the visual otherwise.
    snapshots_nodes = snapshots[:, nodes]                          # [n_snap, n_nodes]

    # ── PyFibers comparison: two simultaneous IClamps at the end nodes ───────
    print(f"  PyFibers comparison ...", flush=True)
    fiber = build_mrg_pyfibers(diameter=D, n_nodes=N_NODES, temperature=CELSIUS)
    fiber.record_vm()
    # IClamp at the first and last node-of-Ranvier sections (centre = 0.5).
    # h.IClamp.amp is in nA, matching our PULSE_AMP_NA convention.
    iclamp0 = h.IClamp(fiber[0](0.5))
    iclamp0.delay = DELAY; iclamp0.dur = PULSE_PW_MS; iclamp0.amp = PULSE_AMP_NA
    iclampN = h.IClamp(fiber[-1](0.5))
    iclampN.delay = DELAY; iclampN.dur = PULSE_PW_MS; iclampN.amp = PULSE_AMP_NA
    # Recordings must be set up BEFORE h.finitialize, else they capture nothing.
    t_vec = h.Vector().record(h._ref_t)
    h.celsius = CELSIUS
    h.dt = DT
    h.finitialize(fiber.v_rest)
    h.continuerun(TSTOP)
    vm_pf = np.array([np.array(v) for v in fiber.vm])     # [n_nodes, n_t]
    t_pf  = np.array(t_vec)
    # Resample PyFibers traces to the same time grid as JAX (output[s] at t=(s+1)*dt)
    pf_snap_indices = [np.argmin(np.abs(t_pf - (DELAY + s))) for s in SNAPSHOT_TS]
    snapshots_nodes_pf = vm_pf[:, pf_snap_indices].T               # [n_snap, n_nodes]
    print(f"    JAX peak {Vm_all.max():.1f} mV, PyFibers peak {vm_pf.max():.1f} mV  "
          f"(|diff|={abs(Vm_all.max() - vm_pf.max()):.2f} mV)", flush=True)

    return {
        "diameter_um":        D,
        "n_comp":             n_comp,
        "n_nodes":            len(nodes),
        "node_indices":       nodes,
        "centers_um":         centers,
        "snapshots":          snapshots,                             # [n_snap, n_comp]
        "snapshots_nodes":    snapshots_nodes,                       # [n_snap, n_nodes]
        "snapshots_nodes_pf": snapshots_nodes_pf,                    # [n_snap, n_nodes]
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
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(2.4 * n_cols, 2.0 * n_rows),
                              sharey=True, constrained_layout=True)
    if n_rows == 1:
        axes = axes[None, :]

    # Hussain Fig 3c: x-axis = node number (0..n_nodes-1), node-only Vm.
    # PyFibers (blue, solid) and jaxfibers (orange, dashed) — matches Hussain
    # convention where NEURON is solid blue and S-MF is dashed orange.
    for r, res in enumerate(results):
        D = res["diameter_um"]
        n_nodes_r = res["n_nodes"]
        node_x = np.arange(n_nodes_r)
        for c, t in enumerate(SNAPSHOT_TS):
            ax = axes[r, c]
            ax.plot(node_x, res["snapshots_nodes_pf"][c], color="C0", lw=1.6,
                    label="PyFibers" if (r == 0 and c == 0) else None)
            ax.plot(node_x, res["snapshots_nodes"][c],    color="C1", lw=1.2, ls="--",
                    label="jaxfibers" if (r == 0 and c == 0) else None)
            ax.axhline(V_REST, color="gray", lw=0.4, ls=":")
            ax.set_xlim(0, n_nodes_r - 1)
            ax.set_ylim(-90, 50)
            if r == 0:
                ax.set_title(f"t = {t} ms", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"D={D} µm\nV$_m$ (mV)", fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel("node #", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3, lw=0.4)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("AP collision — symmetric annihilation in MRG fibers "
                 f"(intracellular pulse {PULSE_AMP_NA} nA × {PULSE_PW_MS} ms "
                 "at both ends)", fontsize=10)
    path = OUT / "fig_ap_collision.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  -> {path}", flush=True)


def main():
    print(f"=== AP collision demo (Hussain 2024 Fig 3c equivalent) ===")
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

    # Save the data
    save_json({
        "diameters_um":   DIAMETERS,
        "snapshot_t_ms":  SNAPSHOT_TS,
        "pulse_amp_nA":   PULSE_AMP_NA,
        "pulse_pw_ms":    PULSE_PW_MS,
        "delay_ms":       DELAY,
        "dt_ms":          DT,
        "tstop_ms":       TSTOP,
        "waterfall":      waterfall,
        "n_nodes":        N_NODES,
        "celsius":        CELSIUS,
        "results":        [
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
    }, OUT / "data_ap_collision.json")

    make_figure(results)
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
