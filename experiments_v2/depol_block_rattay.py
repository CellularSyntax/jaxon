"""DC depolarization block — Rattay C-fiber (waterfall, one-arm block).

A single AP is initiated intracellularly at the MIDDLE node and propagates in
both directions.  A sustained cathodic extracellular field at an offset node
(the "block node", on the lower arm) holds that region in Na inactivation, so the
downward-propagating AP is blocked there while the upward arm propagates freely —
an asymmetric "V" in the node-vs-time waterfall.  Single blocking amplitude.

Deterministic DC phenomenon, so JAX and PyFibers match node-for-node.

Outputs (outputs/depol_block_rattay/)
-------------------------------------
  data_depol_block_rattay.json
  fig_depol_block_rattay.png

Run from project root:
    python experiments_v2/depol_block_rattay.py
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

from jaxfibers.fibers.rattay import (
    build_rattay, node_indices, section_centers_um,
    V_REST, CM,
)
from jaxfibers.channels.rattay_channels import RattayHH
from jaxfibers.stim.extracellular import point_source_potentials_mV
from jaxfibers.stim.extracellular_coupled import arrays_from_geometry, integrate
from jaxfibers.nrn_baseline import build_rattay_pyfibers
from pyfibers import ScaledStim

from experiments_v2.utils import ensure_dir, save_json

OUT = ensure_dir(ROOT / "outputs" / "depol_block_rattay")

# ── shared constants ──────────────────────────────────────────────────────────
CELSIUS   = 37.0
DT        = 0.005    # ms
TSTOP     = 6.0      # ms — covers AP propagation across the fiber
DELAY     = 1.0      # ms — init pulse onset (block field already settled)
N_NODES   = 101
N_STEPS   = int(TSTOP / DT)
SIGMA     = 10.0     # S/m (block electrode)
SRC_H     = 100.0    # µm — block electrode height above fibre

INIT_AMP_NA = 2.0    # nA — intracellular initiation pulse at middle node
INIT_PW_MS  = 0.2    # ms
BLOCK_FRAC  = 0.25   # block node position (fraction of fibre length from node 0)
BLOCK_AMP_MA = -15.0 # mA — single cathodic blocking amplitude

DIAMETER  = 0.8      # µm

# RattayHH default channel parameters
GNABAR = 0.12;  GKBAR = 0.036;  GL = 3e-4
EL = -59.4;  ENA = 45.0;  EK = -82.0


def _make_jax_setup():
    _, geom = build_rattay(diameter=DIAMETER, n_nodes=N_NODES)
    nodes   = node_indices(geom)
    centers = np.array(section_centers_um(geom))
    n_comp  = geom.n_comp

    init_node  = nodes[len(nodes) // 2]
    block_node = nodes[int(round(BLOCK_FRAC * (len(nodes) - 1)))]

    geom_c = dataclasses.replace(geom, cm_uF_cm2=[CM] * n_comp)
    static  = arrays_from_geometry(geom_c, DT)

    # Block electrode: point source above the block node.
    Ve_block = np.asarray(point_source_potentials_mV(
        list(centers), src_x_um=0.0, src_y_um=SRC_H,
        src_z_um=float(centers[block_node]), i0_mA=1.0, sigma_S_m=SIGMA,
    ))

    A_in = static["A_in_cm2"]
    v0   = jnp.float64(V_REST)
    (am0, bm0), (ah0, bh0), (an0, bn0) = RattayHH._alpha_beta(v0, CELSIUS)
    state0 = (
        jnp.full(n_comp, float(am0 / (am0 + bm0)), dtype=jnp.float64),
        jnp.full(n_comp, float(ah0 / (ah0 + bh0)), dtype=jnp.float64),
        jnp.full(n_comp, float(an0 / (an0 + bn0)), dtype=jnp.float64),
    )

    def membrane_fn(Vm, state, dt):
        M, H, N = state
        (am, bm), (ah, bh), (an, bn) = RattayHH._alpha_beta(Vm, CELSIUS)
        M2 = solve_gate_exponential(M, dt, am, bm)
        H2 = solve_gate_exponential(H, dt, ah, bh)
        N2 = solve_gate_exponential(N, dt, an, bn)
        g_na  = GNABAR * M2**3 * H2
        g_k   = GKBAR  * N2**4
        g_tot = (g_na + g_k + GL) * A_in * 1e6
        i_ion = (g_na * (Vm - ENA) + g_k * (Vm - EK) + GL * (Vm - EL)) * A_in * 1e6
        return g_tot, i_ion, (M2, H2, N2)

    # Intracellular initiation pulse at the middle node.
    t_step  = (np.arange(N_STEPS) + 1) * DT
    i_intra = np.zeros((N_STEPS, n_comp), dtype=np.float64)
    on = (t_step >= DELAY) & (t_step < DELAY + INIT_PW_MS)
    i_intra[on, init_node] = INIT_AMP_NA

    # Sustained block field (constant for the whole run).
    pulse_mask = np.ones(N_STEPS, dtype=np.float64)

    return dict(
        geom=geom, static=static, membrane_fn=membrane_fn, state0=state0,
        nodes=nodes, centers=centers, init_node=init_node, block_node=block_node,
        Ve_block=Ve_block, i_intra=i_intra, pulse_mask=pulse_mask, t_axis=t_step,
    )


def _jax_waterfall(S, amp_mA: float) -> np.ndarray:
    Ve = jnp.asarray(S["Ve_block"], dtype=jnp.float64) * amp_mA
    (Vi_all, Vp_all), _ = integrate(
        S["static"], S["membrane_fn"], S["state0"],
        Ve, jnp.asarray(S["pulse_mask"]), DT, v_rest=V_REST, record="all",
        i_intra=jnp.asarray(S["i_intra"]),
    )
    return np.asarray(Vi_all) - np.asarray(Vp_all)   # [n_steps, n_comp]


def _pf_waterfall(S, amp_mA: float):
    fiber = build_rattay_pyfibers(diameter=DIAMETER, n_nodes=N_NODES, temperature=CELSIUS)
    fiber.record_vm()
    fiber.potentials = fiber.point_source_potentials(
        0.0, SRC_H, float(S["centers"][S["block_node"]]), 1.0, SIGMA,
    )
    fiber.add_intrinsic_activity(
        loc=0.5, start_time=DELAY, avg_interval=1e9, num_stims=1,
    )
    wav = lambda t: 1.0   # sustained block field
    stim = ScaledStim(waveform=wav, dt=DT, tstop=TSTOP)
    stim.run_sim(amp_mA, fiber)
    vm = np.array([np.array(v) for v in fiber.vm])   # [n_nodes, n_t]
    t  = np.array(stim.time)
    return t, vm.T                                   # [n_t, n_nodes]


def main():
    print(f"=== DC block (waterfall) — Rattay C-fiber  block={BLOCK_AMP_MA} mA ===\n")
    S = _make_jax_setup()
    nodes   = S["nodes"]
    centers = S["centers"]

    t0 = time.time()
    Vm_all   = _jax_waterfall(S, BLOCK_AMP_MA)
    jax_nodes = Vm_all[:, nodes]
    print(f"  JAX waterfall {jax_nodes.shape}  peak {jax_nodes.max():.1f} mV  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    t_pf, pf_nodes = _pf_waterfall(S, BLOCK_AMP_MA)
    print(f"  PyFibers waterfall {pf_nodes.shape}  peak {pf_nodes.max():.1f} mV  ({time.time()-t0:.1f}s)")

    t_jax = (np.arange(N_STEPS) + 1) * DT
    node_pos_mm = centers[nodes] / 1000.0
    block_pos_mm = float(centers[S["block_node"]]) / 1000.0
    init_pos_mm  = float(centers[S["init_node"]]) / 1000.0

    # ── sanity waterfall figure ───────────────────────────────────────────────
    inter = node_pos_mm[1] - node_pos_mm[0]
    scale = (inter * 0.7) / 150.0
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for i, y0 in enumerate(node_pos_mm):
        ax.plot(t_pf,  y0 + (pf_nodes[:, i]  - V_REST) * scale, color="C0", lw=0.6)
        ax.plot(t_jax, y0 + (jax_nodes[:, i] - V_REST) * scale, color="C1", lw=0.6, ls="--")
    ax.axhline(block_pos_mm, color="red", lw=0.8, ls=":")
    ax.set_xlim(0, TSTOP); ax.set_xlabel("time (ms)"); ax.set_ylabel("node position (mm)")
    ax.set_title(f"Rattay DC block waterfall — block {BLOCK_AMP_MA} mA @ {block_pos_mm:.2f} mm")
    fig.savefig(OUT / "fig_depol_block_rattay.png", dpi=150, bbox_inches="tight")
    print(f"  -> {OUT / 'fig_depol_block_rattay.png'}")

    save_json({
        "fiber_model":    "RATTAY",
        "diameter_um":    DIAMETER,
        "n_nodes":        N_NODES,
        "dt_ms":          DT,
        "tstop_ms":       TSTOP,
        "block_amp_mA":   BLOCK_AMP_MA,
        "block_pos_mm":   block_pos_mm,
        "init_pos_mm":    init_pos_mm,
        "node_pos_mm":    node_pos_mm.tolist(),
        "jax_t_ms":       t_jax.tolist(),
        "jax_vm_nodes":   jax_nodes.tolist(),
        "pf_t_ms":        t_pf.tolist(),
        "pf_vm_nodes":    pf_nodes.tolist(),
    }, OUT / "data_depol_block_rattay.json")
    print(f"  -> {OUT / 'data_depol_block_rattay.json'}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
