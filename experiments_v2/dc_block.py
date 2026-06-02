"""DC block — emergent phenomenon demo (Hussain 2024 Fig 3a).

A single rectangular cathodic monophasic pulse delivered extracellularly
to an MRG fiber produces qualitatively different responses as a function
of amplitude:

  * sub-threshold: no AP
  * just above threshold: single AP from the centre node, propagates
    bidirectionally to both ends
  * 2-3x threshold: AP fires at centre but Na inactivation under the
    sustained depolarisation prevents distal propagation — *bidirectional
    block* of the propagating AP
  * higher still: virtual-anode hyperpolarisation at the END nodes is
    released after the pulse and triggers anodal-break excitation at the
    ends — *re-excitation* with a delayed AP

These four regimes emerge from the cable equation + MRG channel kinetics
without any tuning.  This panel demonstrates that the coupled (V_i, V_pax)
solver reproduces complex propagation phenomena beyond the simple
activation regime — important for vagus-block applications where the same
electrode delivers stimulation that may transition between regimes during
a single therapy session.

Setup
-----
* MRG fiber, D=12 µm, N_NODES=101 (~145 mm fiber).
* Cathodic monophasic rectangular pulse, PW=0.75 ms.
* Extracellular field from a point source at y=1 mm above the fiber centre
  (sigma=0.3 S/m), aligned with the central node.
* Threshold found by JAX bisection; four amplitudes spanning
  sub-threshold to ~3x threshold.
* Snapshots of V_m along the node chain at t = {1.0, 1.5, 2.0, 2.5} ms.

Outputs (outputs/dc_block/)
---------------------------
  data_dc_block.json   — V_m snapshots + threshold + metadata
  fig_dc_block.png     — 4x4 panel: rows = amplitude, cols = timepoint

Run from project root:
    python experiments_v2/dc_block.py
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
from jaxfibers.stim.extracellular import point_source_potentials_mV
from jaxfibers.stim.extracellular_coupled import arrays_from_geometry, integrate

from experiments_v2.utils import (
    PULSES, make_pulse_array, jax_bisect, ensure_dir, save_json,
)

OUT = ensure_dir(ROOT / "outputs" / "dc_block")

# ── shared constants ──────────────────────────────────────────────────────────
CELSIUS  = 37.0
DT       = 0.005       # ms
TSTOP    = 3.0         # ms — covers snapshot range with pulse onset at 0.1 ms
DELAY    = 0.1         # ms
N_NODES  = 101
N_STEPS  = int(TSTOP / DT)
SIGMA    = 0.3         # S/m
SRC_H    = 1000.0      # µm — point source 1 mm above fibre centre

DIAMETER     = 12.8    # µm — closest MRG_DISCRETE diameter to Hussain's 12 µm
PW_MS        = 0.75    # ms
SNAPSHOT_TS  = [1.0, 1.5, 2.0, 2.5]   # ms (absolute time)
AMP_FACTORS  = [0.5, 1.05, 3.0, 15.0]  # x threshold
                                        # 0.5x: sub-threshold (no AP)
                                        # 1.05x: just suprathreshold (canonical
                                        #        bidirectional propagation)
                                        # 3.0x: strong activation, faster
                                        #       inward propagation
                                        # 15x: deep block regime — sustained
                                        #      depolarisation prevents normal
                                        #      AP formation at the centre; APs
                                        #      may only form from virtual-anode
                                        #      release at the ends (anodal-break)

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


def main():
    print(f"=== DC block demo (Hussain 2024 Fig 3a equivalent) ===")
    print(f"D = {DIAMETER} µm, N = {N_NODES} nodes, PW = {PW_MS} ms cathodic")

    # ── build fiber ──────────────────────────────────────────────────────────
    _, geom = build_mrg(diameter=DIAMETER, n_nodes=N_NODES)
    nodes   = node_indices(geom)
    centers = np.array(section_centers_um(geom))
    n_comp  = geom.n_comp
    mid_node = nodes[len(nodes) // 2]

    print(f"Fiber length: {centers[-1] / 1000:.1f} mm, n_comp = {n_comp}", flush=True)

    geom_c = dataclasses.replace(geom, cm_uF_cm2=[CM_AXON] * n_comp)
    static = arrays_from_geometry(geom_c, DT)
    membrane_fn, state0 = _build_membrane_fn(static, geom)

    Ve_unit = np.asarray(point_source_potentials_mV(
        list(centers), src_x_um=0., src_y_um=SRC_H,
        src_z_um=float(centers[mid_node]), i0_mA=-1.0,
    ))

    is_node_arr = static["is_node"]

    # ── compile a peak-Vm runner for bisection ───────────────────────────────
    @jax.jit
    def run_peak(amp_mA: float, pulse_arr: jnp.ndarray) -> float:
        Ve = jnp.asarray(Ve_unit, dtype=jnp.float64) * (amp_mA / -1.0)
        (Vi_all, Vp_all), _ = integrate(
            static, membrane_fn, state0,
            Ve, pulse_arr, DT, v_rest=V_REST, record="all",
        )
        Vm = Vi_all - Vp_all
        Vm_nodes = jnp.where(is_node_arr[None, :], Vm, -jnp.inf)
        return jnp.max(Vm_nodes)

    pulse_arr = make_pulse_array("mono_c", PW_MS, N_STEPS, DT, DELAY)
    spec = PULSES["mono_c"]

    print(f"\nFinding threshold (mono_c, PW = {PW_MS} ms) ...", flush=True)
    thr = jax_bisect(run_peak, pulse_arr, lo=spec.lo, hi=spec.hi)
    print(f"Threshold: {thr:.4f} mA", flush=True)

    # ── run at each amplitude, take snapshots ────────────────────────────────
    results = []
    for fac in AMP_FACTORS:
        amp = thr * fac
        print(f"\n  amp = {amp:.4f} mA  ({fac}x threshold) ...", flush=True)
        Ve = jnp.asarray(Ve_unit, dtype=jnp.float64) * (amp / -1.0)
        (Vi_all, Vp_all), _ = integrate(
            static, membrane_fn, state0,
            Ve, jnp.asarray(pulse_arr), DT,
            v_rest=V_REST, record="all",
        )
        Vm_all = np.asarray(Vi_all) - np.asarray(Vp_all)   # [nsteps, n_comp]

        # Snapshot indices: output[s] is at t = (s+1)*dt
        snap_idx = [int(round(s / DT)) - 1 for s in SNAPSHOT_TS]
        snapshots_all   = np.stack([Vm_all[i] for i in snap_idx], axis=0)
        snapshots_nodes = snapshots_all[:, nodes]

        results.append({
            "amp_mA":           amp,
            "amp_factor":       fac,
            "snapshots_all":    snapshots_all,
            "snapshots_nodes":  snapshots_nodes,
            "vm_peak_mV":       float(Vm_all.max()),
            "fires":            bool(Vm_all.max() > -20.0),
        })
        print(f"    vm_peak = {Vm_all.max():.1f} mV  "
              f"(fires = {Vm_all.max() > -20.0})", flush=True)

    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        len(AMP_FACTORS), len(SNAPSHOT_TS),
        figsize=(2.4 * len(SNAPSHOT_TS), 2.0 * len(AMP_FACTORS)),
        sharey=True, constrained_layout=True,
    )

    for r, res in enumerate(results):
        n_nodes_r = len(nodes)
        node_x = np.arange(n_nodes_r)
        for c, t in enumerate(SNAPSHOT_TS):
            ax = axes[r, c]
            ax.plot(node_x, res["snapshots_nodes"][c], color="C0", lw=1.4)
            ax.axhline(V_REST, color="gray", lw=0.4, ls=":")
            ax.set_xlim(0, n_nodes_r - 1)
            ax.set_ylim(-110, 50)
            if r == 0:
                ax.set_title(f"t = {t} ms", fontsize=9)
            if c == 0:
                ax.set_ylabel(
                    f"{res['amp_mA']:.3f} mA\n({res['amp_factor']}x thr)\n"
                    "V$_m$ (mV)", fontsize=9,
                )
            if r == len(AMP_FACTORS) - 1:
                ax.set_xlabel("node #", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3, lw=0.4)

    fig.suptitle(
        f"DC block — MRG D={DIAMETER} µm, PW={PW_MS} ms cathodic, "
        f"point source 1 mm above center", fontsize=10,
    )
    path = OUT / "fig_dc_block.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  -> {path}", flush=True)

    # ── save data ────────────────────────────────────────────────────────────
    save_json({
        "diameter_um":  DIAMETER,
        "n_nodes":      N_NODES,
        "pw_ms":        PW_MS,
        "delay_ms":     DELAY,
        "dt_ms":        DT,
        "tstop_ms":     TSTOP,
        "celsius":      CELSIUS,
        "src_height_um": SRC_H,
        "sigma_S_m":    SIGMA,
        "threshold_mA": thr,
        "snapshot_t_ms": SNAPSHOT_TS,
        "amp_factors":  AMP_FACTORS,
        "results": [
            {
                "amp_mA":            r["amp_mA"],
                "amp_factor":        r["amp_factor"],
                "vm_peak_mV":        r["vm_peak_mV"],
                "fires":             r["fires"],
                "snapshots_nodes_mV": r["snapshots_nodes"].tolist(),
            }
            for r in results
        ],
    }, OUT / "data_dc_block.json")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
