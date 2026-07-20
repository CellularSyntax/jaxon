"""Multiple conduction responses — Rattay C-fiber (waterfall visualisation).

Analogous to dc_block_sundt.py but for the Rattay HH unmyelinated C-fiber model.

Setup
-----
* Rattay fiber, D=0.8 µm, N_NODES=101 (101 × 8.333 µm ≈ 0.84 mm)
* Point-source extracellular at y=1 mm above fibre centre
* Cathodic monophasic pulse, PW=0.2 ms
* 4 amplitudes: 0.5×, 1.1×, 1.5×, 2.5× threshold

Outputs (outputs/dc_block_rattay/)
----------------------------------
  data_dc_block_rattay.json
  fig_dc_block_rattay.png

Run from project root:
    python experiments_v2/dc_block_rattay.py
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

from jaxon.fibers.rattay import (
    build_rattay, node_indices, section_centers_um,
    V_REST, CM,
)
from jaxon.channels.rattay_channels import RattayHH
from jaxon.stim.extracellular import point_source_potentials_mV
from jaxon.stim.extracellular_coupled import arrays_from_geometry, integrate
from jaxon.nrn_baseline import build_rattay_pyfibers
from pyfibers import ScaledStim
from scipy.interpolate import interp1d

from experiments_v2.utils import (
    PULSES, make_pulse_array, jax_bisect, ensure_dir, save_json,
)

OUT = ensure_dir(ROOT / "outputs" / "dc_block_rattay")

# ── shared constants ──────────────────────────────────────────────────────────
CELSIUS     = 37.0
DT          = 0.005     # ms
TSTOP       = 10.0      # ms — covers full AP propagation across 0.84 mm C-fiber
DELAY       = 0.5       # ms
N_NODES     = 101       # 101 × 8.333 µm ≈ 0.84 mm fiber
N_STEPS     = int(TSTOP / DT)
SIGMA       = 0.3       # S/m
SRC_H       = 1000.0    # µm

DIAMETER    = 0.8       # µm — representative Rattay C-fiber
PW_MS       = 0.2       # ms — longer than MRG (C-fibers need wider pulses)
AMP_FACTORS = [0.5, 1.1, 1.5, 2.5]

# RattayHH default channel parameters (identical to rattay_validation.py)
GNABAR = 0.12     # S/cm²
GKBAR  = 0.036    # S/cm²
GL     = 3e-4     # S/cm²
EL     = -59.4    # mV
ENA    = 45.0     # mV
EK     = -82.0    # mV


def _make_jax_setup(diameter: float, n_nodes: int):
    _, geom = build_rattay(diameter=diameter, n_nodes=n_nodes)
    nodes   = node_indices(geom)
    centers = np.array(section_centers_um(geom))
    mid     = nodes[len(nodes) // 2]
    n_comp  = geom.n_comp

    geom_c = dataclasses.replace(geom, cm_uF_cm2=[CM] * n_comp)
    static  = arrays_from_geometry(geom_c, DT)

    Ve_unit = np.array(point_source_potentials_mV(
        list(centers), src_x_um=0., src_y_um=SRC_H,
        src_z_um=float(centers[mid]), i0_mA=-1.0,
    ))

    A_in = static["A_in_cm2"]

    # Steady-state gate values at V_REST
    v0 = jnp.float64(V_REST)
    (am0, bm0), (ah0, bh0), (an0, bn0) = RattayHH._alpha_beta(v0, CELSIUS)
    m0 = float(am0 / (am0 + bm0))
    h0 = float(ah0 / (ah0 + bh0))
    n0 = float(an0 / (an0 + bn0))
    state0 = (
        jnp.full(n_comp, m0, dtype=jnp.float64),
        jnp.full(n_comp, h0, dtype=jnp.float64),
        jnp.full(n_comp, n0, dtype=jnp.float64),
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

    return static, membrane_fn, state0, Ve_unit, nodes, centers, geom


def _run_pyfibers(amp_mA: float, diameter: float, n_nodes: int):
    fiber = build_rattay_pyfibers(diameter=diameter, n_nodes=n_nodes, temperature=CELSIUS)
    fiber.record_vm()
    fiber.potentials = fiber.point_source_potentials(
        x=0.0, y=SRC_H, z=fiber.length / 2.0, i0=amp_mA, sigma=SIGMA,
    )
    pulse_arr = make_pulse_array("mono_c", PW_MS, N_STEPS, DT, DELAY)
    t_pts = np.concatenate([[0.0], (np.arange(len(pulse_arr)) + 1) * DT])
    v_pts = np.concatenate([[0.0], pulse_arr])
    f   = interp1d(t_pts, v_pts, bounds_error=False, fill_value=0.0)
    wav = lambda t: float(f(t))
    stim = ScaledStim(waveform=wav, dt=DT, tstop=TSTOP)
    stim.run_sim(stimamp=1.0, fiber=fiber, ap_detect_location=0.5,
                 fail_on_end_excitation=False)
    vm   = np.array([np.array(v) for v in fiber.vm])
    t_ms = np.array(fiber.time)
    return t_ms, vm


def main():
    print(f"=== DC block — Rattay C-fiber D={DIAMETER} µm, PW={PW_MS} ms ===\n")

    static, membrane_fn, state0, Ve_unit, nodes, centers, geom = \
        _make_jax_setup(DIAMETER, N_NODES)
    n_comp      = geom.n_comp
    is_node_arr = static["is_node"]

    print(f"Fiber length: {centers[-1]/1000:.3f} mm  n_comp={n_comp}", flush=True)

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
    spec      = PULSES["mono_c"]

    print("Finding threshold ...", flush=True)
    thr = jax_bisect(run_peak, pulse_arr, lo=spec.lo, hi=spec.hi)
    print(f"Threshold (JAX): {thr:.4f} mA\n", flush=True)

    t_jax_axis = (np.arange(N_STEPS) + 1) * DT
    results = []
    for fac in AMP_FACTORS:
        amp = thr * fac
        print(f"  amp = {amp:.4f} mA  ({fac}× threshold)", flush=True)

        t0 = time.time()
        Vm_all    = np.asarray(run_full(jnp.float64(amp), pulse_j))
        jax_nodes = Vm_all[:, nodes]
        print(f"    JAX done in {time.time()-t0:.1f}s  peak {jax_nodes.max():.1f} mV",
              flush=True)

        t0 = time.time()
        t_pf, vm_pf = _run_pyfibers(amp, DIAMETER, N_NODES)
        print(f"    PyFibers done in {time.time()-t0:.1f}s  peak {vm_pf.max():.1f} mV",
              flush=True)

        results.append({
            "amp_mA":       amp,
            "amp_factor":   fac,
            "jax_t_ms":     t_jax_axis,
            "jax_vm_nodes": jax_nodes,        # [N_STEPS, n_nodes]
            "pf_t_ms":      t_pf,
            "pf_vm_nodes":  vm_pf.T,           # [n_t, n_nodes]
            "node_pos_mm":  centers[nodes] / 1000.0,
        })

    # ── quick sanity figure ───────────────────────────────────────────────────
    inter_node_mm = results[0]["node_pos_mm"][1] - results[0]["node_pos_mm"][0]
    v_range       = 150.0
    scale         = (inter_node_mm * 0.7) / v_range
    fig, axes = plt.subplots(1, len(AMP_FACTORS), figsize=(3.6*len(AMP_FACTORS), 5),
                              sharey=True, constrained_layout=True)
    for c, res in enumerate(results):
        ax   = axes[c]
        npos = res["node_pos_mm"]
        for i, y0 in enumerate(npos):
            ax.plot(res["jax_t_ms"],
                    y0 + (res["jax_vm_nodes"][:, i] - V_REST) * scale,
                    color="C1", lw=0.8)
            ax.plot(res["pf_t_ms"],
                    y0 + (res["pf_vm_nodes"][:, i] - V_REST) * scale,
                    color="C0", lw=0.8, ls="--", alpha=0.7)
        ax.set_xlim(0, TSTOP)
        ax.set_xlabel("time (ms)")
        ax.set_title(f"{res['amp_mA']:.3f} mA ({res['amp_factor']}×)")
        if c == 0:
            ax.set_ylabel("node position (mm)")
    fig.suptitle(f"DC block — Rattay D={DIAMETER} µm  PW={PW_MS} ms", fontsize=10)
    path = OUT / "fig_dc_block_rattay.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  -> {path}", flush=True)

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
    }, OUT / "data_dc_block_rattay.json")
    print(f"  -> {OUT / 'data_dc_block_rattay.json'}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
