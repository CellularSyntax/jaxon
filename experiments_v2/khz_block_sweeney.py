"""Kilohertz frequency block — Sweeney A-fiber.

Analogous to khz_block.py (MRG) but for the Sweeney (1987) myelinated A-fiber model.

Setup
-----
* Sweeney fiber, D=10 µm, N_NODES=25
* Point source at y=250 µm, sigma=10 S/m
* 20 kHz square wave, on from t=50 to t=100 ms
* Intrinsic pacing every 10 ms starting at t=15 ms, 14 pulses, amp=1.5 nA
* AP detection at loc=0.9

Outputs (outputs/khz_block_sweeney/)
------------------------------------
  data_khz_block_sweeney.json
  fig_khz_block_sweeney_traces.png

Run from project root:
    python experiments_v2/khz_block_sweeney.py
"""

from __future__ import annotations

import sys
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.signal as sg

import jax
import jax.numpy as jnp
from jaxley.solver_gate import solve_gate_exponential

jax.config.update("jax_enable_x64", True)

from jaxfibers.fibers.sweeney import (
    build_sweeney, node_indices, section_centers_um,
    V_REST,
)
from jaxfibers.channels.sweeney_channels import SweeneyNode
from jaxfibers.stim.extracellular import point_source_potentials_mV
from jaxfibers.stim.extracellular_coupled import arrays_from_geometry, integrate
from jaxfibers.nrn_baseline import build_sweeney_pyfibers
from pyfibers import ScaledStim

from experiments_v2.utils import ensure_dir, save_json

OUT = ensure_dir(ROOT / "outputs" / "khz_block_sweeney")

# ── shared setup ──────────────────────────────────────────────────────────────
DIAMETER  = 10.0    # µm — Sweeney validated at D=10 µm
N_NODES   = 25
CELSIUS   = 37.0
DT        = 0.001   # ms — fine dt needed for accurate kHz waveform
TSTOP     = 150.0   # ms
N_STEPS   = int(TSTOP / DT)

KHZ_FREQ  = 20.0    # kHz — myelinated A-fiber block frequency
KHZ_ON    = 50.0    # ms
KHZ_OFF   = 100.0   # ms

SRC_X     = 0.0
SRC_Y     = 250.0   # µm
SRC_I0    = 1.0
SIGMA     = 10.0    # S/m

PACE_LOC      = 0.1
PACE_START    = 15.0    # ms
PACE_INTERVAL = 10.0    # ms
PACE_N        = 14
PACE_PW       = 0.1     # ms
PACE_AMP_NA   = 1.5     # nA

AP_DETECT_LOC = 0.9
AMPLITUDES    = [-0.5, -1.5, -2.5, -3.0]   # mA: ctrl → kHz-induced → partial block → block

# SweeneyNode default channel parameters
GNABAR = 1.445    # S/cm²
GL     = 0.128    # S/cm²
EL     = -80.01   # mV
ENA    = 35.64    # mV


def waveform_full(t: float) -> float:
    if KHZ_ON < t < KHZ_OFF:
        return float(sg.square(2 * np.pi * KHZ_FREQ * t))
    return 0.0


def _build_jax_setup():
    _, geom = build_sweeney(diameter=DIAMETER, n_nodes=N_NODES)
    nodes   = node_indices(geom)
    centers = np.array(section_centers_um(geom))
    n_comp  = geom.n_comp

    fiber_length_um = centers[-1] + (centers[-1] - centers[-2])
    z_mid = fiber_length_um / 2.0
    Ve_unit = np.asarray(point_source_potentials_mV(
        list(centers),
        src_x_um=SRC_X, src_y_um=SRC_Y, src_z_um=z_mid,
        i0_mA=SRC_I0, sigma_S_m=SIGMA,
    ))

    # Use geometry capacitances as-is (node=CM_NODE, myelin≈1e-6).
    static = arrays_from_geometry(geom, DT)

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

    pace_node = nodes[int(round(PACE_LOC * (len(nodes) - 1)))]
    far_node  = nodes[int(round(AP_DETECT_LOC * (len(nodes) - 1)))]
    t_step    = (np.arange(N_STEPS) + 1) * DT
    i_intra   = np.zeros((N_STEPS, n_comp), dtype=np.float64)
    for ps in PACE_START + np.arange(PACE_N) * PACE_INTERVAL:
        on = (t_step >= ps) & (t_step < ps + PACE_PW)
        i_intra[on, pace_node] = PACE_AMP_NA

    pulse_mask = np.zeros(N_STEPS, dtype=np.float64)
    in_window  = (t_step > KHZ_ON) & (t_step < KHZ_OFF)
    pulse_mask[in_window] = sg.square(2 * np.pi * KHZ_FREQ * t_step[in_window])

    return dict(
        geom=geom, static=static, membrane_fn=membrane_fn, state0=state0,
        nodes=nodes, centers=centers, pace_node=pace_node, far_node=far_node,
        Ve_unit=Ve_unit, i_intra=i_intra, pulse_mask=pulse_mask,
        t_axis=t_step,
    )


def _count_aps(vm: np.ndarray, v_thresh: float = -20.0):
    rises = np.where((vm[:-1] <= v_thresh) & (vm[1:] > v_thresh))[0]
    if len(rises) == 0:
        return 0, None
    t_axis = (np.arange(len(vm)) + 1) * DT
    return int(len(rises)), float(t_axis[rises[-1]])


_JAX  = None
_JIT  = None


def _jax_run(amp_mA: float):
    global _JAX, _JIT
    if _JAX is None:
        _JAX = _build_jax_setup()
        S = _JAX
        Ve_unit_j    = jnp.asarray(S["Ve_unit"], dtype=jnp.float64)
        pulse_mask_j = jnp.asarray(S["pulse_mask"])
        i_intra_j    = jnp.asarray(S["i_intra"])
        far_node     = int(S["far_node"])

        @jax.jit
        def _run(amp):
            Ve = Ve_unit_j * amp
            vm_far_trace, _ = integrate(
                S["static"], S["membrane_fn"], S["state0"],
                Ve, pulse_mask_j, DT, v_rest=V_REST,
                record="center", center_comp=far_node,
                i_intra=i_intra_j,
            )
            return vm_far_trace
        _JIT = _run

    vm_far = np.asarray(_JIT(jnp.float64(amp_mA)))
    n_aps, last_t = _count_aps(vm_far)
    return n_aps, last_t, vm_far


def _pf_run(amp_mA: float):
    fiber = build_sweeney_pyfibers(diameter=DIAMETER, n_nodes=N_NODES, temperature=CELSIUS)
    fiber.potentials = fiber.point_source_potentials(
        SRC_X, SRC_Y, fiber.length / 2.0, SRC_I0, SIGMA,
    )
    fiber.record_vm()
    fiber.add_intrinsic_activity(
        loc=PACE_LOC, start_time=PACE_START,
        avg_interval=PACE_INTERVAL, num_stims=PACE_N,
    )
    stim = ScaledStim(waveform=waveform_full, dt=DT, tstop=TSTOP)
    n_aps, ap_time = stim.run_sim(amp_mA, fiber)
    vm_far = np.array(fiber.vm[fiber.loc_index(AP_DETECT_LOC)])
    t_pf   = np.array(stim.time)
    return int(n_aps), (float(ap_time) if ap_time is not None else None), vm_far, t_pf


def main():
    print("=== kHz block — Sweeney A-fiber ===")
    print(f"D={DIAMETER} µm  N={N_NODES}  dt={DT} ms")
    print(f"kHz: {KHZ_FREQ} kHz  on {KHZ_ON}-{KHZ_OFF} ms")
    print(f"Pacing: loc={PACE_LOC}  start={PACE_START} ms  interval={PACE_INTERVAL} ms"
          f"  n={PACE_N}  amp={PACE_AMP_NA} nA\n")

    results = []
    for amp in AMPLITUDES:
        print(f"--- amp = {amp} mA ---", flush=True)
        t0 = time.time()
        n_jax, last_jax, vm_jax = _jax_run(amp)
        print(f"  JAX     : {n_jax} APs  last_t={last_jax}  ({time.time()-t0:.1f}s)")
        t0 = time.time()
        n_pf, last_pf, vm_pf, t_pf = _pf_run(amp)
        print(f"  PyFibers: {n_pf} APs  last_t={last_pf}  ({time.time()-t0:.1f}s)")
        results.append({
            "amp_mA":     amp,
            "n_aps_pf":   n_pf,
            "last_pf":    last_pf,
            "n_aps_jax":  n_jax,
            "last_jax":   last_jax,
            "t_pf_ms":    t_pf.tolist(),
            "vm_pf_90":   vm_pf.tolist(),
            "vm_jax_90":  vm_jax.tolist(),
        })

    fig, axes = plt.subplots(len(AMPLITUDES), 1, figsize=(10, 2.5*len(AMPLITUDES)),
                              sharex=True, constrained_layout=True)
    for ax, res in zip(axes, results):
        t_pf  = np.array(res["t_pf_ms"])
        vm_pf = np.array(res["vm_pf_90"])
        vm_jx = np.array(res["vm_jax_90"])
        t_jx  = (np.arange(len(vm_jx)) + 1) * DT
        ax.plot(t_pf, vm_pf, color="C0", lw=1.0,
                label=f"PyFibers ({res['n_aps_pf']} APs)")
        ax.plot(t_jx, vm_jx, color="C1", lw=0.8, ls="--",
                label=f"JAXON ({res['n_aps_jax']} APs)")
        ax.axvspan(KHZ_ON, KHZ_OFF, alpha=0.2, color="red")
        ax.set_ylabel(f"{res['amp_mA']} mA\nVm (mV)")
        ax.set_ylim(-100, 50)
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle(f"kHz block — Sweeney D={DIAMETER} µm", fontsize=11)
    path = OUT / "fig_khz_block_sweeney_traces.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  -> {path}")

    save_json({
        "fiber_model":   "SWEENEY",
        "diameter_um":   DIAMETER,
        "n_nodes":       N_NODES,
        "dt_ms":         DT,
        "tstop_ms":      TSTOP,
        "khz_freq":      KHZ_FREQ,
        "src_y_um":      SRC_Y,
        "sigma_S_m":     SIGMA,
        "pace":          {"loc": PACE_LOC, "start_ms": PACE_START,
                          "interval_ms": PACE_INTERVAL, "n_pulses": PACE_N,
                          "amp_nA": PACE_AMP_NA},
        "ap_detect_loc": AP_DETECT_LOC,
        "results":       results,
    }, OUT / "data_khz_block_sweeney.json")
    print(f"  -> {OUT / 'data_khz_block_sweeney.json'}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
