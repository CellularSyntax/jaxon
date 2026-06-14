"""Kilohertz frequency block — Rattay C-fiber.

Analogous to khz_block_sundt.py but for the Rattay HH unmyelinated C-fiber model.

Setup
-----
* Rattay fiber, D=0.8 µm, N_NODES=25
* Point source at y=100 µm, sigma=10 S/m
* 1 kHz square wave, on from t=50 to t=100 ms
* Intrinsic pacing every 20 ms starting at t=15 ms, 7 pulses, amp=0.5 nA
* AP detection at loc=0.9

Outputs (outputs/khz_block_rattay/)
------------------------------------
  data_khz_block_rattay.json
  fig_khz_block_rattay_traces.png

Run from project root:
    python experiments_v2/khz_block_rattay.py
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
import scipy.signal as sg

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

OUT = ensure_dir(ROOT / "outputs" / "khz_block_rattay")

# ── shared setup ──────────────────────────────────────────────────────────────
DIAMETER  = 0.8     # µm — representative Rattay C-fiber
N_NODES   = 25
CELSIUS   = 37.0
DT        = 0.001   # ms — fine dt needed for accurate kHz waveform
TSTOP     = 150.0   # ms
N_STEPS   = int(TSTOP / DT)

KHZ_FREQ  = 1.0     # kHz — C-fibers block at lower frequency than myelinated fibers
KHZ_ON    = 50.0    # ms
KHZ_OFF   = 100.0   # ms

SRC_X     = 0.0
SRC_Y     = 100.0   # µm — closer electrode needed for unmyelinated C-fiber
SRC_I0    = 1.0
SIGMA     = 10.0    # S/m

PACE_LOC      = 0.1
PACE_START    = 15.0    # ms
PACE_INTERVAL = 20.0    # ms — slower than MRG (C-fiber refractory period ~10 ms)
PACE_N        = 7       # pulses: at 15, 35, 55, 75, 95, 115, 135 ms
PACE_PW       = 0.2     # ms — wider intra pulse for C-fiber
PACE_AMP_NA   = 0.5     # nA

AP_DETECT_LOC = 0.9
# Rattay D=0.8 µm block curve (scanned): -0.5 → 7 APs (control, pacing only),
# -5 → 54 (kHz-induced firing every cycle), >=-30 → 4 (depolarisation block).
AMPLITUDES    = [-0.5, -5.0, -30.0, -50.0]   # mA: ctrl → kHz-induced → block → block

# RattayHH default channel parameters
GNABAR = 0.12     # S/cm²
GKBAR  = 0.036    # S/cm²
GL     = 3e-4     # S/cm²
EL     = -59.4    # mV
ENA    = 45.0     # mV
EK     = -82.0    # mV


def waveform_full(t: float) -> float:
    if KHZ_ON < t < KHZ_OFF:
        return float(sg.square(2 * np.pi * KHZ_FREQ * t))
    return 0.0


def _build_jax_setup():
    _, geom = build_rattay(diameter=DIAMETER, n_nodes=N_NODES)
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

    geom_c = dataclasses.replace(geom, cm_uF_cm2=[CM] * n_comp)
    static  = arrays_from_geometry(geom_c, DT)

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
    fiber = build_rattay_pyfibers(diameter=DIAMETER, n_nodes=N_NODES, temperature=CELSIUS)
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
    print("=== kHz block — Rattay C-fiber ===")
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
        ax.set_ylim(-70, 30)
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle(f"kHz block — Rattay D={DIAMETER} µm", fontsize=11)
    path = OUT / "fig_khz_block_rattay_traces.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  -> {path}")

    save_json({
        "fiber_model":   "RATTAY",
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
    }, OUT / "data_khz_block_rattay.json")
    print(f"  -> {OUT / 'data_khz_block_rattay.json'}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
