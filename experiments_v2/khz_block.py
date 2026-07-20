"""Kilohertz frequency block — PyFibers tutorial reproduction + jaxon comparison.

Mirrors pyfibers/tutorials/5_block_threshold:
  * MRG_INTERPOLATION, D=10 µm, N=25 nodes
  * Point source at y=250 µm, sigma=10 S/m
  * 20 kHz square wave, on from t=50 to t=100 ms
  * Intrinsic pacing every 10 ms starting at t=15 ms, 14 pulses
  * AP detection at loc=0.9 (90% of fiber length)

Runs both PyFibers and the JAX coupled solver at the four tutorial
amplitudes (-0.5, -1.5, -2.5, -3.0 mA) and compares Vm at loc=0.9 +
AP counts.  The four-point qualitative match is enough to verify the
JAX model reproduces the kHz block phenomenon; threshold bisection is
deferred (it produces a single number that doesn't add evidence
beyond the recruitment-curve shape).

Outputs (outputs/khz_block/)
----------------------------
  data_khz_block.json       — per-amplitude AP counts + Vm traces
  fig_khz_block_traces.png  — Vm(t) at loc=0.9 for each amplitude, both
                               solvers overlaid

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
import scipy.signal as sg

import jax
import jax.numpy as jnp
from jaxley.solver_gate import solve_gate_exponential

jax.config.update("jax_enable_x64", True)

from jaxon.fibers.mrg import (
    build_mrg_interp, node_indices, section_centers_um,
    V_REST, CM_AXON, G_PAS_MYSA, G_PAS_FLUT, G_PAS_STIN,
)
from jaxon.channels.mrg_axnode import AxnodeMyel
from jaxon.stim.extracellular import point_source_potentials_mV
from jaxon.stim.extracellular_coupled import arrays_from_geometry, integrate
from jaxon.nrn_baseline import build_mrg_pyfibers  # noqa: F401 (PYTHONPATH fix)

from pyfibers import build_fiber, FiberModel, ScaledStim

from experiments_v2.utils import ensure_dir, save_json

OUT = ensure_dir(ROOT / "outputs" / "khz_block")

# ── shared setup (matches PyFibers tutorial 5 verbatim) ───────────────────────
DIAMETER  = 10.0
N_NODES   = 25
CELSIUS   = 37.0
DT        = 0.001
TSTOP     = 150.0
N_STEPS   = int(TSTOP / DT)

KHZ_FREQ  = 20.0
KHZ_ON    = 50.0
KHZ_OFF   = 100.0

SRC_X     = 0.0
SRC_Y     = 250.0
SRC_I0    = 1.0
SIGMA     = 10.0

PACE_LOC      = 0.1
PACE_START    = 15.0
PACE_INTERVAL = 10.0
PACE_N        = 14
PACE_PW       = 0.1
PACE_AMP_NA   = 1.5

AP_DETECT_LOC = 0.9
AMPLITUDES    = [-0.5, -1.5, -2.5, -3.0]

# MRG channel constants
GNABAR = 3.0;  GNAPBAR = 0.01; GKBAR = 0.08; GL = 0.007
ENA    = 50.0; EK      = -90.0; EL    = -90.0


def waveform_full(t: float) -> float:
    """20 kHz square wave, on for t in [KHZ_ON, KHZ_OFF]."""
    if KHZ_ON < t < KHZ_OFF:
        return float(sg.square(2 * np.pi * KHZ_FREQ * t))
    return 0.0


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


def _build_jax_setup():
    _, geom = build_mrg_interp(diameter=DIAMETER, n_nodes=N_NODES)
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

    geom_c = dataclasses.replace(geom, cm_uF_cm2=[CM_AXON] * n_comp)
    static = arrays_from_geometry(geom_c, DT)
    membrane_fn, state0 = _build_membrane_fn(static, geom)

    pace_node = nodes[int(round(PACE_LOC * (len(nodes) - 1)))]
    far_node  = nodes[int(round(AP_DETECT_LOC * (len(nodes) - 1)))]
    t_step = (np.arange(N_STEPS) + 1) * DT
    i_intra = np.zeros((N_STEPS, n_comp), dtype=np.float64)
    pulse_starts = PACE_START + np.arange(PACE_N) * PACE_INTERVAL
    for ps in pulse_starts:
        on = (t_step >= ps) & (t_step < ps + PACE_PW)
        i_intra[on, pace_node] = PACE_AMP_NA

    pulse_mask = np.zeros(N_STEPS, dtype=np.float64)
    in_window = (t_step > KHZ_ON) & (t_step < KHZ_OFF)
    pulse_mask[in_window] = sg.square(2 * np.pi * KHZ_FREQ * t_step[in_window])

    return dict(
        geom=geom, static=static, membrane_fn=membrane_fn, state0=state0,
        nodes=nodes, centers=centers, pace_node=pace_node, far_node=far_node,
        Ve_unit=Ve_unit, i_intra=i_intra, pulse_mask=pulse_mask,
        t_axis=t_step,
    )


def _count_aps(vm: np.ndarray, v_thresh: float = -20.0) -> tuple[int, float | None]:
    rises = np.where((vm[:-1] <= v_thresh) & (vm[1:] > v_thresh))[0]
    if len(rises) == 0:
        return 0, None
    t_axis = (np.arange(len(vm)) + 1) * DT
    return int(len(rises)), float(t_axis[rises[-1]])


# Cached JAX setup + JIT runner
_JAX = None
_JIT_RUN = None


def _jax_run(amp_mA: float) -> tuple[int, float | None, np.ndarray]:
    """Run jaxon kHz block; return (n_aps, last_ap_time, vm_far[N_STEPS])."""
    global _JAX, _JIT_RUN
    if _JAX is None:
        _JAX = _build_jax_setup()
        S = _JAX
        Ve_unit_j    = jnp.asarray(S["Ve_unit"], dtype=jnp.float64)
        pulse_mask_j = jnp.asarray(S["pulse_mask"])
        i_intra_j    = jnp.asarray(S["i_intra"])
        far_node     = int(S["far_node"])

        @jax.jit
        def _run(amp):
            # PyFibers waveform is unitless (-1 / +1).  PyFibers' run_sim
            # multiplies fiber.potentials by `stimamp` directly, with
            # potentials computed at i0=+1 mA.  So our equivalent Ve scaling
            # is `Ve_unit * amp` (no -1 flip — PyFibers does not invert sign
            # via i0; we already used i0=+1 above).
            Ve = Ve_unit_j * amp
            vm_far_trace, _ = integrate(
                S["static"], S["membrane_fn"], S["state0"],
                Ve, pulse_mask_j, DT, v_rest=V_REST,
                record="center", center_comp=far_node,
                i_intra=i_intra_j,
            )
            return vm_far_trace
        _JIT_RUN = _run

    vm_far = np.asarray(_JIT_RUN(jnp.float64(amp_mA)))
    n_aps, last_t = _count_aps(vm_far)
    return n_aps, last_t, vm_far


def _pf_run(amp_mA: float) -> tuple[int, float | None, np.ndarray, np.ndarray]:
    fiber = build_fiber(FiberModel.MRG_INTERPOLATION,
                         diameter=DIAMETER, n_nodes=N_NODES)
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
    print("=== kHz block — PyFibers tutorial + jaxon comparison ===")
    print(f"Fiber: MRG_INTERP D={DIAMETER} µm  N={N_NODES} nodes  dt={DT} ms")
    print(f"kHz: {KHZ_FREQ} kHz square, on {KHZ_ON}-{KHZ_OFF} ms")
    print(f"Pacing: loc={PACE_LOC}, start={PACE_START} ms, interval={PACE_INTERVAL} ms, "
          f"n={PACE_N}\n")

    results = []
    for amp in AMPLITUDES:
        print(f"--- amp = {amp} mA ---", flush=True)
        t0 = time.time()
        n_jax, last_jax, vm_jax = _jax_run(amp)
        t_jax_wall = time.time() - t0
        t0 = time.time()
        n_pf, last_pf, vm_pf, t_pf = _pf_run(amp)
        t_pf_wall = time.time() - t0
        print(f"  PyFibers : {n_pf} APs, last t = {last_pf}  ({t_pf_wall:.1f} s)")
        print(f"  jaxon: {n_jax} APs, last t = {last_jax}  ({t_jax_wall:.1f} s)")
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

    fig, axes = plt.subplots(len(AMPLITUDES), 1,
                              figsize=(10, 2.5 * len(AMPLITUDES)),
                              sharex=True, constrained_layout=True)
    for ax, res in zip(axes, results):
        t_pf  = np.array(res["t_pf_ms"])
        vm_pf = np.array(res["vm_pf_90"])
        vm_jx = np.array(res["vm_jax_90"])
        t_jx  = (np.arange(len(vm_jx)) + 1) * DT
        ax.plot(t_pf, vm_pf, color="C0", lw=1.0,
                label=f"PyFibers ({res['n_aps_pf']} APs)")
        ax.plot(t_jx, vm_jx, color="C1", lw=0.8, ls="--",
                label=f"jaxon ({res['n_aps_jax']} APs)")
        ax.axvspan(KHZ_ON, KHZ_OFF, alpha=0.2, color="red", label="kHz on")
        for s in PACE_START + np.arange(PACE_N) * PACE_INTERVAL:
            ax.axvline(s, color="k", ls="--", lw=0.4, alpha=0.4)
        ax.set_ylabel(f"{res['amp_mA']} mA\nVm (mV)")
        ax.set_ylim(-90, 30)
        ax.grid(alpha=0.3, lw=0.4)
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle(
        f"kHz block — MRG_INTERP D={DIAMETER} µm  (PyFibers solid blue, "
        f"jaxon dashed orange)",
        fontsize=11,
    )
    path = OUT / "fig_khz_block_traces.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  -> {path}")

    save_json({
        "fiber_model":   "MRG_INTERPOLATION",
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
    }, OUT / "data_khz_block.json")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
