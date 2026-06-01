"""Thin PyFibers (NEURON) baseline runner used for side-by-side comparison.

Wraps PyFibers' `build_fiber` + `IntraStim` / `ScaledStim` for MRG_DISCRETE.
Returns numpy arrays in the same shape as our Jaxley outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from neuron import h
import pyfibers
from pyfibers import build_fiber, FiberModel, IntraStim, ScaledStim


@dataclass
class NrnRunResult:
    t_ms: np.ndarray                # (T,)
    vm_mV: np.ndarray               # (n_nodes, T)
    gates: dict[str, np.ndarray]    # per gate: (T,) at probed node
    n_nodes: int
    probe_node_idx: int
    n_aps_at_probe: int


def build_mrg_pyfibers(diameter: float = 10.0, n_nodes: int = 11, temperature: float = 37.0,
                       passive_end_nodes: bool = False):
    """Wrap PyFibers `build_fiber(MRG_DISCRETE, ...)`.

    passive_end_nodes=False matches the JAX coupled solver (all nodes active).
    """
    return build_fiber(
        fiber_model=FiberModel.MRG_DISCRETE,
        diameter=diameter,
        n_nodes=n_nodes,
        temperature=temperature,
        passive_end_nodes=passive_end_nodes,
    )


def run_intracellular(
    diameter: float = 10.0,
    n_nodes: int = 11,
    temperature: float = 37.0,
    i_delay_ms: float = 1.0,
    i_dur_ms: float = 0.1,
    i_amp_nA: float = 1.0,
    dt_ms: float = 0.005,
    tstop_ms: float = 5.0,
    probe_loc: float = 0.5,
    passive_end_nodes: bool = False,
) -> NrnRunResult:
    """Intracellular pulse at probe_loc, record V_m at all nodes and gates at probe."""
    fiber = build_mrg_pyfibers(diameter=diameter, n_nodes=n_nodes, temperature=temperature,
                               passive_end_nodes=passive_end_nodes)
    probe_idx = fiber.loc_index(probe_loc)
    # Record vm at all nodes and gates at probe node
    fiber.record_vm()
    fiber.record_gating(indices=[probe_idx])

    stim = IntraStim(
        dt=dt_ms,
        tstop=tstop_ms,
        istim_ind=probe_idx,
        clamp_kws=dict(
            delay=i_delay_ms,
            pw=i_dur_ms,
            dur=tstop_ms,
            freq=1000.0 / max(tstop_ms, 1.0),   # one pulse fits in tstop
            amp=i_amp_nA,
        ),
    )
    stim.run_sim(stimamp=1.0, fiber=fiber, ap_detect_location=probe_loc, fail_on_end_excitation=False)
    t = np.array(fiber.time)
    vm = np.array([np.array(v) for v in fiber.vm])           # (n_nodes, T)
    gates = {k: np.array(v[0]) for k, v in fiber.gating.items()}  # (T,) per gate
    n_aps = fiber.apc[probe_idx].n
    return NrnRunResult(t_ms=t, vm_mV=vm, gates=gates, n_nodes=fiber.nodecount,
                        probe_node_idx=probe_idx, n_aps_at_probe=int(n_aps))


def run_extracellular(
    diameter: float = 10.0,
    n_nodes: int = 11,
    temperature: float = 37.0,
    # Point source location (µm) and physics:
    src_x_um: float = 0.0,
    src_y_um: float = 0.0,
    src_z_um: float | None = None,   # default = fiber center
    src_height_um: float = 1000.0,   # offset above the fiber for a "1 mm" stim
    sigma_S_m: float = 0.3,
    # Waveform (monophasic rectangular, mA):
    pw_ms: float = 0.1,
    delay_ms: float = 1.0,
    amp_mA: float = -1.0,            # cathodic
    dt_ms: float = 0.005,
    tstop_ms: float = 5.0,
    probe_loc: float = 0.5,
    passive_end_nodes: bool = False,
) -> NrnRunResult:
    """Extracellular point-source rectangular pulse, record V_m at all nodes."""
    fiber = build_mrg_pyfibers(diameter=diameter, n_nodes=n_nodes, temperature=temperature,
                               passive_end_nodes=passive_end_nodes)
    probe_idx = fiber.loc_index(probe_loc)
    fiber.record_vm()
    fiber.record_gating(indices=[probe_idx])

    if src_z_um is None:
        src_z_um = fiber.length / 2.0

    # potentials per fiber section (mV) for a unit-magnitude 1 mA source
    fiber.potentials = fiber.point_source_potentials(
        x=src_x_um, y=src_height_um, z=src_z_um, i0=amp_mA, sigma=sigma_S_m,
    )

    waveform = lambda t: np.where((t >= delay_ms) & (t < delay_ms + pw_ms), 1.0, 0.0)
    stim = ScaledStim(waveform=waveform, dt=dt_ms, tstop=tstop_ms)
    stim.run_sim(stimamp=1.0, fiber=fiber, ap_detect_location=probe_loc, fail_on_end_excitation=False)
    t = np.array(fiber.time)
    vm = np.array([np.array(v) for v in fiber.vm])
    gates = {k: np.array(v[0]) for k, v in fiber.gating.items()}
    n_aps = fiber.apc[probe_idx].n
    return NrnRunResult(t_ms=t, vm_mV=vm, gates=gates, n_nodes=fiber.nodecount,
                        probe_node_idx=probe_idx, n_aps_at_probe=int(n_aps))


def find_threshold_extracellular(
    diameter: float,
    n_nodes: int = 21,
    temperature: float = 37.0,
    src_height_um: float = 1000.0,
    sigma_S_m: float = 0.3,
    pw_ms: float = 0.1,
    delay_ms: float = 1.0,
    dt_ms: float = 0.005,
    tstop_ms: float = 5.0,
    bounds_mA: tuple[float, float] = (-1.0, -0.001),
    rel_tol: float = 1e-3,
    biphasic: bool = False,
) -> float:
    """Bisection: find minimum |amp_mA| cathodic that elicits an AP at probe node.

    biphasic=True: cathodic-first symmetric biphasic (pw_ms per phase, zero inter-phase gap).
    Returns the threshold amplitude in mA (negative = cathodic).
    """
    fiber = build_mrg_pyfibers(diameter=diameter, n_nodes=n_nodes, temperature=temperature)
    probe_idx = fiber.loc_index(0.5)
    # Unit-mag potentials (we'll scale via waveform amplitude during bisection):
    fiber.potentials = fiber.point_source_potentials(
        x=0.0, y=src_height_um, z=fiber.length / 2.0, i0=1.0, sigma=sigma_S_m,
    )
    if biphasic:
        wav = lambda t: (
            np.where((t >= delay_ms) & (t < delay_ms + pw_ms), 1.0, 0.0) -
            np.where((t >= delay_ms + pw_ms) & (t < delay_ms + 2 * pw_ms), 1.0, 0.0)
        )
    else:
        wav = lambda t: np.where((t >= delay_ms) & (t < delay_ms + pw_ms), 1.0, 0.0)
    stim = ScaledStim(waveform=wav, dt=dt_ms, tstop=tstop_ms)

    def fires(amp_mA: float) -> bool:
        stim.run_sim(stimamp=amp_mA, fiber=fiber, ap_detect_location=0.5, fail_on_end_excitation=False)
        return fiber.apc[probe_idx].n > 0

    lo, hi = bounds_mA  # both negative; |lo| > |hi|
    # Expand lo if not firing at it:
    expand_attempts = 0
    while not fires(lo) and expand_attempts < 6:
        lo *= 2.0
        expand_attempts += 1
    if not fires(lo):
        raise RuntimeError(f"PyFibers fiber d={diameter}µm did not fire even at {lo} mA")
    if fires(hi):
        raise RuntimeError(f"PyFibers fiber d={diameter}µm fires at the tiny bound {hi} mA")

    while abs(hi - lo) / abs(lo) > rel_tol:
        mid = (lo + hi) / 2.0
        if fires(mid):
            lo = mid
        else:
            hi = mid
    return lo  # threshold (most-conservative firing value)


# ── Sundt (unmyelinated C-fiber) wrappers ─────────────────────────────────────

def build_sundt_pyfibers(diameter: float = 0.8, n_nodes: int = 21,
                         temperature: float = 37.0,
                         passive_end_nodes: bool = False):
    """Wrap PyFibers `build_fiber(SUNDT, ...)`.

    passive_end_nodes=False matches JAX (all compartments active).
    """
    return build_fiber(
        fiber_model=FiberModel.SUNDT,
        diameter=diameter,
        n_nodes=n_nodes,
        temperature=temperature,
        passive_end_nodes=passive_end_nodes,
    )


def run_intracellular_sundt(
    diameter: float = 0.8,
    n_nodes: int = 21,
    temperature: float = 37.0,
    i_delay_ms: float = 1.0,
    i_dur_ms: float = 0.1,
    i_amp_nA: float = 0.5,
    dt_ms: float = 0.005,
    tstop_ms: float = 10.0,
    probe_loc: float = 0.5,
    passive_end_nodes: bool = False,
) -> NrnRunResult:
    """Intracellular pulse on Sundt C-fiber, record V_m and gates."""
    fiber = build_sundt_pyfibers(diameter=diameter, n_nodes=n_nodes,
                                 temperature=temperature,
                                 passive_end_nodes=passive_end_nodes)
    probe_idx = fiber.loc_index(probe_loc)
    fiber.record_vm()
    fiber.record_gating(indices=[probe_idx])

    stim = IntraStim(
        dt=dt_ms,
        tstop=tstop_ms,
        istim_ind=probe_idx,
        clamp_kws=dict(
            delay=i_delay_ms,
            pw=i_dur_ms,
            dur=tstop_ms,
            freq=1000.0 / max(tstop_ms, 1.0),
            amp=i_amp_nA,
        ),
    )
    stim.run_sim(stimamp=1.0, fiber=fiber, ap_detect_location=probe_loc,
                 fail_on_end_excitation=False)
    t  = np.array(fiber.time)
    vm = np.array([np.array(v) for v in fiber.vm])
    gates = {k: np.array(v[0]) for k, v in fiber.gating.items()}
    n_aps = fiber.apc[probe_idx].n
    return NrnRunResult(t_ms=t, vm_mV=vm, gates=gates, n_nodes=fiber.nodecount,
                        probe_node_idx=probe_idx, n_aps_at_probe=int(n_aps))


def run_extracellular_sundt(
    diameter: float = 0.8,
    n_nodes: int = 21,
    temperature: float = 37.0,
    src_height_um: float = 1000.0,
    sigma_S_m: float = 0.3,
    pw_ms: float = 0.1,
    delay_ms: float = 1.0,
    amp_mA: float = -1.0,
    dt_ms: float = 0.005,
    tstop_ms: float = 10.0,
    probe_loc: float = 0.5,
    passive_end_nodes: bool = False,
) -> NrnRunResult:
    """Extracellular point-source rectangular pulse on Sundt C-fiber."""
    fiber = build_sundt_pyfibers(diameter=diameter, n_nodes=n_nodes,
                                 temperature=temperature,
                                 passive_end_nodes=passive_end_nodes)
    probe_idx = fiber.loc_index(probe_loc)
    fiber.record_vm()
    fiber.record_gating(indices=[probe_idx])

    fiber.potentials = fiber.point_source_potentials(
        x=0.0, y=src_height_um, z=fiber.length / 2.0, i0=amp_mA, sigma=sigma_S_m,
    )
    waveform = lambda t: np.where((t >= delay_ms) & (t < delay_ms + pw_ms), 1.0, 0.0)
    stim = ScaledStim(waveform=waveform, dt=dt_ms, tstop=tstop_ms)
    stim.run_sim(stimamp=1.0, fiber=fiber, ap_detect_location=probe_loc,
                 fail_on_end_excitation=False)
    t  = np.array(fiber.time)
    vm = np.array([np.array(v) for v in fiber.vm])
    gates = {k: np.array(v[0]) for k, v in fiber.gating.items()}
    n_aps = fiber.apc[probe_idx].n
    return NrnRunResult(t_ms=t, vm_mV=vm, gates=gates, n_nodes=fiber.nodecount,
                        probe_node_idx=probe_idx, n_aps_at_probe=int(n_aps))


def find_threshold_extracellular_waveform(
    fiber_model: str,
    diameter: float,
    n_nodes: int,
    waveform_callable,
    tstop_ms: float,
    dt_ms: float,
    temperature: float = 37.0,
    src_height_um: float = 1000.0,
    sigma_S_m: float = 0.3,
    bounds_mA: tuple = (-1.0, -0.001),
    rel_tol: float = 1e-3,
) -> float:
    """Generic threshold finder for arbitrary waveform and fiber model.

    fiber_model: "mrg" or "sundt".
    waveform_callable: t (ms) -> float, normalized in [-1, 1].
    bounds_mA: (lo, hi) where lo has larger |magnitude| (firing) and hi is non-firing.
      Use negative values for cathodic, positive for anodic.
    Returns threshold amplitude in mA (sign matches bounds_mA convention).
    """
    if fiber_model == "mrg":
        fiber = build_mrg_pyfibers(diameter=diameter, n_nodes=n_nodes, temperature=temperature)
    elif fiber_model == "sundt":
        fiber = build_sundt_pyfibers(diameter=diameter, n_nodes=n_nodes, temperature=temperature)
    else:
        raise ValueError(f"Unknown fiber_model '{fiber_model}'. Use 'mrg' or 'sundt'.")

    probe_idx = fiber.loc_index(0.5)
    fiber.potentials = fiber.point_source_potentials(
        x=0.0, y=src_height_um, z=fiber.length / 2.0, i0=1.0, sigma=sigma_S_m,
    )
    stim = ScaledStim(waveform=waveform_callable, dt=dt_ms, tstop=tstop_ms)

    def fires(amp: float) -> bool:
        stim.run_sim(stimamp=amp, fiber=fiber, ap_detect_location=0.5, fail_on_end_excitation=False)
        return fiber.apc[probe_idx].n > 0

    lo, hi = bounds_mA
    for _ in range(10):
        if fires(lo):
            break
        lo *= 2.0
    if not fires(lo):
        raise RuntimeError(f"{fiber_model} d={diameter}µm did not fire at {lo} mA")
    if fires(hi):
        raise RuntimeError(f"{fiber_model} d={diameter}µm fires at sub-threshold bound {hi} mA")

    while abs(hi - lo) / max(abs(lo), 1e-12) > rel_tol:
        mid = (lo + hi) / 2.0
        if fires(mid):
            lo = mid
        else:
            hi = mid
    return lo


def find_threshold_extracellular_sundt(
    diameter: float = 0.8,
    n_nodes: int = 21,
    temperature: float = 37.0,
    src_height_um: float = 1000.0,
    sigma_S_m: float = 0.3,
    pw_ms: float = 0.1,
    delay_ms: float = 1.0,
    dt_ms: float = 0.005,
    tstop_ms: float = 10.0,
    bounds_mA: tuple[float, float] = (-5.0, -0.001),
    rel_tol: float = 1e-3,
) -> float:
    """Bisection: minimum cathodic amplitude that fires the centre Sundt compartment."""
    fiber = build_sundt_pyfibers(diameter=diameter, n_nodes=n_nodes, temperature=temperature)
    probe_idx = fiber.loc_index(0.5)
    fiber.potentials = fiber.point_source_potentials(
        x=0.0, y=src_height_um, z=fiber.length / 2.0, i0=1.0, sigma=sigma_S_m,
    )
    wav = lambda t: np.where((t >= delay_ms) & (t < delay_ms + pw_ms), 1.0, 0.0)
    stim = ScaledStim(waveform=wav, dt=dt_ms, tstop=tstop_ms)

    def fires(amp_mA: float) -> bool:
        stim.run_sim(stimamp=amp_mA, fiber=fiber, ap_detect_location=0.5,
                     fail_on_end_excitation=False)
        return fiber.apc[probe_idx].n > 0

    lo, hi = bounds_mA
    for _ in range(8):
        if fires(lo):
            break
        lo *= 2.0
    if not fires(lo):
        raise RuntimeError(f"Sundt d={diameter}µm did not fire at {lo} mA")
    if fires(hi):
        raise RuntimeError(f"Sundt d={diameter}µm fires at sub-threshold bound {hi} mA")

    while abs(hi - lo) / abs(lo) > rel_tol:
        mid = (lo + hi) / 2.0
        if fires(mid):
            lo = mid
        else:
            hi = mid
    return lo
