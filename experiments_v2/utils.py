"""Shared utilities for experiments_v2 scripts.

Provides:
  - PulseSpec registry and pulse-array generators (JAX + PyFibers compatible)
  - JAX bisection
  - JSON serialisation (numpy-safe)
  - Output-directory helpers
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import jax.numpy as jnp
import jax


# ── Pulse shape registry ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class PulseSpec:
    key: str
    name: str
    group: str            # "mono" | "bi" | "shaped"
    # Bisection search bounds (lo = firing bound, hi = sub-threshold).
    # Negative → cathodic search; positive → anodic search.
    lo: float
    hi: float


PULSES: dict[str, PulseSpec] = {
    "mono_c":   PulseSpec("mono_c",   "Monophasic cathodic",       "mono",   -5.0, -0.001),
    "mono_a":   PulseSpec("mono_a",   "Monophasic anodic",         "mono",   +5.0, +0.001),
    "bi_ca":    PulseSpec("bi_ca",    "Biphasic cathodic-anodic",  "bi",     -5.0, -0.001),
    "bi_ac":    PulseSpec("bi_ac",    "Biphasic anodic-cathodic",  "bi",     -5.0, -0.001),
    "sine":     PulseSpec("sine",     "Sine (cathodic half-cycle)","shaped", -5.0, -0.001),
    "sawtooth": PulseSpec("sawtooth", "Sawtooth (cathodic)",       "shaped", -5.0, -0.001),
    "exp":      PulseSpec("exp",      "Exponential (cathodic)",    "shaped", -5.0, -0.001),
    "gaussian": PulseSpec("gaussian", "Gaussian (cathodic)",       "shaped", -5.0, -0.001),
}

GROUP_COLORS = {"mono": "C0", "bi": "C2", "shaped": "C4"}


# ── Pulse array generator ─────────────────────────────────────────────────────

def make_pulse_array(
    key: str,
    pw: float,
    n_steps: int,
    dt: float,
    delay: float,
) -> np.ndarray:
    """Return a normalised pulse array of length n_steps.

    Sign convention (shared with JAX coupled solver and PyFibers ScaledStim):
      +1 at the cathodic-direction phase, -1 at the anodic-direction phase.
    For cathodic stimulation the amplitude (stimamp / amp_mA) is negative;
    for anodic stimulation it is positive.

    Biphasic total duration = 2 × pw; tstop must accommodate this.
    """
    t = (np.arange(n_steps) + 1) * dt
    on  = (t >= delay)
    off = (t <  delay + pw)
    act = on & off

    if key == "mono_c":
        return act.astype(np.float64)

    elif key == "mono_a":
        # Waveform is still +1 at active phase; sign of amplitude flips polarity.
        return act.astype(np.float64)

    elif key == "bi_ca":
        p1 = on & (t < delay + pw)
        p2 = (t >= delay + pw) & (t < delay + 2 * pw)
        return p1.astype(np.float64) - p2.astype(np.float64)

    elif key == "bi_ac":
        p1 = on & (t < delay + pw)
        p2 = (t >= delay + pw) & (t < delay + 2 * pw)
        return -p1.astype(np.float64) + p2.astype(np.float64)

    elif key == "sine":
        arr = np.zeros(n_steps)
        arr[act] = np.sin(np.pi * (t[act] - delay) / pw)
        return arr

    elif key == "sawtooth":
        arr = np.zeros(n_steps)
        arr[act] = 1.0 - (t[act] - delay) / pw
        return arr

    elif key == "exp":
        arr = np.zeros(n_steps)
        tau = pw / 3.0
        raw = np.exp(-(t[act] - delay) / tau)
        arr[act] = raw / raw.max() if raw.max() > 0 else raw
        return arr

    elif key == "gaussian":
        arr = np.zeros(n_steps)
        mu    = delay + pw / 2.0
        sigma = pw / 6.0
        arr[act] = np.exp(-0.5 * ((t[act] - mu) / sigma) ** 2)
        return arr

    else:
        raise ValueError(f"Unknown pulse key: '{key}'")


def pulse_array_to_callable(arr: np.ndarray, dt: float) -> Callable[[float], float]:
    """Convert a pulse array to a PyFibers-compatible callable (t in ms → float).

    Uses linear interpolation; returns 0 outside the array's time range.
    """
    from scipy.interpolate import interp1d
    t_pts = np.concatenate([[0.0], (np.arange(len(arr)) + 1) * dt])
    v_pts = np.concatenate([[0.0], arr])
    f = interp1d(t_pts, v_pts, bounds_error=False, fill_value=0.0)
    return lambda t: float(f(t))


# ── JAX bisection ─────────────────────────────────────────────────────────────

def jax_bisect(
    run_fn,
    pulse_arr: np.ndarray,
    lo: float,
    hi: float,
    tol: float = 1e-4,
    v_thresh: float = -30.0,
    max_expand: int = 10,
    debug: bool = False,
) -> float:
    """Binary search for firing threshold using a compiled JAX runner.

    run_fn: jit-compiled (amp_float64, pulse_shape_float64) -> peak_Vm_float.
    lo: starting bound that should fire (larger |magnitude|).
    hi: sub-threshold bound (smaller |magnitude|).
    Works for both cathodic (lo < hi < 0) and anodic (lo > hi > 0) bounds.
    Returns threshold amplitude in mA.

    debug=True: print wall time for each individual runner call so you can see
    whether the first call (JIT compile) dominates or all calls are slow.
    """
    pm = jnp.asarray(pulse_arr, dtype=jnp.float64)
    call_times: list[float] = []

    def fires(a: float) -> bool:
        t0 = time.perf_counter()
        result = bool(run_fn(jnp.float64(a), pm) > v_thresh)
        dt = time.perf_counter() - t0
        call_times.append(dt)
        if debug:
            n = len(call_times)
            tag = "COMPILE?" if n == 1 else f"call {n:2d}"
            print(f"          [{tag}] {dt:.3f} s  fires={result}", flush=True)
        return result

    for _ in range(max_expand):
        if fires(lo):
            break
        # Double lo in its current direction: cathodic (lo<0) becomes more negative,
        # anodic (lo>0) becomes more positive. Both directions expand the firing search.
        lo = lo * 2.0
    if not fires(lo):
        raise RuntimeError(f"Fiber did not fire even at {lo} mA — check bounds or model")
    if fires(hi):
        raise RuntimeError(f"Fiber fires at sub-threshold bound {hi} mA — check bounds")

    while abs(hi - lo) > tol:
        mid = 0.5 * (lo + hi)
        if fires(mid):
            lo = mid
        else:
            hi = mid

    if debug and len(call_times) > 1:
        rest = call_times[1:]
        print(f"          [summary] {len(call_times)} calls | "
              f"1st={call_times[0]:.3f}s | rest avg={np.mean(rest):.3f}s "
              f"min={np.min(rest):.3f}s max={np.max(rest):.3f}s", flush=True)

    return 0.5 * (lo + hi)


# ── AP arrival time (rising-edge crossing, sub-step interpolated) ─────────────

def ap_arrival_time(
    vm_trace: np.ndarray,
    t_arr: np.ndarray,
    v_thresh_mV: float = 0.0,
    onset_idx: int = 0,
) -> float:
    """Time (ms) of the first rising-edge crossing of v_thresh_mV.

    Linear interpolation between adjacent samples gives sub-step precision.
    This avoids the single-step discretisation noise that argmax-based
    AP-arrival detection suffers from when the AP peak is broad and flat
    (unmyelinated C-fibers, e.g. Sundt and Rattay at D >= 0.5 µm).

    Returns NaN if no crossing is found after onset_idx.

    Why: for slow C-fiber conduction the inter-node Δt can be only ~0.3-0.5 ms;
    a single dt=0.005 ms argmax error then maps to 1-2 % CV error. The rising-
    edge crossing is locked to the actual depolarisation event, which is
    identical between JAX and NEURON traces to many decimal places, so the
    interpolated crossing time agrees to machine precision.
    """
    vm = np.asarray(vm_trace[onset_idx:], dtype=np.float64)
    t  = np.asarray(t_arr[onset_idx:],   dtype=np.float64)
    above = vm > v_thresh_mV
    if not above.any():
        return float("nan")
    crossings = np.where((vm[:-1] <= v_thresh_mV) & (vm[1:] > v_thresh_mV))[0]
    if len(crossings) == 0:
        return float("nan")
    i = int(crossings[0])
    v0, v1 = float(vm[i]), float(vm[i + 1])
    t0, t1 = float(t[i]),  float(t[i + 1])
    if v1 == v0:
        return t0
    frac = (v_thresh_mV - v0) / (v1 - v0)
    return t0 + frac * (t1 - t0)


# ── PyFibers threshold wrapper ────────────────────────────────────────────────

def pf_find_threshold(
    fiber_model: str,
    diameter: float,
    n_nodes: int,
    pulse_arr: np.ndarray,
    dt: float,
    tstop: float,
    lo: float,
    hi: float,
    temperature: float = 37.0,
    src_height_um: float = 1000.0,
    sigma_S_m: float = 0.3,
    rel_tol: float = 1e-3,
) -> float:
    """PyFibers threshold finding for any pulse array and fiber model.

    Converts the pulse array to a callable, then delegates to
    nrn_baseline.find_threshold_extracellular_waveform.
    """
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from jaxfibers.nrn_baseline import find_threshold_extracellular_waveform

    wav = pulse_array_to_callable(pulse_arr, dt)
    return find_threshold_extracellular_waveform(
        fiber_model=fiber_model,
        diameter=diameter,
        n_nodes=n_nodes,
        waveform_callable=wav,
        tstop_ms=tstop,
        dt_ms=dt,
        temperature=temperature,
        src_height_um=src_height_um,
        sigma_S_m=sigma_S_m,
        bounds_mA=(lo, hi),
        rel_tol=rel_tol,
    )


# ── Output helpers ────────────────────────────────────────────────────────────

def ensure_dir(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(data, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)
    print(f"  → {path}")
