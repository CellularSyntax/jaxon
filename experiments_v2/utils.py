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
    print(f"  -> {path}")


# ─────────────────────────────────── selectivity per-seed summary figure ──────

def plot_seed_cross_section(
    out_path: pathlib.Path,
    nerve,                    # NerveGeometry with .fiber_x_um, .fiber_y_um,
                                # .fiber_diam, .target_mask
    contact_xyz: np.ndarray,  # [K, 3] contact positions (mm)
    amps_mA: np.ndarray,      # [K] final per-contact amplitudes
    acts: np.ndarray,         # [n_fibers] activation proxy ∈ [0, 1]
    si: float,
    si_baseline: float,
    title: str = "",
    contact_xyz_init: np.ndarray | None = None,
    nerve_radius_um: float = 500.0,
    cuff_radius_um: float = 1500.0,
    activation_threshold: float = 0.5,
) -> None:
    """Stand-alone, manuscript-quality nerve cross-section figure.

    Draws:
      - Nerve outline (black ring)
      - **Target fascicle** as a filled blue disk (boundary inferred from
        the target_mask: centroid + 1.1 × max distance of target fibres
        from that centroid).
      - **Off-target region** as the rest of the nerve, with an unfilled
        outline so the difference reads at a glance.
      - **Target fibres**: circles.  Fired → solid blue.  Silent → hollow
        blue (outline only).
      - **Off-target fibres**: squares.  Fired → solid red (alarm — leak).
        Silent → hollow red (outline only).
      - **Contacts**: triangles around the cuff, green if cathodic, red if
        anodic.  Marker area scales with |amp|; label shows ±X.XX mA.
      - For joint-opt, optional arrows from the initial to the final
        contact positions.
      - Big title with SI baseline, SI achieved, and target-fired /
        off-target-fired counts.

    Layout: single axis, square aspect ratio, ~7×7 inches by default.
    Designed to drop into a manuscript figure or supplementary as-is.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    target = np.asarray(nerve.target_mask, dtype=bool)
    n_fibers = nerve.n_fibers
    n_contacts = len(amps_mA)
    acts = np.asarray(acts)
    fired = acts > activation_threshold

    fig, ax = plt.subplots(figsize=(8.0, 8.0), constrained_layout=True)
    ax.set_aspect("equal")

    # nerve outline (a thicker, slightly grey ring so contacts pop)
    ax.add_patch(plt.Circle((0, 0), nerve_radius_um,
                              fill=False, edgecolor="0.25", lw=2.0, zorder=1))

    # ── target/off-target divider line (Hussain-style nerves only) ───────────
    divider = getattr(nerve, "divider_angle_deg", None)
    if divider is not None:
        th = np.deg2rad(divider)
        # Line direction = (cosθ, sinθ); normal = (-sinθ, cosθ).  Draw the
        # chord from (-nerve_r·cosθ, -nerve_r·sinθ) to
        # (+nerve_r·cosθ, +nerve_r·sinθ).
        L = nerve_radius_um * 1.05
        ax.plot([-L * np.cos(th), +L * np.cos(th)],
                [-L * np.sin(th), +L * np.sin(th)],
                color="0.35", lw=1.6, linestyle="--", dashes=(8, 4),
                zorder=2)
        # tiny label "target" on the + normal side
        nx, ny = -np.sin(th), np.cos(th)
        ax.annotate("target ↑", xy=(nx * nerve_radius_um * 0.7,
                                     ny * nerve_radius_um * 0.7),
                     fontsize=9, color="C0", fontweight="bold",
                     ha="center", va="center", zorder=7,
                     bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                edgecolor="C0", alpha=0.9))
        ax.annotate("off-target ↓", xy=(-nx * nerve_radius_um * 0.7,
                                         -ny * nerve_radius_um * 0.7),
                     fontsize=9, color="#c0392b", fontweight="bold",
                     ha="center", va="center", zorder=7,
                     bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                edgecolor="#c0392b", alpha=0.9))

    # ── draw fascicles ───────────────────────────────────────────────────────
    # Prefer the *actual* fascicle outlines (FascicleOutline list on the
    # NerveGeometry) so the boundary matches what the geometry builder
    # used to place fibres.  Fall back to inferring a single outline from
    # target-fibre positions if the nerve has no fascicle metadata
    # (legacy single-disk scatter case).
    fascicles_meta = getattr(nerve, "fascicles", None) or []
    if fascicles_meta:
        for fc in fascicles_meta:
            if fc.is_target:
                # Filled disk for the target fascicle.
                ax.add_patch(plt.Circle(
                    (fc.cx_um, fc.cy_um), fc.r_um,
                    facecolor="#cfe2f3", edgecolor="none",
                    alpha=0.55, zorder=2,
                ))
                ax.add_patch(plt.Circle(
                    (fc.cx_um, fc.cy_um), fc.r_um,
                    facecolor="none", edgecolor="C0",
                    lw=2.5, zorder=6,
                ))
            else:
                # Off-target: hollow with a thick coloured rim so it reads
                # without competing visually with the target fill.
                ax.add_patch(plt.Circle(
                    (fc.cx_um, fc.cy_um), fc.r_um,
                    facecolor="none", edgecolor="#c0392b",
                    lw=2.5, linestyle="-", zorder=6,
                ))
    else:
        # Legacy fallback: estimate a single target fascicle boundary.
        tgt_x = np.asarray(nerve.fiber_x_um)[target]
        tgt_y = np.asarray(nerve.fiber_y_um)[target]
        if len(tgt_x) > 0:
            fc_cx = float(tgt_x.mean())
            fc_cy = float(tgt_y.mean())
            max_r = float(np.max(np.hypot(tgt_x - fc_cx, tgt_y - fc_cy)))
            fc_r  = max(max_r * 1.15, 60.0)
        else:
            fc_cx, fc_cy, fc_r = 0.0, 0.0, nerve_radius_um * 0.3
        ax.add_patch(plt.Circle((fc_cx, fc_cy), fc_r,
                                  facecolor="#cfe2f3", edgecolor="C0",
                                  lw=1.8, alpha=0.55, zorder=2))

    # ── fibres ───────────────────────────────────────────────────────────────
    # Use *distinct markers* so target vs off-target reads even in greyscale,
    # and *fill vs hollow* so fired vs silent reads independently.
    tgt_idx = np.where(target)[0]
    off_idx = np.where(~target)[0]

    # target fibres = circles, blue
    ax.scatter(
        [nerve.fiber_x_um[i] for i in tgt_idx if fired[i]],
        [nerve.fiber_y_um[i] for i in tgt_idx if fired[i]],
        s=160, marker="o", facecolor="C0", edgecolor="k", linewidth=1.0,
        zorder=5,
    )
    ax.scatter(
        [nerve.fiber_x_um[i] for i in tgt_idx if not fired[i]],
        [nerve.fiber_y_um[i] for i in tgt_idx if not fired[i]],
        s=160, marker="o", facecolor="none", edgecolor="C0",
        linewidth=1.8, zorder=5,
    )
    # off-target fibres = squares, red
    ax.scatter(
        [nerve.fiber_x_um[i] for i in off_idx if fired[i]],
        [nerve.fiber_y_um[i] for i in off_idx if fired[i]],
        s=140, marker="s", facecolor="#c0392b", edgecolor="k",
        linewidth=1.0, zorder=5,
    )
    ax.scatter(
        [nerve.fiber_x_um[i] for i in off_idx if not fired[i]],
        [nerve.fiber_y_um[i] for i in off_idx if not fired[i]],
        s=140, marker="s", facecolor="none", edgecolor="#c0392b",
        linewidth=1.4, zorder=5,
    )

    # ── contacts ─────────────────────────────────────────────────────────────
    amax = max(float(np.max(np.abs(amps_mA))), 1e-6)
    for k in range(n_contacts):
        cx, cy = float(contact_xyz[k, 0]), float(contact_xyz[k, 1])
        a = float(amps_mA[k])
        if   a < -1e-4: ccol = "#2ecc71"        # cathodic = green
        elif a > +1e-4: ccol = "#e67e22"        # anodic   = orange (distinct from off-target red)
        else:           ccol = "0.6"            # off
        ms = 180 + 320 * (abs(a) / amax)
        ax.scatter([cx], [cy], s=ms, marker="^", facecolor=ccol,
                    edgecolor="k", linewidth=1.0, zorder=4)
        # outward label
        rr = (cx**2 + cy**2) ** 0.5
        if rr > 0:
            tx, ty = cx * 1.10, cy * 1.10
        else:
            tx, ty = cx + 80, cy + 80
        ax.annotate(f"C{k}\n{a:+.2f} mA", xy=(cx, cy), xytext=(tx, ty),
                    fontsize=9, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                edgecolor="0.7", alpha=0.85))
        if contact_xyz_init is not None:
            ix, iy = float(contact_xyz_init[k, 0]), float(contact_xyz_init[k, 1])
            if abs(ix - cx) + abs(iy - cy) > 1e-3:
                ax.annotate("", xy=(cx, cy), xytext=(ix, iy),
                            arrowprops=dict(arrowstyle="->", color="k",
                                             lw=0.9, alpha=0.6, mutation_scale=12),
                            zorder=3)
                ax.scatter([ix], [iy], s=40, marker="o", facecolor="0.5",
                            edgecolor="k", lw=0.5, alpha=0.7, zorder=3)

    lim = cuff_radius_um * 1.25
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("x (µm)", fontsize=11)
    ax.set_ylabel("y (µm)", fontsize=11)
    ax.grid(alpha=0.25, lw=0.4)

    # ── legend & title ──────────────────────────────────────────────────────
    n_tgt_fired = int(fired[target].sum());  n_tgt = int(target.sum())
    n_off_fired = int(fired[~target].sum()); n_off = int((~target).sum())

    legend_handles = [
        mpatches.Patch(facecolor="#cfe2f3", edgecolor="C0", label="target fascicle"),
        mpatches.Patch(facecolor="none", edgecolor="#c0392b",
                        linewidth=1.8, label="off-target fascicle"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C0",
                markeredgecolor="k", markersize=11, label="target fibre — fired"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                markeredgecolor="C0", markeredgewidth=1.8, markersize=11,
                label="target fibre — silent"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#c0392b",
                markeredgecolor="k", markersize=10, label="off-target fibre — fired"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="none",
                markeredgecolor="#c0392b", markeredgewidth=1.4, markersize=10,
                label="off-target fibre — silent"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#2ecc71",
                markeredgecolor="k", markersize=12, label="contact (cathodic)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#e67e22",
                markeredgecolor="k", markersize=12, label="contact (anodic)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left",
                bbox_to_anchor=(1.02, 1.0), fontsize=9, framealpha=0.95)

    fig.suptitle(
        f"{title}\nSI baseline = {si_baseline:+.3f}   →   SI = {si:+.3f}    "
        f"target fired {n_tgt_fired}/{n_tgt}    "
        f"off-target fired {n_off_fired}/{n_off}",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_seed_summary(
    out_path: pathlib.Path,
    nerve,                    # NerveGeometry with .fiber_x_um, .fiber_y_um,
                                # .fiber_diam, .target_mask
    contact_xyz_final,        # [K, 3] mm-position of contacts at end of opt
    amps_mA,                  # [K] final rect amplitudes (mA)
    acts,                     # [n_fibers] activation proxy ∈ [0, 1]
    loss_hist,                # list of per-iter loss values
    si_hist,                  # list of per-iter SI values
    si_baseline: float,
    title: str = "",
    contact_xyz_init=None,    # optional [K, 3] for joint-opt position arrows
    nerve_radius_um: float = 500.0,
    cuff_radius_um: float = 1500.0,
    activation_threshold: float = 0.5,
) -> None:
    """One per-seed summary figure for selectivity optimisation runs.

    Panels (2 × 2 grid):
      A — nerve cross-section: fibers coloured by (target / off-target) AND
          filled-vs-hollow by activation status (fired / silent).  Contacts
          drawn as triangles, colour-coded cathodic (green) / anodic (red),
          marker size proportional to |amp|.
      B — per-fiber activation proxy bars (coloured by target / off-target,
          threshold line at 0.5).
      C — loss curve (left axis) and SI curve (right axis) over iterations.
      D — final per-contact amplitudes (bars coloured cathodic / anodic).

    For joint-opt, pass `contact_xyz_init` to overlay arrows from the
    initial to the final contact positions on panel A.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    target = np.asarray(nerve.target_mask, dtype=bool)
    n_fibers = nerve.n_fibers
    n_contacts = len(amps_mA)
    acts = np.asarray(acts)
    fired = acts > activation_threshold
    si_best = max(si_hist) if len(si_hist) else float("nan")
    si_last = si_hist[-1] if len(si_hist) else float("nan")

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)

    # ── A: cross-section ────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.set_aspect("equal")
    ax.add_patch(plt.Circle((0, 0), nerve_radius_um, fill=False,
                              edgecolor="k", lw=1.5))
    fiber_r = max(nerve_radius_um * 0.025, 12.0)   # uniform display radius
    for f in range(n_fibers):
        is_tgt = bool(target[f])
        col = "C0" if is_tgt else "C1"            # blue=target, orange=off
        if fired[f]:
            face, edge, lw, alpha = col, "k",      1.6, 1.0
        else:
            face, edge, lw, alpha = "white", col, 1.2, 0.9
        ax.add_patch(plt.Circle(
            (nerve.fiber_x_um[f], nerve.fiber_y_um[f]), fiber_r,
            facecolor=face, edgecolor=edge, lw=lw, alpha=alpha, zorder=3,
        ))
        ax.text(nerve.fiber_x_um[f], nerve.fiber_y_um[f], f"F{f}",
                ha="center", va="center", fontsize=6,
                color="white" if fired[f] else col, zorder=4,
                fontweight="bold")
    # contacts
    amax = max(float(np.max(np.abs(amps_mA))), 1e-6)
    for k in range(n_contacts):
        cx, cy = float(contact_xyz_final[k, 0]), float(contact_xyz_final[k, 1])
        a = float(amps_mA[k])
        if   a < -1e-4: ccol = "C2"               # cathodic = green
        elif a > +1e-4: ccol = "C3"               # anodic   = red
        else:           ccol = "0.6"              # off
        ms = 8 + 14 * (abs(a) / amax)
        ax.plot(cx, cy, marker="^", color=ccol, ms=ms,
                markeredgecolor="k", markeredgewidth=0.8, zorder=2)
        # label with amp
        rr = (cx**2 + cy**2) ** 0.5
        if rr > 0:
            tx, ty = cx * 1.1, cy * 1.1
        else:
            tx, ty = cx + 50, cy + 50
        ax.text(tx, ty, f"C{k}\n{a:+.2f}", fontsize=7, ha="center", va="center")
        if contact_xyz_init is not None:
            ix, iy = float(contact_xyz_init[k, 0]), float(contact_xyz_init[k, 1])
            if abs(ix - cx) + abs(iy - cy) > 1e-3:
                ax.annotate("", xy=(cx, cy), xytext=(ix, iy),
                            arrowprops=dict(arrowstyle="->", color="k",
                                             lw=0.7, alpha=0.6, mutation_scale=10),
                            zorder=1)
                ax.plot(ix, iy, "o", color="0.4", ms=4, alpha=0.6, zorder=1)
    lim = cuff_radius_um * 1.25
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    n_tgt_fired = int(fired[target].sum());  n_tgt = int(target.sum())
    n_off_fired = int(fired[~target].sum()); n_off = int((~target).sum())
    ax.set_title(
        f"A   Cross-section + activation\n"
        f"target fired {n_tgt_fired}/{n_tgt}   off-target fired {n_off_fired}/{n_off}",
        fontsize=10, fontweight="bold",
    )
    leg_handles = [
        mpatches.Patch(facecolor="C0", edgecolor="k", label="target fired"),
        mpatches.Patch(facecolor="white", edgecolor="C0", label="target silent"),
        mpatches.Patch(facecolor="C1", edgecolor="k", label="off-target fired"),
        mpatches.Patch(facecolor="white", edgecolor="C1", label="off-target silent"),
    ]
    ax.legend(handles=leg_handles, loc="upper right", fontsize=7, framealpha=0.85)
    ax.grid(alpha=0.25, lw=0.4)

    # ── B: per-fiber activation bars ────────────────────────────────────────
    ax = axes[0, 1]
    colors = ["C0" if t else "C1" for t in target]
    bars = ax.bar(np.arange(n_fibers), acts, color=colors,
                   edgecolor="k", lw=0.5)
    for f in range(n_fibers):
        if not fired[f]:
            bars[f].set_alpha(0.45)
    ax.axhline(activation_threshold, color="gray", ls="--", lw=0.8,
                label=f"threshold = {activation_threshold}")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("fiber #"); ax.set_ylabel("activation proxy")
    ax.set_title(
        f"B   Per-fiber activation   "
        f"(SI baseline {si_baseline:+.3f}, best {si_best:+.3f}, last {si_last:+.3f})",
        fontsize=10, fontweight="bold",
    )
    ax.legend(handles=[
        mpatches.Patch(facecolor="C0", label="target"),
        mpatches.Patch(facecolor="C1", label="off-target"),
    ] + ax.get_legend_handles_labels()[0], loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3, lw=0.4)
    if n_fibers <= 30:
        ax.set_xticks(np.arange(n_fibers))
        ax.set_xticklabels([f"F{f}" for f in range(n_fibers)],
                            rotation=45, fontsize=7)

    # ── C: training curves ──────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(loss_hist, color="C0", lw=1.5, label="WQ loss")
    ax.set_xlabel("iteration"); ax.set_ylabel("WQ loss", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax.grid(alpha=0.3, lw=0.4)
    ax2 = ax.twinx()
    ax2.plot(si_hist, color="C3", lw=1.5, ls="--", label="SI")
    ax2.axhline(0, color="k", lw=0.4, alpha=0.4)
    ax2.set_ylabel("SI", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax2.set_ylim(-1.05, 1.05)
    ax.set_title("C   Training curves", fontsize=10, fontweight="bold")

    # ── D: per-contact amplitudes ───────────────────────────────────────────
    ax = axes[1, 1]
    ccols = ["C2" if a < 0 else ("C3" if a > 0 else "0.6") for a in amps_mA]
    bars = ax.bar(np.arange(n_contacts), amps_mA, color=ccols,
                   edgecolor="k", lw=0.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(np.arange(n_contacts))
    ax.set_xticklabels([f"C{k}" for k in range(n_contacts)])
    ax.set_ylabel("amplitude (mA)")
    for k in range(n_contacts):
        ax.text(k, amps_mA[k] + 0.02 * np.sign(amps_mA[k] or 1),
                f"{amps_mA[k]:+.2f}",
                ha="center", va="bottom" if amps_mA[k] >= 0 else "top",
                fontsize=8)
    ax.set_title("D   Optimised amplitudes   (green = cathodic, red = anodic)",
                  fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, lw=0.4)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt2
    plt2.close(fig)
    print(f"  -> {out_path}")
