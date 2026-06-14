"""Figure 1: JAXON validation and computational efficacy.

16:9 landscape (14 × 7.875 in), 3-row × 4-col outer grid, Nature Medicine style:

  [0, 0:2]  a — AP traces + gating variables (2-subrow × 4 models; stim overlaid)
  [0, 2:4]  b — Strength-Duration Curves (2×2)
  [1:3, 0:2] c — Activation Thresholds: scatter + per-model + by-waveform + by-diam (2×2, TALL)
  [1, 2:4]  d — Conduction Velocities (1×2)
  [2, 2:4]  e — Computational Efficacy (1×2)

Run from project root:
    python -m experiments_v2.figures_validation_main
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "manuscript" / "figures" / "main"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "MRG":     "#0072B2",
    "Sundt":   "#D55E00",
    "Sweeney": "#009E73",
    "Rattay":  "#CC79A7",
    "grey":    "#3B3B3B",
    "lgrey":   "#AAAAAA",
}
GATE_COL = {
    "m":  "#E41A1C",
    "h":  "#4DAF4A",
    "mp": "#984EA3",
    "s":  "#FF7F00",
    "n":  "#377EB8",
    "l":  "#A65628",
}
MODELS = ["MRG", "Sundt", "Sweeney", "Rattay"]

PULSE_LABELS = {
    "bi_ac": "bi AC", "bi_ca": "bi CA", "exp": "exp",
    "gaussian": "gauss", "mono_a": "mono A", "mono_c": "mono C",
    "sawtooth": "saw", "sine": "sine",
}
DIAM_BINS = [
    ("<1.5µm",      lambda d: d < 1.5),
    ("1.5–5µm",     lambda d: 1.5 <= d < 5.0),
    ("5–8µm",       lambda d: 5.0 <= d < 8.0),
    ("8–12µm",      lambda d: 8.0 <= d < 12.0),
    ("12–16µm",     lambda d: 12.0 <= d <= 16.0),
]

FS    = 8
FS_SM = 7

mpl.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size":         FS,
    "axes.labelsize":    FS,
    "axes.titlesize":    FS_SM,
    "xtick.labelsize":   FS_SM,
    "ytick.labelsize":   FS_SM,
    "legend.fontsize":   FS_SM,
    "axes.linewidth":    0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size":  2.5,
    "ytick.major.size":  2.5,
    "lines.linewidth":   0.8,
    "patch.linewidth":   0.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
    "savefig.dpi":       600,
    "savefig.bbox":      "tight",
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "pdf.fonttype":      42,
})


# ── Visual helpers ────────────────────────────────────────────────────────────
def _lighten(color, amount: float = 0.55):
    """Blend a color toward white. amount=0 → original, 1 → white.

    Used so PyFibers (dashed, on top) appears as a lighter tint of the same
    hue as JAXON (solid, beneath); since the curves overlap perfectly, the
    darker JAXON line shows through the dashes."""
    r, g, b = mpl.colors.to_rgb(color)
    return (r + (1.0 - r) * amount,
            g + (1.0 - g) * amount,
            b + (1.0 - b) * amount)


def _pulse_icon_xy(kind: str):
    """Normalised (t in [0,1], y in [-1,1]) glyph for a stimulus waveform,
    drawn as a small icon above the by-waveform error boxes.  Cathodic =
    downward by convention."""
    t = np.linspace(0.0, 1.0, 240)
    z = np.zeros_like(t)
    if kind == "mono_c":            # monophasic cathodic
        return t, np.where((t > 0.38) & (t < 0.62), -1.0, 0.0)
    if kind == "mono_a":            # monophasic anodic
        return t, np.where((t > 0.38) & (t < 0.62), 1.0, 0.0)
    if kind == "bi_ca":             # biphasic, cathodic-first
        y = np.where((t > 0.30) & (t < 0.50), -1.0, z)
        return t, np.where((t > 0.50) & (t < 0.70), 1.0, y)
    if kind == "bi_ac":             # biphasic, anodic-first
        y = np.where((t > 0.30) & (t < 0.50), 1.0, z)
        return t, np.where((t > 0.50) & (t < 0.70), -1.0, y)
    if kind == "sine":              # one period, cathodic-first
        m = (t > 0.20) & (t < 0.80)
        return t, np.where(m, -np.sin(2 * np.pi * (t - 0.20) / 0.60), 0.0)
    if kind == "sawtooth":          # cathodic ramp with sharp reset
        m = (t > 0.28) & (t < 0.74)
        return t, np.where(m, -(t - 0.28) / 0.46, 0.0)
    if kind == "exp":               # exponentially-decaying cathodic
        m = (t > 0.32) & (t < 0.82)
        return t, np.where(m, -np.exp(-(t - 0.32) / 0.46 * 4.0), 0.0)
    if kind == "gaussian":          # gaussian cathodic bump
        return t, -np.exp(-((t - 0.50) / 0.09) ** 2)
    return t, z


def _draw_axon_glyph(ax, color, lw: float = 4.0, myelinated: bool = False,
                     border: str = "#3a3a3a"):
    """Draw a small horizontal axon icon as an outlined tube into inset axes
    ``ax``.  Myelinated = capsule internodes separated by node-of-Ranvier
    gaps over a thin core; unmyelinated = one smooth tube.  ``lw`` (points)
    sets the fibre thickness; a very thin dark-grey border outlines the tube
    (drawn as a slightly wider line beneath the coloured fill)."""
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-1.0, 1.0)
    ax.axis("off")
    bw = 0.7  # total border width added around the fill (points)
    if myelinated:
        core = max(lw * 0.30, 0.6)
        ax.plot([0.03, 0.97], [0, 0], color=border, lw=core + bw,
                solid_capstyle="butt", zorder=1)
        ax.plot([0.03, 0.97], [0, 0], color=color, lw=core,
                solid_capstyle="butt", zorder=2)
        for x0, x1 in [(0.05, 0.30), (0.37, 0.62), (0.69, 0.95)]:
            ax.plot([x0, x1], [0, 0], color=border, lw=lw + bw,
                    solid_capstyle="round", zorder=3)
            ax.plot([x0, x1], [0, 0], color=color, lw=lw,
                    solid_capstyle="round", zorder=4)
    else:
        ax.plot([0.04, 0.96], [0, 0], color=border, lw=lw + bw,
                solid_capstyle="round", zorder=1)
        ax.plot([0.04, 0.96], [0, 0], color=color, lw=lw,
                solid_capstyle="round", zorder=2)


# ── Data helpers ──────────────────────────────────────────────────────────────
def _load_sd(model: str) -> dict:
    p = ROOT / "outputs" / f"{model.lower()}_validation" / f"data_{model.lower()}_sd.json"
    with open(p) as f:
        return json.load(f)


def _load_cv(model: str) -> dict:
    p = ROOT / "outputs" / f"{model.lower()}_validation" / f"data_{model.lower()}_cv.json"
    with open(p) as f:
        return json.load(f)


def _load_traces(model: str) -> dict:
    p = ROOT / "outputs" / f"{model.lower()}_validation" / f"data_{model.lower()}_traces.json"
    with open(p) as f:
        return json.load(f)


def _infer_pw_ms(model: str, trace: dict) -> float:
    """Return pulse width (ms) used for this trace by exact-matching its threshold to the SD curve."""
    thr = float(trace.get("threshold_extra_mA", trace.get("threshold_intra_mA", 0.0)))
    diam = float(trace.get("diameter", 0.0))
    try:
        sd = _load_sd(model)
    except FileNotFoundError:
        return 0.2
    diam_key = next((k for k in sd if abs(float(k) - diam) < 0.1), None)
    if diam_key is None:
        return 0.2
    best_pw, best_err = 0.2, float("inf")
    for pt_data in sd[diam_key].values():
        for pw_str, val in pt_data.items():
            if not isinstance(val, dict) or "jax" not in val:
                continue
            err = abs(float(val["jax"]) - thr)
            if err < best_err:
                best_err, best_pw = err, float(pw_str)
    return best_pw


def _flatten_sd(sd: dict, model: str) -> list[dict]:
    rows: list[dict] = []
    for diam_str, by_pulse in sd.items():
        diam = float(diam_str)
        for pulse, by_pw in by_pulse.items():
            for pw_str, vals in by_pw.items():
                if not isinstance(vals, dict) or "jax" not in vals:
                    continue
                rows.append(dict(
                    model=model, diam_um=diam, pulse=pulse,
                    PW_ms=float(pw_str),
                    jax_mA=float(vals["jax"]),
                    pyfibers_mA=float(vals["pyfibers"]),
                    err_abs_pct=abs(float(vals.get("err_pct", float("nan")))),
                ))
    return rows


def _gather_all_rows() -> list[dict]:
    out: list[dict] = []
    for m in MODELS:
        try:
            out.extend(_flatten_sd(_load_sd(m), m))
        except FileNotFoundError as e:
            print(f"  [warn] {e}", file=sys.stderr)
    return out


# ── Panel heading ─────────────────────────────────────────────────────────────
def _heading(ax: plt.Axes, letter: str, title: str,
             dx: float = -0.20, dy: float = 1.10) -> None:
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="bottom", ha="left", clip_on=False)
    ax.text(0.0, dy, title, transform=ax.transAxes,
            fontsize=11, va="bottom", ha="left", clip_on=False)


# ── Panel a: AP traces (Vm + gating), stim overlaid on Vm ────────────────────
def _panel_a(gs_cell, fig) -> plt.Axes:
    gs_ap = gridspec.GridSpecFromSubplotSpec(
        2, 4, subplot_spec=gs_cell,
        hspace=0.22, wspace=0.28,
        height_ratios=[1.0, 0.65],
    )

    T_STIM  = 1.0     # stimulus onset (ms) — 1 ms equilibration before pulse
    VM_CLIP = 8.0     # trace time window (ms)

    first_ax = None

    for mi, m in enumerate(MODELS):
        ax_vm = fig.add_subplot(gs_ap[0, mi])
        ax_ga = fig.add_subplot(gs_ap[1, mi])
        if first_ax is None:
            first_ax = ax_vm
        col = PALETTE[m]

        try:
            d = _load_traces(m)
        except FileNotFoundError:
            for ax in (ax_vm, ax_ga):
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, fontsize=FS_SM - 1,
                        color=PALETTE["lgrey"])
                ax.set_xticks([]); ax.set_yticks([])
            continue

        t_j  = np.asarray(d["t_extra_jax"],  dtype=float)
        vm_j = np.asarray(d["vm_extra_jax"], dtype=float)
        t_n  = np.asarray(d.get("t_extra_nrn",  d["t_intra_nrn"]),  dtype=float)
        vm_n = np.asarray(d.get("vm_extra_nrn", d["vm_intra_nrn"]), dtype=float)
        gj   = d.get("gates_extra_jax", d.get("gates_intra_jax", {}))
        gn   = d.get("gates_extra_nrn", d.get("gates_intra_nrn", {}))
        amp  = float(d.get("amp_extra_mA", d.get("amp_intra_mA", -0.1)))
        pw_ms = _infer_pw_ms(m, d)  # exact PW used for this trace

        # ── Vm traces ──────────────────────────────────────────────
        mj = t_j <= VM_CLIP
        mn = t_n <= VM_CLIP
        ax_vm.plot(t_j[mj], vm_j[mj], color=col, lw=1.0, zorder=3)
        ax_vm.plot(t_n[mn], vm_n[mn], color=PALETTE["lgrey"],
                   lw=0.9, ls="--", zorder=4)
        ax_vm.set_xlim(0, VM_CLIP)
        ax_vm.set_xticks([])

        # ── Stimulus: solid black step drawn in Vm axes at resting potential ──
        # Baseline sits at vm_rest; cathodic pulse steps down by 10% of AP height.
        vm_rest_val = float(vm_j[mj][0])
        vm_max_val  = float(np.max(vm_j[mj]))
        vm_height   = max(vm_max_val - vm_rest_val, 1.0)
        pulse_vis   = 0.10 * vm_height

        t_s      = np.linspace(0, VM_CLIP, 4000)
        stim_arr = np.full_like(t_s, vm_rest_val)
        stim_arr[(t_s >= T_STIM) & (t_s < T_STIM + pw_ms)] -= pulse_vis

        ax_vm.plot(t_s, stim_arr, color="black", lw=0.9, zorder=5)
        ax_vm.set_ylim(vm_rest_val - 0.18 * vm_height,
                       vm_max_val  + 0.04 * vm_height)

        ax_vm.text(0.50, 1.04, m, transform=ax_vm.transAxes,
                   ha="center", va="bottom", fontsize=FS_SM,
                   color=col, weight="semibold", clip_on=False)

        if mi == 0:
            ax_vm.set_ylabel(r"$V_m$ (mV)", fontsize=FS_SM)
            ax_vm.legend(
                handles=[
                    mlines.Line2D([], [], color=PALETTE["lgrey"], ls="--",
                                  lw=0.9, label="PyFibers"),
                    mlines.Line2D([], [], color=PALETTE["grey"],  ls="-",
                                  lw=1.0, label="JAXON"),
                    mlines.Line2D([], [], color="black", ls="-", lw=0.9,
                                  label="stim"),
                ],
                frameon=False, fontsize=FS_SM - 1, loc="upper right",
                handlelength=1.0, handletextpad=0.3, labelspacing=0.20,
            )
        else:
            ax_vm.set_yticklabels([])

        # ── Gating variables ───────────────────────────────────────
        for gname, gdata in gj.items():
            gcol = GATE_COL.get(gname, "#888888")
            mg   = t_j <= VM_CLIP
            ax_ga.plot(t_j[mg], np.asarray(gdata)[mg],
                       color=gcol, lw=0.75, label=gname)
            if gname in gn:
                mn2 = t_n <= VM_CLIP
                ax_ga.plot(t_n[mn2], np.asarray(gn[gname])[mn2],
                           color=gcol, lw=0.75, ls="--", alpha=0.55)
        ax_ga.set_xlim(0, VM_CLIP)
        ax_ga.set_ylim(0, 1.0)
        ax_ga.set_xlabel("time (ms)", fontsize=FS_SM)
        if mi == 0:
            ax_ga.set_ylabel("gating (a.u.)", fontsize=FS_SM)
        else:
            ax_ga.set_yticklabels([])
        ax_ga.legend(frameon=False, fontsize=FS_SM - 1, loc="upper right",
                     handlelength=0.9, labelspacing=0.18,
                     ncol=2, columnspacing=0.4)

    return first_ax


# ── Panel b: SD curves (2×2) ──────────────────────────────────────────────────
def _panel_b(gs_cell, fig) -> plt.Axes:
    gs_in = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_cell, wspace=0.32, hspace=0.44,
    )
    axes = [[fig.add_subplot(gs_in[r, c]) for c in range(2)] for r in range(2)]
    for ax_idx, (m, ax) in enumerate(zip(MODELS, [axes[r][c]
                                                    for r in range(2) for c in range(2)])):
        try:
            sd = _load_sd(m)
        except FileNotFoundError:
            ax.text(0.5, 0.5, f"{m}\nno data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=FS_SM)
            continue
        diams = sorted(float(d) for d in sd.keys())
        cmap  = plt.cm.viridis(np.linspace(0.05, 0.90, len(diams)))
        for D, clr in zip(diams, cmap):
            key   = str(D) if str(D) in sd else f"{D:.1f}"
            pk    = "mono_c" if "mono_c" in sd[key] else next(iter(sd[key]))
            by_pw = sd[key][pk]
            pws   = sorted(float(pw) for pw in by_pw.keys())
            def _v(pw, k, _b=by_pw):
                ky = str(pw) if str(pw) in _b else f"{pw:.2f}"
                return _b[ky][k]
            thr_j  = np.array([abs(_v(pw, "jax"))      for pw in pws])
            thr_py = np.array([abs(_v(pw, "pyfibers")) for pw in pws])
            ax.plot(pws, thr_j,  ls="-",  marker="o", ms=3.5,
                    color=clr, lw=0.65, zorder=2)
            ax.plot(pws, thr_py, ls="--", marker="x", ms=3.5,
                    color=_lighten(clr), lw=0.65, zorder=3)
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        ax.text(0.97, 0.97, m, transform=ax.transAxes,
                ha="right", va="top", fontsize=FS_SM, weight="semibold",
                color=PALETTE[m])
        if ax_idx in (0, 2):
            ax.set_ylabel("threshold (mA)")
        if ax_idx in (2, 3):
            ax.set_xlabel("pulse width (ms)")
        if ax_idx == 0:
            ax.legend(
                handles=[
                    mlines.Line2D([], [], color="black", marker="o", ls="-",
                                  ms=3.5, lw=0.65, label="JAXON"),
                    mlines.Line2D([], [], color=_lighten("black"), marker="x",
                                  ls="--", ms=3.5, lw=0.65, label="PyFibers"),
                ],
                frameon=False, fontsize=FS_SM, loc="upper right",
                bbox_to_anchor=(1.0, 0.86),
                handlelength=1.1,
            )
    return axes[0][0]


# ── Panel c: Activation thresholds (2×2) — spans 2 rows, TALL ────────────────
def _panel_c(gs_cell, fig, rows: list[dict]) -> plt.Axes:
    gs_in = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_cell, wspace=0.45, hspace=0.48,
    )
    ax_sc  = fig.add_subplot(gs_in[0, 0])
    ax_mod = fig.add_subplot(gs_in[0, 1])
    ax_wav = fig.add_subplot(gs_in[1, 0])
    ax_dia = fig.add_subplot(gs_in[1, 1])
    rng = np.random.default_rng(0)

    # ── scatter ───────────────────────────────────────────────────────────────
    all_py, all_jx = [], []
    for m in MODELS:
        pairs = [(abs(r["pyfibers_mA"]), abs(r["jax_mA"])) for r in rows
                 if r["model"] == m and np.isfinite(r["pyfibers_mA"])
                 and np.isfinite(r["jax_mA"]) and r["pyfibers_mA"] != 0.0]
        if not pairs:
            continue
        x = np.array([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])
        ax_sc.scatter(x, y, s=12, alpha=0.55, color=PALETTE[m], label=m,
                      edgecolors="none")
        all_py.append(x); all_jx.append(y)
    if all_py:
        xa, ya = np.concatenate(all_py), np.concatenate(all_jx)
        hi = max(xa.max(), ya.max()) * 1.04
        ax_sc.plot([0, hi], [0, hi], color=PALETTE["grey"], ls="--", lw=0.8)
        r2   = 1.0 - float(np.sum((ya - xa)**2) / np.sum((ya - ya.mean())**2))
        mape = float(np.median(np.abs((ya - xa) / xa))) * 100.0
        ax_sc.text(0.04, 0.96,
                   f"$n$ = {len(xa)}\n$R^2$ = {r2:.5f}\nmedian |err| = {mape:.2f}%",
                   transform=ax_sc.transAxes, ha="left", va="top", fontsize=FS_SM,
                   bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                             edgecolor=PALETTE["lgrey"], linewidth=0.5))
        ax_sc.set_xlim(0, hi); ax_sc.set_ylim(0, hi)
    ax_sc.set_xlabel("PyFibers / NEURON (mA)")
    ax_sc.set_ylabel("JAXON (mA)")
    ax_sc.legend(frameon=False, fontsize=FS_SM, loc="lower right",
                 markerscale=1.5, handletextpad=0.3)

    # ── per-model error ───────────────────────────────────────────────────────
    for xi, m in enumerate(MODELS):
        errs = np.array([r["err_abs_pct"] for r in rows
                         if r["model"] == m and np.isfinite(r["err_abs_pct"])])
        if not errs.size:
            continue
        col = PALETTE[m]
        bp = ax_mod.boxplot(
            [errs], positions=[xi], widths=0.42,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=1.4),
            boxprops=dict(facecolor=col, alpha=0.30, edgecolor=col, linewidth=0.6),
            whiskerprops=dict(color=col, linewidth=0.7),
            capprops=dict(color=col, linewidth=0.7),
        )
        jit = rng.uniform(-0.14, 0.14, errs.size)
        ax_mod.scatter(xi + jit, errs, s=3, color=col,
                       alpha=0.40, edgecolors="none", zorder=3)
    ax_mod.set_xticks(range(len(MODELS)))
    ax_mod.set_xticklabels(MODELS, fontsize=FS_SM)
    ax_mod.set_ylabel("|err| (%)")
    ax_mod.set_xlim(-0.55, len(MODELS) - 0.45)
    ax_mod.set_yscale("log")
    ax_mod.tick_params(axis="x", length=0)

    # ── by waveform ───────────────────────────────────────────────────────────
    pulses = sorted({r["pulse"] for r in rows})
    wav_colors = plt.cm.tab10(np.arange(len(pulses)) / 10.0)
    for xi, pulse in enumerate(pulses):
        errs = np.array([r["err_abs_pct"] for r in rows
                         if r["pulse"] == pulse and np.isfinite(r["err_abs_pct"])])
        if not errs.size:
            continue
        col = wav_colors[xi]
        bp = ax_wav.boxplot(
            [errs], positions=[xi], widths=0.42,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=1.4),
            boxprops=dict(facecolor=col, alpha=0.30, edgecolor=col, linewidth=0.6),
            whiskerprops=dict(color=col, linewidth=0.7),
            capprops=dict(color=col, linewidth=0.7),
        )
        jit = rng.uniform(-0.14, 0.14, errs.size)
        ax_wav.scatter(xi + jit, errs, s=3, color=col,
                       alpha=0.40, edgecolors="none", zorder=3)
    ax_wav.set_xticks(range(len(pulses)))
    ax_wav.set_xticklabels([PULSE_LABELS.get(p, p) for p in pulses],
                           rotation=35, ha="right", fontsize=FS_SM)
    ax_wav.set_ylabel("|err| (%)")
    ax_wav.set_xlim(-0.6, len(pulses) - 0.4)
    ax_wav.set_yscale("log")
    ax_wav.tick_params(axis="x", length=0)

    # Open a little headroom above the data so the waveform glyphs sit just
    # above the top whisker caps rather than floating above the axes.
    _ylo, _yhi = ax_wav.get_ylim()
    ax_wav.set_ylim(_ylo, _yhi * 4.5)

    # Representative stimulus-waveform glyph over each box, in a single row
    # tucked into the headroom band just above the whiskers.  Insets are in
    # axes-fraction coords (x mapped from the box position).
    _xspan = (len(pulses) - 0.4) - (-0.6)        # axis x-range in data units
    _wf    = 0.74 / _xspan                        # icon width  (axes fraction)
    for xi, pulse in enumerate(pulses):
        xf = (xi - (-0.6)) / _xspan               # box centre  (axes fraction)
        ic = ax_wav.inset_axes([xf - _wf / 2, 0.85, _wf, 0.12])
        tt, yy = _pulse_icon_xy(pulse)
        ic.plot(tt, yy, color=wav_colors[xi], lw=0.9, solid_capstyle="round")
        ic.axhline(0.0, color=PALETTE["lgrey"], lw=0.3, zorder=0)
        ic.set_xlim(-0.05, 1.05)
        ic.set_ylim(-1.35, 1.35)
        ic.axis("off")

    # ── by diameter range (finer bins, pooled across models, per-bin color) ───
    dia_colors = plt.cm.tab10(np.arange(len(DIAM_BINS)) / 10.0)
    _xtick_labels = []
    for xi, (label, cond) in enumerate(DIAM_BINS):
        errs = np.array([r["err_abs_pct"] for r in rows
                         if cond(r["diam_um"]) and np.isfinite(r["err_abs_pct"])])
        if not errs.size:
            _xtick_labels.append(f"{label}\n(n=0)")
            continue
        _dia_col = dia_colors[xi]
        bp = ax_dia.boxplot(
            [errs], positions=[xi], widths=0.42,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=1.4),
            boxprops=dict(facecolor=_dia_col, alpha=0.30,
                          edgecolor=_dia_col, linewidth=0.6),
            whiskerprops=dict(color=_dia_col, linewidth=0.7),
            capprops=dict(color=_dia_col, linewidth=0.7),
        )
        jit = rng.uniform(-0.14, 0.14, errs.size)
        ax_dia.scatter(xi + jit, errs, s=3, color=_dia_col,
                       alpha=0.40, edgecolors="none", zorder=3)
        _xtick_labels.append(f"{label}\n($n$={errs.size})")
    ax_dia.set_xticks(range(len(DIAM_BINS)))
    ax_dia.set_xticklabels(_xtick_labels, fontsize=FS_SM)
    ax_dia.set_ylabel("|err| (%)")
    ax_dia.set_xlim(-0.6, len(DIAM_BINS) - 0.4)
    ax_dia.set_yscale("log")
    ax_dia.tick_params(axis="x", length=0)

    # Headroom + axon-thickness glyph over each diameter box (thin -> thick).
    _dlo, _dhi = ax_dia.get_ylim()
    ax_dia.set_ylim(_dlo, _dhi * 4.5)
    _dxspan = (len(DIAM_BINS) - 0.4) - (-0.6)
    _dwf    = 0.74 / _dxspan
    _nb     = len(DIAM_BINS)
    for xi in range(_nb):
        xf = (xi - (-0.6)) / _dxspan
        ic = ax_dia.inset_axes([xf - _dwf / 2, 0.85, _dwf, 0.12])
        _draw_axon_glyph(ic, dia_colors[xi],
                         lw=1.6 + 4.2 * (xi / max(_nb - 1, 1)),
                         myelinated=False)
    return ax_sc


# ── Panel d: Conduction velocity ──────────────────────────────────────────────
def _panel_d(gs_cell, fig) -> plt.Axes:
    gs_in = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_cell, wspace=0.45)
    ax_a  = fig.add_subplot(gs_in[0])
    ax_c  = fig.add_subplot(gs_in[1])
    for models, ax, title, myel in [
        (["MRG", "Sweeney"], ax_a, "A-fibres", True),
        (["Sundt", "Rattay"], ax_c, "C-fibres", False),
    ]:
        all_diams = []
        for m in models:
            try:
                cv = _load_cv(m)
            except FileNotFoundError:
                continue
            diams = sorted(float(d) for d in cv.keys())
            if not diams:
                continue
            all_diams.extend(diams)
            def _val(d, k, _cv=cv):
                key = str(d) if str(d) in _cv else f"{d:.1f}"
                return _cv[key][k]
            jax_v = np.array([_val(d, "jax")      for d in diams])
            py_v  = np.array([_val(d, "pyfibers") for d in diams])
            if not np.all(np.isfinite(jax_v)):
                continue
            col = PALETTE[m]
            ax.plot(diams, jax_v, ls="-",  marker="o", ms=4.5,
                    color=col, label=f"JAXON {m}", lw=0.8, zorder=2)
            ax.plot(diams, py_v, ls="--", marker="x", ms=4.5,
                    color=_lighten(col), label=f"PyFibers {m}", lw=0.8, zorder=3)
        ax.text(0.04, 0.96, title, transform=ax.transAxes,
                ha="left", va="top", fontsize=FS_SM,
                color=PALETTE["grey"], weight="semibold")
        # Myelinated (A) vs unmyelinated (C) axon glyph, upper-left.
        gic = ax.inset_axes([0.06, 0.79, 0.20, 0.11])
        _draw_axon_glyph(gic, PALETTE["grey"], lw=4.0, myelinated=myel)
        ax.set_xlabel(r"fibre diameter (µm)")
        ax.set_ylabel("CV (m/s)")
        ax.set_ylim(bottom=0)
        if all_diams:
            span = max(all_diams) - min(all_diams)
            ax.set_xlim(min(all_diams) - span * 0.06,
                        max(all_diams) + span * 0.06)
        ax.legend(frameon=False, fontsize=FS_SM, loc="lower right",
                  ncol=1, handlelength=1.0, handletextpad=0.3)
    return ax_a


# ── Panel e: Computational efficacy ──────────────────────────────────────────
def _panel_e(gs_cell, fig) -> plt.Axes:
    gs_in = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_cell, wspace=0.48)
    ax_t  = fig.add_subplot(gs_in[0])
    ax_s  = fig.add_subplot(gs_in[1])
    p = ROOT / "outputs" / "scaling" / "data_scaling.json"
    if not p.exists():
        for ax in (ax_t, ax_s):
            ax.text(0.5, 0.5, "scaling data\nmissing", ha="center", va="center",
                    transform=ax.transAxes, fontsize=FS_SM)
        return ax_t
    with open(p) as f:
        data = json.load(f)
    for m_entry in data["models"]:
        model = m_entry["model"]
        if model not in MODELS:
            continue
        col = PALETTE[model]
        N   = np.array([int(n) for n in m_entry["N"]])
        py  = np.array([m_entry["pyfibers"][str(n)] for n in N])
        ek  = "jaxley_gpu" if "jaxley_gpu" in m_entry else "jaxley_cpu"
        jax = np.array([m_entry[ek]["run"][str(n)] for n in N])
        ax_t.plot(N, jax,      color=col, ls="-",  marker="o", ms=4.0, lw=0.7, zorder=2)
        ax_t.plot(N, py,       color=col, ls="--", marker="x", ms=4.0, lw=0.7, alpha=0.55, zorder=3)
        ax_s.plot(N, py / jax, color=col, marker="o", ms=4.0, label=model, lw=0.7)
    for ax in (ax_t, ax_s):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("N axons")
    ax_t.set_ylabel("wall-clock (s)")
    ax_t.legend(
        handles=[
            mlines.Line2D([], [], color=PALETTE["grey"], ls="-",  marker="o",
                          ms=4.0, lw=0.7, label="JAXON"),
            mlines.Line2D([], [], color=PALETTE["grey"], ls="--", marker="x",
                          ms=4.0, lw=0.7, alpha=0.55, label="PyFibers"),
        ],
        frameon=False, fontsize=FS_SM, loc="upper left", handlelength=1.1)
    ax_s.axhline(1.0, color=PALETTE["grey"], ls=":", lw=0.6)
    ax_s.set_ylabel("speedup (PyFibers / JAXON)")
    ax_s.legend(frameon=False, fontsize=FS_SM, loc="upper left",
                ncol=2, columnspacing=0.4, handlelength=0.8)
    return ax_t


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    rows = _gather_all_rows()
    if not rows:
        print("[figures_validation_main] no SD rows found", file=sys.stderr)
        return 1
    print(f"[figures_validation_main] {len(rows)} threshold rows")

    # 16:9 landscape
    fig = plt.figure(figsize=(14.0, 7.875))

    # Two-column outer grid.
    #   Left  column: panel a (compact AP panel, ~35% height)
    #                 panel c (large thresholds,  ~65% height)
    #   Right column: panel b (SD curves, 50% height — 2 internal subplot rows)
    #                 panel d (CV,         25% height — 1 internal subplot row)
    #                 panel e (scaling,    25% height — 1 internal subplot row)
    # height_ratios=[2,1,1] on the right means each SD-curve subplot row has the
    # same physical height as the CV subplot row and the scaling subplot row.
    gs_outer = gridspec.GridSpec(
        1, 2, figure=fig,
        left=0.06, right=0.98, top=0.93, bottom=0.10,
        wspace=0.18,
    )

    gs_left = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_outer[0, 0],
        height_ratios=[0.52, 1.0], hspace=0.28,
    )
    gs_right = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs_outer[0, 1],
        height_ratios=[2.0, 1.0, 1.0], hspace=0.52,
    )

    ax_a = _panel_a(gs_left[0], fig)
    ax_c = _panel_c(gs_left[1], fig, rows)
    ax_b = _panel_b(gs_right[0], fig)
    ax_d = _panel_d(gs_right[1], fig)
    ax_e = _panel_e(gs_right[2], fig)

    _heading(ax_a, "a", "Action Potentials, Gating & Stimulus",
             dx=-0.14, dy=1.18)
    _heading(ax_c, "c", "Activation Thresholds",
             dx=-0.14, dy=1.06)
    _heading(ax_b, "b", "Strength-Duration Curves",
             dx=-0.14, dy=1.10)
    _heading(ax_d, "d", "Conduction Velocities",
             dx=-0.14, dy=1.14)
    _heading(ax_e, "e", "Computational Efficacy",
             dx=-0.14, dy=1.14)

    for ext in (".png", ".svg"):
        path = OUT_DIR / f"fig1_validation{ext}"
        fig.savefig(path, dpi=300 if ext == ".png" else None)
        print(f"  -> {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
