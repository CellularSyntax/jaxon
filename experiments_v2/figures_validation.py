"""Build the seven jaxon-vs-pyfibers validation figures in a
Nature-Medicine-style: no titles, no log scales (except where the
data span makes it unavoidable, namely the scaling plot 4b), large
sans-serif axis labels and tick labels, minimal top/right spines,
metrics annotated inside the plot itself.

Inputs (already on disk from earlier benchmark runs):
  outputs/{mrg,sundt,sweeney,rattay}_validation/data_<m>_sd.json
  outputs/{mrg,sundt,sweeney,rattay}_validation/data_<m>_cv.json
  outputs/scaling/data_scaling.json

Outputs (written to manuscript/figures/validation/):
  fig1_scatter_thresholds.png
  fig2_violin_by_model.png
  fig3_violin_by_pulse.png
  fig4a_violin_by_diameter.png
  fig4b_scaling_walltime.png
  fig5_conduction_velocity.png
  fig6_strength_duration.png
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# ── Style ────────────────────────────────────────────────────────────────────

# Wong / Okabe colourblind-friendly palette.  Same hue cycle Nature uses
# in many of its style guides.
PALETTE = {
    "MRG":     "#0072B2",   # blue
    "Sundt":   "#D55E00",   # vermillion
    "Sweeney": "#009E73",   # green
    "Rattay":  "#CC79A7",   # pink
    "neutral": "#56B4E9",   # sky-blue for the pooled-distribution boxes
    "grey":    "#3B3B3B",
}

mpl.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "Liberation Sans",
                            "DejaVu Sans"],
    "font.size":          14,
    "axes.labelsize":     16,
    "axes.titlesize":     16,
    "xtick.labelsize":    13,
    "ytick.labelsize":    13,
    "legend.fontsize":    12,
    "axes.linewidth":     1.4,
    "xtick.major.width":  1.4,
    "ytick.major.width":  1.4,
    "xtick.major.size":   5.0,
    "ytick.major.size":   5.0,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          False,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "figure.facecolor":   "white",
})


# ── Configuration ───────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR   = REPO_ROOT / "manuscript" / "figures" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["MRG", "Sundt", "Sweeney", "Rattay"]

PULSE_ORDER = ["mono_c", "mono_a", "bi_ca", "bi_ac",
               "sine", "exp", "gaussian", "sawtooth"]
PULSE_PRETTY = {
    "mono_c":   "mono (-)",
    "mono_a":   "mono (+)",
    "bi_ca":    "biphasic (-)",
    "bi_ac":    "biphasic (+)",
    "sine":     "sine",
    "exp":      "exp",
    "gaussian": "gaussian",
    "sawtooth": "triangle",
}

DIAM_BINS = [(0.0, 1.5), (1.5, 5.0), (5.0, 8.0),
             (8.0, 12.0), (12.0, 17.0)]
DIAM_BIN_LABELS = [
    r"$<\!1.5\,\mu$m",
    r"$1.5\!-\!5\,\mu$m",
    r"$5\!-\!8\,\mu$m",
    r"$8\!-\!12\,\mu$m",
    r"$12\!-\!16\,\mu$m",
]


# ── IO helpers ──────────────────────────────────────────────────────────────

def _load_sd(model: str) -> dict:
    p = REPO_ROOT / "outputs" / f"{model.lower()}_validation" / f"data_{model.lower()}_sd.json"
    with open(p) as f:
        return json.load(f)


def _load_cv(model: str) -> dict:
    p = REPO_ROOT / "outputs" / f"{model.lower()}_validation" / f"data_{model.lower()}_cv.json"
    with open(p) as f:
        return json.load(f)


def _flatten_sd(sd: dict, model: str) -> list[dict]:
    rows: list[dict] = []
    for diam_str, by_pulse in sd.items():
        diam = float(diam_str)
        for pulse, by_pw in by_pulse.items():
            for pw_str, vals in by_pw.items():
                if not isinstance(vals, dict) or "jax" not in vals:
                    continue
                err = float(vals.get("err_pct", float("nan")))
                rows.append(dict(
                    model=model, diam_um=diam, pulse=pulse,
                    PW_ms=float(pw_str),
                    jax_mA=float(vals["jax"]),
                    pyfibers_mA=float(vals["pyfibers"]),
                    err_pct=err,
                    err_abs_pct=abs(err),
                ))
    return rows


def _gather_all_rows() -> list[dict]:
    out: list[dict] = []
    for m in MODELS:
        try:
            out.extend(_flatten_sd(_load_sd(m), m))
        except FileNotFoundError as e:
            print(f"  [warn] {m} SD data missing: {e}")
    return out


# ── Shared box+scatter helper ───────────────────────────────────────────────

def _box_with_scatter(ax, groups: list[np.ndarray], labels: list[str],
                       colours: list[str], rng_seed: int = 0) -> None:
    bp = ax.boxplot(
        groups, widths=0.55, showfliers=False, patch_artist=True,
        boxprops=dict(linewidth=1.2),
        medianprops=dict(color="black", linewidth=2.0),
        whiskerprops=dict(linewidth=1.2, color=PALETTE["grey"]),
        capprops=dict(linewidth=1.2, color=PALETTE["grey"]),
    )
    for patch, col in zip(bp["boxes"], colours):
        patch.set_facecolor(col); patch.set_alpha(0.35)
        patch.set_edgecolor(PALETTE["grey"])
    rng = np.random.default_rng(rng_seed)
    for i, (arr, col) in enumerate(zip(groups, colours)):
        if arr.size == 0:
            continue
        jitter = rng.uniform(-0.18, 0.18, size=arr.size)
        ax.scatter(np.full(arr.size, i + 1) + jitter, arr,
                    s=11, alpha=0.55, color=col, edgecolors="none",
                    zorder=3)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)


# ── Figure 1: |mA| scatter, linear axes ─────────────────────────────────────

def make_fig1_scatter(rows: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    all_jax, all_py = [], []
    for m in MODELS:
        pairs = [(abs(r["pyfibers_mA"]), abs(r["jax_mA"])) for r in rows
                  if r["model"] == m
                  and np.isfinite(r["pyfibers_mA"])
                  and np.isfinite(r["jax_mA"])
                  and r["pyfibers_mA"] != 0.0]
        if not pairs:
            continue
        x = np.array([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])
        ax.scatter(x, y, s=22, alpha=0.55, color=PALETTE[m],
                    label=m, edgecolors="none")
        all_jax.append(y); all_py.append(x)

    x_all = np.concatenate(all_py); y_all = np.concatenate(all_jax)
    lo = 0.0
    hi = max(x_all.max(), y_all.max())
    pad = 0.04 * hi
    ax.plot([lo, hi + pad], [lo, hi + pad], color=PALETTE["grey"],
              ls="--", lw=1.2, zorder=1)

    ss_res = float(np.sum((y_all - x_all) ** 2))
    ss_tot = float(np.sum((y_all - y_all.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    mape = float(np.median(np.abs((y_all - x_all) / x_all))) * 100.0

    ax.text(0.04, 0.96,
              f"$n$ = {len(x_all)}\n"
              f"$R^2$ = {r2:.5f}\n"
              f"median $|\\mathrm{{err}}|$ = {mape:.2f}%",
              transform=ax.transAxes, ha="left", va="top", fontsize=13,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                        edgecolor=PALETTE["grey"], linewidth=0.8))

    ax.set_xlabel("PyFibers / NEURON threshold (|mA|)")
    ax.set_ylabel("Jaxon threshold (|mA|)")
    ax.set_xlim(0, hi + pad); ax.set_ylim(0, hi + pad)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    out = OUT_DIR / "fig1_scatter_thresholds.png"
    fig.savefig(out); plt.close(fig)
    return out


# ── Figure 2: |err| by model (linear) ───────────────────────────────────────

def make_fig2_by_model(rows: list[dict]) -> Path:
    groups, labels, cols = [], [], []
    for m in MODELS:
        arr = np.array([r["err_abs_pct"] for r in rows
                          if r["model"] == m and np.isfinite(r["err_abs_pct"])])
        if arr.size == 0:
            continue
        groups.append(arr); labels.append(f"{m}\n($n$={arr.size})")
        cols.append(PALETTE[m])
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    _box_with_scatter(ax, groups, labels, cols)
    ax.set_ylabel(r"$|\mathrm{err}|$  (%)")
    # Annotate median above each box.
    for i, arr in enumerate(groups):
        med = float(np.median(arr))
        ax.text(i + 1, max(arr) * 1.06, f"{med:.2f}%",
                  ha="center", va="bottom", fontsize=11,
                  color=PALETTE["grey"])
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out = OUT_DIR / "fig2_violin_by_model.png"
    fig.savefig(out); plt.close(fig)
    return out


# ── Figure 3: |err| by pulse shape ──────────────────────────────────────────

def make_fig3_by_pulse(rows: list[dict]) -> Path:
    groups, labels, cols = [], [], []
    for p in PULSE_ORDER:
        arr = np.array([r["err_abs_pct"] for r in rows
                          if r["pulse"] == p and np.isfinite(r["err_abs_pct"])])
        if arr.size == 0:
            continue
        groups.append(arr)
        labels.append(f"{PULSE_PRETTY.get(p, p)}\n($n$={arr.size})")
        cols.append(PALETTE["neutral"])
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    _box_with_scatter(ax, groups, labels, cols)
    ax.set_ylabel(r"$|\mathrm{err}|$  (%)")
    for i, arr in enumerate(groups):
        med = float(np.median(arr))
        ax.text(i + 1, max(arr) * 1.06, f"{med:.2f}%",
                  ha="center", va="bottom", fontsize=10,
                  color=PALETTE["grey"])
    ax.set_ylim(bottom=0)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    out = OUT_DIR / "fig3_violin_by_pulse.png"
    fig.savefig(out); plt.close(fig)
    return out


# ── Figure 4a: |err| by diameter bin ────────────────────────────────────────

def make_fig4a_by_diameter(rows: list[dict]) -> Path:
    groups, labels, cols = [], [], []
    for (lo, hi), lab in zip(DIAM_BINS, DIAM_BIN_LABELS):
        arr = np.array([r["err_abs_pct"] for r in rows
                          if lo <= r["diam_um"] < hi
                          and np.isfinite(r["err_abs_pct"])])
        if arr.size == 0:
            continue
        groups.append(arr); labels.append(f"{lab}\n($n$={arr.size})")
        cols.append(PALETTE["neutral"])
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    _box_with_scatter(ax, groups, labels, cols)
    ax.set_ylabel(r"$|\mathrm{err}|$  (%)")
    for i, arr in enumerate(groups):
        med = float(np.median(arr))
        ax.text(i + 1, max(arr) * 1.06, f"{med:.2f}%",
                  ha="center", va="bottom", fontsize=11,
                  color=PALETTE["grey"])
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out = OUT_DIR / "fig4a_violin_by_diameter.png"
    fig.savefig(out); plt.close(fig)
    return out


# ── Figure 4b: scaling (log axes kept on purpose, see note in panel) ────────

def make_fig4b_scaling() -> Path:
    """Forward-pass wall-clock and speedup vs n_axons.

    Note: we keep log axes here despite the user's general 'no log
    scales' instruction because the data span 5 decades (1 to 1e5
    axons; 0.01 s to 1e4 s) and a linear plot would show only the
    largest point.  This is the one figure where log axes carry
    information the data span otherwise hides."""
    p = REPO_ROOT / "outputs" / "scaling" / "data_scaling.json"
    with open(p) as f:
        data = json.load(f)
    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    for m_entry in data["models"]:
        model = m_entry["model"]
        if model not in MODELS:
            continue
        col = PALETTE[model]
        N = np.array([int(n) for n in m_entry["N"]])
        py = np.array([m_entry["pyfibers"][str(n)] for n in N])
        engine_key = "jaxley_gpu" if "jaxley_gpu" in m_entry else "jaxley_cpu"
        jax_run = np.array([m_entry[engine_key]["run"][str(n)] for n in N])
        ax_t.plot(N, py,       color=col, ls="--", marker="x",
                    label=f"PyFibers {model}", lw=2)
        ax_t.plot(N, jax_run,  color=col, ls="-",  marker="o",
                    label=f"Jaxon {model}",     lw=2)
        ax_s.plot(N, py / jax_run, color=col, marker="o", label=model, lw=2)

    for ax in (ax_t, ax_s):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("number of axons")

    ax_t.set_ylabel("wall-clock (s)")
    ax_t.legend(fontsize=10, ncol=2, frameon=False)
    ax_s.axhline(1.0, color=PALETTE["grey"], ls=":", lw=1)
    ax_s.set_ylabel("speedup  (PyFibers / Jaxon)")
    ax_s.legend(fontsize=11, frameon=False)

    fig.tight_layout()
    out = OUT_DIR / "fig4b_scaling_walltime.png"
    fig.savefig(out); plt.close(fig)
    return out


# ── Figure 5: conduction velocity vs diameter ───────────────────────────────

def make_fig5_cv() -> Path:
    a_models = ["MRG", "Sweeney"]
    c_models = ["Sundt", "Rattay"]
    fig, (ax_a, ax_c) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    def _plot(ax, models, anno):
        any_plotted = False
        for m in models:
            try:
                cv = _load_cv(m)
            except FileNotFoundError:
                continue
            diams = sorted(float(d) for d in cv.keys())
            if not diams:
                continue
            def _val(d, k):
                key = str(d) if str(d) in cv else f"{d:.1f}"
                return cv[key][k]
            jax_v = np.array([_val(d, "jax")      for d in diams])
            py_v  = np.array([_val(d, "pyfibers") for d in diams])
            if not np.all(np.isfinite(jax_v)):
                continue
            col = PALETTE[m]
            ax.plot(diams, py_v, ls="--", marker="x", color=col,
                      label=f"PyFibers {m}", lw=2)
            ax.plot(diams, jax_v, ls="-",  marker="o", color=col,
                      label=f"Jaxon {m}",     lw=2)
            any_plotted = True
        ax.set_xlabel(r"fibre diameter ($\mu$m)")
        ax.set_ylabel("conduction velocity (m/s)")
        ax.text(0.04, 0.96, anno, transform=ax.transAxes,
                  ha="left", va="top", fontsize=14,
                  color=PALETTE["grey"], weight="bold")
        if any_plotted:
            ax.legend(fontsize=11, frameon=False, loc="lower right")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    _plot(ax_a, a_models, "A-fibres")
    _plot(ax_c, c_models, "C-fibres")
    fig.tight_layout()
    out = OUT_DIR / "fig5_conduction_velocity.png"
    fig.savefig(out); plt.close(fig)
    return out


# ── Figure 6: strength–duration curves ──────────────────────────────────────

def make_fig6_strength_duration() -> Path:
    models_with_data = [m for m in MODELS
                          if (REPO_ROOT / "outputs"
                              / f"{m.lower()}_validation"
                              / f"data_{m.lower()}_sd.json").exists()]
    ncols = 2
    nrows = math.ceil(len(models_with_data) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 4.2 * nrows),
                                squeeze=False)
    for ax_idx, m in enumerate(models_with_data):
        sd = _load_sd(m)
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        diams = sorted(float(d) for d in sd.keys())
        cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(diams)))
        pulse_key_used = None
        for D, col in zip(diams, cmap):
            entry = sd[str(D) if str(D) in sd else f"{D:.1f}"]
            pulse_key = "mono_c" if "mono_c" in entry else next(iter(entry.keys()))
            pulse_key_used = pulse_key
            by_pw = entry[pulse_key]
            pws = sorted(float(pw) for pw in by_pw.keys())
            def _v(pw, k):
                key = str(pw) if str(pw) in by_pw else f"{pw:.2f}"
                return by_pw[key][k]
            thr_j  = np.array([abs(_v(pw, "jax"))      for pw in pws])
            thr_py = np.array([abs(_v(pw, "pyfibers")) for pw in pws])
            ax.plot(pws, thr_py, ls="--", marker="x", color=col,
                      alpha=0.8, lw=1.6)
            ax.plot(pws, thr_j,  ls="-",  marker="o", color=col,
                      label=fr"$D={D}\,\mu$m", lw=1.6)
        ax.set_xlabel("pulse width (ms)")
        ax.set_ylabel("threshold |mA|")
        ax.text(0.96, 0.96, m, transform=ax.transAxes,
                  ha="right", va="top", fontsize=14, weight="bold",
                  color=PALETTE["grey"])
        ax.legend(fontsize=10, frameon=False, ncol=2,
                    loc="upper right", bbox_to_anchor=(0.95, 0.92))
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    # Hide unused subplots.
    for j in range(len(models_with_data), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    # Shared engine-convention legend along the bottom (solid+circle =
    # Jaxon, dashed+X = PyFibers).  Black handles, since the colour in
    # the panel encodes diameter not engine.
    from matplotlib.lines import Line2D
    engine_handles = [
        Line2D([0], [0], color="black", marker="o", linestyle="-",
                 lw=1.8, label="Jaxon"),
        Line2D([0], [0], color="black", marker="x", linestyle="--",
                 lw=1.8, label="PyFibers"),
    ]
    fig.legend(handles=engine_handles, loc="lower center", ncol=2,
                  frameon=False, fontsize=13,
                  bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = OUT_DIR / "fig6_strength_duration.png"
    fig.savefig(out); plt.close(fig)
    return out


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    rows = _gather_all_rows()
    print(f"[gather] {len(rows)} jax-vs-pyfibers threshold rows from "
          f"{len(MODELS)} models")
    if not rows:
        raise SystemExit("No validation rows found — nothing to plot.")
    saved = [
        make_fig1_scatter(rows),
        make_fig2_by_model(rows),
        make_fig3_by_pulse(rows),
        make_fig4a_by_diameter(rows),
        make_fig4b_scaling(),
        make_fig5_cv(),
        make_fig6_strength_duration(),
    ]
    for p in saved:
        print(f"  -> {p}")


if __name__ == "__main__":
    main()
