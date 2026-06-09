"""Build the six jaxon-vs-pyfibers validation figures.

Inputs (already on disk from earlier benchmark runs):
  outputs/{mrg,sundt,sweeney,rattay}_validation/data_<m>_sd.json
      Nested {diameter_um -> pulse_shape -> PW_ms -> {jax, pyfibers, err_pct}}
      threshold_mA values (signed; cathodic monophasic is negative, etc.).
  outputs/{mrg,sundt,sweeney,rattay}_validation/data_<m>_cv.json
      Per-diameter conduction velocity for jax and pyfibers (m/s).
  outputs/scaling/data_scaling.json
      Per-model wall-clock dict with pyfibers and jaxley_{cpu,gpu} keys
      (each split into compile vs run, indexed by n_axons).

Outputs (written to manuscript/figures/validation/):
  fig1_scatter_thresholds.png       all-model, all-PW, all-pulse scatter
  fig2_violin_by_model.png          |err_pct| violins per model
  fig3_violin_by_pulse.png          |err_pct| violins per pulse shape
  fig4a_violin_by_diameter.png      |err_pct| violins per diameter bin
  fig4b_scaling_walltime.png        wall-clock vs n_axons + speedup
  fig5_conduction_velocity.png      CV(D) per model, jax vs pyfibers
  fig6_strength_duration.png        SD curves per model, one diam per line
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Configuration ──────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR   = REPO_ROOT / "manuscript" / "figures" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["MRG", "Sundt", "Sweeney", "Rattay"]
MODEL_COLOURS = {
    "MRG":     "tab:blue",
    "Sundt":   "tab:orange",
    "Sweeney": "tab:green",
    "Rattay":  "tab:red",
}

# Human-readable pulse-shape labels matching the JSON keys.
PULSE_ORDER = ["mono_c", "mono_a", "bi_ca", "bi_ac",
               "sine", "exp", "gaussian", "sawtooth"]
PULSE_PRETTY = {
    "mono_c":   "mono (-)",
    "mono_a":   "mono (+)",
    "bi_ca":    "biphasic (-)",
    "bi_ac":    "biphasic (+)",
    "sine":     "sine",
    "exp":      "exponential",
    "gaussian": "gaussian",
    "sawtooth": "triangle",
}

# Diameter bin edges (µm) for plot (4a).  Lump the very thick MRG range
# together with the smaller A-fibres separately from C-fibres.
DIAM_BINS = [(0.0, 1.5), (1.5, 5.0), (5.0, 8.0),
             (8.0, 12.0), (12.0, 17.0)]
DIAM_BIN_LABELS = [
    r"$<\!1.5\,\mu$m",
    r"$1.5\!-\!5\,\mu$m",
    r"$5\!-\!8\,\mu$m",
    r"$8\!-\!12\,\mu$m",
    r"$12\!-\!16\,\mu$m",
]


# ── IO helpers ────────────────────────────────────────────────────────────────

def _load_sd(model: str) -> dict:
    """Return the SD JSON for ``model`` (case-insensitive lookup)."""
    p = REPO_ROOT / "outputs" / f"{model.lower()}_validation" / f"data_{model.lower()}_sd.json"
    with open(p) as f:
        return json.load(f)


def _load_cv(model: str) -> dict:
    p = REPO_ROOT / "outputs" / f"{model.lower()}_validation" / f"data_{model.lower()}_cv.json"
    with open(p) as f:
        return json.load(f)


def _flatten_sd(sd: dict, model: str) -> list[dict]:
    """Flatten an SD dict into a per-condition row list.

    Each row: {model, diam_um, pulse, PW_ms, jax_mA, pyfibers_mA, err_pct,
                err_abs_pct}.
    """
    rows: list[dict] = []
    for diam_str, by_pulse in sd.items():
        diam = float(diam_str)
        for pulse, by_pw in by_pulse.items():
            for pw_str, vals in by_pw.items():
                if not isinstance(vals, dict) or "jax" not in vals:
                    continue
                err = float(vals.get("err_pct", float("nan")))
                rows.append(dict(
                    model=model,
                    diam_um=diam,
                    pulse=pulse,
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


# ── Plot 1: scatter of all thresholds ────────────────────────────────────────

def make_fig1_scatter(rows: list[dict]) -> Path:
    """All thresholds are plotted as ABSOLUTE values (|mA|) so cathodic
    and anodic responses live in the same positive quadrant -- the
    validation question is whether the magnitudes agree, not the
    polarity (the sign is set deterministically by the pulse-shape key
    in the JSON and is identical between jaxon and pyfibers by
    construction).
    """
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
        ax.scatter(x, y, s=12, alpha=0.6, color=MODEL_COLOURS[m],
                    label=f"{m}  (n={len(x)})", edgecolors="none")
        all_jax.append(y)
        all_py.append(x)

    x_all = np.concatenate(all_py)
    y_all = np.concatenate(all_jax)
    lo = min(x_all.min(), y_all.min())
    hi = max(x_all.max(), y_all.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="y = x")

    ss_res = float(np.sum((y_all - x_all) ** 2))
    ss_tot = float(np.sum((y_all - y_all.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    mape = float(np.median(np.abs((y_all - x_all) / x_all))) * 100.0

    ax.set_xlabel("pyFibers / NEURON threshold $|$mA$|$")
    ax.set_ylabel("jaxon threshold $|$mA$|$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"Activation thresholds (n = {len(x_all)} finite pairs)\n"
                  f"$R^2$ = {r2:.5f},  median $|{{\\rm err}}|$ = {mape:.2f}%")
    ax.legend(loc="upper left", frameon=False)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    out = OUT_DIR / "fig1_scatter_thresholds.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ── Plot 2: violin |err_pct| per model ───────────────────────────────────────

def _boxplot_with_scatter(ax, groups: list[np.ndarray], labels: list[str],
                            colours: list[str] | None = None,
                            rng_seed: int = 0) -> None:
    """Boxplot with median + IQR, individual points jittered over the top.
    Reads better than a violin on log-scaled error data because the box
    bounds are exact percentiles and the scatter shows the distribution
    shape directly."""
    if colours is None:
        colours = ["lightsteelblue"] * len(groups)
    bp = ax.boxplot(
        groups, widths=0.55, showfliers=False, patch_artist=True,
        boxprops=dict(linewidth=1.0),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
    )
    for patch, col in zip(bp["boxes"], colours):
        patch.set_facecolor(col); patch.set_alpha(0.45)
        patch.set_edgecolor("black")
    # Jittered scatter on top.
    rng = np.random.default_rng(rng_seed)
    for i, (arr, col) in enumerate(zip(groups, colours)):
        if arr.size == 0:
            continue
        jitter = rng.uniform(-0.18, 0.18, size=arr.size)
        ax.scatter(np.full(arr.size, i + 1) + jitter, arr,
                    s=9, alpha=0.45, color=col, edgecolors="none",
                    zorder=3)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")


def make_fig2_by_model(rows: list[dict]) -> Path:
    groups, labels, cols = [], [], []
    for m in MODELS:
        arr = np.array([r["err_abs_pct"] for r in rows
                          if r["model"] == m and np.isfinite(r["err_abs_pct"])])
        if arr.size == 0:
            continue
        groups.append(arr); labels.append(f"{m}\n(n={arr.size})")
        cols.append(MODEL_COLOURS[m])
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    _boxplot_with_scatter(ax, groups, labels, cols)
    ax.set_ylabel(r"$|\mathrm{err}|$  (%)")
    ax.set_ylim(bottom=1e-4)
    ax.set_title("Threshold |err| by model")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    out = OUT_DIR / "fig2_violin_by_model.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ── Plot 3: violin |err_pct| per pulse type ──────────────────────────────────

def make_fig3_by_pulse(rows: list[dict]) -> Path:
    groups, labels = [], []
    for p in PULSE_ORDER:
        arr = np.array([r["err_abs_pct"] for r in rows
                          if r["pulse"] == p and np.isfinite(r["err_abs_pct"])])
        if arr.size == 0:
            continue
        groups.append(arr)
        labels.append(f"{PULSE_PRETTY.get(p, p)}\n(n={arr.size})")
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    _boxplot_with_scatter(ax, groups, labels)
    ax.set_ylabel(r"$|\mathrm{err}|$  (%)")
    ax.set_ylim(bottom=1e-4)
    ax.set_title("Threshold |err| by pulse shape (all models pooled)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    out = OUT_DIR / "fig3_violin_by_pulse.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ── Plot 4a: violin |err_pct| per diameter bin ───────────────────────────────

def make_fig4a_by_diameter(rows: list[dict]) -> Path:
    groups, labels = [], []
    for (lo, hi), lab in zip(DIAM_BINS, DIAM_BIN_LABELS):
        arr = np.array([r["err_abs_pct"] for r in rows
                          if lo <= r["diam_um"] < hi
                          and np.isfinite(r["err_abs_pct"])])
        if arr.size == 0:
            continue
        groups.append(arr)
        labels.append(f"{lab}\n(n={arr.size})")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    _boxplot_with_scatter(ax, groups, labels)
    ax.set_ylabel(r"$|\mathrm{err}|$  (%)")
    ax.set_ylim(bottom=1e-4)
    ax.set_title("Threshold |err| by fibre diameter (all models pooled)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    out = OUT_DIR / "fig4a_violin_by_diameter.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ── Plot 4b: wall-clock + speedup ────────────────────────────────────────────

def make_fig4b_scaling() -> Path:
    p = REPO_ROOT / "outputs" / "scaling" / "data_scaling.json"
    with open(p) as f:
        data = json.load(f)
    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(11, 4.2))

    for m_entry in data["models"]:
        model = m_entry["model"]
        if model not in MODELS:
            continue
        col = MODEL_COLOURS[model]
        N = np.array([int(n) for n in m_entry["N"]])
        py = np.array([m_entry["pyfibers"][str(n)] for n in N])
        # Prefer GPU run-time when available, else CPU.
        engine_key = "jaxley_gpu" if "jaxley_gpu" in m_entry else "jaxley_cpu"
        jax_run = np.array([m_entry[engine_key]["run"][str(n)] for n in N])

        ax_t.plot(N, py,       color=col, ls="--", marker="x",
                    label=f"pyFibers ({model})")
        ax_t.plot(N, jax_run,  color=col, ls="-",  marker="o",
                    label=f"jaxon ({model})")

        ax_s.plot(N, py / jax_run, color=col, marker="o", label=model)

    for ax in (ax_t, ax_s):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlabel("number of axons")
    ax_t.set_ylabel("wall-clock (s, log)")
    ax_t.set_title("Forward-pass wall-clock")
    ax_t.legend(fontsize=7, ncol=2, frameon=False)

    ax_s.axhline(1.0, color="k", ls=":", lw=1)
    ax_s.set_ylabel("speedup  (pyfibers / jaxon)")
    ax_s.set_title("Speedup")
    ax_s.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    out = OUT_DIR / "fig4b_scaling_walltime.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ── Plot 5: conduction velocity per model ───────────────────────────────────

def make_fig5_cv() -> Path:
    """Two panels side-by-side: A-fibres (myelinated, ~5-16 m/s per um)
    and C-fibres (unmyelinated, <1 m/s).  Plotting them together
    visually crushes the C-fibre data into a flat line near zero, so
    split by typical CV magnitude."""
    a_fibre_models = ["MRG", "Sweeney"]
    c_fibre_models = ["Sundt", "Rattay"]

    fig, (ax_a, ax_c) = plt.subplots(1, 2, figsize=(11, 4.2))

    def _plot(ax, models, title):
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
            col = MODEL_COLOURS[m]
            ax.plot(diams, py_v, ls="--", marker="x", color=col,
                      label=f"pyFibers ({m})")
            ax.plot(diams, jax_v, ls="-",  marker="o", color=col,
                      label=f"jaxon ({m})")
            any_plotted = True
        ax.set_xlabel(r"fibre diameter ($\mu$m)")
        ax.set_ylabel("conduction velocity (m/s)")
        ax.set_title(title)
        if any_plotted:
            ax.legend(fontsize=8, frameon=False)
        ax.grid(True, alpha=0.3)

    _plot(ax_a, a_fibre_models, "A-fibres (myelinated)")
    _plot(ax_c, c_fibre_models, "C-fibres (unmyelinated)")
    fig.tight_layout()
    out = OUT_DIR / "fig5_conduction_velocity.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ── Plot 6: strength–duration per model ──────────────────────────────────────

def make_fig6_strength_duration() -> Path:
    """One panel per model.  For each diameter we plot |threshold| vs PW
    using the cathodic-monophasic (``mono_c``) values where available, else
    the first available pulse shape (some C-fibre models only have monophasic
    in their SD sweep).  Lines are coloured by diameter."""
    n_models_with_data = 0
    for m in MODELS:
        try:
            _load_sd(m)
            n_models_with_data += 1
        except FileNotFoundError:
            pass
    ncols = 2
    nrows = math.ceil(n_models_with_data / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.0 * nrows),
                                squeeze=False)

    ax_idx = 0
    for m in MODELS:
        try:
            sd = _load_sd(m)
        except FileNotFoundError:
            continue
        ax = axes[ax_idx // ncols][ax_idx % ncols]; ax_idx += 1
        diams = sorted(float(d) for d in sd.keys())
        cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(diams)))
        for D, col in zip(diams, cmap):
            entry = sd[str(D) if str(D) in sd else f"{D:.1f}"]
            # Pick mono_c if available, else first pulse with data.
            pulse_key = "mono_c" if "mono_c" in entry else next(iter(entry.keys()))
            by_pw = entry[pulse_key]
            pws = sorted(float(pw) for pw in by_pw.keys())
            thr_j  = np.array([abs(by_pw[str(pw) if str(pw) in by_pw
                                            else f"{pw:.2f}"]["jax"])
                                  for pw in pws])
            thr_py = np.array([abs(by_pw[str(pw) if str(pw) in by_pw
                                            else f"{pw:.2f}"]["pyfibers"])
                                  for pw in pws])
            ax.plot(pws, thr_py, ls="--", marker="x", color=col, alpha=0.7)
            ax.plot(pws, thr_j,  ls="-",  marker="o", color=col,
                      label=fr"$D={D}\,\mu$m")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("pulse width (ms)")
        ax.set_ylabel("threshold |mA|")
        ax.set_title(f"{m}  ({pulse_key})")
        ax.legend(fontsize=7, frameon=False, ncol=2)
        ax.grid(True, which="both", alpha=0.3)

    # Hide any unused axes.
    for j in range(ax_idx, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Strength–duration curves (solid = jaxon, dashed = pyFibers)",
                   y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "fig6_strength_duration.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

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
