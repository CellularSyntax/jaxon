"""Figure 2: Computational scaling — main manuscript figure.

Two panels, 183 mm wide, Nature Medicine 7 pt style:
  a  — wall-clock time vs number of axons (log-log): PyFibers dashed,
        JAXON GPU solid; 4 models
  b  — speedup factor (PyFibers / JAXON) vs number of axons (log-log)

Run from project root:
    python -m experiments_v2.figures_scaling_main
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "outputs" / "scaling" / "data_scaling.json"
OUT_DIR = ROOT / "manuscript" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "MRG":     "#0072B2",
    "Sundt":   "#D55E00",
    "Sweeney": "#009E73",
    "Rattay":  "#CC79A7",
    "grey":    "#3B3B3B",
}
MODELS = ["MRG", "Sundt", "Sweeney", "Rattay"]

FS    = 7
FS_SM = 6
FS_AX = 7

mpl.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size":         FS,
    "axes.labelsize":    FS_AX,
    "axes.titlesize":    FS,
    "xtick.labelsize":   FS,
    "ytick.labelsize":   FS,
    "legend.fontsize":   FS_SM,
    "axes.linewidth":    0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size":  2.5,
    "ytick.major.size":  2.5,
    "lines.linewidth":   0.9,
    "patch.linewidth":   0.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "pdf.fonttype":      42,
})


def main() -> int:
    if not DATA.exists():
        print(f"[figures_scaling_main] missing {DATA}", file=sys.stderr)
        return 1

    with open(DATA) as f:
        data = json.load(f)

    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(5.5, 2.8))

    for m_entry in data["models"]:
        model = m_entry["model"]
        if model not in MODELS:
            continue
        col = PALETTE[model]
        N = np.array([int(n) for n in m_entry["N"]])
        py = np.array([m_entry["pyfibers"][str(n)] for n in N])
        engine_key = "jaxley_gpu" if "jaxley_gpu" in m_entry else "jaxley_cpu"
        jax_run = np.array([m_entry[engine_key]["run"][str(n)] for n in N])

        ax_t.plot(N, py,      color=col, ls="--", marker="x",
                  markersize=3, lw=0.9)
        ax_t.plot(N, jax_run, color=col, ls="-",  marker="o",
                  markersize=3, lw=0.9)
        ax_s.plot(N, py / jax_run, color=col, marker="o",
                  markersize=3, label=model, lw=0.9)

    for ax in (ax_t, ax_s):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of axons")

    ax_t.set_ylabel("wall-clock time (s)")
    # Compact legend: line style only (colours encode model, shown in panel b)
    engine_handles = [
        mlines.Line2D([], [], color=PALETTE["grey"], ls="-",  marker="o",
                      markersize=3, lw=0.9, label="JAXON (GPU)"),
        mlines.Line2D([], [], color=PALETTE["grey"], ls="--", marker="x",
                      markersize=3, lw=0.9, label="PyFibers (CPU)"),
    ]
    ax_t.legend(handles=engine_handles, frameon=False, fontsize=FS_SM,
                loc="upper left", handlelength=1.4)
    ax_s.axhline(1.0, color=PALETTE["grey"], ls=":", lw=0.7)
    ax_s.set_ylabel("speedup  (PyFibers / JAXON)")
    ax_s.legend(frameon=False, loc="upper left")

    # Panel letters
    for letter, ax in [("a", ax_t), ("b", ax_s)]:
        ax.text(-0.18, 1.05, letter, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="bottom", ha="left")

    fig.tight_layout(w_pad=1.5)
    for ext in (".png", ".svg"):
        p = OUT_DIR / f"fig2_scaling{ext}"
        fig.savefig(p, dpi=300 if ext == ".png" else None)
        print(f"  -> {p}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
