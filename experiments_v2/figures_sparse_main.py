"""Figure 5: Sparse fibre-sampling analysis — main manuscript figure.

Layout (183 mm wide, Nature Medicine style):
  a  — schematic of centroid vs sparse vs dense fibre sampling strategies
  b  — connected strip plot: SI_sparse vs SI_dense per strategy (swine + human)
  c  — gap violin: SI_dense_opt − SI_sparse_reported per strategy
  d  — [PENDING §3.5] transfer evaluation placeholder

Run from project root:
    python -m experiments_v2.figures_sparse_main
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from experiments_v2.figures_sparse_sampling import (
    _load,
    _strip_panel,
    _gap_violin_panel,
    PALETTE,
    STRAT_LAYOUT,
    X_TICKS,
    X_LABELS,
)

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "manuscript" / "figures" / "main"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
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
    "legend.fontsize":   FS,
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
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "pdf.fonttype":      42,
})


# ── Panel a: sampling schematic ───────────────────────────────────────────────
def _draw_fascicle(ax, cx, cy, r, n_dots, dot_col, bg_col, rng,
                   label_top="", label_bot=""):
    """Draw a circular fascicle with n_dots randomly placed fibre dots."""
    circle = mpatches.Circle(
        (cx, cy), r, facecolor=bg_col, edgecolor="#888888",
        linewidth=0.6, zorder=2,
    )
    ax.add_patch(circle)

    if n_dots > 0:
        # Generate random positions inside circle
        angles  = rng.uniform(0, 2 * np.pi, n_dots * 6)
        radii   = np.sqrt(rng.uniform(0, 1, n_dots * 6)) * r * 0.88
        xs = cx + radii * np.cos(angles)
        ys = cy + radii * np.sin(angles)
        inside = (xs - cx) ** 2 + (ys - cy) ** 2 < (r * 0.88) ** 2
        xs, ys = xs[inside][:n_dots], ys[inside][:n_dots]
        ax.scatter(xs, ys, s=4, color=dot_col, edgecolors="none",
                   zorder=3, alpha=0.85)

    if label_top:
        ax.text(cx, cy + r + 0.07, label_top, ha="center", va="bottom",
                fontsize=FS_SM, color="#444444", weight="semibold")
    if label_bot:
        ax.text(cx, cy - r - 0.07, label_bot, ha="center", va="top",
                fontsize=FS_SM, color="#666666")


def _panel_a_schematic(ax: plt.Axes) -> None:
    ax.set_xlim(-0.1, 3.3)
    ax.set_ylim(-0.55, 0.80)
    ax.set_aspect("equal")
    ax.axis("off")

    rng = np.random.default_rng(42)
    FASC_R = 0.34
    BG     = "#f0f0f0"
    GREEN  = "#1A7340"
    GREY   = "#AAAAAA"
    RED    = "#888888"

    configs = [
        # cx, cy, n_dots_grey, n_dots_green, label_top, label_bot
        (0.42,  0.0,  40,  0, "dense", "(~1 000 fibres)"),
        (1.20,  0.0,   0,  1, "centroid", "(1/fascicle)"),
        (1.98,  0.0,   0,  3, "random-3", "(3/fascicle)"),
        (2.76,  0.0,   0, 10, "random-10", "(10/fascicle)"),
    ]

    for cx, cy, n_grey, n_green, lbl_top, lbl_bot in configs:
        _draw_fascicle(ax, cx, cy, FASC_R, n_grey, GREY, BG, rng,
                       label_top=lbl_top, label_bot=lbl_bot)
        if n_green > 0:
            # Place centroid first in centre, then random
            if n_green == 1:
                ax.scatter([cx], [cy], s=10, color=GREEN,
                           edgecolors="none", zorder=4)
            else:
                angles  = rng.uniform(0, 2 * np.pi, n_green * 5)
                radii   = np.sqrt(rng.uniform(0, 1, n_green * 5)) * FASC_R * 0.80
                xs = cx + radii * np.cos(angles)
                ys = cy + radii * np.sin(angles)
                inside = (xs - cx) ** 2 + (ys - cy) ** 2 < (FASC_R * 0.80) ** 2
                xs, ys = xs[inside][:n_green], ys[inside][:n_green]
                ax.scatter(xs, ys, s=10, color=GREEN,
                           edgecolors="none", zorder=4, alpha=0.90)

    # Down-arrow: "dense → sparse"
    ax.annotate("", xy=(1.10, 0.0), xytext=(0.76, 0.0),
                 arrowprops=dict(arrowstyle="-|>", color="#555555",
                                 lw=0.7, mutation_scale=8))

    # Legend
    leg_dots = [
        mpatches.Patch(facecolor=GREEN, edgecolor="none", label="sampled fibre"),
        mpatches.Patch(facecolor=GREY,  edgecolor="none", label="all fibres (dense)"),
    ]
    ax.legend(handles=leg_dots, loc="lower right", fontsize=FS_SM,
              frameon=False, handlelength=1.0, handletextpad=0.4,
              bbox_to_anchor=(1.0, -0.05))


# ── Panel d: transfer evaluation placeholder ──────────────────────────────────
def _panel_d_placeholder(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.08), 0.90, 0.84,
        boxstyle="round,pad=0.02",
        facecolor="#F7F7F7", edgecolor="#CCCCCC", linewidth=0.8,
        zorder=1,
    ))
    ax.text(0.50, 0.62,
            "Transfer evaluation",
            ha="center", va="center", fontsize=FS,
            color="#555555", weight="semibold", zorder=2)
    ax.text(0.50, 0.48,
            r"$\mathrm{SI}_\mathrm{sparse}$ $-$ $\mathrm{SI}_\mathrm{dense,transfer}$ (§3.5)",
            ha="center", va="center", fontsize=FS_SM,
            color="#888888", zorder=2)
    ax.text(0.50, 0.30,
            "Pending full-cohort results",
            ha="center", va="center", fontsize=FS_SM,
            style="italic", color="#AAAAAA", zorder=2)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    rows = _load()
    if not rows:
        print("[figures_sparse_main] no paired data found", file=sys.stderr)
        return 1
    n_samples = len({r["sample"] for r in rows})
    print(f"[figures_sparse_main] {len(rows)} seed×sample records, {n_samples} samples")

    fig = plt.figure(figsize=(7.2, 6.0))
    gs  = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.08, right=0.98, top=0.97, bottom=0.08,
        wspace=0.38, hspace=0.55,
    )
    ax_a = fig.add_subplot(gs[0, 0])   # schematic
    ax_b = fig.add_subplot(gs[0, 1])   # strip plot (2-panel built inside)
    ax_c = fig.add_subplot(gs[1, 0])   # gap violin (2-panel built inside)
    ax_d = fig.add_subplot(gs[1, 1])   # placeholder

    # Panels b and c use 2 sub-panels (swine + human) — build with inset gridspecs
    ax_b.set_visible(False)
    gs_b = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[0, 1], wspace=0.25
    )
    ax_b_sw = fig.add_subplot(gs_b[0])
    ax_b_hu = fig.add_subplot(gs_b[1])

    ax_c.set_visible(False)
    gs_c = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[1, 0], wspace=0.25
    )
    ax_c_sw = fig.add_subplot(gs_c[0])
    ax_c_hu = fig.add_subplot(gs_c[1])

    _panel_a_schematic(ax_a)
    _strip_panel(ax_b_sw, rows, "swine", PALETTE["swine"])
    _strip_panel(ax_b_hu, rows, "human", PALETTE["human"])
    ax_b_hu.set_ylabel("")

    _gap_violin_panel(ax_c_sw, rows, "swine", PALETTE["swine"])
    _gap_violin_panel(ax_c_hu, rows, "human", PALETTE["human"])
    ax_c_hu.set_ylabel("")

    _panel_d_placeholder(ax_d)

    # Panel letters — anchor to the first visible axis per panel
    for letter, ax in [
        ("a", ax_a), ("b", ax_b_sw), ("c", ax_c_sw), ("d", ax_d)
    ]:
        ax.text(-0.15, 1.06, letter, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="bottom", ha="left")

    # Shared strip legend
    med_line  = mlines.Line2D([], [], color="#555555", lw=1.8, label="median")
    band_p    = mpatches.Patch(facecolor="#AAAAAA", alpha=0.4,
                                label="±0.05 band")
    ax_b_sw.legend(handles=[med_line, band_p], loc="lower left",
                   frameon=False, fontsize=FS_SM)

    out_png = OUT_DIR / "fig5_sparse_main.png"
    out_svg = OUT_DIR / "fig5_sparse_main.svg"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"  -> {out_png}")
    print(f"  -> {out_svg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
