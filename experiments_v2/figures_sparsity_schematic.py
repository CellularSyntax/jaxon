"""Schematic illustration of axon sampling sparsity levels within a single fascicle.

5 panels (dense | centroid | rand-1 | rand-3 | rand-10), each showing a circular
fascicle outline with ~50 fibre dots; selected fibres are highlighted.

Run from project root:
    python -m experiments_v2.figures_sparsity_schematic
"""
from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "manuscript" / "figures" / "main"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS    = 7
FS_SM = 6

mpl.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size":         FS,
    "axes.labelsize":    FS,
    "axes.titlesize":    FS,
    "xtick.labelsize":   FS_SM,
    "ytick.labelsize":   FS_SM,
    "axes.linewidth":    0.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.spines.bottom": False,
    "axes.grid":         False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "pdf.fonttype":      42,
})

# ── Colours ───────────────────────────────────────────────────────────────────
COL_FASC_FILL   = "#F5F3F7"   # warm off-white with slight violet tint
COL_FASC_EDGE   = "#3D3550"   # deep slate-violet outline
COL_FIBER_BG    = "#C4BECC"   # muted lavender-grey (unselected)
COL_FIBER_SEL   = "#6B4FAB"   # muted violet
COL_CENTROID    = "#C0392B"   # deep red centroid cross

# ── Generate schematic fibre positions inside unit circle ─────────────────────
def _fibers_in_circle(n: int = 30, seed: int = 3) -> np.ndarray:
    """Return (n, 2) array of positions inside the unit circle."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        xy = rng.uniform(-1, 1, (n * 4, 2))
        xy = xy[np.hypot(xy[:, 0], xy[:, 1]) < 0.88]
        pts.extend(xy.tolist())
    return np.array(pts[:n])


def _centroid_idx(xy: np.ndarray) -> int:
    """Index of fibre closest to centroid (origin for unit circle)."""
    return int(np.argmin(np.hypot(xy[:, 0], xy[:, 1])))


def _random_idx(xy: np.ndarray, n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(len(xy), size=min(n, len(xy)), replace=False)


# ── Strategy definitions ───────────────────────────────────────────────────────
def _get_strategies(xy: np.ndarray) -> list[tuple[str, str, np.ndarray]]:
    """Return list of (key, label, selected_indices)."""
    all_idx = np.arange(len(xy))
    return [
        ("dense",    f"dense\n(all)",  all_idx),
        ("centroid", "centroid\n(1 / fasc)",         np.array([_centroid_idx(xy)])),
        ("rand1",    "random\n(1 / fasc)",           _random_idx(xy, 1,  seed=7)),
        ("rand3",    "random\n(3 / fasc)",           _random_idx(xy, 3,  seed=7)),
        ("rand10",   "random\n(10 / fasc)",          _random_idx(xy, 10, seed=7)),
    ]


# ── Draw one panel ─────────────────────────────────────────────────────────────
def _draw_panel(ax: plt.Axes, xy: np.ndarray, sel_idx: np.ndarray,
                key: str, label: str) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.axis("off")

    # fascicle circle
    circ = mpatches.Circle((0, 0), 1.0, facecolor=COL_FASC_FILL,
                            edgecolor=COL_FASC_EDGE, linewidth=1.2)
    ax.add_patch(circ)

    # background (unselected) fibres
    mask = np.ones(len(xy), dtype=bool)
    mask[sel_idx] = False
    ax.scatter(xy[mask, 0], xy[mask, 1],
               s=7, color=COL_FIBER_BG, edgecolors="none", zorder=2)

    # selected fibres
    ax.scatter(xy[sel_idx, 0], xy[sel_idx, 1],
               s=38, color=COL_FIBER_SEL, edgecolors="white",
               linewidths=0.5, zorder=4)

    # centroid marker (cross) for centroid strategy
    if key == "centroid":
        ax.plot(0, 0, "+", color=COL_CENTROID, ms=7, mew=1.2, zorder=5)



# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    xy = _fibers_in_circle(n=52, seed=3)
    strategies = _get_strategies(xy)

    n = len(strategies)
    fig, axes = plt.subplots(1, n, figsize=(1.05 * n, 1.5))

    for ax, (key, label, sel_idx) in zip(axes, strategies):
        _draw_panel(ax, xy, sel_idx, key, label)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.12, wspace=0.08)

    for ext in (".png", ".svg"):
        path = OUT_DIR / f"fig_sparsity_schematic{ext}"
        fig.savefig(path, dpi=300 if ext == ".png" else None, bbox_inches="tight")
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
