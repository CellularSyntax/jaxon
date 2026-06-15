"""Supplementary anatomy & FEM-field pages: one A4 page per cohort nerve.

Composites the per-nerve render components produced by the Golgi FE pipeline
(stored in duke_Ves/<nerve>/) into a single A4-portrait multi-panel page:

    a  uCT cross-section with segmentation overlay (+ scale bar)
    b  area-preserving deformed cross-section used for the 3-D model (+ bar)
    c  3-D nerve + 12-contact cuff rendering (+ scale bar)  | component legend
    d  per-contact extracellular potential V_e
    e  activating function (second spatial difference of V_e)
    f  extracellular E-field magnitude

Scale-bar PNGs that match their parent's dimensions are transparent overlays
and are alpha-composited onto the parent automatically.

Saved to manuscript/figures/supp/figA_<nerve>.png.

Run from project root:
    python -m experiments_v2.figures_supp_anatomy            # all cohort nerves
    python -m experiments_v2.figures_supp_anatomy sub-11_sam-3   # one nerve (preview)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from experiments_v2.figures_duke_main import PALETTE, FS, FS_SM, _load_all_seeds

ROOT     = Path(__file__).resolve().parent.parent
DUKE     = ROOT / "duke_Ves"
SUPP_DIR = ROOT / "manuscript" / "figures" / "supp"
SUPP_DIR.mkdir(parents=True, exist_ok=True)


def _species(name: str) -> str:
    return "human" if name.startswith("human") else "swine"


def _load(path: Path):
    return Image.open(path).convert("RGBA") if path.exists() else None


def _with_overlay(base_path: Path, overlay_path: Path):
    """Alpha-composite a same-size transparent scale-bar overlay onto base."""
    base = _load(base_path)
    if base is None:
        return None
    ov = _load(overlay_path)
    if ov is not None and ov.size == base.size:
        base = Image.alpha_composite(base, ov)
    return base


def _with_corner_bar(base_path: Path, bar_path: Path, margin_frac: float = 0.02):
    """Composite a small standalone scale bar at the bottom-right corner."""
    base = _load(base_path)
    if base is None:
        return None
    bar = _load(bar_path)
    if bar is not None:
        m = int(base.width * margin_frac)
        base.alpha_composite(bar, (base.width - bar.width - m,
                                   base.height - bar.height - m))
    return base


def _panel(ax, img, label: str | None, title: str | None, col):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if img is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center",
                transform=ax.transAxes, fontsize=FS_SM, color=PALETTE["lgrey"])
        return
    ax.imshow(np.asarray(img))
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=FS_SM, pad=3)
    if label:
        ax.text(-0.02, 1.02, label, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=FS + 1, fontweight="bold", color=col)


def make_anatomy_page(nerve: str) -> bool:
    d = DUKE / nerve
    if not (d / "render_components.png").exists():
        print(f"  [skip] {nerve}: no render_components.png", file=sys.stderr)
        return False
    sp = _species(nerve)
    col = PALETTE[sp]

    uct      = _with_overlay(d / "render_components_uct.png",
                             d / "render_components_uct_scalebar.png")
    deformed = _with_overlay(d / "render_components_deformed.png",
                             d / "render_components_deformed_scalebar.png")
    render3d = _with_corner_bar(d / "render_components.png",
                                d / "render_components_scalebar.png")
    legend   = _load(d / "render_components_legend.png")
    ve       = _load(d / "render_components_ve.png")
    af       = _load(d / "render_components_af.png")
    efield   = _load(d / "render_components_efield.png")

    fig = plt.figure(figsize=(13.33, 7.5))           # 16:9 landscape
    gs = gridspec.GridSpec(
        2, 3, figure=fig, left=0.03, right=0.985, top=0.91, bottom=0.04,
        wspace=0.10, hspace=0.18,
    )
    sample_id = nerve.replace("human_", "").replace("human", "")
    fig.suptitle(f"{sp.capitalize()} ({sample_id}) — anatomy and FEM fields",
                 fontsize=FS + 2, fontweight="bold", color=col, y=0.975)

    # row 1: anatomy -> 3-D model
    _panel(fig.add_subplot(gs[0, 0]), uct, "a",
           "µCT with segmentation", col)
    _panel(fig.add_subplot(gs[0, 1]), deformed, "b",
           "deformed cross-section (3-D model)", col)
    ax_c = fig.add_subplot(gs[0, 2])
    _panel(ax_c, render3d, "c", "3-D nerve + 12-contact cuff", col)
    if legend is not None:                            # scaled-down legend inset
        axl = ax_c.inset_axes([0.0, 0.0, 0.30, 0.30])
        axl.imshow(np.asarray(legend))
        axl.set_xticks([]); axl.set_yticks([])
        for s in axl.spines.values():
            s.set_visible(False)

    # row 2: the FEM fields, per contact
    _panel(fig.add_subplot(gs[1, 0]), ve, "d",
           "per-contact $V_e$", col)
    _panel(fig.add_subplot(gs[1, 1]), af, "e",
           "per-contact activating function", col)
    _panel(fig.add_subplot(gs[1, 2]), efield, "f",
           "per-contact extracellular E-field", col)

    out = SUPP_DIR / f"figA_{nerve}.png"
    fig.savefig(out, dpi=200)
    print(f"  -> {out}")
    plt.close(fig)
    return True


def main() -> int:
    if len(sys.argv) > 1:                              # one nerve (preview)
        ok = make_anatomy_page(sys.argv[1])
        return 0 if ok else 1
    rows = _load_all_seeds()
    nerves = sorted({r["sample"] for r in rows},
                    key=lambda n: (_species(n), n))
    print(f"[figures_supp_anatomy] {len(nerves)} cohort nerves")
    n_ok = sum(make_anatomy_page(nv) for nv in nerves)
    print(f"[figures_supp_anatomy] wrote {n_ok}/{len(nerves)} pages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
