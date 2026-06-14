"""Supplementary figure: AP collision across diameters — PyFibers vs JAXON.

The main figure (fig2) shows AP collision as a single-diameter node-vs-time
waterfall for visual consistency with the propagation and DC-block panels.  This
supplement keeps the multi-diameter validation: for each model, Vm vs node # at 5
time points × 4 fibre diameters (the original snapshot grid), demonstrating
collision/annihilation reproduces across fibre sizes.

2x2 grid, one cell per model.

Run from project root:
    python -m experiments_v2.figures_phenomena_supp_collision
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from experiments_v2.figures_phenomena_main import (
    ROOT, MODELS, _load, _panel_c, _heading,
)

OUT_DIR = ROOT / "manuscript" / "figures" / "supp"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    FIG_W, FIG_H = 14.0, 12.0
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    outer = gridspec.GridSpec(2, 2, figure=fig,
                              left=0.06, right=0.98, bottom=0.05, top=0.92,
                              wspace=0.22, hspace=0.30)

    for idx, m in enumerate(MODELS):
        r, c = divmod(idx, 2)
        name = f"ap_collision{m['suffix']}"
        coll = _load(ROOT / "outputs" / name / f"data_{name}.json")
        _panel_c(outer[r, c], fig, coll, v_rest=m["v_rest"])

        bbox = outer[r, c].get_position(fig)
        _heading(fig, bbox.x0 - 0.045, bbox.y1 + 0.012, "abcd"[idx],
                 f"{m['key']} — AP collision (by diameter)")

    for ext in (".png", ".svg"):
        p = OUT_DIR / f"figS_collision{ext}"
        fig.savefig(p, dpi=600 if ext == ".png" else None)
        print(f"  -> {p}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
