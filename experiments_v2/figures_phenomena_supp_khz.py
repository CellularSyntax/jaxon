"""Supplementary figure: kHz frequency block — PyFibers vs JAXON, all 4 models.

Moved out of the main propagation-phenomena figure (fig2) because the strongly
driven kHz regime is intrinsically scheme-sensitive: JAX (backward-Euler) and
NEURON (Crank-Nicolson) agree on the block *outcome* (AP counts) but the
in-window forced sub-threshold oscillation is not reproducible pointwise (it does
not converge under dt refinement).  The main figure uses DC depolarization block
instead, which is deterministic and matches to the AP.

This supplement keeps the kHz comparison for completeness: 2x2 grid, one cell per
model, each cell = 4 stimulation amplitudes (control -> kHz-induced -> block).

Run from project root:
    python -m experiments_v2.figures_phenomena_supp_khz
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from experiments_v2.figures_phenomena_main import (
    ROOT, MODELS, _load, _panel_b, _heading, FS_SM,
)

OUT_DIR = ROOT / "manuscript" / "figures" / "supp"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    FIG_W, FIG_H = 12.0, 11.0
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # 2x2 grid of model cells; generous spacing for the per-cell legend/heading.
    outer = gridspec.GridSpec(2, 2, figure=fig,
                              left=0.07, right=0.97, bottom=0.05, top=0.92,
                              wspace=0.30, hspace=0.32)

    for idx, m in enumerate(MODELS):
        r, c = divmod(idx, 2)
        name = f"khz_block{m['suffix']}"
        khz = _load(ROOT / "outputs" / name / f"data_{name}.json")
        _panel_b(outer[r, c], fig, khz, v_rest=m["v_rest"])

        # Heading per cell (letter + model name).
        bbox = outer[r, c].get_position(fig)
        _heading(fig, bbox.x0 - 0.045, bbox.y1 + 0.012, "abcd"[idx],
                 f"{m['key'].upper()} — kHz frequency block")

    for ext in (".png", ".svg"):
        p = OUT_DIR / f"figS_khz_block{ext}"
        fig.savefig(p, dpi=600 if ext == ".png" else None)
        print(f"  -> {p}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
