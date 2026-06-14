"""Supplementary figures: one figure per subject, all seeds side by side.

Each figure shows all optimisation seeds for one specimen, ordered by SI
descending. Each seed column shows BOTH the dense-optimised activation map and
the sparse-optimised (transfer) map evaluated on the full nerve, plus their
optimised cuff patterns. Style matches fig3a. Saved to manuscript/figures/supp/.

Run from project root:
    python -m experiments_v2.figures_supp_activation
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from experiments_v2.figures_duke import (
    _draw_xsection,
    _draw_cuff_grid,
    _load_geometry,
)
from experiments_v2.figures_duke_main import (
    PALETTE, FS, FS_SM,
    _add_scale_bar, _draw_compass,
    _load_all_seeds, _dense_transfer_example,
)

ROOT     = Path(__file__).resolve().parent.parent
SUPP_DIR = ROOT / "manuscript" / "figures" / "supp"
SUPP_DIR.mkdir(parents=True, exist_ok=True)

MAX_COLS   = 5
# each seed column: dense xsec | dense cuff | sparse xsec | sparse cuff
H_RATIOS   = [1.8, 1.0, 1.8, 1.0]


# ── Data ──────────────────────────────────────────────────────────────────────
def _seeds_by_sample(rows: list[dict], species: str
                     ) -> dict[str, list[tuple[int, float, dict]]]:
    """Group seeds by sample; each list sorted by SI descending."""
    by_sample: dict[str, list] = {}
    for r in rows:
        if r["species"] != species or not np.isfinite(r["si"]):
            continue
        try:
            seed_num = int(Path(r["raw_path"]).stem.split("_")[-1])
        except ValueError:
            seed_num = 0
        raw = json.loads(Path(r["raw_path"]).read_text())
        by_sample.setdefault(r["sample"], []).append((seed_num, r["si"], raw))

    for sample in by_sample:
        by_sample[sample].sort(key=lambda x: -x[1])

    # Sort samples by their best SI descending
    return dict(sorted(by_sample.items(),
                        key=lambda kv: -kv[1][0][1]))


# ── Drawing ───────────────────────────────────────────────────────────────────
def _draw_one(ax_xsec: plt.Axes, ax_cuff: plt.Axes,
              sample: str, raw: dict | None, species: str,
              top_label: str, top_color, si: float,
              is_first: bool = False) -> None:
    """Draw a single xsec activation map + its cuff grid from `raw`."""
    if raw is None:
        ax_xsec.axis("off")
        ax_xsec.text(0.5, 0.5, "no sparse\nresult", ha="center", va="center",
                     transform=ax_xsec.transAxes, fontsize=FS_SM,
                     color=PALETTE["lgrey"])
        ax_cuff.axis("off")
        return
    try:
        _draw_xsection(ax_xsec, sample, raw, draw_electrodes=True, draw_label=False)
        _add_scale_bar(ax_xsec)
    except Exception:
        ax_xsec.text(0.5, 0.5, "geometry\nmissing", ha="center", va="center",
                     transform=ax_xsec.transAxes, fontsize=FS_SM,
                     color=PALETTE["lgrey"])

    ax_xsec.text(0.50, 1.03, top_label, transform=ax_xsec.transAxes,
                 ha="center", va="bottom", fontsize=FS_SM,
                 color=top_color, weight="bold", clip_on=False)
    ax_xsec.text(0.97, 0.97, f"SI = {si:.3f}" if np.isfinite(si) else "SI = n/a",
                 transform=ax_xsec.transAxes, ha="right", va="top",
                 fontsize=FS_SM, color="black", clip_on=False)
    if is_first:
        _draw_compass(ax_xsec)

    try:
        _, _, contact_xyz, _ = _load_geometry(sample)
        amps = np.asarray(raw["rect"]["amps_mA"], dtype=float)
        _draw_cuff_grid(ax_cuff, contact_xyz, amps)
    except Exception:
        ax_cuff.axis("off")
        ax_cuff.text(0.5, 0.5, "cuff missing", ha="center", va="center",
                     transform=ax_cuff.transAxes, fontsize=FS_SM,
                     color=PALETTE["lgrey"])


def _draw_seed(axes4: list[plt.Axes],
               sample: str, seed_num: int, si: float, raw: dict,
               species: str, is_first: bool = False) -> None:
    """Draw dense (rows 0-1) and sparse-transfer (rows 2-3) for one seed."""
    col = PALETTE[species]
    ax_dx, ax_dc, ax_sx, ax_sc = axes4

    _draw_one(ax_dx, ax_dc, sample, raw, species,
              f"seed {seed_num} — dense", col, si, is_first=is_first)

    _, trans = _dense_transfer_example(sample, seed_num)
    si_tr = float((trans or {}).get("rect", {}).get("achievable_si", float("nan"))) \
        if trans is not None else float("nan")
    _draw_one(ax_sx, ax_sc, sample, trans, species,
              "sparse (transfer)", PALETTE["dgrey"], si_tr)


# ── Per-subject figure ────────────────────────────────────────────────────────
def _make_subject_fig(sample: str,
                      seeds: list[tuple[int, float, dict]],
                      species: str) -> None:
    n      = len(seeds)
    n_cols = min(MAX_COLS, n)
    n_rows = (n + n_cols - 1) // n_cols

    cell_w = 2.0
    row_h  = 6.8   # dense xsec + cuff + sparse xsec + cuff + spacing
    fig_w  = n_cols * cell_w + 0.4
    fig_h  = n_rows * row_h  + 0.7

    fig = plt.figure(figsize=(fig_w, fig_h))

    outer = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        left=0.03, right=0.97,
        top=0.90, bottom=0.03,
        wspace=0.12, hspace=0.30,
    )

    for idx, (seed_num, si, raw) in enumerate(seeds):
        r, c = divmod(idx, n_cols)
        inner = gridspec.GridSpecFromSubplotSpec(
            4, 1, subplot_spec=outer[r, c],
            height_ratios=H_RATIOS, hspace=0.30,
        )
        axes4 = [fig.add_subplot(inner[k]) for k in range(4)]
        _draw_seed(axes4, sample, seed_num, si, raw, species,
                   is_first=(idx == 0))

    # Hide unused cells
    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        fig.add_subplot(outer[r, c]).axis("off")

    # Strip species prefix for title
    sample_id = sample
    for pfx in (species, "swine", "human"):
        if sample_id.lower().startswith(pfx):
            sample_id = sample_id[len(pfx):].lstrip("_- ")
            break

    best_si = seeds[0][1]
    fig.suptitle(
        f"{species.capitalize()} ({sample_id})   "
        f"n = {n} seeds  |  best SI = {best_si:.3f}   "
        f"(dense vs sparse-transfer, 1 fibre/fascicle)",
        fontsize=FS + 1, fontweight="bold", y=0.985,
    )

    # Safe filename — replace special chars
    safe_name = sample.replace("/", "_").replace(" ", "_")
    for ext in (".png", ".svg"):
        p = SUPP_DIR / f"figS_{safe_name}{ext}"
        fig.savefig(p, dpi=300 if ext == ".png" else None)
        print(f"  -> {p}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    rows = _load_all_seeds()
    if not rows:
        print("[figures_supp_activation] no seed data found", file=sys.stderr)
        return 1

    for species in ("swine", "human"):
        by_sample = _seeds_by_sample(rows, species)
        print(f"[figures_supp_activation] {species}: "
              f"{len(by_sample)} subjects, "
              f"{sum(len(v) for v in by_sample.values())} seeds total")
        for sample, seeds in by_sample.items():
            _make_subject_fig(sample, seeds, species)

    return 0


if __name__ == "__main__":
    sys.exit(main())
