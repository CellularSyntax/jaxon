"""Sparse fiber sampling figures for §3.4.

Two figures:
  fig_sparse_si_strips.png    — per-seed connected strip plot: SI for each
                                sampling strategy, swine and human panels.
  fig_sparse_gap.png          — gap distribution: SI_dense_opt - SI_sparse
                                per strategy (violin + scatter).

Run from project root:
    python -m experiments_v2.figures_sparse_sampling
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

ROOT  = Path(__file__).resolve().parent.parent
SWEEP = ROOT / "outputs" / "duke_sweeps"
OUT_DIR = ROOT / "manuscript" / "figures" / "duke" / "sparse_sampling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
PALETTE = {
    "swine":   "#D6604D",
    "human":   "#4393C3",
    "dense":   "#333333",
    "lgrey":   "#AAAAAA",
    "dgrey":   "#555555",
}
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

# ── Strategy layout ──────────────────────────────────────────────────────────
# (x_index, strategy_key, n_per_fasc, x_label)
STRAT_LAYOUT = [
    (0, "dense",    None, "dense\n(~1000)"),
    (1, "centroid",    1, "centroid\n(1/fasc)"),
    (2, "random",      1, "rand\n(1/fasc)"),
    (3, "random",      3, "rand\n(3/fasc)"),
    (4, "random",     10, "rand\n(10/fasc)"),
]
X_TICKS  = [t[0] for t in STRAT_LAYOUT]
X_LABELS = [t[3] for t in STRAT_LAYOUT]


# ── Data loader ──────────────────────────────────────────────────────────────
def _species_of(sample: str) -> str:
    return "human" if sample.startswith("human") else "swine"


def _load() -> list[dict]:
    rows = []
    for sample_dir in sorted(SWEEP.iterdir()):
        if not sample_dir.is_dir():
            continue
        sample = sample_dir.name
        for sparse_path in sorted(sample_dir.glob("sparse_sampling_seed_*.json")):
            seed = int(sparse_path.stem.split("_")[-1])
            dense_path = sample_dir / f"data_seed_{seed:04d}.json"
            if not dense_path.exists():
                continue
            try:
                dense_d  = json.loads(dense_path.read_text())
                sparse_d = json.loads(sparse_path.read_text())
            except Exception:
                continue
            _nan = float("nan")
            rect    = dense_d.get("rect") or {}
            dense_si = float(rect.get("achievable_si",
                                      abs(rect.get("final_si", _nan))))
            dense_firing = rect.get("firing") or {}

            # Per-strategy maps keyed by (strategy, n_per_fascicle):
            #   si           = in-sample SI on the sparse model
            #   si_transfer  = sparse-optimised amps re-evaluated on the FULL nerve
            #   transfer.firing = on-/off-target recruitment on the FULL nerve
            si_map: dict[tuple, float] = {}
            si_tr_map: dict[tuple, float] = {}
            fir_tr_map: dict[tuple, dict] = {}
            for r in sparse_d.get("results", []):
                if r.get("skipped"):
                    continue
                key = (r["strategy"], int(r["n_per_fascicle"]))
                si_map[key] = float(r["si"]) if r.get("si") is not None else _nan
                st = r.get("si_transfer")
                si_tr_map[key] = float(st) if st is not None else _nan
                fir_tr_map[key] = (r.get("transfer") or {}).get("firing")

            si_by_xi: dict[int, float] = {0: dense_si}
            si_tr_by_xi: dict[int, float] = {0: dense_si}    # full->full is its own transfer
            fir_tr_by_xi: dict[int, dict] = {0: dense_firing}
            for xi, strat_key, n_pf, _ in STRAT_LAYOUT[1:]:
                si_by_xi[xi]    = si_map.get((strat_key, n_pf), _nan)
                si_tr_by_xi[xi] = si_tr_map.get((strat_key, n_pf), _nan)
                fir_tr_by_xi[xi] = fir_tr_map.get((strat_key, n_pf))

            rows.append(dict(
                sample=sample,
                species=_species_of(sample),
                seed=seed,
                si=si_by_xi,
                si_transfer=si_tr_by_xi,
                dense_si=dense_si,
                dense_firing=dense_firing,
                transfer_firing=fir_tr_by_xi,
                n_fasc=int(sparse_d.get("dense_n_fibers", 0)),  # fiber count proxy
            ))
    return rows


# ── Figure 1: connected strip plot ───────────────────────────────────────────
def _strip_panel(ax, rows: list[dict], species: str, color: str) -> None:
    sp_rows = [r for r in rows if r["species"] == species]
    if not sp_rows:
        ax.set_visible(False)
        return

    rng = np.random.default_rng(42)

    for r in sp_rows:
        xi_list = sorted(r["si"].keys())
        y_list  = [r["si"][xi] for xi in xi_list]
        valid   = [np.isfinite(y) for y in y_list]
        # Draw connecting lines between valid adjacent points
        xs_v = [xi_list[i] for i in range(len(xi_list)) if valid[i]]
        ys_v = [y_list[i]  for i in range(len(xi_list)) if valid[i]]
        ax.plot(xs_v, ys_v, color=color, alpha=0.18, lw=0.6, zorder=1)

    # Scatter with small jitter
    for xi, _, _, _ in STRAT_LAYOUT:
        ys = [r["si"][xi] for r in sp_rows if np.isfinite(r["si"].get(xi, float("nan")))]
        if not ys:
            continue
        jitter = rng.uniform(-0.12, 0.12, len(ys))
        ax.scatter(xi + jitter, ys, s=10, color=color, alpha=0.55,
                   edgecolors="none", zorder=2)

    # Median marker per strategy
    for xi, _, _, _ in STRAT_LAYOUT:
        ys = [r["si"][xi] for r in sp_rows if np.isfinite(r["si"].get(xi, float("nan")))]
        if not ys:
            continue
        med = float(np.median(ys))
        ax.plot([xi - 0.3, xi + 0.3], [med, med],
                color=color, lw=1.8, solid_capstyle="round", zorder=3)

    # Reference band: ±0.05 around median dense SI
    dense_vals = [r["dense_si"] for r in sp_rows if np.isfinite(r["dense_si"])]
    if dense_vals:
        med_dense = float(np.median(dense_vals))
        ax.axhspan(med_dense - 0.05, med_dense + 0.05,
                   color=PALETTE["lgrey"], alpha=0.15, linewidth=0, zorder=0)

    ax.axhline(0.95, color=PALETTE["lgrey"], ls="--", lw=0.6, zorder=0)
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels(X_LABELS, fontsize=FS_SM)
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.6, 4.6)
    ax.set_ylabel("selectivity index (SI)")
    n = len(sp_rows)
    ax.set_title(f"{species}  (n={n} seeds)", color=color, fontsize=FS)
    ax.tick_params(axis="x", length=0)


def fig_strips(rows: list[dict]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    _strip_panel(axes[0], rows, "swine", PALETTE["swine"])
    _strip_panel(axes[1], rows, "human", PALETTE["human"])
    axes[1].set_ylabel("")

    # Legend
    med_line  = mlines.Line2D([], [], color=PALETTE["dgrey"], lw=1.8,
                               label="median")
    band_patch = mpl.patches.Patch(facecolor=PALETTE["lgrey"], alpha=0.4,
                                    label="±0.05 around\nmedian dense")
    axes[0].legend(handles=[med_line, band_patch], loc="lower left",
                   frameon=False, fontsize=FS_SM)

    fig.tight_layout(w_pad=1.5)
    out = OUT_DIR / "fig_sparse_si_strips.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}")
    return out


# ── Figure 2: gap distribution ───────────────────────────────────────────────
def _gap_violin_panel(ax, rows: list[dict], species: str, color: str) -> None:
    sp_rows = [r for r in rows if r["species"] == species]
    if not sp_rows:
        ax.set_visible(False)
        return

    rng = np.random.default_rng(0)
    gap_by_xi: dict[int, list[float]] = {xi: [] for xi, *_ in STRAT_LAYOUT[1:]}
    for r in sp_rows:
        for xi, _, _, _ in STRAT_LAYOUT[1:]:
            si = r["si"].get(xi, float("nan"))
            if np.isfinite(si) and np.isfinite(r["dense_si"]):
                gap_by_xi[xi].append(r["dense_si"] - si)

    positions = [xi for xi, *_ in STRAT_LAYOUT[1:]]
    data      = [gap_by_xi[xi] for xi in positions]
    data_nonempty = [d for d in data if d]

    if data_nonempty:
        parts = ax.violinplot(
            [d if d else [float("nan")] for d in data],
            positions=positions,
            widths=0.55,
            showmedians=False, showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.35)
            pc.set_edgecolor(color)
            pc.set_linewidth(0.5)

    for xi in positions:
        gaps = gap_by_xi[xi]
        if not gaps:
            continue
        jitter = rng.uniform(-0.12, 0.12, len(gaps))
        ax.scatter(xi + jitter, gaps, s=9, color=color, alpha=0.6,
                   edgecolors="none", zorder=3)
        med = float(np.median(gaps))
        ax.plot([xi - 0.28, xi + 0.28], [med, med],
                color=color, lw=1.8, solid_capstyle="round", zorder=4)

    ax.axhline(0, color=PALETTE["dgrey"], ls="-", lw=0.7, zorder=0)
    ax.axhline(0.05, color=PALETTE["lgrey"], ls="--", lw=0.5, zorder=0)
    ax.axhline(-0.05, color=PALETTE["lgrey"], ls="--", lw=0.5, zorder=0)

    xlabs = [t[3] for t in STRAT_LAYOUT[1:]]
    ax.set_xticks(positions)
    ax.set_xticklabels(xlabs, fontsize=FS_SM)
    ax.set_ylabel("SI$_{\\mathrm{dense}}$ − SI$_{\\mathrm{sparse}}$ (gap)")
    n = len(sp_rows)
    ax.set_title(f"{species}  (n={n} seeds)", color=color, fontsize=FS)
    ax.tick_params(axis="x", length=0)


def fig_gap(rows: list[dict]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    _gap_violin_panel(axes[0], rows, "swine", PALETTE["swine"])
    _gap_violin_panel(axes[1], rows, "human", PALETTE["human"])
    axes[1].set_ylabel("")

    # Annotation
    for ax in axes:
        ax.text(0.02, 0.97, "positive gap → dense better\nnegative gap → sparse overestimates",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=FS_SM, color=PALETTE["dgrey"])

    fig.tight_layout(w_pad=1.5)
    out = OUT_DIR / "fig_sparse_gap.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}")
    return out


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    rows = _load()
    if not rows:
        print("[figures_sparse_sampling] no paired data found", file=sys.stderr)
        return 1
    n_samples = len({r["sample"] for r in rows})
    print(f"[figures_sparse_sampling] {len(rows)} seed×sample records, "
          f"{n_samples} samples")
    fig_strips(rows)
    fig_gap(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
