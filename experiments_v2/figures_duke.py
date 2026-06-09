"""Build the Duke-cohort results figures in the Nature-Medicine style
established by figures_validation.py:

  fig_duke_per_sample.png        per-sample bar chart with rect, L1, wave SI
  fig_duke_probe_vs_l1.png       agreement scatter (probe SI vs L1 SI)
  fig_duke_selectivity_summary.png   two-panel combined version of the
                                       above (for tab:duke-summary)

Run from project root::

    python -m experiments_v2.figures_duke
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# ── Style (matches figures_validation.py) ───────────────────────────────────

PALETTE = {
    "rect":     "#0072B2",   # blue   — probe-based smart-init Adam-FD
    "l1":       "#D55E00",   # orange — random-init L1-discovery
    "wave":     "#009E73",   # green  — waveform autodiff
    "swine":    "#CC79A7",   # pink   — used for species marker styling
    "human":    "#56B4E9",   # sky-blue
    "grey":     "#3B3B3B",
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


# ── Config ──────────────────────────────────────────────────────────────────

ROOT     = Path(__file__).resolve().parent.parent
SWEEP    = ROOT / "outputs" / "duke_sweeps"
OUT_DIR  = ROOT / "manuscript" / "figures" / "duke"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AGREE_TOL = 0.05


# ── Loader ──────────────────────────────────────────────────────────────────

def _species_of(sample: str) -> str:
    return "human" if sample.startswith("human") else "swine"


def _short(sample: str) -> str:
    """Strip 'human_' / leading 'sub-' clutter so x-tick labels are readable."""
    s = sample.replace("human_", "")
    return s


def _load_rows() -> list[dict]:
    rows = []
    for d in sorted(SWEEP.iterdir()):
        if not d.is_dir():
            continue
        j = d / "data_seed_0000.json"
        if not j.exists():
            continue
        raw = json.loads(j.read_text())
        rect = raw.get("rect") or {}
        l1   = raw.get("rect_l1") or {}
        wav  = raw.get("waveform") or {}
        rows.append(dict(
            sample=raw.get("sample", d.name),
            species=_species_of(raw.get("sample", d.name)),
            rect_si=float(rect.get("achievable_si",
                                       abs(rect.get("final_si", 0.0)))),
            l1_si=float(l1.get("achievable_si",
                                  abs(l1.get("final_si", 0.0))))
                  if l1 else float("nan"),
            wave_si=float(wav.get("achievable_si",
                                      abs(wav.get("final_si", 0.0)))),
        ))
    return rows


# ── Figure: per-sample triple bar ───────────────────────────────────────────

def _per_sample_bars(ax, rows: list[dict]) -> None:
    # Sort: swine first, then human; within species by descending rect_si.
    sp_order = {"swine": 0, "human": 1}
    rows = sorted(rows,
                    key=lambda r: (sp_order.get(r["species"], 2),
                                    -r["rect_si"]))
    n = len(rows)
    x = np.arange(n)
    w = 0.27
    rect = np.array([r["rect_si"] for r in rows])
    l1   = np.array([r["l1_si"]   for r in rows])
    wave = np.array([r["wave_si"] for r in rows])

    ax.bar(x - w, rect, width=w, color=PALETTE["rect"], alpha=0.9,
              edgecolor=PALETTE["grey"], linewidth=0.7,
              label="probe Adam-FD")
    ax.bar(x,     l1,   width=w, color=PALETTE["l1"],   alpha=0.9,
              edgecolor=PALETTE["grey"], linewidth=0.7,
              label="L1 random init")
    ax.bar(x + w, wave, width=w, color=PALETTE["wave"], alpha=0.9,
              edgecolor=PALETTE["grey"], linewidth=0.7,
              label="waveform Adam")

    # Noise-floor line at SI=0.95
    ax.axhline(0.95, color=PALETTE["grey"], ls=":", lw=1.2, zorder=0)

    # Find species boundary (first index where species changes from swine).
    swine_indices = [i for i, r in enumerate(rows) if r["species"] == "swine"]
    human_indices = [i for i, r in enumerate(rows) if r["species"] == "human"]
    if swine_indices:
        s_lo, s_hi = min(swine_indices) - 0.5, max(swine_indices) + 0.5
        ax.axvspan(s_lo, s_hi, color=PALETTE["swine"], alpha=0.10,
                      zorder=0)
        ax.text((s_lo + s_hi) / 2, 1.04, "swine",
                  ha="center", va="bottom", fontsize=14,
                  color=PALETTE["grey"], weight="bold",
                  transform=ax.get_xaxis_transform())
    if human_indices:
        h_lo, h_hi = min(human_indices) - 0.5, max(human_indices) + 0.5
        ax.axvspan(h_lo, h_hi, color=PALETTE["human"], alpha=0.10,
                      zorder=0)
        ax.text((h_lo + h_hi) / 2, 1.04, "human",
                  ha="center", va="bottom", fontsize=14,
                  color=PALETTE["grey"], weight="bold",
                  transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels([_short(r["sample"]) for r in rows],
                          rotation=30, ha="right")
    ax.set_ylabel("achievable SI")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.5, n - 0.5)
    ax.legend(loc="lower center", ncol=3, frameon=False,
                bbox_to_anchor=(0.5, -0.45))


# ── Figure: probe-vs-L1 agreement scatter ──────────────────────────────────

def _probe_vs_l1_scatter(ax, rows: list[dict]) -> None:
    finite = [r for r in rows if np.isfinite(r["l1_si"])]
    rect = np.array([r["rect_si"] for r in finite])
    l1   = np.array([r["l1_si"]   for r in finite])
    species = np.array([r["species"] for r in finite])

    # Agreement band
    xx = np.linspace(0, 1, 100)
    ax.fill_between(xx, xx - AGREE_TOL, xx + AGREE_TOL,
                       color=PALETTE["grey"], alpha=0.10, linewidth=0,
                       label=f"$|\\Delta \\mathrm{{SI}}| \\leq {AGREE_TOL:.2f}$")
    ax.plot([0, 1], [0, 1], color=PALETTE["grey"], ls="--", lw=1.0,
              zorder=1)

    # Swine vs human markers
    for sp, marker, col in [("swine", "s", PALETTE["swine"]),
                                ("human", "o", PALETTE["human"])]:
        m = species == sp
        if not np.any(m):
            continue
        ax.scatter(rect[m], l1[m], marker=marker, s=85,
                      facecolor=col, edgecolor=PALETTE["grey"],
                      linewidth=1.0, alpha=0.9, label=sp, zorder=3)

    # Annotate each point with the sample name (short)
    for r in finite:
        ax.annotate(_short(r["sample"]),
                       (r["rect_si"], r["l1_si"]),
                       xytext=(6, -3), textcoords="offset points",
                       fontsize=9, color=PALETTE["grey"])

    # Annotation: agreement metric
    agree = sum(abs(r["rect_si"] - r["l1_si"]) <= AGREE_TOL for r in finite)
    n_fin = len(finite)
    ax.text(0.03, 0.97,
              f"$n$ = {n_fin}\n"
              f"agree $|\\Delta| \\leq {AGREE_TOL:.2f}$: "
              f"{agree}/{n_fin} ({100*agree/n_fin:.0f}%)\n"
              f"mean $|\\Delta \\mathrm{{SI}}|$ = "
              f"{float(np.mean(np.abs(rect - l1))):.3f}",
              transform=ax.transAxes, ha="left", va="top", fontsize=12,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=PALETTE["grey"], linewidth=0.8))

    ax.set_xlabel("probe Adam-FD SI")
    ax.set_ylabel("L1-discovery SI")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", frameon=False)


# ── Public callables ────────────────────────────────────────────────────────

def make_per_sample_fig(rows: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    _per_sample_bars(ax, rows)
    out = OUT_DIR / "fig_duke_per_sample.png"
    fig.savefig(out); plt.close(fig)
    return out


def make_agreement_fig(rows: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    _probe_vs_l1_scatter(ax, rows)
    out = OUT_DIR / "fig_duke_probe_vs_l1.png"
    fig.savefig(out); plt.close(fig)
    return out


def make_summary_fig(rows: list[dict]) -> Path:
    fig = plt.figure(figsize=(15.5, 5.6))
    ax1 = fig.add_axes([0.05, 0.18, 0.55, 0.74])
    ax2 = fig.add_axes([0.66, 0.10, 0.32, 0.84])
    _per_sample_bars(ax1, rows)
    _probe_vs_l1_scatter(ax2, rows)
    out = OUT_DIR / "fig_duke_selectivity_summary.png"
    fig.savefig(out); plt.close(fig)
    return out


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    rows = _load_rows()
    print(f"[load] {len(rows)} samples")
    paths = [
        make_per_sample_fig(rows),
        make_agreement_fig(rows),
        make_summary_fig(rows),
    ]
    for p in paths:
        print(f"  -> {p}")


if __name__ == "__main__":
    main()
