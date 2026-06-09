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


# ── Figure: SI by species × method ──────────────────────────────────────────

def _box_by_species_method(ax, rows: list[dict]) -> None:
    """Two clusters (swine, human) of three boxes each (probe, L1, wave).
    Box face colour encodes method; scatter markers encode species
    (square = swine, circle = human) to match the agreement scatter in
    the companion panel."""
    METHODS  = ["rect", "l1", "wave"]
    METHOD_LABELS = ["probe Adam-FD", "L1 random init", "waveform Adam"]
    SPECIES  = ["swine", "human"]
    SPECIES_MARKERS = {"swine": "s", "human": "o"}

    n_sp     = len(SPECIES)
    n_meth   = len(METHODS)
    box_w    = 0.55
    group_w  = n_meth * box_w + 0.7   # spacing between species clusters
    positions: list[float] = []
    groups:    list[np.ndarray] = []
    species_for_box: list[str] = []
    method_for_box:  list[str] = []
    centres = []   # x position of each species cluster centre

    for s_idx, sp in enumerate(SPECIES):
        base = s_idx * group_w
        cluster_xs = []
        for m_idx, method in enumerate(METHODS):
            x = base + (m_idx - (n_meth - 1) / 2.0) * box_w
            positions.append(x); cluster_xs.append(x)
            vals = np.array([r[f"{method}_si"] for r in rows
                                if r["species"] == sp
                                and np.isfinite(r[f"{method}_si"])])
            groups.append(vals); species_for_box.append(sp)
            method_for_box.append(method)
        centres.append(float(np.mean(cluster_xs)))

    # Boxes (face-coloured by method, edge dark grey)
    bp = ax.boxplot(
        groups, positions=positions, widths=box_w * 0.85,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=2.0),
        boxprops=dict(linewidth=1.2),
        whiskerprops=dict(linewidth=1.2, color=PALETTE["grey"]),
        capprops=dict(linewidth=1.2, color=PALETTE["grey"]),
    )
    for patch, method in zip(bp["boxes"], method_for_box):
        patch.set_facecolor(PALETTE[method])
        patch.set_alpha(0.35)
        patch.set_edgecolor(PALETTE["grey"])

    # Jittered scatter: marker by species, colour by method
    rng = np.random.default_rng(0)
    for x, vals, sp, method in zip(positions, groups, species_for_box,
                                       method_for_box):
        if vals.size == 0:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=vals.size)
        ax.scatter(np.full(vals.size, x) + jitter, vals,
                      s=55, marker=SPECIES_MARKERS[sp],
                      facecolor=PALETTE[method], edgecolor=PALETTE["grey"],
                      linewidth=0.8, alpha=0.95, zorder=3)

    # Noise-floor line at SI=0.95
    ax.axhline(0.95, color=PALETTE["grey"], ls=":", lw=1.2, zorder=0)

    # X ticks at cluster centres
    ax.set_xticks(centres)
    ax.set_xticklabels([f"swine\n($n$=" +
                            str(sum(1 for r in rows if r["species"] == "swine"))
                            + ")",
                          f"human\n($n$=" +
                            str(sum(1 for r in rows if r["species"] == "human"))
                            + ")"])

    # Method legend (face colours)
    from matplotlib.patches import Patch
    method_handles = [Patch(facecolor=PALETTE[m], edgecolor=PALETTE["grey"],
                                alpha=0.55, label=lab)
                          for m, lab in zip(METHODS, METHOD_LABELS)]
    ax.legend(handles=method_handles, loc="lower center", ncol=3,
                frameon=False, bbox_to_anchor=(0.5, -0.28))

    ax.set_ylabel("achievable SI")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(positions[0] - box_w, positions[-1] + box_w)


def make_species_method_fig(rows: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    _box_by_species_method(ax, rows)
    out = OUT_DIR / "fig_duke_si_by_species_method.png"
    fig.savefig(out); plt.close(fig)
    return out


# ── Cross-section gallery ───────────────────────────────────────────────────

DUKE_VES = ROOT / "duke_Ves"

# Fiber-state colours (target / off-target × fired / silent).
FIB_TGT_FIRED   = "#2E7D32"   # green
FIB_TGT_SILENT  = "#B7D8B6"   # very light green
FIB_OFF_FIRED   = "#D55E00"   # vermillion / orange
FIB_OFF_SILENT  = "#E0E0E0"   # light grey

# Electrode encoding.
ELEC_CATHODE = "#0072B2"   # blue
ELEC_ANODE   = "#D55E00"   # orange


def _load_geometry(sample_dir_name: str):
    """Return (outline_xy, fascicles_list_of_polygons, contact_xy_um, target_fasc_ids).

    Outline is an (N, 2) array in um.  Fascicles is a list of dicts
    {polygon, id}.  contact_xy_um is a (K, 2) array.  target_fasc_ids
    is read from the sweep JSON's ``cluster.target_ids`` field via the
    caller.
    """
    d = DUKE_VES / sample_dir_name
    nx = json.loads((d / "nerve_xsec.json").read_text())
    outline = np.asarray(nx["nerve_outline_xy_um"], dtype=float)
    fascs = [dict(id=int(f["id"]),
                    polygon=np.asarray(f["polygon_xy_um"], dtype=float))
                for f in nx["fascicles"]]
    ec = json.loads((d / "electrode_config.json").read_text())
    # Cylindrical (R [m], phi [rad], z [m]) → Cartesian xy [um].
    contact_xy = np.array([
        [p["R"] * np.cos(p["phi"]) * 1e6,
         p["R"] * np.sin(p["phi"]) * 1e6]
        for p in ec.get("patches", [])
    ], dtype=float)
    return outline, fascs, contact_xy


def _draw_xsection(ax, sample_dir_name: str, raw: dict) -> None:
    outline, fascs, contact_xy = _load_geometry(sample_dir_name)
    fiber_x = np.asarray(raw["nerve"]["fiber_x_um"], dtype=float)
    fiber_y = np.asarray(raw["nerve"]["fiber_y_um"], dtype=float)
    tgt = np.asarray(raw["nerve"]["target_mask"], dtype=bool)
    acts = np.asarray(raw["rect"]["final_acts"], dtype=float)
    fired = acts > 0.5
    amps = np.asarray(raw["rect"]["amps_mA"], dtype=float)
    target_fasc_ids = set(raw["cluster"]["target_ids"])

    # Nerve outline.
    ax.fill(outline[:, 0], outline[:, 1],
              facecolor="#fafafa", edgecolor=PALETTE["grey"], linewidth=1.2,
              zorder=1)
    # Fascicles -- target faintly tinted green, off-target plain.
    for f in fascs:
        is_target = f["id"] in target_fasc_ids
        ax.fill(f["polygon"][:, 0], f["polygon"][:, 1],
                  facecolor=("#dff0e0" if is_target else "#f0f0f0"),
                  edgecolor=PALETTE["grey"], linewidth=0.6, zorder=2)

    # Fibres -- four-colour scheme.
    masks = [
        (~tgt & ~fired, FIB_OFF_SILENT, "off-target silent"),
        ( tgt & ~fired, FIB_TGT_SILENT, "target silent"),
        (~tgt &  fired, FIB_OFF_FIRED,  "off-target fired"),
        ( tgt &  fired, FIB_TGT_FIRED,  "target fired"),
    ]
    for m, col, _lab in masks:
        if np.any(m):
            ax.scatter(fiber_x[m], fiber_y[m], s=3.5, color=col,
                          edgecolors="none", zorder=4, alpha=0.95)

    # Electrodes -- signed amp gives colour; |amp| gives size.
    if contact_xy.size and amps.size:
        K = min(len(contact_xy), len(amps))
        radii = 0.5 * (np.abs(amps[:K]) ** 0.5) * 90.0 + 22.0
        for k in range(K):
            ax.add_patch(plt.Circle(
                (contact_xy[k, 0], contact_xy[k, 1]), radii[k],
                facecolor=(ELEC_CATHODE if amps[k] < -1e-9 else
                            ELEC_ANODE if amps[k] > 1e-9 else "white"),
                edgecolor=PALETTE["grey"], linewidth=0.9, alpha=0.85,
                zorder=5,
            ))

    # Aesthetics.
    pad = 100.0
    xy = np.concatenate([outline, contact_xy]) if contact_xy.size else outline
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Top-left: sample name; top-right: SI badge.
    name = _short(raw.get("sample", sample_dir_name))
    si = float(raw["rect"]["achievable_si"])
    ax.text(0.03, 0.97, name, transform=ax.transAxes,
              ha="left", va="top", fontsize=11, weight="bold",
              color=PALETTE["grey"])
    ax.text(0.97, 0.97, f"SI = {si:.2f}", transform=ax.transAxes,
              ha="right", va="top", fontsize=11,
              color=("black" if si >= 0.95 else PALETTE["grey"]),
              weight="bold" if si >= 0.95 else "normal",
              bbox=dict(boxstyle="round,pad=0.25",
                          facecolor=("#dff0e0" if si >= 0.95 else "white"),
                          edgecolor=PALETTE["grey"], linewidth=0.8))


def _xsection_gallery(samples: list[tuple[str, dict]],
                        nrows: int, ncols: int,
                        out_path: Path, title: str) -> Path:
    fig, axes = plt.subplots(nrows, ncols,
                                figsize=(3.6 * ncols, 3.6 * nrows),
                                squeeze=False)
    for ax_idx, (dir_name, raw) in enumerate(samples):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        _draw_xsection(ax, dir_name, raw)
    # Hide unused subplots.
    for j in range(len(samples), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    # Shared legend at the bottom.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=FIB_TGT_FIRED,
                 markersize=8, label="target fired"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=FIB_TGT_SILENT,
                 markersize=8, markeredgecolor=PALETTE["grey"],
                 label="target silent"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=FIB_OFF_FIRED,
                 markersize=8, label="off-target fired"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=FIB_OFF_SILENT,
                 markersize=8, markeredgecolor=PALETTE["grey"],
                 label="off-target silent"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ELEC_CATHODE,
                 markersize=11, markeredgecolor=PALETTE["grey"],
                 label="cathode (-)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ELEC_ANODE,
                 markersize=11, markeredgecolor=PALETTE["grey"],
                 label="anode (+)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6,
                  frameon=False, fontsize=12,
                  bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("")   # Nature-style: no title (species implied by filename)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path); plt.close(fig)
    return out_path


def make_xsection_galleries(rows: list[dict]) -> list[Path]:
    """Two PNGs: one for swine samples, one for human samples."""
    # Load each sample's raw JSON.
    samples_by_species: dict[str, list[tuple[str, dict]]] = {
        "swine": [], "human": []
    }
    for r in rows:
        d = SWEEP / r["sample"]
        j = d / "data_seed_0000.json"
        if not j.exists():
            continue
        raw = json.loads(j.read_text())
        samples_by_species[r["species"]].append((r["sample"], raw))

    paths = []
    for sp in ("swine", "human"):
        s = samples_by_species[sp]
        if not s:
            continue
        s.sort(key=lambda t: -float(t[1]["rect"]["achievable_si"]))
        n = len(s)
        # Layout: aim for ~3 columns
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        out = OUT_DIR / f"fig_duke_xsections_{sp}.png"
        _xsection_gallery(s, nrows, ncols, out,
                            title=f"{sp} cohort cross-sections")
        paths.append(out)
    return paths


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
        make_species_method_fig(rows),
        make_summary_fig(rows),
    ]
    paths.extend(make_xsection_galleries(rows))
    for p in paths:
        print(f"  -> {p}")


if __name__ == "__main__":
    main()
