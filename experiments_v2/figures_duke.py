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


def _stage_metrics(stage: dict) -> dict:
    """Return (frac_tgt, frac_off, loss, time_s) for one stage's sub-dict."""
    fire = stage.get("firing") or {}
    loss = stage.get("final_loss")
    return dict(
        frac_tgt=float(fire.get("frac_fired_target", float("nan"))),
        frac_off=float(fire.get("frac_fired_nontarget", float("nan"))),
        loss=float(loss) if loss is not None else float("nan"),
        time_s=float(stage.get("time_s", float("nan"))),
    )


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
        m_rect = _stage_metrics(rect)
        m_l1   = _stage_metrics(l1) if l1 else dict(
            frac_tgt=float("nan"), frac_off=float("nan"),
            loss=float("nan"), time_s=float("nan"),
        )
        m_wave = _stage_metrics(wav)
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
            # Per-stage metrics (frac in [0,1]; loss; time s).
            rect_frac_tgt=m_rect["frac_tgt"], rect_frac_off=m_rect["frac_off"],
            rect_loss=m_rect["loss"],         rect_time=m_rect["time_s"],
            l1_frac_tgt=m_l1["frac_tgt"],     l1_frac_off=m_l1["frac_off"],
            l1_loss=m_l1["loss"],             l1_time=m_l1["time_s"],
            wave_frac_tgt=m_wave["frac_tgt"], wave_frac_off=m_wave["frac_off"],
            wave_loss=m_wave["loss"],         wave_time=m_wave["time_s"],
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


# ── Figure: bar + strip + error-bar metrics panel ──────────────────────────

def _bar_strip_panel(ax, rows: list[dict], field_template: str,
                       ylabel: str, ylim: tuple[float, float] | None = None,
                       scaling: float = 1.0, log: bool = False) -> None:
    """One panel: clusters of three bars per species, with median +
    IQR error bars and a jittered strip plot of individual samples.

    ``field_template`` is the row key with a method placeholder, e.g.
    ``"{method}_frac_tgt"``.  ``scaling`` multiplies the values (use
    100.0 to convert a [0,1] fraction to a percent).
    """
    METHODS  = ["rect", "l1", "wave"]
    SPECIES  = ["swine", "human"]
    SP_MARK  = {"swine": "s", "human": "o"}
    n_meth   = len(METHODS)
    bar_w    = 0.6
    group_w  = n_meth * bar_w + 0.7
    rng      = np.random.default_rng(0)
    centres  = []

    for s_idx, sp in enumerate(SPECIES):
        base = s_idx * group_w
        cluster_xs = []
        for m_idx, method in enumerate(METHODS):
            x = base + (m_idx - (n_meth - 1) / 2.0) * bar_w
            cluster_xs.append(x)
            vals = np.array([r[field_template.format(method=method)] * scaling
                              for r in rows if r["species"] == sp])
            finite = vals[np.isfinite(vals)]
            if finite.size:
                med  = float(np.median(finite))
                q25  = float(np.quantile(finite, 0.25))
                q75  = float(np.quantile(finite, 0.75))
            else:
                med = q25 = q75 = float("nan")
            ax.bar(x, med, width=bar_w * 0.9,
                      color=PALETTE[method], alpha=0.45,
                      edgecolor=PALETTE["grey"], linewidth=1.2,
                      zorder=2)
            # Error bars (IQR).
            if np.isfinite(med):
                ax.errorbar(x, med, yerr=[[med - q25], [q75 - med]],
                              fmt="none", ecolor=PALETTE["grey"],
                              elinewidth=1.4, capsize=4, capthick=1.4,
                              zorder=3)
            # Strip plot.
            if finite.size:
                jitter = rng.uniform(-0.18, 0.18, size=finite.size)
                ax.scatter(np.full(finite.size, x) + jitter, finite,
                              s=45, marker=SP_MARK[sp],
                              facecolor=PALETTE[method],
                              edgecolor=PALETTE["grey"], linewidth=0.8,
                              alpha=0.95, zorder=4)
        centres.append(float(np.mean(cluster_xs)))

    ax.set_xticks(centres)
    ax.set_xticklabels([
        f"swine\n($n$=" + str(sum(1 for r in rows if r["species"] == "swine")) + ")",
        f"human\n($n$=" + str(sum(1 for r in rows if r["species"] == "human")) + ")",
    ])
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if log:
        ax.set_yscale("log")
    # Leftmost bar centre is at -bar_w (m_idx=0, offset -1), rightmost at
    # (n_species-1)*group_w + bar_w.  Bars have width bar_w*0.9 so each
    # half-width is 0.45*bar_w.  Pad by an extra bar_w on each side so
    # the strip-plot jitter and IQR caps are not clipped.
    half_bar = 0.5 * bar_w * 0.9
    left_edge  = -bar_w - half_bar
    right_edge = (len(SPECIES) - 1) * group_w + bar_w + half_bar
    pad = bar_w
    ax.set_xlim(left_edge - pad, right_edge + pad)


def make_metrics_fig(rows: list[dict]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    (ax_tgt, ax_off), (ax_loss, ax_time) = axes

    _bar_strip_panel(ax_tgt,  rows, "{method}_frac_tgt",
                        "target fibres activated (%)",
                        ylim=(0, 105), scaling=100.0)
    _bar_strip_panel(ax_off,  rows, "{method}_frac_off",
                        "off-target fibres activated (%)",
                        ylim=(0, 105), scaling=100.0)
    _bar_strip_panel(ax_loss, rows, "{method}_loss",
                        "final loss")
    ax_loss.set_ylim(bottom=0)
    _bar_strip_panel(ax_time, rows, "{method}_time",
                        "wall-clock time (s)",
                        log=True)

    # One shared legend at the bottom (method colours).
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=PALETTE["rect"], edgecolor=PALETTE["grey"],
                alpha=0.55, label="probe Adam-FD"),
        Patch(facecolor=PALETTE["l1"],   edgecolor=PALETTE["grey"],
                alpha=0.55, label="L1 random init"),
        Patch(facecolor=PALETTE["wave"], edgecolor=PALETTE["grey"],
                alpha=0.55, label="waveform Adam"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
                  frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = OUT_DIR / "fig_duke_metrics.png"
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
    """Return (outline_xy, fascicles, contact_xyz_um).

    Outline is an (N, 2) array in um.  Fascicles is a list of dicts
    {polygon, id}.  contact_xyz_um is a (K, 3) array of contact
    positions in um (x, y, z).
    """
    d = DUKE_VES / sample_dir_name
    nx = json.loads((d / "nerve_xsec.json").read_text())
    outline = np.asarray(nx["nerve_outline_xy_um"], dtype=float)
    fascs = [dict(id=int(f["id"]),
                    polygon=np.asarray(f["polygon_xy_um"], dtype=float))
                for f in nx["fascicles"]]
    ec = json.loads((d / "electrode_config.json").read_text())
    # Cylindrical (R [m], phi [rad], z [m]) → Cartesian xyz [um].
    contact_xyz = np.array([
        [p["R"] * np.cos(p["phi"]) * 1e6,
         p["R"] * np.sin(p["phi"]) * 1e6,
         p["z"] * 1e6]
        for p in ec.get("patches", [])
    ], dtype=float)
    return outline, fascs, contact_xyz


def _draw_xsection(ax, sample_dir_name: str, raw: dict,
                     draw_electrodes: bool = True,
                     draw_label: bool = True) -> None:
    outline, fascs, contact_xyz = _load_geometry(sample_dir_name)
    contact_xy = contact_xyz[:, :2]
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
    if draw_electrodes and contact_xy.size and amps.size:
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

    # Top-left: sample name; top-right: SI badge.  Skipped when the
    # composite per-sample figure draws its own headings.
    name = _short(raw.get("sample", sample_dir_name))
    si = float(raw["rect"]["achievable_si"])
    if draw_label:
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


def _contact_to_grid(contact_xyz: np.ndarray) -> tuple[np.ndarray, list[str], list[float]]:
    """Map each of the K contacts to a (column_idx, row_idx) cell in a
    4-column-by-3-row grid where columns are angular positions
    (E, N, W, S in CCW order) and rows are z-levels (z_max on top,
    z_min on bottom).

    Returns
    -------
    cell_of_contact : (K, 2) int array with [column_idx, row_idx]
    col_labels      : list of 4 strings, left-to-right ('E', 'N', 'W', 'S')
    row_z_values    : list of 3 floats, top-to-bottom (z_max, z_mid, z_min)
    """
    K = contact_xyz.shape[0]
    # Bin each contact's phi into one of {0, 90, 180, 270} deg.
    phi_deg = np.degrees(np.arctan2(contact_xyz[:, 1], contact_xyz[:, 0])) % 360.0
    angle_bin = (np.round(phi_deg / 90.0).astype(int)) % 4
    # angle_bin: 0 = E (phi 0°), 1 = N (90°), 2 = W (180°), 3 = S (270°).
    col_labels = ["E", "N", "W", "S"]

    # Determine the three unique z levels.
    z = contact_xyz[:, 2]
    uniq = np.sort(np.unique(np.round(z, 3)))
    if uniq.size != 3:
        # Fall back: just rank by z.
        order = np.argsort(z)
        rank = np.empty(K, dtype=int); rank[order] = np.arange(K) // (K // 3)
        rank = np.clip(rank, 0, 2)
    else:
        # Row 0 = highest z (top), row 2 = lowest z (bottom).
        z_top, z_mid, z_bot = uniq[2], uniq[1], uniq[0]
        row_of = lambda zi: (
            0 if abs(zi - z_top) < 1e-3 else
            2 if abs(zi - z_bot) < 1e-3 else 1
        )
        rank = np.array([row_of(zi) for zi in z], dtype=int)

    cell = np.column_stack([angle_bin, rank])
    row_z_values = [float(uniq[2]) if uniq.size == 3 else float(z.max()),
                     float(uniq[1]) if uniq.size == 3 else float(np.median(z)),
                     float(uniq[0]) if uniq.size == 3 else float(z.min())]
    return cell, col_labels, row_z_values


def _draw_cuff_grid(ax, contact_xyz: np.ndarray, amps_mA: np.ndarray) -> None:
    """Draw the unrolled 4×3 cuff schematic with each cell coloured by
    the signed amplitude in mA (blue cathode, orange anode, white
    inactive).  Cell size encodes |amp|; centre prints the signed amp."""
    cell, col_labels, row_z_values = _contact_to_grid(contact_xyz)
    K = contact_xyz.shape[0]
    amps = amps_mA[:K]
    max_abs = max(0.5, float(np.max(np.abs(amps))))

    # Grid geometry
    ncols, nrows = 4, 3
    spacing_x, spacing_y = 1.0, 1.0
    for c_idx in range(ncols):
        for r_idx in range(nrows):
            ax.plot(c_idx * spacing_x, -(r_idx * spacing_y),
                      "o", markersize=8, markerfacecolor="white",
                      markeredgecolor="#cfcfcf", markeredgewidth=1.0,
                      zorder=1)
    # Plot each contact's circle.
    for k in range(K):
        c_idx, r_idx = int(cell[k, 0]), int(cell[k, 1])
        x_pos = c_idx * spacing_x
        y_pos = -(r_idx * spacing_y)
        amp = float(amps[k])
        rel = abs(amp) / max_abs if max_abs > 0 else 0.0
        size = 22.0 + 60.0 * rel
        if amp < -1e-9:
            facecolor = ELEC_CATHODE
        elif amp > 1e-9:
            facecolor = ELEC_ANODE
        else:
            facecolor = "white"
        ax.scatter(x_pos, y_pos, s=size**2 * 0.4,
                      facecolor=facecolor, edgecolor=PALETTE["grey"],
                      linewidth=1.2, alpha=0.9, zorder=3)
        if abs(amp) > 1e-9:
            ax.text(x_pos, y_pos,
                      f"{amp:+.2f}", ha="center", va="center",
                      fontsize=9,
                      color=("white" if abs(amp) / max_abs > 0.45
                              else "black"),
                      weight="bold", zorder=4)

    # Column labels (angular position) at the top, just above the
    # first row of circles.
    for c_idx, label in enumerate(col_labels):
        ax.text(c_idx * spacing_x, 0.55, label,
                  ha="center", va="center", fontsize=13,
                  color=PALETTE["grey"], weight="bold")
    # Row labels (z direction) on the left.
    row_pretty = ["+z (top)", "z = 0 (middle)", "-z (bottom)"]
    for r_idx, label in enumerate(row_pretty):
        ax.text(-0.75, -(r_idx * spacing_y), label,
                  ha="right", va="center", fontsize=11,
                  color=PALETTE["grey"])

    # Tight axis -- leave a bit more headroom at the top so the
    # column labels and the panel title don't overlap.
    ax.set_xlim(-1.7, ncols * spacing_x - 0.4)
    ax.set_ylim(-(nrows - 1) * spacing_y - 0.6, 1.3)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_pulse_trace(ax, pulse_shape: str, pw_ms: float,
                         asym_ratio: float) -> None:
    """Render the unit-amplitude biphasic_asym waveform that the
    optimiser used.  The y-axis is fractional amplitude (×amp_k for
    each electrode), the x-axis is time in ms.

    Phase 1 (cathodic): unit amplitude, duration pw_ms.
    Phase 2 (anodic recharge): -1/asym_ratio amplitude, duration
    pw_ms * asym_ratio (so the integrated charge is balanced).
    """
    if pulse_shape != "biphasic_asym":
        ax.text(0.5, 0.5, f"pulse: {pulse_shape}", transform=ax.transAxes,
                  ha="center", va="center", fontsize=12,
                  color=PALETTE["grey"])
        return
    # Pulse-mask convention: phase 1 = +1 (the strong "active" phase --
    # cathodic at the cathode, anodic at the anode, since the actual
    # delivered current is amp_k * pulse[t] and amp_k carries the
    # signed polarity).  Phase 2 = -1/asym_ratio (charge-recharge of
    # the opposite polarity).
    p1_amp = 1.0
    p2_amp = -1.0 / max(asym_ratio, 1e-6)
    p1_dur = pw_ms
    p2_dur = pw_ms * asym_ratio
    t = [0.0, 0.0, p1_dur, p1_dur, p1_dur + p2_dur,
            p1_dur + p2_dur, p1_dur + p2_dur + 0.3 * (p1_dur + p2_dur)]
    a = [0.0, p1_amp, p1_amp, p2_amp, p2_amp, 0.0, 0.0]
    # Neutral grey fill -- colour-coding by sign here would conflict
    # with the cathode/anode colours used elsewhere in the figure
    # (the SIGN of the pulse mask is not the polarity of the
    # delivered current; that depends on each electrode's amp_k).
    ax.plot(t, a, color="black", lw=1.8, zorder=3)
    ax.fill_between(t, a, 0, alpha=0.20, color=PALETTE["grey"],
                       linewidth=0, zorder=2)
    ax.axhline(0, color=PALETTE["grey"], lw=0.8, zorder=1)
    ax.set_xlabel("time (ms)", fontsize=11)
    ax.set_ylabel("amplitude\n(× amp$_k$)", fontsize=10)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(p2_amp - 0.2, p1_amp + 0.2)
    ax.tick_params(labelsize=10)


def _make_xsection_legend(out_path: Path) -> Path:
    """Standalone reference card explaining the per-sample
    cross-section + cuff-grid colour conventions."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.axis("off")

    rows = [
        ("fibres", [
            ("target fibre, fired",     FIB_TGT_FIRED,  "o", None),
            ("target fibre, silent",    FIB_TGT_SILENT, "o", PALETTE["grey"]),
            ("off-target fibre, fired", FIB_OFF_FIRED,  "o", None),
            ("off-target fibre, silent",FIB_OFF_SILENT, "o", PALETTE["grey"]),
        ]),
        ("fascicle outline", [
            ("target fascicle (light green tint)", "#dff0e0", "s", PALETTE["grey"]),
            ("off-target fascicle (grey)",          "#f0f0f0", "s", PALETTE["grey"]),
        ]),
        ("cuff electrode pattern", [
            ("cathode (amp < 0)",  ELEC_CATHODE, "o", PALETTE["grey"]),
            ("anode (amp > 0)",    ELEC_ANODE,   "o", PALETTE["grey"]),
            ("inactive (amp = 0)", "white",      "o", PALETTE["grey"]),
        ]),
    ]
    # Render headers + entries in two columns to keep the figure
    # compact.
    y = 0.96
    line_dy = 0.075
    section_dy = 0.05
    for header, entries in rows:
        ax.text(0.02, y, header, transform=ax.transAxes,
                  ha="left", va="top", fontsize=14, weight="bold",
                  color=PALETTE["grey"])
        y -= line_dy
        for label, col, marker, edge in entries:
            ax.scatter(0.06, y, transform=ax.transAxes,
                          s=180, marker=marker, facecolor=col,
                          edgecolor=(edge if edge is not None else "none"),
                          linewidth=1.0, zorder=3)
            ax.text(0.11, y - 0.005, label, transform=ax.transAxes,
                      ha="left", va="center", fontsize=12,
                      color="black")
            y -= line_dy
        y -= section_dy

    # SI badge demo at bottom
    ax.text(0.02, y, "SI badge", transform=ax.transAxes,
              ha="left", va="top", fontsize=14, weight="bold",
              color=PALETTE["grey"])
    y -= line_dy
    for label, fc in [("SI ≥ 0.95 (near-noise-floor)", "#dff0e0"),
                        ("SI < 0.95", "white")]:
        ax.text(0.06, y, f"SI = X.XX", transform=ax.transAxes,
                  ha="left", va="center", fontsize=12,
                  color="black",
                  bbox=dict(boxstyle="round,pad=0.25", facecolor=fc,
                              edgecolor=PALETTE["grey"], linewidth=0.8))
        ax.text(0.30, y, label, transform=ax.transAxes,
                  ha="left", va="center", fontsize=12, color="black")
        y -= line_dy
    fig.savefig(out_path); plt.close(fig)
    return out_path


def make_per_sample_xsection(rows: list[dict]) -> list[Path]:
    """For each sample, create a subfolder under figures/duke/per_sample/
    containing four standalone PNGs:

      <sample>/pulse.png         stimulation pulse waveform
      <sample>/cross_section.png anatomical cross-section
      <sample>/cuff_pattern.png  schematic 4x3 unrolled cuff grid
      <sample>/legend.png        reference card explaining the colour
                                   conventions (one copy per sample so
                                   each folder is self-contained)
    """
    base_dir = OUT_DIR / "per_sample"
    base_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for r in rows:
        d = SWEEP / r["sample"]
        j = d / "data_seed_0000.json"
        if not j.exists():
            continue
        raw = json.loads(j.read_text())
        outline, fascs, contact_xyz = _load_geometry(r["sample"])
        amps = np.asarray(raw["rect"]["amps_mA"], dtype=float)
        sample_dir = base_dir / r["sample"]
        sample_dir.mkdir(parents=True, exist_ok=True)

        # 1. Pulse trace.
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        _draw_pulse_trace(ax, raw["pulse_shape"],
                              float(raw["pulse_pw_ms"]),
                              float(raw["pulse_asym_ratio"]))
        fig.tight_layout()
        p_pulse = sample_dir / "pulse.png"
        fig.savefig(p_pulse); plt.close(fig); written.append(p_pulse)

        # 2. Cross-section.
        fig, ax = plt.subplots(figsize=(7.5, 7.0))
        _draw_xsection(ax, r["sample"], raw, draw_electrodes=False,
                          draw_label=False)
        si = float(raw["rect"]["achievable_si"])
        ax.set_title(r["sample"], fontsize=15, weight="bold",
                        color=PALETTE["grey"], loc="left", pad=10)
        ax.text(0.98, 1.02, f"SI = {si:.2f}",
                  transform=ax.transAxes, ha="right", va="bottom",
                  fontsize=13,
                  color=("black" if si >= 0.95 else PALETTE["grey"]),
                  weight="bold" if si >= 0.95 else "normal",
                  bbox=dict(boxstyle="round,pad=0.25",
                              facecolor=("#dff0e0" if si >= 0.95
                                          else "white"),
                              edgecolor=PALETTE["grey"], linewidth=0.8))
        fig.tight_layout()
        p_xs = sample_dir / "cross_section.png"
        fig.savefig(p_xs); plt.close(fig); written.append(p_xs)

        # 3. Cuff grid.
        fig, ax = plt.subplots(figsize=(6.0, 5.5))
        _draw_cuff_grid(ax, contact_xyz, amps)
        ax.set_title("Cuff electrode pattern (mA)", fontsize=14,
                        weight="bold", color=PALETTE["grey"],
                        loc="center", pad=8)
        fig.tight_layout()
        p_cuff = sample_dir / "cuff_pattern.png"
        fig.savefig(p_cuff); plt.close(fig); written.append(p_cuff)

        # 4. Legend.
        p_leg = sample_dir / "legend.png"
        _make_xsection_legend(p_leg); written.append(p_leg)
    return written


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
        make_metrics_fig(rows),
        make_summary_fig(rows),
    ]
    paths.extend(make_xsection_galleries(rows))
    paths.extend(make_per_sample_xsection(rows))
    for p in paths:
        print(f"  -> {p}")


if __name__ == "__main__":
    main()
