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
    "rect":     "#2166AC",   # muted dark blue — probe Adam-FD
    "swine":    "#D6604D",   # muted terracotta — swine species
    "human":    "#4393C3",   # steel blue — human species
    "grey":     "#555555",   # soft dark grey
    "lgrey":    "#AAAAAA",   # light grey for guides / decorations
}

# NM font sizes (points)
FS     = 7    # body / tick labels
FS_SM  = 6    # minor annotations
FS_AX  = 7    # axis labels
FS_TTL = 7    # panel titles / headings

mpl.rcParams.update({
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Arial", "Helvetica", "Liberation Sans",
                               "DejaVu Sans"],
    "font.size":             FS,
    "axes.labelsize":        FS_AX,
    "axes.titlesize":        FS_TTL,
    "xtick.labelsize":       FS,
    "ytick.labelsize":       FS,
    "legend.fontsize":       FS,
    "axes.linewidth":        0.5,
    "xtick.major.width":     0.5,
    "ytick.major.width":     0.5,
    "xtick.minor.width":     0.35,
    "ytick.minor.width":     0.35,
    "xtick.major.size":      2.5,
    "ytick.major.size":      2.5,
    "xtick.minor.size":      1.2,
    "ytick.minor.size":      1.2,
    "lines.linewidth":       0.8,
    "patch.linewidth":       0.5,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.grid":             False,
    "savefig.dpi":           600,
    "savefig.bbox":          "tight",
    "figure.facecolor":      "white",
    "axes.facecolor":        "white",
    "pdf.fonttype":          42,
    "svg.fonttype":          "none",
})


# ── Config ──────────────────────────────────────────────────────────────────

ROOT     = Path(__file__).resolve().parent.parent
SWEEP    = ROOT / "outputs" / "duke_sweeps"
OUT_DIR  = ROOT / "manuscript" / "figures" / "duke"
PNG_DIR  = OUT_DIR / "png"
SVG_DIR  = OUT_DIR / "svg"
for _d in (OUT_DIR, PNG_DIR, SVG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

AGREE_TOL = 0.05


def _savefig(fig, name: str) -> list[Path]:
    """Save fig to both PNG_DIR/<name>.png and SVG_DIR/<name>.svg."""
    paths = []
    for d, ext in [(PNG_DIR, ".png"), (SVG_DIR, ".svg")]:
        p = d / (name + ext)
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def _savefig_sample(fig, sample_dir: Path, name: str) -> list[Path]:
    """Save per-sample fig to <sample_dir>/png/ and <sample_dir>/svg/."""
    paths = []
    for sub, ext in [("png", ".png"), ("svg", ".svg")]:
        d = sample_dir / sub
        d.mkdir(parents=True, exist_ok=True)
        p = d / (name + ext)
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


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
    sp_order = {"swine": 0, "human": 1}
    rows = sorted(rows,
                    key=lambda r: (sp_order.get(r["species"], 2),
                                    -r["rect_si"]))
    n = len(rows)
    x = np.arange(n)
    w = 0.6
    colors = [PALETTE["swine"] if r["species"] == "swine"
               else PALETTE["human"] for r in rows]

    ax.bar(x, [r["rect_si"] for r in rows], width=w,
              color=colors, alpha=0.85,
              edgecolor="white", linewidth=0.4)

    # Threshold line
    ax.axhline(0.90, color=PALETTE["lgrey"], ls="--", lw=0.7, zorder=0)

    # Subtle species spans + small labels at top of span
    swine_i = [i for i, r in enumerate(rows) if r["species"] == "swine"]
    human_i = [i for i, r in enumerate(rows) if r["species"] == "human"]
    for idx_list, col, label in [
        (swine_i, PALETTE["swine"], "swine"),
        (human_i, PALETTE["human"], "human"),
    ]:
        if not idx_list:
            continue
        lo, hi = min(idx_list) - 0.5, max(idx_list) + 0.5
        ax.axvspan(lo, hi, color=col, alpha=0.06, zorder=0, linewidth=0)
        ax.text((lo + hi) / 2, 1.02, label,
                  ha="center", va="bottom", fontsize=FS_SM,
                  color=col, weight="semibold",
                  transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels([_short(r["sample"]) for r in rows],
                          rotation=35, ha="right", fontsize=FS_SM)
    ax.set_ylabel("selectivity index (SI)")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.5, n - 0.5)
    ax.tick_params(axis="x", length=0)


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
    METHODS  = ["rect"]
    METHOD_LABELS = ["probe Adam-FD"]
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

    # Boxes — face-coloured by species
    bp = ax.boxplot(
        groups, positions=positions, widths=box_w * 0.82,
        patch_artist=True, showfliers=False,
        medianprops=dict(color=PALETTE["grey"], linewidth=1.0),
        boxprops=dict(linewidth=0.5),
        whiskerprops=dict(linewidth=0.5, color=PALETTE["lgrey"]),
        capprops=dict(linewidth=0.5, color=PALETTE["lgrey"]),
    )
    for patch, sp in zip(bp["boxes"], species_for_box):
        patch.set_facecolor(PALETTE[sp])
        patch.set_alpha(0.30)
        patch.set_edgecolor(PALETTE["lgrey"])

    # Jittered scatter coloured by species
    rng = np.random.default_rng(0)
    for x, vals, sp in zip(positions, groups, species_for_box):
        if vals.size == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=vals.size)
        ax.scatter(np.full(vals.size, x) + jitter, vals,
                      s=12, marker=SPECIES_MARKERS[sp],
                      facecolor=PALETTE[sp], edgecolor="white",
                      linewidth=0.3, alpha=0.90, zorder=3)

    ax.axhline(0.90, color=PALETTE["lgrey"], ls="--", lw=0.7, zorder=0)

    ax.set_xticks(centres)
    n_sw = sum(1 for r in rows if r["species"] == "swine")
    n_hu = sum(1 for r in rows if r["species"] == "human")
    ax.set_xticklabels([f"swine\n(n={n_sw})", f"human\n(n={n_hu})"],
                         fontsize=FS)

    ax.set_ylabel("selectivity index (SI)")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(positions[0] - box_w, positions[-1] + box_w)


def make_species_method_fig(rows: list[dict]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    _box_by_species_method(ax, rows)
    fig.tight_layout()
    return _savefig(fig, "fig_duke_si_by_species_method")


# ── Figure: bar + strip + error-bar metrics panel ──────────────────────────

def _bar_strip_panel(ax, rows: list[dict], field_template: str,
                       ylabel: str, ylim: tuple[float, float] | None = None,
                       scaling: float = 1.0, log: bool = False,
                       methods: list[str] | None = None) -> None:
    """One panel: clusters of bars per species, with median +
    IQR error bars and a jittered strip plot of individual samples.

    ``field_template`` is the row key with a method placeholder, e.g.
    ``"{method}_frac_tgt"``.  ``scaling`` multiplies the values (use
    100.0 to convert a [0,1] fraction to a percent).
    ``methods`` controls which methods are shown (default: all three).
    """
    METHODS  = methods if methods is not None else ["rect"]
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
                      color=PALETTE[sp], alpha=0.40,
                      edgecolor="none", linewidth=0,
                      zorder=2)
            if np.isfinite(med):
                ax.errorbar(x, med, yerr=[[med - q25], [q75 - med]],
                              fmt="none", ecolor=PALETTE["lgrey"],
                              elinewidth=0.6, capsize=2, capthick=0.6,
                              zorder=3)
            if finite.size:
                jitter = rng.uniform(-0.14, 0.14, size=finite.size)
                ax.scatter(np.full(finite.size, x) + jitter, finite,
                              s=12, marker=SP_MARK[sp],
                              facecolor=PALETTE[sp],
                              edgecolor="white", linewidth=0.3,
                              alpha=0.90, zorder=4)
        centres.append(float(np.mean(cluster_xs)))

    n_sw = sum(1 for r in rows if r["species"] == "swine")
    n_hu = sum(1 for r in rows if r["species"] == "human")
    ax.set_xticks(centres)
    ax.set_xticklabels([f"swine\n(n={n_sw})", f"human\n(n={n_hu})"],
                         fontsize=FS)
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


def make_metrics_fig(rows: list[dict]) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))
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

    fig.tight_layout()
    return _savefig(fig, "fig_duke_metrics")


def _violin_panel(ax, rows: list[dict], field_template: str,
                   ylabel: str, ylim: tuple[float, float] | None = None,
                   scaling: float = 1.0, log: bool = False) -> None:
    """Violin + jittered-scatter panel: same layout as _bar_strip_panel
    but shows full distribution shape.  Falls back to scatter-only when
    n < 2 (kernel density undefined)."""
    METHODS  = ["rect"]
    SPECIES  = ["swine", "human"]
    SP_MARK  = {"swine": "s", "human": "o"}
    n_meth   = len(METHODS)
    viol_w   = 0.5
    group_w  = n_meth * viol_w + 0.7
    rng      = np.random.default_rng(0)
    centres  = []

    for s_idx, sp in enumerate(SPECIES):
        base = s_idx * group_w
        cluster_xs = []
        for m_idx, method in enumerate(METHODS):
            x = base + (m_idx - (n_meth - 1) / 2.0) * viol_w
            cluster_xs.append(x)
            vals = np.array([r[field_template.format(method=method)] * scaling
                              for r in rows if r["species"] == sp])
            finite = vals[np.isfinite(vals)]
            if finite.size >= 2:
                vp = ax.violinplot([finite], positions=[x],
                                    widths=viol_w * 0.85,
                                    showmeans=False, showmedians=True,
                                    showextrema=True)
                for body in vp["bodies"]:
                    body.set_facecolor(PALETTE[sp])
                    body.set_edgecolor("none")
                    body.set_alpha(0.30)
                    body.set_linewidth(0)
                for part in ("cmedians", "cmaxes", "cmins", "cbars"):
                    if part in vp:
                        vp[part].set_color(PALETTE["lgrey"])
                        vp[part].set_linewidth(0.6)
            if finite.size >= 1:
                jitter = rng.uniform(-0.08, 0.08, size=finite.size)
                ax.scatter(np.full(finite.size, x) + jitter, finite,
                              s=12, marker=SP_MARK[sp],
                              facecolor=PALETTE[sp],
                              edgecolor="white", linewidth=0.3,
                              alpha=0.90, zorder=5)
        centres.append(float(np.mean(cluster_xs)))

    n_sw = sum(1 for r in rows if r["species"] == "swine")
    n_hu = sum(1 for r in rows if r["species"] == "human")
    ax.set_xticks(centres)
    ax.set_xticklabels([f"swine\n(n={n_sw})", f"human\n(n={n_hu})"],
                         fontsize=FS)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if log:
        ax.set_yscale("log")
    half_v = 0.5 * viol_w * 0.85
    left_edge  = 0 - viol_w - half_v
    right_edge = (len(SPECIES) - 1) * group_w + viol_w + half_v
    pad = viol_w
    ax.set_xlim(left_edge - pad, right_edge + pad)


def make_metrics_violin_fig(rows: list[dict]) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))
    (ax_tgt, ax_off), (ax_loss, ax_time) = axes

    _violin_panel(ax_tgt,  rows, "{method}_frac_tgt",
                   "target fibres activated (%)",
                   ylim=(0, 105), scaling=100.0)
    _violin_panel(ax_off,  rows, "{method}_frac_off",
                   "off-target fibres activated (%)",
                   ylim=(0, 105), scaling=100.0)
    _violin_panel(ax_loss, rows, "{method}_loss", "final loss")
    ax_loss.set_ylim(bottom=0)
    _violin_panel(ax_time, rows, "{method}_time",
                   "wall-clock time (s)", log=True)

    fig.tight_layout()
    return _savefig(fig, "fig_duke_metrics_violin")


# ── Cross-section gallery ───────────────────────────────────────────────────

DUKE_VES = ROOT / "duke_Ves"

# Fiber-state colours (target / off-target × fired / silent) — muted NM palette.
FIB_TGT_FIRED   = "#1A7340"   # forest green
FIB_TGT_SILENT  = "#C8E6C9"   # pale green
FIB_OFF_FIRED   = "#C0392B"   # muted red
FIB_OFF_SILENT  = "#E8E8E8"   # near-white grey

# Electrode encoding — matches _draw_cuff_grid colours.
ELEC_CATHODE = "#0E7C7B"   # deep teal (distinct from human blue)
ELEC_ANODE   = "#D97706"   # warm amber (distinct from swine red)


def _load_geometry(sample_dir_name: str):
    """Return (outline_xy, fascicles, contact_xyz_um, patches).

    Outline is an (N, 2) array in um.  Fascicles is a list of dicts
    {polygon, id}.  contact_xyz_um is a (K, 3) array of contact
    positions in um (x, y, z).  patches is the raw list of dicts from
    electrode_config.json (each with R, phi, dphi, z, dz in SI units).
    """
    d = DUKE_VES / sample_dir_name
    nx = json.loads((d / "nerve_xsec.json").read_text())
    outline = np.asarray(nx["nerve_outline_xy_um"], dtype=float)
    fascs = [dict(id=int(f["id"]),
                    polygon=np.asarray(f["polygon_xy_um"], dtype=float))
                for f in nx["fascicles"]]
    ec = json.loads((d / "electrode_config.json").read_text())
    patches = ec.get("patches", [])
    # Cylindrical (R [m], phi [rad], z [m]) → Cartesian xyz [um].
    contact_xyz = np.array([
        [p["R"] * np.cos(p["phi"]) * 1e6,
         p["R"] * np.sin(p["phi"]) * 1e6,
         p["z"] * 1e6]
        for p in patches
    ], dtype=float)
    return outline, fascs, contact_xyz, patches


def _draw_cuff_arcs(ax, patches: list[dict], amps_mA: np.ndarray) -> None:
    """Draw the cuff electrode in the x-y plane as arc-shaped Wedge patches.

    The cuff centre is at (0, 0) µm (the electrode_config.json coordinate
    origin).  We draw:
      - a thin silicone ring (full 360°) at R
      - one arc contact per unique angular position, coloured by signed
        amplitude.  When multiple z-levels exist we use the middle row.
    """
    from matplotlib.patches import Wedge

    if not patches or amps_mA.size == 0:
        return

    CONTACT_THICK_UM = 160.0   # radial thickness of each contact patch
    SILICONE_THICK_UM = 70.0   # thin silicone backing ring

    R_um = float(patches[0]["R"]) * 1e6

    # ── silicone ring ────────────────────────────────────────────────────────
    ax.add_patch(Wedge(
        (0.0, 0.0), R_um + SILICONE_THICK_UM, 0.0, 360.0,
        width=SILICONE_THICK_UM,
        facecolor="#E8E8E8", edgecolor="#BBBBBB", linewidth=0.35,
        alpha=0.7, zorder=3,
    ))

    # ── pick middle z-row ────────────────────────────────────────────────────
    z_vals = np.array([p["z"] for p in patches])
    unique_z = np.unique(z_vals)
    z_mid = unique_z[len(unique_z) // 2]
    mid_idx = [i for i, p in enumerate(patches) if abs(p["z"] - z_mid) < 1e-9]

    # ── contact arcs ─────────────────────────────────────────────────────────
    for i in mid_idx:
        if i >= len(amps_mA):
            continue
        p = patches[i]
        amp = float(amps_mA[i])
        phi_deg  = float(np.degrees(p["phi"]))
        dphi_deg = float(np.degrees(p["dphi"]))
        theta1 = phi_deg - dphi_deg / 2.0
        theta2 = phi_deg + dphi_deg / 2.0
        if amp < -1e-9:
            fc = ELEC_CATHODE
        elif amp > 1e-9:
            fc = ELEC_ANODE
        else:
            fc = "white"
        ec = "#1A5276" if amp < -1e-9 else ("#873600" if amp > 1e-9 else "#BBBBBB")
        ax.add_patch(Wedge(
            (0.0, 0.0), R_um + CONTACT_THICK_UM,
            theta1, theta2,
            width=CONTACT_THICK_UM,
            facecolor=fc, edgecolor=ec, linewidth=0.35,
            alpha=0.88, zorder=4,
        ))


def _draw_xsection(ax, sample_dir_name: str, raw: dict,
                     draw_electrodes: bool = True,
                     draw_label: bool = True) -> None:
    outline, fascs, contact_xyz, patches = _load_geometry(sample_dir_name)
    contact_xy = contact_xyz[:, :2]
    fiber_x = np.asarray(raw["nerve"]["fiber_x_um"], dtype=float)
    fiber_y = np.asarray(raw["nerve"]["fiber_y_um"], dtype=float)
    tgt = np.asarray(raw["nerve"]["target_mask"], dtype=bool)
    acts = np.asarray(raw["rect"]["final_acts"], dtype=float)
    fired = acts > 0.5
    amps = np.asarray(raw["rect"]["amps_mA"], dtype=float)
    target_fasc_ids = set(raw["cluster"]["target_ids"])

    # Saline bath: light-blue fill of the cuff lumen (radius = cuff inner edge),
    # drawn beneath the nerve so only the nerve-to-cuff annulus reads blue —
    # indicating the modelled conductive saline between nerve and electrode.
    if draw_electrodes and patches:
        from matplotlib.patches import Circle
        _R_in = float(patches[0]["R"]) * 1e6
        ax.add_patch(Circle((0.0, 0.0), _R_in, facecolor="#f0f7fd",
                            edgecolor="none", zorder=0.5))

    # Nerve outline (epineurium interior tinted a very light yellow).
    ax.fill(outline[:, 0], outline[:, 1],
              facecolor="#fcfae6", edgecolor="#555555", linewidth=0.5,
              zorder=1)
    # Fascicle fills (no edge here — outlines drawn on top of fibers below).
    for f in fascs:
        is_target = f["id"] in target_fasc_ids
        ax.fill(f["polygon"][:, 0], f["polygon"][:, 1],
                  facecolor=("#dff0e0" if is_target else "#eeeeee"),
                  edgecolor="none", zorder=2)

    # Fibres -- four-colour scheme.
    masks = [
        (~tgt & ~fired, FIB_OFF_SILENT, "off-target silent"),
        ( tgt & ~fired, FIB_TGT_SILENT, "target silent"),
        (~tgt &  fired, FIB_OFF_FIRED,  "off-target fired"),
        ( tgt &  fired, FIB_TGT_FIRED,  "target fired"),
    ]
    for m, col, _lab in masks:
        if np.any(m):
            ax.scatter(fiber_x[m], fiber_y[m], s=1.2, color=col,
                          edgecolors="#555555", linewidths=0.15, zorder=4, alpha=0.95)

    # Fascicle outlines on top of fiber dots — target in bold green.
    for f in fascs:
        is_target = f["id"] in target_fasc_ids
        poly = np.vstack([f["polygon"], f["polygon"][:1]])   # close ring
        ax.plot(poly[:, 0], poly[:, 1],
                color=("#1C1C1C" if is_target else "#aaaaaa"),
                linewidth=(0.9 if is_target else 0.4),
                zorder=6)

    # Electrodes -- cuff silicone ring + arc-shaped contacts.
    if draw_electrodes and patches and amps.size:
        _draw_cuff_arcs(ax, patches, amps)

    # Aesthetics — axis limits include cuff extent when electrodes are drawn.
    pad = 150.0
    if draw_electrodes and patches:
        R_um = float(patches[0]["R"]) * 1e6 + 200.0  # contact outer edge
        cuff_box = np.array([[-R_um, -R_um], [R_um, R_um]])
        xy = np.concatenate([outline, cuff_box])
    else:
        xy = outline
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
                  ha="left", va="top", fontsize=7, weight="bold",
                  color="#444444")
        ax.text(0.97, 0.97, f"SI = {si:.2f}", transform=ax.transAxes,
                  ha="right", va="top", fontsize=7,
                  color=("black" if si >= 0.90 else "#888888"),
                  weight="bold" if si >= 0.90 else "normal",
                  bbox=dict(boxstyle="round,pad=0.2",
                              facecolor=("#dff0e0" if si >= 0.90 else "white"),
                              edgecolor="#aaaaaa", linewidth=0.5))


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
    """Unrolled 4×3 cuff schematic — contacts coloured by polarity."""
    cell, col_labels, row_z_values = _contact_to_grid(contact_xyz)
    K = contact_xyz.shape[0]
    amps = amps_mA[:K]
    max_abs = max(0.5, float(np.max(np.abs(amps))))

    ncols, nrows = 4, 3
    sx, sy = 1.8, 1.5       # cell spacing: large enough circles never overlap
    dot_r  = 0.52           # circle radius in data units → sets physical size
    # Convert: at figsize=(3.0, 2.8) with ~5 data-unit range, ≈ 0.5in/unit
    # → dot_r=0.52 ≈ 0.26in → ~22pt radius → s≈(22*72/72)^2*π not needed,
    # we'll use ax.add_patch for exact sizing.
    from matplotlib.patches import Circle
    for c_idx in range(ncols):
        for r_idx in range(nrows):
            # Background placeholder circle (empty)
            ax.add_patch(Circle(
                (c_idx * sx, -(r_idx * sy)), dot_r,
                facecolor="white", edgecolor="#CCCCCC",
                linewidth=0.5, zorder=1,
            ))

    for k in range(K):
        c_idx, r_idx = int(cell[k, 0]), int(cell[k, 1])
        xp, yp = c_idx * sx, -(r_idx * sy)
        amp = float(amps[k])
        if amp < -1e-9:
            fc, ec = ELEC_CATHODE, "#1A5276"
        elif amp > 1e-9:
            fc, ec = ELEC_ANODE, "#873600"
        else:
            fc, ec = "#F5F5F5", "#CCCCCC"
        ax.add_patch(Circle(
            (xp, yp), dot_r,
            facecolor=fc, edgecolor=ec,
            linewidth=0.5, alpha=0.92, zorder=2,
        ))
        if abs(amp) > 1e-9:
            ax.text(xp, yp, f"{amp:+.2f}",
                      ha="center", va="center", fontsize=FS_SM,
                      color="white", weight="bold", zorder=3)

    for c_idx, label in enumerate(col_labels):
        ax.text(c_idx * sx, 0.9, label,
                  ha="center", va="center", fontsize=FS,
                  color=PALETTE["grey"], weight="semibold")

    row_pretty = ["+z", "0", "−z"]
    for r_idx, label in enumerate(row_pretty):
        ax.text(-0.7, -(r_idx * sy), label,
                  ha="right", va="center", fontsize=FS_SM,
                  color=PALETTE["lgrey"])

    ax.set_xlim(-1.4, (ncols - 1) * sx + 1.2)
    ax.set_ylim(-(nrows - 1) * sy - 0.9, 1.5)
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
    ax.plot(t, a, color=PALETTE["grey"], lw=0.9, zorder=3)
    ax.fill_between(t, a, 0, alpha=0.15, color=PALETTE["grey"],
                       linewidth=0, zorder=2)
    ax.axhline(0, color=PALETTE["lgrey"], lw=0.5, zorder=1)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("amplitude\n(× amp$_k$)")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(p2_amp - 0.15, p1_amp + 0.15)


def _make_xsection_legend_ax(ax) -> None:
    """Draw the legend reference card onto ax."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

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
    line_dy = 0.085
    section_dy = 0.04
    for header, entries in rows:
        ax.text(0.02, y, header, transform=ax.transAxes,
                  ha="left", va="top", fontsize=FS, weight="semibold",
                  color=PALETTE["grey"])
        y -= line_dy
        for label, col, marker, edge in entries:
            ax.scatter(0.05, y, transform=ax.transAxes,
                          s=40, marker=marker, facecolor=col,
                          edgecolor=(edge if edge is not None else "none"),
                          linewidth=0.5, zorder=3)
            ax.text(0.10, y, label, transform=ax.transAxes,
                      ha="left", va="center", fontsize=FS_SM,
                      color="#333333")
            y -= line_dy
        y -= section_dy

    ax.text(0.02, y, "SI badge", transform=ax.transAxes,
              ha="left", va="top", fontsize=FS, weight="semibold",
              color=PALETTE["grey"])
    y -= line_dy
    for label, fc in [("SI ≥ 0.90", "#dff0e0"), ("SI < 0.90", "white")]:
        ax.text(0.05, y, "SI = X.XX", transform=ax.transAxes,
                  ha="left", va="center", fontsize=FS_SM,
                  color="#333333",
                  bbox=dict(boxstyle="round,pad=0.2", facecolor=fc,
                              edgecolor=PALETTE["lgrey"], linewidth=0.5))
        ax.text(0.28, y, label, transform=ax.transAxes,
                  ha="left", va="center", fontsize=FS_SM, color="#333333")
        y -= line_dy


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
        outline, fascs, contact_xyz, patches = _load_geometry(r["sample"])
        amps = np.asarray(raw["rect"]["amps_mA"], dtype=float)
        sample_dir = base_dir / r["sample"]
        sample_dir.mkdir(parents=True, exist_ok=True)

        # 1. Pulse trace.
        fig, ax = plt.subplots(figsize=(3.5, 2.2))
        _draw_pulse_trace(ax, raw["pulse_shape"],
                              float(raw["pulse_pw_ms"]),
                              float(raw["pulse_asym_ratio"]))
        fig.tight_layout()
        written.extend(_savefig_sample(fig, sample_dir, "pulse"))

        # 2. Cross-section.
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        _draw_xsection(ax, r["sample"], raw, draw_electrodes=True,
                          draw_label=False)
        si = float(raw["rect"]["achievable_si"])
        ax.set_title(_short(r["sample"]), fontsize=8, weight="bold",
                        color=PALETTE["grey"], loc="left", pad=6)
        ax.text(0.98, 1.02, f"SI = {si:.2f}",
                  transform=ax.transAxes, ha="right", va="bottom",
                  fontsize=7,
                  color=("black" if si >= 0.90 else PALETTE["grey"]),
                  weight="bold" if si >= 0.90 else "normal",
                  bbox=dict(boxstyle="round,pad=0.2",
                              facecolor=("#dff0e0" if si >= 0.90
                                          else "white"),
                              edgecolor=PALETTE["grey"], linewidth=0.5))
        fig.tight_layout()
        written.extend(_savefig_sample(fig, sample_dir, "cross_section"))

        # 3. Cuff grid.
        fig, ax = plt.subplots(figsize=(3.0, 2.8))
        _draw_cuff_grid(ax, contact_xyz, amps)
        ax.set_title("Cuff electrode pattern (mA)", fontsize=8,
                        weight="bold", color=PALETTE["grey"],
                        loc="center", pad=6)
        fig.tight_layout()
        written.extend(_savefig_sample(fig, sample_dir, "cuff_pattern"))

        # 4. Legend.
        fig_leg, ax_leg = plt.subplots(figsize=(5.0, 3.5))
        _make_xsection_legend_ax(ax_leg)
        fig_leg.tight_layout()
        written.extend(_savefig_sample(fig_leg, sample_dir, "legend"))
    return written


def _xsection_gallery(samples: list[tuple[str, dict]],
                        nrows: int, ncols: int,
                        name: str) -> list[Path]:
    fig, axes = plt.subplots(nrows, ncols,
                                figsize=(3.2 * ncols, 3.2 * nrows),
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
                 markersize=6, label="target fired"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=FIB_TGT_SILENT,
                 markersize=6, markeredgecolor=PALETTE["grey"],
                 label="target silent"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=FIB_OFF_FIRED,
                 markersize=6, label="off-target fired"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=FIB_OFF_SILENT,
                 markersize=6, markeredgecolor=PALETTE["grey"],
                 label="off-target silent"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ELEC_CATHODE,
                 markersize=8, markeredgecolor=PALETTE["grey"],
                 label="cathode (-)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ELEC_ANODE,
                 markersize=8, markeredgecolor=PALETTE["grey"],
                 label="anode (+)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6,
                  frameon=False, fontsize=7,
                  bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _savefig(fig, name)


def make_xsection_galleries(rows: list[dict]) -> list[Path]:
    """Two gallery figures: one for swine samples, one for human samples."""
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

    paths: list[Path] = []
    for sp in ("swine", "human"):
        s = samples_by_species[sp]
        if not s:
            continue
        s.sort(key=lambda t: -float(t[1]["rect"]["achievable_si"]))
        n = len(s)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        paths.extend(_xsection_gallery(s, nrows, ncols,
                                        f"fig_duke_xsections_{sp}"))
    return paths


# ── Public callables ────────────────────────────────────────────────────────

def make_per_sample_fig(rows: list[dict]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    _per_sample_bars(ax, rows)
    fig.tight_layout()
    return _savefig(fig, "fig_duke_per_sample")


def make_summary_fig(rows: list[dict]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    _per_sample_bars(ax, rows)
    fig.tight_layout()
    return _savefig(fig, "fig_duke_selectivity_summary")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    rows = _load_rows()
    print(f"[load] {len(rows)} samples")
    all_paths: list[Path] = []
    for fn in (make_per_sample_fig, make_summary_fig,
               make_species_method_fig, make_metrics_fig,
               make_metrics_violin_fig):
        all_paths.extend(fn(rows))
    all_paths.extend(make_xsection_galleries(rows))
    all_paths.extend(make_per_sample_xsection(rows))
    for p in all_paths:
        print(f"  -> {p}")


if __name__ == "__main__":
    main()
