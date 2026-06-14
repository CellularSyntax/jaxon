"""Figure 3: Duke-cohort selectivity — anatomy, performance, stimulus patterns,
and sparse fibre-sampling analysis.

2×2 major panel layout, Nature Medicine style:
  a — Swine + human nerve cross-sections with unrolled cuff patterns
      (anatomy cells enlarged; fibre legend placed outside nerve image)
  b — Selectivity performance metrics (SI / on-target / off-target / loss)
  c — Stimulus amplitude by contact position (target-normalised phi_bin × z_row)
  d — Sparse fibre-sampling:
        left   = SI violins per strategy (swine + human)
        centre = SI_dense − SI_sparse gap, swine
        right  = SI_dense − SI_sparse gap, human (own y-range)

Run from project root:
    python -m experiments_v2.figures_duke_main
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import io

import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from experiments_v2.figures_duke import (
    _draw_xsection,
    _draw_cuff_grid,
    _load_geometry,
)
from experiments_v2.figures_sparse_sampling import (
    _load as _load_sparse,
    STRAT_LAYOUT,
    X_LABELS,
)

ROOT     = Path(__file__).resolve().parent.parent
SWEEP    = ROOT / "outputs" / "duke_sweeps"
OUT_DIR  = ROOT / "manuscript" / "figures" / "main"
ICON_DIR = ROOT / "manuscript" / "figures" / "assets" / "electrode_config_icons"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "swine": "#D6604D",
    "human": "#4393C3",
    "grey":  "#555555",
    "lgrey": "#AAAAAA",
    "dgrey": "#333333",
}
SP_MARKER = {"swine": "s", "human": "o"}

# Validity filter shared by panels b and d: seeds whose dense optimisation failed
# to converge (dense SI below this) are excluded — they are neither a fair
# performance sample (panel b) nor a valid ceiling for the transfer penalty (d).
_DENSE_MIN = 0.5

FS    = 8
FS_SM = 7

mpl.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size":         FS,
    "axes.labelsize":    FS,
    "axes.titlesize":    FS_SM,
    "xtick.labelsize":   FS_SM,
    "ytick.labelsize":   FS_SM,
    "legend.fontsize":   FS_SM,
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
    "savefig.dpi":       600,
    "savefig.bbox":      "tight",
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "pdf.fonttype":      42,
})

# ── Shared helpers ─────────────────────────────────────────────────────────────
def _species_of(sample: str) -> str:
    return "human" if sample.startswith("human") else "swine"


def _panel_heading(ax: plt.Axes, letter: str, title: str,
                   dx: float = -0.18, dy: float = 1.14) -> None:
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="bottom", ha="left", clip_on=False)
    ax.text(0.0, dy, title, transform=ax.transAxes,
            fontsize=11, va="bottom", ha="left", clip_on=False)


def _violin(ax: plt.Axes, data: np.ndarray, pos: float,
            col: str, width: float = 0.36, fill: bool = True) -> None:
    if data.size < 3:
        return
    vp = ax.violinplot([data], positions=[pos], widths=width,
                       showmedians=True, showextrema=False)
    for body in vp["bodies"]:
        if fill:
            body.set_facecolor(col); body.set_alpha(0.42)
        else:                                  # hollow outline (in-sample)
            body.set_facecolor("none"); body.set_alpha(0.9)
        body.set_edgecolor(col); body.set_linewidth(0.8 if not fill else 0.4)
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(1.4)


# ── Data: duke seeds ──────────────────────────────────────────────────────────
def _load_all_seeds() -> list[dict]:
    rows = []
    for sd in sorted(SWEEP.iterdir()):
        if not sd.is_dir():
            continue
        sample = sd.name
        for jp in sorted(sd.glob("data_seed_*.json")):
            try:
                raw = json.loads(jp.read_text())
            except Exception:
                continue
            rect = raw.get("rect") or {}
            fire = rect.get("firing") or {}
            si = float(rect.get("achievable_si",
                                  abs(rect.get("final_si", float("nan")))))
            rows.append(dict(
                sample=sample,
                species=_species_of(sample),
                si=si,
                frac_tgt=float(fire.get("frac_fired_target",  float("nan"))),
                frac_off=float(fire.get("frac_fired_nontarget", float("nan"))),
                loss=float(rect.get("final_loss", float("nan"))),
                raw_path=str(jp),
            ))
    return rows


def _best_seed_raw(rows: list[dict], species: str) -> tuple[str, int, dict] | None:
    sp = [r for r in rows if r["species"] == species and np.isfinite(r["si"])]
    if not sp:
        return None
    best = max(sp, key=lambda r: r["si"])
    try:
        seed_num = int(Path(best["raw_path"]).stem.split("_")[-1])
    except ValueError:
        seed_num = 0
    return best["sample"], seed_num, json.loads(Path(best["raw_path"]).read_text())


def _worst_seed_raw(rows: list[dict], species: str) -> tuple[str, int, dict] | None:
    sp = [r for r in rows if r["species"] == species and np.isfinite(r["si"])]
    if not sp:
        return None
    worst = min(sp, key=lambda r: r["si"])
    try:
        seed_num = int(Path(worst["raw_path"]).stem.split("_")[-1])
    except ValueError:
        seed_num = 0
    return worst["sample"], seed_num, json.loads(Path(worst["raw_path"]).read_text())


# ── Data: contact amplitudes ──────────────────────────────────────────────────
def _phi_bin(phi_rad: float, target_center_deg: float) -> int:
    rel = (np.degrees(phi_rad) - target_center_deg + 360.0) % 360.0
    return int((rel + 45.0) // 90.0) % 4


def _load_contact_amplitudes() -> list[dict]:
    rows = []
    for sd in sorted(SWEEP.iterdir()):
        if not sd.is_dir():
            continue
        sample = sd.name
        species = _species_of(sample)
        try:
            _, _, contact_xyz, patches = _load_geometry(sample)
        except Exception:
            continue
        if not patches:
            continue
        # patches["z"] is in SI meters; build uniq_z in the same unit
        z_from_patches = np.array([float(p["z"]) for p in patches])
        uniq_z = np.sort(np.unique(np.round(z_from_patches, 9)))

        def _z_row(z: float, _uniq=uniq_z) -> int:
            """Nearest-neighbour z-level mapping; top (+z) → 0, bot (-z) → 2."""
            if _uniq.size == 0:
                return 1
            idx = int(np.argmin(np.abs(_uniq - z)))
            if _uniq.size == 3:
                return 2 - idx   # ascending sort: [bot, mid, top] → [2,1,0]
            return min(idx * 3 // _uniq.size, 2)

        for jp in sorted(sd.glob("data_seed_*.json")):
            try:
                raw = json.loads(jp.read_text())
            except Exception:
                continue
            rect = raw.get("rect") or {}
            amps = np.asarray(rect.get("amps_mA", []), dtype=float)
            cluster = raw.get("cluster") or {}
            tc = ((float(cluster.get("window_start_deg", 0.0)) +
                   float(cluster.get("window_end_deg", 0.0))) / 2.0)
            for k, p in enumerate(patches):
                if k >= len(amps):
                    break
                rows.append(dict(
                    species=species,
                    phi_bin=_phi_bin(float(p["phi"]), tc),
                    z_row=_z_row(float(p["z"])),
                    amp=float(amps[k]),
                ))
    return rows


# ── Scale bar ────────────────────────────────────────────────────────────────
def _add_scale_bar(ax: plt.Axes, length_um: float = 500.0,
                   pad_frac: float = 0.05) -> None:
    """Horizontal scale bar at bottom-right; axes units are µm."""
    xl, xr = ax.get_xlim()
    yb, yt = ax.get_ylim()
    xspan = xr - xl
    yspan = yt - yb

    x1 = xr - pad_frac * xspan
    x0 = x1 - length_um
    y  = yb + pad_frac * yspan

    ax.plot([x0, x1], [y, y], color="black", lw=1.2,
            solid_capstyle="butt", clip_on=False, zorder=10)
    ax.text((x0 + x1) / 2, y - 0.02 * yspan,
            f"{int(length_um)} µm",
            ha="center", va="top", fontsize=FS_SM - 1,
            color="black", clip_on=False, zorder=10)


# ── Compass rose ──────────────────────────────────────────────────────────────
def _draw_compass(ax: plt.Axes) -> None:
    """Inset N/E/S/W compass rose at top-left of ax."""
    ins = ax.inset_axes([-0.06, -0.10, 0.22, 0.22])
    ins.set_xlim(-1.6, 1.6)
    ins.set_ylim(-1.6, 1.6)
    ins.set_aspect("equal")
    ins.axis("off")

    col = PALETTE["dgrey"]
    for dx, dy, lbl, ha, va in [
        ( 0,  1, "N", "center", "bottom"),
        ( 1,  0, "E", "left",   "center"),
        ( 0, -1, "S", "center", "top"),
        (-1,  0, "W", "right",  "center"),
    ]:
        ins.annotate("", xy=(dx * 0.80, dy * 0.80), xytext=(dx * 0.12, dy * 0.12),
                     arrowprops=dict(arrowstyle="-|>", color=col,
                                     lw=0.7, mutation_scale=7))
        ins.text(dx * 1.25, dy * 1.25, lbl,
                 ha=ha, va=va, fontsize=FS_SM, fontweight="bold", color=col)
    ins.plot(0, 0, "o", color=col, ms=2, zorder=5)


# ── Panel a: anatomy (swine + human, 2 rows; each row = xsec + cuff) ─────────
def _draw_example(ax_xsec: plt.Axes, ax_cuff: plt.Axes,
                  sample: str, seed_num: int, raw: dict, species: str,
                  case: str = "best") -> None:
    col = PALETTE[species]
    try:
        _draw_xsection(ax_xsec, sample, raw, draw_electrodes=True, draw_label=False)
        _add_scale_bar(ax_xsec)
    except Exception:
        ax_xsec.text(0.5, 0.5, "geometry missing", ha="center", va="center",
                     transform=ax_xsec.transAxes, fontsize=FS_SM,
                     color=PALETTE["lgrey"])

    rect   = raw.get("rect") or {}
    si_val = float(rect.get("achievable_si", rect.get("final_si", float("nan"))))
    si_txt = f"SI = {si_val:.3f}" if np.isfinite(si_val) else "SI = n/a"

    # Strip species prefix from sample ID for a compact top label
    sample_id = sample
    for pfx in (species, "swine", "human"):
        if sample_id.lower().startswith(pfx):
            sample_id = sample_id[len(pfx):].lstrip("_- ")
            break

    # Top centre: "Swine (sample-id · sN)" in species colour
    ax_xsec.text(0.50, 1.03,
                 f"{species.capitalize()} ({sample_id} · s{seed_num})",
                 transform=ax_xsec.transAxes,
                 ha="center", va="bottom", fontsize=FS_SM,
                 color=col, weight="bold", clip_on=False)

    # Top-right: SI value, clean black
    ax_xsec.text(0.97, 0.97, si_txt,
                 transform=ax_xsec.transAxes,
                 ha="right", va="top", fontsize=FS_SM,
                 color="black", clip_on=False)

    # Bottom centre: "best" / "worst" in black italic
    ax_xsec.text(0.50, -0.04, case,
                 transform=ax_xsec.transAxes,
                 ha="center", va="top", fontsize=FS_SM,
                 color="black", style="italic", clip_on=False)

    try:
        _, _, contact_xyz, _ = _load_geometry(sample)
        amps = np.asarray(raw["rect"]["amps_mA"], dtype=float)
        _draw_cuff_grid(ax_cuff, contact_xyz, amps)
        ax_cuff.set_title("contact amplitudes", fontsize=FS_SM, pad=2, fontweight="bold")
    except Exception:
        ax_cuff.axis("off")
        ax_cuff.text(0.5, 0.5, "cuff missing", ha="center", va="center",
                     transform=ax_cuff.transAxes, fontsize=FS_SM,
                     color=PALETTE["lgrey"])


def _panel_a(gs, fig, rows: list[dict]) -> plt.Axes:
    """2-row × 4-col anatomy panel.

    Row 0 (tall):  sw-best | sw-worst | hu-best | hu-worst  (nerve xsec)
    Row 1 (short): corresponding activation maps (cuff grids)
    """
    from experiments_v2.figures_duke import (
        FIB_TGT_FIRED, FIB_TGT_SILENT, FIB_OFF_FIRED, FIB_OFF_SILENT,
    )
    gs_inner = gridspec.GridSpecFromSubplotSpec(
        2, 4, subplot_spec=gs,
        wspace=0.08, hspace=0.30,
        height_ratios=[1.8, 1.0],
    )

    examples = [
        ("swine", "best",  _best_seed_raw(rows,  "swine")),
        ("swine", "worst", _worst_seed_raw(rows, "swine")),
        ("human", "best",  _best_seed_raw(rows,  "human")),
        ("human", "worst", _worst_seed_raw(rows, "human")),
    ]

    first_ax = last_ax = None
    for ci, (species, case, res) in enumerate(examples):
        ax_x = fig.add_subplot(gs_inner[0, ci])
        ax_c = fig.add_subplot(gs_inner[1, ci])
        if ci == 0:
            first_ax = ax_x
        if ci == 3:
            last_ax = ax_x
        if res:
            _draw_example(ax_x, ax_c, res[0], res[1], res[2], species, case=case)
        else:
            ax_x.text(0.5, 0.5, f"no {species} data",
                      ha="center", va="center",
                      transform=ax_x.transAxes, fontsize=FS_SM,
                      color=PALETTE["lgrey"])

    last_ax.legend(
        handles=[
            mpatches.Patch(color=FIB_TGT_FIRED,  label="target, fired"),
            mpatches.Patch(color=FIB_TGT_SILENT, label="target, silent"),
            mpatches.Patch(color=FIB_OFF_FIRED,  label="off-tgt, fired"),
            mpatches.Patch(color=FIB_OFF_SILENT, label="off-tgt, silent"),
        ],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.24),
        ncol=4,
        fontsize=FS, frameon=False,
        handlelength=0.8, borderpad=0.0, labelspacing=0.18, columnspacing=0.6,
    )
    _draw_compass(first_ax)
    return first_ax


# ── Panel b: performance metrics 2×2 ─────────────────────────────────────────
def _metric_bar(ax: plt.Axes, rows: list[dict], field: str, ylabel: str,
                scale: float = 1.0, ylim: tuple | None = None,
                log: bool = False) -> None:
    rng = np.random.default_rng(0)
    for xi, sp in enumerate(["swine", "human"]):
        vals = np.array([r[field] * scale for r in rows
                         if r["species"] == sp
                         and np.isfinite(r.get(field, float("nan")))])
        if not vals.size:
            continue
        med = float(np.median(vals))
        q25 = float(np.quantile(vals, 0.25))
        q75 = float(np.quantile(vals, 0.75))
        col = PALETTE[sp]
        ax.bar(xi, med, width=0.50, color=col, alpha=0.38, edgecolor="none", zorder=2)
        ax.errorbar(xi, med, yerr=[[med - q25], [q75 - med]], fmt="none",
                    ecolor=PALETTE["dgrey"], elinewidth=1.1,
                    capsize=3.0, capthick=1.0, zorder=5)
        jit = rng.uniform(-0.13, 0.13, vals.size)
        ax.scatter(xi + jit, vals, s=8, color=col, marker=SP_MARKER[sp],
                   edgecolors="white", linewidth=0.3, alpha=0.85, zorder=4)
    n_sw = sum(1 for r in rows if r["species"] == "swine"
               and np.isfinite(r.get(field, float("nan"))))
    n_hu = sum(1 for r in rows if r["species"] == "human"
               and np.isfinite(r.get(field, float("nan"))))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["swine", "human"], fontsize=FS_SM)
    ax.set_ylabel(ylabel, fontsize=FS_SM)
    ax.set_xlim(-0.55, 1.55)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if log:
        ax.set_yscale("log")


def _panel_b(gs, fig, rows: list[dict]) -> plt.Axes:
    # Same validity filter as panel d: drop dense seeds that did not converge
    # (dense SI < _DENSE_MIN) so the two panels can't drift.
    rows = [r for r in rows
            if np.isfinite(r.get("si", np.nan)) and r["si"] >= _DENSE_MIN]
    gs_inner = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs, wspace=0.42, hspace=0.48,
    )
    axes = [[fig.add_subplot(gs_inner[r, c]) for c in range(2)] for r in range(2)]
    _metric_bar(axes[0][0], rows, "si",       "SI",          ylim=(0, 1.10))
    _metric_bar(axes[0][1], rows, "frac_tgt", "on-tgt (%)",  scale=100.0, ylim=(0, 110))
    _metric_bar(axes[1][0], rows, "frac_off", "off-tgt (%)", scale=100.0)
    _metric_bar(axes[1][1], rows, "loss",     "loss",        log=True)
    return axes[0][0]


# ── Electrode-config icon helpers ─────────────────────────────────────────────
_ELEC_ICON_CACHE: dict[str, np.ndarray] = {}

def _elec_icon(filename: str) -> np.ndarray:
    if filename in _ELEC_ICON_CACHE:
        return _ELEC_ICON_CACHE[filename]
    arr = plt.imread(str(ICON_DIR / filename))
    _ELEC_ICON_CACHE[filename] = arr
    return arr


def _place_elec_icons(ax: plt.Axes, xpos_files: list[tuple[float, str]],
                      zoom: float = 0.06, y_offset_pt: float = 0.0) -> None:
    for xpos, filename in xpos_files:
        arr = _elec_icon(filename)
        ab = AnnotationBbox(
            OffsetImage(arr, zoom=zoom),
            xy=(xpos, 1), xycoords=("data", "axes fraction"),
            xybox=(0, y_offset_pt), boxcoords="offset points",
            box_alignment=(0.5, 0.0), frameon=False, clip_on=False,
        )
        ax.add_artist(ab)


def _place_elec_icons_at_whisker(
        ax: plt.Axes,
        xpos_files_y: list[tuple[float, str, float]],
        zoom: float = 0.06, y_offset_pt: float = 3.0) -> None:
    """Place icons in data coordinates, anchored at the upper-whisker height."""
    for xpos, filename, y_data in xpos_files_y:
        arr = _elec_icon(filename)
        ab = AnnotationBbox(
            OffsetImage(arr, zoom=zoom),
            xy=(xpos, y_data), xycoords="data",
            xybox=(0, y_offset_pt), boxcoords="offset points",
            box_alignment=(0.5, 0.0), frameon=False, clip_on=False,
        )
        ax.add_artist(ab)


def _whisker_top(data: np.ndarray) -> float:
    """Upper whisker limit (matplotlib default: max ≤ Q3 + 1.5·IQR)."""
    q1, q3 = np.percentile(data, [25, 75])
    fence = q3 + 1.5 * (q3 - q1)
    above = data[data <= fence]
    return float(above.max()) if above.size else float(q3)


# ── Panel c: amplitude distributions by phi_bin and z_row ────────────────────
def _panel_c(gs, fig, crows: list[dict]) -> plt.Axes:
    gs_inner = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs, wspace=0.24,
    )
    ax_phi = fig.add_subplot(gs_inner[0])
    ax_z   = fig.add_subplot(gs_inner[1])

    PHI_LABELS  = ["N", "E", "S", "W"]   # target-normalised: N = toward target
    Z_LABELS    = ["top\n(+z)", "mid", "bot\n(−z)"]
    PHI_ICONS   = ["config_N.png", "config_E.png", "config_S.png", "config_W.png"]
    Z_ICONS     = ["config_top.png", "config_mid.png", "config_bot.png"]
    ACTIVE_THR  = 0.01
    rng = np.random.default_rng(2)

    # upper[ax] maps xi → max upper-whisker across both species at that x-bin
    upper: dict[plt.Axes, dict[int, float]] = {ax_phi: {}, ax_z: {}}

    for ax, key, n_bins, tick_labels in [
        (ax_phi, "phi_bin", 4, PHI_LABELS),
        (ax_z,   "z_row",   3, Z_LABELS),
    ]:
        for xi in range(n_bins):
            for sp, xoff in [("swine", -0.22), ("human", 0.22)]:
                all_v  = np.array([r["amp"] for r in crows
                                   if r["species"] == sp and r[key] == xi])
                active = np.abs(all_v[np.abs(all_v) > ACTIVE_THR])
                if active.size < 3:
                    continue
                col  = PALETTE[sp]
                xpos = xi + xoff
                ax.boxplot(
                    [active], positions=[xpos], widths=0.32,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.4),
                    boxprops=dict(facecolor=col, alpha=0.35,
                                  edgecolor=col, linewidth=0.5),
                    whiskerprops=dict(color=col, linewidth=0.6),
                    capprops=dict(color=col, linewidth=0.6),
                )
                jit = rng.uniform(-0.07, 0.07, active.size)
                ax.scatter(xpos + jit, active, s=3, color=col,
                           alpha=0.38, edgecolors="none", zorder=3)
                # track max Q3 (top of box) across species for this x-bin
                q3 = float(np.percentile(active, 75))
                upper[ax][xi] = max(upper[ax].get(xi, 0.0), q3)

        ax.set_ylim(bottom=0)
        ax.set_xticks(range(n_bins))
        ax.set_xticklabels(tick_labels, fontsize=FS_SM)
        ax.set_ylabel("|amplitude| (mA)", fontsize=FS_SM)
        ax.set_xlim(-0.55, n_bins - 0.45)

    ax_phi.set_xlabel("contact position (N = toward target)", fontsize=FS_SM)

    leg = [mpatches.Patch(color=PALETTE["swine"], alpha=0.7, label="swine"),
           mpatches.Patch(color=PALETTE["human"], alpha=0.7, label="human")]
    ax_phi.legend(handles=leg, frameon=False, fontsize=FS_SM,
                  loc="upper right", handlelength=0.8)

    _place_elec_icons_at_whisker(ax_phi,
        [(xi, PHI_ICONS[xi], upper[ax_phi].get(xi, 0.0)) for xi in range(4)], zoom=0.05)
    _place_elec_icons_at_whisker(ax_z,
        [(xi, Z_ICONS[xi],   upper[ax_z].get(xi,  0.0)) for xi in range(3)], zoom=0.05)
    return ax_phi


# ── Panel d: sparse sampling (SI violins + two gap plots) ─────────────────────
# ── Sparsity icon helpers ─────────────────────────────────────────────────────
_ICON_COL_FILL = "#F5F3F7"   # warm off-white fascicle fill
_ICON_COL_EDGE = "#3D3550"   # (unused — border now drawn separately)
_ICON_COL_BG   = "#E8E8E8"   # unselected fibers — matches FIB_OFF_SILENT
_ICON_COL_SEL  = "#1A7340"   # selected fibers — matches FIB_TGT_FIRED
_ICON_COL_CENT = "#C0392B"   # centroid cross

def _icon_xy(n: int = 30, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts: list = []
    while len(pts) < n:
        xy = rng.uniform(-1, 1, (n * 4, 2))
        xy = xy[np.hypot(xy[:, 0], xy[:, 1]) < 0.88]
        pts.extend(xy.tolist())
    return np.array(pts[:n])

_ICON_XY = _icon_xy()

def _icon_sel(strat_key: str, n_per) -> np.ndarray:
    xy = _ICON_XY
    if strat_key == "dense":
        return np.arange(len(xy))
    if strat_key == "centroid":
        return np.array([int(np.argmin(np.hypot(xy[:, 0], xy[:, 1])))])
    n = int(n_per) if n_per is not None else 1
    return np.random.default_rng(7).choice(len(xy), size=min(n, len(xy)), replace=False)

_ICON_CACHE: dict[str, np.ndarray] = {}

def _sparsity_icon(strat_key: str, n_per, dpi: int = 180) -> np.ndarray:
    cache_key = f"{strat_key}_{n_per}"
    if cache_key in _ICON_CACHE:
        return _ICON_CACHE[cache_key]
    xy = _ICON_XY
    sel = _icon_sel(strat_key, n_per)
    mask = np.ones(len(xy), dtype=bool); mask[sel] = False

    fig_ic, ax_ic = plt.subplots(1, 1, figsize=(0.45, 0.45))
    fig_ic.subplots_adjust(0, 0, 1, 1)
    ax_ic.set_aspect("equal"); ax_ic.set_xlim(-1.15, 1.15); ax_ic.set_ylim(-1.15, 1.15)
    ax_ic.axis("off")
    ax_ic.add_patch(mpatches.Circle((0, 0), 1.0, facecolor=_ICON_COL_FILL,
                                     edgecolor="none", zorder=1))
    ax_ic.add_patch(mpatches.Circle((0, 0), 1.0, facecolor="none",
                                     edgecolor="#1C1C1C", linewidth=0.9, zorder=6))
    ax_ic.scatter(xy[mask, 0], xy[mask, 1], s=6,  color=_ICON_COL_BG,
                  edgecolors="#555555", linewidths=0.15, zorder=2)
    ax_ic.scatter(xy[sel,  0], xy[sel,  1], s=28, color=_ICON_COL_SEL,
                  edgecolors="#555555", linewidths=0.15, zorder=4)
    if strat_key == "centroid":
        ax_ic.plot(0, 0, "+", color=_ICON_COL_CENT, ms=6, mew=1.0, zorder=5)

    buf = io.BytesIO()
    fig_ic.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                   pad_inches=0.01, transparent=True)
    plt.close(fig_ic)
    buf.seek(0)
    arr = plt.imread(buf)
    _ICON_CACHE[cache_key] = arr
    return arr


def _place_icons(ax: plt.Axes, xpos_strats: list[tuple[float, str, object]],
                 zoom: float = 0.28, y_offset_pt: float = 6.0) -> None:
    """Place sparsity icons just above the top of the axes at given data-x positions."""
    for xpos, strat_key, n_per in xpos_strats:
        arr = _sparsity_icon(strat_key, n_per)
        ab = AnnotationBbox(
            OffsetImage(arr, zoom=zoom),
            xy=(xpos, 1), xycoords=("data", "axes fraction"),
            xybox=(0, y_offset_pt), boxcoords="offset points",
            box_alignment=(0.5, 0.0), frameon=False, clip_on=False,
        )
        ax.add_artist(ab)


# Panel d order: 10/fasc → 3/fasc → 1/fasc → centroid (dense shown as a band).
# Tuple: (data_xi, strat_key, n_per_fasc, label)
_D_SPARSE = [
    (4, "random",    10, "10/fasc"),
    (3, "random",     3, "3/fasc"),
    (2, "random",     1, "1/fasc"),
    (1, "centroid",   1, "centroid"),
]


def _ps_mean(sparse_rows, species, valfn) -> np.ndarray:
    """Aggregate seed-rows to one value per nerve (avoids pseudoreplication from
    seeds sharing a geometry)."""
    by = defaultdict(list)
    for r in sparse_rows:
        if r["species"] != species:
            continue
        v = valfn(r)
        if v is not None and np.isfinite(v):
            by[r["sample"]].append(v)
    return np.array([float(np.mean(v)) for v in by.values()])


def _boot_ci(x, n: int = 4000, seed: int = 0):
    if x.size < 2:
        return (float(x.mean()) if x.size else np.nan,) * 2
    rng = np.random.default_rng(seed)
    bs = [np.mean(rng.choice(x, x.size, replace=True)) for _ in range(n)]
    return tuple(np.percentile(bs, [2.5, 97.5]))


def _holm(pvals):
    p = [pp if np.isfinite(pp) else 1.0 for pp in pvals]
    m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min((m - k) * p[i], 1.0)); adj[i] = run
    return adj


def _pfmt(p):
    """Compact p-value label (journals often prefer values over stars)."""
    if not np.isfinite(p):
        return ""
    return "p<0.001" if p < 1e-3 else f"p={p:.3f}"


def _mean_transfer(r) -> float:
    v = [r["si_transfer"][xi] for xi, *_ in _D_SPARSE
         if np.isfinite(r["si_transfer"].get(xi, np.nan))]
    return float(np.mean(v)) if v else np.nan


def _panel_d(gs, fig, sparse_rows: list[dict]) -> plt.Axes:
    """Sparse fiber-sampling, per nerve: dense ceiling vs sparse-optimised
    transfer SI (swine | human), and the mechanism — penalty vs target-fascicle
    size.  Seeds whose dense optimisation failed (dense_si < _DENSE_MIN) are
    excluded (no valid ceiling).  Stars: transfer vs dense (two-sided per-nerve
    Wilcoxon, Holm-corrected across the 8 cells)."""
    rows = [r for r in sparse_rows
            if np.isfinite(r.get("dense_si", np.nan)) and r["dense_si"] >= _DENSE_MIN]

    gs_inner = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs, wspace=0.42, width_ratios=[1.0, 1.0, 1.25])
    ax_sw = fig.add_subplot(gs_inner[0])
    ax_hu = fig.add_subplot(gs_inner[1], sharey=ax_sw)
    ax_me = fig.add_subplot(gs_inner[2])

    # Holm-adjusted transfer-vs-dense p-values across both species × 4 strategies.
    def _pen(xi):
        return lambda r: (r["dense_si"] - r["si_transfer"][xi]
                          if np.isfinite(r["si_transfer"].get(xi, np.nan)) else None)
    cells = [(sp, xi) for sp in ("swine", "human") for xi, *_ in _D_SPARSE]
    raw = []
    for sp, xi in cells:
        pen = _ps_mean(rows, sp, _pen(xi))
        raw.append(stats.wilcoxon(pen, alternative="two-sided").pvalue
                   if pen.size >= 6 else np.nan)
    padj = {c: a for c, a in zip(cells, _holm(raw))}

    rng = np.random.default_rng(42)
    for ax, sp in [(ax_sw, "swine"), (ax_hu, "human")]:
        col = PALETTE[sp]
        dense = _ps_mean(rows, sp, lambda r: r["dense_si"])
        dlo, dhi = _boot_ci(dense)
        ax.axhspan(dlo, dhi, color=PALETTE["lgrey"], alpha=0.30, zorder=0)
        ax.axhline(float(dense.mean()), color=PALETTE["grey"], ls="--", lw=0.8, zorder=1)
        for gi, (xi, _, _, lbl) in enumerate(_D_SPARSE):
            tr = _ps_mean(rows, sp, lambda r, xi=xi: r["si_transfer"].get(xi, np.nan))
            if not tr.size:
                continue
            ax.scatter(gi + rng.uniform(-0.13, 0.13, tr.size), tr, s=8, color=col,
                       alpha=0.40, edgecolors="none", zorder=3)
            m = float(tr.mean()); lo, hi = _boot_ci(tr)
            ax.errorbar(gi, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=5, mfc=col,
                        mec="black", color="black", elinewidth=1.2, capsize=3, zorder=5)
            _pt = ax.text(gi, hi + 0.025, _pfmt(padj[(sp, xi)]), ha="center",
                          va="bottom", fontsize=FS_SM, color="#222222")
            _pt.set_zorder(60)
        ax.set_xticks(range(len(_D_SPARSE)))
        ax.set_xticklabels([t[3] for t in _D_SPARSE], fontsize=FS_SM)
        ax.set_xlim(-0.55, len(_D_SPARSE) - 0.45)
        ax.set_ylim(0, 1.20)
        ax.tick_params(axis="x", length=0)
        _place_icons(ax, [(float(gi), t[1], t[2]) for gi, t in enumerate(_D_SPARSE)],
                     zoom=0.24, y_offset_pt=-12)
        _t = ax.set_title(sp.capitalize(), fontsize=FS_SM + 1, color=col,
                          weight="bold", y=1.18)
        _t.set_zorder(1000)
    ax_sw.set_ylabel("SI (full nerve)")
    plt.setp(ax_hu.get_yticklabels(), visible=False)
    leg = [Line2D([], [], color=PALETTE["grey"], ls="--", label="dense-optimised"),
           Line2D([], [], marker="o", ls="none", mfc=PALETTE["dgrey"], mec="black",
                  label="sparse-optimised")]
    ax_sw.legend(handles=leg, frameon=False, fontsize=FS_SM - 1,
                 loc="lower left", handlelength=1.4, borderaxespad=0.3)

    # ── mechanism: deployment penalty vs target-fascicle size (per nerve) ──────
    byn = defaultdict(lambda: {"fpf": [], "pen": []}); spof = {}
    for r in rows:
        mt = _mean_transfer(r); fpf = r.get("fibers_per_target_fasc", np.nan)
        if np.isfinite(mt) and np.isfinite(fpf):
            spof[r["sample"]] = r["species"]
            byn[r["sample"]]["fpf"].append(fpf)
            byn[r["sample"]]["pen"].append(r["dense_si"] - mt)
    nf  = {s: float(np.mean(v["fpf"])) for s, v in byn.items()}
    npn = {s: float(np.mean(v["pen"])) for s, v in byn.items()}
    for sp in ("swine", "human"):
        ss = [s for s in nf if spof[s] == sp]
        ax_me.scatter([nf[s] for s in ss], [npn[s] for s in ss], s=14,
                      color=PALETTE[sp], alpha=0.6, edgecolors="none", label=sp.capitalize())
    allx = np.array([nf[s] for s in nf]); ally = np.array([npn[s] for s in nf])
    if allx.size >= 5:
        b = np.polyfit(np.log(allx), ally, 1)
        xx = np.geomspace(allx.min(), allx.max(), 60)
        ax_me.plot(xx, b[0] * np.log(xx) + b[1], color=PALETTE["grey"], ls="--", lw=1.0)
        rho, pv = stats.spearmanr(allx, ally)
        ax_me.text(0.05, 0.96, f"$r$={rho:+.2f}\n{_pfmt(pv)}",
                   transform=ax_me.transAxes, va="top", ha="left", fontsize=FS_SM)
    ax_me.axhline(0, color=PALETTE["lgrey"], lw=0.6)
    ax_me.set_xscale("log")
    ax_me.set_xlabel("fibers / target fascicle", fontsize=FS_SM)
    ax_me.set_ylabel(r"penalty  $\mathrm{SI_{dense}}-\mathrm{SI_{transfer}}$", fontsize=FS_SM)
    ax_me.tick_params(labelsize=FS_SM)
    ax_me.legend(frameon=False, fontsize=FS_SM - 1, loc="lower right")
    _tm = ax_me.set_title("mechanism", fontsize=FS_SM + 1, color=PALETTE["dgrey"],
                          weight="bold", y=1.18)
    _tm.set_zorder(1000)
    return ax_sw


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    rows   = _load_all_seeds()
    crows  = _load_contact_amplitudes()
    sparse = _load_sparse()

    if not rows:
        print("[figures_duke_main] no seed data found", file=sys.stderr)
        return 1
    n_samples = len({r["sample"] for r in rows})
    print(f"[figures_duke_main] {len(rows)} seed records, {n_samples} samples")
    print(f"[figures_duke_main] {len(crows)} contact amplitude records")
    if crows:
        z_vals = [r["z_row"] for r in crows]
        for z in [0, 1, 2]:
            print(f"  z_row={z}: {z_vals.count(z)} contacts")
    print(f"[figures_duke_main] {len(sparse)} sparse sampling records")

    # 16:9 landscape: wider than tall
    fig = plt.figure(figsize=(14.0, 7.875))

    # Independent height ratios per column so panel a can be taller
    # without forcing panel b / d to also change height.
    # Manual layout so top row (a wide, b narrow) and bottom row (c=d width) are
    # independent — impossible to achieve with a single shared GridSpec.
    L, R, T, B = 0.06, 0.98, 0.94, 0.12
    W, H = R - L, T - B           # usable width / height in figure fraction

    row_gap  = 0.15                # vertical gap between rows (icons need this headroom)
    c_gap    = 0.10                # gap above c — icons are small so less headroom needed
    col_gap  = 0.07                # horizontal gap between columns

    # Row heights: top 1.8×, bottom 1.0×
    bot_h = (H - row_gap) / 2.8
    top_h = 1.8 * bot_h
    bot_bot, bot_top = B,           B + bot_h
    top_bot, top_top = bot_top + row_gap, bot_top + row_gap + top_h

    # Panel c gets a taller extent (smaller gap above it than d)
    c_top = top_bot - c_gap

    # Top-row column widths: panel a 2.2× wider than panel b
    b_w = (W - col_gap) / 3.2
    a_w = 2.2 * b_w
    a_l, a_r = L,           L + a_w
    b_l, b_r = a_r + col_gap, a_r + col_gap + b_w

    # Bottom-row column widths: d is wider (3 sub-panels) than c
    c_w = (W - col_gap) * 0.40
    d_w = (W - col_gap) * 0.60
    c_l, c_r = L,           L + c_w
    d_l, d_r = c_r + col_gap, c_r + col_gap + d_w

    def _gs1(left, right, bottom, top):
        return gridspec.GridSpec(1, 1, figure=fig,
                                 left=left, right=right,
                                 bottom=bottom, top=top)

    ax_a_ref = _panel_a(_gs1(a_l, a_r, top_bot, top_top)[0, 0], fig, rows)
    ax_b_ref = _panel_b(_gs1(b_l, b_r, top_bot, top_top)[0, 0], fig, rows)
    ax_c_ref = _panel_c(_gs1(c_l, c_r, bot_bot, c_top)[0, 0],   fig, crows)
    ax_d_ref = _panel_d(_gs1(d_l, d_r, bot_bot, bot_top)[0, 0], fig, sparse)

    _panel_heading(ax_a_ref, "a", "Fiber activation maps",            dx=-0.11)
    _panel_heading(ax_b_ref, "b", "Selectivity performance",         dx=-0.16)
    _panel_heading(ax_c_ref, "c", "Stimulus amplitude by contact position",
                   dx=-0.14, dy=1.08)
    _panel_heading(ax_d_ref, "d", "Sparse fiber-sampling analysis",  dx=-0.14, dy=1.34)

    for ext in (".png", ".svg"):
        p = OUT_DIR / f"fig3_duke{ext}"
        fig.savefig(p, dpi=600 if ext == ".png" else None)
        print(f"  -> {p}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
