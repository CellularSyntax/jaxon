"""Figure 4: Duke-cohort population analysis — electrode patterns and SI correlates.

Layout (183 mm wide, Nature Medicine style):
  a — target-normalised electrode heatmap (swine)
  b — target-normalised electrode heatmap (human)
  c — SI vs total stimulus amplitude (L1 norm)
  d — SI vs number of cathodic contacts
  e — SI vs number of fascicles

Target-normalised heatmap:
  For each seed, each contact's angular position is rotated by the target
  cluster centre so that 0° always points toward the target fascicle group.
  Contacts are binned into 4 angular bins × 3 z-levels; the heatmap value
  is the mean |amplitude| across all seeds for that (bin, z-level) cell.

Run from project root:
    python -m experiments_v2.figures_duke_population
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT     = Path(__file__).resolve().parent.parent
SWEEP    = ROOT / "outputs" / "duke_sweeps"
DUKE_VES = ROOT / "duke_Ves"
OUT_DIR  = ROOT / "manuscript" / "figures" / "main"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = {
    "swine": "#D6604D",
    "human": "#4393C3",
    "grey":  "#555555",
    "lgrey": "#AAAAAA",
    "dgrey": "#333333",
}
SP_MARKER = {"swine": "s", "human": "o"}

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
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "pdf.fonttype":      42,
})

# ── Data loading ──────────────────────────────────────────────────────────────
def _species_of(sample: str) -> str:
    return "human" if sample.startswith("human") else "swine"


def _n_fascicles(sample: str) -> int:
    p = DUKE_VES / sample / "nerve_xsec.json"
    if not p.exists():
        return 0
    data = json.loads(p.read_text())
    return len(data.get("fascicles", []))


def _load_electrode_config(sample: str) -> list[dict]:
    p = DUKE_VES / sample / "electrode_config.json"
    if not p.exists():
        return []
    return json.loads(p.read_text()).get("patches", [])


def _phi_bin_target_normalised(phi_rad: float, target_center_deg: float) -> int:
    """Map a contact's phi (rad) to one of 4 target-normalised bins.
    Bin 0 = toward target (±45° of target center).
    Bin 1 = 90° CW, Bin 2 = opposite, Bin 3 = 90° CCW.
    """
    rel_deg = (np.degrees(phi_rad) - target_center_deg + 360.0) % 360.0
    return int((rel_deg + 45.0) // 90.0) % 4


def _load_population() -> list[dict]:
    """Load all seeds; compute per-seed heatmap contributions + scalar features."""
    rows = []
    for sample_dir in sorted(SWEEP.iterdir()):
        if not sample_dir.is_dir():
            continue
        sample = sample_dir.name
        patches = _load_electrode_config(sample)
        n_fascs = _n_fascicles(sample)
        phi_rads = [p["phi"] for p in patches]   # in radians

        for jpath in sorted(sample_dir.glob("data_seed_*.json")):
            try:
                raw = json.loads(jpath.read_text())
            except Exception:
                continue
            rect = raw.get("rect") or {}
            fire = rect.get("firing") or {}
            si = float(rect.get("achievable_si",
                                  abs(rect.get("final_si", float("nan")))))
            amps = np.asarray(rect.get("amps_mA", []), dtype=float)
            cluster = raw.get("cluster") or {}
            w_start = float(cluster.get("window_start_deg", 0.0))
            w_end   = float(cluster.get("window_end_deg",   0.0))
            target_center = (w_start + w_end) / 2.0

            # target-normalised amplitude grid [4 phi_bins × 3 z_rows]
            grid = np.zeros((4, 3), dtype=float)
            grid_count = np.zeros((4, 3), dtype=int)

            # Determine z-levels
            if patches:
                z_vals = np.array([p["z"] for p in patches])
                uniq_z = np.sort(np.unique(np.round(z_vals, 6)))
                if uniq_z.size == 3:
                    z_top, z_mid, z_bot = uniq_z[2], uniq_z[1], uniq_z[0]
                    def _z_row(zi):
                        if abs(zi - z_top) < 1e-6:
                            return 0
                        elif abs(zi - z_bot) < 1e-6:
                            return 2
                        return 1
                else:
                    z_sorted = np.sort(z_vals)
                    def _z_row(zi):
                        rank = np.searchsorted(z_sorted, zi)
                        return min(int(rank * 3 // len(z_sorted)), 2)

                for k, p in enumerate(patches):
                    if k >= len(amps):
                        break
                    phi_bin = _phi_bin_target_normalised(p["phi"], target_center)
                    z_row   = _z_row(p["z"])
                    grid[phi_bin, z_row]       += abs(float(amps[k]))
                    grid_count[phi_bin, z_row] += 1

            # Scalar features
            l1_norm     = float(np.sum(np.abs(amps))) if amps.size else float("nan")
            n_cathodic  = int(np.sum(amps < -1e-9))   if amps.size else 0

            rows.append(dict(
                sample=sample,
                species=_species_of(sample),
                seed=int(jpath.stem.split("_")[-1]),
                si=si,
                l1_norm=l1_norm,
                n_cathodic=n_cathodic,
                n_fascicles=n_fascs,
                amp_grid=grid,
                amp_grid_count=grid_count,
            ))
    return rows


# ── Panel a/b: target-normalised electrode heatmap ───────────────────────────
def _heatmap_panel(ax: plt.Axes, rows: list[dict], species: str) -> None:
    sp_rows = [r for r in rows if r["species"] == species and np.isfinite(r["si"])]
    if not sp_rows:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=FS_SM, color=PALETTE["lgrey"])
        return

    total_grid = np.zeros((4, 3), dtype=float)
    total_count = np.zeros((4, 3), dtype=int)
    for r in sp_rows:
        total_grid  += r["amp_grid"]
        total_count += r["amp_grid_count"]

    mean_grid = np.where(total_count > 0, total_grid / total_count, 0.0)
    # shape (4 phi_bins, 3 z_rows) → plot as heatmap with phi_bins on x, z_rows on y
    # transpose for imshow: rows = z (top=row 0), cols = phi_bins
    img = mean_grid.T   # (3, 4)

    vmax = max(float(img.max()), 0.01)
    im = ax.imshow(img, aspect="auto", origin="upper",
                   cmap="Blues", vmin=0.0, vmax=vmax,
                   interpolation="nearest")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("mean |amp| (mA)", fontsize=FS_SM)
    cbar.ax.tick_params(labelsize=FS_SM)

    col = PALETTE[species]
    ax.set_title(species, fontsize=FS, color=col, weight="semibold", pad=4)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["toward\ntarget", "right", "opposite", "left"],
                       fontsize=FS_SM)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["top (+z)", "mid", "bot (−z)"], fontsize=FS_SM)
    ax.set_xlabel("contact position (target-normalised)", fontsize=FS_SM)
    ax.set_ylabel("z-level", fontsize=FS_SM)

    # Annotate cells with mean value
    for zi in range(3):
        for pi in range(4):
            val = img[zi, pi]
            if val > 0.005:
                ax.text(pi, zi, f"{val:.2f}", ha="center", va="center",
                        fontsize=FS_SM - 1, color="white" if val > vmax * 0.6 else "#333333")
    n = len(sp_rows)
    ax.text(0.98, 0.02, f"n = {n} seeds", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=FS_SM, color=PALETTE["lgrey"])


# ── Panels c, d, e: SI scatter vs scalar feature ─────────────────────────────
def _scatter_panel(ax: plt.Axes, rows: list[dict],
                   x_field: str, xlabel: str,
                   x_log: bool = False) -> None:
    for species in ["swine", "human"]:
        sp = [r for r in rows
              if r["species"] == species
              and np.isfinite(r["si"])
              and np.isfinite(r.get(x_field, float("nan")))]
        if not sp:
            continue
        x = np.array([r[x_field] for r in sp])
        y = np.array([r["si"] for r in sp])
        col = PALETTE[species]
        ax.scatter(x, y, s=9, color=col, marker=SP_MARKER[species],
                   alpha=0.65, edgecolors="none", zorder=3, label=species)

    ax.axhline(0.90, color=PALETTE["lgrey"], ls="--", lw=0.6, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("selectivity index (SI)")
    ax.set_ylim(0, 1.08)
    if x_log:
        ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=FS_SM, loc="lower right",
              handletextpad=0.3, markerscale=1.2)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    rows = _load_population()
    if not rows:
        print("[figures_duke_population] no data found", file=sys.stderr)
        return 1
    n_samples = len({r["sample"] for r in rows})
    print(f"[figures_duke_population] {len(rows)} seed×sample records, "
          f"{n_samples} samples")

    fig = plt.figure(figsize=(7.2, 6.4))

    # Two-row layout: heatmaps (top) + scatter panels (bottom)
    gs_outer = gridspec.GridSpec(
        2, 1, figure=fig,
        left=0.08, right=0.97, top=0.96, bottom=0.09,
        hspace=0.50,
        height_ratios=[1.0, 1.0],
    )

    # Top row: two heatmaps (swine | human)
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_outer[0], wspace=0.50,
    )
    ax_sw_heat = fig.add_subplot(gs_top[0])
    ax_hu_heat = fig.add_subplot(gs_top[1])

    # Bottom row: 3 scatter panels
    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_outer[1], wspace=0.45,
    )
    ax_l1    = fig.add_subplot(gs_bot[0])
    ax_cath  = fig.add_subplot(gs_bot[1])
    ax_fascs = fig.add_subplot(gs_bot[2])

    # ── Populate ─────────────────────────────────────────────────────────────
    _heatmap_panel(ax_sw_heat, rows, "swine")
    _heatmap_panel(ax_hu_heat, rows, "human")

    _scatter_panel(ax_l1,    rows, "l1_norm",    "total |amplitude| L1 (mA)",
                   x_log=False)
    _scatter_panel(ax_cath,  rows, "n_cathodic", "# cathodic contacts")
    _scatter_panel(ax_fascs, rows, "n_fascicles","# fascicles in nerve")

    # ── Panel letters ─────────────────────────────────────────────────────────
    for letter, ax in [
        ("a", ax_sw_heat), ("b", ax_hu_heat),
        ("c", ax_l1),      ("d", ax_cath), ("e", ax_fascs),
    ]:
        ax.text(-0.16, 1.06, letter, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="bottom", ha="left")

    for ext in (".png", ".svg"):
        p = OUT_DIR / f"fig4_duke_population{ext}"
        fig.savefig(p, dpi=300 if ext == ".png" else None)
        print(f"  -> {p}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
