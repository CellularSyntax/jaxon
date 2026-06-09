"""Sanity-check that contact_xyz_um[k] is co-located with the spatial
peak of Ve_unit[k, :, :] -- i.e. that the kth patch in nerve_xsec.json
corresponds to the kth column in Ve_mat.

If the FEM pipeline writes patches and Ve channels in different orders,
all our spatial-pattern reasoning ('south column = cathode at south
contact') is wrong because contact k physically lives somewhere
different from where its Ve hotspot is.

For each contact k:
  * Read its (x, y) from contact_xyz_um.
  * Find the top-N fibres by max|Ve_unit[k, f, :]| and compute their
    fibre-position centroid in (x, y).
  * Compute angular position phi for both: the "claimed" contact phi
    and the "Ve hotspot" phi.  If they disagree by > 22.5° the
    indexing is broken.

Run:
    python -m experiments_v2.duke_contact_ve_alignment <sample_dir>

where <sample_dir> is e.g.
    /msc/home/.../duke_swine_human/human/sub-53_sam-2

Output: a print table and a PNG side-by-side figure
(contact_xyz_um[k] vs. Ve_unit[k] hotspot).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Ensure project root is on sys.path so we can import duke_loader.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from experiments_v2.duke_loader import load_duke_sample  # noqa: E402


def _phi_deg(x: float, y: float) -> float:
    """Return atan2(y, x) in degrees, normalised to [-180, +180]."""
    return float(np.degrees(np.arctan2(y, x)))


def _phi_to_compass(phi_deg: float) -> str:
    phi = float(phi_deg) % 360.0
    compass = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    return compass[int(np.round(phi / 45.0)) % 8]


def _hotspot_centroid(Ve_unit_k: np.ndarray,
                       fiber_xy: np.ndarray,
                       top_frac: float = 0.05) -> tuple[float, float, float]:
    """Locate the (x, y) centroid of the top-frac fibres by max|Ve_unit|.

    Returns (cx, cy, mean_top_Ve).
    """
    per_fibre_ve = np.max(np.abs(Ve_unit_k), axis=1)   # [n_fibers]
    n_top = max(1, int(round(top_frac * per_fibre_ve.shape[0])))
    top_idx = np.argsort(per_fibre_ve)[-n_top:]
    cx = float(np.mean(fiber_xy[top_idx, 0]))
    cy = float(np.mean(fiber_xy[top_idx, 1]))
    mean_top = float(np.mean(per_fibre_ve[top_idx]))
    return cx, cy, mean_top


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_dir", type=str)
    parser.add_argument("--top-frac", type=float, default=0.05,
                        help="Fraction of fibres used to define Ve hotspot")
    parser.add_argument("--out-png", type=str, default=None,
                        help="Write side-by-side figure to this PNG path")
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir).resolve()
    print(f"[load] {sample_dir}")
    sample = load_duke_sample(sample_dir, fiber_diam_um=10.0, n_nodes=21,
                                 verbose=True)
    Ve_unit       = sample["Ve_unit"]          # [K, n_fibers, n_comp]
    contact_xyz   = sample["contact_xyz_um"]   # [K, 3]
    nerve_geom    = sample["nerve_geom"]
    fiber_xy      = np.stack([nerve_geom.fiber_x_um, nerve_geom.fiber_y_um],
                              axis=1)
    K = Ve_unit.shape[0]

    print()
    print(f"{'k':>3} | {'contact (x,y) [um]':>22}  {'phi_c':>7}  {'side_c':>5}"
          f" | {'Ve hotspot (x,y)':>22}  {'phi_v':>7}  {'side_v':>5}"
          f" | {'delta_phi':>10}  ok?")
    print("-" * 110)
    mismatches: list[int] = []
    for k in range(K):
        cx_c, cy_c = float(contact_xyz[k, 0]), float(contact_xyz[k, 1])
        phi_c = _phi_deg(cx_c, cy_c)
        side_c = _phi_to_compass(phi_c)

        cx_v, cy_v, _ = _hotspot_centroid(Ve_unit[k], fiber_xy,
                                            top_frac=args.top_frac)
        phi_v = _phi_deg(cx_v, cy_v)
        side_v = _phi_to_compass(phi_v)

        # angular distance accounting for wrap-around
        d_phi = (phi_v - phi_c + 540.0) % 360.0 - 180.0
        ok = abs(d_phi) <= 22.5
        if not ok:
            mismatches.append(k)
        print(f"{k:>3} | ({cx_c:+8.0f},{cy_c:+8.0f})  {phi_c:+7.1f}  {side_c:>5}"
              f" | ({cx_v:+8.0f},{cy_v:+8.0f})  {phi_v:+7.1f}  {side_v:>5}"
              f" | {d_phi:+10.1f}  {'OK' if ok else 'MISMATCH'}")
    print()
    if mismatches:
        print(f"[ALIGNMENT WARNING] {len(mismatches)}/{K} contacts have "
              f"|delta_phi| > 22.5°: {mismatches}")
        print("contact_xyz_um indexing does NOT align with Ve_unit channels.")
        print("Every spatial-pattern argument (tripolar_S etc.) is wrong "
              "because the cathode physically sits at a different column "
              "than the code thinks.")
    else:
        print(f"[ALIGNMENT OK] all {K} contacts within 22.5° between claimed "
              "position and Ve hotspot.")

    if args.out_png is None:
        out_png = sample_dir.parent.parent / "duke_contact_ve_alignment" / (
            sample_dir.name + ".png"
        )
    else:
        out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, title, get_xy, color in [
        (axes[0], "Contact positions (contact_xyz_um)", lambda k: contact_xyz[k, :2], "tab:blue"),
        (axes[1], "Ve_unit hotspots (top fibres per contact)",
         lambda k: np.array(_hotspot_centroid(Ve_unit[k], fiber_xy,
                                                args.top_frac)[:2]),
         "tab:red"),
    ]:
        # Draw fascicle outlines
        for fc in sample["fasc_meta"]:
            poly = fc["polygon_xy_um"]
            ax.fill(poly[:, 0], poly[:, 1], alpha=0.15, edgecolor="k",
                     facecolor="lightgray", linewidth=0.5)
        # Draw fibres
        ax.scatter(fiber_xy[:, 0], fiber_xy[:, 1], s=2, color="0.7",
                    alpha=0.3, label="fibres")
        # Draw the K markers
        for k in range(K):
            x, y = get_xy(k)
            ax.scatter(x, y, s=200, color=color, edgecolor="k", zorder=10)
            ax.annotate(str(k), (x, y), ha="center", va="center", fontsize=9,
                          color="white", weight="bold", zorder=11)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x [um]")
        ax.set_ylabel("y [um]")
    plt.suptitle(f"{sample_dir.name} -- contact vs. Ve hotspot alignment "
                  f"({len(mismatches)} mismatches)")
    plt.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"[saved] {out_png}")


if __name__ == "__main__":
    main()
