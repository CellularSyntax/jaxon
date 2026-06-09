"""Sanity-check the peripheral-cluster target selection visually.

For each Duke bundle in ``duke_Ves/`` (or ``duke_meshes/``), draws a
per-sample 2-panel figure:

  - Overview panel: nerve outline, the radial peripheral threshold
    ring, the chosen angular target sector, all fascicle polygons
    coloured by their role (target / peripheral-but-not-target /
    central), all 12 contacts, and fascicle centroids labelled with
    their id and fibre count.
  - Target/off-target panel: the same nerve cross-section but with
    per-fibre target/off-target colouring matching what the sweep
    code will use.

n_fascicle < ``--min-fascicles`` bundles are skipped (the cluster
paradigm is degenerate for them and they will be excluded from the
manuscript cohort).  Likewise nerves whose best cluster contains
fewer than ``--n-min-target-fibers`` fibres are flagged.

Outputs land in ``outputs/duke_target_sanity/<sample>.png``.

Run from the project root:

    python -m experiments_v2.duke_target_sanity
    python -m experiments_v2.duke_target_sanity --angular-window 120
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from experiments_v2.duke_loader import select_peripheral_cluster_target
from experiments_v2.utils import ensure_dir


# ─── defaults (CLI-overridable) ──────────────────────────────────────
DEFAULT_RADIUS_QUANTILE     = 0.6
DEFAULT_ANGULAR_WINDOW_DEG  = 90.0
DEFAULT_N_MIN_TARGET_FIBERS = 50
DEFAULT_MIN_FASCICLES       = 3
DEFAULT_OUT_SUBDIR          = "duke_target_sanity"


def _resolve_duke_root(project_root: Path) -> Path | None:
    for sub in ("duke_meshes", "duke_Ves"):
        p = project_root / sub
        if p.is_dir():
            return p
    return None


def _lite_load(sample_dir: Path) -> dict:
    """Read only what we need for drawing — no MRG geometry, no Ve."""
    nx = json.loads((sample_dir / "nerve_xsec.json").read_text())
    ec = json.loads((sample_dir / "electrode_config.json").read_text())

    fiber_xy = np.asarray(nx["fibers"]["xy_um"], dtype=np.float64)
    fasc_id  = np.asarray(nx["fibers"]["fascicle"], dtype=int)

    fasc_meta = []
    for fc in nx["fascicles"]:
        poly = (np.asarray(fc["polygon_xy_um"], dtype=np.float64)
                if "polygon_xy_um" in fc else None)
        cx, cy = fc["centroid_xy_um"]
        fid = int(fc["id"])
        fasc_meta.append({
            "id":             fid,
            "centroid_xy_um": [float(cx), float(cy)],
            "polygon_xy_um":  poly,
            "n_fibers":       int(np.sum(fasc_id == fid)),
        })

    outline = np.asarray(nx["nerve_outline_xy_um"], dtype=np.float64)

    contacts = ec.get("patches") or ec.get("contacts") or []
    K = len(contacts)
    contact_xyz = np.zeros((K, 3), dtype=np.float64)
    for i, p in enumerate(contacts):
        cx = float(p.get("x") or p.get("cx") or 0.0)
        cy = float(p.get("y") or p.get("cy") or 0.0)
        cz = float(p.get("z") or p.get("cz") or 0.0)
        if abs(cx) < 0.1 and abs(cy) < 0.1 and abs(cz) < 0.1:
            cx, cy, cz = cx * 1e6, cy * 1e6, cz * 1e6
        contact_xyz[i] = (cx, cy, cz)

    return dict(fiber_xy=fiber_xy, fasc_id=fasc_id, fasc_meta=fasc_meta,
                outline=outline, contact_xyz=contact_xyz,
                sample_name=sample_dir.name)


def _draw_outline(ax, outline: np.ndarray) -> None:
    ax.plot(outline[:, 0], outline[:, 1], "k-", lw=0.8, zorder=1)
    ax.fill(outline[:, 0], outline[:, 1], color="0.95", alpha=0.5,
            edgecolor="none", zorder=0)


def _draw_peripheral_ring(ax, centroid_xy, r_threshold,
                          outline: np.ndarray) -> None:
    """Dashed circle at the radial peripheral threshold."""
    cx0, cy0 = centroid_xy
    theta = np.linspace(0, 2 * np.pi, 200)
    rx = cx0 + r_threshold * np.cos(theta)
    ry = cy0 + r_threshold * np.sin(theta)
    ax.plot(rx, ry, "--", color="0.55", lw=0.8, alpha=0.7, zorder=2)
    ax.plot(cx0, cy0, "+", color="0.3", ms=8, mew=1.0, zorder=2)


def _draw_angular_sector(ax, centroid_xy, start_deg, end_deg,
                         outline: np.ndarray) -> None:
    """Translucent wedge from centroid through the outer boundary
    covering [start_deg, end_deg]."""
    cx0, cy0 = centroid_xy
    # Use a large radius to cover the full nerve.
    R = float(np.max(np.hypot(outline[:, 0] - cx0, outline[:, 1] - cy0))) * 1.05
    width = (end_deg - start_deg) % 360.0
    if width == 0.0:
        width = 360.0
    wedge = mpatches.Wedge(
        (cx0, cy0), R, start_deg, start_deg + width,
        facecolor="#ffcccb", edgecolor="red", alpha=0.25, lw=0.8, zorder=1,
    )
    ax.add_patch(wedge)


def _draw_fascicle_polygons(ax, fasc_meta, target_id_set, peripheral_id_set,
                            for_target_view: bool = False) -> None:
    """Colour fascicles by role.

       target_id_set ∋ fid              → green
       fid ∈ peripheral_id_set (not tgt) → light blue with grey edge
       else                             → grey
    """
    for m in fasc_meta:
        poly = m.get("polygon_xy_um")
        if poly is None or len(poly) < 3:
            continue
        fid = int(m["id"])
        if fid in target_id_set:
            face, edge, lw, alpha = "#7ec97e", "black", 1.0, 0.7
        elif fid in peripheral_id_set:
            face, edge, lw, alpha = "#cfe2f3", "steelblue", 0.6, 0.55
        else:
            face, edge, lw, alpha = "#d8d8d8", "0.6", 0.4, 0.45
        ax.fill(poly[:, 0], poly[:, 1], facecolor=face, edgecolor=edge,
                lw=lw, alpha=alpha, zorder=3)


def _draw_contacts(ax, contact_xyz: np.ndarray) -> None:
    ax.scatter(contact_xyz[:, 0], contact_xyz[:, 1],
               s=70, c="black", edgecolor="white", lw=1.0, zorder=6)


def _annotate_centroids(ax, fasc_meta, target_id_set,
                        peripheral_id_set) -> None:
    for m in fasc_meta:
        fid = int(m["id"])
        cx, cy = m["centroid_xy_um"]
        n = int(m["n_fibers"])
        if fid in target_id_set:
            ax.plot(cx, cy, "o", color="red", ms=8, mec="black", mew=0.8,
                    zorder=5)
            ax.annotate(f"{fid}\n(n={n})", (cx, cy), ha="center",
                        va="center", fontsize=6, color="white",
                        weight="bold", zorder=7)
        else:
            face = "0.4" if fid in peripheral_id_set else "0.7"
            ax.plot(cx, cy, "o", color=face, ms=3, zorder=5)
            ax.annotate(f"{fid}", (cx, cy), ha="center", va="center",
                        fontsize=5, color="0.3", zorder=7,
                        xytext=(0, -8), textcoords="offset points")


def _set_axis(ax) -> None:
    ax.set_aspect("equal")
    ax.axis("off")


def draw_sample(sample: dict, cluster: dict, out_path: Path,
                radius_quantile: float, angular_window_deg: float,
                n_min_target_fibers: int) -> None:
    name = sample["sample_name"]
    fasc_meta = sample["fasc_meta"]
    fasc_id = sample["fasc_id"]
    fiber_xy = sample["fiber_xy"]
    outline = sample["outline"]
    contact_xyz = sample["contact_xyz"]

    target_id_set = set(int(t) for t in cluster["target_ids"])
    peripheral_id_set = set(int(t) for t in cluster["peripheral_ids"])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.6),
                              constrained_layout=True)

    # ── Overview panel ──────────────────────────────────────────────
    ax = axes[0]
    _draw_outline(ax, outline)
    if target_id_set:
        _draw_angular_sector(ax, cluster["nerve_centroid_xy_um"],
                             cluster["window_start_deg"],
                             cluster["window_end_deg"],
                             outline)
    _draw_peripheral_ring(ax, cluster["nerve_centroid_xy_um"],
                          cluster["r_threshold_um"], outline)
    _draw_fascicle_polygons(ax, fasc_meta, target_id_set, peripheral_id_set)
    _annotate_centroids(ax, fasc_meta, target_id_set, peripheral_id_set)
    _draw_contacts(ax, contact_xyz)

    n_fasc = len(fasc_meta)
    n_per = len(peripheral_id_set)
    n_tgt_fasc = len(target_id_set)
    n_tgt_fibers = cluster["n_target_fibers"]
    tids_str = ", ".join(map(str, sorted(target_id_set))) if target_id_set else "none"
    title = (
        f"{name}\n"
        f"n_fasc={n_fasc}, peripheral={n_per}, target cluster={n_tgt_fasc}\n"
        f"target_ids=[{tids_str}], n_target_fibers={n_tgt_fibers}\n"
        f"(r_qtl={radius_quantile}, sector={angular_window_deg:.0f}°, "
        f"window {cluster['window_start_deg']:.0f}-{cluster['window_end_deg']:.0f}°)"
    )
    ax.set_title(title, fontsize=8)
    _set_axis(ax)

    # ── Target/off-target panel ──────────────────────────────────────
    ax = axes[1]
    _draw_outline(ax, outline)
    if target_id_set:
        target_mask = np.array(
            [int(f) in target_id_set for f in fasc_id], dtype=bool
        )
        n_tgt = int(target_mask.sum())
        n_nt = int((~target_mask).sum())
        # Fascicle polygons: target green, others light grey
        for m in fasc_meta:
            poly = m.get("polygon_xy_um")
            if poly is None or len(poly) < 3:
                continue
            fid = int(m["id"])
            face = "#7ec97e" if fid in target_id_set else "#e9e9e9"
            edge = "black" if fid in target_id_set else "0.6"
            lw = 1.0 if fid in target_id_set else 0.4
            alpha = 0.7 if fid in target_id_set else 0.5
            ax.fill(poly[:, 0], poly[:, 1], facecolor=face, edgecolor=edge,
                    lw=lw, alpha=alpha, zorder=3)
        # Fibre dots
        ax.scatter(fiber_xy[target_mask, 0], fiber_xy[target_mask, 1],
                   s=1.5, c="#1e7d1e", alpha=0.75, zorder=4)
        ax.scatter(fiber_xy[~target_mask, 0], fiber_xy[~target_mask, 1],
                   s=1.0, c="#666666", alpha=0.45, zorder=4)
        ax.set_title(f"target cluster: {n_tgt} target / {n_nt} off-target fibres",
                     fontsize=9)
    else:
        # Failed selection — explain why
        ax.set_title(
            f"NO TARGET\n"
            f"best cluster had only {cluster['n_target_fibers']} fibres "
            f"(< {n_min_target_fibers})",
            fontsize=9, color="red",
        )
    _draw_contacts(ax, contact_xyz)
    _set_axis(ax)

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duke-root", type=Path, default=None,
                    help="Override duke_Ves / duke_meshes location.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Override output directory.")
    ap.add_argument("--radius-quantile", type=float,
                    default=DEFAULT_RADIUS_QUANTILE,
                    help=f"Radial cut for peripheral fascicles, as a "
                         f"fraction of the maximum fascicle radial "
                         f"position (default {DEFAULT_RADIUS_QUANTILE}).")
    ap.add_argument("--angular-window", type=float,
                    default=DEFAULT_ANGULAR_WINDOW_DEG,
                    help=f"Angular sector width in degrees that defines "
                         f"the target cluster (default "
                         f"{DEFAULT_ANGULAR_WINDOW_DEG}).")
    ap.add_argument("--n-min-target-fibers", type=int,
                    default=DEFAULT_N_MIN_TARGET_FIBERS,
                    help=f"Reject nerves whose best cluster has fewer "
                         f"target fibres than this (default "
                         f"{DEFAULT_N_MIN_TARGET_FIBERS}).")
    ap.add_argument("--min-fascicles", type=int,
                    default=DEFAULT_MIN_FASCICLES,
                    help="Skip bundles with fewer than this many fascicles.")
    args = ap.parse_args(argv)

    duke_root = args.duke_root or _resolve_duke_root(ROOT)
    if duke_root is None or not duke_root.exists():
        print(f"[error] no duke_meshes/ or duke_Ves/ found at project root "
              f"({ROOT})", file=sys.stderr)
        return 1
    out_dir = ensure_dir(args.out or (ROOT / "outputs" / DEFAULT_OUT_SUBDIR))

    samples = sorted(d for d in duke_root.iterdir() if d.is_dir())
    print(f"[duke_target_sanity] scanning {duke_root} "
          f"({len(samples)} subdirs)")
    print(f"[duke_target_sanity] params: radius_quantile={args.radius_quantile}, "
          f"angular_window={args.angular_window}°, "
          f"n_min_target_fibers={args.n_min_target_fibers}, "
          f"min_fascicles={args.min_fascicles}")
    print(f"[duke_target_sanity] output: {out_dir}")
    print()

    n_drawn = n_skip_low_fasc = n_skip_no_cluster = n_missing = 0
    for sample_dir in samples:
        nx_path = sample_dir / "nerve_xsec.json"
        if not nx_path.exists():
            continue
        try:
            sample = _lite_load(sample_dir)
        except Exception as e:
            print(f"  [error] {sample_dir.name}: load failed ({e})")
            n_missing += 1
            continue
        n_fasc = len(sample["fasc_meta"])
        if n_fasc < args.min_fascicles:
            print(f"  [skip] {sample_dir.name}: only {n_fasc} fascicle(s)")
            n_skip_low_fasc += 1
            continue
        if sample["contact_xyz"].shape[0] == 0:
            print(f"  [skip] {sample_dir.name}: no contacts")
            n_missing += 1
            continue
        cluster = select_peripheral_cluster_target(
            sample["fasc_meta"], sample["fasc_id"], sample["outline"],
            radius_quantile=args.radius_quantile,
            angular_window_deg=args.angular_window,
            n_min_target_fibers=args.n_min_target_fibers,
        )
        out_path = out_dir / f"{sample_dir.name}.png"
        draw_sample(sample, cluster, out_path,
                    radius_quantile=args.radius_quantile,
                    angular_window_deg=args.angular_window,
                    n_min_target_fibers=args.n_min_target_fibers)
        if not cluster["target_ids"]:
            print(f"  -> {sample_dir.name}: NO TARGET "
                  f"(best cluster had {cluster['n_target_fibers']} fibres, "
                  f"need >= {args.n_min_target_fibers})")
            n_skip_no_cluster += 1
        else:
            print(f"  -> {sample_dir.name}: target={cluster['target_ids']} "
                  f"({cluster['n_target_fibers']} fibres, sector "
                  f"{cluster['window_start_deg']:.0f}-"
                  f"{cluster['window_end_deg']:.0f}°)")
        n_drawn += 1

    print()
    print(f"[duke_target_sanity] done: {n_drawn} figures drawn "
          f"({n_skip_no_cluster} with no acceptable cluster), "
          f"{n_skip_low_fasc} skipped (n_fasc < {args.min_fascicles}), "
          f"{n_missing} bundle load errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
