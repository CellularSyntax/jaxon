"""Sanity-check the peripheral-cluster target selection visually.

For each Duke bundle in ``duke_Ves/``, draws a per-sample figure with
``--n-positions`` columns (default 4).  Each column shows one random
angular window:

  - Row 0 (overview): nerve outline, peripheral-threshold ring, chosen
    sector wedge, fascicle polygons coloured by role, contacts,
    centroid annotations.
  - Row 1 (fibre view): per-fibre target / off-target colouring.

Outputs land in ``outputs/duke_target_sanity/<sample>.png``.

Run from the project root::

    python -m experiments_v2.duke_target_sanity
    python -m experiments_v2.duke_target_sanity --n-positions 4 --rng-seed 0
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

from experiments_v2.duke_loader import select_random_cluster_targets
from experiments_v2.utils import ensure_dir


# ─── defaults ────────────────────────────────────────────────────────
DEFAULT_RADIUS_QUANTILE     = 0.32
DEFAULT_ANGULAR_WINDOW_DEG  = 90.0
DEFAULT_N_MIN_TARGET_FIBERS = 50
DEFAULT_MIN_FASCICLES       = 3
DEFAULT_N_POSITIONS         = 4
DEFAULT_RNG_SEED            = 0
DEFAULT_OUT_SUBDIR          = "duke_target_sanity"


def _resolve_duke_root(project_root: Path) -> Path | None:
    for sub in ("duke_Ves", "duke_meshes"):
        p = project_root / sub
        if p.is_dir():
            return p
    return None


def _lite_load(sample_dir: Path) -> dict:
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
            "radius_um":      float(fc.get("radius_um", 0.0)),
            "n_fibers":       int(np.sum(fasc_id == fid)),
        })

    outline = np.asarray(nx["nerve_outline_xy_um"], dtype=np.float64)

    contacts = ec.get("patches") or ec.get("contacts") or []
    K = len(contacts)
    contact_xyz = np.zeros((K, 3), dtype=np.float64)
    for i, p in enumerate(contacts):
        if "R" in p and "phi" in p:
            R   = float(p["R"])
            phi = float(p["phi"])
            z   = float(p.get("z", 0.0))
            cx2, cy2, cz = R * np.cos(phi), R * np.sin(phi), z
        else:
            cx2 = float(p.get("x") or p.get("cx") or 0.0)
            cy2 = float(p.get("y") or p.get("cy") or 0.0)
            cz  = float(p.get("z") or p.get("cz") or 0.0)
        # Convert m → µm if coordinates look like SI metres
        if abs(cx2) < 0.1 and abs(cy2) < 0.1 and abs(cz) < 0.1:
            cx2, cy2, cz = cx2 * 1e6, cy2 * 1e6, cz * 1e6
        contact_xyz[i] = (cx2, cy2, cz)

    return dict(fiber_xy=fiber_xy, fasc_id=fasc_id, fasc_meta=fasc_meta,
                outline=outline, contact_xyz=contact_xyz,
                sample_name=sample_dir.name)


# ─── drawing primitives ──────────────────────────────────────────────

def _draw_outline(ax, outline):
    ax.plot(outline[:, 0], outline[:, 1], "-", color="#888888",
            lw=0.6, zorder=1)
    ax.fill(outline[:, 0], outline[:, 1], color="#F5F5F5",
            alpha=0.5, edgecolor="none", zorder=0)


def _draw_peripheral_ring(ax, centroid_xy, r_threshold):
    cx0, cy0 = centroid_xy
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(cx0 + r_threshold * np.cos(theta),
            cy0 + r_threshold * np.sin(theta),
            "--", color="#AAAAAA", lw=0.7, alpha=0.8, zorder=2)
    ax.plot(cx0, cy0, "+", color="#666666", ms=6, mew=0.8, zorder=2)


def _draw_angular_sector(ax, centroid_xy, start_deg, end_deg, outline):
    cx0, cy0 = centroid_xy
    R = float(np.max(np.hypot(outline[:, 0] - cx0,
                               outline[:, 1] - cy0))) * 1.05
    width = (end_deg - start_deg) % 360.0 or 360.0
    ax.add_patch(mpatches.Wedge(
        (cx0, cy0), R, start_deg, start_deg + width,
        facecolor="#FFC0B0", edgecolor="#CC4444",
        alpha=0.22, lw=0.7, zorder=1,
    ))


def _draw_fascicle_polygons(ax, fasc_meta, target_id_set, peripheral_id_set):
    for m in fasc_meta:
        poly = m.get("polygon_xy_um")
        if poly is None or len(poly) < 3:
            continue
        fid = int(m["id"])
        if fid in target_id_set:
            face, edge, lw, alpha = "#7EC97E", "#2E7D32", 0.8, 0.75
        elif fid in peripheral_id_set:
            face, edge, lw, alpha = "#C8DCEF", "#4472A8", 0.5, 0.55
        else:
            face, edge, lw, alpha = "#D8D8D8", "#999999", 0.4, 0.45
        ax.fill(poly[:, 0], poly[:, 1], facecolor=face, edgecolor=edge,
                lw=lw, alpha=alpha, zorder=3)


def _draw_contacts(ax, contact_xyz):
    ax.scatter(contact_xyz[:, 0], contact_xyz[:, 1],
               s=40, c="#333333", edgecolor="white", lw=0.6, zorder=6)


def _annotate_centroids(ax, fasc_meta, target_id_set, peripheral_id_set):
    for m in fasc_meta:
        fid = int(m["id"])
        cx, cy = m["centroid_xy_um"]
        n = int(m["n_fibers"])
        if fid in target_id_set:
            ax.plot(cx, cy, "o", color="#CC2222", ms=6,
                    mec="#880000", mew=0.6, zorder=5)
            ax.annotate(f"{fid}\n(n={n})", (cx, cy),
                        ha="center", va="center", fontsize=5,
                        color="white", weight="bold", zorder=7)
        else:
            face = "#555555" if fid in peripheral_id_set else "#AAAAAA"
            ax.plot(cx, cy, "o", color=face, ms=2.5, zorder=5)
            ax.annotate(str(fid), (cx, cy),
                        ha="center", va="top", fontsize=4,
                        color="#666666", zorder=7,
                        xytext=(0, -6), textcoords="offset points")


def _set_axis(ax):
    ax.set_aspect("equal")
    ax.axis("off")


# ─── per-sample figure ───────────────────────────────────────────────

def draw_sample(sample: dict, clusters: list[dict], out_path: Path,
                radius_quantile: float, angular_window_deg: float,
                n_min_target_fibers: int) -> None:
    name      = sample["sample_name"]
    fasc_meta = sample["fasc_meta"]
    fasc_id   = sample["fasc_id"]
    fiber_xy  = sample["fiber_xy"]
    outline   = sample["outline"]
    contact_xyz = sample["contact_xyz"]
    n_pos = len(clusters)

    fig, axes = plt.subplots(2, n_pos,
                              figsize=(4.5 * n_pos, 8.5),
                              constrained_layout=True)
    if n_pos == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        f"{name}   |   r_qtl={radius_quantile}, "
        f"sector={angular_window_deg:.0f}°, "
        f"n_min={n_min_target_fibers}",
        fontsize=9, weight="semibold",
    )

    for col, cluster in enumerate(clusters):
        target_id_set    = set(int(t) for t in cluster["target_ids"])
        peripheral_id_set = set(int(t) for t in cluster["peripheral_ids"])
        start = cluster["window_start_deg"]
        end   = cluster["window_end_deg"]
        n_tgt_f = cluster["n_target_fibers"]
        ok = bool(target_id_set)

        # ── Row 0: overview ──────────────────────────────────────────
        ax = axes[0, col]
        _draw_outline(ax, outline)
        if ok:
            _draw_angular_sector(ax, cluster["nerve_centroid_xy_um"],
                                 start, end, outline)
        _draw_peripheral_ring(ax, cluster["nerve_centroid_xy_um"],
                              cluster["r_threshold_um"])
        _draw_fascicle_polygons(ax, fasc_meta, target_id_set, peripheral_id_set)
        _annotate_centroids(ax, fasc_meta, target_id_set, peripheral_id_set)
        _draw_contacts(ax, contact_xyz)
        _set_axis(ax)
        tids_str = (", ".join(map(str, sorted(target_id_set)))
                    if target_id_set else "none")
        ax.set_title(
            f"pos {col+1}: {start:.0f}–{end:.0f}°\n"
            f"target_ids=[{tids_str}]  n={n_tgt_f}",
            fontsize=7,
            color=("black" if ok else "#CC0000"),
        )

        # ── Row 1: fibre target/off-target ───────────────────────────
        ax = axes[1, col]
        _draw_outline(ax, outline)
        for m in fasc_meta:
            poly = m.get("polygon_xy_um")
            if poly is None or len(poly) < 3:
                continue
            fid = int(m["id"])
            if fid in target_id_set:
                face, edge, lw = "#7EC97E", "#2E7D32", 0.7
            else:
                face, edge, lw = "#EBEBEB", "#AAAAAA", 0.35
            ax.fill(poly[:, 0], poly[:, 1], facecolor=face, edgecolor=edge,
                    lw=lw, alpha=0.65, zorder=3)
        if ok:
            tgt_mask = np.array([int(f) in target_id_set for f in fasc_id],
                                 dtype=bool)
            ax.scatter(fiber_xy[tgt_mask, 0], fiber_xy[tgt_mask, 1],
                       s=1.2, c="#1A7340", alpha=0.80, zorder=4)
            ax.scatter(fiber_xy[~tgt_mask, 0], fiber_xy[~tgt_mask, 1],
                       s=0.8, c="#888888", alpha=0.40, zorder=4)
            n_t = int(tgt_mask.sum())
            n_o = int((~tgt_mask).sum())
            ax.set_title(f"target: {n_t}   off-target: {n_o}", fontsize=7)
        else:
            ax.set_title(
                f"NO TARGET  (best={n_tgt_f} < {n_min_target_fibers})",
                fontsize=7, color="#CC0000",
            )
        _draw_contacts(ax, contact_xyz)
        _set_axis(ax)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duke-root", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--radius-quantile", type=float,
                    default=DEFAULT_RADIUS_QUANTILE)
    ap.add_argument("--angular-window", type=float,
                    default=DEFAULT_ANGULAR_WINDOW_DEG)
    ap.add_argument("--n-min-target-fibers", type=int,
                    default=DEFAULT_N_MIN_TARGET_FIBERS)
    ap.add_argument("--min-fascicles", type=int,
                    default=DEFAULT_MIN_FASCICLES)
    ap.add_argument("--n-positions", type=int,
                    default=DEFAULT_N_POSITIONS,
                    help="Number of random angular positions per sample "
                         f"(default {DEFAULT_N_POSITIONS}).")
    ap.add_argument("--rng-seed", type=int,
                    default=DEFAULT_RNG_SEED,
                    help=f"Seed for random window placement "
                         f"(default {DEFAULT_RNG_SEED}).")
    args = ap.parse_args(argv)

    duke_root = args.duke_root or _resolve_duke_root(ROOT)
    if duke_root is None or not duke_root.exists():
        print(f"[error] no duke_Ves/ or duke_meshes/ found at {ROOT}",
              file=sys.stderr)
        return 1
    out_dir = ensure_dir(args.out or (ROOT / "outputs" / DEFAULT_OUT_SUBDIR))

    samples = sorted(d for d in duke_root.iterdir() if d.is_dir())
    print(f"[duke_target_sanity] {duke_root}  ({len(samples)} subdirs)")
    print(f"  params: r_qtl={args.radius_quantile}, "
          f"window={args.angular_window}°, "
          f"n_positions={args.n_positions}, "
          f"rng_seed={args.rng_seed}, "
          f"n_min={args.n_min_target_fibers}")
    print(f"  output: {out_dir}\n")

    n_drawn = n_skip = n_err = 0
    for sample_dir in samples:
        if not (sample_dir / "nerve_xsec.json").exists():
            continue
        try:
            sample = _lite_load(sample_dir)
        except Exception as e:
            print(f"  [error] {sample_dir.name}: {e}")
            n_err += 1
            continue
        if len(sample["fasc_meta"]) < args.min_fascicles:
            print(f"  [skip]  {sample_dir.name}: "
                  f"only {len(sample['fasc_meta'])} fascicle(s)")
            n_skip += 1
            continue
        if sample["contact_xyz"].shape[0] == 0:
            print(f"  [skip]  {sample_dir.name}: no contacts")
            n_skip += 1
            continue

        clusters = select_random_cluster_targets(
            sample["fasc_meta"], sample["fasc_id"], sample["outline"],
            n_positions=args.n_positions,
            rng_seed=args.rng_seed,
            radius_quantile=args.radius_quantile,
            angular_window_deg=args.angular_window,
            n_min_target_fibers=args.n_min_target_fibers,
        )
        out_path = out_dir / f"{sample_dir.name}.png"
        draw_sample(sample, clusters, out_path,
                    radius_quantile=args.radius_quantile,
                    angular_window_deg=args.angular_window,
                    n_min_target_fibers=args.n_min_target_fibers)

        ok_counts = [c["n_target_fibers"] for c in clusters if c["target_ids"]]
        print(f"  -> {sample_dir.name}: "
              f"{len(ok_counts)}/{args.n_positions} valid positions  "
              f"n_target=[{', '.join(str(n) for n in ok_counts)}]")
        n_drawn += 1

    print(f"\n[duke_target_sanity] done: {n_drawn} figures, "
          f"{n_skip} skipped, {n_err} errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
