"""Sanity-check the peripheral target-candidate selection visually.

For each Duke bundle in ``duke_Ves/`` (or ``duke_meshes/``), draws a
per-sample multi-panel figure:

  - Overview panel: nerve outline, all fascicle polygons, all 12
    contacts, with the selected target candidates highlighted in
    red.
  - One panel per target candidate: the same nerve cross-section but
    coloured by the per-fibre target/off-target assignment that the
    sweep code will use when that fascicle is chosen as the target.

n_fascicle < 3 bundles are skipped (the divider / single-fascicle
paradigm is degenerate for them and they will be excluded from the
manuscript cohort).

Outputs land in ``outputs/duke_target_sanity/<sample>.png``.

Run from the project root:

    python -m experiments_v2.duke_target_sanity
    python -m experiments_v2.duke_target_sanity --top-k 5
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
import matplotlib.pyplot as plt

from experiments_v2.duke_loader import select_peripheral_target_candidates
from experiments_v2.utils import ensure_dir


# ─── defaults (CLI-overridable) ──────────────────────────────────────
DEFAULT_N_MIN_FIBERS       = 30
DEFAULT_TOP_K              = 3
DEFAULT_PERIPHERAL_QUANTILE = 0.5
DEFAULT_OUT_SUBDIR         = "duke_target_sanity"


def _resolve_duke_root(project_root: Path) -> Path | None:
    """Prefer duke_meshes/ if it exists, otherwise duke_Ves/."""
    for sub in ("duke_meshes", "duke_Ves"):
        p = project_root / sub
        if p.is_dir():
            return p
    return None


def _lite_load(sample_dir: Path) -> dict:
    """Read only what we need for drawing — no MRG geometry, no Ve
    interpolation.  Skips the expensive parts of ``load_duke_sample``
    so the sanity check runs in seconds per sample."""
    nx = json.loads((sample_dir / "nerve_xsec.json").read_text())
    ec = json.loads((sample_dir / "electrode_config.json").read_text())

    fiber_xy = np.asarray(nx["fibers"]["xy_um"], dtype=np.float64)   # [F, 2]
    fasc_id  = np.asarray(nx["fibers"]["fascicle"], dtype=int)        # [F]

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

    return dict(
        fiber_xy=fiber_xy, fasc_id=fasc_id, fasc_meta=fasc_meta,
        outline=outline, contact_xyz=contact_xyz,
        sample_name=sample_dir.name,
    )


def _draw_outline(ax, outline: np.ndarray) -> None:
    ax.plot(outline[:, 0], outline[:, 1], "k-", lw=0.8)
    ax.fill(outline[:, 0], outline[:, 1], color="0.95", alpha=0.5,
            edgecolor="none")


def _draw_fascicle_polygons(ax, fasc_meta, target_id=None,
                            candidate_ids=None) -> None:
    """Fill each fascicle polygon.  Colour depends on role:
       target_id present  -> green for that fascicle, grey for others
       candidate_ids set  -> red outline on candidates, blue fill on rest
       neither            -> uniform blue fill."""
    for m in fasc_meta:
        poly = m.get("polygon_xy_um")
        if poly is None or len(poly) < 3:
            continue
        fid = int(m["id"])
        if target_id is not None:
            is_tgt = (fid == target_id)
            face = "#7ec97e" if is_tgt else "#d8d8d8"
            edge = "black" if is_tgt else "0.6"
            lw = 1.0 if is_tgt else 0.5
            alpha = 0.7 if is_tgt else 0.5
        elif candidate_ids is not None:
            is_cand = fid in candidate_ids
            face = "#cfe2f3" if is_cand else "#e9e9e9"
            edge = "red" if is_cand else "0.6"
            lw = 1.5 if is_cand else 0.5
            alpha = 0.75 if is_cand else 0.5
        else:
            face, edge, lw, alpha = "#cfe2f3", "steelblue", 0.5, 0.5
        ax.fill(poly[:, 0], poly[:, 1], facecolor=face,
                edgecolor=edge, lw=lw, alpha=alpha)


def _draw_contacts(ax, contact_xyz: np.ndarray) -> None:
    ax.scatter(contact_xyz[:, 0], contact_xyz[:, 1],
               s=70, c="black", edgecolor="white", lw=1.0, zorder=5)


def _annotate_centroids(ax, fasc_meta, candidate_ids, all_items=None) -> None:
    """Label every fascicle centroid with its id and fibre count.
    Candidate fascicles get a red dot and bold white-on-red label;
    others get a small grey dot."""
    for m in fasc_meta:
        fid = int(m["id"])
        cx, cy = m["centroid_xy_um"]
        n = int(m["n_fibers"])
        if fid in candidate_ids:
            ax.plot(cx, cy, "o", color="red", ms=8, mec="black", mew=0.8,
                    zorder=4)
            ax.annotate(f"{fid}\n(n={n})", (cx, cy), ha="center", va="center",
                        fontsize=6, color="white", weight="bold", zorder=6)
        else:
            ax.plot(cx, cy, "o", color="0.4", ms=3, zorder=4)
            ax.annotate(f"{fid}", (cx, cy), ha="center", va="center",
                        fontsize=5, color="0.3", zorder=6,
                        xytext=(0, -8), textcoords="offset points")


def _set_axis(ax) -> None:
    ax.set_aspect("equal")
    ax.axis("off")


def draw_sample(sample: dict, candidate_ids: list[int],
                out_path: Path, n_min_fibers: int,
                peripheral_quantile: float) -> None:
    name = sample["sample_name"]
    fasc_meta = sample["fasc_meta"]
    fasc_id = sample["fasc_id"]
    fiber_xy = sample["fiber_xy"]
    outline = sample["outline"]
    contact_xyz = sample["contact_xyz"]

    n_panels = 1 + len(candidate_ids)
    fig_w = max(5.5 * n_panels, 6.0)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 5.6),
                              constrained_layout=True)
    if n_panels == 1:
        axes = np.array([axes])

    # ── Overview panel ──────────────────────────────────────────────
    ax = axes[0]
    _draw_outline(ax, outline)
    _draw_fascicle_polygons(ax, fasc_meta, candidate_ids=set(candidate_ids))
    _annotate_centroids(ax, fasc_meta, candidate_ids=set(candidate_ids))
    _draw_contacts(ax, contact_xyz)
    n_cand = len(candidate_ids)
    n_fasc = len(fasc_meta)
    cand_str = ", ".join(map(str, candidate_ids)) if candidate_ids else "none"
    ax.set_title(
        f"{name}\n"
        f"n_fasc={n_fasc}, candidates(red)={n_cand}: [{cand_str}]\n"
        f"(N_min={n_min_fibers}, periph_qtl={peripheral_quantile})",
        fontsize=8,
    )
    _set_axis(ax)

    # ── Per-candidate panels ────────────────────────────────────────
    for i, tid in enumerate(candidate_ids):
        ax = axes[i + 1]
        target_mask = (fasc_id == tid)
        n_tgt = int(target_mask.sum())
        n_nt = int((~target_mask).sum())
        _draw_outline(ax, outline)
        _draw_fascicle_polygons(ax, fasc_meta, target_id=tid)
        # Fibre dots: target green, off-target grey
        ax.scatter(fiber_xy[target_mask, 0], fiber_xy[target_mask, 1],
                   s=1.5, c="#1e7d1e", alpha=0.75, zorder=3)
        ax.scatter(fiber_xy[~target_mask, 0], fiber_xy[~target_mask, 1],
                   s=1.0, c="#666666", alpha=0.45, zorder=3)
        _draw_contacts(ax, contact_xyz)
        # Mark the target fascicle centroid
        for m in fasc_meta:
            if int(m["id"]) == tid:
                cx, cy = m["centroid_xy_um"]
                ax.plot(cx, cy, "o", color="red", ms=8, mec="black", mew=0.8,
                        zorder=5)
                break
        ax.set_title(f"target = fascicle #{tid}\n"
                     f"{n_tgt} target / {n_nt} off-target fibres",
                     fontsize=9)
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
    ap.add_argument("--n-min-fibers", type=int, default=DEFAULT_N_MIN_FIBERS,
                    help=f"Minimum fibre count per target candidate "
                         f"(default {DEFAULT_N_MIN_FIBERS}).")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                    help=f"Max number of target candidates per nerve "
                         f"(default {DEFAULT_TOP_K}).")
    ap.add_argument("--peripheral-quantile", type=float,
                    default=DEFAULT_PERIPHERAL_QUANTILE,
                    help=f"Fraction of fascicles kept in the peripheral "
                         f"pool, ranked by proximity to nearest contact "
                         f"(default {DEFAULT_PERIPHERAL_QUANTILE}).")
    ap.add_argument("--min-fascicles", type=int, default=3,
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
    print(f"[duke_target_sanity] params: n_min_fibers={args.n_min_fibers}, "
          f"top_k={args.top_k}, "
          f"peripheral_quantile={args.peripheral_quantile}, "
          f"min_fascicles={args.min_fascicles}")
    print(f"[duke_target_sanity] output: {out_dir}")
    print()

    n_drawn = n_skipped_low_fasc = n_skipped_no_cand = n_missing = 0
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
            n_skipped_low_fasc += 1
            continue
        if sample["contact_xyz"].shape[0] == 0:
            print(f"  [skip] {sample_dir.name}: no contacts in electrode_config.json")
            n_missing += 1
            continue
        candidates = select_peripheral_target_candidates(
            sample["fasc_meta"], sample["contact_xyz"], sample["fasc_id"],
            n_min_fibers=args.n_min_fibers, top_k=args.top_k,
            peripheral_quantile=args.peripheral_quantile,
        )
        if not candidates:
            print(f"  [skip] {sample_dir.name}: no candidates pass filters "
                  f"(n_fasc={n_fasc})")
            n_skipped_no_cand += 1
            continue
        out_path = out_dir / f"{sample_dir.name}.png"
        draw_sample(sample, candidates, out_path,
                    n_min_fibers=args.n_min_fibers,
                    peripheral_quantile=args.peripheral_quantile)
        n_drawn += 1
        print(f"  -> {sample_dir.name}: candidates={candidates} "
              f"(n_fasc={n_fasc})")

    print()
    print(f"[duke_target_sanity] done: {n_drawn} drawn, "
          f"{n_skipped_low_fasc} skipped (n_fasc < {args.min_fascicles}), "
          f"{n_skipped_no_cand} skipped (no candidate passes filters), "
          f"{n_missing} bundle load errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
