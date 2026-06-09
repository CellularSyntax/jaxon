"""Load a Duke FEM nerve-sample bundle into the jaxon optimisation pipeline.

The FEniCSx FEM pipeline produces, per Duke vagus-nerve sample, a directory
``duke_Ves/<sub>_<sam>/`` containing:

  - ``nerve_xsec.json``      — outline + polygonal fascicles + per-fibre xy (µm)
  - ``electrode_config.json``— 12-contact cuff (3 axial × 4 azimuthal) metadata
  - ``paths_Ve.npz``         — per-fibre arc-length grid and lead-field tensor
                               ``Ve_mat[Σpts, 12]`` in V/A (S.I.)

This loader returns the same triplet
``(nerve_geom, Ve_unit, node_indices, geoms, extra)``
that ``jaxfibers.stim.multichannel_field.precompute_ve_unit`` returns for the
synthetic Hussain-style pipeline, so the rest of the optimisation stack
(``run_rect_optimization`` / ``run_waveform_optimization``) is unchanged.

Key conventions for this conversion
-----------------------------------

* ``Ve_VperA`` is potential per amp.  ``1 V/A = 1 mV/mA`` numerically, so
  feeding ``Ve_VperA`` straight into ``Ve_unit`` (which the optimiser
  multiplies by ``amps_mA`` to get mV) gives correct units **without any
  scaling**.  Sign convention: positive ``amps_mA[k]`` corresponds to
  current injection into contact ``k`` of the same polarity as the FEM
  ``inject_A`` direction.

* FEM ``paths_flat`` is in metres.  jaxon's MRG compartment centres are in
  µm and reference the proximal end at z = 0; we shift them so the nodal
  midpoint sits at z = 0 (i.e.\ at the cuff centre) before interpolating
  the FEM ``Ve(z)`` onto them, mirroring the synthetic-Ve recentering
  step in ``selectivity_sweep.py``.

* A few % of the FEM Ve samples are NaN (points near fascicle ends that
  fell just off the FE mesh).  We fill them per-fibre with linear
  interpolation along z over the finite samples; if a fibre has no
  finite points for some contact (shouldn't happen in practice), we
  fall back to zero.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from jaxfibers.nerve.geometry import NerveGeometry, FascicleOutline
from jaxfibers.fibers.mrg import _mrg_geometry, section_centers_um


def _fill_nan_along_z(z: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Linearly interpolate v(z) across NaN entries in v.

    Parameters
    ----------
    z : [n] sorted (or unsorted) z-coordinates.
    v : [n] values, possibly with NaNs.

    Returns
    -------
    [n] array with NaNs filled by linear interpolation over the finite
    samples, with constant extrapolation at the ends (``np.interp``
    semantics).  If ``v`` has no finite entries, returns zeros.
    """
    finite = np.isfinite(v)
    if not finite.any():
        return np.zeros_like(v)
    if finite.all():
        return v
    # Sort by z for monotonic interpolation
    order = np.argsort(z)
    z_s = z[order]
    v_s = v[order]
    finite_s = np.isfinite(v_s)
    filled_s = np.interp(z_s, z_s[finite_s], v_s[finite_s])
    # Un-sort back to original index order
    out = np.empty_like(v)
    out[order] = filled_s
    return out


def _stratified_subsample(fasc_id: np.ndarray, max_fibers: int,
                          rng_seed: int = 0) -> np.ndarray:
    """Pick ~``max_fibers`` fibre indices stratified across fascicles.

    Each fascicle's share is proportional to its fibre count, rounded up,
    capped at the fascicle's own count.  Returns a sorted int array of
    fibre indices into the original 1...n_fibers space.  Deterministic
    given ``rng_seed``."""
    rng = np.random.default_rng(rng_seed)
    n_total = len(fasc_id)
    if max_fibers >= n_total:
        return np.arange(n_total)
    unique = np.unique(fasc_id)
    weights = np.array([(fasc_id == f).sum() for f in unique], dtype=float)
    weights /= weights.sum()
    quotas = np.ceil(weights * max_fibers).astype(int)  # rounded up
    picked = []
    for f, q in zip(unique, quotas):
        idx_f = np.where(fasc_id == f)[0]
        q_eff = min(q, len(idx_f))
        choice = rng.choice(idx_f, size=q_eff, replace=False)
        picked.extend(choice.tolist())
    return np.sort(np.array(picked, dtype=int))


def load_duke_sample(
    sample_dir: str | Path,
    fiber_diam_um: float = 5.7,
    n_nodes: int = 21,
    max_fibers: int | None = None,
    subsample_seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Load a Duke FEM bundle and produce jaxon-compatible solver inputs.

    Parameters
    ----------
    sample_dir : path
        Directory containing ``nerve_xsec.json`` / ``electrode_config.json`` /
        ``paths_Ve.npz``.
    fiber_diam_um : float
        MRG outer diameter to assign to all fibres (the FEM bundle does
        not provide individual diameters).  Default 5.7 µm.
    n_nodes : int
        Nodes per MRG fibre.  Default 21 (n_comp = 221).
    verbose : bool
        Print a summary line on load.

    Returns
    -------
    dict with keys:
        nerve_geom     — NerveGeometry (n_fibers, fiber_x/y_um, fiber_diam,
                         target_mask placeholder, fascicles list, divider angle)
        Ve_unit        — [K=12, n_fibers, n_comp] float64 mV at 1 mA
        node_indices   — [n_fibers, n_nodes] int32
        geoms          — list[MrgGeometry], one per fibre (identical structure)
        contact_xyz_um — [K, 3] contact positions in µm (for plotting only)
        fasc_id        — [n_fibers] int  fascicle-id per fibre
        fasc_meta      — list of dicts (centroid_xy_um, polygon_xy_um) per fascicle
        nerve_outline_xy_um — [N, 2] polygon for the nerve outer outline
        sample_name    — short tag, e.g. "sub-10_sam-1"
    """
    sample_dir = Path(sample_dir)
    nx_path  = sample_dir / "nerve_xsec.json"
    ec_path  = sample_dir / "electrode_config.json"
    npz_path = sample_dir / "paths_Ve.npz"
    for p in (nx_path, ec_path, npz_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Duke bundle missing required file: {p}"
            )
        if p.stat().st_size == 0:
            raise ValueError(
                f"Duke bundle file is empty (0 bytes), likely a failed "
                f"transfer: {p}"
            )
    try:
        nx = json.loads(nx_path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(
            f"nerve_xsec.json malformed in {sample_dir.name}: {e}"
        ) from e
    try:
        ec = json.loads(ec_path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(
            f"electrode_config.json malformed in {sample_dir.name}: {e}"
        ) from e
    npz = np.load(npz_path)

    # ── Nerve geometry ──────────────────────────────────────────────────────
    fiber_xy_all  = np.asarray(nx["fibers"]["xy_um"], dtype=np.float64)  # [F_full, 2]
    fasc_id_all   = np.asarray(nx["fibers"]["fascicle"], dtype=int)        # [F_full]
    n_fibers_full = fiber_xy_all.shape[0]

    # Optional stratified subsample (e.g. for CPU smoketest).
    if max_fibers is not None and max_fibers < n_fibers_full:
        keep_idx = _stratified_subsample(fasc_id_all, max_fibers, rng_seed=subsample_seed)
        fiber_xy = fiber_xy_all[keep_idx]
        fasc_id  = fasc_id_all[keep_idx]
        if verbose:
            print(f"[duke loader] {sample_dir.name}: stratified subsample "
                  f"{len(keep_idx)}/{n_fibers_full} fibres "
                  f"(seed={subsample_seed})")
    else:
        keep_idx = np.arange(n_fibers_full)
        fiber_xy = fiber_xy_all
        fasc_id  = fasc_id_all
    n_fibers = fiber_xy.shape[0]
    fiber_diam = np.full(n_fibers, fiber_diam_um, dtype=np.float64)

    fascicles_meta = []
    fascicles = []
    for fc in nx["fascicles"]:
        fid = int(fc["id"])
        cx, cy = fc["centroid_xy_um"]
        n_in_fc = int(np.sum(fasc_id == fid))
        fascicles.append(FascicleOutline(
            cx_um=float(cx), cy_um=float(cy),
            r_um=float(fc["radius_um"]),
            is_target=False,
            n_fibers=n_in_fc,
        ))
        fascicles_meta.append(dict(
            id=fid,
            centroid_xy_um=[float(cx), float(cy)],
            polygon_xy_um=np.asarray(fc["polygon_xy_um"], dtype=np.float64),
            radius_um=float(fc["radius_um"]),
            area_um2=float(fc["area_um2"]),
        ))

    nerve_geom = NerveGeometry(
        n_fibers=n_fibers,
        fiber_x_um=fiber_xy[:, 0].copy(),
        fiber_y_um=fiber_xy[:, 1].copy(),
        fiber_diam=fiber_diam,
        target_mask=np.zeros(n_fibers, dtype=bool),  # set by sweep code
        fascicles=fascicles,
        divider_angle_deg=None,
    )

    # ── Per-fibre MRG geometry (identical for all since same diameter) ──────
    geoms = [_mrg_geometry(fiber_diam_um, n_nodes) for _ in range(n_fibers)]
    n_comp = geoms[0].n_comp
    centers_z_um_raw = np.asarray(section_centers_um(geoms[0]))  # µm
    # Centre the fibre about z=0 (cuff midplane) so we can directly align
    # with the FEM data which is also in cuff-local coordinates.
    centers_z_um = centers_z_um_raw - centers_z_um_raw.mean()
    centers_z_m = centers_z_um * 1e-6

    node_indices = np.array(
        [[i for i, b in enumerate(geoms[f].is_node) if b] for f in range(n_fibers)],
        dtype=np.int32,
    )

    # ── FEM lead-field tensor → jaxon compartment grid ──────────────────────
    paths_flat        = npz["paths_flat"]    # [Σpts, 3] metres
    path_lengths_full = npz["path_lengths"]  # [F_full]
    Ve_mat            = npz["Ve_mat"]        # [Σpts, K] V/A (= mV/mA numerically)
    K = Ve_mat.shape[1]
    if int(path_lengths_full.sum()) != paths_flat.shape[0]:
        raise ValueError(
            f"paths_Ve.npz inconsistent: sum(path_lengths)={path_lengths_full.sum()}, "
            f"paths_flat.shape[0]={paths_flat.shape[0]}"
        )
    if int(path_lengths_full.shape[0]) != n_fibers_full:
        raise ValueError(
            f"fibre count mismatch: nerve_xsec.json {n_fibers_full}, "
            f"paths_Ve.npz {path_lengths_full.shape[0]}"
        )
    # Build per-fibre offset table over the full set, then index by keep_idx.
    offsets_full = np.concatenate([[0], np.cumsum(path_lengths_full)])  # [F_full + 1]

    Ve_unit = np.zeros((K, n_fibers, n_comp), dtype=np.float64)
    n_nan_filled = 0
    for fi, f_orig in enumerate(keep_idx):
        L = int(path_lengths_full[f_orig])
        a, b = offsets_full[f_orig], offsets_full[f_orig] + L
        z_fem_m = paths_flat[a:b, 2]
        Ve_fem  = Ve_mat[a:b, :]
        order = np.argsort(z_fem_m)
        z_fem_s = z_fem_m[order]
        for k in range(K):
            v = Ve_fem[order, k]
            finite = np.isfinite(v)
            n_nan_filled += int((~finite).sum())
            if not finite.any():
                continue
            v_filled = np.interp(z_fem_s, z_fem_s[finite], v[finite])
            Ve_unit[k, fi, :] = np.interp(centers_z_m, z_fem_s, v_filled)

    # ── Contact positions in µm (for plotting/diagnostic only — solver
    #    doesn't need them since Ve_unit already encodes them) ──────────────
    contacts = ec.get("patches") or ec.get("contacts") or []
    contact_xyz_um = np.zeros((K, 3), dtype=np.float64)
    for i, p in enumerate(contacts[:K]):
        # The Duke JSON stores positions in metres; defensively support
        # both metres and µm by checking magnitude.
        cx = float(p.get("x") or p.get("cx") or 0.0)
        cy = float(p.get("y") or p.get("cy") or 0.0)
        cz = float(p.get("z") or p.get("cz") or 0.0)
        # If magnitudes look like metres (sub-mm), promote to µm.
        if abs(cx) < 0.1 and abs(cy) < 0.1 and abs(cz) < 0.1:
            cx, cy, cz = cx * 1e6, cy * 1e6, cz * 1e6
        contact_xyz_um[i] = (cx, cy, cz)

    if verbose:
        print(
            f"[duke loader] {sample_dir.name}: "
            f"{n_fibers} fibres, {len(fascicles)} fascicles, K={K} contacts, "
            f"n_comp={n_comp}, NaNs filled along z = {n_nan_filled} "
            f"({100.0 * n_nan_filled / Ve_mat.size:.2f} % of Ve_mat)"
        )

    return dict(
        nerve_geom=nerve_geom,
        Ve_unit=Ve_unit,
        node_indices=node_indices,
        geoms=geoms,
        contact_xyz_um=contact_xyz_um,
        fasc_id=fasc_id,
        fasc_meta=fascicles_meta,
        nerve_outline_xy_um=np.asarray(nx["nerve_outline_xy_um"], dtype=np.float64),
        sample_name=sample_dir.name,
    )


def select_peripheral_target_candidates(
    fasc_meta: list,
    contact_xyz_um: np.ndarray,
    fasc_id: np.ndarray,
    n_min_fibers: int = 30,
    top_k: int = 3,
    peripheral_quantile: float = 0.5,
) -> list[int]:
    """Pick up to ``top_k`` fascicle IDs to use as single-fascicle targets.

    The selection paradigm is physics-aware: an extraneural cuff cannot
    selectively activate fascicles deep in the nerve interior because
    every contact reaches them at approximately the same Ve magnitude.
    So we restrict targets to **peripheral** fascicles (those with a
    small centroid-to-nearest-contact distance), and within that pool
    we take the largest fascicles by fibre count (so per-target SI
    statistics aren't degenerate).

    Parameters
    ----------
    fasc_meta : list
        Per-fascicle metadata with ``id``, ``centroid_xy_um``.
    contact_xyz_um : [K, 3]
        Cuff contact positions in µm.  Only the xy plane is used here
        since the cuff is axially extended and per-fascicle proximity
        is dominated by the angular component.
    fasc_id : [n_fibers] int
        Per-fibre fascicle membership (for the size filter).
    n_min_fibers : int, default 30
        Minimum fibre count for a fascicle to be a candidate; smaller
        fascicles give degenerate per-target SI statistics.
    top_k : int, default 3
        Maximum number of candidates to return per nerve.
    peripheral_quantile : float, default 0.5
        Fraction of fascicles to keep in the "peripheral" pool, ranked
        by ascending centroid-to-nearest-contact distance.  0.5 keeps
        the closest-to-contact half of the fascicles.

    Returns
    -------
    list of fascicle IDs (subset of ``[m['id'] for m in fasc_meta]``),
    ordered by fibre count descending.  May be shorter than ``top_k``
    if the size filter rejects everything, or empty if no fascicle
    passes the filters.
    """
    if not fasc_meta:
        return []
    contact_xy = np.asarray(contact_xyz_um, dtype=np.float64)[:, :2]
    items = []
    for m in fasc_meta:
        fid = int(m["id"])
        cx, cy = float(m["centroid_xy_um"][0]), float(m["centroid_xy_um"][1])
        d_min = float(np.min(np.hypot(contact_xy[:, 0] - cx,
                                       contact_xy[:, 1] - cy)))
        n_in_f = int(np.sum(fasc_id == fid))
        items.append({
            "id":           fid,
            "centroid_xy":  (cx, cy),
            "proximity_um": d_min,
            "n_fibers":     n_in_f,
        })

    # Peripheral pool: closest-to-contact half (or whatever fraction).
    items.sort(key=lambda r: r["proximity_um"])
    n_peripheral = max(1, int(np.ceil(len(items) * peripheral_quantile)))
    peripheral = items[:n_peripheral]

    # Size filter.
    candidates = [c for c in peripheral if c["n_fibers"] >= n_min_fibers]
    if not candidates:
        return []

    # Top-k by fibre count.
    candidates.sort(key=lambda r: -r["n_fibers"])
    return [c["id"] for c in candidates[:top_k]]


def single_fascicle_target_mask(
    nerve_geom: NerveGeometry,
    fasc_id: np.ndarray,
    fasc_meta: list,
    target_fascicle_id: int,
) -> np.ndarray:
    """Set the target_mask so that only fibres in ``target_fascicle_id``
    are target; every other fibre is off-target.  Updates the
    ``is_target`` flag on each fascicle in ``nerve_geom.fascicles`` for
    downstream plotting code.  Returns the boolean ``target_mask``.

    Mirrors the ``divider_split_target_mask`` API so the sweep code
    can drop in this targeting paradigm without other changes."""
    target_mask = (np.asarray(fasc_id, dtype=int) == int(target_fascicle_id))
    nerve_geom.target_mask = target_mask
    # Update is_target on each FascicleOutline so xsection plotting
    # paints the right fascicle in the target colour.
    id_to_tgt = {int(m["id"]): (int(m["id"]) == int(target_fascicle_id))
                 for m in fasc_meta}
    for fasc, m in zip(nerve_geom.fascicles, fasc_meta):
        fasc.is_target = id_to_tgt[int(m["id"])]
    # Clear divider angle — we're not using the divider paradigm here.
    nerve_geom.divider_angle_deg = None
    return target_mask


def divider_split_target_mask(
    nerve_geom: NerveGeometry,
    fasc_id: np.ndarray,
    fasc_meta: list,
    divider_angle_deg: float,
) -> np.ndarray:
    """Classify each fibre as target / off-target by a line through the
    origin with angle ``divider_angle_deg`` (anticlockwise from +x).

    A fascicle is classified by the sign of its centroid projection onto
    the line normal n̂ = (-sin θ, cos θ); fibres inherit their fascicle's
    class.  Returns ``target_mask`` (bool, [n_fibers]).

    If the divider happens to place every fascicle on one side, the
    fascicle whose centroid sits closest to the line is flipped so the
    split always yields at least one target and one off-target group
    (mirrors ``make_hussain_style_nerve``).
    """
    th = np.deg2rad(divider_angle_deg)
    nx, ny = -np.sin(th), np.cos(th)
    # Centroid side per fascicle (in same order as nerve_geom.fascicles).
    sides = np.array([nx * m["centroid_xy_um"][0] + ny * m["centroid_xy_um"][1]
                      for m in fasc_meta])
    is_tgt = sides > 0.0
    if is_tgt.all() or (~is_tgt).all():
        flip_i = int(np.argmin(np.abs(sides)))
        is_tgt[flip_i] = not is_tgt[flip_i]
    # Update the fascicles list in place so downstream plotting sees it.
    for fasc, t in zip(nerve_geom.fascicles, is_tgt):
        fasc.is_target = bool(t)
    # Map fascicle id → is_target via fasc_meta ordering (id field).
    id_to_tgt = {m["id"]: bool(t) for m, t in zip(fasc_meta, is_tgt)}
    target_mask = np.array([id_to_tgt[int(fid)] for fid in fasc_id], dtype=bool)
    nerve_geom.target_mask = target_mask
    nerve_geom.divider_angle_deg = float(divider_angle_deg)
    return target_mask
