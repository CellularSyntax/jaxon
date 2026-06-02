"""Synthetic nerve cross-section geometry for selectivity optimization experiments.

Two constructors:

* ``make_multi_fascicle_nerve`` — the realistic one.  Discrete circular
  fascicles inside the nerve outline; fibres are placed *inside* a
  specific fascicle and assigned target / off-target at the *fascicle*
  level.  This matches how real peripheral nerves (and Hussain's P1-P6 /
  H1-H6 anatomies) are organised: fibres of like function bundle into a
  perineurium-bounded fascicle, and the optimiser decides whether that
  *fascicle* should fire.

* ``make_synthetic_nerve`` — legacy single-disk scatter.  Kept for
  backward compatibility (selectivity_demo and a handful of older
  experiments still call it), but **not** recommended for new work:
  it places fibres uniformly across the whole nerve cross-section and
  labels those falling inside one eccentric disk as target, the rest
  off-target.  The result is a population in which target and off-target
  fibres can sit at arbitrarily small spatial separation, which makes
  selective stimulation almost impossible by construction.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

# MRG discrete diameters available for mixed-fibre nerves.
AVAILABLE_DIAMETERS = [5.7, 7.3, 8.7, 10.0, 11.5, 12.8, 14.0, 15.0, 16.0]


@dataclass
class FascicleOutline:
    """One circular fascicle inside the nerve outline.

    The same diameter assumption that Hussain makes (a single representative
    fibre per fascicle, the activation of which proxies for the whole
    cross-sectional area) means a *circular* outline is plenty for our
    purposes — what matters is the centre / radius the FEM Ve template
    sees, not the precise shape.
    """
    cx_um:     float
    cy_um:     float
    r_um:      float
    is_target: bool
    n_fibers:  int


@dataclass
class NerveGeometry:
    """Cross-sectional geometry of a synthetic nerve.

    ``fascicles`` is optional for backward compatibility with the legacy
    single-disk constructor; when populated by ``make_multi_fascicle_nerve``
    it lets the cross-section plot draw the actual fascicle boundaries
    instead of inferring them from fibre positions.
    """
    n_fibers:    int
    fiber_x_um:  np.ndarray    # [n_fibers] x-position in cross-section (µm)
    fiber_y_um:  np.ndarray    # [n_fibers] y-position in cross-section (µm)
    fiber_diam:  np.ndarray    # [n_fibers] MRG outer diameter (µm)
    target_mask: np.ndarray    # [n_fibers] bool: True = belongs to a target fascicle
    fascicles:   list[FascicleOutline] = field(default_factory=list)


def _sample_in_disk(rng, n, cx, cy, r, jitter=0.0):
    """Reject-sample n points uniformly inside a disk centred at (cx, cy)
    with radius r.  ``jitter`` shrinks the sample radius slightly so
    fibres don't sit exactly on the fascicle boundary."""
    rs = r * (1.0 - jitter)
    pts = np.empty((n, 2), dtype=np.float64)
    k = 0
    while k < n:
        x = rng.uniform(-rs, rs)
        y = rng.uniform(-rs, rs)
        if x * x + y * y < rs * rs:
            pts[k, 0] = cx + x
            pts[k, 1] = cy + y
            k += 1
    return pts


def make_multi_fascicle_nerve(
    n_fibers_per_fascicle: int = 100,
    nerve_radius_um:       float = 500.0,
    fascicle_radius_um:    float = 130.0,
    fascicle_centers_um:   list[tuple[float, float]] | None = None,
    target_indices:        list[int] | None = None,
    diameters:             list[float] | None = None,
    seed:                  int = 42,
) -> NerveGeometry:
    """Build a synthetic nerve with discrete circular fascicles.

    Default layout is two fascicles placed symmetrically on the x-axis:
    target on the right (+x), off-target on the left (−x).  Pass
    ``fascicle_centers_um`` and ``target_indices`` to set up arbitrary
    multi-fascicle layouts (e.g. 4 fascicles, 2 target).

    Parameters
    ----------
    n_fibers_per_fascicle : int
        Number of fibres placed uniformly inside each fascicle outline.
    nerve_radius_um : float
        Outer nerve radius (used only for visualisation; fibres are bounded
        by their fascicle, not the nerve).
    fascicle_radius_um : float
        Per-fascicle radius.  With the default 2-fascicle layout at
        ±250 µm separation, 130 µm radius gives a comfortable epineurial
        gap and ~17 % per-fascicle area on a 500 µm-radius nerve, close
        to Hussain's typical pig-vagus aspect ratio.
    fascicle_centers_um : list[(x, y)] | None
        Centres of the fascicles in µm.  Default
        ``[(+250, 0), (−250, 0)]``: 2 fascicles, target right / off-target
        left, 500 µm apart edge-to-centre.
    target_indices : list[int] | None
        Indices into ``fascicle_centers_um`` of the target fascicles.
        Default ``[0]``: the first listed fascicle is the target, all
        others are off-target.
    diameters : list[float] | None
        Pool of MRG outer diameters to sample from.  Default
        ``[FIBER_DIAMETER_UM]`` (single 5.7 µm) when called from the
        selectivity scripts.  Pass ``AVAILABLE_DIAMETERS`` for the
        mixed-diameter regime.
    seed : int
        RNG seed.

    Returns
    -------
    NerveGeometry
        with ``fascicles`` populated.
    """
    rng = np.random.default_rng(seed)
    if fascicle_centers_um is None:
        fascicle_centers_um = [(+250.0, 0.0), (-250.0, 0.0)]
    if target_indices is None:
        target_indices = [0]
    if diameters is None:
        diameters = [5.7]
    target_set = set(int(i) for i in target_indices)

    fascicles:    list[FascicleOutline] = []
    positions:    list[np.ndarray]      = []
    target_mask_chunks: list[np.ndarray] = []

    for i, (cx, cy) in enumerate(fascicle_centers_um):
        is_tgt = (i in target_set)
        pts = _sample_in_disk(rng, n_fibers_per_fascicle, cx, cy,
                                fascicle_radius_um, jitter=0.08)
        positions.append(pts)
        target_mask_chunks.append(
            np.full(n_fibers_per_fascicle, is_tgt, dtype=bool),
        )
        fascicles.append(FascicleOutline(
            cx_um=float(cx), cy_um=float(cy), r_um=float(fascicle_radius_um),
            is_target=is_tgt, n_fibers=int(n_fibers_per_fascicle),
        ))

    positions   = np.concatenate(positions, axis=0)
    target_mask = np.concatenate(target_mask_chunks, axis=0)
    n_fibers    = positions.shape[0]
    fiber_diam  = rng.choice(diameters, size=n_fibers).astype(float)

    return NerveGeometry(
        n_fibers=n_fibers,
        fiber_x_um=positions[:, 0],
        fiber_y_um=positions[:, 1],
        fiber_diam=fiber_diam,
        target_mask=target_mask,
        fascicles=fascicles,
    )


def make_synthetic_nerve(
    n_fibers:                int   = 10,
    nerve_radius_um:         float = 500.0,
    target_fraction:         float = 0.3,
    diameters:               list[float] | None = None,
    seed:                    int   = 42,
    fascicle_offset_fraction: float = 0.35,
) -> NerveGeometry:
    """LEGACY: single-disk scatter nerve.  Kept for backward compatibility.

    See module docstring for why this is *not* recommended for new
    selectivity work.  New code should call
    :func:`make_multi_fascicle_nerve` instead.
    """
    rng = np.random.default_rng(seed)
    if diameters is None:
        diameters = AVAILABLE_DIAMETERS

    usable_r = 0.9 * nerve_radius_um
    positions = []
    while len(positions) < n_fibers:
        x = rng.uniform(-usable_r, usable_r)
        y = rng.uniform(-usable_r, usable_r)
        if x ** 2 + y ** 2 < usable_r ** 2:
            positions.append((x, y))
    positions = np.array(positions)

    fiber_diam = rng.choice(diameters, size=n_fibers).astype(float)

    fascicle_r_um  = usable_r * np.sqrt(target_fraction)
    offset_um      = nerve_radius_um * fascicle_offset_fraction
    fascicle_angle = rng.uniform(0.0, 2.0 * np.pi)
    fascicle_cx    = offset_um * np.cos(fascicle_angle)
    fascicle_cy    = offset_um * np.sin(fascicle_angle)

    dx = positions[:, 0] - fascicle_cx
    dy = positions[:, 1] - fascicle_cy
    target_mask = dx ** 2 + dy ** 2 < fascicle_r_um ** 2

    dist2_to_fascicle = dx ** 2 + dy ** 2
    if not target_mask.any():
        target_mask[np.argmin(dist2_to_fascicle)] = True
    if target_mask.all():
        target_mask[np.argmax(dist2_to_fascicle)] = False

    return NerveGeometry(
        n_fibers=n_fibers,
        fiber_x_um=positions[:, 0],
        fiber_y_um=positions[:, 1],
        fiber_diam=fiber_diam,
        target_mask=target_mask,
        fascicles=[FascicleOutline(
            cx_um=float(fascicle_cx), cy_um=float(fascicle_cy),
            r_um=float(fascicle_r_um), is_target=True,
            n_fibers=int(target_mask.sum()),
        )],
    )
