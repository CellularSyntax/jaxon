"""Synthetic nerve cross-section geometry for selectivity optimization experiments."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# MRG discrete diameters available for mixed-fiber nerves.
AVAILABLE_DIAMETERS = [5.7, 7.3, 8.7, 10.0, 11.5, 12.8, 14.0, 15.0, 16.0]


@dataclass
class NerveGeometry:
    """Cross-sectional geometry of a synthetic nerve with mixed MRG fibers."""
    n_fibers: int
    fiber_x_um: np.ndarray    # [n_fibers] x-position in cross-section (µm)
    fiber_y_um: np.ndarray    # [n_fibers] y-position in cross-section (µm)
    fiber_diam: np.ndarray    # [n_fibers] MRG outer diameter (µm)
    target_mask: np.ndarray   # [n_fibers] bool: True = target fascicle


def make_synthetic_nerve(
    n_fibers: int = 10,
    nerve_radius_um: float = 500.0,
    target_fraction: float = 0.3,
    diameters: list[float] | None = None,
    seed: int = 42,
) -> NerveGeometry:
    """Create a synthetic nerve cross-section with randomly placed MRG fibers.

    Fibers are placed uniformly at random within the nerve cross-section.
    Fibers closer to the center (inner circle of area = target_fraction × total)
    are designated as the target fascicle.

    Parameters
    ----------
    n_fibers : int
        Number of fibers.
    nerve_radius_um : float
        Outer radius of the nerve (µm).
    target_fraction : float
        Fraction of nerve area designated as target fascicle (by inner-circle area).
    diameters : list[float] | None
        Pool of MRG fiber diameters to sample from. Defaults to AVAILABLE_DIAMETERS.
    seed : int
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    if diameters is None:
        diameters = AVAILABLE_DIAMETERS

    # Rejection-sample fiber positions inside 90% of nerve radius (avoid edge)
    usable_r = 0.9 * nerve_radius_um
    positions = []
    while len(positions) < n_fibers:
        x = rng.uniform(-usable_r, usable_r)
        y = rng.uniform(-usable_r, usable_r)
        if x ** 2 + y ** 2 < usable_r ** 2:
            positions.append((x, y))
    positions = np.array(positions)  # [n_fibers, 2]

    fiber_diam = rng.choice(diameters, size=n_fibers).astype(float)

    # Target: fibers in the inner circle whose area = target_fraction × nerve area
    inner_r = nerve_radius_um * np.sqrt(target_fraction)
    r2 = positions[:, 0] ** 2 + positions[:, 1] ** 2
    target_mask = r2 < inner_r ** 2

    # Guarantee at least one fiber on each side
    if not target_mask.any():
        target_mask[np.argmin(r2)] = True
    if target_mask.all():
        target_mask[np.argmax(r2)] = False

    return NerveGeometry(
        n_fibers=n_fibers,
        fiber_x_um=positions[:, 0],
        fiber_y_um=positions[:, 1],
        fiber_diam=fiber_diam,
        target_mask=target_mask,
    )
