"""Intracellular stimulation helpers for Jaxley fibers."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import jaxley as jx


def rectangular_pulse(
    t_grid_ms: np.ndarray,
    delay_ms: float,
    pw_ms: float,
    amp_nA: float,
) -> np.ndarray:
    """Monophasic rectangular intracellular pulse trace (nA) at the given t-grid."""
    return np.where((t_grid_ms >= delay_ms) & (t_grid_ms < delay_ms + pw_ms), amp_nA, 0.0)


def attach_intra_pulse(
    cell: jx.Cell,
    comp_idx: int,
    pulse_trace_nA: np.ndarray,
) -> None:
    """Attach a time-varying intracellular current at one compartment of a single-branch cell."""
    cell.branch(0).comp(comp_idx).stimulate(jnp.asarray(pulse_trace_nA))
