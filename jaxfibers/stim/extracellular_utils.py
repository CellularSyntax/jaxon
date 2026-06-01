"""Extracellular stimulation utilities for Jaxley nerve fiber models.

Converts a spatial extracellular voltage field Ve(x, t) into per-compartment
intracellular injection currents using Rattay's activating function, then
injects them into a Jaxley cell via .stimulate().

Physics
-------
In a discrete multicompartment cable, an extracellular field Ve[i](t) exerts
exactly the same effect as injecting an intracellular current:

    I_inj[i](t) = G_ax_left[i]  * (Ve[i-1] - Ve[i])
                + G_ax_right[i] * (Ve[i+1] - Ve[i])

where G_ax_left/right are the inter-compartment axial conductances (µS).
This is zero at uniform Ve and maximal at curvature peaks (nodes of Ranvier).

Units
-----
Ve  in mV, G_ax in µS  →  I_inj = G * ΔV [µS × mV = nA]  ✓
Jaxley .stimulate() expects current in nA.

Note on Ve vs V_pax
-------------------
For the MRG double-cable model, using V_pax (periaxonal voltage, solved by
solve_vpax_static in extracellular.py) is more accurate than raw Ve because
it accounts for myelin filtering at FLUT/STIN sections. Both inputs are
supported here; pass Ve for a single-cable approximation or V_pax for the
double-cable equivalent.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
import jax
import jax.numpy as jnp

from jaxfibers.fibers.mrg import MrgGeometry


# ---------------------------------------------------------------------------
# Axial geometry
# ---------------------------------------------------------------------------

def axial_conductances_uS(geom: MrgGeometry) -> np.ndarray:
    """Inter-compartment axial conductances in µS.

    Returns Gax of shape (n_comp-1,) where Gax[i] is the conductance between
    compartment i and compartment i+1.

    Uses half-resistances: R_half[i] = Ra * (L[i]/2) / (pi * r[i]^2),
    converted from Ohm to MOhm so that G = 1/R is directly in µS.
    """
    L_um = np.array(geom.length_um)      # µm
    r_um = np.array(geom.diam_um) / 2.0  # µm
    Ra   = np.array(geom.Ra_ohm_cm)      # Ω·cm

    # R_half [MΩ] = Ra[Ω·cm] × (L/2)[µm×1e-4 cm/µm] / (π r²[µm²×1e-8 cm²/µm²]) × 1e-6[MΩ/Ω]
    # = Ra × L/2 × 1e-4 / (π r² × 1e-8) × 1e-6
    # = Ra × L/2 / (π r²) × 0.01
    R_half_MOhm = Ra * (L_um / 2.0) / (np.pi * r_um**2) * 0.01

    # Conductance between compartment i and i+1 [µS = 1/MΩ]
    R_ax_MOhm = R_half_MOhm[:-1] + R_half_MOhm[1:]
    Gax_uS = 1.0 / R_ax_MOhm
    return Gax_uS


# ---------------------------------------------------------------------------
# Activating function
# ---------------------------------------------------------------------------

def compute_activating_currents_nA(
    Ve_matrix_mV: Union[np.ndarray, jnp.ndarray],
    geom: MrgGeometry,
) -> jnp.ndarray:
    """Compute Rattay activating currents for all compartments and timesteps.

    Parameters
    ----------
    Ve_matrix_mV : array, shape (n_comp, n_steps)
        Extracellular voltage at each compartment centre over time (mV).
        Pass Ve for a single-cable model, or V_pax (from solve_vpax_static)
        for the MRG double-cable model.
    geom : MrgGeometry
        Compartment geometry of the fiber.

    Returns
    -------
    I_inj : jnp.array, shape (n_comp, n_steps)
        Activating current at each compartment (nA).
        Boundary compartments (ends of fiber) have only one neighbor term.
    """
    Ve = jnp.asarray(Ve_matrix_mV)          # (n_comp, n_steps)
    Gax = jnp.asarray(axial_conductances_uS(geom))  # (n_comp-1,)

    # Forward differences: dVe[i] = Ve[i+1] - Ve[i],  shape (n_comp-1, n_steps)
    dVe = Ve[1:, :] - Ve[:-1, :]

    # Current flowing INTO compartment i FROM its right neighbor (i+1):
    #   = Gax[i] * (Ve[i+1] - Ve[i]) = Gax[i] * dVe[i]
    # Pad with zero for last compartment (no right neighbour).
    contrib_from_right = jnp.pad(Gax[:, None] * dVe, ((0, 1), (0, 0)))

    # Current flowing INTO compartment i FROM its left neighbor (i-1):
    #   = Gax[i-1] * (Ve[i-1] - Ve[i]) = -Gax[i-1] * dVe[i-1]
    # Pad with zero for first compartment (no left neighbour).
    contrib_from_left = jnp.pad(-Gax[:, None] * dVe, ((1, 0), (0, 0)))

    return contrib_from_right + contrib_from_left   # (n_comp, n_steps)  [nA]


def compute_activating_currents_from_point_source_nA(
    geom: MrgGeometry,
    src_x_um: float,
    src_y_um: float,
    src_z_um: float,
    amp_waveform_mA: Union[np.ndarray, jnp.ndarray],
    sigma_S_m: float = 0.3,
) -> jnp.ndarray:
    """Convenience wrapper: point source → activating currents.

    Parameters
    ----------
    geom : MrgGeometry
    src_x_um, src_y_um, src_z_um : electrode position (µm)
    amp_waveform_mA : shape (n_steps,) — stimulus current amplitude over time
    sigma_S_m : extracellular conductivity (default: 0.3 S/m ≈ 333 Ω·cm, saline)

    Returns
    -------
    I_inj : jnp.array, shape (n_comp, n_steps)  [nA]
    """
    from jaxfibers.stim.extracellular import (
        point_source_potentials_mV, solve_vpax_static,
    )
    from jaxfibers.fibers.mrg import section_centers_um

    centers = section_centers_um(geom)
    # Unit Ve for 1 mA source:
    ve_unit = point_source_potentials_mV(
        centers, src_x_um=src_x_um, src_y_um=src_y_um,
        src_z_um=src_z_um, i0_mA=1.0, sigma_S_m=sigma_S_m,
    )
    # Solve double-cable V_pax:
    vpax_unit = solve_vpax_static(
        ve_unit, geom.length_um, geom.xraxial_Mohm_cm,
        geom.xg_myelin_S_cm2, geom.is_node, geom.diameter,
    )

    # Scale by waveform:  V_pax(t) = vpax_unit * amp(t)
    vpax_unit = jnp.asarray(vpax_unit)            # (n_comp,)
    amp = jnp.asarray(amp_waveform_mA)            # (n_steps,)
    Ve_matrix = vpax_unit[:, None] * amp[None, :] # (n_comp, n_steps)

    return compute_activating_currents_nA(Ve_matrix, geom)


# ---------------------------------------------------------------------------
# Jaxley injection helpers
# ---------------------------------------------------------------------------

def apply_activating_currents_to_cell(
    cell,
    I_inj_nA: np.ndarray,
    dt_ms: float,
) -> None:
    """Inject pre-computed activating currents into a Jaxley cell.

    Calls cell.branch(0).comp(i).stimulate(I_inj_nA[i]) for every
    compartment that has a non-zero waveform.

    Parameters
    ----------
    cell : jaxley.Cell
    I_inj_nA : np.ndarray, shape (n_comp, n_steps)
        Output of compute_activating_currents_nA. Must be a plain numpy
        array (not JAX) because Jaxley stores it as a fixed record.
    dt_ms : float
        Simulation time step in ms. The waveform index n corresponds to
        time n*dt_ms in the Jaxley integrator.
    """
    I = np.asarray(I_inj_nA)
    n_comp, n_steps = I.shape
    assert n_comp == cell.total_nbranches * len(cell.branch(0).comp(0).nodes), \
        "I_inj column count must equal number of compartments in the cell."

    for i in range(n_comp):
        if np.any(I[i] != 0.0):
            cell.branch(0).comp(i).stimulate(I[i])


def build_stimuli_for_jx_integrate(
    cell,
    I_inj_nA: np.ndarray,
) -> tuple:
    """Build a data_stimuli tuple for use with jx.integrate(..., data_stimuli=...).

    This is the preferred path when I_inj depends on a trainable parameter
    (e.g., electrode position or amplitude), because it keeps the computation
    inside the JAX graph.

    Parameters
    ----------
    cell : jaxley.Cell  (must have called .to_jax() already)
    I_inj_nA : jnp.ndarray, shape (n_comp, n_steps)

    Returns
    -------
    data_stimuli : tuple compatible with jx.integrate(data_stimuli=...)
    """
    import pandas as pd

    I = jnp.asarray(I_inj_nA)   # (n_comp, n_steps)
    n_comp = I.shape[0]

    # Build the index DataFrame that Jaxley expects:
    # one row per compartment, columns: 'global_comp_index' (or Jaxley's internal index)
    rows = []
    for i in range(n_comp):
        rows.append({"comp_index": i})
    inds_df = pd.DataFrame(rows)

    # data_stimuli = (state_name, current_matrix, index_dataframe)
    return ("i", I, inds_df)
