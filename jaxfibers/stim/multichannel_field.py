"""Multi-contact extracellular field precomputation for ring cuff electrodes.

Workflow
--------
1. call make_ring_cuff_positions() to get K contact positions.
2. call precompute_ve_unit() to get Ve_unit[K, n_fibers, n_comp].
3. Build Ve_seq at run time:
   - Rect:  Ve_seq[f,t,:] = pulse_mask[t] * sum_k(amps[k] * Ve_unit[k,f,:])
   - Arb:   Ve_seq[f,t,:] = sum_k(u[k,t]  * Ve_unit[k,f,:])
"""
from __future__ import annotations

import numpy as np

from jaxfibers.stim.extracellular import point_source_potentials_mV
from jaxfibers.fibers.mrg import section_centers_um, _mrg_geometry, _mrg_interp_geometry
from jaxfibers.nerve.geometry import NerveGeometry


def make_ring_cuff_positions(
    n_contacts: int = 6,
    cuff_radius_um: float = 1000.0,
    cuff_z_um: float = 0.0,
) -> np.ndarray:
    """Electrode contact positions for a symmetric ring cuff.

    Returns
    -------
    contact_xyz : [K, 3] float64
        (x, y, z) positions of each contact in µm.
        Contacts are evenly spaced around a ring in the x-y plane at z = cuff_z_um.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_contacts, endpoint=False)
    x = cuff_radius_um * np.cos(angles)
    y = cuff_radius_um * np.sin(angles)
    z = np.full(n_contacts, cuff_z_um)
    return np.stack([x, y, z], axis=1)   # [K, 3]


def precompute_ve_unit(
    nerve_geom: NerveGeometry,
    n_nodes: int,
    contact_xyz_um: np.ndarray,
    sigma_S_m: float = 0.3,
    use_interp: bool = False,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Pre-compute unit extracellular potential for every (contact, fiber, compartment).

    For arbitrary-waveform optimization the full Ve time sequence is obtained as:
        Ve_seq[f, t, n] = sum_k  u[k, t] * Ve_unit[k, f, n]

    Parameters
    ----------
    nerve_geom : NerveGeometry
        Cross-sectional geometry (fiber positions and diameters).
    n_nodes : int
        Number of nodes per fiber (same for all fibers).
    contact_xyz_um : [K, 3] array
        Electrode contact positions in µm.
    sigma_S_m : float
        Tissue conductivity (S/m).
    use_interp : bool
        Use polynomial-interpolated MRG geometry (_mrg_interp_geometry) instead of
        the discrete lookup table.

    Returns
    -------
    Ve_unit : [K, n_fibers, n_comp] float64
        Unit potential (mV) per contact, fiber, compartment at i0 = -1 mA.
    node_indices : [n_fibers, n_nodes] int32
        Compartment indices of nodes for each fiber.
    geoms : list[MrgGeometry]
        MrgGeometry objects for each fiber (needed to build solver statics).
    """
    K = len(contact_xyz_um)
    n_fibers = nerve_geom.n_fibers

    geoms = []
    all_centers = []
    all_node_idx = []
    for f in range(n_fibers):
        d = float(nerve_geom.fiber_diam[f])
        if use_interp:
            geom = _mrg_interp_geometry(d, n_nodes)
        else:
            geom = _mrg_geometry(d, n_nodes)
        geoms.append(geom)
        centers = np.array(section_centers_um(geom))
        all_centers.append(centers)
        all_node_idx.append([i for i, isn in enumerate(geom.is_node) if isn])

    n_comp = geoms[0].n_comp   # same for all fibers with the same n_nodes

    Ve_unit = np.zeros((K, n_fibers, n_comp), dtype=np.float64)
    for k, (cx, cy, cz) in enumerate(contact_xyz_um):
        for f in range(n_fibers):
            xi = nerve_geom.fiber_x_um[f]
            yi = nerve_geom.fiber_y_um[f]
            Ve_unit[k, f] = point_source_potentials_mV(
                all_centers[f],
                src_x_um=cx - xi,
                src_y_um=cy - yi,
                src_z_um=float(cz),
                i0_mA=-1.0,
                sigma_S_m=sigma_S_m,
            )

    node_indices = np.array(all_node_idx, dtype=np.int32)  # [n_fibers, n_nodes]
    return Ve_unit, node_indices, geoms
