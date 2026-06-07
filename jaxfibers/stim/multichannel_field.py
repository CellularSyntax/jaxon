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
import jax
import jax.numpy as jnp

from jaxfibers.stim.extracellular import point_source_potentials_mV, point_source_potentials_mV_jax
from jaxfibers.fibers.mrg     import section_centers_um as _mrg_centers
from jaxfibers.fibers.mrg     import _mrg_geometry, _mrg_interp_geometry
from jaxfibers.fibers.sundt   import section_centers_um as _sundt_centers
from jaxfibers.fibers.sundt   import _sundt_geometry
from jaxfibers.fibers.sweeney import section_centers_um as _sweeney_centers
from jaxfibers.fibers.sweeney import _sweeney_geometry
from jaxfibers.fibers.rattay  import section_centers_um as _rattay_centers
from jaxfibers.fibers.rattay  import _rattay_geometry
from jaxfibers.nerve.geometry import NerveGeometry


# Model dispatch for cross-model selectivity sweeps. Each entry maps
# the model name (lowercase) to (geometry_factory, centers_fn). The
# geometry factory takes (diameter, n_nodes) and returns a *Geometry
# dataclass; centers_fn(geom) returns the per-compartment z positions
# in µm. Use the same `section_centers_um` name everywhere so the
# call sites here don't have to dispatch on model.
_MODEL_FACTORIES = {
    "mrg":     (lambda d, n: _mrg_geometry(d, n),       _mrg_centers),
    "mrg_interp": (lambda d, n: _mrg_interp_geometry(d, n), _mrg_centers),
    "sundt":   (lambda d, n: _sundt_geometry(d, n, 8.333), _sundt_centers),
    "sweeney": (lambda d, n: _sweeney_geometry(d, n),  _sweeney_centers),
    "rattay":  (lambda d, n: _rattay_geometry(d, n, 8.333), _rattay_centers),
}

# Default for back-compat: callers that don't pass `model=` get MRG.
def section_centers_um(geom):
    """Back-compat shim: returns the per-comp z centers for any
    supported model.  Dispatches on the geometry's class name."""
    cls = type(geom).__name__
    if cls == "MrgGeometry":      return _mrg_centers(geom)
    if cls == "SundtGeometry":    return _sundt_centers(geom)
    if cls == "SweeneyGeometry":  return _sweeney_centers(geom)
    if cls == "RattayGeometry":   return _rattay_centers(geom)
    raise TypeError(f"section_centers_um: unknown geom class {cls!r}")


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
    model: str = "mrg",
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

    # Resolve the per-model geometry factory.  use_interp=True forces the
    # MRG interpolated variant regardless of `model`; otherwise model="mrg"
    # uses the discrete MRG table.
    key = "mrg_interp" if use_interp else model.lower()
    if key not in _MODEL_FACTORIES:
        raise ValueError(
            f"precompute_ve_unit: unknown model {model!r}. "
            f"Supported: {sorted(_MODEL_FACTORIES)}"
        )
    geom_factory, centers_fn = _MODEL_FACTORIES[key]

    geoms = []
    all_centers = []
    all_node_idx = []
    for f in range(n_fibers):
        d = float(nerve_geom.fiber_diam[f])
        geom = geom_factory(d, n_nodes)
        geoms.append(geom)
        centers = np.array(centers_fn(geom))
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


def build_fiber_arrays(
    nerve_geom: NerveGeometry,
    geoms: list,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract JAX arrays needed for differentiable field computation.

    Returns
    -------
    fiber_xy_um   : [n_fibers, 2]      — fiber (x, y) positions in µm.
    all_centers_um: [n_fibers, n_comp] — compartment z-centers in µm.
    """
    fiber_xy = jnp.stack([
        jnp.asarray(nerve_geom.fiber_x_um, dtype=jnp.float64),
        jnp.asarray(nerve_geom.fiber_y_um, dtype=jnp.float64),
    ], axis=1)                                           # [n_fibers, 2]
    all_centers = jnp.stack([
        jnp.asarray(section_centers_um(g), dtype=jnp.float64) for g in geoms
    ])                                                   # [n_fibers, n_comp]
    return fiber_xy, all_centers


def compute_ve_unit_jax(
    fiber_xy_um: jnp.ndarray,
    all_centers_um: jnp.ndarray,
    contact_xyz_um: jnp.ndarray,
    sigma_S_m: float = 0.3,
) -> jnp.ndarray:
    """JAX-native Ve_unit — differentiable w.r.t. contact_xyz_um.

    Parameters
    ----------
    fiber_xy_um   : [n_fibers, 2]
    all_centers_um: [n_fibers, n_comp]
    contact_xyz_um: [K, 3]

    Returns
    -------
    Ve_unit : [K, n_fibers, n_comp]  (mV at i0 = -1 mA)
    """
    def _one_contact_fiber(contact_xyz, fiber_xy, centers):
        return point_source_potentials_mV_jax(
            centers, contact_xyz, fiber_xy, i0_mA=-1.0, sigma_S_m=sigma_S_m
        )

    _over_fibers  = jax.vmap(_one_contact_fiber, in_axes=(None, 0, 0))
    _over_contacts = jax.vmap(_over_fibers,      in_axes=(0, None, None))
    return _over_contacts(contact_xyz_um, fiber_xy_um, all_centers_um)  # [K, n_fibers, n_comp]
