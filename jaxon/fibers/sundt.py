"""Jaxley reconstruction of the Sundt (2015) unmyelinated C-fiber model.

Single-cable model: every compartment is a uniform active cylinder with no
periaxonal space.  The existing MRG coupled (Vi, Vpax) solver is reused with
`is_node=True` everywhere — this collapses the 2×2 periaxonal row to a trivial
Vp = Ve constraint, reducing the system to the standard single-cable cable equation.

Compartment parameters (verified against PyFibers SUNDT at runtime):
  diameter  = fiber_diameter (µm)  — same inner and outer diam (no myelin)
  length    = delta_z = 8.333 µm   (default; can be overridden)
  Ra        = 100 Ω·cm
  Cm        = 1.0 µF/cm²
  v_rest    = -60 mV

Extracellular coupling: call `arrays_from_geometry(geom_c, dt)` from
`jaxon.stim.extracellular_coupled` (same import as for MRG).  Set
  xraxial_Mohm_cm  = 1e6  (dummy — periaxonal axial never used for nodes)
  xc_myelin_uF_cm2 = 0.0  (no myelin capacitance)
  xg_myelin_S_cm2  = 1e10 (short-circuit; overridden to 0 by the is_node mask)
"""

from __future__ import annotations

from dataclasses import dataclass

import jaxley as jx
from jaxley.channels import Leak

from jaxon.channels.sundt_channels import SundtAxon

# ── constants ──────────────────────────────────────────────────────────────────
V_REST   = -60.0   # mV
CM       =  1.0    # µF/cm²
RA       = 100.0   # Ω·cm
DELTA_Z  =  8.333  # µm  (PyFibers default delta_z for SUNDT)
# Dummy periaxonal values (is_node=True everywhere so they never enter the solve):
_XRAX_DUMMY  = 1e6    # MΩ/cm  — large → Gp ≈ 0 (avoids div/0 in arrays_from_geometry)
_XCMY_DUMMY  = 0.0    # µF/cm²
_XGMY_DUMMY  = 1e10   # S/cm²


@dataclass
class SundtGeometry:
    """Per-compartment parameters for a Sundt C-fiber (MrgGeometry-compatible)."""
    diameter: float          # fiber diameter (µm)
    n_nodes: int
    delta_z: float           # compartment length (µm)
    section_type: list[str]
    length_um: list[float]
    diam_um: list[float]
    Ra_ohm_cm: list[float]
    cm_uF_cm2: list[float]
    gleak_S_cm2: list[float]
    is_node: list[bool]
    xraxial_Mohm_cm: list[float]
    xc_myelin_uF_cm2: list[float]
    xg_myelin_S_cm2: list[float]

    @property
    def n_comp(self) -> int:
        return len(self.section_type)

    @property
    def total_length_um(self) -> float:
        return sum(self.length_um)


def _sundt_geometry(diameter: float, n_nodes: int, delta_z: float) -> SundtGeometry:
    """Build per-compartment arrays for a uniform unmyelinated C-fiber."""
    n = n_nodes
    return SundtGeometry(
        diameter=diameter,
        n_nodes=n,
        delta_z=delta_z,
        section_type=["node"] * n,
        length_um=[delta_z] * n,
        diam_um=[diameter] * n,
        Ra_ohm_cm=[RA] * n,
        cm_uF_cm2=[CM] * n,
        gleak_S_cm2=[0.0] * n,      # leak is inside SundtAxon channel
        is_node=[True] * n,
        xraxial_Mohm_cm=[_XRAX_DUMMY] * n,
        xc_myelin_uF_cm2=[_XCMY_DUMMY] * n,
        xg_myelin_S_cm2=[_XGMY_DUMMY] * n,
    )


def build_sundt(
    diameter: float = 0.8,
    n_nodes: int = 21,
    temperature: float = 37.0,
    delta_z: float = DELTA_Z,
) -> tuple[jx.Cell, SundtGeometry]:
    """Build a Sundt C-fiber as a Jaxley Cell (single straight branch).

    Parameters
    ----------
    diameter  : fiber diameter (µm); 0.8 µm is the standard Sundt/PyFibers default
    n_nodes   : number of compartments
    temperature : simulation temperature (°C); passed to SundtAxon_celsius
    delta_z   : compartment spacing (µm)

    Returns
    -------
    cell : jaxley.Cell
    geom : SundtGeometry
    """
    geom = _sundt_geometry(diameter=diameter, n_nodes=n_nodes, delta_z=delta_z)
    branch = jx.Branch(jx.Compartment(), ncomp=geom.n_comp)
    cell = jx.Cell(branch, parents=[-1])

    for i in range(geom.n_comp):
        comp = cell.branch(0).comp(i)
        comp.set("length",           geom.length_um[i])
        comp.set("radius",           geom.diam_um[i] / 2.0)
        comp.set("axial_resistivity", geom.Ra_ohm_cm[i])
        comp.set("capacitance",      geom.cm_uF_cm2[i])
        comp.set("v", V_REST)

    # Insert all channels in one sweep (Jaxley quirk: no .set() on same view right after .insert())
    for i in range(geom.n_comp):
        cell.branch(0).comp(i).insert(SundtAxon())

    # Set temperature in second sweep
    for i in range(geom.n_comp):
        cell.branch(0).comp(i).set("SundtAxon_celsius", temperature)

    return cell, geom


def node_indices(geom: SundtGeometry) -> list[int]:
    """All compartment indices (every section is active in Sundt)."""
    return list(range(geom.n_comp))


def section_centers_um(geom: SundtGeometry) -> list[float]:
    """Center-of-compartment position along the fiber (µm), starting at 0."""
    centers, z = [], 0.0
    for L in geom.length_um:
        centers.append(z + L / 2.0)
        z += L
    return centers
