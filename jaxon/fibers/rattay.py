"""Jaxley reconstruction of the Rattay (1993) unmyelinated C-fiber model.

Single-cable model identical in geometry to the Sundt fiber: every compartment
is a uniform active cylinder. The MRG coupled (Vi, Vpax) solver is reused with
is_node=True everywhere (trivial periaxonal collapse, Vp = Ve).

Compartment parameters (from RattayAberham.mod / rattay.py in PyFibers):
  diameter  = fiber_diameter (µm)
  length    = delta_z = 8.333 µm  (PyFibers default)
  Ra        = 100 Ω·cm
  Cm        = 1.0 µF/cm²
  V_rest    = -70 mV

Channels: m³h Na  (gnabar=0.12)  |  n⁴ K (gkbar=0.036)  |  leak (gl=3e-4)
"""

from __future__ import annotations

from dataclasses import dataclass

import jaxley as jx

from jaxon.channels.rattay_channels import RattayHH

# ── constants ──────────────────────────────────────────────────────────────────
V_REST  = -70.0   # mV
CM      =   1.0   # µF/cm²
RA      = 100.0   # Ω·cm
DELTA_Z =   8.333 # µm

_XRAX_DUMMY = 1e6   # MΩ/cm — large → Gp ≈ 0 (periaxonal not used)
_XCMY_DUMMY = 0.0   # µF/cm²
_XGMY_DUMMY = 1e10  # S/cm²


@dataclass
class RattayGeometry:
    """Per-compartment parameters for a Rattay C-fiber (MrgGeometry-compatible)."""
    diameter: float
    n_nodes: int
    delta_z: float
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


def _rattay_geometry(diameter: float, n_nodes: int, delta_z: float) -> RattayGeometry:
    n = n_nodes
    return RattayGeometry(
        diameter=diameter,
        n_nodes=n,
        delta_z=delta_z,
        section_type=["node"] * n,
        length_um=[delta_z] * n,
        diam_um=[diameter] * n,
        Ra_ohm_cm=[RA] * n,
        cm_uF_cm2=[CM] * n,
        gleak_S_cm2=[0.0] * n,
        is_node=[True] * n,
        xraxial_Mohm_cm=[_XRAX_DUMMY] * n,
        xc_myelin_uF_cm2=[_XCMY_DUMMY] * n,
        xg_myelin_S_cm2=[_XGMY_DUMMY] * n,
    )


def build_rattay(
    diameter: float = 0.8,
    n_nodes: int = 21,
    temperature: float = 37.0,
    delta_z: float = DELTA_Z,
) -> tuple[jx.Cell, RattayGeometry]:
    """Build a Rattay C-fiber as a Jaxley Cell (single straight branch).

    Parameters
    ----------
    diameter    : fiber diameter (µm)
    n_nodes     : number of compartments
    temperature : simulation temperature (°C)
    delta_z     : compartment spacing (µm)

    Returns
    -------
    cell : jaxley.Cell
    geom : RattayGeometry
    """
    geom = _rattay_geometry(diameter=diameter, n_nodes=n_nodes, delta_z=delta_z)
    branch = jx.Branch(jx.Compartment(), ncomp=geom.n_comp)
    cell = jx.Cell(branch, parents=[-1])

    for i in range(geom.n_comp):
        comp = cell.branch(0).comp(i)
        comp.set("length",            geom.length_um[i])
        comp.set("radius",            geom.diam_um[i] / 2.0)
        comp.set("axial_resistivity", geom.Ra_ohm_cm[i])
        comp.set("capacitance",       geom.cm_uF_cm2[i])
        comp.set("v", V_REST)

    for i in range(geom.n_comp):
        cell.branch(0).comp(i).insert(RattayHH())

    for i in range(geom.n_comp):
        cell.branch(0).comp(i).set("RattayHH_celsius", temperature)

    return cell, geom


def node_indices(geom: RattayGeometry) -> list[int]:
    """All compartment indices (every section is active in Rattay)."""
    return list(range(geom.n_comp))


def section_centers_um(geom: RattayGeometry) -> list[float]:
    """Center-of-compartment position along the fiber (µm), starting at 0."""
    centers, z = [], 0.0
    for L in geom.length_um:
        centers.append(z + L / 2.0)
        z += L
    return centers
