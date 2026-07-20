"""Jaxley reconstruction of the Sweeney (1987) myelinated fiber model.

Node-myelin alternating geometry: (2*n_nodes - 1) sections total.
Channel: SweeneyNode at nodes only (m²h Na + leak); myelin sections passive.

Critical design choices:
  - ALL sections is_node=True  → Vpax = Ve everywhere (PyFibers xg[0]=1e10 for both
    node and myelin sections, i.e. periaxonal space is shorted to bath for all).
  - Myelin cm = 1e-6 µF/cm²   → avoids singular block-Thomas diagonal (PyFibers cm=0
    is mathematically valid but numerically unsafe in the ODE solver).
  - Myelin g_leak = 0          → purely passive (no active conductance in internodes).

Parameters (from sweeney.py / sweeney.mod in PyFibers):
  Node:   diam = D×0.6 µm,  L = 1.5 µm,          cm = 2.5 µF/cm²,   Ra = 54.7 Ω·cm
  Myelin: diam = D×0.6 µm,  L = 100×D - 1.5 µm,  cm = 1e-6 µF/cm²,  Ra = 54.7 Ω·cm
  V_rest = -80 mV
"""

from __future__ import annotations

from dataclasses import dataclass

import jaxley as jx
from jaxley.channels import Leak

from jaxon.channels.sweeney_channels import SweeneyNode

# ── constants ──────────────────────────────────────────────────────────────────
V_REST         = -80.0   # mV
RA             =  54.7   # Ω·cm
CM_NODE        =   2.5   # µF/cm²
CM_MYELIN      =   1e-6  # µF/cm²  (numerical safety replacement for 0)
NODE_L         =   1.5   # µm
DIAM_FACTOR    =   0.6   # inner_diam = fiber_diam × 0.6

_XRAX_DUMMY = 1e6   # MΩ/cm — large → Gp ≈ 0 (periaxonal not used for is_node=True)
_XCMY_DUMMY = 0.0   # µF/cm²
_XGMY_DUMMY = 1e10  # S/cm²


@dataclass
class SweeneyGeometry:
    """Per-compartment parameters for a Sweeney myelinated fiber (MrgGeometry-compatible)."""
    diameter: float
    n_nodes: int
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
    def delta_z(self) -> float:
        return self.diameter * 100.0

    @property
    def total_length_um(self) -> float:
        return sum(self.length_um)


def _sweeney_geometry(diameter: float, n_nodes: int) -> SweeneyGeometry:
    inner_d  = diameter * DIAM_FACTOR
    myelin_L = diameter * 100.0 - NODE_L    # internodal spacing minus node length

    section_type, length_um, diam_um = [], [], []
    Ra_arr, cm_arr, gl_arr, is_node_arr = [], [], [], []
    xrax_arr, xc_arr, xg_arr = [], [], []

    for k in range(n_nodes):
        # Node
        section_type.append("node"); length_um.append(NODE_L); diam_um.append(inner_d)
        Ra_arr.append(RA); cm_arr.append(CM_NODE); gl_arr.append(0.0)
        is_node_arr.append(True)
        xrax_arr.append(_XRAX_DUMMY); xc_arr.append(_XCMY_DUMMY); xg_arr.append(_XGMY_DUMMY)

        # Myelin — not after the last node
        if k < n_nodes - 1:
            section_type.append("myelin"); length_um.append(myelin_L); diam_um.append(inner_d)
            Ra_arr.append(RA); cm_arr.append(CM_MYELIN); gl_arr.append(0.0)
            is_node_arr.append(True)
            xrax_arr.append(_XRAX_DUMMY); xc_arr.append(_XCMY_DUMMY); xg_arr.append(_XGMY_DUMMY)

    return SweeneyGeometry(
        diameter=diameter, n_nodes=n_nodes,
        section_type=section_type, length_um=length_um, diam_um=diam_um,
        Ra_ohm_cm=Ra_arr, cm_uF_cm2=cm_arr, gleak_S_cm2=gl_arr,
        is_node=is_node_arr,
        xraxial_Mohm_cm=xrax_arr,
        xc_myelin_uF_cm2=xc_arr,
        xg_myelin_S_cm2=xg_arr,
    )


def build_sweeney(
    diameter: float = 10.0,
    n_nodes: int = 21,
    temperature: float = 37.0,
) -> tuple[jx.Cell, SweeneyGeometry]:
    """Build a Sweeney myelinated fiber as a Jaxley Cell (single straight branch).

    Parameters
    ----------
    diameter    : fiber diameter (µm)
    n_nodes     : number of nodes of Ranvier
    temperature : simulation temperature (°C) — unused (Sweeney has no Q10)

    Returns
    -------
    cell : jaxley.Cell
    geom : SweeneyGeometry
    """
    geom = _sweeney_geometry(diameter=diameter, n_nodes=n_nodes)
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
        comp = cell.branch(0).comp(i)
        if geom.section_type[i] == "node":
            comp.insert(SweeneyNode())
        else:
            comp.insert(Leak())

    for i in range(geom.n_comp):
        if geom.section_type[i] == "myelin":
            comp = cell.branch(0).comp(i)
            comp.set("Leak_gLeak", 0.0)
            comp.set("Leak_eLeak", V_REST)

    return cell, geom


def node_indices(geom: SweeneyGeometry) -> list[int]:
    """Compartment indices of nodes of Ranvier."""
    return [i for i, st in enumerate(geom.section_type) if st == "node"]


def section_centers_um(geom: SweeneyGeometry) -> list[float]:
    """Center-of-compartment position along the fiber (µm), starting at 0."""
    centers, z = [], 0.0
    for L in geom.length_um:
        centers.append(z + L / 2.0)
        z += L
    return centers
