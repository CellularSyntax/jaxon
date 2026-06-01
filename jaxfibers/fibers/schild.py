"""Jaxley reconstruction of the Schild (1994/1997) unmyelinated C-fiber models.

Uniform active cable: every compartment is a cylinder of length delta_z with the
full Schild channel suite (naf/nas/kd/ka/kds/kca/can/cat + leaks + three pumps
+ Ca2+ dynamics).

Compartment parameters (from Schild 1994 / schild.py in PyFibers):
  length    = delta_z = 8.333 µm  (PyFibers default)
  Ra        = 100 Ohm.cm
  Cm        = 1.326291192 uF/cm2
  V_rest    = -46.5 mV
  fhspace   = 1 µm  (periaxonal shell for caextscale, overridden in schild.py)

References:
  Schild JH et al (1994) J Neurophysiol 71:2338-2358
  Schild JH & Bhatt DL (1997) J Neurophysiol 78:3198-3209
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jaxley as jx

from jaxfibers.channels.schild_channels import (
    SchildCombined94,
    SchildCombined97,
    ca_geometry,
)

# ── Constants ──────────────────────────────────────────────────────────────────
V_REST   = -46.5           # mV
CM       = 1.326291192     # uF/cm2
RA       = 100.0           # Ohm.cm
DELTA_Z  = 8.333           # um  compartment length
FHSPACE  = 1.0             # um  periaxonal shell (caextscale: fhspace_caextscale=1)

_XRAX_DUMMY = 1e6          # Mohm/cm — large → Gp ≈ 0 (periaxonal space unused)
_XCMY_DUMMY = 0.0          # uF/cm2
_XGMY_DUMMY = 1e10         # S/cm2


@dataclass
class SchildGeometry:
    """Per-compartment parameters for a Schild C-fiber (MrgGeometry-compatible)."""
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


def _schild_geometry(diameter: float, n_nodes: int, delta_z: float) -> SchildGeometry:
    n = n_nodes
    return SchildGeometry(
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


def _set_ca_geometry_params(cell, geom: SchildGeometry, channel_name: str) -> None:
    """Compute and set SA/Vol/Vol_peri channel params for each compartment."""
    for i, (L, d) in enumerate(zip(geom.length_um, geom.diam_um)):
        SA, Vol, Vol_peri = ca_geometry(L, d, fhspace_um=FHSPACE)
        comp = cell.branch(0).comp(i)
        comp.set(f"{channel_name}_SA_cm2",       SA)
        comp.set(f"{channel_name}_Vol_cm3",      Vol)
        comp.set(f"{channel_name}_Vol_peri_cm3", Vol_peri)


def build_schild94(
    diameter: float = 0.8,
    n_nodes:  int   = 21,
    temperature: float = 37.0,
    delta_z:  float = DELTA_Z,
) -> tuple[jx.Cell, SchildGeometry]:
    """Build a Schild 1994 C-fiber as a Jaxley Cell.

    Parameters
    ----------
    diameter    : fiber diameter (um)
    n_nodes     : number of compartments
    temperature : simulation temperature (degC)
    delta_z     : compartment spacing (um)

    Returns
    -------
    cell : jaxley.Cell
    geom : SchildGeometry
    """
    geom   = _schild_geometry(diameter=diameter, n_nodes=n_nodes, delta_z=delta_z)
    branch = jx.Branch(jx.Compartment(), ncomp=geom.n_comp)
    cell   = jx.Cell(branch, parents=[-1])

    for i in range(geom.n_comp):
        comp = cell.branch(0).comp(i)
        comp.set("length",            geom.length_um[i])
        comp.set("radius",            geom.diam_um[i] / 2.0)
        comp.set("axial_resistivity", geom.Ra_ohm_cm[i])
        comp.set("capacitance",       geom.cm_uF_cm2[i])
        comp.set("v", V_REST)

    for i in range(geom.n_comp):
        cell.branch(0).comp(i).insert(SchildCombined94())

    ch_name = "SchildCombined94"
    for i in range(geom.n_comp):
        cell.branch(0).comp(i).set(f"{ch_name}_celsius", temperature)

    _set_ca_geometry_params(cell, geom, ch_name)

    return cell, geom


def build_schild97(
    diameter: float = 0.8,
    n_nodes:  int   = 21,
    temperature: float = 37.0,
    delta_z:  float = DELTA_Z,
) -> tuple[jx.Cell, SchildGeometry]:
    """Build a Schild 1997 C-fiber as a Jaxley Cell."""
    geom   = _schild_geometry(diameter=diameter, n_nodes=n_nodes, delta_z=delta_z)
    branch = jx.Branch(jx.Compartment(), ncomp=geom.n_comp)
    cell   = jx.Cell(branch, parents=[-1])

    for i in range(geom.n_comp):
        comp = cell.branch(0).comp(i)
        comp.set("length",            geom.length_um[i])
        comp.set("radius",            geom.diam_um[i] / 2.0)
        comp.set("axial_resistivity", geom.Ra_ohm_cm[i])
        comp.set("capacitance",       geom.cm_uF_cm2[i])
        comp.set("v", V_REST)

    for i in range(geom.n_comp):
        cell.branch(0).comp(i).insert(SchildCombined97())

    ch_name = "SchildCombined97"
    for i in range(geom.n_comp):
        cell.branch(0).comp(i).set(f"{ch_name}_celsius", temperature)

    _set_ca_geometry_params(cell, geom, ch_name)

    return cell, geom


def node_indices(geom: SchildGeometry) -> list[int]:
    """All compartment indices (every section is active in Schild)."""
    return list(range(geom.n_comp))


def section_centers_um(geom: SchildGeometry) -> list[float]:
    """Center-of-compartment position along the fiber (um), starting at 0."""
    centers, z = [], 0.0
    for L in geom.length_um:
        centers.append(z + L / 2.0)
        z += L
    return centers
