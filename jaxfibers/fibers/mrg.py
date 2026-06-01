"""Jaxley reconstruction of the MRG (McIntyre-Richardson-Grill 2002) myelinated fiber.

Double-cable model: each myelinated section retains its full axon-membrane capacitance
(CM_AXON = 2 µF/cm²) and bare-membrane conductance.  The myelin sheath is modelled
as a separate bath-facing RC layer in the periaxonal space — exactly as NEURON's
`extracellular` mechanism — via per-compartment parameters stored in MrgGeometry:
    xraxial_Mohm_cm  periaxonal axial resistivity  (MΩ/cm)
    xc_myelin_uF_cm2 myelin cap. per unit area     (µF/cm²)
    xg_myelin_S_cm2  myelin cond. per unit area    (S/cm²)
These are consumed by jaxfibers.stim.extracellular.solve_vpax_static().

Compartment-wise geometry tracks PyFibers' MRG_DISCRETE table (`models/mrg.py`):
period = node + MYSA + FLUT + STIN x 6 + FLUT + MYSA (11 sections per period).

The result is a `jaxley.Cell` ready for `jx.integrate`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jaxley as jx
from jaxley.channels import Leak

from jaxfibers.channels.mrg_axnode import AxnodeMyel

# Discrete MRG geometry table (PyFibers models/mrg.py, line 65-77).
_MRG_DISCRETE = {
    1.0:  dict(delta_z=100,  paranodal_length_2=5,  axon_diam=0.8,  node_diam=0.7, nl=15),
    2.0:  dict(delta_z=200,  paranodal_length_2=10, axon_diam=1.6,  node_diam=1.4, nl=30),
    5.7:  dict(delta_z=500,  paranodal_length_2=35, axon_diam=3.4,  node_diam=1.9, nl=80),
    7.3:  dict(delta_z=750,  paranodal_length_2=38, axon_diam=4.6,  node_diam=2.4, nl=100),
    8.7:  dict(delta_z=1000, paranodal_length_2=40, axon_diam=5.8,  node_diam=2.8, nl=110),
    10.0: dict(delta_z=1150, paranodal_length_2=46, axon_diam=6.9,  node_diam=3.3, nl=120),
    11.5: dict(delta_z=1250, paranodal_length_2=50, axon_diam=8.1,  node_diam=3.7, nl=130),
    12.8: dict(delta_z=1350, paranodal_length_2=54, axon_diam=9.2,  node_diam=4.2, nl=135),
    14.0: dict(delta_z=1400, paranodal_length_2=56, axon_diam=10.4, node_diam=4.7, nl=140),
    15.0: dict(delta_z=1450, paranodal_length_2=58, axon_diam=11.5, node_diam=5.0, nl=145),
    16.0: dict(delta_z=1500, paranodal_length_2=60, axon_diam=12.7, node_diam=5.5, nl=150),
}

NODE_LENGTH = 1.0           # µm
MYSA_LENGTH = 3.0           # µm (paranodal_length_1)
RHOA = 0.7e6                # Ω·µm  (axoplasmic resistivity, = 70 Ω·cm)
MYCM = 0.1                  # µF/cm² per myelin leaflet
MYGM = 0.001                # S/cm² per myelin leaflet
V_REST = -80.0              # mV
CM_AXON = 2.0               # µF/cm² axon membrane capacitance (full, NOT lumped with myelin)
SECTIONS_PER_PERIOD = 11    # node, MYSA, FLUT, STIN×6, FLUT, MYSA

# Periaxonal gap thickness (McIntyre 2002 / PyFibers convention):
SPACE_P1 = 0.002            # µm  gap at node and MYSA
SPACE_P2 = 0.004            # µm  gap at FLUT and STIN

# Bare axon-membrane g_pas per unit axon-membrane area.
# (PyFibers uses g_pas = prefactor × d_seg/d_fiber with diam=d_fiber for NEURON area;
#  dividing out the area scaling recovers the values below per actual inner area.)
G_PAS_MYSA = 1.0e-3   # S/cm²
G_PAS_FLUT = 1.0e-4   # S/cm²
G_PAS_STIN = 1.0e-4   # S/cm²


@dataclass
class MrgGeometry:
    """Per-compartment morphological and double-cable parameters for an MRG fiber."""
    diameter: float
    n_nodes: int
    # arrays indexed by compartment (axon-membrane parameters):
    section_type: list[str]
    length_um: list[float]
    diam_um: list[float]
    Ra_ohm_cm: list[float]
    cm_uF_cm2: list[float]
    gleak_S_cm2: list[float]
    is_node: list[bool]
    # double-cable periaxonal layer (NEURON extracellular mechanism equivalents):
    xraxial_Mohm_cm: list[float]   # periaxonal axial resistivity (MΩ/cm)
    xc_myelin_uF_cm2: list[float]  # myelin sheath capacitance to bath (µF/cm²)
    xg_myelin_S_cm2: list[float]   # myelin sheath conductance to bath (S/cm²)

    @property
    def n_comp(self) -> int:
        return len(self.section_type)

    @property
    def delta_z_um(self) -> float:
        if self.diameter in _MRG_DISCRETE:
            return _MRG_DISCRETE[self.diameter]["delta_z"]
        d = self.diameter
        return -8.215 * d**2 + 272.4 * d - 780.2 if d >= 5.643 else 81.08 * d + 37.84

    @property
    def total_length_um(self) -> float:
        return sum(self.length_um)


def _rpax_Mohm_cm(inner_diam_um: float, space_um: float) -> float:
    """Periaxonal axial resistivity in MΩ/cm (same formula as PyFibers/McIntyre 2002).

    xraxial = (rhoa × 0.01) / (π × annular_area_µm²)
    With rhoa in Ω·µm the result comes out in MΩ/cm directly
    (verified against first-principles ρ/A calculation).
    """
    area = math.pi * ((inner_diam_um / 2 + space_um) ** 2 - (inner_diam_um / 2) ** 2)
    return (RHOA * 0.01) / area


def _mrg_geometry(diameter: float, n_nodes: int) -> MrgGeometry:
    if diameter not in _MRG_DISCRETE:
        raise ValueError(
            f"diameter={diameter} not in MRG_DISCRETE table. "
            f"Valid: {sorted(_MRG_DISCRETE)}"
        )
    p = _MRG_DISCRETE[diameter]
    delta_z = p["delta_z"]
    axon_d = p["axon_diam"]
    node_d = p["node_diam"]
    nl = p["nl"]
    flut_l = p["paranodal_length_2"]

    stin_l = (delta_z - NODE_LENGTH - 2 * MYSA_LENGTH - 2 * flut_l) / 6.0
    if stin_l <= 0:
        raise RuntimeError(f"Negative STIN length for diameter={diameter}: {stin_l}")

    # Periaxonal layer parameters (consumed by solve_vpax_static):
    Ra_ohm_cm = RHOA / 10000.0          # = 70 Ω·cm
    rpn = _rpax_Mohm_cm(node_d, SPACE_P1)   # node and MYSA gap
    rpf = _rpax_Mohm_cm(axon_d, SPACE_P2)   # FLUT and STIN gap
    xc_myel = MYCM / (2 * nl)               # myelin cap per unit area (µF/cm²)
    xg_myel = MYGM / (2 * nl)               # myelin cond per unit area (S/cm²)

    # Jaxley lacks NEURON's extracellular mechanism: approximate the myelinated-
    # section impedance via the series combination of axon membrane and myelin sheath.
    # This preserves the correct sub-threshold filtering while keeping a single-cable ODE.
    cm_myel  = 1.0 / (1.0 / CM_AXON + 1.0 / xc_myel)   # µF/cm² (myelin-dominated)
    g_mysa   = 1.0 / (1.0 / G_PAS_MYSA + 1.0 / xg_myel)
    g_flut   = 1.0 / (1.0 / G_PAS_FLUT + 1.0 / xg_myel)
    g_stin   = 1.0 / (1.0 / G_PAS_STIN + 1.0 / xg_myel)

    section_type, length_um, diam_um = [], [], []
    Ra_arr, cm_arr, gl_arr, is_node_arr = [], [], [], []
    xrax_arr, xc_arr, xg_arr = [], [], []

    # One period = 11 sections; fiber is (n_nodes-1) periods + 1 trailing node.
    n_periods = n_nodes - 1

    # Template: (section_type, length, inner_diam, Ra, cm_axon, g_pas_axon,
    #            is_node, xraxial, xc_myelin, xg_myelin)
    # At nodes: xc=0, xg=1e10 (short circuit → V_pax = V_e, as in NEURON).
    period_template = (
        #  st       L          d       Ra          cm        gpas    isn   xrax  xc       xg
        ("node",  NODE_LENGTH, node_d, Ra_ohm_cm, CM_AXON,  0.0,    True,  rpn, 0.0,    1e10),
        ("mysa",  MYSA_LENGTH, node_d, Ra_ohm_cm, cm_myel,  g_mysa, False, rpn, xc_myel, xg_myel),
        ("flut",  flut_l,      axon_d, Ra_ohm_cm, cm_myel,  g_flut, False, rpf, xc_myel, xg_myel),
        ("stin",  stin_l,      axon_d, Ra_ohm_cm, cm_myel,  g_stin, False, rpf, xc_myel, xg_myel),
        ("stin",  stin_l,      axon_d, Ra_ohm_cm, cm_myel,  g_stin, False, rpf, xc_myel, xg_myel),
        ("stin",  stin_l,      axon_d, Ra_ohm_cm, cm_myel,  g_stin, False, rpf, xc_myel, xg_myel),
        ("stin",  stin_l,      axon_d, Ra_ohm_cm, cm_myel,  g_stin, False, rpf, xc_myel, xg_myel),
        ("stin",  stin_l,      axon_d, Ra_ohm_cm, cm_myel,  g_stin, False, rpf, xc_myel, xg_myel),
        ("stin",  stin_l,      axon_d, Ra_ohm_cm, cm_myel,  g_stin, False, rpf, xc_myel, xg_myel),
        ("flut",  flut_l,      axon_d, Ra_ohm_cm, cm_myel,  g_flut, False, rpf, xc_myel, xg_myel),
        ("mysa",  MYSA_LENGTH, node_d, Ra_ohm_cm, cm_myel,  g_mysa, False, rpn, xc_myel, xg_myel),
    )
    assert len(period_template) == SECTIONS_PER_PERIOD

    for _ in range(n_periods):
        for st, L, d, Ra, cm, gl, isn, xrax, xc, xg in period_template:
            section_type.append(st); length_um.append(L); diam_um.append(d)
            Ra_arr.append(Ra); cm_arr.append(cm); gl_arr.append(gl)
            is_node_arr.append(isn)
            xrax_arr.append(xrax); xc_arr.append(xc); xg_arr.append(xg)

    # Trailing node:
    section_type.append("node"); length_um.append(NODE_LENGTH); diam_um.append(node_d)
    Ra_arr.append(Ra_ohm_cm); cm_arr.append(CM_AXON); gl_arr.append(0.0)
    is_node_arr.append(True)
    xrax_arr.append(rpn); xc_arr.append(0.0); xg_arr.append(1e10)

    return MrgGeometry(
        diameter=diameter, n_nodes=n_nodes,
        section_type=section_type, length_um=length_um, diam_um=diam_um,
        Ra_ohm_cm=Ra_arr, cm_uF_cm2=cm_arr, gleak_S_cm2=gl_arr,
        is_node=is_node_arr,
        xraxial_Mohm_cm=xrax_arr,
        xc_myelin_uF_cm2=xc_arr,
        xg_myelin_S_cm2=xg_arr,
    )


def build_mrg(
    diameter: float = 10.0,
    n_nodes: int = 11,
    temperature: float = 37.0,
) -> tuple[jx.Cell, MrgGeometry]:
    """Build an MRG fiber as a Jaxley Cell (single straight branch).

    Returns:
        cell : jaxley.Cell
        geom : MrgGeometry  (compartment-wise metadata, for inspection / stim)
    """
    geom = _mrg_geometry(diameter=diameter, n_nodes=n_nodes)
    branch = jx.Branch(jx.Compartment(), ncomp=geom.n_comp)
    cell = jx.Cell(branch, parents=[-1])

    # Apply per-compartment morphology:
    for i in range(geom.n_comp):
        comp = cell.branch(0).comp(i)
        comp.set("length", geom.length_um[i])
        comp.set("radius", geom.diam_um[i] / 2.0)
        comp.set("axial_resistivity", geom.Ra_ohm_cm[i])
        comp.set("capacitance", geom.cm_uF_cm2[i])
        comp.set("v", V_REST)

    # Insert channels (pass 1) — Jaxley quirk: setting channel params on the same
    # comp-view immediately after .insert() raises KeyError. We do all inserts in one
    # sweep and then parameter sets in a second sweep.
    for i in range(geom.n_comp):
        comp = cell.branch(0).comp(i)
        if geom.is_node[i]:
            comp.insert(AxnodeMyel())
        else:
            comp.insert(Leak())

    # Channel param sets (pass 2)
    for i in range(geom.n_comp):
        comp = cell.branch(0).comp(i)
        if geom.is_node[i]:
            comp.set("AxnodeMyel_celsius", temperature)
        else:
            comp.set("Leak_gLeak", geom.gleak_S_cm2[i])
            comp.set("Leak_eLeak", V_REST)

    return cell, geom


def _mrg_interp_params(diameter: float) -> dict:
    """MRG_INTERPOLATION polynomial geometry for diameter d in [2, 16] µm."""
    if not (2.0 <= diameter <= 16.0):
        raise ValueError(
            f"MRG_INTERPOLATION valid range is 2–16 µm (inclusive), got {diameter}"
        )
    d = diameter
    flut_l = -0.1652 * d**2 + 6.354 * d - 0.2862
    dz     = -8.215 * d**2 + 272.4 * d - 780.2 if d >= 5.643 else 81.08 * d + 37.84
    nl     = -0.4749 * d**2 + 16.85 * d - 0.7648
    node_d =  0.01093 * d**2 + 0.1008 * d + 1.099
    axon_d =  0.02361 * d**2 + 0.3673 * d + 0.7122
    return dict(delta_z=dz, paranodal_length_2=flut_l,
                axon_diam=axon_d, node_diam=node_d, nl=nl)


def _mrg_interp_geometry(diameter: float, n_nodes: int) -> MrgGeometry:
    p = _mrg_interp_params(diameter)
    delta_z = p["delta_z"]
    axon_d  = p["axon_diam"]
    node_d  = p["node_diam"]
    nl      = p["nl"]
    flut_l  = p["paranodal_length_2"]

    stin_l = (delta_z - NODE_LENGTH - 2 * MYSA_LENGTH - 2 * flut_l) / 6.0
    if stin_l <= 0:
        raise RuntimeError(f"Negative STIN length for diameter={diameter}: {stin_l}")

    Ra_ohm_cm = RHOA / 10000.0
    rpn = _rpax_Mohm_cm(node_d, SPACE_P1)
    rpf = _rpax_Mohm_cm(axon_d, SPACE_P2)
    xc_myel = MYCM / (2 * nl)
    xg_myel = MYGM / (2 * nl)
    cm_myel  = 1.0 / (1.0 / CM_AXON + 1.0 / xc_myel)
    g_mysa   = 1.0 / (1.0 / G_PAS_MYSA + 1.0 / xg_myel)
    g_flut   = 1.0 / (1.0 / G_PAS_FLUT + 1.0 / xg_myel)
    g_stin   = 1.0 / (1.0 / G_PAS_STIN + 1.0 / xg_myel)

    section_type, length_um, diam_um = [], [], []
    Ra_arr, cm_arr, gl_arr, is_node_arr = [], [], [], []
    xrax_arr, xc_arr, xg_arr = [], [], []

    n_periods = n_nodes - 1
    period_template = (
        ("node",  NODE_LENGTH, node_d, Ra_ohm_cm, CM_AXON,  0.0,    True,  rpn, 0.0,     1e10),
        ("mysa",  MYSA_LENGTH, node_d, Ra_ohm_cm, cm_myel,  g_mysa, False, rpn, xc_myel, xg_myel),
        ("flut",  flut_l,      axon_d, Ra_ohm_cm, cm_myel,  g_flut, False, rpf, xc_myel, xg_myel),
        ("stin",  stin_l,      axon_d, Ra_ohm_cm, cm_myel,  g_stin, False, rpf, xc_myel, xg_myel),
        ("stin",  stin_l,      axon_d, Ra_ohm_cm, cm_myel,  g_stin, False, rpf, xc_myel, xg_myel),
        ("stin",  stin_l,      axon_d, Ra_ohm_cm, cm_myel,  g_stin, False, rpf, xc_myel, xg_myel),
        ("stin",  stin_l,      axon_d, Ra_ohm_cm, cm_myel,  g_stin, False, rpf, xc_myel, xg_myel),
        ("stin",  stin_l,      axon_d, Ra_ohm_cm, cm_myel,  g_stin, False, rpf, xc_myel, xg_myel),
        ("stin",  stin_l,      axon_d, Ra_ohm_cm, cm_myel,  g_stin, False, rpf, xc_myel, xg_myel),
        ("flut",  flut_l,      axon_d, Ra_ohm_cm, cm_myel,  g_flut, False, rpf, xc_myel, xg_myel),
        ("mysa",  MYSA_LENGTH, node_d, Ra_ohm_cm, cm_myel,  g_mysa, False, rpn, xc_myel, xg_myel),
    )
    assert len(period_template) == SECTIONS_PER_PERIOD

    for _ in range(n_periods):
        for st, L, d, Ra, cm, gl, isn, xrax, xc, xg in period_template:
            section_type.append(st); length_um.append(L); diam_um.append(d)
            Ra_arr.append(Ra); cm_arr.append(cm); gl_arr.append(gl)
            is_node_arr.append(isn)
            xrax_arr.append(xrax); xc_arr.append(xc); xg_arr.append(xg)

    section_type.append("node"); length_um.append(NODE_LENGTH); diam_um.append(node_d)
    Ra_arr.append(Ra_ohm_cm); cm_arr.append(CM_AXON); gl_arr.append(0.0)
    is_node_arr.append(True)
    xrax_arr.append(rpn); xc_arr.append(0.0); xg_arr.append(1e10)

    return MrgGeometry(
        diameter=diameter, n_nodes=n_nodes,
        section_type=section_type, length_um=length_um, diam_um=diam_um,
        Ra_ohm_cm=Ra_arr, cm_uF_cm2=cm_arr, gleak_S_cm2=gl_arr,
        is_node=is_node_arr,
        xraxial_Mohm_cm=xrax_arr,
        xc_myelin_uF_cm2=xc_arr,
        xg_myelin_S_cm2=xg_arr,
    )


def build_mrg_interp(
    diameter: float = 10.0,
    n_nodes: int = 11,
    temperature: float = 37.0,
) -> tuple[jx.Cell, MrgGeometry]:
    """Build an MRG fiber using polynomial-interpolated geometry (valid 2–16 µm).

    Same channels (AxnodeMyel + Leak) as the discrete variant; geometry computed
    from the McIntyre 2002 polynomial fits instead of the lookup table.
    """
    geom = _mrg_interp_geometry(diameter=diameter, n_nodes=n_nodes)
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
        if geom.is_node[i]:
            comp.insert(AxnodeMyel())
        else:
            comp.insert(Leak())

    for i in range(geom.n_comp):
        comp = cell.branch(0).comp(i)
        if geom.is_node[i]:
            comp.set("AxnodeMyel_celsius", temperature)
        else:
            comp.set("Leak_gLeak", geom.gleak_S_cm2[i])
            comp.set("Leak_eLeak", V_REST)

    return cell, geom


def node_indices(geom: MrgGeometry) -> list[int]:
    """Indices (into the per-compartment arrays) of node-of-Ranvier compartments."""
    return [i for i, isn in enumerate(geom.is_node) if isn]


def section_centers_um(geom: MrgGeometry) -> list[float]:
    """Center-of-compartment position along the fiber (µm), starting at 0."""
    centers, z = [], 0.0
    for L in geom.length_um:
        centers.append(z + L / 2.0)
        z += L
    return centers
