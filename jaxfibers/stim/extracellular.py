"""Extracellular stimulation in Jaxley via the activating-function approximation.

The cable equation under a known extracellular potential V_e(z, t):

    C dV_m/dt + I_ion = (1/Ra) d^2 V_m / dz^2  +  (1/Ra) d^2 V_e / dz^2

In Jaxley (which doesn't have NEURON's `extracellular` mechanism), we treat the
second term as a *per-compartment injected current*, computed once for the
unit-amplitude spatial potential profile, then scaled in time by the stimulus
waveform.  This is the "activating function" of Rattay (1986/1989).

For a non-uniform cable (different L, r per compartment), the right discretisation
is the Kirchhoff sum of axial currents *driven by V_e* at each node:

    I_e(i) = (V_e(i+1) - V_e(i)) / R_ax(i, i+1)
           - (V_e(i)   - V_e(i-1)) / R_ax(i-1, i)

where R_ax(i, j) = (R_a_i * L_i/2 + R_a_j * L_j/2) / (pi * r_eff^2) (Ω).

This injected current has units of A; convert to nA for Jaxley `stimulate`.

The point-source extracellular potential at distance r is:
    V_e(r) = I0 / (4 * pi * sigma * r),     I0 in A, sigma in S/m, r in m, V in V
The PyFibers convention is I0 in mA, in which case V_e comes out in mV directly.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import jax.numpy as jnp


def point_source_potentials_mV(
    section_centers_um: list[float] | np.ndarray,
    src_x_um: float = 0.0,
    src_y_um: float = 1000.0,   # default 1 mm "above" the fiber
    src_z_um: Optional[float] = None,
    i0_mA: float = -1.0,
    sigma_S_m: float = 0.3,
) -> np.ndarray:
    """Static spatial potential profile (mV) at each compartment center.

    Fiber is assumed to run along z; the source is placed at (src_x, src_y, src_z).
    The factor `i0_mA` is the *unit* current magnitude — set it to ±1 mA here and
    scale via the waveform amplitude at run time.
    """
    centers = np.asarray(section_centers_um)
    if src_z_um is None:
        src_z_um = (centers[0] + centers[-1]) / 2.0
    dx = (src_x_um - 0.0) * 1e-6                  # m
    dy = (src_y_um - 0.0) * 1e-6                  # m
    dz = (src_z_um - centers) * 1e-6              # m
    r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    # i0 in mA, r in m, sigma in S/m → V in mV directly.
    return i0_mA / (4.0 * math.pi * sigma_S_m * r)


def activating_currents_nA(
    v_ext_mV: np.ndarray,
    length_um: list[float] | np.ndarray,
    diam_um: list[float] | np.ndarray,
    Ra_ohm_cm: list[float] | np.ndarray,
) -> np.ndarray:
    """Per-compartment equivalent injected current (nA) for the given V_ext profile.

    Computes the activating-function as the Kirchhoff sum of axial currents
    driven by V_ext across the (heterogeneous) cable.  End-compartments are
    sealed (no axial current beyond the boundary).
    """
    v = np.asarray(v_ext_mV)                      # mV
    L = np.asarray(length_um) * 1e-4              # cm
    r = (np.asarray(diam_um) / 2.0) * 1e-4        # cm
    Ra = np.asarray(Ra_ohm_cm)                    # Ω·cm
    n = len(v)

    # Half-segment axial resistance for each compartment (Ω):
    half_R = Ra * (L / 2.0) / (math.pi * r ** 2)  # (n,)

    # Inter-compartment axial resistance between i and i+1:
    R_ax = half_R[:-1] + half_R[1:]               # (n-1,)
    # Axial currents driven by V_ext, from i+1 → i (positive into compartment i):
    # I_e->in = (V_e(i+1) - V_e(i)) / R_ax * (-1)  — sign convention chosen so
    # that a *negative* extracellular potential outside the membrane (cathodic source)
    # gives a *positive* intracellular injected current (depolarizing).
    # Practically: keep the conventional activating-function sign and tune via amp:
    dv = (v[1:] - v[:-1])                         # mV
    i_inter_mA = dv / R_ax                        # mA (V/Ω = A; mV/Ω = mA)
    # Per-compartment net injected current:
    inj_mA = np.zeros(n)
    inj_mA[1:]   += -i_inter_mA      # current arriving at i+1 from i side
    inj_mA[:-1]  +=  i_inter_mA      # current leaving i toward i+1
    # The two lines above implement -d/dz (axial flux): inj(i) = +(I_in(i)) - I_out(i)
    # which is the discrete (1/Ra) d^2 V_e / dz^2.
    # Convert to nA:
    return inj_mA * 1e6


def solve_vpax_static(
    v_ext_mV: np.ndarray,
    length_um: list[float] | np.ndarray,
    xraxial_Mohm_cm: list[float] | np.ndarray,
    xg_S_cm2: list[float] | np.ndarray,
    is_node: list[bool] | np.ndarray,
    fiber_diam_um: float,
) -> np.ndarray:
    """Quasi-static periaxonal voltage V_pax (mV) along the fiber.

    Solves the steady-state linear system for the periaxonal space that is
    equivalent to NEURON's `extracellular` mechanism at each time point:

        G_myelin_i * (V_pax_i - V_e_i)
        + (V_pax_i - V_pax_{i-1}) / R_pax_{i-1,i}
        + (V_pax_i - V_pax_{i+1}) / R_pax_{i,i+1}  = 0

    Boundary conditions: at nodes xg=1e10 S/cm² enforces V_pax ≈ V_e (short
    circuit through the node gap), handled here as Dirichlet BCs for stability.

    Units used internally:
        xraxial [MΩ/cm] × L [µm] × 1e-4 [cm/µm]  →  R [MΩ]  →  G = 1/(R×1e6) [S]
        xg [S/cm²] × π × fiber_diam [cm] × L [cm]  →  G_myelin [S]
    """
    v = np.asarray(v_ext_mV, dtype=float)
    L_um = np.asarray(length_um, dtype=float)
    xr = np.asarray(xraxial_Mohm_cm, dtype=float)
    xg = np.asarray(xg_S_cm2, dtype=float)
    isn = np.asarray(is_node, dtype=bool)
    n = len(v)

    v_pax = np.zeros(n)
    v_pax[isn] = v[isn]             # Dirichlet BCs at nodes

    myel_idx = np.where(~isn)[0]
    n_myel = len(myel_idx)
    if n_myel == 0:
        return v_pax

    # Myelin conductance per section [S]:
    fib_cm = fiber_diam_um * 1e-4   # cm
    L_cm = L_um * 1e-4
    G_myelin = xg * (np.pi * fib_cm * L_cm)   # S/cm² × cm² = S

    # Half-section periaxonal axial resistance [MΩ]:
    half_R_MΩ = xr * (L_cm / 2.0)             # MΩ/cm × cm/2 = MΩ

    # Inter-compartment periaxonal conductance [S]:
    R_inter_MΩ = half_R_MΩ[:-1] + half_R_MΩ[1:]
    G_inter = 1.0 / (R_inter_MΩ * 1e6)        # 1/Ω = S

    # Map global index → myelinated-only index:
    g2m = {int(g): m for m, g in enumerate(myel_idx)}

    A = np.zeros((n_myel, n_myel))
    b = np.zeros(n_myel)

    for m, i in enumerate(myel_idx):
        diag = G_myelin[i]
        rhs = G_myelin[i] * v[i]

        if i > 0:
            gl = G_inter[i - 1]
            diag += gl
            if isn[i - 1]:
                rhs += gl * v_pax[i - 1]   # node BC
            else:
                A[m, g2m[i - 1]] -= gl

        if i < n - 1:
            gr = G_inter[i]
            diag += gr
            if isn[i + 1]:
                rhs += gr * v_pax[i + 1]   # node BC
            else:
                A[m, g2m[i + 1]] -= gr

        A[m, m] = diag
        b[m] = rhs

    v_pax[myel_idx] = np.linalg.solve(A, b)
    return v_pax


def rectangular_waveform(
    t_grid_ms: np.ndarray,
    delay_ms: float,
    pw_ms: float,
    amp: float = 1.0,
) -> np.ndarray:
    """Monophasic rectangular shape sampled on `t_grid_ms`."""
    return np.where((t_grid_ms >= delay_ms) & (t_grid_ms < delay_ms + pw_ms), amp, 0.0)


def biphasic_waveform(
    t_grid_ms: np.ndarray,
    delay_ms: float,
    pw_ms: float,
    amp: float = 1.0,
    inter_phase_ms: float = 0.0,
) -> np.ndarray:
    """Symmetric cathodic-leading biphasic rectangular pulse."""
    w = np.zeros_like(t_grid_ms)
    p1 = (t_grid_ms >= delay_ms) & (t_grid_ms < delay_ms + pw_ms)
    p2_start = delay_ms + pw_ms + inter_phase_ms
    p2 = (t_grid_ms >= p2_start) & (t_grid_ms < p2_start + pw_ms)
    w[p1] = -amp
    w[p2] = +amp
    return w
