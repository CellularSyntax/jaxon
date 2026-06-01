"""Custom backward-Euler integrator for the MRG double-cable model.

State variable: V_i (intracellular voltage). Channels see V_m = V_i - V_pax.
This matches NEURON's extracellular mechanism exactly:
  - Axial currents driven by V_i (not V_m)
  - Channel kinetics driven by V_m = V_i - V_pax
  - At onset of a rectangular pulse, Vm = Vi - Vpax_new shows the full
    depolarization from direct voltage coupling (not just activating function)

Cable ODE (in terms of V_i):
    C_m dV_i/dt = G_ax * (V_i_neighbors - V_i) - I_channels(V_i - V_pax) + I_stim

This implicitly includes both the spatial activating function (second diff of Vpax)
AND the capacitive coupling term (-Cm * dVpax/dt) that a Vm-state formulation misses.

V_pax is the quasi-static periaxonal voltage (see solve_vpax_static), precomputed
for a unit-amplitude stimulus and scaled by amp × shape(t) at each timestep.

Units throughout:
    voltage : mV
    current  : nA  (converted from mA/cm² × area_cm² × 1e6)
    time     : ms
    capacitance : nF  (µF/cm² × area_cm² × 1e3)
    conductance : µS  (S/cm² × area_cm² × 1e6)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp

from jaxfibers.fibers.mrg import (
    MrgGeometry, NODE_LENGTH, CM_AXON, RHOA,
    G_PAS_MYSA, G_PAS_FLUT, G_PAS_STIN, V_REST,
)
from jaxfibers.channels.mrg_axnode import AxnodeMyel


# ---------------------------------------------------------------------------
# AxnodeMyel channel equations (inline, JAX-friendly)
# ---------------------------------------------------------------------------

# Default parameters from AxnodeMyel.channel_params:
_GNABAR  = 3.0    # S/cm²
_GNAPBAR = 0.01
_GKBAR   = 0.08
_GL      = 0.007
_ENA     = 50.0   # mV
_EK      = -90.0
_EL      = -90.0


def _axnode_currents_mA_cm2(v, m, h, mp, s):
    """Total ionic current in mA/cm² (outward positive) for a single node."""
    ina  = _GNABAR  * m**3 * h  * (v - _ENA)
    inap = _GNAPBAR * mp**3     * (v - _ENA)
    ik   = _GKBAR   * s         * (v - _EK)
    il   = _GL                  * (v - _EL)
    return ina + inap + ik + il


def _axnode_conductance_mS_cm2(m, h, mp, s):
    """Total chord conductance in S/cm² (= G_total for linearisation)."""
    return _GNABAR * m**3 * h + _GNAPBAR * mp**3 + _GKBAR * s + _GL


def _axnode_Eeff(m, h, mp, s):
    """Effective reversal potential weighted by conductances (mV)."""
    g_na  = _GNABAR  * m**3 * h
    g_nap = _GNAPBAR * mp**3
    g_k   = _GKBAR   * s
    g_l   = _GL
    g_tot = g_na + g_nap + g_k + g_l
    return (g_na * _ENA + g_nap * _ENA + g_k * _EK + g_l * _EL) / g_tot


def _update_gates(v_eff, m, h, mp, s, dt, celsius=37.0):
    """Analytic exponential gate update at voltage v_eff for one step dt (ms)."""
    (a_m, b_m), (a_h, b_h), (a_mp, b_mp), (a_s, b_s) = AxnodeMyel._alpha_beta(v_eff, celsius)

    def _exp_solver(x, a, b):
        tau = 1.0 / (a + b)
        xinf = a * tau
        return xinf + (x - xinf) * jnp.exp(-dt / tau)

    return (
        _exp_solver(m,  a_m,  b_m),
        _exp_solver(h,  a_h,  b_h),
        _exp_solver(mp, a_mp, b_mp),
        _exp_solver(s,  a_s,  b_s),
    )


def _init_gates(v_eff, celsius=37.0):
    (a_m, b_m), (a_h, b_h), (a_mp, b_mp), (a_s, b_s) = AxnodeMyel._alpha_beta(v_eff, celsius)
    m  = a_m  / (a_m  + b_m)
    h  = a_h  / (a_h  + b_h)
    mp = a_mp / (a_mp + b_mp)
    s  = a_s  / (a_s  + b_s)
    return m, h, mp, s


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _build_geometry_arrays(geom: MrgGeometry):
    """Return per-compartment arrays needed for the integrator.

    Returns
    -------
    Cm_pF : jnp array (n_comp,)  — total membrane capacitance [pF]
    Gax_uS: jnp array (n_comp-1,) — inter-compartment axial conductance [µS]
    Gpas_uS: jnp array (n_comp,) — total leak conductance [µS]
    Epas_mV: jnp array (n_comp,) — effective reversal for leak [mV]
    is_node: jnp bool array (n_comp,)
    """
    n = geom.n_comp
    L = np.array(geom.length_um) * 1e-4   # cm
    r = np.array(geom.diam_um) / 2.0 * 1e-4  # cm
    Ra = np.array(geom.Ra_ohm_cm)

    # Node diam for NEURON: use the node_d for nodes, axon_d for myelinated.
    # Both are already in geom.diam_um per compartment.

    # Membrane area [cm²]:
    A = np.pi * (2 * r) * L   # π × d × L

    # Capacitance [nF]: use per-compartment cm from geometry.
    # Nodes: cm = CM_AXON = 2 µF/cm² (no myelin).
    # Myelinated: cm = series(CM_AXON, xc_myelin) ≈ xc_myelin — matches NEURON MRG.
    cm_arr = np.array(geom.cm_uF_cm2)       # µF/cm² per compartment
    Cm_nF = cm_arr * A * 1e3                # µF/cm² × cm² × 1e3 nF/µF = nF

    # Axial half-resistances [MΩ]:  Ra × (L/2) / (π × r²)
    half_R = Ra * (L / 2.0) / (np.pi * r**2)   # Ω·cm × cm / cm² = Ω  ... × 1e-6 → MΩ
    # Wait: Ra [Ω·cm] × L/2 [cm] / (π r² [cm²]) = Ω → convert to µS
    # Actually: Ra [Ω·cm] × L/2 [cm] / (π r² [cm²]) = Ω (absolute resistance)
    # G_axial [µS] = 1 / R_total [MΩ] × 1e6 ... let me be careful.
    # R_half [Ω] = Ra [Ω·cm] × (L/2 [cm]) / (π × r² [cm²])
    R_half_Ohm = Ra * (L / 2.0) / (np.pi * r**2)   # Ω
    # Inter-compartment R_ax = R_half[i] + R_half[i+1]  [Ω]
    R_ax_Ohm = R_half_Ohm[:-1] + R_half_Ohm[1:]    # (n-1,)
    # Conductance [µS] = 1e6 / R [Ω]
    Gax_uS = 1e6 / R_ax_Ohm   # µS

    # Leak conductance [µS] for myelinated sections:
    # Use TRUE g_pas per section type.
    Gpas_S_cm2 = np.zeros(n)
    for i, stype in enumerate(geom.section_type):
        if geom.is_node[i]:
            Gpas_S_cm2[i] = 0.0   # node has no Leak (AxnodeMyel only)
        elif stype == "mysa":
            Gpas_S_cm2[i] = G_PAS_MYSA
        elif stype == "flut":
            Gpas_S_cm2[i] = G_PAS_FLUT
        else:   # stin
            Gpas_S_cm2[i] = G_PAS_STIN

    Gpas_uS = Gpas_S_cm2 * A * 1e6   # S/cm² × cm² × 1e6 µS/S = µS

    Epas_mV = np.full(n, V_REST)

    return (
        jnp.array(Cm_nF),
        jnp.array(Gax_uS),
        jnp.array(Gpas_uS),
        jnp.array(Epas_mV),
        jnp.array(geom.is_node, dtype=bool),
        jnp.array(A),   # membrane areas [cm²], needed for current conversion
    )


# ---------------------------------------------------------------------------
# Main integrator
# ---------------------------------------------------------------------------

def integrate_mrg_extracellular(
    geom: MrgGeometry,
    vpax_waveform_mV: np.ndarray,   # (n_comp, n_steps+1) — precomputed V_pax(t)
    stim_waveform_nA: Optional[np.ndarray],  # (n_comp, n_steps+1) or None
    dt_ms: float,
    t_max_ms: float,
    record_comps: list[int],
    temperature: float = 37.0,
) -> np.ndarray:
    """Backward-Euler MRG integration with explicit V_pax coupling.

    Returns V_i (mV) at compartments in `record_comps`, shape (len(record_comps), n_steps+1).
    """
    n_comp = geom.n_comp
    n_steps = round(t_max_ms / dt_ms)

    Cm, Gax, Gpas, Epas, is_node, A_cm2 = _build_geometry_arrays(geom)

    vpax = jnp.asarray(vpax_waveform_mV, dtype=jnp.float32)   # (n_comp, n_steps+1)
    if stim_waveform_nA is not None:
        stim = jnp.asarray(stim_waveform_nA, dtype=jnp.float32)
    else:
        stim = jnp.zeros((n_comp, n_steps + 1), dtype=jnp.float32)

    node_idx = jnp.where(is_node)[0]
    myel_idx = jnp.where(~is_node)[0]

    # ---- Initial conditions --------------------------------------------------
    # At t=0: V_pax=0, so V_m = V_i = V_rest. Gates at SS(V_rest).
    Vi0 = jnp.full((n_comp,), V_REST)
    m0, h0, mp0, s0 = _init_gates(jnp.float32(V_REST), temperature)
    M0  = jnp.full((len(node_idx),), m0)
    H0  = jnp.full((len(node_idx),), h0)
    MP0 = jnp.full((len(node_idx),), mp0)
    S0  = jnp.full((len(node_idx),), s0)

    # ---- Build the constant parts of the tridiagonal system ------------------
    # Upper / lower diagonals from axial conductance (constant):
    lower = -Gax          # (n_comp-1,)  connecting i to i-1
    upper = -Gax          # (n_comp-1,)  connecting i to i+1
    # Diagonal axial contributions (sum of connected G_ax):
    diag_ax = jnp.zeros(n_comp)
    diag_ax = diag_ax.at[:-1].add(Gax)
    diag_ax = diag_ax.at[1:].add(Gax)

    A_jnp = jnp.array(A_cm2, dtype=jnp.float32)  # membrane areas

    record_idx = jnp.array(record_comps, dtype=jnp.int32)

    # ---- Scan over timesteps -------------------------------------------------
    # State variable: V_i (intracellular voltage). Channels see V_m = V_i - V_pax.
    # This is NEURON's extracellular mechanism in backward-Euler form:
    #   (Cm/dt + G_ax + G_ion) * Vi_new = (Cm/dt) * Vi_old
    #                                     + G_ion * (E_eff_Vm + Vpax_new)
    #                                     + G_ax * Vi_new_neighbors
    #                                     + I_stim
    # The Vpax_new shift on E_eff_Vi captures both the spatial activating function
    # and the capacitive coupling (-Cm * dVpax/dt) automatically.

    def step(carry, t_idx):
        Vi, M, H, MP, S = carry
        vp_old = vpax[:, t_idx]          # V_pax at current time (gate update)
        vp_new = vpax[:, t_idx + 1]      # V_pax at next time (RHS)
        I_s = stim[:, t_idx + 1]

        # Membrane voltage at current time for gate kinetics
        Vm_old = Vi - vp_old
        M_n, H_n, MP_n, S_n = _update_gates(Vm_old[node_idx], M, H, MP, S, dt_ms, temperature)

        # Conductance and effective reversal for AxnodeMyel (in Vm frame):
        Gnode_uS = _axnode_conductance_mS_cm2(M_n, H_n, MP_n, S_n) * A_jnp[node_idx] * 1e6
        Enode_mV = _axnode_Eeff(M_n, H_n, MP_n, S_n)

        G_ch = jnp.zeros(n_comp)
        G_ch = G_ch.at[node_idx].set(Gnode_uS)
        G_ch = G_ch.at[myel_idx].set(Gpas[myel_idx])

        diag = Cm / dt_ms + diag_ax + G_ch

        # RHS: Cm/dt * Vi_old + G_ch * (E_eff_Vm + Vpax_new) + I_stim
        # The +Vpax_new shift converts ion reversal potentials from Vm frame to Vi frame.
        # This is the key term that replicates NEURON's direct extracellular coupling.
        E_eff = jnp.full(n_comp, V_REST)
        E_eff = E_eff.at[node_idx].set(Enode_mV)
        rhs = (Cm / dt_ms) * Vi + G_ch * (E_eff + vp_new) + I_s

        Vi_new = _tridiagonal_solve(lower, diag, upper, rhs)

        # Record membrane voltage: Vm = Vi - Vpax at new time
        rec = Vi_new[record_idx] - vp_new[record_idx]
        return (Vi_new, M_n, H_n, MP_n, S_n), rec

    _, recordings = jax.lax.scan(step, (Vi0, M0, H0, MP0, S0), jnp.arange(n_steps))
    # recordings: (n_steps, len(record_comps))  — Vm at each recorded compartment
    # Add initial state (Vm_0 = Vi_0 - Vpax_0 = -80 - 0 = -80 mV):
    rec0 = Vi0[record_idx][None, :]
    all_rec = jnp.concatenate([rec0, recordings], axis=0)  # (n_steps+1, n_rec)
    return all_rec.T   # (n_rec, n_steps+1)


# ---------------------------------------------------------------------------
# Tridiagonal solve (Thomas algorithm, JAX-compatible)
# ---------------------------------------------------------------------------

def _tridiagonal_solve(lower: jnp.ndarray, diag: jnp.ndarray,
                       upper: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    """Solve tridiagonal system with Thomas algorithm.

    lower[i] = coefficient of x[i-1] in equation i  (length n-1, indices 1..n-1)
    diag[i]  = coefficient of x[i]   in equation i  (length n)
    upper[i] = coefficient of x[i+1] in equation i  (length n-1, indices 0..n-2)

    Forward sweep modifies diag/rhs in place (functionally via scan).
    """
    n = diag.shape[0]

    # Forward elimination:
    def fwd(carry, i):
        c_prev, d_prev = carry
        # Eliminate lower[i] (coefficient of x[i-1] in row i)
        w = lower[i - 1] / c_prev    # lower[i-1] is the sub-diagonal at row i
        c_new = diag[i] - w * upper[i - 1]
        d_new = rhs[i]  - w * d_prev
        return (c_new, d_new), (c_new, d_new)

    c0 = diag[0]
    d0 = rhs[0]
    _, (c_fwd, d_fwd) = jax.lax.scan(fwd, (c0, d0), jnp.arange(1, n))
    c_all = jnp.concatenate([c0[None], c_fwd])   # (n,)
    d_all = jnp.concatenate([d0[None], d_fwd])

    # Back substitution:
    x_last = d_all[-1] / c_all[-1]

    def back(x_next, i):
        x_i = (d_all[i] - upper[i] * x_next) / c_all[i]
        return x_i, x_i

    _, x_rest = jax.lax.scan(back, x_last, jnp.arange(n - 2, -1, -1))
    x_rev = jnp.concatenate([x_last[None], x_rest])
    return jnp.flip(x_rev)


# ---------------------------------------------------------------------------
# Convenience: threshold bisection using this integrator
# ---------------------------------------------------------------------------

def find_threshold_extracellular_dc(
    geom: MrgGeometry,
    vpax_unit_mV: np.ndarray,        # (n_comp,) from solve_vpax_static, i0_mA=1
    t_grid_ms: np.ndarray,           # (n_steps+1,)
    shape_waveform: np.ndarray,      # (n_steps+1,) — values in [0,1]
    dt_ms: float,
    t_max_ms: float,
    mid_comp: int,
    ap_threshold_mV: float = -20.0,
    rel_tol: float = 1e-3,
    initial_lo_mA: float = -15.0,
    initial_hi_mA: float = -0.001,
    temperature: float = 37.0,
) -> float:
    """Bisect over cathodic amplitude to find threshold using the DC integrator."""
    from jaxfibers.fibers.mrg import node_indices

    n_steps = len(t_grid_ms) - 1
    vpax_u = np.asarray(vpax_unit_mV)

    def fires(amp_mA: float) -> bool:
        # V_pax waveform: (n_comp, n_steps+1)
        vpax_wave = np.outer(vpax_u, shape_waveform * amp_mA).astype(np.float32)
        rec = integrate_mrg_extracellular(
            geom=geom,
            vpax_waveform_mV=vpax_wave,
            stim_waveform_nA=None,
            dt_ms=dt_ms,
            t_max_ms=t_max_ms,
            record_comps=[mid_comp],
            temperature=temperature,
        )
        return float(jnp.max(rec)) > ap_threshold_mV

    lo, hi = initial_lo_mA, initial_hi_mA
    for _ in range(6):
        if fires(lo):
            break
        lo *= 2.0

    while abs(hi - lo) / abs(lo) > rel_tol:
        mid = (lo + hi) / 2.0
        if fires(mid):
            lo = mid
        else:
            hi = mid
    return lo
