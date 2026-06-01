"""Threshold search using the full coupled (Vi, Vpax) backward-Euler solver.

Wires AxnodeMyel at nodes + bare g_pas at internodes into membrane_fn.
Expects NEURON ref = -0.18317 mA (D=10 µm, N=21, 0.1 ms rect pulse).
"""
import sys, pathlib, dataclasses
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import jax.numpy as jnp
from jaxley.solver_gate import solve_gate_exponential

from jaxfibers.fibers.mrg import (
    build_mrg, section_centers_um, node_indices,
    V_REST, CM_AXON, G_PAS_MYSA, G_PAS_FLUT, G_PAS_STIN,
)
from jaxfibers.channels.mrg_axnode import AxnodeMyel
from jaxfibers.stim.extracellular import point_source_potentials_mV, rectangular_waveform
from mrg_extracellular_coupled import arrays_from_geometry, find_threshold

# ── parameters ─────────────────────────────────────────────────────────────
D = 10.0; N = 21; DT = 0.005; TSTOP = 5.0; PW = 0.1; DELAY = 1.0
CELSIUS = 37.0
NEURON_REF = -0.18317   # mA

# ── geometry ────────────────────────────────────────────────────────────────
_, geom = build_mrg(diameter=D, n_nodes=N)
nodes   = node_indices(geom)
centers = section_centers_um(geom)
mid     = nodes[len(nodes) // 2]
n_comp  = geom.n_comp

# Coupled solver needs un-lumped CM_AXON on the axon membrane (not the series
# cm_myel stored in geom for the single-cable solver).  Myelin capacitance is
# already stored separately in geom.xc_myelin_uF_cm2 and handled by the solver.
geom_c = dataclasses.replace(geom, cm_uF_cm2=[CM_AXON] * n_comp)
static  = arrays_from_geometry(geom_c, DT)

# ── extracellular field (cathodic reference: i0 = -1 mA) ──────────────────
Ve_unit = np.array(point_source_potentials_mV(
    centers, src_x_um=0., src_y_um=1000., src_z_um=centers[mid], i0_mA=-1.0
))

# ── pulse mask: on[s] = field active during step s ─────────────────────────
n_steps   = int(TSTOP / DT)
shape     = rectangular_waveform(np.arange(n_steps + 1) * DT, DELAY, PW, amp=1.0)
pulse_mask = jnp.asarray(shape[:n_steps] > 0.5)
print(f"Pulse steps: {int(jnp.sum(pulse_mask))}  (expect {int(PW/DT)})")

# ── static arrays for membrane_fn ─────────────────────────────────────────
is_node  = static["is_node"]   # (n_comp,) bool
A_in     = static["A_in_cm2"]  # (n_comp,) float64  inner membrane area [cm²]

# Bare axon g_pas per compartment type (S/cm²); zero at nodes (AxnodeMyel handles those)
_stype_gpas = {"node": 0.0, "mysa": G_PAS_MYSA, "flut": G_PAS_FLUT, "stin": G_PAS_STIN}
g_pas_arr = jnp.asarray([_stype_gpas[s] for s in geom.section_type])

# AxnodeMyel channel parameters (from AXNODE_myel.mod PARAMETER block)
_gnabar = 3.0; _gnapbar = 0.01; _gkbar = 0.08; _gl = 0.007
_ena = 50.0; _ek = -90.0; _el = -90.0

# Resting gate steady states at V_REST
_v0 = jnp.float64(V_REST)
(a_m0, b_m0), (a_h0, b_h0), (a_mp0, b_mp0), (a_s0, b_s0) = AxnodeMyel._alpha_beta(_v0, CELSIUS)
m0  = float(a_m0  / (a_m0  + b_m0))
h0  = float(a_h0  / (a_h0  + b_h0))
mp0 = float(a_mp0 / (a_mp0 + b_mp0))
s0  = float(a_s0  / (a_s0  + b_s0))
print(f"Resting gates: m={m0:.4f}  h={h0:.4f}  mp={mp0:.4f}  s={s0:.4f}")


def membrane_fn_factory():
    """Fresh (membrane_fn, state0) per bisection trial — gates reset to rest."""
    state0 = (
        jnp.where(is_node, m0,  0.0),
        jnp.where(is_node, h0,  0.0),
        jnp.where(is_node, mp0, 0.0),
        jnp.where(is_node, s0,  0.0),
    )

    def membrane_fn(Vm, state, dt):
        M, H, MP, S = state
        # 1. Advance gates (cnexp, Q10-scaled, at current Vm)
        (a_m, b_m), (a_h, b_h), (a_mp, b_mp), (a_s, b_s) = AxnodeMyel._alpha_beta(Vm, CELSIUS)
        M2  = solve_gate_exponential(M,  dt, a_m,  b_m)
        H2  = solve_gate_exponential(H,  dt, a_h,  b_h)
        MP2 = solve_gate_exponential(MP, dt, a_mp, b_mp)
        S2  = solve_gate_exponential(S,  dt, a_s,  b_s)

        # 2. Node: conductances (S/cm²) and currents (mA/cm²) with new gates
        g_na  = _gnabar  * M2**3 * H2
        g_nap = _gnapbar * MP2**3
        g_k   = _gkbar   * S2
        g_tot = g_na + g_nap + g_k + _gl           # total chord conductance
        g_node = g_tot * A_in * 1e6                 # µS
        i_node = (
            (g_na + g_nap) * (Vm - _ena) +
            g_k            * (Vm - _ek)  +
            _gl            * (Vm - _el)
        ) * A_in * 1e6                              # nA (outward positive)

        # 3. Internode: bare axon g_pas (myelin handled separately by solver)
        g_pas_us = g_pas_arr * A_in * 1e6           # µS
        i_pas    = g_pas_us  * (Vm - V_REST)        # nA

        g_eff = jnp.where(is_node, g_node,  g_pas_us)
        i_ion = jnp.where(is_node, i_node,  i_pas)
        return g_eff, i_ion, (M2, H2, MP2, S2)

    return membrane_fn, state0


# ── run bisection ──────────────────────────────────────────────────────────
print("Running bisection (coupled solver) …")
thr = find_threshold(
    static, membrane_fn_factory, Ve_unit, pulse_mask, DT,
    v_thresh=-30.0, lo=-1.0, hi=-0.01, tol=1e-4, v_rest=V_REST,
)
err = abs(thr - NEURON_REF) / abs(NEURON_REF) * 100
print(f"Threshold = {thr:.5f} mA")
print(f"NEURON ref = {NEURON_REF} mA")
print(f"Error      = {err:.1f}%")
