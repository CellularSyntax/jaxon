"""Jaxley translation of `AXNODE_myel.mod` (MRG node-of-Ranvier mechanism).

Source: pyfibers/MOD/AXNODE_myel.mod (McIntyre, Richardson & Grill 2002).
Reference: J Neurophysiol 87:995-1006.

The mechanism comprises four conductances at the node:
  * `gnabar`  — fast transient Na (gates m^3 h, reversal ena=+50 mV)
  * `gnapbar` — persistent Na             (gates mp^3,  reversal ena=+50 mV)
  * `gkbar`   — slow K                    (gate s,      reversal ek=-90 mV)
  * `gl`      — leak                      (reversal el=-90 mV, rest -80 mV)

All four channels live on the *same* compartment in NEURON. To match that
exactly in Jaxley while keeping each gate independently inspectable, we
expose **one** Channel (`AxnodeMyel`) that owns all four conductances and
all four gating variables. This mirrors the NMODL `SUFFIX axnode_myel`.

Units (matching the .mod file):
  voltage in mV, current in mA/cm^2, conductance in S/cm^2, time in ms.

All math uses `jax.numpy` only, so this Channel is fully differentiable.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxley.channels import Channel
from jaxley.solver_gate import solve_gate_exponential


# ---------------------------------------------------------------------------
# vtrap helpers (translated verbatim from the NMODL FUNCTION vtrap1..vtrap11)
# ---------------------------------------------------------------------------
#
# In NMODL each vtrap branches on:
#   - |arg/scale| < 1e-6        -> L'Hopital constant
#   - x out of an asymptote     -> hard-coded tail value
#   - otherwise                 -> the analytic expression
#
# In JAX we cannot branch on traced values, so we use `jnp.where` and
# rely on the safe expression being computed everywhere (with the
# small-denominator guard handled by adding a tiny epsilon to the denom
# only when we are about to divide). This preserves differentiability.

_EPS = 1e-30  # never differentiated through; only there to silence inf when masked


def _safe_exp(x: jnp.ndarray) -> jnp.ndarray:
    """Match the .mod FUNCTION Exp: returns 0 below x=-100 to suppress overflow."""
    return jnp.where(x < -100.0, 0.0, jnp.exp(jnp.clip(x, a_min=-100.0, a_max=100.0)))


def _vtrap1(v):
    # alpha_mp: (ampA*(v+ampB)) / (1 - exp(-(v+ampB)/ampC))
    A, B, C = 0.01, 27.0, 10.2
    arg = (v + B) / C
    denom = 1.0 - _safe_exp(-arg)
    main = (A * (v + B)) / jnp.where(jnp.abs(denom) < _EPS, _EPS, denom)
    lhopital = A * C
    tail = 0.00086725  # value at x < -150
    out = jnp.where(jnp.abs(arg) < 1e-6, lhopital, main)
    return jnp.where(v < -150.0, tail, out)


def _vtrap2(v):
    # beta_mp: (bmpA*(-(v+bmpB))) / (1 - exp((v+bmpB)/bmpC))
    A, B, C = 0.00025, 34.0, 10.0
    arg = (v + B) / C
    denom = 1.0 - _safe_exp(arg)
    main = (A * (-(v + B))) / jnp.where(jnp.abs(denom) < _EPS, _EPS, denom)
    lhopital = A * C
    tail = 1.5855e-05  # value at x > 150
    out = jnp.where(jnp.abs(arg) < 1e-6, lhopital, main)
    return jnp.where(v > 150.0, tail, out)


def _vtrap6(v):
    # alpha_m
    A, B, C = 1.86, 21.4, 10.3
    arg = (v + B) / C
    denom = 1.0 - _safe_exp(-arg)
    main = (A * (v + B)) / jnp.where(jnp.abs(denom) < _EPS, _EPS, denom)
    lhopital = A * C
    tail = 0.15733
    out = jnp.where(jnp.abs(arg) < 1e-6, lhopital, main)
    return jnp.where(v < -150.0, tail, out)


def _vtrap7(v):
    # beta_m
    A, B, C = 0.086, 25.7, 9.16
    arg = (v + B) / C
    denom = 1.0 - _safe_exp(arg)
    main = (A * (-(v + B))) / jnp.where(jnp.abs(denom) < _EPS, _EPS, denom)
    lhopital = A * C
    tail = 0.0057268
    out = jnp.where(jnp.abs(arg) < 1e-6, lhopital, main)
    return jnp.where(v > 150.0, tail, out)


def _vtrap8(v):
    # alpha_h
    A, B, C = 0.062, 114.0, 11.0
    arg = (v + B) / C
    denom = 1.0 - _safe_exp(arg)
    main = (A * (-(v + B))) / jnp.where(jnp.abs(denom) < _EPS, _EPS, denom)
    lhopital = A * C
    tail = 0.0032594
    out = jnp.where(jnp.abs(arg) < 1e-6, lhopital, main)
    return jnp.where(v > 150.0, tail, out)


def _vtrap9(v):
    # beta_h
    A, B, C = 2.3, 31.8, 13.4
    main = A / (1.0 + _safe_exp(-(v + B) / C))
    tail = 0.0014054
    return jnp.where(v < -150.0, tail, main)


def _vtrap10(v):
    # alpha_s: vtraub-shifted, asC negative
    A, B, C = 0.3, -27.0, -5.0
    vtraub = -80.0
    main = A / (_safe_exp((v - vtraub + B) / C) + 1.0)
    tail = 3.3484e-05
    return jnp.where(v < -150.0, tail, main)


def _vtrap11(v):
    # beta_s
    A, B, C = 0.03, 10.0, -1.0
    vtraub = -80.0
    main = A / (_safe_exp((v - vtraub + B) / C) + 1.0)
    tail = 3.3484e-06
    return jnp.where(v < -150.0, tail, main)


# ---------------------------------------------------------------------------
# The Channel class
# ---------------------------------------------------------------------------


class AxnodeMyel(Channel):
    """Combined MRG node mechanism (Fast Na + Persistent Na + Slow K + Leak).

    States: m, h (fast Na), mp (persistent Na), s (slow K).
    All conductances live on the same compartment with shared `celsius` and
    `q10_*` factors as in the original NMODL.
    """

    current_is_in_mA_per_cm2 = True  # Jaxley >= 0.5: opt-in to mA/cm^2 units

    def __init__(self, name=None):
        super().__init__(name)
        prefix = self._name
        # Default reversal/parameters from AXNODE_myel.mod PARAMETER block:
        self.channel_params = {
            f"{prefix}_gnabar":  3.0,    # S/cm^2 (mho/cm2)
            f"{prefix}_gnapbar": 0.01,   # S/cm^2
            f"{prefix}_gkbar":   0.08,   # S/cm^2
            f"{prefix}_gl":      0.007,  # S/cm^2
            f"{prefix}_ena":     50.0,   # mV
            f"{prefix}_ek":      -90.0,  # mV
            f"{prefix}_el":      -90.0,  # mV
            f"{prefix}_celsius": 37.0,   # degC (NEURON default in PyFibers)
        }
        # Steady-state values at v_rest = -80 mV are computed lazily in init_state.
        # Placeholder seeds (close to rest-state values) until init_state is called:
        self.channel_states = {
            f"{prefix}_m":  0.0303,
            f"{prefix}_h":  0.7506,
            f"{prefix}_mp": 0.0494,
            f"{prefix}_s":  0.1248,
        }
        # Jaxley uses `current_name` to aggregate ionic currents. We register
        # this channel under a NONSPECIFIC bucket because the .mod file does
        # the same (NONSPECIFIC_CURRENT ina, inap, ik, il).
        self.current_name = f"i_{prefix}"
        self.META = {
            "reference": "McIntyre CC, Richardson AG, Grill WM (2002). J Neurophysiol 87:995-1006.",
            "source": "AXNODE_myel.mod (PyFibers)",
        }

    # ---- internals -------------------------------------------------------

    @staticmethod
    def _q10s(celsius):
        q10_1 = 2.2 ** ((celsius - 20.0) / 10.0)
        q10_2 = 2.9 ** ((celsius - 20.0) / 10.0)
        q10_3 = 3.0 ** ((celsius - 36.0) / 10.0)
        return q10_1, q10_2, q10_3

    @classmethod
    def _alpha_beta(cls, v, celsius):
        """Return raw (alpha, beta) for each of (m, h, mp, s), Q10-scaled."""
        q10_1, q10_2, q10_3 = cls._q10s(celsius)
        a_m  = q10_1 * _vtrap6(v);   b_m  = q10_1 * _vtrap7(v)
        a_h  = q10_2 * _vtrap8(v);   b_h  = q10_2 * _vtrap9(v)
        a_mp = q10_1 * _vtrap1(v);   b_mp = q10_1 * _vtrap2(v)
        a_s  = q10_3 * _vtrap10(v);  b_s  = q10_3 * _vtrap11(v)
        return (a_m, b_m), (a_h, b_h), (a_mp, b_mp), (a_s, b_s)

    # ---- Jaxley API ------------------------------------------------------

    def update_states(self, states, dt, v, params):
        """Advance gates by one dt using analytic exponential solver."""
        prefix = self._name
        celsius = params[f"{prefix}_celsius"]
        (a_m, b_m), (a_h, b_h), (a_mp, b_mp), (a_s, b_s) = self._alpha_beta(v, celsius)
        m  = solve_gate_exponential(states[f"{prefix}_m"],  dt, a_m,  b_m)
        h  = solve_gate_exponential(states[f"{prefix}_h"],  dt, a_h,  b_h)
        mp = solve_gate_exponential(states[f"{prefix}_mp"], dt, a_mp, b_mp)
        s  = solve_gate_exponential(states[f"{prefix}_s"],  dt, a_s,  b_s)
        return {
            f"{prefix}_m":  m,
            f"{prefix}_h":  h,
            f"{prefix}_mp": mp,
            f"{prefix}_s":  s,
        }

    def compute_current(self, states, v, params):
        """Total ionic current in mA/cm^2 (outward positive — Jaxley sign)."""
        prefix = self._name
        m  = states[f"{prefix}_m"]
        h  = states[f"{prefix}_h"]
        mp = states[f"{prefix}_mp"]
        s  = states[f"{prefix}_s"]
        gna  = params[f"{prefix}_gnabar"]
        gnap = params[f"{prefix}_gnapbar"]
        gk   = params[f"{prefix}_gkbar"]
        gl   = params[f"{prefix}_gl"]
        ena  = params[f"{prefix}_ena"]
        ek   = params[f"{prefix}_ek"]
        el   = params[f"{prefix}_el"]

        ina  = gna  * m**3 * h * (v - ena)
        inap = gnap * mp**3      * (v - ena)
        ik   = gk   * s          * (v - ek)
        il   = gl                * (v - el)
        return ina + inap + ik + il

    def init_state(self, states, v, params, delta_t):
        """Initialize gates to their voltage-dependent steady states (the .mod INITIAL block)."""
        prefix = self._name
        celsius = params[f"{prefix}_celsius"]
        (a_m, b_m), (a_h, b_h), (a_mp, b_mp), (a_s, b_s) = self._alpha_beta(v, celsius)
        return {
            f"{prefix}_m":  a_m  / (a_m  + b_m),
            f"{prefix}_h":  a_h  / (a_h  + b_h),
            f"{prefix}_mp": a_mp / (a_mp + b_mp),
            f"{prefix}_s":  a_s  / (a_s  + b_s),
        }
