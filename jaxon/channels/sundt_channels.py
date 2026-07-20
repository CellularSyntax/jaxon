"""Jaxley translation of `nahh.mod` + `borgkdr.mod` (Sundt 2015 C-fiber mechanisms).

Combined `SundtAxon` channel: transient Na (nahh) + delayed-rectifier K (borgkdr) + leak (pas).
All three live on the same NEURON section in PyFibers' SUNDT model, so we bundle
them into one Channel exactly as AxnodeMyel bundles the MRG node mechanisms.

Sources:
  nahh.mod   — Traub & Miles (1991), Cummins et al. (2007), Sheets et al. (2007)
  borgkdr.mod — Borg-Graham (1987)
  As packaged in PyFibers (FiberModel.SUNDT)

PyFibers parameters (verified at runtime):
  gnabar = 0.04 S/cm², mshift = -6 mV, hshift = 6 mV, ena = 50 mV
  gkdrbar = 0.04 S/cm², ek = -90 mV
  g_pas = 1e-4 S/cm², e_pas = v_rest = -60 mV
  Ra = 100 Ω·cm, cm = 1 µF/cm², delta_z = 8.333 µm, diam = fiber_diameter

Units: voltage mV, current mA/cm², conductance S/cm², time ms.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxley.channels import Channel
from jaxley.solver_gate import solve_gate_exponential

_EPS = 1e-30


def _expM1(x, y):
    """NMODL expM1(x,y) = x / (exp(x/y) - 1), L'Hôpital near 0.

    Safe for JAX: jnp.where evaluates both branches, so we guard the denominator.
    """
    ratio = x / y
    exp_term = jnp.exp(jnp.clip(ratio, -50.0, 50.0)) - 1.0
    safe_denom = jnp.where(jnp.abs(exp_term) < _EPS, _EPS, exp_term)
    main = x / safe_denom
    lhopital = y * (1.0 - 0.5 * ratio)
    return jnp.where(jnp.abs(ratio) < 1e-6, lhopital, main)


class SundtAxon(Channel):
    """Combined Sundt C-fiber mechanism: nahh (Na) + borgkdr (K) + pas (leak).

    States: m, h (fast Na); n, l (delayed-rectifier K).
    Passive leak has no state (instantaneous).
    """

    current_is_in_mA_per_cm2 = True

    def __init__(self, name=None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            # nahh (transient Na)
            f"{prefix}_gnabar":  0.04,    # S/cm²
            f"{prefix}_mshift":  -6.0,    # mV — NaV1.7/1.8 activation shift
            f"{prefix}_hshift":   6.0,    # mV — NaV1.7/1.8 inactivation shift
            f"{prefix}_ishift":   0.0,    # mV — h-gate extra shift (default 0)
            f"{prefix}_ena":     50.0,    # mV
            # borgkdr (delayed-rectifier K)
            f"{prefix}_gkdrbar": 0.04,    # S/cm²
            f"{prefix}_ek":     -90.0,    # mV
            # borgkdr shape parameters (from mod PARAMETER block):
            f"{prefix}_vhalfn":  -32.0,   # mV
            f"{prefix}_vhalfl":  -61.0,   # mV
            f"{prefix}_a0n":      0.03,   # /ms
            f"{prefix}_a0l":      0.001,  # /ms
            f"{prefix}_zetan":   -5.0,
            f"{prefix}_zetal":    2.0,
            f"{prefix}_gmn":      0.4,
            f"{prefix}_gml":      1.0,
            # passive leak
            f"{prefix}_g_pas":   1e-4,    # S/cm²
            f"{prefix}_e_pas":  -60.0,    # mV (= v_rest; set before finitialize in PyFibers)
            # simulation temperature
            f"{prefix}_celsius": 37.0,    # degC
        }
        # Steady-state seeds at V_rest = -60 mV, T = 37 °C:
        #   m/h: q10 cancels in inf, so same as at any T
        #   n/l: FRT changes with T, computed at 37 °C
        self.channel_states = {
            f"{prefix}_m": 0.01175,
            f"{prefix}_h": 0.93669,
            f"{prefix}_n": 0.00527,
            f"{prefix}_l": 0.48137,
        }
        self.current_name = f"i_{prefix}"
        self.META = {
            "reference": "Sundt et al. (2015). J Neurophysiol 114:3075-3091.",
            "source": "nahh.mod + borgkdr.mod (PyFibers)",
        }

    # ── nahh rate functions ────────────────────────────────────────────────────

    @staticmethod
    def _nahh_alpha_beta(v, celsius, mshift, hshift, ishift):
        """(alpha_m, beta_m), (alpha_h, beta_h) — Q10 scaled, shifts applied."""
        q10 = 3.0 ** ((celsius - 30.0) / 10.0)
        # Gate m  (v shifted by mshift relative to the v+65 frame)
        vm = v + 65.0 + mshift
        a_m = q10 * 0.32 * _expM1(13.1 - vm, 4.0)
        b_m = q10 * 0.28 * _expM1(vm - 40.1, 5.0)
        # Gate h  (v shifted by hshift)
        vh = v + 65.0 + hshift
        a_h = q10 * 0.128 * jnp.exp((17.0 - vh + ishift) / 18.0)
        b_h = q10 * 4.0 / (jnp.exp((40.0 - vh) / 5.0) + 1.0)
        return (a_m, b_m), (a_h, b_h)

    # ── borgkdr rate functions ─────────────────────────────────────────────────

    @staticmethod
    def _borgkdr_alpha_beta(v, celsius, vhalfn, vhalfl, a0n, a0l,
                            zetan, zetal, gmn, gml):
        """(alpha_n, beta_n), (alpha_l, beta_l) in cnexp form.

        NMODL uses ninf/taun form; we convert:
            alpha = ninf / taun = q10 * a0 / bet
            beta  = (1-ninf)/taun = q10 * a0 * alp / bet
        so solve_gate_exponential reproduces the identical steady state and τ.
        """
        q10 = 3.0 ** ((celsius - 30.0) / 10.0)
        # Faraday/RT in 1/mV  (exact NMODL formula: 9.648e4 / (8.315*(273.16+celsius)))
        FRT = 9.648e4 / (8.315 * (273.16 + celsius)) * 1e-3  # 1/mV (factor 1e-3 from mV→V)

        # n gate
        alpn = jnp.exp(zetan * (v - vhalfn) * FRT)
        betn = jnp.exp(zetan * gmn * (v - vhalfn) * FRT)
        a_n  = q10 * a0n / betn
        b_n  = q10 * a0n * alpn / betn

        # l gate
        alpl = jnp.exp(zetal * (v - vhalfl) * FRT)
        betl = jnp.exp(zetal * gml * (v - vhalfl) * FRT)
        a_l  = q10 * a0l / betl
        b_l  = q10 * a0l * alpl / betl

        return (a_n, b_n), (a_l, b_l)

    # ── Jaxley API ─────────────────────────────────────────────────────────────

    def update_states(self, states, dt, v, params):
        prefix = self._name
        celsius = params[f"{prefix}_celsius"]
        mshift  = params[f"{prefix}_mshift"]
        hshift  = params[f"{prefix}_hshift"]
        ishift  = params[f"{prefix}_ishift"]
        (a_m, b_m), (a_h, b_h) = self._nahh_alpha_beta(v, celsius, mshift, hshift, ishift)
        m = solve_gate_exponential(states[f"{prefix}_m"], dt, a_m, b_m)
        h = solve_gate_exponential(states[f"{prefix}_h"], dt, a_h, b_h)

        (a_n, b_n), (a_l, b_l) = self._borgkdr_alpha_beta(
            v, celsius,
            params[f"{prefix}_vhalfn"], params[f"{prefix}_vhalfl"],
            params[f"{prefix}_a0n"],    params[f"{prefix}_a0l"],
            params[f"{prefix}_zetan"],  params[f"{prefix}_zetal"],
            params[f"{prefix}_gmn"],    params[f"{prefix}_gml"],
        )
        n = solve_gate_exponential(states[f"{prefix}_n"], dt, a_n, b_n)
        l = solve_gate_exponential(states[f"{prefix}_l"], dt, a_l, b_l)
        return {
            f"{prefix}_m": m, f"{prefix}_h": h,
            f"{prefix}_n": n, f"{prefix}_l": l,
        }

    def compute_current(self, states, v, params):
        prefix = self._name
        m = states[f"{prefix}_m"]; h = states[f"{prefix}_h"]
        n = states[f"{prefix}_n"]; l = states[f"{prefix}_l"]
        i_na  = params[f"{prefix}_gnabar"]  * m**3 * h * (v - params[f"{prefix}_ena"])
        i_k   = params[f"{prefix}_gkdrbar"] * n**3 * l * (v - params[f"{prefix}_ek"])
        i_pas = params[f"{prefix}_g_pas"]              * (v - params[f"{prefix}_e_pas"])
        return i_na + i_k + i_pas

    def init_state(self, states, v, params, delta_t):
        prefix = self._name
        celsius = params[f"{prefix}_celsius"]
        mshift  = params[f"{prefix}_mshift"]
        hshift  = params[f"{prefix}_hshift"]
        ishift  = params[f"{prefix}_ishift"]
        (a_m, b_m), (a_h, b_h) = self._nahh_alpha_beta(v, celsius, mshift, hshift, ishift)
        (a_n, b_n), (a_l, b_l) = self._borgkdr_alpha_beta(
            v, celsius,
            params[f"{prefix}_vhalfn"], params[f"{prefix}_vhalfl"],
            params[f"{prefix}_a0n"],    params[f"{prefix}_a0l"],
            params[f"{prefix}_zetan"],  params[f"{prefix}_zetal"],
            params[f"{prefix}_gmn"],    params[f"{prefix}_gml"],
        )
        return {
            f"{prefix}_m": a_m / (a_m + b_m),
            f"{prefix}_h": a_h / (a_h + b_h),
            f"{prefix}_n": a_n / (a_n + b_n),
            f"{prefix}_l": a_l / (a_l + b_l),
        }
