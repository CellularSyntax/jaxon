"""Jaxley translation of RattayAberham.mod (Rattay 1993 HH mechanism).

Classic Hodgkin-Huxley kinetics with vtraub = 70 mV shift (V_rest = -70 mV).
Source: RattayAberham.mod as packaged in PyFibers (FiberModel.RATTAY).
Reference: Rattay F (1993) IEEE Trans Biomed Eng 40:1145-1150.

Channels: m³h Na  |  n⁴ K  |  leak
Temperature: Q10 = 2.247^((celsius-6.3)/10), calibrated for k=12 at 37 °C.
Units: voltage mV, current mA/cm², conductance S/cm², time ms.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxley.channels import Channel
from jaxley.solver_gate import solve_gate_exponential

_EPS = 1e-30


def _vtrap(x: jnp.ndarray, y: float) -> jnp.ndarray:
    """NMODL vtrap(x,y) = x/(exp(x/y)-1), L'Hôpital near zero."""
    z = x / y
    denom = jnp.expm1(jnp.clip(z, -50.0, 50.0))
    safe = jnp.where(jnp.abs(denom) < _EPS, _EPS, denom)
    return jnp.where(jnp.abs(z) < 1e-6, y * (1.0 - 0.5 * z), y * z / safe)


class RattayHH(Channel):
    """Rattay-Aberham (1993) HH channel: m³h Na + n⁴ K + leak.

    Matches RattayAberham.mod exactly (vtraub offset = 70 mV for V_rest = -70 mV).
    """

    current_is_in_mA_per_cm2 = True

    def __init__(self, name=None):
        super().__init__(name)
        p = self._name
        self.channel_params = {
            f"{p}_gnabar":  0.12,    # S/cm²
            f"{p}_gkbar":   0.036,   # S/cm²
            f"{p}_gl":      3e-4,    # S/cm²  (= 0.0003 mho/cm² in MOD)
            f"{p}_el":     -59.4,    # mV
            f"{p}_ena":     45.0,    # mV
            f"{p}_ek":     -82.0,    # mV
            f"{p}_celsius": 37.0,    # °C
        }
        # Steady-state seeds at V_rest = -70 mV, T = 37 °C:
        self.channel_states = {
            f"{p}_m": 0.053,
            f"{p}_h": 0.596,
            f"{p}_n": 0.318,
        }
        self.current_name = f"i_{p}"
        self.META = {
            "reference": "Rattay F (1993). IEEE Trans Biomed Eng 40:1145-1150.",
            "source": "RattayAberham.mod (PyFibers)",
        }

    @staticmethod
    def _alpha_beta(v, celsius):
        """Q10-scaled (alpha, beta) pairs for m, h, n gates.

        v in mV; uses the vsh = v+70 shift from RattayAberham.mod.
        Returns ((am,bm), (ah,bh), (an,bn)).
        """
        q10 = 2.24659524757 ** ((celsius - 6.3) / 10.0)
        vsh = v + 70.0   # shift so v_rest=-70 maps to 0 (standard HH origin)

        # Symmetric ±50 clip on the exponent: prevents overflow at extreme V
        # while preserving the correct LARGE rate constants at hyperpolarised
        # voltages (v < V_rest).  An earlier (-50, 0) cap silently truncated
        # exp(+x) → 1 for v < -70 mV, underestimating β_m, α_h, β_n by up to
        # ~16× at v ≈ -120 mV (end nodes during the cathodic phase of bi_ca).
        # This produced the +6-8 % bi_ca threshold over-estimate diagnosed
        # 2026-06-02 — h failed to recover during prolonged hyperpolarisation,
        # so virtual-cathode anodal break required more current.

        # m gate (Na activation, m³)
        am = q10 * _vtrap(2.5 - 0.1 * vsh, 1.0)
        bm = q10 * 4.0 * jnp.exp(jnp.clip(-vsh / 18.0, -50.0, 50.0))

        # h gate (Na inactivation)
        ah = q10 * 0.07 * jnp.exp(jnp.clip(-vsh / 20.0, -50.0, 50.0))
        bh = q10 / (jnp.exp(jnp.clip(3.0 - 0.1 * vsh, -50.0, 50.0)) + 1.0)

        # n gate (K activation, n⁴)
        an = q10 * 0.1 * _vtrap(1.0 - 0.1 * vsh, 1.0)
        bn = q10 * 0.125 * jnp.exp(jnp.clip(-vsh / 80.0, -50.0, 50.0))

        return (am, bm), (ah, bh), (an, bn)

    def update_states(self, states, dt, v, params):
        p = self._name
        (am, bm), (ah, bh), (an, bn) = self._alpha_beta(v, params[f"{p}_celsius"])
        return {
            f"{p}_m": solve_gate_exponential(states[f"{p}_m"], dt, am, bm),
            f"{p}_h": solve_gate_exponential(states[f"{p}_h"], dt, ah, bh),
            f"{p}_n": solve_gate_exponential(states[f"{p}_n"], dt, an, bn),
        }

    def compute_current(self, states, v, params):
        p = self._name
        m, h, n = states[f"{p}_m"], states[f"{p}_h"], states[f"{p}_n"]
        ina = params[f"{p}_gnabar"] * m**3 * h * (v - params[f"{p}_ena"])
        ik  = params[f"{p}_gkbar"]  * n**4      * (v - params[f"{p}_ek"])
        il  = params[f"{p}_gl"]                 * (v - params[f"{p}_el"])
        return ina + ik + il

    def init_state(self, states, v, params, delta_t):
        p = self._name
        (am, bm), (ah, bh), (an, bn) = self._alpha_beta(v, params[f"{p}_celsius"])
        return {
            f"{p}_m": am / (am + bm),
            f"{p}_h": ah / (ah + bh),
            f"{p}_n": an / (an + bn),
        }
