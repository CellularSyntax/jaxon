"""Jaxley translation of sweeney.mod (Sweeney 1987 myelinated-fiber mechanism).

Fast Na + leak channels at the node of Ranvier of the Sweeney myelinated fiber.
Source: sweeney.mod as packaged in PyFibers (FiberModel.SWEENEY).
Reference: Sweeney JD, Mortimer JT, Durand D (1987).
           Modeling of mammalian myelinated nerve for functional neuromuscular
           stimulation. Proc 9th IEEE EMBS Conf, pp 1577-1578.

Channels: m²h Na  |  leak  (no K channel in Sweeney)
Temperature: No Q10 — kinetics are fixed for celsius = 37 °C.
Units: voltage mV, current mA/cm², conductance S/cm², time ms.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxley.channels import Channel
from jaxley.solver_gate import solve_gate_exponential


def _safe_exp(x: jnp.ndarray) -> jnp.ndarray:
    """Clipped exp — matches the Exp() function in sweeney.mod (returns 0 below -100)."""
    return jnp.where(x < -100.0, 0.0, jnp.exp(jnp.clip(x, -100.0, 100.0)))


class SweeneyNode(Channel):
    """Sweeney (1987) node-of-Ranvier channel: m²h Na + leak.

    Parameter values match sweeney.mod exactly (vtraub = 0).
    Note: no K channel and no Q10 — the model is designed for 37 °C.
    """

    current_is_in_mA_per_cm2 = True

    # MOD constants (not exposed as per-compartment params — model is not
    # intended for temperature or pharmacological variation).
    _amA = 49.0;  _amB = 126.0; _amC = 0.363; _amD = 5.3
    _bmA = 56.2;  _bmB = 4.17
    _ahA = 56.0;  _ahB = 15.6
    _bhA = 10.0;  _bhB = 74.5;  _bhC = 5.0

    def __init__(self, name=None):
        super().__init__(name)
        p = self._name
        self.channel_params = {
            f"{p}_gnabar": 1.445,    # S/cm²  (= 1.445 mho/cm² in MOD)
            f"{p}_gl":     0.128,    # S/cm²
            f"{p}_el":    -80.01,    # mV
            f"{p}_ena":    35.64,    # mV
        }
        # Steady-state seeds at V_rest = -80 mV:
        self.channel_states = {
            f"{p}_m": 0.003,
            f"{p}_h": 0.75,
        }
        self.current_name = f"i_{p}"
        self.META = {
            "reference": (
                "Sweeney JD, Mortimer JT, Durand D (1987). "
                "Proc 9th IEEE EMBS Conf, pp 1577-1578."
            ),
            "source": "sweeney.mod (PyFibers)",
        }

    @staticmethod
    def _alpha_beta(v):
        """Alpha/beta for m and h gates at voltage v (mV).

        Translated verbatim from the PROCEDURE rates(v) in sweeney.mod.
        No Q10 — model is fixed for T = 37 °C.
        Returns ((am, bm), (ah, bh)).
        """
        amA, amB, amC, amD = 49.0, 126.0, 0.363, 5.3
        bmA, bmB = 56.2, 4.17
        ahA, ahB, bhA, bhB, bhC = 56.0, 15.6, 10.0, 74.5, 5.0

        # m gate
        denom_m = 1.0 + _safe_exp(-(amA + v) / amD)
        am = (amB + amC * v) / jnp.where(jnp.abs(denom_m) < 1e-30, 1e-30, denom_m)
        bm = am * _safe_exp(-(v + bmA) / bmB)

        # h gate
        denom_h = 1.0 + _safe_exp(-(v + ahA) / bhA)
        ah = ahB * _safe_exp(-(v + bhB) / bhC) / jnp.where(jnp.abs(denom_h) < 1e-30, 1e-30, denom_h)
        bh = ahB / jnp.where(jnp.abs(denom_h) < 1e-30, 1e-30, denom_h)

        return (am, bm), (ah, bh)

    def update_states(self, states, dt, v, params):
        p = self._name
        (am, bm), (ah, bh) = self._alpha_beta(v)
        return {
            f"{p}_m": solve_gate_exponential(states[f"{p}_m"], dt, am, bm),
            f"{p}_h": solve_gate_exponential(states[f"{p}_h"], dt, ah, bh),
        }

    def compute_current(self, states, v, params):
        p = self._name
        m, h = states[f"{p}_m"], states[f"{p}_h"]
        ina = params[f"{p}_gnabar"] * m**2 * h * (v - params[f"{p}_ena"])
        il  = params[f"{p}_gl"]                 * (v - params[f"{p}_el"])
        return ina + il

    def init_state(self, states, v, params, delta_t):
        p = self._name
        (am, bm), (ah, bh) = self._alpha_beta(v)
        return {
            f"{p}_m": am / (am + bm),
            f"{p}_h": ah / (ah + bh),
        }
