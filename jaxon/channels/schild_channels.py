"""Jaxley translation of Schild 1994 / Schild 1997 C-fiber mechanisms.

All mechanisms from the original NEURON MOD files are merged into two monolithic
channel classes (SchildCombined94, SchildCombined97) to handle the coupled
Ca2+ dynamics (cai, cao, Oc) correctly within Jaxley's channel framework.

Mechanisms included (Schild 1994):
  naf         m3hl  fast TTX-sensitive Na
  nas         m3h   slow TTX-insensitive Na
  kd          n     delayed-rectifier K
  ka          p3q   transient K
  kds         x3y   slowly-inactivating K
  kca         c     Ca-activated K   (depends on cai)
  can         d*(0.55f1+0.45f2)  high-threshold Ca (N-type)
  cat         d*f   low-threshold Ca (T-type)
  leakSchild  background Na + Ca leaks (ECa dynamic)
  NaKpumpSchild  Na/K pump (algebraic)
  CaPump      sarcolemmal Ca pump (Michaelis-Menten on cai)
  NaCaPump    Na/Ca exchanger
  caextscale  cao ODE
  caintscale  cai + Oc ODEs

Schild 1997 differs in:
  naf97mean   m3h fast Na (no l gate, different kinetics, Q10TempA=22)
  nas97mean   m3h slow Na (different kinetics, Q10TempA=22)
  All other channels identical; conductances set by build_schild97.

References:
  Schild JH et al (1994) J Neurophysiol 71:2338-2358
  Schild JH & Bhatt DL (1997) J Neurophysiol 78:3198-3209
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from jaxley.channels import Channel
from jaxley.solver_gate import solve_gate_exponential

# Physical constants
_F   = 96500.0   # C/mol  (Faraday)
_R   = 8.314     # J/mol/K
_EPS = 1e-30

# Calmodulin (caintscale.mod)
_KU = 100.0   # /mM/ms  binding rate
_KR = 0.238   # /ms     release rate
_NB = 4       # binding sites per calmodulin
_BI = 0.001   # mM  calmodulin concentration


def _safe_cai(cai):
    """Clamp cai to avoid log(0) / division by zero."""
    return jnp.where(cai <= 0.0, 1e-9, cai)


def _eca(cai, cao, celsius):
    """Dynamic Ca reversal potential from Schild 1994."""
    T_K = celsius + 273.15
    return 1000.0 * (_R * T_K / (2.0 * _F)) * jnp.log(cao / _safe_cai(cai)) - 78.7


def _tau_gauss(A, B, C, Vp, v):
    """Gaussian tau: A*exp(-(B*(v-Vp))^2) + C."""
    return A * jnp.exp(-(B * (v - Vp)) ** 2) + C


def _q10_scale(Q10, Q10TempA, celsius):
    """Q10-based temperature scaling factor."""
    return Q10 ** ((Q10TempA - celsius) / 10.0)


def _reversal_potentials(celsius, nai, nao, ki, ko, cai, cao):
    """Nernst potentials for Na, K, Ca."""
    T_K = celsius + 273.15
    ena = 1000.0 * (_R * T_K / _F) * jnp.log(nao / nai)
    ek  = 1000.0 * (_R * T_K / _F) * jnp.log(ko  / ki)
    eca = _eca(cai, cao, celsius)
    return ena, ek, eca


def ca_geometry(length_um: float, diam_um: float, fhspace_um: float = 1.0):
    """(SA_cm2, Vol_cm3, Vol_peri_cm3) for a cylindrical Schild compartment.

    fhspace_um: periaxonal shell thickness [µm] (schild.py sets this to 1 µm).
    """
    lseg     = length_um * 1e-4
    r_in     = diam_um * 0.5 * 1e-4
    r_out    = (diam_um * 0.5 + fhspace_um) * 1e-4
    SA       = math.pi * diam_um * 1e-4 * lseg
    Vol      = math.pi * r_in  ** 2 * lseg
    Vol_peri = math.pi * r_out ** 2 * lseg - Vol
    return SA, Vol, Vol_peri


# ── Shared kinetic functions ──────────────────────────────────────────────────

def _kd_kinetics(v, celsius):
    Q10  = _q10_scale(1.40, 22.85, celsius)
    Vm   = v + 3.0   # shiftkd = +3 mV
    num  = 0.001265 * (Vm + 14.273)
    den  = 1.0 - jnp.exp((Vm + 14.273) / (-10.0))
    alphan = num / jnp.where(jnp.abs(den) < _EPS, _EPS, den)
    betan  = 0.125 * jnp.exp((Vm + 55.0) / (-2.5))
    tau_n  = (1.0 / (alphan + betan) + 1.0) * Q10
    ninf   = 1.0 / (1.0 + jnp.exp((v + 14.62 + 3.0) / (-18.38)))
    return tau_n, ninf


def _ka_kinetics(v, celsius):
    Q10   = _q10_scale(1.93, 22.85, celsius)
    tau_p = _tau_gauss(5.0, 0.022, 2.5, -65.0, v) * Q10
    pinf  = 1.0 / (1.0 + jnp.exp((v + 28.0 + 3.0) / (-28.0)))
    tau_q = _tau_gauss(100.0, 0.035, 10.5, -30.0, v) * Q10
    qinf  = 1.0 / (1.0 + jnp.exp((v + 58.0 + 3.0) / 7.0))
    return (tau_p, pinf), (tau_q, qinf)


def _kds_kinetics(v, celsius):
    Q10   = _q10_scale(1.93, 22.85, celsius)
    tau_x = _tau_gauss(5.0, 0.022, 2.5, -65.0, v) * Q10
    xinf  = 1.0 / (1.0 + jnp.exp((v + 39.59 + 3.0) / (-14.68)))
    tau_y = 7500.0 * Q10
    yinf  = 1.0 / (1.0 + jnp.exp((v + 48.0  + 3.0) / 7.0))
    return (tau_x, xinf), (tau_y, yinf)


def _kca_kinetics(v, cai, celsius):
    Q10    = _q10_scale(2.30, 22.85, celsius)
    alphac = 750.0 * _safe_cai(cai) * jnp.exp((v - 10.0) / 12.0)
    betac  = 0.05  * jnp.exp((v - 10.0) / (-60.0))
    tau_c  = (4.5 / (alphac + betac)) * Q10
    cinf   = alphac / (alphac + betac)
    return tau_c, cinf


def _can_kinetics(v, celsius):
    Q10   = _q10_scale(4.30, 22.85, celsius)
    vs    = v - 7.0   # shiftcan = -7 mV
    tau_d = _tau_gauss(3.25,  0.042,  0.395, -31.0, v) * Q10
    dinf  = 1.0 / (1.0 + jnp.exp((vs + 20.0) / (-4.5)))
    tau_f1 = _tau_gauss(33.5, 0.0395, 5.0,   -30.0, v) * Q10
    f1inf  = 1.0 / (1.0 + jnp.exp((vs + 20.0) / 25.0))
    tau_f2 = _tau_gauss(225.0, 0.0275, 75.0,  -40.0, v) * Q10
    rn     = 0.2 / (1.0 + jnp.exp((vs + 5.0) / (-10.0)))
    f2inf  = rn + 1.0 / (1.0 + jnp.exp((vs + 40.0) / 10.0))
    return (tau_d, dinf), (tau_f1, f1inf), (tau_f2, f2inf)


def _cat_kinetics(v, celsius):
    Q10d  = _q10_scale(1.90, 22.85, celsius)
    Q10f  = _q10_scale(2.20, 22.85, celsius)
    vs    = v - 7.0   # shiftcat = -7 mV
    tau_d = _tau_gauss(22.0,  0.052, 2.5,  -68.0, v) * Q10d
    dinf  = 1.0 / (1.0 + jnp.exp((vs + 54.0) / (-5.75)))
    tau_f = _tau_gauss(103.0, 0.050, 12.5, -58.0, v) * Q10f
    finf  = 1.0 / (1.0 + jnp.exp((vs + 68.0) / 6.0))
    return (tau_d, dinf), (tau_f, finf)


def _nakpump_current(v, nai, ko, celsius, INaKmax22, Kmnai, Kmko):
    INaKmax = INaKmax22 * _q10_scale(1.16, 22.85, celsius)
    fnk     = (v + 150.0) / (v + 200.0)
    return INaKmax * fnk * (nai / (nai + Kmnai)) ** 3 * (ko / (ko + Kmko)) ** 2


def _capump_current(cai, celsius, ICaPmax22, KmCa):
    ICaPmax = ICaPmax22 * _q10_scale(2.30, 22.0, celsius)
    return ICaPmax * _safe_cai(cai) / (_safe_cai(cai) + KmCa)


def _nacapump_current(v, cai, cao, nai, nao, celsius, KNaCa22, DNaCa):
    KNaCa   = KNaCa22 * _q10_scale(2.20, 22.85, celsius)
    T_K     = celsius + 273.15
    S       = 1.0 + DNaCa * (_safe_cai(cai) * nao**3 + cao * nai**3)
    exp_in  = jnp.exp((1.0 * 0.5 * v * _F) / (1000.0 * _R * T_K))
    exp_out = jnp.exp((1.0 * (0.5 - 1.0) * v * _F) / (1000.0 * _R * T_K))
    DFin    = nai**3 * cao               * exp_in
    DFout   = nao**3 * _safe_cai(cai)   * exp_out
    return KNaCa * (DFin - DFout) / S   # inca; ina=3*inca, ica=-2*inca


def _ca_ode_step(ica_total, cai, cao, Oc, dt,
                 SA, Vol, Vol_peri, cabath, txfer):
    """Explicit Euler update of cai, cao, Oc."""
    diffOc   = _KU * _safe_cai(cai) * (1.0 - Oc) - _KR * Oc
    dcao     = ica_total * SA / (2.0 * Vol_peri * _F) + (cabath - cao) / txfer
    dcai     = -ica_total * SA / Vol / (2.0 * _F) - _NB * _BI * diffOc
    new_cao  = cao + dt * dcao
    new_cai  = jnp.maximum(1e-9, cai + dt * dcai)
    new_Oc   = jnp.clip(Oc + dt * diffOc, 0.0, 1.0)
    return new_cai, new_cao, new_Oc


def _update_gate(x, dt, tau, inf_):
    return solve_gate_exponential(x, dt, inf_ / tau, (1.0 - inf_) / tau)


# ═══════════════════════════════════════════════════════════════════════════
# Schild 1994 combined channel
# ═══════════════════════════════════════════════════════════════════════════

class SchildCombined94(Channel):
    """All Schild 1994 mechanisms in one Channel class.

    Ca2+ dynamics (cai, cao, Oc) are state variables updated via explicit Euler
    each timestep, coupled to gate kinetics through ECa and KCa alpha rate.

    Gate states (16): m_naf, h_naf, l_naf, m_nas, h_nas,
                      n_kd, p_ka, q_ka, x_kds, y_kds, c_kca,
                      d_can, f1_can, f2_can, d_cat, f_cat
    Ca states   ( 3): cai, cao, Oc
    """

    current_is_in_mA_per_cm2 = True

    def __init__(self, name: str | None = None):
        super().__init__(name)
        p = self._name
        self.channel_params = {
            # Conductances (S/cm2) — Schild 1994 values
            f"{p}_gbar_naf":   0.068967142,
            f"{p}_gbar_nas":   0.001043349,
            f"{p}_gbar_kd":    0.000180376,
            f"{p}_gbar_ka":    0.000141471,
            f"{p}_gbar_kds":   0.000106103,
            f"{p}_gbar_kca":   0.000141471,
            f"{p}_gbar_can":   0.000106103,
            f"{p}_gbar_cat":   1.23787e-5,
            f"{p}_gbna_leak":  1.85681e-5,
            f"{p}_gbca_leak":  3.00626e-6,
            # Fixed ionic concentrations (mM)
            f"{p}_nai":  8.9,
            f"{p}_nao":  154.0,
            f"{p}_ki":   145.0,
            f"{p}_ko":   5.4,
            # Pumps
            f"{p}_INaKmax22":  0.009726135,
            f"{p}_Kmnai":      5.46,
            f"{p}_Kmko":       0.621,
            f"{p}_ICaPmax22":  0.000859437,
            f"{p}_KmCa":       0.0005,
            f"{p}_KNaCa22":    1.27324e-6,
            f"{p}_DNaCa":      0.0036,
            # Ca bath / exchange
            f"{p}_cabath":     2.0,
            f"{p}_txfer":      4511.0,
            # Geometry for Ca ODEs (set per-compartment in build_schild)
            f"{p}_SA_cm2":        2.094e-7,    # default: d=0.8 µm, L=8.333 µm
            f"{p}_Vol_cm3":       4.189e-12,
            f"{p}_Vol_peri_cm3":  1.702e-11,
            # Temperature
            f"{p}_celsius":    37.0,
        }
        # Approximate SS seeds at V_rest = -46.5 mV, celsius=37
        self.channel_states = {
            f"{p}_m_naf":  0.008,
            f"{p}_h_naf":  0.610,
            f"{p}_l_naf":  0.987,
            f"{p}_m_nas":  3.0e-5,
            f"{p}_h_nas":  1.000,
            f"{p}_n_kd":   0.172,
            f"{p}_p_ka":   0.365,
            f"{p}_q_ka":   0.112,
            f"{p}_x_kds":  0.434,
            f"{p}_y_kds":  0.345,
            f"{p}_c_kca":  0.006,
            f"{p}_d_can":  5.86e-4,
            f"{p}_f1_can": 0.792,
            f"{p}_f2_can": 0.795,
            f"{p}_d_cat":  0.521,
            f"{p}_f_cat":  0.082,
            f"{p}_cai":    0.000117,
            f"{p}_cao":    2.0,
            f"{p}_Oc":     0.05,
        }
        self.current_name = f"i_{p}"
        self.META = {
            "reference": "Schild JH et al (1994) J Neurophysiol 71:2338-2358",
        }

    @staticmethod
    def _naf94_kinetics(v, celsius):
        """tau/inf for naf m, h, l gates (Schild 1994)."""
        Q10m = _q10_scale(2.30, 22.85, celsius)
        Q10h = _q10_scale(1.50, 22.85, celsius)
        # m gate  (S0p5m negative → activation)
        tau_m = _tau_gauss(0.75, 0.0635, 0.12, -40.35, v) * Q10m
        minf  = 1.0 / (1.0 + jnp.exp((v + 41.35 - 17.5) / (-4.75)))
        # h gate  (S0p5h positive → inactivation)
        tau_h = _tau_gauss(6.5, 0.0295, 0.55, -75.0, v) * Q10h
        hinf  = 1.0 / (1.0 + jnp.exp((v + 62.0 - 17.5) / 4.50))
        # l gate  (j in paper; no voltage shift, no Q10)
        tau_l = (25.0 / (1.0 + jnp.exp((v - 20.0) / 4.50))) + 0.01
        linf  = 1.0 / (1.0 + jnp.exp((v + 40.0) / 1.50))
        return (tau_m, minf), (tau_h, hinf), (tau_l, linf)

    @staticmethod
    def _nas94_kinetics(v, celsius):
        Q10m  = _q10_scale(2.30, 22.85, celsius)
        Q10h  = _q10_scale(1.50, 22.85, celsius)
        tau_m = _tau_gauss(1.50, 0.0595, 0.15, -20.35, v) * Q10m
        minf  = 1.0 / (1.0 + jnp.exp((v + 20.35 - 20.0) / (-4.45)))
        tau_h = _tau_gauss(4.95, 0.0335, 0.75, -20.0,  v) * Q10h
        hinf  = 1.0 / (1.0 + jnp.exp((v + 18.0  - 20.0) / 4.50))
        return (tau_m, minf), (tau_h, hinf)

    def _compute_all_currents(self, states, v, params):
        p = self._name
        celsius  = params[f"{p}_celsius"]
        cai, cao = states[f"{p}_cai"], states[f"{p}_cao"]
        nai, nao = params[f"{p}_nai"], params[f"{p}_nao"]
        ki,  ko  = params[f"{p}_ki"],  params[f"{p}_ko"]

        ena, ek, eca = _reversal_potentials(celsius, nai, nao, ki, ko, cai, cao)

        # Naf (m3hl)
        ina_naf = (params[f"{p}_gbar_naf"]
                   * states[f"{p}_m_naf"]**3 * states[f"{p}_h_naf"] * states[f"{p}_l_naf"]
                   * (v - ena))
        # Nas (m3h)
        ina_nas = (params[f"{p}_gbar_nas"]
                   * states[f"{p}_m_nas"]**3 * states[f"{p}_h_nas"]
                   * (v - ena))
        # K channels
        ik_kd   = params[f"{p}_gbar_kd"]  * states[f"{p}_n_kd"]                              * (v - ek)
        ik_ka   = params[f"{p}_gbar_ka"]  * states[f"{p}_p_ka"]**3 * states[f"{p}_q_ka"]     * (v - ek)
        ik_kds  = params[f"{p}_gbar_kds"] * states[f"{p}_x_kds"]**3 * states[f"{p}_y_kds"]   * (v - ek)
        ik_kca  = params[f"{p}_gbar_kca"] * states[f"{p}_c_kca"]                              * (v - ek)
        # Ca channels
        ica_can = (params[f"{p}_gbar_can"]
                   * states[f"{p}_d_can"] * (0.55 * states[f"{p}_f1_can"] + 0.45 * states[f"{p}_f2_can"])
                   * (v - eca))
        ica_cat = (params[f"{p}_gbar_cat"]
                   * states[f"{p}_d_cat"] * states[f"{p}_f_cat"]
                   * (v - eca))
        # Leaks
        ina_leak = params[f"{p}_gbna_leak"] * (v - ena)
        ica_leak = params[f"{p}_gbca_leak"] * (v - eca)
        # Pumps
        ink      = _nakpump_current(v, nai, ko, celsius,
                                    params[f"{p}_INaKmax22"], params[f"{p}_Kmnai"], params[f"{p}_Kmko"])
        ina_nak  = 3.0 * ink
        ik_nak   = -2.0 * ink
        ica_cap  = _capump_current(cai, celsius, params[f"{p}_ICaPmax22"], params[f"{p}_KmCa"])
        inca     = _nacapump_current(v, cai, cao, nai, nao, celsius,
                                     params[f"{p}_KNaCa22"], params[f"{p}_DNaCa"])
        ina_naca =  3.0 * inca
        ica_naca = -2.0 * inca

        i_total = (ina_naf + ina_nas
                   + ik_kd + ik_ka + ik_kds + ik_kca
                   + ica_can + ica_cat
                   + ina_leak + ica_leak
                   + ina_nak + ik_nak
                   + ica_cap
                   + ina_naca + ica_naca)
        ica_total = ica_can + ica_cat + ica_leak + ica_cap + ica_naca
        return i_total, ica_total

    def update_states(self, states, dt, v, params):
        p       = self._name
        celsius = params[f"{p}_celsius"]
        cai     = states[f"{p}_cai"]
        cao     = states[f"{p}_cao"]
        Oc      = states[f"{p}_Oc"]

        _, ica_total = self._compute_all_currents(states, v, params)
        new_cai, new_cao, new_Oc = _ca_ode_step(
            ica_total, cai, cao, Oc, dt,
            params[f"{p}_SA_cm2"], params[f"{p}_Vol_cm3"], params[f"{p}_Vol_peri_cm3"],
            params[f"{p}_cabath"], params[f"{p}_txfer"],
        )

        (tau_m, minf), (tau_h, hinf), (tau_l, linf) = self._naf94_kinetics(v, celsius)
        (tau_mn, mninf), (tau_hn, hninf)             = self._nas94_kinetics(v, celsius)
        tau_n, ninf                                   = _kd_kinetics(v, celsius)
        (tau_p, pinf), (tau_q, qinf)                 = _ka_kinetics(v, celsius)
        (tau_x, xinf), (tau_y, yinf)                 = _kds_kinetics(v, celsius)
        tau_c, cinf                                   = _kca_kinetics(v, cai, celsius)
        (tau_d, dinf), (tau_f1, f1inf), (tau_f2, f2inf) = _can_kinetics(v, celsius)
        (tau_dc, dcinf), (tau_fc, fcinf)             = _cat_kinetics(v, celsius)

        return {
            f"{p}_m_naf":  _update_gate(states[f"{p}_m_naf"],  dt, tau_m,  minf),
            f"{p}_h_naf":  _update_gate(states[f"{p}_h_naf"],  dt, tau_h,  hinf),
            f"{p}_l_naf":  _update_gate(states[f"{p}_l_naf"],  dt, tau_l,  linf),
            f"{p}_m_nas":  _update_gate(states[f"{p}_m_nas"],  dt, tau_mn, mninf),
            f"{p}_h_nas":  _update_gate(states[f"{p}_h_nas"],  dt, tau_hn, hninf),
            f"{p}_n_kd":   _update_gate(states[f"{p}_n_kd"],   dt, tau_n,  ninf),
            f"{p}_p_ka":   _update_gate(states[f"{p}_p_ka"],   dt, tau_p,  pinf),
            f"{p}_q_ka":   _update_gate(states[f"{p}_q_ka"],   dt, tau_q,  qinf),
            f"{p}_x_kds":  _update_gate(states[f"{p}_x_kds"],  dt, tau_x,  xinf),
            f"{p}_y_kds":  _update_gate(states[f"{p}_y_kds"],  dt, tau_y,  yinf),
            f"{p}_c_kca":  _update_gate(states[f"{p}_c_kca"],  dt, tau_c,  cinf),
            f"{p}_d_can":  _update_gate(states[f"{p}_d_can"],  dt, tau_d,  dinf),
            f"{p}_f1_can": _update_gate(states[f"{p}_f1_can"], dt, tau_f1, f1inf),
            f"{p}_f2_can": _update_gate(states[f"{p}_f2_can"], dt, tau_f2, f2inf),
            f"{p}_d_cat":  _update_gate(states[f"{p}_d_cat"],  dt, tau_dc, dcinf),
            f"{p}_f_cat":  _update_gate(states[f"{p}_f_cat"],  dt, tau_fc, fcinf),
            f"{p}_cai":    new_cai,
            f"{p}_cao":    new_cao,
            f"{p}_Oc":     new_Oc,
        }

    def compute_current(self, states, v, params):
        i_total, _ = self._compute_all_currents(states, v, params)
        return i_total

    def init_state(self, states, v, params, delta_t):
        p       = self._name
        celsius = params[f"{p}_celsius"]
        cai0    = states[f"{p}_cai"]
        cao0    = states[f"{p}_cao"]

        (_, minf), (_, hinf), (_, linf)             = self._naf94_kinetics(v, celsius)
        (_, mninf), (_, hninf)                       = self._nas94_kinetics(v, celsius)
        _, ninf                                       = _kd_kinetics(v, celsius)
        (_, pinf), (_, qinf)                         = _ka_kinetics(v, celsius)
        (_, xinf), (_, yinf)                         = _kds_kinetics(v, celsius)
        _, cinf                                       = _kca_kinetics(v, cai0, celsius)
        (_, dinf), (_, f1inf), (_, f2inf)            = _can_kinetics(v, celsius)
        (_, dcinf), (_, fcinf)                       = _cat_kinetics(v, celsius)

        return {
            f"{p}_m_naf":  minf,
            f"{p}_h_naf":  hinf,
            f"{p}_l_naf":  linf,
            f"{p}_m_nas":  mninf,
            f"{p}_h_nas":  hninf,
            f"{p}_n_kd":   ninf,
            f"{p}_p_ka":   pinf,
            f"{p}_q_ka":   qinf,
            f"{p}_x_kds":  xinf,
            f"{p}_y_kds":  yinf,
            f"{p}_c_kca":  cinf,
            f"{p}_d_can":  dinf,
            f"{p}_f1_can": f1inf,
            f"{p}_f2_can": f2inf,
            f"{p}_d_cat":  dcinf,
            f"{p}_f_cat":  fcinf,
            f"{p}_cai":    cai0,
            f"{p}_cao":    cao0,
            f"{p}_Oc":     states[f"{p}_Oc"],
        }


# ═══════════════════════════════════════════════════════════════════════════
# Schild 1997 combined channel
# ═══════════════════════════════════════════════════════════════════════════

class SchildCombined97(Channel):
    """All Schild 1997 mechanisms in one Channel class.

    Differs from SchildCombined94:
      - naf97mean: m3h fast Na (no l gate, different kinetics, Q10TempA=22)
      - nas97mean: m3h slow Na (different kinetics, Q10TempA=22)
      - Different conductances (set in build_schild97)

    Gate states (15): m_naf97, h_naf97, m_nas97, h_nas97,
                      n_kd, p_ka, q_ka, x_kds, y_kds, c_kca,
                      d_can, f1_can, f2_can, d_cat, f_cat
    Ca states   ( 3): cai, cao, Oc
    """

    current_is_in_mA_per_cm2 = True

    def __init__(self, name: str | None = None):
        super().__init__(name)
        p = self._name
        self.channel_params = {
            # Conductances (S/cm2) — Schild 1997 values
            f"{p}_gbar_naf":   0.022434928,
            f"{p}_gbar_nas":   0.022434928,
            f"{p}_gbar_kd":    0.001956534,
            f"{p}_gbar_ka":    0.001304356,
            f"{p}_gbar_kds":   0.000782614,
            f"{p}_gbar_kca":   0.000913049,
            f"{p}_gbar_can":   0.000521743,
            f"{p}_gbar_cat":   0.00018261,
            f"{p}_gbna_leak":  1.8261e-5,
            f"{p}_gbca_leak":  9.13049e-6,
            # Fixed ionic concentrations (mM) — same as 1994
            f"{p}_nai":  8.9,
            f"{p}_nao":  154.0,
            f"{p}_ki":   145.0,
            f"{p}_ko":   5.4,
            # Pumps — same as 1994
            f"{p}_INaKmax22":  0.009726135,
            f"{p}_Kmnai":      5.46,
            f"{p}_Kmko":       0.621,
            f"{p}_ICaPmax22":  0.000859437,
            f"{p}_KmCa":       0.0005,
            f"{p}_KNaCa22":    1.27324e-6,
            f"{p}_DNaCa":      0.0036,
            # Ca bath / exchange
            f"{p}_cabath":     2.0,
            f"{p}_txfer":      4511.0,
            # Geometry (set per-compartment in build_schild)
            f"{p}_SA_cm2":        2.094e-7,
            f"{p}_Vol_cm3":       4.189e-12,
            f"{p}_Vol_peri_cm3":  1.702e-11,
            # Temperature
            f"{p}_celsius":    37.0,
        }
        self.channel_states = {
            f"{p}_m_naf97": 0.050,
            f"{p}_h_naf97": 0.720,
            f"{p}_m_nas97": 0.020,
            f"{p}_h_nas97": 0.600,
            f"{p}_n_kd":    0.172,
            f"{p}_p_ka":    0.365,
            f"{p}_q_ka":    0.112,
            f"{p}_x_kds":   0.434,
            f"{p}_y_kds":   0.345,
            f"{p}_c_kca":   0.006,
            f"{p}_d_can":   5.86e-4,
            f"{p}_f1_can":  0.792,
            f"{p}_f2_can":  0.795,
            f"{p}_d_cat":   0.521,
            f"{p}_f_cat":   0.082,
            f"{p}_cai":     0.000117,
            f"{p}_cao":     2.0,
            f"{p}_Oc":      0.05,
        }
        self.current_name = f"i_{p}"
        self.META = {
            "reference": "Schild JH & Bhatt DL (1997) J Neurophysiol 78:3198-3209",
        }

    @staticmethod
    def _naf97_kinetics(v, celsius):
        """tau/inf for naf97mean m, h (no l gate, Q10TempA=22)."""
        Q10m = _q10_scale(2.30, 22.0, celsius)
        Q10h = _q10_scale(1.50, 22.0, celsius)
        tau_m = _tau_gauss(1.15, 0.06, 0.21, -40.0, v) * Q10m
        minf  = 1.0 / (1.0 + jnp.exp((-31.62 - v) / 6.98))
        tau_h = _tau_gauss(18.0, 0.043, 1.35, -62.5, v) * Q10h
        hinf  = 1.0 / (1.0 + jnp.exp((-65.99 - v) / (-5.97)))
        return (tau_m, minf), (tau_h, hinf)

    @staticmethod
    def _nas97_kinetics(v, celsius):
        """tau/inf for nas97mean m, h (Q10TempA=22)."""
        Q10m  = _q10_scale(2.30, 22.0, celsius)
        Q10h  = _q10_scale(1.50, 22.0, celsius)
        tau_m = _tau_gauss(1.45, 0.058, 0.26, -14.5, v) * Q10m
        minf  = 1.0 / (1.0 + jnp.exp((-11.29 - v) / 5.54))
        tau_h = _tau_gauss(10.75, 0.067, 3.15, -13.5, v) * Q10h
        hinf  = 1.0 / (1.0 + jnp.exp((-31.0  - v) / (-5.20)))
        return (tau_m, minf), (tau_h, hinf)

    def _compute_all_currents(self, states, v, params):
        p = self._name
        celsius  = params[f"{p}_celsius"]
        cai, cao = states[f"{p}_cai"], states[f"{p}_cao"]
        nai, nao = params[f"{p}_nai"], params[f"{p}_nao"]
        ki,  ko  = params[f"{p}_ki"],  params[f"{p}_ko"]

        ena, ek, eca = _reversal_potentials(celsius, nai, nao, ki, ko, cai, cao)

        # Naf97 (m3h, no l)
        ina_naf = (params[f"{p}_gbar_naf"]
                   * states[f"{p}_m_naf97"]**3 * states[f"{p}_h_naf97"]
                   * (v - ena))
        # Nas97 (m3h)
        ina_nas = (params[f"{p}_gbar_nas"]
                   * states[f"{p}_m_nas97"]**3 * states[f"{p}_h_nas97"]
                   * (v - ena))
        # K channels
        ik_kd   = params[f"{p}_gbar_kd"]  * states[f"{p}_n_kd"]                              * (v - ek)
        ik_ka   = params[f"{p}_gbar_ka"]  * states[f"{p}_p_ka"]**3 * states[f"{p}_q_ka"]     * (v - ek)
        ik_kds  = params[f"{p}_gbar_kds"] * states[f"{p}_x_kds"]**3 * states[f"{p}_y_kds"]   * (v - ek)
        ik_kca  = params[f"{p}_gbar_kca"] * states[f"{p}_c_kca"]                              * (v - ek)
        # Ca channels
        ica_can = (params[f"{p}_gbar_can"]
                   * states[f"{p}_d_can"] * (0.55 * states[f"{p}_f1_can"] + 0.45 * states[f"{p}_f2_can"])
                   * (v - eca))
        ica_cat = (params[f"{p}_gbar_cat"]
                   * states[f"{p}_d_cat"] * states[f"{p}_f_cat"]
                   * (v - eca))
        # Leaks
        ina_leak = params[f"{p}_gbna_leak"] * (v - ena)
        ica_leak = params[f"{p}_gbca_leak"] * (v - eca)
        # Pumps
        ink      = _nakpump_current(v, nai, ko, celsius,
                                    params[f"{p}_INaKmax22"], params[f"{p}_Kmnai"], params[f"{p}_Kmko"])
        ina_nak  = 3.0 * ink
        ik_nak   = -2.0 * ink
        ica_cap  = _capump_current(cai, celsius, params[f"{p}_ICaPmax22"], params[f"{p}_KmCa"])
        inca     = _nacapump_current(v, cai, cao, nai, nao, celsius,
                                     params[f"{p}_KNaCa22"], params[f"{p}_DNaCa"])
        ina_naca =  3.0 * inca
        ica_naca = -2.0 * inca

        i_total = (ina_naf + ina_nas
                   + ik_kd + ik_ka + ik_kds + ik_kca
                   + ica_can + ica_cat
                   + ina_leak + ica_leak
                   + ina_nak + ik_nak
                   + ica_cap
                   + ina_naca + ica_naca)
        ica_total = ica_can + ica_cat + ica_leak + ica_cap + ica_naca
        return i_total, ica_total

    def update_states(self, states, dt, v, params):
        p       = self._name
        celsius = params[f"{p}_celsius"]
        cai     = states[f"{p}_cai"]
        cao     = states[f"{p}_cao"]
        Oc      = states[f"{p}_Oc"]

        _, ica_total = self._compute_all_currents(states, v, params)
        new_cai, new_cao, new_Oc = _ca_ode_step(
            ica_total, cai, cao, Oc, dt,
            params[f"{p}_SA_cm2"], params[f"{p}_Vol_cm3"], params[f"{p}_Vol_peri_cm3"],
            params[f"{p}_cabath"], params[f"{p}_txfer"],
        )

        (tau_m, minf), (tau_h, hinf) = self._naf97_kinetics(v, celsius)
        (tau_mn, mninf), (tau_hn, hninf) = self._nas97_kinetics(v, celsius)
        tau_n, ninf                   = _kd_kinetics(v, celsius)
        (tau_p, pinf), (tau_q, qinf) = _ka_kinetics(v, celsius)
        (tau_x, xinf), (tau_y, yinf) = _kds_kinetics(v, celsius)
        tau_c, cinf                   = _kca_kinetics(v, cai, celsius)
        (tau_d, dinf), (tau_f1, f1inf), (tau_f2, f2inf) = _can_kinetics(v, celsius)
        (tau_dc, dcinf), (tau_fc, fcinf) = _cat_kinetics(v, celsius)

        return {
            f"{p}_m_naf97":  _update_gate(states[f"{p}_m_naf97"],  dt, tau_m,  minf),
            f"{p}_h_naf97":  _update_gate(states[f"{p}_h_naf97"],  dt, tau_h,  hinf),
            f"{p}_m_nas97":  _update_gate(states[f"{p}_m_nas97"],  dt, tau_mn, mninf),
            f"{p}_h_nas97":  _update_gate(states[f"{p}_h_nas97"],  dt, tau_hn, hninf),
            f"{p}_n_kd":     _update_gate(states[f"{p}_n_kd"],     dt, tau_n,  ninf),
            f"{p}_p_ka":     _update_gate(states[f"{p}_p_ka"],     dt, tau_p,  pinf),
            f"{p}_q_ka":     _update_gate(states[f"{p}_q_ka"],     dt, tau_q,  qinf),
            f"{p}_x_kds":    _update_gate(states[f"{p}_x_kds"],    dt, tau_x,  xinf),
            f"{p}_y_kds":    _update_gate(states[f"{p}_y_kds"],    dt, tau_y,  yinf),
            f"{p}_c_kca":    _update_gate(states[f"{p}_c_kca"],    dt, tau_c,  cinf),
            f"{p}_d_can":    _update_gate(states[f"{p}_d_can"],    dt, tau_d,  dinf),
            f"{p}_f1_can":   _update_gate(states[f"{p}_f1_can"],   dt, tau_f1, f1inf),
            f"{p}_f2_can":   _update_gate(states[f"{p}_f2_can"],   dt, tau_f2, f2inf),
            f"{p}_d_cat":    _update_gate(states[f"{p}_d_cat"],    dt, tau_dc, dcinf),
            f"{p}_f_cat":    _update_gate(states[f"{p}_f_cat"],    dt, tau_fc, fcinf),
            f"{p}_cai":      new_cai,
            f"{p}_cao":      new_cao,
            f"{p}_Oc":       new_Oc,
        }

    def compute_current(self, states, v, params):
        i_total, _ = self._compute_all_currents(states, v, params)
        return i_total

    def init_state(self, states, v, params, delta_t):
        p       = self._name
        celsius = params[f"{p}_celsius"]
        cai0    = states[f"{p}_cai"]
        cao0    = states[f"{p}_cao"]

        (_, minf), (_, hinf)          = self._naf97_kinetics(v, celsius)
        (_, mninf), (_, hninf)        = self._nas97_kinetics(v, celsius)
        _, ninf                        = _kd_kinetics(v, celsius)
        (_, pinf), (_, qinf)          = _ka_kinetics(v, celsius)
        (_, xinf), (_, yinf)          = _kds_kinetics(v, celsius)
        _, cinf                        = _kca_kinetics(v, cai0, celsius)
        (_, dinf), (_, f1inf), (_, f2inf) = _can_kinetics(v, celsius)
        (_, dcinf), (_, fcinf)        = _cat_kinetics(v, celsius)

        return {
            f"{p}_m_naf97":  minf,
            f"{p}_h_naf97":  hinf,
            f"{p}_m_nas97":  mninf,
            f"{p}_h_nas97":  hninf,
            f"{p}_n_kd":     ninf,
            f"{p}_p_ka":     pinf,
            f"{p}_q_ka":     qinf,
            f"{p}_x_kds":    xinf,
            f"{p}_y_kds":    yinf,
            f"{p}_c_kca":    cinf,
            f"{p}_d_can":    dinf,
            f"{p}_f1_can":   f1inf,
            f"{p}_f2_can":   f2inf,
            f"{p}_d_cat":    dcinf,
            f"{p}_f_cat":    fcinf,
            f"{p}_cai":      cai0,
            f"{p}_cao":      cao0,
            f"{p}_Oc":       states[f"{p}_Oc"],
        }
