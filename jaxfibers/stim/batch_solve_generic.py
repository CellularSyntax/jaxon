"""Model-agnostic batched m_max integrator.

The original batch_solve.py is MRG-specific: it imports AxnodeMyel and
hardcodes the 4-gate (m, h, mp, s) state.  The unmyelinated models
(Sundt, Sweeney, Rattay) use different channels and gate counts:

  - Sundt:    SundtAxon    states (m, h, n, l)
  - Sweeney:  SweeneyNode  states (m, h)
  - Rattay:   RattayHH     states (m, h, n)

Rather than 4× copy-paste the per-step coupled backward-Euler solver,
this module reuses the existing model-agnostic ``integrate()`` from
``extracellular_coupled.py`` (which already takes a model-specific
``membrane_fn`` callback) and vmaps it across fibres.

Public API (consumed by experiments_v2/selectivity_sweep.py):

  build_model_batch(model, geoms, Ve_seq_batch, dt)
      → returns (per_fiber_integrate_fn, m_max_batch_fn) closures
      that selectivity_sweep / optimizer can call without caring
      which axon model is underneath.

  batch_integrate_m_max_generic(model, fs_list, s0_list, Ve_seq_batch, dt)
      → drop-in equivalent of batch_solve.batch_integrate_m_max
      but model-aware.

Currently registered models: 'mrg' (uses the legacy MRG path),
'sundt', 'sweeney', 'rattay'.  Schild variants are not registered for
selectivity sweeps because they were only benchmarked for scaling.
"""
from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp

from jaxley.solver_gate import solve_gate_exponential

from jaxfibers.stim.extracellular_coupled import (
    arrays_from_geometry, integrate, _be_step,
)

# Per-model channel imports.
from jaxfibers.channels.sundt_channels   import SundtAxon
from jaxfibers.channels.sweeney_channels import SweeneyNode
from jaxfibers.channels.rattay_channels  import RattayHH


# ─── Per-model constants (gates + V_REST + sim params) ────────────────────────

# Sundt — from experiments_v2/sundt_validation.py
_SUNDT = dict(
    V_REST  = -60.0,
    CELSIUS = 37.0,
    MSHIFT  = -6.0, HSHIFT  =  6.0, ISHIFT  = 0.0,
    VHALFN  = -32.0, VHALFL = -61.0,
    A0N     = 0.03,  A0L    = 0.001,
    ZETAN   = -5.0,  ZETAL  = 2.0,
    GMN     = 0.4,   GML    = 1.0,
    GNABAR  = 0.04,  GKDRBAR= 0.04,
    G_PAS   = 1e-4,  E_PAS  = -60.0,
    ENA     = 50.0,  EK     = -90.0,
)

# Sweeney — verified against experiments_v2/sweeney_validation.py
_SWEENEY = dict(
    V_REST  = -80.0,
    GNABAR  = 1.445,    ENA = 35.64,
    GL      = 0.128,    EL  = -80.01,
)

# Rattay — verified against experiments_v2/rattay_validation.py
_RATTAY = dict(
    V_REST  = -70.0,
    CELSIUS = 37.0,
    GNABAR  = 0.12,
    GKBAR   = 0.036,
    G_PAS   = 3e-4,
    ENA     =  45.0,
    EK      = -82.0,
    E_PAS   = -59.4,
)


# ─── Per-model (state0, membrane_fn) builders ─────────────────────────────────
#
# Each builder takes a geom (one fibre) + dt + the arrays_from_geometry static
# dict and returns:
#   state0      — initial gate state pytree (tuple of arrays, one per gate)
#   membrane_fn — (Vm, state, dt) → (g_eff_uS, i_ion_nA, new_state)
# The returned closures are pure-jax and can be vmapped across fibres.

def _build_sundt(geom, dt, static):
    P = _SUNDT
    A_in = static["A_in_cm2"]
    is_node = static["is_node"]
    has_channel = is_node                       # Sundt: every section is a node

    v0 = jnp.float64(P["V_REST"])
    (am0, bm0), (ah0, bh0) = SundtAxon._nahh_alpha_beta(
        v0, P["CELSIUS"], P["MSHIFT"], P["HSHIFT"], P["ISHIFT"])
    (an0, bn0), (al0, bl0) = SundtAxon._borgkdr_alpha_beta(
        v0, P["CELSIUS"], P["VHALFN"], P["VHALFL"], P["A0N"], P["A0L"],
        P["ZETAN"], P["ZETAL"], P["GMN"], P["GML"])
    m0, h0 = float(am0 / (am0 + bm0)), float(ah0 / (ah0 + bh0))
    n0, l0 = float(an0 / (an0 + bn0)), float(al0 / (al0 + bl0))

    state0 = (
        jnp.where(has_channel, m0, 0.0),
        jnp.where(has_channel, h0, 0.0),
        jnp.where(has_channel, n0, 0.0),
        jnp.where(has_channel, l0, 0.0),
    )

    def membrane_fn(Vm, state, dt):
        M, H, N, L = state
        (am, bm), (ah, bh) = SundtAxon._nahh_alpha_beta(
            Vm, P["CELSIUS"], P["MSHIFT"], P["HSHIFT"], P["ISHIFT"])
        M2 = solve_gate_exponential(M, dt, am, bm)
        H2 = solve_gate_exponential(H, dt, ah, bh)
        (an, bn), (al, bl) = SundtAxon._borgkdr_alpha_beta(
            Vm, P["CELSIUS"], P["VHALFN"], P["VHALFL"],
            P["A0N"], P["A0L"], P["ZETAN"], P["ZETAL"], P["GMN"], P["GML"])
        N2 = solve_gate_exponential(N, dt, an, bn)
        L2 = solve_gate_exponential(L, dt, al, bl)
        g_na = P["GNABAR"]  * M2**3 * H2
        g_k  = P["GKDRBAR"] * N2**3 * L2
        g_tot = (g_na + g_k + P["G_PAS"]) * A_in * 1e6
        i_ion = (g_na * (Vm - P["ENA"]) + g_k * (Vm - P["EK"])
                 + P["G_PAS"] * (Vm - P["E_PAS"])) * A_in * 1e6
        return g_tot, i_ion, (M2, H2, N2, L2)

    return state0, membrane_fn, P["V_REST"]


def _build_sweeney(geom, dt, static):
    P = _SWEENEY
    A_in = static["A_in_cm2"]
    is_node = static["is_node"]
    has_channel = jnp.asarray(
        [st == "node" for st in geom.section_type], dtype=jnp.bool_,
    )

    v0 = jnp.float64(P["V_REST"])
    (am0, bm0), (ah0, bh0) = SweeneyNode._alpha_beta(v0)
    m0, h0 = float(am0 / (am0 + bm0)), float(ah0 / (ah0 + bh0))

    state0 = (
        jnp.where(has_channel, m0, 0.0),
        jnp.where(has_channel, h0, 0.0),
    )

    def membrane_fn(Vm, state, dt):
        M, H = state
        (am, bm), (ah, bh) = SweeneyNode._alpha_beta(Vm)
        M2 = solve_gate_exponential(M, dt, am, bm)
        H2 = solve_gate_exponential(H, dt, ah, bh)
        g_na   = P["GNABAR"] * M2**2 * H2
        g_node = (g_na + P["GL"]) * A_in * 1e6
        i_node = (g_na * (Vm - P["ENA"]) + P["GL"] * (Vm - P["EL"])) * A_in * 1e6
        # Myelin sections: tiny non-zero leak so block-Thomas stays non-singular
        g_mye = jnp.full_like(g_node, 1e-9)
        i_mye = jnp.zeros_like(i_node)
        g_eff = jnp.where(has_channel, g_node, g_mye)
        i_ion = jnp.where(has_channel, i_node, i_mye)
        return g_eff, i_ion, (M2, H2)

    return state0, membrane_fn, P["V_REST"]


def _build_rattay(geom, dt, static):
    P = _RATTAY
    A_in = static["A_in_cm2"]
    is_node = static["is_node"]
    has_channel = is_node                       # Rattay: every section is a node

    v0 = jnp.float64(P["V_REST"])
    (am0, bm0), (ah0, bh0), (an0, bn0) = RattayHH._alpha_beta(v0, P["CELSIUS"])
    m0 = float(am0 / (am0 + bm0))
    h0 = float(ah0 / (ah0 + bh0))
    n0 = float(an0 / (an0 + bn0))

    state0 = (
        jnp.where(has_channel, m0, 0.0),
        jnp.where(has_channel, h0, 0.0),
        jnp.where(has_channel, n0, 0.0),
    )

    def membrane_fn(Vm, state, dt):
        M, H, N = state
        (am, bm), (ah, bh), (an, bn) = RattayHH._alpha_beta(Vm, P["CELSIUS"])
        M2 = solve_gate_exponential(M, dt, am, bm)
        H2 = solve_gate_exponential(H, dt, ah, bh)
        N2 = solve_gate_exponential(N, dt, an, bn)
        g_na = P["GNABAR"] * M2**3 * H2
        g_k  = P["GKBAR"]  * N2**4
        g_tot = (g_na + g_k + P["G_PAS"]) * A_in * 1e6
        i_ion = (g_na * (Vm - P["ENA"]) + g_k * (Vm - P["EK"])
                 + P["G_PAS"] * (Vm - P["E_PAS"])) * A_in * 1e6
        return g_tot, i_ion, (M2, H2, N2)

    return state0, membrane_fn, P["V_REST"]


_BUILDERS = {
    "sundt":   _build_sundt,
    "sweeney": _build_sweeney,
    "rattay":  _build_rattay,
}


# ─── Generic batched m_max integrator (unmyelinated models) ───────────────────

def _integrate_one_fiber_m_max_generic(
    static, state0, membrane_fn, Ve_seq, dt, v_rest, pulse_mask=None,
):
    """Single-fibre forward pass, tracking max(m) per compartment.

    Vm = Vi - Vp.  Returns m_max array [n_comp] = element-wise max of the
    first gate state over the entire scan.  This is the activation proxy
    consumed by jaxfibers.optim.losses.activation_proxy_batch.
    """
    n = static["is_node"].shape[0]
    T = Ve_seq.shape[0]

    # ``integrate`` expects Ve as the [n_comp] *spatial* profile, scaled
    # element-wise by pulse_mask[s] each step.  We instead get Ve_seq already
    # combined ([T, n_comp]) from the selectivity-sweep pipeline.  So inline
    # the step here rather than calling integrate(), using a constant
    # pulse_mask = 1.  This matches what _integrate_one_fiber_m_max does in
    # batch_solve.py.
    Ve_prev = jnp.concatenate(
        [jnp.zeros((1, n), dtype=jnp.float64), Ve_seq[:-1]], axis=0,
    )

    @jax.checkpoint
    def step(carry, s):
        Vi, Vp, st, m_max = carry
        Vm = Vi - Vp
        g_eff, i_ion, st2 = membrane_fn(Vm, st, dt)
        Vi2, Vp2 = _be_step(
            Vi, Vp, Ve_seq[s], Ve_prev[s], g_eff, i_ion,
            static["Cm_dt"], static["Cmy_dt"], static["gmy"],
            static["Gi_diag"], static["Gp_diag"],
            static["Up"], static["Low"], static["is_node"],
        )
        # First gate in st2 is always m (we enforce this in the builders).
        m_max_new = jnp.maximum(m_max, st2[0])
        return (Vi2, Vp2, st2, m_max_new), None

    Vi0 = jnp.full(n, v_rest, dtype=jnp.float64)
    Vp0 = jnp.zeros(n, dtype=jnp.float64)
    m_max0 = jnp.zeros(n, dtype=jnp.float64)
    (_, _, _, m_max_f), _ = jax.lax.scan(
        step, (Vi0, Vp0, state0, m_max0), jnp.arange(T),
    )
    return m_max_f


def make_batch_integrate_m_max_generic(model: str, geoms: list, dt: float):
    """Build a closure that maps Ve_seq_batch [n_fibers, T, n_comp]
    to m_max_batch [n_fibers, n_comp] for a non-MRG model.

    Pre-computes the per-fibre static arrays + membrane functions once;
    the returned function is JIT-friendly and can be called repeatedly
    inside the optimisation loop with different Ve_seq inputs.
    """
    if model not in _BUILDERS:
        raise ValueError(
            f"Cross-model selectivity not supported for model={model!r}. "
            f"Supported: 'mrg' (use the legacy batch_solve path) or "
            f"{sorted(_BUILDERS)}."
        )
    builder = _BUILDERS[model]

    # Build per-fibre statics + closures once.  We stack everything that's
    # array-valued; the closures themselves are kept in a Python list.
    statics_list = [arrays_from_geometry(g, dt) for g in geoms]
    state0_list, membrane_fn_list, v_rest_list = [], [], []
    for g, s in zip(geoms, statics_list):
        s0, mfn, vr = builder(g, dt, s)
        state0_list.append(s0)
        membrane_fn_list.append(mfn)
        v_rest_list.append(vr)
    assert all(v == v_rest_list[0] for v in v_rest_list), \
        f"v_rest must be identical across fibres for model {model}"
    v_rest = v_rest_list[0]

    def batch_integrate_m_max(Ve_seq_batch):
        """Ve_seq_batch: [n_fibers, T, n_comp] → m_max: [n_fibers, n_comp]."""
        m_max_per_fiber = []
        for f, (static, s0, mfn) in enumerate(
            zip(statics_list, state0_list, membrane_fn_list)
        ):
            m_max_per_fiber.append(
                _integrate_one_fiber_m_max_generic(
                    static, s0, mfn, Ve_seq_batch[f], dt, v_rest,
                )
            )
        return jnp.stack(m_max_per_fiber, axis=0)

    return batch_integrate_m_max


def supports_cross_model(model: str) -> bool:
    """True if a non-MRG model is registered for cross-model selectivity."""
    return model in _BUILDERS


# ─── Optimizer-compatible wrappers (signatures match batch_solve.py) ──────────
#
# The optimizer (jaxfibers/optim/optimizer.py) calls
#   batch_integrate_m_max_fd(fs_tiled, s0_tiled, Ve_comb_flat, pulse_j,
#                            pulse_prev_j, dt)
# with Ve_comb_flat shape [n_configs*n_fibers, n_comp].  The 7 call sites
# inside the optimizer don't know which model is running — they just call
# the imported names.  For cross-model support, selectivity_sweep
# monkey-patches optimizer.batch_integrate_m_max / _fd at startup with
# the closures returned below.

def make_batch_integrate_m_max_fd_factory(model: str, geoms: list, dt: float):
    """Build a closure with the same signature as
    batch_solve.batch_integrate_m_max_fd, but driven by the generic
    per-model membrane functions.

    The closure ignores the fs_tiled / s0_tiled inputs (they were
    MRG-specific) and uses the per-fibre static arrays + membrane fns
    captured here at registration time.

    Signature of returned closure:
      m_max_flat = fn(fs_tiled, s0_tiled, Ve_comb_flat,
                       pulse_mask, pulse_mask_prev, dt)
    where:
      Ve_comb_flat   : [n_configs * n_fibers, n_comp]
      pulse_mask     : [T]   (1.0 where the pulse is on, else 0)
      pulse_mask_prev: [T]   (1.0 where the previous-step pulse was on)
      Returns m_max_flat : [n_configs * n_fibers, n_comp]
    """
    if model not in _BUILDERS:
        raise ValueError(f"make_batch_integrate_m_max_fd_factory: unsupported "
                         f"model {model!r}")
    builder = _BUILDERS[model]

    # Build per-fibre statics + closures once.
    n_fibers = len(geoms)
    statics_list = [arrays_from_geometry(g, dt) for g in geoms]
    state0_list  = []
    membrane_fn_list = []
    v_rest_list = []
    for g, s in zip(geoms, statics_list):
        s0, mfn, vr = builder(g, dt, s)
        state0_list.append(s0)
        membrane_fn_list.append(mfn)
        v_rest_list.append(vr)
    v_rest = float(v_rest_list[0])

    def batch_integrate_m_max_fd(
        fs_tiled,        # unused; the optimiser still passes it for API parity
        s0_tiled,        # unused
        Ve_comb_flat,    # [n_configs * n_fibers, n_comp]
        pulse_mask,      # [T]
        pulse_mask_prev, # [T]
        dt_,
    ):
        # Reshape Ve_comb_flat back to [n_configs, n_fibers, n_comp].
        n_total = Ve_comb_flat.shape[0]
        n_comp  = Ve_comb_flat.shape[1]
        assert n_total % n_fibers == 0, (
            f"Ve_comb_flat first dim {n_total} not divisible by "
            f"n_fibers={n_fibers}")
        n_configs = n_total // n_fibers
        Ve_comb_all = Ve_comb_flat.reshape(n_configs, n_fibers, n_comp)

        # Build Ve_seq[c, f, t, n_comp] = Ve_comb[c, f, n_comp] * pulse[t].
        # Then per-(c, f) integrate one fibre.
        pulse_j = jnp.asarray(pulse_mask, dtype=jnp.float64)

        # The selectivity-sweep convention: Ve_seq is [T, n_comp] per fibre.
        # We iterate Python-side over fibres (closure list of statics differs
        # per fibre, hence vmap is hard); JIT inside _integrate_one_fiber.
        out = []
        for c in range(n_configs):
            for f in range(n_fibers):
                Ve_seq_f = pulse_j[:, None] * Ve_comb_all[c, f][None, :]
                out.append(_integrate_one_fiber_m_max_generic(
                    statics_list[f], state0_list[f], membrane_fn_list[f],
                    Ve_seq_f, dt_, v_rest,
                ))
        return jnp.stack(out, axis=0)   # [n_configs * n_fibers, n_comp]

    return batch_integrate_m_max_fd


def make_batch_integrate_m_max_factory(model: str, geoms: list, dt: float):
    """Build a closure matching batch_solve.batch_integrate_m_max for
    a non-MRG model.

    Signature of returned closure:
      m_max_batch = fn(fs_batch, s0_batch, Ve_seq_batch, dt)
    where Ve_seq_batch is [n_fibers, T, n_comp].
    """
    if model not in _BUILDERS:
        raise ValueError(f"make_batch_integrate_m_max_factory: unsupported "
                         f"model {model!r}")
    builder = _BUILDERS[model]

    n_fibers = len(geoms)
    statics_list = [arrays_from_geometry(g, dt) for g in geoms]
    state0_list, membrane_fn_list, v_rest_list = [], [], []
    for g, s in zip(geoms, statics_list):
        s0, mfn, vr = builder(g, dt, s)
        state0_list.append(s0)
        membrane_fn_list.append(mfn)
        v_rest_list.append(vr)
    v_rest = float(v_rest_list[0])

    def batch_integrate_m_max(fs_batch, s0_batch, Ve_seq_batch, dt_):
        out = []
        for f in range(n_fibers):
            out.append(_integrate_one_fiber_m_max_generic(
                statics_list[f], state0_list[f], membrane_fn_list[f],
                Ve_seq_batch[f], dt_, v_rest,
            ))
        return jnp.stack(out, axis=0)

    return batch_integrate_m_max
