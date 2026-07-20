"""Model-agnostic batched m_max integrator, vmapped over fibres.

The original ``batch_solve.py`` is MRG-specific: it imports
``AxnodeMyel`` and hardcodes the 4-gate (m, h, mp, s) state.  The
unmyelinated models (Sundt, Sweeney, Rattay) use different channels
and gate counts:

  - Sundt:    SundtAxon    states (m, h, n, l)
  - Sweeney:  SweeneyNode  states (m, h)
  - Rattay:   RattayHH     states (m, h, n)

This module provides a generic per-fibre forward pass parameterised on
a per-model ``step_fn`` (Vm, state, A_in, has_channel) → (g_eff, i_ion,
new_state).  Static per-fibre arrays are stacked into a single
namedtuple (``FiberStaticsGeneric``) that ``jax.vmap`` iterates over —
so unmyelinated fibres are GPU-parallelised exactly the way the MRG
path is.  No Python loop over fibres anywhere.

Public API consumed by ``experiments_v2/selectivity_sweep.py``:

  make_batch_integrate_m_max_factory(model, geoms, dt)
      → closure with signature
        (fs_batch_unused, s0_batch_unused, Ve_seq_batch, dt) → m_max_batch
        Drop-in for batch_solve.batch_integrate_m_max.

  make_batch_integrate_m_max_fd_factory(model, geoms, dt)
      → closure with signature
        (fs_tiled_unused, s0_tiled_unused, Ve_comb_flat,
         pulse_mask, pulse_mask_prev, dt) → m_max_flat
        Drop-in for batch_solve.batch_integrate_m_max_fd.

Both closures capture the stacked per-fibre statics once at
registration time and JIT-compile a vmapped scan; subsequent calls
pay only the GPU launch cost.
"""
from __future__ import annotations

from typing import NamedTuple, Callable

import numpy as np
import jax
import jax.numpy as jnp

from jaxley.solver_gate import solve_gate_exponential

from jaxon.stim.extracellular_coupled import arrays_from_geometry, _be_step

from jaxon.channels.sundt_channels   import SundtAxon
from jaxon.channels.sweeney_channels import SweeneyNode
from jaxon.channels.rattay_channels  import RattayHH


# ─── Per-model constants (gates + V_REST + channel reversal potentials) ───────
# Numbers verified line-by-line against experiments_v2/{sundt,sweeney,rattay}_validation.py.

_SUNDT = dict(
    V_REST  = -60.0,
    CELSIUS = 37.0,
    MSHIFT  = -6.0,  HSHIFT  =  6.0, ISHIFT  = 0.0,
    VHALFN  = -32.0, VHALFL  = -61.0,
    A0N     =  0.03, A0L     =  0.001,
    ZETAN   = -5.0,  ZETAL   =  2.0,
    GMN     =  0.4,  GML     =  1.0,
    GNABAR  =  0.04, GKDRBAR =  0.04,
    G_PAS   =  1e-4, E_PAS   = -60.0,
    ENA     = 50.0,  EK      = -90.0,
)

_SWEENEY = dict(
    V_REST  = -80.0,
    GNABAR  =  1.445, ENA     =  35.64,
    GL      =  0.128, EL      = -80.01,
)

_RATTAY = dict(
    V_REST  = -70.0,
    CELSIUS = 37.0,
    GNABAR  =  0.12,  GKBAR   =   0.036, G_PAS = 3e-4,
    ENA     = 45.0,   EK      = -82.0,   E_PAS = -59.4,
)


# ─── Stacked per-fibre statics ────────────────────────────────────────────────
#
# These are the arrays the per-step membrane functions actually need
# (A_in, is_node, has_channel) plus everything ``_be_step`` needs to
# advance Vi, Vp by one backward-Euler step.  Stacked across fibres so
# jax.vmap can iterate over the leading axis.

class FiberStaticsGeneric(NamedTuple):
    # Per-step backward-Euler arrays (model-independent: come from
    # arrays_from_geometry).  All shape [n_fibers, n_comp] except Up/Low
    # which are [n_fibers, n_comp, 2, 2].
    Cm_dt:    jnp.ndarray
    Cmy_dt:   jnp.ndarray
    gmy:      jnp.ndarray
    A_in_cm2: jnp.ndarray
    is_node:  jnp.ndarray
    Gi_diag:  jnp.ndarray
    Gp_diag:  jnp.ndarray
    Up:       jnp.ndarray
    Low:      jnp.ndarray
    # Per-model: which compartments have active channels.  Same shape
    # [n_fibers, n_comp]; for Sundt and Rattay it equals is_node; for
    # Sweeney it equals (section_type == "node").
    has_channel: jnp.ndarray


def _stack_statics(model: str, geoms: list, dt: float) -> FiberStaticsGeneric:
    """Build the per-fibre static arrays for every fibre, then stack."""
    Cm_dt, Cmy_dt, gmy, A_in, is_node = [], [], [], [], []
    Gi_diag, Gp_diag, Up, Low, has_channel = [], [], [], [], []
    for g in geoms:
        s = arrays_from_geometry(g, dt)
        Cm_dt.append(s["Cm_dt"]);   Cmy_dt.append(s["Cmy_dt"])
        gmy.append(s["gmy"]);       A_in.append(s["A_in_cm2"])
        is_node.append(s["is_node"])
        Gi_diag.append(s["Gi_diag"]); Gp_diag.append(s["Gp_diag"])
        Up.append(s["Up"]);          Low.append(s["Low"])
        if model in ("sundt", "rattay"):
            # Every section is a node for unmyelinated.
            has_channel.append(s["is_node"])
        elif model == "sweeney":
            has_channel.append(jnp.asarray(
                [st == "node" for st in g.section_type], dtype=jnp.bool_,
            ))
        else:
            raise ValueError(f"_stack_statics: unsupported model {model!r}")
    return FiberStaticsGeneric(
        Cm_dt   = jnp.stack(Cm_dt),
        Cmy_dt  = jnp.stack(Cmy_dt),
        gmy     = jnp.stack(gmy),
        A_in_cm2= jnp.stack(A_in),
        is_node = jnp.stack(is_node),
        Gi_diag = jnp.stack(Gi_diag),
        Gp_diag = jnp.stack(Gp_diag),
        Up      = jnp.stack(Up),
        Low     = jnp.stack(Low),
        has_channel = jnp.stack(has_channel),
    )


# ─── Per-model pure-JAX step functions ────────────────────────────────────────
#
# Each step_fn closes over the model's α/β rates + reversal potentials
# and operates on a single fibre's (Vm, state, A_in, has_channel) — no
# Python state, fully traceable, vmap-friendly.

def _sundt_step(Vm, state, dt, A_in, has_channel):
    P = _SUNDT
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
    g_na  = P["GNABAR"]  * M2**3 * H2
    g_k   = P["GKDRBAR"] * N2**3 * L2
    g_tot = (g_na + g_k + P["G_PAS"]) * A_in * 1e6
    i_ion = (g_na * (Vm - P["ENA"]) + g_k * (Vm - P["EK"])
             + P["G_PAS"] * (Vm - P["E_PAS"])) * A_in * 1e6
    return g_tot, i_ion, (M2, H2, N2, L2)


def _sweeney_step(Vm, state, dt, A_in, has_channel):
    P = _SWEENEY
    M, H = state
    (am, bm), (ah, bh) = SweeneyNode._alpha_beta(Vm)
    M2 = solve_gate_exponential(M, dt, am, bm)
    H2 = solve_gate_exponential(H, dt, ah, bh)
    g_na   = P["GNABAR"] * M2**2 * H2
    g_node = (g_na + P["GL"]) * A_in * 1e6
    i_node = (g_na * (Vm - P["ENA"]) + P["GL"] * (Vm - P["EL"])) * A_in * 1e6
    g_mye  = jnp.full_like(g_node, 1e-9)
    i_mye  = jnp.zeros_like(i_node)
    g_eff  = jnp.where(has_channel, g_node, g_mye)
    i_ion  = jnp.where(has_channel, i_node, i_mye)
    return g_eff, i_ion, (M2, H2)


def _rattay_step(Vm, state, dt, A_in, has_channel):
    P = _RATTAY
    M, H, N = state
    (am, bm), (ah, bh), (an, bn) = RattayHH._alpha_beta(Vm, P["CELSIUS"])
    M2 = solve_gate_exponential(M, dt, am, bm)
    H2 = solve_gate_exponential(H, dt, ah, bh)
    N2 = solve_gate_exponential(N, dt, an, bn)
    g_na  = P["GNABAR"] * M2**3 * H2
    g_k   = P["GKBAR"]  * N2**4
    g_tot = (g_na + g_k + P["G_PAS"]) * A_in * 1e6
    i_ion = (g_na * (Vm - P["ENA"]) + g_k * (Vm - P["EK"])
             + P["G_PAS"] * (Vm - P["E_PAS"])) * A_in * 1e6
    return g_tot, i_ion, (M2, H2, N2)


def _initial_state_sundt(has_channel):
    P = _SUNDT
    v0 = jnp.float64(P["V_REST"])
    (am, bm), (ah, bh) = SundtAxon._nahh_alpha_beta(
        v0, P["CELSIUS"], P["MSHIFT"], P["HSHIFT"], P["ISHIFT"])
    (an, bn), (al, bl) = SundtAxon._borgkdr_alpha_beta(
        v0, P["CELSIUS"], P["VHALFN"], P["VHALFL"],
        P["A0N"], P["A0L"], P["ZETAN"], P["ZETAL"], P["GMN"], P["GML"])
    m0 = am / (am + bm); h0 = ah / (ah + bh)
    n0 = an / (an + bn); l0 = al / (al + bl)
    z = jnp.zeros_like(has_channel, dtype=jnp.float64)
    return (jnp.where(has_channel, m0, z),
            jnp.where(has_channel, h0, z),
            jnp.where(has_channel, n0, z),
            jnp.where(has_channel, l0, z))


def _initial_state_sweeney(has_channel):
    P = _SWEENEY
    v0 = jnp.float64(P["V_REST"])
    (am, bm), (ah, bh) = SweeneyNode._alpha_beta(v0)
    m0 = am / (am + bm); h0 = ah / (ah + bh)
    z = jnp.zeros_like(has_channel, dtype=jnp.float64)
    return (jnp.where(has_channel, m0, z),
            jnp.where(has_channel, h0, z))


def _initial_state_rattay(has_channel):
    P = _RATTAY
    v0 = jnp.float64(P["V_REST"])
    (am, bm), (ah, bh), (an, bn) = RattayHH._alpha_beta(v0, P["CELSIUS"])
    m0 = am / (am + bm); h0 = ah / (ah + bh); n0 = an / (an + bn)
    z = jnp.zeros_like(has_channel, dtype=jnp.float64)
    return (jnp.where(has_channel, m0, z),
            jnp.where(has_channel, h0, z),
            jnp.where(has_channel, n0, z))


_MODEL_REGISTRY = {
    "sundt":   dict(step_fn=_sundt_step,   state0_fn=_initial_state_sundt,   v_rest=_SUNDT["V_REST"]),
    "sweeney": dict(step_fn=_sweeney_step, state0_fn=_initial_state_sweeney, v_rest=_SWEENEY["V_REST"]),
    "rattay":  dict(step_fn=_rattay_step,  state0_fn=_initial_state_rattay,  v_rest=_RATTAY["V_REST"]),
}


def supports_cross_model(model: str) -> bool:
    return model in _MODEL_REGISTRY


# ─── Single-fibre forward pass (vmappable) ────────────────────────────────────

def _make_one_fiber_m_max(step_fn, state0_fn, v_rest):
    """Build a (jit-compiled) one-fibre m_max integrator for one model.

    Returns a function: (fs_one, Ve_seq, dt) -> m_max [n_comp]
    where fs_one is a FiberStaticsGeneric with the leading fibre dim
    removed (i.e. plain [n_comp] arrays).
    """
    def integrate_one(fs_one, Ve_seq, dt):
        n = fs_one.is_node.shape[0]
        T = Ve_seq.shape[0]
        Ve_prev = jnp.concatenate(
            [jnp.zeros((1, n), dtype=jnp.float64), Ve_seq[:-1]], axis=0,
        )

        @jax.checkpoint
        def step(carry, s):
            Vi, Vp, st, m_max = carry
            Vm = Vi - Vp
            g_eff, i_ion, st2 = step_fn(Vm, st, dt, fs_one.A_in_cm2, fs_one.has_channel)
            Vi2, Vp2 = _be_step(
                Vi, Vp, Ve_seq[s], Ve_prev[s], g_eff, i_ion,
                fs_one.Cm_dt, fs_one.Cmy_dt, fs_one.gmy,
                fs_one.Gi_diag, fs_one.Gp_diag,
                fs_one.Up, fs_one.Low, fs_one.is_node,
            )
            # First entry of state tuple is always m (we enforce in builders).
            m_max_new = jnp.maximum(m_max, st2[0])
            return (Vi2, Vp2, st2, m_max_new), None

        Vi0 = jnp.full(n, v_rest, dtype=jnp.float64)
        Vp0 = jnp.zeros(n, dtype=jnp.float64)
        st0 = state0_fn(fs_one.has_channel)
        m_max0 = jnp.zeros(n, dtype=jnp.float64)
        (_, _, _, m_max_f), _ = jax.lax.scan(
            step, (Vi0, Vp0, st0, m_max0), jnp.arange(T),
        )
        return m_max_f

    return integrate_one


# ─── Public factories (drop-in for batch_solve.batch_integrate_m_max{,_fd}) ──

def make_batch_integrate_m_max_factory(model: str, geoms: list, dt: float):
    """vmap-batched m_max integrator for one of the supported models.

    Signature of the returned closure:
      m_max = fn(fs_batch_unused, s0_batch_unused, Ve_seq_batch, dt)
    where Ve_seq_batch has shape [n_fibers, T, n_comp] and m_max has
    shape [n_fibers, n_comp].  The first two arguments are the
    optimizer-side fiber_statics_batch / state0_batch and are ignored
    here (we use the closed-over statics built at registration).
    """
    if model not in _MODEL_REGISTRY:
        raise ValueError(f"make_batch_integrate_m_max_factory: unsupported model "
                         f"{model!r}.  Supported: {sorted(_MODEL_REGISTRY)}")
    M = _MODEL_REGISTRY[model]
    fs_batch = _stack_statics(model, geoms, dt)
    integrate_one = _make_one_fiber_m_max(M["step_fn"], M["state0_fn"], M["v_rest"])
    # vmap across the leading (fibre) axis of fs_batch and Ve_seq_batch.
    batched = jax.jit(jax.vmap(integrate_one, in_axes=(0, 0, None)))

    def batch_integrate_m_max(fs_unused, s0_unused, Ve_seq_batch, dt_):
        return batched(fs_batch, Ve_seq_batch, dt_)

    return batch_integrate_m_max


def make_batch_integrate_m_max_fd_factory(model: str, geoms: list, dt: float):
    """vmap-batched m_max integrator for the FD-gradient path.

    Signature matches batch_solve.batch_integrate_m_max_fd:
      m_max_flat = fn(fs_tiled_unused, s0_tiled_unused, Ve_comb_flat,
                      pulse_mask, pulse_mask_prev, dt)
    where Ve_comb_flat has shape [n_configs * n_fibers, n_comp] (the
    optimizer tiles n_configs = K+1 perturbed amplitude configurations
    along the fibre axis).  Returns m_max_flat: [n_configs * n_fibers, n_comp].
    """
    if model not in _MODEL_REGISTRY:
        raise ValueError(f"make_batch_integrate_m_max_fd_factory: unsupported "
                         f"model {model!r}")
    M = _MODEL_REGISTRY[model]
    fs_batch = _stack_statics(model, geoms, dt)
    n_fibers = fs_batch.is_node.shape[0]
    integrate_one = _make_one_fiber_m_max(M["step_fn"], M["state0_fn"], M["v_rest"])
    batched = jax.vmap(integrate_one, in_axes=(0, 0, None))

    # Tile statics across n_configs once at registration; vmap handles the
    # whole [n_configs * n_fibers] axis as one big vmapped batch.  This
    # mirrors what batch_solve._tile_fiber_statics does for MRG.
    def _tile_n_configs(fs: FiberStaticsGeneric, n_configs: int):
        return FiberStaticsGeneric(*[
            jnp.tile(getattr(fs, field), (n_configs,) + (1,) * (getattr(fs, field).ndim - 1))
            for field in FiberStaticsGeneric._fields
        ])

    @jax.jit
    def _impl(fs_tiled, Ve_comb_flat, pulse_mask):
        # Ve_comb_flat: [n_total, n_comp], pulse_mask: [T].
        # Ve_seq[f, t, n] = pulse_mask[t] * Ve_comb_flat[f, n].
        Ve_seq_batch = pulse_mask[None, :, None] * Ve_comb_flat[:, None, :]
        # vmap across the leading fibre axis (which spans all configs).
        return batched(fs_tiled, Ve_seq_batch, dt)

    # Cache tiled-statics across iterations so we don't rebuild them every call.
    _cache = {}
    def batch_integrate_m_max_fd(
        fs_tiled_unused, s0_tiled_unused, Ve_comb_flat,
        pulse_mask, pulse_mask_prev_unused, dt_,
    ):
        # Determine n_configs from the flat batch size; cache the tiled fs.
        n_total = Ve_comb_flat.shape[0]
        n_configs = n_total // n_fibers
        if n_configs not in _cache:
            _cache[n_configs] = _tile_n_configs(fs_batch, n_configs)
        return _impl(_cache[n_configs], Ve_comb_flat, pulse_mask)

    return batch_integrate_m_max_fd
