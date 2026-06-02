"""Vmapped multi-fiber coupled solver for spatial selectivity optimization.

All fibers share the same n_nodes (and therefore the same n_comp), so their
static geometry arrays can be stacked into [n_fibers, n_comp] batches and
differentiated through with jax.vmap + jax.grad.

Public API
----------
FiberStatics         : NamedTuple of per-compartment solver arrays for one fiber.
build_fiber_statics  : MrgGeometry → FiberStatics (one fiber).
stack_fiber_statics  : list[MrgGeometry] → FiberStatics batched [n_fibers, n_comp, ...].
initial_states_batch : list[MrgGeometry] → initial (M,H,MP,S) stacked [n_fibers, n_comp].
batch_integrate_m_max: vmap over fibers, returns m_max [n_fibers, n_comp].
"""
from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxley.solver_gate import solve_gate_exponential

from jaxfibers.stim.extracellular_coupled import _be_step, arrays_from_geometry
from jaxfibers.channels.mrg_axnode import AxnodeMyel
from jaxfibers.fibers.mrg import V_REST, CM_AXON, G_PAS_MYSA, G_PAS_FLUT, G_PAS_STIN

jax.config.update("jax_enable_x64", True)

# MRG channel constants (same as mrg_validation.py)
_GNABAR  = 3.0
_GNAPBAR = 0.01
_GKBAR   = 0.08
_GL      = 0.007
_ENA     = 50.0
_EK      = -90.0
_EL      = -90.0
_CELSIUS = 37.0

_STYPE_GP = {"node": 0.0, "mysa": G_PAS_MYSA, "flut": G_PAS_FLUT, "stin": G_PAS_STIN}


class FiberStatics(NamedTuple):
    """Per-compartment static solver arrays for one MRG fiber.

    When stacked across fibers these become [n_fibers, n_comp, ...] batches
    that jax.vmap can iterate over.
    """
    Cm_dt:    jnp.ndarray   # [n_comp]  membrane Cm / dt  (nF/ms = µS)
    Cmy_dt:   jnp.ndarray   # [n_comp]  myelin Cm / dt
    gmy:      jnp.ndarray   # [n_comp]  myelin conductance (µS)
    A_in_cm2: jnp.ndarray   # [n_comp]  inner axon membrane area (cm²)
    is_node:  jnp.ndarray   # [n_comp]  bool: True at nodes of Ranvier
    Gi_diag:  jnp.ndarray   # [n_comp]  intracellular axial conductance diagonal
    Gp_diag:  jnp.ndarray   # [n_comp]  periaxonal axial conductance diagonal
    Up:       jnp.ndarray   # [n_comp, 2, 2]  upper off-diagonal blocks
    Low:      jnp.ndarray   # [n_comp, 2, 2]  lower off-diagonal blocks
    g_pas:    jnp.ndarray   # [n_comp]  passive leak conductance (S/cm²)


def build_fiber_statics(geom, dt: float) -> FiberStatics:
    """Build FiberStatics for a single MrgGeometry."""
    geom_c = dataclasses.replace(geom, cm_uF_cm2=[CM_AXON] * geom.n_comp)
    s = arrays_from_geometry(geom_c, dt)
    g_pas = jnp.asarray([_STYPE_GP[st] for st in geom.section_type], dtype=jnp.float64)
    return FiberStatics(
        Cm_dt=s["Cm_dt"],
        Cmy_dt=s["Cmy_dt"],
        gmy=s["gmy"],
        A_in_cm2=s["A_in_cm2"],
        is_node=s["is_node"],
        Gi_diag=s["Gi_diag"],
        Gp_diag=s["Gp_diag"],
        Up=s["Up"],
        Low=s["Low"],
        g_pas=g_pas,
    )


def stack_fiber_statics(geom_list: list, dt: float) -> FiberStatics:
    """Build and stack FiberStatics for all fibers into a [n_fibers, ...] batch."""
    statics = [build_fiber_statics(g, dt) for g in geom_list]
    return FiberStatics(*[
        jnp.stack([getattr(s, field) for s in statics])
        for field in FiberStatics._fields
    ])


def initial_states_batch(geom_list: list) -> tuple:
    """Initial resting gate states for all fibers, stacked [n_fibers, n_comp].

    Returns
    -------
    (M0, H0, MP0, S0) : each [n_fibers, n_comp] float64
        Gate states at V_rest. Internodes (no active channels) are set to 0.
    """
    v0 = jnp.float64(V_REST)
    (a_m, b_m), (a_h, b_h), (a_mp, b_mp), (a_s, b_s) = AxnodeMyel._alpha_beta(v0, _CELSIUS)
    m0  = float(a_m  / (a_m  + b_m))
    h0  = float(a_h  / (a_h  + b_h))
    mp0 = float(a_mp / (a_mp + b_mp))
    s0  = float(a_s  / (a_s  + b_s))

    M_rows, H_rows, MP_rows, S_rows = [], [], [], []
    for geom in geom_list:
        isn = jnp.asarray(geom.is_node, dtype=jnp.float64)
        M_rows.append(isn * m0)
        H_rows.append(isn * h0)
        MP_rows.append(isn * mp0)
        S_rows.append(isn * s0)

    return (
        jnp.stack(M_rows),   # [n_fibers, n_comp]
        jnp.stack(H_rows),
        jnp.stack(MP_rows),
        jnp.stack(S_rows),
    )


# ─────────────────────────────────────────────── single-fiber inner loop ─────

def _integrate_one_fiber_m_max(
    fs: FiberStatics,
    state0: tuple,
    Ve_seq: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Integrate one fiber; return max m-gate at each compartment over time.

    Parameters
    ----------
    fs : FiberStatics
        Static solver arrays for this fiber (not batched).
    state0 : (M, H, MP, S) each [n_comp]
        Initial gate states.
    Ve_seq : [T, n_comp]
        Extracellular potential sequence (mV), already including all contacts.
    dt : float
        Time step (ms).

    Returns
    -------
    m_max : [n_comp]
        Max m-gate value at each compartment over the simulation.
        Internodes are 0. Used as activation proxy at nodes.
    """
    n = fs.is_node.shape[0]
    T = Ve_seq.shape[0]

    # Pre-shift Ve for backward-Euler (Ve at previous step = 0 before step 0)
    Ve_prev = jnp.concatenate(
        [jnp.zeros((1, n), dtype=jnp.float64), Ve_seq[:-1]], axis=0
    )

    @jax.checkpoint
    def step(carry, s):
        Vi, Vp, st, m_max = carry
        M, H, MP, S = st
        Vm = Vi - Vp

        (a_m, b_m), (a_h, b_h), (a_mp, b_mp), (a_s, b_s) = AxnodeMyel._alpha_beta(
            Vm, _CELSIUS
        )
        M2  = solve_gate_exponential(M,  dt, a_m,  b_m)
        H2  = solve_gate_exponential(H,  dt, a_h,  b_h)
        MP2 = solve_gate_exponential(MP, dt, a_mp, b_mp)
        S2  = solve_gate_exponential(S,  dt, a_s,  b_s)

        g_na   = _GNABAR  * M2 ** 3 * H2
        g_nap  = _GNAPBAR * MP2 ** 3
        g_k    = _GKBAR   * S2
        A_in   = fs.A_in_cm2
        g_pas_us = fs.g_pas * A_in * 1e6
        g_node   = (g_na + g_nap + g_k + _GL) * A_in * 1e6
        i_node   = (
            (g_na + g_nap) * (Vm - _ENA)
            + g_k * (Vm - _EK)
            + _GL * (Vm - _EL)
        ) * A_in * 1e6
        i_pas  = g_pas_us * (Vm - V_REST)
        g_eff  = jnp.where(fs.is_node, g_node, g_pas_us)
        i_ion  = jnp.where(fs.is_node, i_node, i_pas)

        Vi2, Vp2 = _be_step(
            Vi, Vp, Ve_seq[s], Ve_prev[s], g_eff, i_ion,
            fs.Cm_dt, fs.Cmy_dt, fs.gmy,
            fs.Gi_diag, fs.Gp_diag, fs.Up, fs.Low, fs.is_node,
        )
        m_max_new = jnp.maximum(m_max, M2)
        return (Vi2, Vp2, (M2, H2, MP2, S2), m_max_new), None

    Vi0    = jnp.full(n, V_REST, dtype=jnp.float64)
    Vp0    = jnp.zeros(n, dtype=jnp.float64)
    m_max0 = jnp.zeros(n, dtype=jnp.float64)
    (_, _, _, m_max_f), _ = jax.lax.scan(
        step, (Vi0, Vp0, state0, m_max0), jnp.arange(T)
    )
    return m_max_f  # [n_comp]


# ─────────────────────────────────────────── vmapped batch entry point ────────

def batch_integrate_m_max(
    fiber_statics_batch: FiberStatics,
    state0_batch: tuple,
    Ve_seq_batch: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Vmap over fibers: simulate all fibers in parallel and return m_max.

    Parameters
    ----------
    fiber_statics_batch : FiberStatics
        Each field is [n_fibers, n_comp, ...].
    state0_batch : (M, H, MP, S) each [n_fibers, n_comp]
        Initial gate states for all fibers.
    Ve_seq_batch : [n_fibers, T, n_comp]
        Per-fiber extracellular potential sequence.
    dt : float
        Time step (ms).

    Returns
    -------
    m_max_batch : [n_fibers, n_comp]
        Max m-gate at each compartment for each fiber.
    """
    return jax.vmap(
        lambda fs, s0, ve: _integrate_one_fiber_m_max(fs, s0, ve, dt)
    )(fiber_statics_batch, state0_batch, Ve_seq_batch)


# ──────────────────────────────── memory-efficient FD forward pass ────────────

def _integrate_one_fiber_m_max_fd(
    fs: FiberStatics,
    state0: tuple,
    Ve_comb: jnp.ndarray,       # [n_comp]  static spatial profile
    pulse_seq: jnp.ndarray,     # [T]       pulse amplitude at each step
    pulse_prev_seq: jnp.ndarray,# [T]       pulse at previous step (0 at t=0)
    dt: float,
) -> jnp.ndarray:
    """Like _integrate_one_fiber_m_max but avoids the [T, n_comp] Ve_seq tensor.

    Ve at each step is computed on-the-fly as pulse[t] * Ve_comb.
    This lets the FD optimizer pack (K+1)*N_FIBERS effective fibers into a
    single batched forward pass without the T-dimension memory overhead.
    """
    n = fs.is_node.shape[0]
    T = pulse_seq.shape[0]

    # Checkpoint policy selectable per job via JAXLEY_FIBERS_FD_CHECKPOINT.
    #   "1" (default) — wrap step in @jax.checkpoint.  Caps memory at the
    #     forward-only footprint when this scan runs under autodiff
    #     (LBFGS rect path); essential for N_FIBERS≥~100 on A16.
    #   "0" — no checkpoint.  Forces full-trajectory storage during
    #     backward, but eliminates a known XLA pathology where the remat
    #     planner stalls for 25+ min trying to schedule a 5-level-nested
    #     vmap/scan/value_and_grad/scan/checkpoint stack under
    #     optax.zoom_linesearch.  Use for small N_FIBERS that fit without
    #     remat, or for the Adam-FD path (which has no backward and is
    #     insensitive to this flag).
    import os as _os
    _use_checkpoint = _os.environ.get("JAXLEY_FIBERS_FD_CHECKPOINT", "1") != "0"

    def step(carry, s):
        Vi, Vp, st, m_max = carry
        M, H, MP, S = st
        Vm = Vi - Vp
        (a_m, b_m), (a_h, b_h), (a_mp, b_mp), (a_s, b_s) = AxnodeMyel._alpha_beta(Vm, _CELSIUS)
        M2  = solve_gate_exponential(M,  dt, a_m,  b_m)
        H2  = solve_gate_exponential(H,  dt, a_h,  b_h)
        MP2 = solve_gate_exponential(MP, dt, a_mp, b_mp)
        S2  = solve_gate_exponential(S,  dt, a_s,  b_s)
        g_na     = _GNABAR  * M2 ** 3 * H2
        g_nap    = _GNAPBAR * MP2 ** 3
        g_k      = _GKBAR   * S2
        A_in     = fs.A_in_cm2
        g_pas_us = fs.g_pas * A_in * 1e6
        g_node   = (g_na + g_nap + g_k + _GL) * A_in * 1e6
        i_node   = (
            (g_na + g_nap) * (Vm - _ENA)
            + g_k          * (Vm - _EK)
            + _GL          * (Vm - _EL)
        ) * A_in * 1e6
        i_pas    = g_pas_us * (Vm - V_REST)
        g_eff    = jnp.where(fs.is_node, g_node,  g_pas_us)
        i_ion    = jnp.where(fs.is_node, i_node,  i_pas)
        Ve_cur   = pulse_seq[s]      * Ve_comb
        Ve_prev  = pulse_prev_seq[s] * Ve_comb
        Vi2, Vp2 = _be_step(
            Vi, Vp, Ve_cur, Ve_prev, g_eff, i_ion,
            fs.Cm_dt, fs.Cmy_dt, fs.gmy,
            fs.Gi_diag, fs.Gp_diag, fs.Up, fs.Low, fs.is_node,
        )
        return (Vi2, Vp2, (M2, H2, MP2, S2), jnp.maximum(m_max, M2)), None

    if _use_checkpoint:
        step = jax.checkpoint(step)

    Vi0    = jnp.full(n, V_REST, dtype=jnp.float64)
    Vp0    = jnp.zeros(n, dtype=jnp.float64)
    m_max0 = jnp.zeros(n, dtype=jnp.float64)
    (_, _, _, m_max_f), _ = jax.lax.scan(
        step, (Vi0, Vp0, state0, m_max0), jnp.arange(T)
    )
    return m_max_f


def batch_integrate_m_max_fd(
    fiber_statics_batch: FiberStatics,
    state0_batch: tuple,
    Ve_comb_batch: jnp.ndarray,     # [n_fibers_total, n_comp]
    pulse_seq: jnp.ndarray,         # [T]
    pulse_prev_seq: jnp.ndarray,    # [T]
    dt: float,
) -> jnp.ndarray:
    """Memory-efficient vmap for FD gradient computation.

    Takes a static Ve_comb [n_comp] per fiber instead of a full Ve_seq [T, n_comp].
    Ve at each timestep is computed as pulse[t] * Ve_comb inside the scan, so only
    [n_fibers_total, n_comp] memory is needed (not [n_fibers_total, T, n_comp]).

    Designed for use with the packed-FD optimizer: pass (K+1)*N_FIBERS effective
    fibers (one per amplitude config) to compute all K+1 forward passes in parallel.

    Returns
    -------
    m_max_batch : [n_fibers_total, n_comp]
    """
    return jax.vmap(
        lambda fs, s0, vc: _integrate_one_fiber_m_max_fd(
            fs, s0, vc, pulse_seq, pulse_prev_seq, dt
        )
    )(fiber_statics_batch, state0_batch, Ve_comb_batch)
