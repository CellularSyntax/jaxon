"""Optimization loops for spatial selectivity.

Two modes
---------
run_rect_optimization  : optimize per-contact amplitudes (K DOF).
run_waveform_optimization : optimize full per-contact waveforms (K×T DOF).

Both use Adam (optax) and return a history dict.
"""
from __future__ import annotations

import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxfibers.stim.batch_solve import FiberStatics, batch_integrate_m_max
from jaxfibers.optim.losses import activation_proxy_batch, wq_loss, wbce, selectivity_index


# ─────────────────────────────────────────── rectangular pulse optimization ──

def run_rect_optimization(
    fiber_statics_batch: FiberStatics,
    state0_batch: tuple,
    Ve_unit: jnp.ndarray,             # [K, n_fibers, n_comp]
    pulse_mask: jnp.ndarray,          # [T] shared monophasic pulse shape (0/1)
    node_indices: np.ndarray,         # [n_fibers, n_nodes]
    target_mask: np.ndarray,          # [n_fibers] bool
    dt: float,
    n_steps: int = 100,
    lr: float = 5e-2,
    amp_init_mA: float = -0.3,
    amp_clip: tuple[float, float] = (-5.0, 0.0),
    weights: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """Optimize per-contact rectangular pulse amplitudes.

    All contacts share the same pulse timing (pulse_mask); only amplitudes differ.
    Optimization variable: amps [K] (mA, cathodic = negative).

    Returns
    -------
    result : dict with keys
        'amps'    : [K] final amplitudes (mA)
        'history' : dict of lists {'loss', 'bce', 'si', 'amps', 'acts'}
    """
    K = Ve_unit.shape[0]
    n_fibers = Ve_unit.shape[1]
    T = pulse_mask.shape[0]

    Ve_unit_j = jnp.asarray(Ve_unit, dtype=jnp.float64)
    pulse_j   = jnp.asarray(pulse_mask, dtype=jnp.float64)
    node_idx_j = jnp.asarray(node_indices, dtype=jnp.int32)
    w = jnp.ones(n_fibers, dtype=jnp.float64) / n_fibers if weights is None else jnp.asarray(weights, dtype=jnp.float64)

    amps0 = jnp.full(K, amp_init_mA, dtype=jnp.float64)

    def loss_fn(amps):
        # Sign convention: amps < 0 = cathodic (matching the validation scripts).
        # Ve_unit[k,f,n] < 0 near contact k (i0_mA=-1 cathodic reference).
        # Multiply by -amps so negative amps → negative Ve (depolarising).
        Ve_comb = jnp.einsum("k,kfn->fn", -amps, Ve_unit_j)          # [n_fibers, n_comp]
        # Ve_seq_batch[f, t, n] = pulse[t] * Ve_comb[f, n]
        Ve_seq_batch = pulse_j[:, None, None] * Ve_comb[None, :, :]  # [T, n_fibers, n_comp]
        Ve_seq_batch = jnp.transpose(Ve_seq_batch, (1, 0, 2))        # [n_fibers, T, n_comp]

        m_max_batch = batch_integrate_m_max(
            fiber_statics_batch, state0_batch, Ve_seq_batch, dt
        )
        acts = activation_proxy_batch(m_max_batch, node_idx_j)
        loss = wq_loss(acts, target_mask, w)
        return loss, acts

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    # Adam + gradient clipping to prevent threshold overshoot.
    optimizer  = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state  = optimizer.init(amps0)
    amps       = amps0
    history    = {"loss": [], "bce": [], "si": [], "amps": [], "acts": []}
    best_loss  = float("inf")
    best_amps  = np.array(amps0)

    if verbose:
        print(f"  Rect optimisation: K={K} contacts, T={T} steps, {n_steps} iters")

    for i in range(n_steps):
        t0 = time.time()
        (loss_val, acts_val), grads = loss_and_grad(amps)
        updates, opt_state = optimizer.update(grads, opt_state)
        amps = optax.apply_updates(amps, updates)
        amps = jnp.clip(amps, amp_clip[0], amp_clip[1])

        acts_np = np.array(acts_val)
        history["loss"].append(float(loss_val))
        history["bce"].append(float(wbce(acts_val, target_mask, w)))
        history["si"].append(selectivity_index(acts_np, target_mask))
        history["amps"].append(np.array(amps))
        history["acts"].append(acts_np)
        if float(loss_val) < best_loss:
            best_loss = float(loss_val)
            best_amps = np.array(amps)

        if verbose and (i % max(1, n_steps // 10) == 0 or i == n_steps - 1):
            dt_ms = (time.time() - t0) * 1000
            print(
                f"  [{i:3d}/{n_steps}] loss={float(loss_val):.4f}  "
                f"SI={history['si'][-1]:+.3f}  dt={dt_ms:.0f}ms"
            )

    return {"amps": best_amps, "history": history}


# ─────────────────────────────────────────── arbitrary waveform optimization ─

def run_waveform_optimization(
    fiber_statics_batch: FiberStatics,
    state0_batch: tuple,
    Ve_unit: jnp.ndarray,             # [K, n_fibers, n_comp]
    node_indices: np.ndarray,         # [n_fibers, n_nodes]
    target_mask: np.ndarray,          # [n_fibers] bool
    dt: float,
    T: int,
    n_steps: int = 100,
    lr: float = 5e-3,
    u_init: np.ndarray | None = None,  # [K, T] initial waveforms; None = zeros
    u_clip: tuple[float, float] = (-5.0, 5.0),
    weights: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """Optimize arbitrary per-contact waveforms u[K, T].

    Optimization variable: u [K, T] (mA, signed).

    Returns
    -------
    result : dict with keys
        'u'       : [K, T] final waveforms (mA)
        'history' : dict of lists {'loss', 'bce', 'si', 'acts'}
    """
    K = Ve_unit.shape[0]
    n_fibers = Ve_unit.shape[1]

    Ve_unit_j  = jnp.asarray(Ve_unit, dtype=jnp.float64)
    node_idx_j = jnp.asarray(node_indices, dtype=jnp.int32)
    w = jnp.ones(n_fibers, dtype=jnp.float64) / n_fibers if weights is None else jnp.asarray(weights, dtype=jnp.float64)

    if u_init is None:
        u0 = jnp.zeros((K, T), dtype=jnp.float64)
    else:
        u0 = jnp.asarray(u_init, dtype=jnp.float64)

    def loss_fn(u):
        # Same sign convention as rect: u < 0 cathodic → negate so Ve < 0 = depolarising.
        Ve_seq_batch = jnp.einsum("kt,kfn->ftn", -u, Ve_unit_j)  # [n_fibers, T, n_comp]
        m_max_batch = batch_integrate_m_max(
            fiber_statics_batch, state0_batch, Ve_seq_batch, dt
        )
        acts = activation_proxy_batch(m_max_batch, node_idx_j)
        loss = wq_loss(acts, target_mask, w)
        return loss, acts

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = optimizer.init(u0)
    u = u0
    history = {"loss": [], "bce": [], "si": [], "acts": []}
    best_loss = float("inf")
    best_u    = np.array(u0)

    if verbose:
        print(f"  Waveform optimisation: K={K} contacts, T={T} timesteps, {n_steps} iters")

    for i in range(n_steps):
        t0 = time.time()
        (loss_val, acts_val), grads = loss_and_grad(u)
        updates, opt_state = optimizer.update(grads, opt_state)
        u = optax.apply_updates(u, updates)
        u = jnp.clip(u, u_clip[0], u_clip[1])

        acts_np = np.array(acts_val)
        history["loss"].append(float(loss_val))
        history["bce"].append(float(wbce(acts_val, target_mask, w)))
        history["si"].append(selectivity_index(acts_np, target_mask))
        history["acts"].append(acts_np)
        if float(loss_val) < best_loss:
            best_loss = float(loss_val)
            best_u = np.array(u)

        if verbose and (i % max(1, n_steps // 10) == 0 or i == n_steps - 1):
            dt_ms = (time.time() - t0) * 1000
            print(
                f"  [{i:3d}/{n_steps}] loss={float(loss_val):.4f}  "
                f"SI={history['si'][-1]:+.3f}  dt={dt_ms:.0f}ms"
            )

    return {"u": best_u, "history": history}
