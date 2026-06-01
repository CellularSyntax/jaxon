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
from jaxfibers.stim.multichannel_field import compute_ve_unit_jax
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
        print(f"  Rect optimisation: K={K} contacts, T={T} steps, {n_steps} iters", flush=True)
        print(f"  [iter 0] XLA compile — may take 10-30 min on first run ...", flush=True)

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

        if verbose and (i % max(1, n_steps // 20) == 0 or i == n_steps - 1):
            dt_ms = (time.time() - t0) * 1000
            amps_np = np.array(amps)
            amp_str = "  ".join(f"{a:+.2f}" for a in amps_np)
            print(
                f"  [{i:3d}/{n_steps}] loss={float(loss_val):.4f}  "
                f"SI={history['si'][-1]:+.3f}  "
                f"amps=[{amp_str}] mA  dt={dt_ms:.0f}ms",
                flush=True,
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
        print(f"  Waveform optimisation: K={K} contacts, T={T} timesteps, {n_steps} iters", flush=True)
        print(f"  [iter 0] XLA compile — may take 10-30 min on first run ...", flush=True)

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

        if verbose and (i % max(1, n_steps // 20) == 0 or i == n_steps - 1):
            dt_ms = (time.time() - t0) * 1000
            u_np = np.array(u)
            rms = float(np.sqrt(np.mean(u_np ** 2)))
            peak = float(np.abs(u_np).max())
            print(
                f"  [{i:3d}/{n_steps}] loss={float(loss_val):.4f}  "
                f"SI={history['si'][-1]:+.3f}  "
                f"rms={rms:.3f} mA  peak={peak:.3f} mA  dt={dt_ms:.0f}ms",
                flush=True,
            )

    return {"u": best_u, "history": history}


# ─────────────────────────────────── joint waveform + electrode position opt ──

def run_joint_optimization(
    fiber_statics_batch: FiberStatics,
    state0_batch: tuple,
    fiber_xy_um: jnp.ndarray,           # [n_fibers, 2]
    all_centers_um: jnp.ndarray,        # [n_fibers, n_comp]
    pulse_mask: jnp.ndarray,            # [T]
    node_indices: np.ndarray,           # [n_fibers, n_nodes]
    target_mask: np.ndarray,            # [n_fibers] bool
    dt: float,
    contact_xyz_init: np.ndarray,       # [K, 3] initial contact positions (µm)
    amps_init: np.ndarray,              # [K] initial amplitudes (mA)
    n_steps: int = 100,
    lr_amp: float = 5e-2,
    lr_pos: float = 10.0,              # µm / step
    amp_clip: tuple[float, float] = (-5.0, 0.0),
    xyz_min: float = -3000.0,          # µm
    xyz_max: float = 3000.0,           # µm
    weights: np.ndarray | None = None,
    verbose: bool = True,
    sigma_S_m: float = 0.3,
) -> dict:
    """Jointly optimize per-contact amplitudes and electrode positions.

    Optimization variables:
        amps           [K]    — per-contact amplitudes (mA)
        contact_xyz_um [K, 3] — contact positions (µm)

    Uses two independent Adam optimizers (separate learning rates for
    amplitude and position). Returns best checkpoint over all iterations.

    Returns
    -------
    result : dict with keys
        'amps'           : [K] best amplitudes (mA)
        'contact_xyz_um' : [K, 3] best contact positions (µm)
        'history'        : dict of lists {'loss', 'si', 'amps', 'xyz'}
    """
    K = contact_xyz_init.shape[0]
    n_fibers = fiber_xy_um.shape[0]

    fiber_xy_j    = jnp.asarray(fiber_xy_um,   dtype=jnp.float64)
    all_centers_j = jnp.asarray(all_centers_um, dtype=jnp.float64)
    pulse_j       = jnp.asarray(pulse_mask,     dtype=jnp.float64)
    node_idx_j    = jnp.asarray(node_indices,   dtype=jnp.int32)
    w = (jnp.ones(n_fibers, dtype=jnp.float64) / n_fibers
         if weights is None else jnp.asarray(weights, dtype=jnp.float64))

    amps0 = jnp.asarray(amps_init,       dtype=jnp.float64)
    xyz0  = jnp.asarray(contact_xyz_init, dtype=jnp.float64)

    def loss_fn(params):
        amps, contact_xyz = params
        Ve_unit   = compute_ve_unit_jax(fiber_xy_j, all_centers_j, contact_xyz, sigma_S_m)
        Ve_comb   = jnp.einsum("k,kfn->fn", -amps, Ve_unit)
        Ve_seq    = pulse_j[:, None, None] * Ve_comb[None, :, :]
        Ve_seq    = jnp.transpose(Ve_seq, (1, 0, 2))       # [n_fibers, T, n_comp]
        m_max     = batch_integrate_m_max(fiber_statics_batch, state0_batch, Ve_seq, dt)
        acts      = activation_proxy_batch(m_max, node_idx_j)
        loss      = wq_loss(acts, target_mask, w)
        return loss, acts

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    opt_amp = optax.chain(optax.clip_by_global_norm(1.0),   optax.adam(lr_amp))
    opt_pos = optax.chain(optax.clip_by_global_norm(100.0), optax.adam(lr_pos))
    st_amp  = opt_amp.init(amps0)
    st_pos  = opt_pos.init(xyz0)
    amps, xyz = amps0, xyz0

    history   = {"loss": [], "si": [], "amps": [], "xyz": []}
    best_loss = float("inf")
    best_amps = np.array(amps0)
    best_xyz  = np.array(contact_xyz_init)

    if verbose:
        print(f"  Joint optimisation: K={K} contacts, {n_steps} iters", flush=True)
        print(f"  [iter 0] XLA compile — may take 10-30 min on first run ...", flush=True)

    for i in range(n_steps):
        t0 = time.time()
        (loss_val, acts_val), (g_amp, g_xyz) = loss_and_grad((amps, xyz))
        upd_amp, st_amp = opt_amp.update(g_amp, st_amp)
        upd_pos, st_pos = opt_pos.update(g_xyz, st_pos)
        amps = jnp.clip(optax.apply_updates(amps, upd_amp), amp_clip[0], amp_clip[1])
        xyz  = jnp.clip(optax.apply_updates(xyz,  upd_pos), xyz_min, xyz_max)

        si = selectivity_index(np.array(acts_val), target_mask)
        history["loss"].append(float(loss_val))
        history["si"].append(si)
        history["amps"].append(np.array(amps))
        history["xyz"].append(np.array(xyz))
        if float(loss_val) < best_loss:
            best_loss = float(loss_val)
            best_amps = np.array(amps)
            best_xyz  = np.array(xyz)

        if verbose and (i % max(1, n_steps // 20) == 0 or i == n_steps - 1):
            dt_ms = (time.time() - t0) * 1000
            xyz_np = np.array(xyz)
            disp   = np.linalg.norm(xyz_np - contact_xyz_init, axis=1)
            amps_np = np.array(amps)
            amp_str = "  ".join(f"{a:+.2f}" for a in amps_np)
            print(
                f"  [{i:3d}/{n_steps}] loss={float(loss_val):.4f}  SI={si:+.3f}  "
                f"disp mean={disp.mean():.1f} max={disp.max():.1f} µm  "
                f"amps=[{amp_str}] mA  dt={dt_ms:.0f}ms",
                flush=True,
            )

    return {"amps": best_amps, "contact_xyz_um": best_xyz, "history": history}
