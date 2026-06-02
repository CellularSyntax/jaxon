"""Optimization loops for spatial selectivity.

Three modes
-----------
run_rect_optimization      : optimize per-contact amplitudes (K DOF).
run_waveform_optimization  : optimize full per-contact waveforms (K×T DOF).
run_joint_optimization     : jointly optimize amplitudes + contact positions.

Gradient strategy
-----------------
Rect and joint use *finite-difference (FD) gradients* packed into a single
batched forward pass — no autodiff backward pass through the ODE scan.

The key trick: instead of K+1 sequential forward calls, we treat each
amplitude/position config as an extra "fiber" and run one big
batch_integrate_m_max_fd call with (K+1)*N_FIBERS effective fibers.
Ve at each step is computed as pulse[t] * Ve_comb inside the scan so
only [n_fibers_total, n_comp] memory is needed (not T times that).

For K=6, N_FIBERS=1000:
  Rect : 7 configs  →  7 000 effective fibers — one forward pass
  Joint: 25 configs → 25 000 effective fibers — one forward pass

Waveform optimization keeps autodiff (the only feasible option for K×T DOF).
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxfibers.stim.batch_solve import (
    FiberStatics,
    batch_integrate_m_max,
    batch_integrate_m_max_fd,
)
from jaxfibers.stim.multichannel_field import compute_ve_unit_jax
from jaxfibers.optim.losses import activation_proxy_batch, wq_loss, wbce, selectivity_index


# ──────────────────────────────────────────────── statics tiling helpers ──────

def _tile_fiber_statics(fs: FiberStatics, n_tiles: int) -> FiberStatics:
    """Repeat FiberStatics n_tiles times along the fiber (first) axis."""
    return FiberStatics(*[
        jnp.tile(field, (n_tiles,) + (1,) * (field.ndim - 1))
        for field in fs
    ])


def _tile_states(s0: tuple, n_tiles: int) -> tuple:
    """Repeat initial gate states n_tiles times along the fiber (first) axis."""
    return tuple(
        jnp.tile(s, (n_tiles,) + (1,) * (s.ndim - 1)) for s in s0
    )


# ─────────────────────────────────────────── rectangular pulse optimization ──

def run_rect_optimization(
    fiber_statics_batch: FiberStatics,
    state0_batch: tuple,
    Ve_unit: jnp.ndarray,              # [K, n_fibers, n_comp]
    pulse_mask: jnp.ndarray,           # [T]
    node_indices: np.ndarray,          # [n_fibers, n_nodes]
    target_mask: np.ndarray,           # [n_fibers] bool
    dt: float,
    n_steps: int = 100,
    lr: float = 8e-2,
    amp_init_mA: float = -0.4,
    amp_clip: tuple[float, float] = (-2.5, 2.5),
    weights: np.ndarray | None = None,
    fd_eps: float = 5e-2,
    verbose: bool = True,
) -> dict:
    """Optimize per-contact rectangular pulse amplitudes via FD gradient.

    Each optimization step runs ONE batched forward pass over (K+1)*N_FIBERS
    effective fibers: one base config + K perturbed configs (amps[k] += fd_eps).

    Returns
    -------
    result : dict with keys
        'amps'    : [K] best amplitudes (mA)
        'history' : dict of lists {'loss', 'bce', 'si', 'amps', 'acts'}
    """
    K        = Ve_unit.shape[0]
    n_fibers = Ve_unit.shape[1]
    n_comp   = Ve_unit.shape[2]
    n_configs = K + 1

    Ve_unit_j    = jnp.asarray(Ve_unit,      dtype=jnp.float64)
    pulse_j      = jnp.asarray(pulse_mask,   dtype=jnp.float64)
    pulse_prev_j = jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), pulse_j[:-1]])
    node_idx_j   = jnp.asarray(node_indices, dtype=jnp.int32)
    tgt_j        = jnp.asarray(target_mask,  dtype=jnp.float64)
    w = (jnp.ones(n_fibers, dtype=jnp.float64) / n_fibers
         if weights is None else jnp.asarray(weights, dtype=jnp.float64))

    # Tile statics once outside JIT — constant across all iterations.
    fs_tiled = _tile_fiber_statics(fiber_statics_batch, n_configs)
    s0_tiled = _tile_states(state0_batch, n_configs)

    @jax.jit
    def _fd_step(amps):
        # Base Ve_comb [n_fibers, n_comp].
        Ve_comb_base = jnp.einsum("k,kfn->fn", -amps, Ve_unit_j)

        # Perturbed configs: amps[k] += fd_eps  →  Ve_comb[k] -= fd_eps * Ve_unit[k].
        Ve_comb_pert = Ve_comb_base[None] - fd_eps * Ve_unit_j  # [K, n_f, n_c]

        # All K+1 configs stacked: [K+1, n_f, n_c].
        Ve_comb_all  = jnp.concatenate([Ve_comb_base[None], Ve_comb_pert], axis=0)

        # Flatten for batch solver: [(K+1)*n_fibers, n_comp].
        Ve_comb_flat = Ve_comb_all.reshape(n_configs * n_fibers, n_comp)

        m_max_flat = batch_integrate_m_max_fd(
            fs_tiled, s0_tiled, Ve_comb_flat, pulse_j, pulse_prev_j, dt
        )
        m_max_all = m_max_flat.reshape(n_configs, n_fibers, n_comp)

        # Activation proxy and loss for every config.
        acts_all   = jax.vmap(lambda m: activation_proxy_batch(m, node_idx_j))(m_max_all)
        losses_all = jax.vmap(lambda a: wq_loss(a, tgt_j, w))(acts_all)

        loss_base = losses_all[0]
        acts_base = acts_all[0]
        grad      = (losses_all[1:] - loss_base) / fd_eps  # [K]
        return (loss_base, acts_base), grad

    # Initialize amplitudes proportional to each contact's mean |Ve| at target
    # fibers.  This immediately breaks ring symmetry for eccentric fascicles:
    # the contact closest to the target fascicle gets amp_init_mA, the contact
    # farthest away gets ~0 (the neutral "off" baseline).  Uniform-magnitude
    # init across all contacts otherwise.
    #
    # The "off" anchor is 0, NOT amp_clip[1].  An earlier formula used
    # amp_clip[1] as the anchor, which only worked for cathodic-only clips
    # like (-5, -0.05): with a symmetric clip like (-2.5, +2.5) it would
    # pin the farthest contact at the positive (anodic) clip — actively
    # firing the off-target fibers from the wrong side.  Anchoring at 0
    # is clip-invariant.
    ve_tgt_sum  = jnp.where(tgt_j[None, :, None], jnp.abs(Ve_unit_j), 0.0).sum((1, 2))
    ve_tgt_mean = ve_tgt_sum / jnp.maximum(tgt_j.sum(), 1.0)   # [K]
    ve_range    = ve_tgt_mean.max() - ve_tgt_mean.min()
    ve_norm     = (ve_tgt_mean - ve_tgt_mean.min()) / (ve_range + 1e-10)   # [K] in [0,1]
    amps0 = jnp.clip(
        ve_norm * amp_init_mA,
        amp_clip[0], amp_clip[1],
    ).astype(jnp.float64)

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = optimizer.init(amps0)
    amps      = amps0
    history   = {"loss": [], "bce": [], "si": [], "amps": [], "acts": []}
    best_loss = float("inf")
    best_amps = np.array(amps0)

    if verbose:
        amp_init_str = "  ".join(f"{a:+.2f}" for a in np.array(amps0))
        print(
            f"  Rect opt (FD): K={K} contacts, "
            f"{n_configs} configs × {n_fibers} fibers = "
            f"{n_configs * n_fibers} effective fibers per pass, "
            f"{n_steps} iters",
            flush=True,
        )
        print(f"  init amps=[{amp_init_str}] mA", flush=True)
        print("  [iter 0] XLA compile — first call only ...", flush=True)

    for i in range(n_steps):
        t0 = time.time()
        (loss_val, acts_val), grads = _fd_step(amps)
        updates, opt_state = optimizer.update(grads, opt_state)
        amps = jnp.clip(optax.apply_updates(amps, updates), amp_clip[0], amp_clip[1])

        acts_np = np.array(acts_val)
        history["loss"].append(float(loss_val))
        history["bce"].append(float(wbce(acts_val, tgt_j, w)))
        history["si"].append(selectivity_index(acts_np, target_mask))
        history["amps"].append(np.array(amps))
        history["acts"].append(acts_np)
        if float(loss_val) < best_loss:
            best_loss = float(loss_val)
            best_amps = np.array(amps)

        if verbose and (i % max(1, n_steps // 20) == 0 or i == n_steps - 1):
            dt_ms   = (time.time() - t0) * 1000
            amp_str = "  ".join(f"{a:+.2f}" for a in np.array(amps))
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
    Ve_unit: jnp.ndarray,              # [K, n_fibers, n_comp]
    node_indices: np.ndarray,          # [n_fibers, n_nodes]
    target_mask: np.ndarray,           # [n_fibers] bool
    dt: float,
    T: int,
    n_steps: int = 100,
    lr: float = 5e-3,
    u_init: np.ndarray | None = None,  # [K, T]
    u_clip: tuple[float, float] = (-5.0, 5.0),
    weights: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """Optimize arbitrary per-contact waveforms u[K, T] via autodiff.

    NOTE: K×T degrees of freedom (e.g. 6×1600 = 9600) make FD impractical.
    Autodiff backward through the ODE scan is slow (~100s/iter for T=1600).
    Reduce T_STOP / DT or switch to a surrogate for large-scale sweeps.

    Returns
    -------
    result : dict with keys
        'u'       : [K, T] best waveforms (mA)
        'history' : dict of lists {'loss', 'bce', 'si', 'acts'}
    """
    K        = Ve_unit.shape[0]
    n_fibers = Ve_unit.shape[1]

    Ve_unit_j  = jnp.asarray(Ve_unit,      dtype=jnp.float64)
    node_idx_j = jnp.asarray(node_indices, dtype=jnp.int32)
    tgt_j      = jnp.asarray(target_mask,  dtype=jnp.float64)
    w = (jnp.ones(n_fibers, dtype=jnp.float64) / n_fibers
         if weights is None else jnp.asarray(weights, dtype=jnp.float64))

    u0 = (jnp.zeros((K, T), dtype=jnp.float64)
          if u_init is None else jnp.asarray(u_init, dtype=jnp.float64))

    def loss_fn(u):
        Ve_seq_batch = jnp.einsum("kt,kfn->ftn", -u, Ve_unit_j)   # [n_f, T, n_c]
        m_max_batch  = batch_integrate_m_max(
            fiber_statics_batch, state0_batch, Ve_seq_batch, dt
        )
        acts = activation_proxy_batch(m_max_batch, node_idx_j)
        return wq_loss(acts, tgt_j, w), acts

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = optimizer.init(u0)
    u         = u0
    history   = {"loss": [], "bce": [], "si": [], "acts": []}
    best_loss = float("inf")
    best_u    = np.array(u0)

    if verbose:
        print(
            f"  Waveform opt (autodiff): K={K} contacts, T={T} steps, {n_steps} iters",
            flush=True,
        )
        print("  [iter 0] XLA compile — first call only ...", flush=True)

    for i in range(n_steps):
        t0 = time.time()
        (loss_val, acts_val), grads = loss_and_grad(u)
        updates, opt_state = optimizer.update(grads, opt_state)
        u = jnp.clip(optax.apply_updates(u, updates), u_clip[0], u_clip[1])

        acts_np = np.array(acts_val)
        history["loss"].append(float(loss_val))
        history["bce"].append(float(wbce(acts_val, tgt_j, w)))
        history["si"].append(selectivity_index(acts_np, target_mask))
        history["acts"].append(acts_np)
        if float(loss_val) < best_loss:
            best_loss = float(loss_val)
            best_u    = np.array(u)

        if verbose and (i % max(1, n_steps // 20) == 0 or i == n_steps - 1):
            dt_ms = (time.time() - t0) * 1000
            u_np  = np.array(u)
            print(
                f"  [{i:3d}/{n_steps}] loss={float(loss_val):.4f}  "
                f"SI={history['si'][-1]:+.3f}  "
                f"rms={float(np.sqrt(np.mean(u_np**2))):.3f} mA  "
                f"peak={float(np.abs(u_np).max()):.3f} mA  dt={dt_ms:.0f}ms",
                flush=True,
            )

    return {"u": best_u, "history": history}


# ─────────────────────────────── joint amplitude + electrode position opt ──────

def run_joint_optimization(
    fiber_statics_batch: FiberStatics,
    state0_batch: tuple,
    fiber_xy_um: jnp.ndarray,          # [n_fibers, 2]
    all_centers_um: jnp.ndarray,       # [n_fibers, n_comp]
    pulse_mask: jnp.ndarray,           # [T]
    node_indices: np.ndarray,          # [n_fibers, n_nodes]
    target_mask: np.ndarray,           # [n_fibers] bool
    dt: float,
    contact_xyz_init: np.ndarray,      # [K, 3]
    amps_init: np.ndarray,             # [K]
    n_steps: int = 100,
    lr_amp: float = 8e-2,
    lr_pos: float = 10.0,
    amp_clip: tuple[float, float] = (-2.5, 2.5),
    xyz_min: float = -3000.0,
    xyz_max: float = 3000.0,
    weights: np.ndarray | None = None,
    fd_eps_amp: float = 5e-2,
    fd_eps_pos: float = 20.0,
    verbose: bool = True,
    sigma_S_m: float = 0.3,
) -> dict:
    """Jointly optimize per-contact amplitudes and electrode positions via FD.

    Each step runs ONE batched forward pass over (4K+1)*N_FIBERS effective fibers:
      1 base + K amplitude perturbations + 3K position perturbations.

    For K=6 this is 25 configs. The 3K position perturbations are built by
    vmapping compute_ve_unit_jax over the K×3 unit perturbation directions —
    all field computations run in parallel before the single fiber simulation.

    Returns
    -------
    result : dict with keys
        'amps'           : [K] best amplitudes (mA)
        'contact_xyz_um' : [K, 3] best contact positions (µm)
        'history'        : dict of lists {'loss', 'si', 'amps', 'xyz'}
    """
    K        = contact_xyz_init.shape[0]
    n_fibers = fiber_xy_um.shape[0]
    n_comp   = fiber_statics_batch.Cm_dt.shape[1]
    n_configs = 4 * K + 1   # 1 base + K amp pert + 3K pos pert

    fiber_xy_j    = jnp.asarray(fiber_xy_um,    dtype=jnp.float64)
    all_centers_j = jnp.asarray(all_centers_um, dtype=jnp.float64)
    pulse_j       = jnp.asarray(pulse_mask,     dtype=jnp.float64)
    pulse_prev_j  = jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), pulse_j[:-1]])
    node_idx_j    = jnp.asarray(node_indices,   dtype=jnp.int32)
    tgt_j         = jnp.asarray(target_mask,    dtype=jnp.float64)
    w = (jnp.ones(n_fibers, dtype=jnp.float64) / n_fibers
         if weights is None else jnp.asarray(weights, dtype=jnp.float64))

    # Tile statics once — 4K+1 copies of each fiber's static arrays.
    fs_tiled = _tile_fiber_statics(fiber_statics_batch, n_configs)
    s0_tiled = _tile_states(state0_batch, n_configs)

    # Unit perturbation directions for positions: [3K, K, 3] each scaled by eps_pos.
    # Row k*3+d perturbs contact k in dimension d.
    I_KD = (jnp.eye(K * 3, dtype=jnp.float64)
            .reshape(K * 3, K, 3) * fd_eps_pos)  # [3K, K, 3]

    @jax.jit
    def _fd_step_joint(amps, contact_xyz):
        # ── base field ────────────────────────────────────────────────────────
        Ve_unit_base = compute_ve_unit_jax(
            fiber_xy_j, all_centers_j, contact_xyz, sigma_S_m
        )                                                       # [K, n_f, n_c]
        Ve_comb_base = jnp.einsum("k,kfn->fn", -amps, Ve_unit_base)  # [n_f, n_c]

        # ── amplitude perturbations ───────────────────────────────────────────
        # amps[k] += fd_eps_amp  →  Ve_comb -= fd_eps_amp * Ve_unit_base[k]
        Ve_comb_amp_pert = Ve_comb_base[None] - fd_eps_amp * Ve_unit_base  # [K, n_f, n_c]

        # ── position perturbations ────────────────────────────────────────────
        # Vmap over 3K unit directions — all contact field computations in parallel.
        def _pos_pert_ve_comb(xyz_delta):
            Ve_u = compute_ve_unit_jax(
                fiber_xy_j, all_centers_j, contact_xyz + xyz_delta, sigma_S_m
            )
            return jnp.einsum("k,kfn->fn", -amps, Ve_u)        # [n_f, n_c]

        Ve_comb_pos_pert = jax.vmap(_pos_pert_ve_comb)(I_KD)   # [3K, n_f, n_c]

        # ── pack all 4K+1 configs ─────────────────────────────────────────────
        Ve_comb_all  = jnp.concatenate([
            Ve_comb_base[None],
            Ve_comb_amp_pert,
            Ve_comb_pos_pert,
        ], axis=0)                                              # [4K+1, n_f, n_c]

        Ve_comb_flat = Ve_comb_all.reshape(n_configs * n_fibers, n_comp)

        m_max_flat = batch_integrate_m_max_fd(
            fs_tiled, s0_tiled, Ve_comb_flat, pulse_j, pulse_prev_j, dt
        )
        m_max_all = m_max_flat.reshape(n_configs, n_fibers, n_comp)

        acts_all   = jax.vmap(lambda m: activation_proxy_batch(m, node_idx_j))(m_max_all)
        losses_all = jax.vmap(lambda a: wq_loss(a, tgt_j, w))(acts_all)

        loss_base = losses_all[0]
        acts_base = acts_all[0]

        # FD gradients.
        g_amp = (losses_all[1:K + 1] - loss_base) / fd_eps_amp        # [K]
        g_pos = ((losses_all[K + 1:] - loss_base) / fd_eps_pos
                 ).reshape(K, 3)                                       # [K, 3]

        return (loss_base, acts_base), (g_amp, g_pos)

    amps0 = jnp.asarray(amps_init,        dtype=jnp.float64)
    xyz0  = jnp.asarray(contact_xyz_init, dtype=jnp.float64)

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
        print(
            f"  Joint opt (FD): K={K} contacts, "
            f"{n_configs} configs × {n_fibers} fibers = "
            f"{n_configs * n_fibers} effective fibers per pass, "
            f"{n_steps} iters",
            flush=True,
        )
        print("  [iter 0] XLA compile — first call only ...", flush=True)

    for i in range(n_steps):
        t0 = time.time()
        (loss_val, acts_val), (g_amp, g_xyz) = _fd_step_joint(amps, xyz)
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
            dt_ms   = (time.time() - t0) * 1000
            xyz_np  = np.array(xyz)
            disp    = np.linalg.norm(xyz_np - contact_xyz_init, axis=1)
            amp_str = "  ".join(f"{a:+.2f}" for a in np.array(amps))
            print(
                f"  [{i:3d}/{n_steps}] loss={float(loss_val):.4f}  SI={si:+.3f}  "
                f"disp mean={disp.mean():.1f} max={disp.max():.1f} µm  "
                f"amps=[{amp_str}] mA  dt={dt_ms:.0f}ms",
                flush=True,
            )

    return {"amps": best_amps, "contact_xyz_um": best_xyz, "history": history}
