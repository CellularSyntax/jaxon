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
from jaxfibers.optim.losses import (
    activation_proxy_batch as _activation_proxy_batch_impl,
    wq_loss as _wq_loss_impl,
    wbce, selectivity_index,
)
import os as _os

# JAXLEY_FIBERS_SOFT_TEMPERATURE: sigmoid-smooth the per-fibre activation
# proxy.  See losses.activation_proxy_batch docstring.  Default 0 = binary.
_SOFT_T = float(_os.environ.get("JAXLEY_FIBERS_SOFT_TEMPERATURE", "0.0"))

# JAXLEY_FIBERS_ENERGY_LAMBDA: amplitude-energy regularisation strength.
# Adds `lambda * mean(amps²)` to the selectivity loss.  Required for the
# mixed-diameter type-selectivity problem where the unregularised optimiser
# falls into the trivial "crank everything to ±3 mA so all fibres fire"
# equilibrium.  Default 0 = no regularisation (single-diameter runs).
# Typical values: 1e-3 to 1e-2.
_ENERGY_LAMBDA = float(_os.environ.get("JAXLEY_FIBERS_ENERGY_LAMBDA", "0.0"))


def activation_proxy_batch(m_max, node_idx):
    return _activation_proxy_batch_impl(m_max, node_idx, soft_temperature=_SOFT_T)


def wq_loss(acts, target_mask, weights=None, amps=None):
    return _wq_loss_impl(
        acts, target_mask, weights=weights,
        amps=amps, energy_lambda=_ENERGY_LAMBDA,
    )


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
    amps_init_vector: np.ndarray | None = None,
    amp_clip: tuple[float, float] = (-2.5, 2.5),
    weights: np.ndarray | None = None,
    fd_eps: float = 5e-2,
    verbose: bool = True,
    # Early-stopping knobs.  Adam-FD often reaches SI=1.0 within the first
    # ~10 iters on the easy single-diameter problem, then runs another 90
    # iters with no improvement — waste of cluster time.
    early_stop_si:        float = 1.0,   # stop as soon as SI >= this value
    early_stop_patience:  int   = 20,    # stop if best_loss hasn't moved for this many iters
    early_stop_loss_tol:  float = 1e-5,  # "moved" means improved by > this
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

    # Boolean target mask + totals for the per-iter firing summary that
    # shows up in the verbose log line (oversaturated / undersaturated /
    # gradient-informative regime is otherwise invisible mid-run).
    target_mask_bool = np.asarray(target_mask, dtype=bool)
    n_target_total    = int(target_mask_bool.sum())
    n_nontarget_total = int((~target_mask_bool).sum())

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

        # Per-config amplitudes for the energy-regularisation term:
        # base = amps, perturbed config k = amps + fd_eps · e_k.
        amps_pert  = amps[None, :] + fd_eps * jnp.eye(K, dtype=amps.dtype)
        amps_all   = jnp.concatenate([amps[None, :], amps_pert], axis=0)
        losses_all = jax.vmap(
            lambda a, ak: wq_loss(a, tgt_j, w, amps=ak)
        )(acts_all, amps_all)

        loss_base = losses_all[0]
        acts_base = acts_all[0]
        grad      = (losses_all[1:] - loss_base) / fd_eps  # [K]
        return (loss_base, acts_base), grad

    # Bipolar Ve-weighted initialisation (guard-pattern init).
    #
    # ve_norm[k] ∈ [0, 1] measures how close contact k is to the target
    # fascicle: 1 = closest, 0 = farthest.  We map
    #
    #     amps0[k] = (2*ve_norm[k] − 1) * |amp_init_mA|
    #
    # so the closest contact gets amp_init_mA (cathodic) and the farthest
    # contact gets −amp_init_mA (anodic), with smooth interpolation in
    # between.  This is the guard-electrode pattern that the geometry
    # sweep at C:/tmp/dbg_geom_sweep.py showed achieves SI = +1.000 at the
    # current 1.5 mm cuff radius on the selectivity_demo nerve.
    #
    # Earlier formulations were one-sided ("ve_norm * amp_init_mA" → all
    # cathodic, or "amp_clip[1] + ve_norm*(amp_init−amp_clip[1])" → pinned
    # at the clip).  Both put the optimiser in a single-polarity basin
    # that Adam-FD / LBFGS cannot escape via gradient steps (flipping
    # contact polarity requires a discrete jump that small steps don't
    # take).  The bipolar init starts inside the mixed-polarity basin
    # where field steering is possible.
    if amps_init_vector is not None:
        # Caller provided a full pre-computed amps0 (e.g. from a smart
        # spatial-contrast init + magnitude probe); we use it verbatim,
        # clipped to the configured range.  amp_init_mA is ignored in
        # this branch.
        amps0 = jnp.clip(
            jnp.asarray(amps_init_vector, dtype=jnp.float64),
            amp_clip[0], amp_clip[1],
        )
    else:
        ve_tgt_sum  = jnp.where(tgt_j[None, :, None], jnp.abs(Ve_unit_j), 0.0).sum((1, 2))
        ve_tgt_mean = ve_tgt_sum / jnp.maximum(tgt_j.sum(), 1.0)   # [K]
        ve_range    = ve_tgt_mean.max() - ve_tgt_mean.min()
        ve_norm     = (ve_tgt_mean - ve_tgt_mean.min()) / (ve_range + 1e-10)   # [K] in [0,1]
        amps0 = jnp.clip(
            (2.0 * ve_norm - 1.0) * jnp.abs(amp_init_mA) * jnp.sign(amp_init_mA),
            amp_clip[0], amp_clip[1],
        ).astype(jnp.float64)
    # `jnp.sign(amp_init_mA)` keeps backward-compat with cathodic-only callers
    # who set amp_init_mA = -0.4 (negative).  Closest contact ends up at the
    # given amp_init_mA (cathodic), farthest at −amp_init_mA (anodic).
    amps0 = amps0.astype(jnp.float64)

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

    stale_iters = 0
    stopped_at  = n_steps
    stop_reason = "max iters"
    for i in range(n_steps):
        t0 = time.time()
        (loss_val, acts_val), grads = _fd_step(amps)
        updates, opt_state = optimizer.update(grads, opt_state)
        amps = jnp.clip(optax.apply_updates(amps, updates), amp_clip[0], amp_clip[1])

        acts_np = np.array(acts_val)
        si_now  = selectivity_index(acts_np, target_mask)
        history["loss"].append(float(loss_val))
        history["bce"].append(float(wbce(acts_val, tgt_j, w)))
        history["si"].append(si_now)
        history["amps"].append(np.array(amps))
        history["acts"].append(acts_np)
        if float(loss_val) < best_loss - early_stop_loss_tol:
            best_loss = float(loss_val)
            best_amps = np.array(amps)
            stale_iters = 0
        else:
            stale_iters += 1

        if verbose and (i % max(1, n_steps // 20) == 0 or i == n_steps - 1):
            dt_ms   = (time.time() - t0) * 1000
            amp_str = "  ".join(f"{a:+.2f}" for a in np.array(amps))
            fired_now = acts_np > 0.5
            nft = int(np.sum(fired_now & target_mask_bool))
            nfn = int(np.sum(fired_now & ~target_mask_bool))
            print(
                f"  [{i:3d}/{n_steps}] loss={float(loss_val):.4f}  "
                f"SI={si_now:+.3f}  "
                f"fired={nft}/{n_target_total}t+{nfn}/{n_nontarget_total}nt  "
                f"amps=[{amp_str}] mA  dt={dt_ms:.0f}ms",
                flush=True,
            )

        # Early-stop checks (after the iter is recorded).
        if si_now >= early_stop_si:
            stopped_at = i + 1
            stop_reason = f"SI >= {early_stop_si:.3f}"
            if verbose:
                print(f"  [early stop @ iter {i}] {stop_reason}", flush=True)
            break
        if early_stop_patience > 0 and stale_iters >= early_stop_patience:
            stopped_at = i + 1
            stop_reason = f"no loss improvement for {stale_iters} iters"
            if verbose:
                print(f"  [early stop @ iter {i}] {stop_reason} "
                      f"(best_loss={best_loss:.4f})", flush=True)
            break

    return {
        "amps":        best_amps,
        "history":     history,
        "stopped_at":  stopped_at,
        "stop_reason": stop_reason,
    }


# ─────────────────────────────────── LBFGS + multi-restart rect optimization ─

def _build_rect_loss_fn(
    fiber_statics_batch: FiberStatics,
    state0_batch: tuple,
    Ve_unit_j: jnp.ndarray,            # [K, n_fibers, n_comp]
    pulse_j: jnp.ndarray,              # [T]
    pulse_prev_j: jnp.ndarray,         # [T]
    node_idx_j: jnp.ndarray,           # [n_fibers, n_nodes]
    tgt_j: jnp.ndarray,                # [n_fibers]
    w_j: jnp.ndarray,                  # [n_fibers]
    dt: float,
):
    """Closes over a single seed's data; returns scalar wq_loss(amps)."""
    def loss_fn(amps):
        # Ve_comb [n_fibers, n_comp] — minus sign because amp<0 = cathodic.
        Ve_comb = jnp.einsum("k,kfn->fn", -amps, Ve_unit_j)
        m_max = batch_integrate_m_max_fd(
            fiber_statics_batch, state0_batch, Ve_comb,
            pulse_j, pulse_prev_j, dt,
        )                                                          # [n_f, n_c]
        acts = activation_proxy_batch(m_max, node_idx_j)           # [n_f]
        return wq_loss(acts, tgt_j, w_j, amps=amps)
    return loss_fn


def _initial_amps_for_seed(
    Ve_unit_j: jnp.ndarray,            # [K, n_fibers, n_comp]
    tgt_j: jnp.ndarray,                # [n_fibers]
    amp_init_mA: float,
    amp_clip: tuple[float, float],
    n_restarts: int,
    rng_key: jnp.ndarray,              # PRNGKey for the random restarts
) -> jnp.ndarray:
    """Build an [n_restarts, K] stack of initial amplitudes.

    Restart 0: deterministic Ve-weighted init (same logic as the legacy
        run_rect_optimization — anchors the search at a known-decent start).
    Restarts 1..M-1: uniform random in (-amp_clip_span/4, +amp_clip_span/4),
        so we sample both polarities but don't pin on the clip boundary.

    rng_key must be a PRNGKey (jax.random.PRNGKey output) — passed as a
    traced array, so this function is vmap-safe over seeds.
    """
    K = Ve_unit_j.shape[0]
    # Bipolar Ve-weighted deterministic restart — see the lengthy comment in
    # run_rect_optimization for the rationale.  Maps ve_norm ∈ [0,1] to amps
    # ∈ [+|amp_init|, −|amp_init|]: closest contact cathodic, farthest anodic.
    # This is the "guard pattern" init that breaks the single-polarity basin
    # trap.
    ve_tgt_sum  = jnp.where(tgt_j[None, :, None], jnp.abs(Ve_unit_j), 0.0).sum((1, 2))
    ve_tgt_mean = ve_tgt_sum / jnp.maximum(tgt_j.sum(), 1.0)
    ve_range    = ve_tgt_mean.max() - ve_tgt_mean.min()
    ve_norm     = (ve_tgt_mean - ve_tgt_mean.min()) / (ve_range + 1e-10)
    amps_ve     = jnp.clip(
        (2.0 * ve_norm - 1.0) * jnp.abs(amp_init_mA) * jnp.sign(amp_init_mA),
        amp_clip[0], amp_clip[1],
    ).astype(jnp.float64)

    if n_restarts <= 1:
        return amps_ve[None, :]

    keys = jax.random.split(rng_key, n_restarts - 1)
    half_range = (amp_clip[1] - amp_clip[0]) / 4.0
    rand_amps = jax.vmap(
        lambda k: jax.random.uniform(
            k, shape=(K,),
            minval=-half_range, maxval=half_range,
            dtype=jnp.float64,
        )
    )(keys)                                                           # [M-1, K]
    return jnp.concatenate([amps_ve[None, :], rand_amps], axis=0)     # [M, K]


def run_rect_optimization_lbfgs(
    fiber_statics_batch: FiberStatics,
    state0_batch: tuple,
    Ve_unit: jnp.ndarray,                      # [K, n_fibers, n_comp]
    pulse_mask: jnp.ndarray,                   # [T]
    node_indices: np.ndarray,                  # [n_fibers, n_nodes]
    target_mask: np.ndarray,                   # [n_fibers] bool
    dt: float,
    n_restarts: int = 4,
    n_steps: int = 30,
    amp_init_mA: float = -0.4,
    amp_clip: tuple[float, float] = (-2.5, 2.5),
    weights: np.ndarray | None = None,
    rng_seed: int = 0,
    lbfgs_memory: int = 10,
    linesearch_max_steps: int = 5,
    verbose: bool = True,
) -> dict:
    """LBFGS with M parallel random restarts (single seed).

    Strategy
    --------
    * LBFGS uses autodiff gradient + Strong Wolfe zoom line search.
      Typically converges in ~20-50 iters vs ~100-200 for Adam-FD on the
      same loss surface (smooth, locally convex near the minimum).
    * M restarts are vmapped, so all run on the GPU in parallel. Restart 0
      starts at the Ve-weighted deterministic init (matches the legacy
      Adam-FD optimiser); 1..M-1 sample uniformly in [-clip/4, +clip/4]
      to spread across cathodic / anodic / mixed initial configurations.
    * At the end, we pick the restart with the smallest final loss.

    Returns
    -------
    result : dict
        'amps'              [K]                     — best amplitudes (mA)
        'final_loss'        float                   — wq_loss at best
        'final_acts'        [n_fibers]              — activation proxies at best
        'best_restart'      int                     — which restart won
        'all_loss_traces'   [n_restarts, n_steps+1] — per-step loss for every restart
        'all_final_amps'    [n_restarts, K]
        'all_final_losses'  [n_restarts]
    """
    K        = Ve_unit.shape[0]
    n_fibers = Ve_unit.shape[1]

    Ve_unit_j    = jnp.asarray(Ve_unit,    dtype=jnp.float64)
    pulse_j      = jnp.asarray(pulse_mask, dtype=jnp.float64)
    pulse_prev_j = jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), pulse_j[:-1]])
    node_idx_j   = jnp.asarray(node_indices, dtype=jnp.int32)
    tgt_j        = jnp.asarray(target_mask,  dtype=jnp.float64)
    w_j = (jnp.ones(n_fibers, dtype=jnp.float64) / n_fibers
           if weights is None else jnp.asarray(weights, dtype=jnp.float64))

    loss_fn = _build_rect_loss_fn(
        fiber_statics_batch, state0_batch,
        Ve_unit_j, pulse_j, pulse_prev_j,
        node_idx_j, tgt_j, w_j, dt,
    )

    init_amps = _initial_amps_for_seed(
        Ve_unit_j, tgt_j, amp_init_mA, amp_clip, n_restarts,
        jax.random.PRNGKey(rng_seed),
    )                                                                  # [M, K]

    optimizer = optax.lbfgs(
        memory_size=lbfgs_memory,
        scale_init_precond=True,
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=linesearch_max_steps,
        ),
    )

    def one_restart(amps0: jnp.ndarray):
        """Run LBFGS for n_steps starting at amps0. vmappable."""
        opt_state = optimizer.init(amps0)

        def step(carry, _):
            params, opt_state = carry
            value, grad = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(
                grad, opt_state, params,
                value=value, grad=grad, value_fn=loss_fn,
            )
            params = jnp.clip(
                optax.apply_updates(params, updates),
                amp_clip[0], amp_clip[1],
            )
            return (params, opt_state), value

        (final_params, _), losses = jax.lax.scan(
            step, (amps0, opt_state), None, length=n_steps,
        )
        final_loss = loss_fn(final_params)
        # Concatenate so loss_traces has length n_steps+1 (init + post-step values)
        loss_traces = jnp.concatenate([losses, final_loss[None]], axis=0)
        return final_params, final_loss, loss_traces

    # lax.map (instead of vmap) over the M restarts.
    #
    # The XLA remat planner has known pathological compile times when
    # @jax.checkpoint is nested inside vmap(scan(value_and_grad(scan(...))))
    # and optax.scale_by_zoom_linesearch is involved — the 5-level-nested
    # control flow takes 25+ min to schedule on A16 even at small N_FIBERS.
    #
    # lax.map runs the M restarts SEQUENTIALLY but reuses a single compiled
    # graph (no outer vmap dim).  This:
    #   - Eliminates the outer vmap level → graph compiles in 3-5 min
    #     instead of 25+ min.
    #   - Peak memory = single-restart footprint (no parallel accumulation
    #     across restarts), so @jax.checkpoint isn't strictly required for
    #     N_FIBERS ≤ ~400 on A16.
    #   - Wall time per restart is the same; total wall is M× single-restart
    #     (e.g. M=2 → 2× the runtime of one restart, but starts after a
    #     much faster compile, so net wall is way smaller).
    if verbose:
        print(
            f"  LBFGS multi-restart: K={K} contacts, n_fibers={n_fibers}, "
            f"{n_restarts} restarts × {n_steps} steps "
            f"(sequential via lax.map). Compiling ...",
            flush=True,
        )

    t0 = time.time()
    final_params_all, final_losses_all, loss_traces_all = jax.jit(
        lambda init: jax.lax.map(one_restart, init),
    )(init_amps)
    # Block on the result so timing is accurate.
    jax.block_until_ready(final_losses_all)
    t_total = time.time() - t0

    best_idx = int(jnp.argmin(final_losses_all))

    # Compute acts at the best restart for reporting.
    Ve_comb_best = jnp.einsum("k,kfn->fn", -final_params_all[best_idx], Ve_unit_j)
    m_max_best = batch_integrate_m_max_fd(
        fiber_statics_batch, state0_batch, Ve_comb_best,
        pulse_j, pulse_prev_j, dt,
    )
    best_acts = activation_proxy_batch(m_max_best, node_idx_j)

    if verbose:
        print(
            f"  LBFGS done in {t_total:.1f} s. Best restart: {best_idx}/{n_restarts}, "
            f"final loss = {float(final_losses_all[best_idx]):.4f}.",
            flush=True,
        )

    return {
        "amps":             np.array(final_params_all[best_idx]),
        "final_loss":       float(final_losses_all[best_idx]),
        "final_acts":       np.array(best_acts),
        "best_restart":     best_idx,
        "all_loss_traces":  np.array(loss_traces_all),
        "all_final_amps":   np.array(final_params_all),
        "all_final_losses": np.array(final_losses_all),
        "wall_time_s":      t_total,
    }


# ─────────────────────────── multi-seed batched LBFGS rect optimization ───────

def run_rect_optimization_lbfgs_batched(
    fiber_statics_batches: list,                # list of S FiberStatics
    state0_batches: list,                        # list of S state tuples
    Ve_units: np.ndarray,                        # [S, K, n_fibers, n_comp]
    pulse_mask: np.ndarray,                      # [T]
    node_indices_batches: np.ndarray,            # [S, n_fibers, n_nodes]
    target_masks: np.ndarray,                    # [S, n_fibers] bool
    dt: float,
    n_restarts: int = 4,
    n_steps: int = 30,
    amp_init_mA: float = -0.4,
    amp_clip: tuple[float, float] = (-2.5, 2.5),
    weights_per_seed: np.ndarray | None = None,  # [S, n_fibers] or None
    rng_seeds: list[int] | None = None,
    lbfgs_memory: int = 10,
    linesearch_max_steps: int = 5,
    verbose: bool = True,
) -> list[dict]:
    """LBFGS multi-restart over a batch of S seeds via outer vmap.

    All S × M × n_fibers effective fibers participate in a single vmapped
    forward pass per LBFGS step. On an a100 (40 GB) S=8, M=8 fits with
    n_fibers=100; on an a16 (16 GB) S=4, M=4 is the safe budget.

    Returns a list of S dicts, one per input seed, each with the same shape
    as the single-seed run_rect_optimization_lbfgs output.
    """
    S = len(fiber_statics_batches)
    assert len(state0_batches) == S
    assert Ve_units.shape[0] == S
    assert node_indices_batches.shape[0] == S
    assert target_masks.shape[0] == S
    if rng_seeds is None:
        rng_seeds = list(range(S))
    assert len(rng_seeds) == S

    n_fibers = Ve_units.shape[2]
    K        = Ve_units.shape[1]

    # Stack the S FiberStatics into FiberStatics-of-arrays with shape [S, n_fibers, ...]
    stacked_fs = FiberStatics(*[
        jnp.stack([getattr(fs, field) for fs in fiber_statics_batches], axis=0)
        for field in FiberStatics._fields
    ])
    # state0 is a tuple of 4 arrays each [n_fibers, n_comp]; stack across seeds.
    stacked_s0 = tuple(
        jnp.stack([s0[i] for s0 in state0_batches], axis=0)            # [S, n_fibers, n_comp]
        for i in range(len(state0_batches[0]))
    )

    Ve_units_j        = jnp.asarray(Ve_units, dtype=jnp.float64)
    node_idx_j        = jnp.asarray(node_indices_batches, dtype=jnp.int32)
    tgt_j             = jnp.asarray(target_masks, dtype=jnp.float64)
    if weights_per_seed is None:
        weights_per_seed_j = jnp.ones((S, n_fibers), dtype=jnp.float64) / n_fibers
    else:
        weights_per_seed_j = jnp.asarray(weights_per_seed, dtype=jnp.float64)
    pulse_j      = jnp.asarray(pulse_mask, dtype=jnp.float64)
    pulse_prev_j = jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), pulse_j[:-1]])
    # One PRNGKey per seed (vmap-safe; can't use Python int inside vmap).
    rng_keys_j = jax.vmap(jax.random.PRNGKey)(jnp.asarray(rng_seeds, dtype=jnp.uint32))

    optimizer = optax.lbfgs(
        memory_size=lbfgs_memory,
        scale_init_precond=True,
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=linesearch_max_steps,
        ),
    )

    def one_seed(fs_one, s0_one, Ve_unit_one, node_idx_one, tgt_one, w_one, rng_key_one):
        """Run M restarts of LBFGS for one seed. vmappable over the S axis."""
        loss_fn = _build_rect_loss_fn(
            fs_one, s0_one,
            Ve_unit_one, pulse_j, pulse_prev_j,
            node_idx_one, tgt_one, w_one, dt,
        )
        init_amps = _initial_amps_for_seed(
            Ve_unit_one, tgt_one, amp_init_mA, amp_clip, n_restarts, rng_key_one,
        )

        def one_restart(amps0):
            opt_state = optimizer.init(amps0)
            def step(carry, _):
                params, opt_state = carry
                value, grad = jax.value_and_grad(loss_fn)(params)
                updates, opt_state = optimizer.update(
                    grad, opt_state, params,
                    value=value, grad=grad, value_fn=loss_fn,
                )
                params = jnp.clip(
                    optax.apply_updates(params, updates),
                    amp_clip[0], amp_clip[1],
                )
                return (params, opt_state), value
            (final_params, _), losses = jax.lax.scan(
                step, (amps0, opt_state), None, length=n_steps,
            )
            final_loss = loss_fn(final_params)
            loss_traces = jnp.concatenate([losses, final_loss[None]], axis=0)
            return final_params, final_loss, loss_traces

        # lax.map (instead of vmap) over the M restarts — same rationale as
        # in run_rect_optimization_lbfgs above.  Sequential restarts; smaller
        # XLA graph; avoids the remat planner pathology under zoom_linesearch.
        final_params_all, final_losses_all, loss_traces_all = jax.lax.map(
            one_restart, init_amps
        )
        best_idx = jnp.argmin(final_losses_all)
        best_amps = final_params_all[best_idx]
        best_loss = final_losses_all[best_idx]

        # Reproduce acts at the best restart
        Ve_comb_best = jnp.einsum("k,kfn->fn", -best_amps, Ve_unit_one)
        m_max_best = batch_integrate_m_max_fd(
            fs_one, s0_one, Ve_comb_best,
            pulse_j, pulse_prev_j, dt,
        )
        best_acts = activation_proxy_batch(m_max_best, node_idx_one)

        return (final_params_all, final_losses_all, loss_traces_all,
                best_amps, best_loss, best_acts, best_idx)

    if verbose:
        print(
            f"  LBFGS batched: S={S} seeds × M={n_restarts} restarts × {n_steps} steps. "
            f"Compiling ...",
            flush=True,
        )

    t0 = time.time()
    batched = jax.jit(jax.vmap(one_seed))(
        stacked_fs, stacked_s0, Ve_units_j, node_idx_j, tgt_j,
        weights_per_seed_j, rng_keys_j,
    )
    jax.block_until_ready(batched[1])   # final_losses_all
    t_total = time.time() - t0

    if verbose:
        print(f"  LBFGS batched done in {t_total:.1f} s for {S} seeds.", flush=True)

    final_params_all_S, final_losses_all_S, loss_traces_all_S, \
        best_amps_S, best_losses_S, best_acts_S, best_idx_S = batched

    results = []
    for s in range(S):
        results.append({
            "amps":             np.array(best_amps_S[s]),
            "final_loss":       float(best_losses_S[s]),
            "final_acts":       np.array(best_acts_S[s]),
            "best_restart":     int(best_idx_S[s]),
            "all_loss_traces":  np.array(loss_traces_all_S[s]),
            "all_final_amps":   np.array(final_params_all_S[s]),
            "all_final_losses": np.array(final_losses_all_S[s]),
            "wall_time_s":      t_total / S,    # amortised across seeds
        })
    return results


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
    early_stop_patience: int = 0,
    # SI-based and loss-tolerance early stop (parallels the rect optimiser).
    # Without these, the wave Adam ran the full 100 iters even when the
    # warm-start already sat at SI=1.0: best_loss kept improving by ~1e-6
    # each iter (microscopic refinement) so the patience counter never
    # accumulated.  Bottom line: 5x cluster time wasted on iters that did
    # nothing visible.
    early_stop_si:       float = 1.0,
    early_stop_loss_tol: float = 1e-5,
) -> dict:
    """Optimize arbitrary per-contact waveforms u[K, T] via autodiff.

    NOTE: K×T degrees of freedom (e.g. 6×1600 = 9600) make FD impractical.
    Autodiff backward through the ODE scan is slow (~100s/iter for T=1600).
    Reduce T_STOP / DT or switch to a surrogate for large-scale sweeps.

    Warm-start note
    ---------------
    When ``u_init`` is the rect-found amps × pulse_mask, iter 0 already sits
    at the rect optimum.  In that case ``lr=5e-3`` is far too aggressive —
    the first Adam step overshoots into the "fire everything" attractor and
    a near-perfect SI≈1 collapses to SI=0 in <20 iters.  For warm-started
    runs use ``lr <= 1e-3`` and ``early_stop_patience`` >0 so the optimiser
    can't destroy a good init.

    Parameters
    ----------
    early_stop_patience : int, default 0 (disabled)
        Stop if best_loss hasn't improved for this many consecutive iters.
        Cheap insurance against Adam wandering off a good warm-start.

    Returns
    -------
    result : dict with keys
        'u'        : [K, T] best-loss waveforms (mA)
        'best_iter': int — iter index where best_loss was attained
        'best_acts': [n_fibers] — activation proxies at best_iter
        'best_si'  : float — SI at best_iter (use this, not history[-1])
        'history'  : dict of lists {'loss', 'bce', 'si', 'acts'}
    """
    K        = Ve_unit.shape[0]
    n_fibers = Ve_unit.shape[1]

    Ve_unit_j  = jnp.asarray(Ve_unit,      dtype=jnp.float64)
    node_idx_j = jnp.asarray(node_indices, dtype=jnp.int32)
    tgt_j      = jnp.asarray(target_mask,  dtype=jnp.float64)
    w = (jnp.ones(n_fibers, dtype=jnp.float64) / n_fibers
         if weights is None else jnp.asarray(weights, dtype=jnp.float64))

    # Boolean target mask + totals for the per-iter firing summary in the
    # verbose log line.  Mirrors the rect optimiser, so over/undersaturation
    # is visible during the wave run too.
    target_mask_bool   = np.asarray(target_mask, dtype=bool)
    n_target_total     = int(target_mask_bool.sum())
    n_nontarget_total  = int((~target_mask_bool).sum())

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
    best_iter = 0
    best_acts = np.zeros(n_fibers, dtype=np.float64)
    best_si   = 0.0
    stale     = 0   # iters since last best_loss improvement

    if verbose:
        print(
            f"  Waveform opt (autodiff): K={K} contacts, T={T} steps, {n_steps} iters, "
            f"lr={lr:.1e}, patience={early_stop_patience}",
            flush=True,
        )
        print("  [iter 0] XLA compile — first call only ...", flush=True)

    for i in range(n_steps):
        t0 = time.time()
        # Evaluate at current u BEFORE stepping so iter 0 records the
        # warm-start performance (critical for u_init = rect amps × pulse).
        (loss_val, acts_val), grads = loss_and_grad(u)

        acts_np = np.array(acts_val)
        si_now  = selectivity_index(acts_np, target_mask)
        history["loss"].append(float(loss_val))
        history["bce"].append(float(wbce(acts_val, tgt_j, w)))
        history["si"].append(si_now)
        history["acts"].append(acts_np)
        # Improvement now requires beating best_loss by > early_stop_loss_tol.
        # Without the tol, microscopic 1e-6 wiggles count as "improvement" and
        # the patience counter never accumulates (the wave Adam's fine-tuning
        # never actually halts even when SI is already at the ceiling).
        if float(loss_val) < best_loss - early_stop_loss_tol:
            best_loss = float(loss_val)
            best_u    = np.array(u)
            best_iter = i
            best_acts = acts_np
            best_si   = si_now
            stale     = 0
        else:
            stale += 1

        # Step AFTER recording so we keep iter 0 = warm-start state.
        updates, opt_state = optimizer.update(grads, opt_state)
        u = jnp.clip(optax.apply_updates(u, updates), u_clip[0], u_clip[1])

        if verbose and (i % max(1, n_steps // 20) == 0 or i == n_steps - 1):
            dt_ms = (time.time() - t0) * 1000
            u_np  = np.array(u)
            fired_now = acts_np > 0.5
            nft = int(np.sum(fired_now & target_mask_bool))
            nfn = int(np.sum(fired_now & ~target_mask_bool))
            print(
                f"  [{i:3d}/{n_steps}] loss={float(loss_val):.4f}  "
                f"SI={si_now:+.3f}  "
                f"fired={nft}/{n_target_total}t+{nfn}/{n_nontarget_total}nt  "
                f"best={best_loss:.4f}@{best_iter}  "
                f"rms={float(np.sqrt(np.mean(u_np**2))):.3f} mA  "
                f"peak={float(np.abs(u_np).max()):.3f} mA  dt={dt_ms:.0f}ms",
                flush=True,
            )

        # SI-based early stop: if the run has already hit (or beaten) the
        # target selectivity, every subsequent iter is wasted cluster time.
        if si_now >= early_stop_si:
            if verbose:
                print(f"  [early stop @ iter {i}] SI >= {early_stop_si:.3f}",
                      flush=True)
            break

        if early_stop_patience > 0 and stale >= early_stop_patience:
            if verbose:
                print(f"  [early stop] no improvement for {stale} iters "
                      f"(best loss {best_loss:.4f} @ iter {best_iter})",
                      flush=True)
            break

    return {
        "u":         best_u,
        "best_iter": best_iter,
        "best_acts": best_acts,
        "best_si":   float(best_si),
        "history":   history,
    }


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
