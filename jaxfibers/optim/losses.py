"""Differentiable losses for spatial selectivity optimization.

Activation proxy
----------------
For fiber f with m-gate trace M[T, n_comp], we define the activation proxy as:

    act_f = min( max_t M[t, first_node],  max_t M[t, last_node] )

This is in [0, 1], approaches ~0.99 when the fiber fires and propagation
reaches both ends, and stays near the resting value (~0.03) otherwise.
Gradients flow cleanly because jnp.maximum and jnp.minimum are differentiable.

WQ loss (Weighted Quality)
--------------------------
    L = sum_f  w_f * [ target_f * (1 - act_f)  +  (1 - target_f) * act_f ]

- target fiber: penalise lack of activation  → want act_f → 1
- off-target:   penalise activation          → want act_f → 0

WBCE (monitoring only)
-----------------------
Weighted binary cross-entropy of act_f against target_mask.  Not used for
backprop (gradients explode near 0/1); only logged during optimisation.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def activation_proxy_batch(
    m_max_batch: jnp.ndarray,
    node_indices: np.ndarray | jnp.ndarray,
    soft_temperature: float = 0.0,
    soft_threshold:  float = 0.5,
) -> jnp.ndarray:
    """Compute activation proxy for all fibers.

    Parameters
    ----------
    m_max_batch : [n_fibers, n_comp]
        Max m-gate at each compartment over the simulation (from batch_integrate_m_max).
    node_indices : [n_fibers, n_nodes] int
        Compartment indices of nodes for each fiber.
    soft_temperature : float, default 0.0
        If 0.0, use the original BINARY-ish proxy:
            min(m_max[first_node], m_max[last_node])
        Returns ~1.0 for a fibre that fires and propagates to both ends,
        ~0.05 (the resting m-gate value) otherwise.  The gradient is
        nearly zero everywhere except in a narrow window at threshold,
        which traps waveform/Adam optimisation in flat-loss saturation
        on mixed-diameter problems.

        If > 0, apply a sigmoid smoother per node:
            soft_act = sigmoid((m_node - soft_threshold) / soft_temperature)
        then take the min across the two end nodes.  Larger temperature
        widens the gradient-informative window; gradients survive far
        from threshold so the optimiser can adjust toward a transition
        from any starting point.  Typical values: 0.05 (mildly smoothed,
        almost binary) to 0.2 (very smooth, gradient everywhere).
    soft_threshold : float, default 0.5
        Sigmoid centre.  0.5 places the soft boundary halfway between
        the resting (~0.03) and saturated (~1.0) m-gate values.

    Returns
    -------
    acts : [n_fibers]  ∈ [0, 1]
    """
    node_indices = jnp.asarray(node_indices, dtype=jnp.int32)
    n_fibers = m_max_batch.shape[0]
    f_idx = jnp.arange(n_fibers)[:, None]            # [n_fibers, 1]
    m_nodes = m_max_batch[f_idx, node_indices]        # [n_fibers, n_nodes]
    if soft_temperature > 0.0:
        # Soft sigmoid proxy — gradient informative far from threshold.
        soft = jax.nn.sigmoid((m_nodes - soft_threshold) / soft_temperature)
        return jnp.minimum(soft[:, 0], soft[:, -1])
    # Binary-ish proxy (original behaviour).
    return jnp.minimum(m_nodes[:, 0], m_nodes[:, -1])


def wq_loss(
    acts: jnp.ndarray,
    target_mask: np.ndarray | jnp.ndarray,
    weights: np.ndarray | jnp.ndarray | None = None,
    amps: jnp.ndarray | None = None,
    energy_lambda: float = 0.0,
) -> jnp.ndarray:
    """Weighted selectivity loss with optional amplitude-energy regularisation.

    Parameters
    ----------
    acts : [n_fibers]
        Activation proxies from activation_proxy_batch.
    target_mask : [n_fibers] bool
        True for target fascicle fibers.
    weights : [n_fibers] or None
        Per-fiber weights (e.g. fascicle cross-sectional area). Uniform if None.
    amps : jnp.ndarray | None, shape [K] or [K, T]
        Per-contact amplitudes (rect mode: [K]) or per-contact waveforms
        (waveform mode: [K, T]).  Required iff ``energy_lambda > 0``.
    energy_lambda : float, default 0.0
        Energy-regularisation strength.  When > 0, adds
        ``energy_lambda * mean(amps²)`` to the loss.  This prevents the
        trivial "fire everything" minimum that the optimiser falls
        into when amps can grow arbitrarily (e.g. mixed-diameter
        type-selectivity problems where rect/wave both plateau at
        SI=0 because cranking all contacts to ±3 mA fires both target
        and off-target indiscriminately).  Typical values:
        1e-3 to 1e-2 in mA².

    Returns
    -------
    loss : scalar  (lower = better spatial selectivity)
    """
    n = acts.shape[0]
    target = jnp.asarray(target_mask, dtype=jnp.float64)
    w = jnp.ones(n, dtype=jnp.float64) / n if weights is None else jnp.asarray(weights, dtype=jnp.float64)
    sel = jnp.sum(w * (target * (1.0 - acts) + (1.0 - target) * acts))
    if energy_lambda > 0.0 and amps is not None:
        amps_j = jnp.asarray(amps, dtype=jnp.float64)
        sel = sel + energy_lambda * jnp.mean(amps_j ** 2)
    return sel


def wbce(
    acts: jnp.ndarray,
    target_mask: np.ndarray | jnp.ndarray,
    weights: np.ndarray | jnp.ndarray | None = None,
    eps: float = 1e-7,
) -> jnp.ndarray:
    """Weighted binary cross-entropy (monitoring only, not used for backprop).

    Parameters
    ----------
    acts : [n_fibers]  activation proxies.
    target_mask : [n_fibers] bool.
    weights : [n_fibers] or None.
    eps : float  clipping to avoid log(0).

    Returns
    -------
    bce : scalar
    """
    n = acts.shape[0]
    target = jnp.asarray(target_mask, dtype=jnp.float64)
    w = jnp.ones(n, dtype=jnp.float64) / n if weights is None else jnp.asarray(weights, dtype=jnp.float64)
    a = jnp.clip(acts, eps, 1.0 - eps)
    bce = -(target * jnp.log(a) + (1.0 - target) * jnp.log(1.0 - a))
    return jnp.sum(w * bce)


def selectivity_index(
    acts: jnp.ndarray,
    target_mask: np.ndarray | jnp.ndarray,
    threshold: float = 0.5,
) -> float:
    """Hard selectivity index: target activation rate minus off-target rate.

    SI = 1.0 means perfect (target fires, off-target silent).
    SI = 0.0 means no discrimination.
    SI = -1.0 means reversed.
    """
    target = np.asarray(target_mask, dtype=bool)
    hard = np.asarray(acts) > threshold
    n_tgt = target.sum()
    n_off = (~target).sum()
    tgt_rate = (hard[target].sum() / n_tgt) if n_tgt > 0 else 0.0
    off_rate = (hard[~target].sum() / n_off) if n_off > 0 else 0.0
    return float(tgt_rate - off_rate)
