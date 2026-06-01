"""Differentiable simulation objectives for gradient-based stimulation optimization.

All functions accept JAX arrays and are fully differentiable via jax.grad / jax.jit.
Intended use: define a loss as ``objective(integrate(...))`` and call ``jax.grad(loss)``.

Gradient regime notes:
  max_vm          — gradient is non-zero wherever V_peak changes with the parameter.
                    Flat (≈0) deep below threshold; large at and above threshold.
  activation_prob — sigmoid smoothing gives a non-zero gradient across ≈4σ around
                    v_thresh (default ±8 mV for σ=2 mV). Best surrogate for optimizing
                    across the threshold transition.
  recruitment_fraction — mean over N fibers; gradient scales as 1/N but remains non-zero.
  selectivity     — r_target − r_off; gradient is non-zero wherever the two populations
                    respond differently to the parameter.
  energy          — always non-zero gradient (d(I²dt)/d(I) = 2I·dt·nsteps).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def max_vm(trace: jnp.ndarray) -> jnp.ndarray:
    """Peak membrane voltage at the probe compartment (mV).

    trace : (nsteps,) — Vm time series at center node from integrate(..., record='center')
    """
    return jnp.max(trace)


def activation_prob(trace: jnp.ndarray,
                    v_thresh: float = -20.0,
                    sigma: float = 2.0) -> jnp.ndarray:
    """Differentiable AP probability: sigmoid( (V_peak − v_thresh) / sigma ).

    Smooth surrogate for the hard threshold indicator I(V_peak >= v_thresh).
    Returns a value in (0, 1). Gradient is non-zero in a window of ≈4σ around
    the threshold, which is the regime where gradient-based optimization is most
    effective.

    trace    : (nsteps,) — Vm time series
    v_thresh : AP detection threshold (mV), default −20 mV
    sigma    : sigmoid width (mV); smaller → sharper but narrower gradient support
    """
    v_peak = jnp.max(trace)
    return jax.nn.sigmoid((v_peak - v_thresh) / sigma)


def recruitment_fraction(population_traces: jnp.ndarray,
                         v_thresh: float = -20.0,
                         sigma: float = 2.0) -> jnp.ndarray:
    """Mean activation probability across a fiber population.

    population_traces : (N, nsteps) — center-node Vm traces for N fibers,
                        typically from jax.vmap over amplitudes or positions
    v_thresh, sigma   : forwarded to activation_prob
    """
    probs = jax.vmap(lambda tr: activation_prob(tr, v_thresh, sigma))(population_traces)
    return jnp.mean(probs)


def selectivity(target_traces: jnp.ndarray,
                off_traces: jnp.ndarray,
                v_thresh: float = -20.0,
                sigma: float = 2.0) -> jnp.ndarray:
    """Target-minus-off-target recruitment fraction ∈ [−1, 1].

    Returns r_target − r_off. Maximize this objective to selectively activate
    target fibers while suppressing off-target activation.

    target_traces : (N_on,  nsteps) — Vm traces for target population
    off_traces    : (N_off, nsteps) — Vm traces for off-target population
    """
    r_on  = recruitment_fraction(target_traces, v_thresh, sigma)
    r_off = recruitment_fraction(off_traces,    v_thresh, sigma)
    return r_on - r_off


def energy(waveform: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Charge-weighted energy ∫I² dt (mA²·ms).

    waveform : (nsteps,) — signed stimulation current at each timestep (mA).
               For a rectangular pulse: ``amp_mA * pulse_mask``.
    dt       : timestep duration (ms)
    """
    return jnp.sum(waveform ** 2) * dt
