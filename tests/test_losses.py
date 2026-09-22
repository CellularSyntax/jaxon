"""Unit tests for the selectivity index. Requires only jax + numpy."""
import numpy as np
from jaxon.optim.losses import selectivity_index


def test_si_perfect():
    # 4 fibers: first two on-target and firing, last two off-target and silent.
    acts = np.array([1.0, 1.0, 0.0, 0.0])
    target = np.array([True, True, False, False])
    assert abs(selectivity_index(acts, target) - 1.0) < 1e-9


def test_si_reversed():
    acts = np.array([0.0, 0.0, 1.0, 1.0])
    target = np.array([True, True, False, False])
    assert abs(selectivity_index(acts, target) - (-1.0)) < 1e-9


def test_si_no_discrimination():
    acts = np.array([1.0, 1.0, 1.0, 1.0])
    target = np.array([True, True, False, False])
    assert abs(selectivity_index(acts, target)) < 1e-9


def test_si_threshold():
    acts = np.array([0.6, 0.4, 0.0, 0.0])
    target = np.array([True, True, False, False])
    # Only the 0.6 target fiber counts as fired at threshold 0.5 -> 0.5 - 0 = 0.5
    assert abs(selectivity_index(acts, target, threshold=0.5) - 0.5) < 1e-9
