"""Unit tests for the coupled (V_i, V_px) block-Thomas solver.

These check the linear-algebra core of jaxon's extracellular solver against a
dense LU reference — the same check the module docstring reports (block-Thomas
matches dense LU to ~1e-11 mV). They require only jax + numpy (no jaxley/NEURON).
"""
import numpy as np
import jax.numpy as jnp
import jax

from jaxon.stim.extracellular_coupled import block_thomas, block_thomas_assoc, _inv2

jax.config.update("jax_enable_x64", True)


def _dense_from_blocks(D, Low, Up):
    """Assemble the dense 2n x 2n matrix from 2x2 block-tridiagonal parts.

    Row k: Low[k] @ x[k-1] + D[k] @ x[k] + Up[k] @ x[k+1] = R[k].
    Low[0] and Up[n-1] are unused.
    """
    n = D.shape[0]
    M = np.zeros((2 * n, 2 * n))
    for k in range(n):
        M[2 * k:2 * k + 2, 2 * k:2 * k + 2] = np.asarray(D[k])
        if k + 1 < n:
            M[2 * k:2 * k + 2, 2 * (k + 1):2 * (k + 1) + 2] = np.asarray(Up[k])
        if k - 1 >= 0:
            M[2 * k:2 * k + 2, 2 * (k - 1):2 * (k - 1) + 2] = np.asarray(Low[k])
    return M


def _random_system(n, seed=0):
    rng = np.random.default_rng(seed)
    # Diagonally dominant 2x2 blocks so the system is well conditioned.
    D = rng.standard_normal((n, 2, 2))
    D += np.eye(2)[None] * 6.0
    Low = 0.2 * rng.standard_normal((n, 2, 2))
    Up = 0.2 * rng.standard_normal((n, 2, 2))
    R = rng.standard_normal((n, 2))
    return (jnp.asarray(D), jnp.asarray(Low), jnp.asarray(Up), jnp.asarray(R))


def test_inv2_matches_numpy():
    rng = np.random.default_rng(1)
    M = rng.standard_normal((5, 2, 2)) + np.eye(2)[None] * 3.0
    got = np.asarray(_inv2(jnp.asarray(M)))
    exp = np.linalg.inv(M)
    assert np.allclose(got, exp, atol=1e-10)


def test_block_thomas_matches_dense_lu():
    for seed in (0, 1, 2):
        D, Low, Up, R = _random_system(24, seed=seed)
        x = np.asarray(block_thomas(D, Low, Up, R)).reshape(-1)
        M = _dense_from_blocks(D, Low, Up)
        x_ref = np.linalg.solve(M, np.asarray(R).reshape(-1))
        assert np.allclose(x, x_ref, atol=1e-9), np.abs(x - x_ref).max()


def test_block_thomas_assoc_matches_sequential():
    D, Low, Up, R = _random_system(24, seed=3)
    x_seq = np.asarray(block_thomas(D, Low, Up, R))
    x_assoc = np.asarray(block_thomas_assoc(D, Low, Up, R))
    assert np.allclose(x_seq, x_assoc, atol=1e-9), np.abs(x_seq - x_assoc).max()
