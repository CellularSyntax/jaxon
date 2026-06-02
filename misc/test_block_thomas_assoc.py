"""Quick correctness check: block_thomas_assoc vs block_thomas."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)

from jaxfibers.stim.extracellular_coupled import block_thomas, block_thomas_assoc

rng = np.random.default_rng(42)

for n in [5, 11, 51, 221, 551]:
    D   = jnp.array(rng.standard_normal((n, 2, 2)) + 5*np.eye(2), dtype=jnp.float64)
    Low = jnp.array(rng.standard_normal((n, 2, 2)) * 0.3, dtype=jnp.float64)
    Up  = jnp.array(rng.standard_normal((n, 2, 2)) * 0.3, dtype=jnp.float64)
    R   = jnp.array(rng.standard_normal((n, 2)), dtype=jnp.float64)

    x_seq   = block_thomas(D, Low, Up, R)
    x_assoc = block_thomas_assoc(D, Low, Up, R)

    err = float(jnp.max(jnp.abs(x_seq - x_assoc)))
    status = "PASS" if err < 1e-11 else "FAIL"
    print(f"  n={n:4d}  max|err|={err:.2e}  {status}")
    assert err < 1e-11, f"n={n}: error {err:.2e} exceeds tolerance"

print("All tests PASSED.")
