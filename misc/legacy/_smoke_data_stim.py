"""Smoke test: can we JIT a function whose data_stimuli amplitude varies?

If yes, we compile once per shape and re-run with different amplitudes cheaply,
which is the key to making threshold bisection fast in Jaxley.
"""
import time
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import jaxley as jx

cell = jx.Cell(jx.Branch(jx.Compartment(), ncomp=5), parents=[-1])
cell.branch(0).comp(2).record("v")

n_steps = 200
t = np.linspace(0, 5, n_steps + 1)
shape = np.where((t > 1) & (t < 1.1), 1.0, 0.0)


def run(amp):
    currents = jnp.asarray(amp * shape)[None, :]
    ds = cell.branch(0).comp(2).data_stimulate(currents)
    return jx.integrate(cell, delta_t=5 / 200, t_max=5.0,
                        data_stimuli=ds, solver="bwd_euler")


run_jit = jax.jit(run)
for amp in [0.5, 0.7, 0.3, 0.4, 0.9]:
    t0 = time.time()
    v = run_jit(amp)
    v.block_until_ready()
    dt = time.time() - t0
    print(f"amp={amp:.2f}  peak={float(jnp.max(v)):.3f}  wall={dt:.3f}s")
