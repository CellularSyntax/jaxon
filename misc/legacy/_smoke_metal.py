"""Quick check: which Jaxley/JAX operations work on Apple Metal?

Tests:
  1. A trivial JAX op (sanity).
  2. A small Jaxley simulation (5 comps, 100 steps).
  3. The full MRG fiber (111 comps) — forward only.
  4. The full MRG fiber — forward + gradient (the gradient test that
     gates exp_3 on Metal).

Each test is run on both CPU and METAL devices; reports compile + run time.
"""
import time
import numpy as np
import jax
import jax.numpy as jnp


def banner(s): print("\n=== " + s + " ===", flush=True)


def time_on(device, fn, *args):
    try:
        with jax.default_device(device):
            t0 = time.time(); out = fn(*args); jax.block_until_ready(out)
            t1 = time.time(); out2 = fn(*args); jax.block_until_ready(out2)
            t2 = time.time()
        return True, t1 - t0, t2 - t1, out
    except Exception as e:
        return False, None, None, f"{type(e).__name__}: {e}"


def main():
    devs = jax.devices()
    print(f"jax.devices(): {devs}", flush=True)
    cpu = next((d for d in jax.devices('cpu')), None)
    metal = next((d for d in devs if d.platform.lower() == 'metal'), None)
    print(f"CPU device: {cpu}\nMetal device: {metal}", flush=True)
    assert cpu is not None and metal is not None

    # --- 1. trivial op
    banner("1. trivial matmul")
    @jax.jit
    def m(x): return (x @ x.T).sum()
    x = jnp.ones((256, 256))
    for dev, name in [(cpu, 'CPU'), (metal, 'METAL')]:
        ok, t1, t2, out = time_on(dev, m, x)
        print(f"  {name:6s}  ok={ok}  compile={t1}  run={t2}  result={float(out):.2e}" if ok
              else f"  {name:6s}  FAIL: {out}", flush=True)

    # --- 2. small Jaxley
    banner("2. small Jaxley (5 comps, 100 steps)")
    import jaxley as jx
    from jaxley.channels import Leak

    def make_small_forward():
        b = jx.Cell(jx.Branch(jx.Compartment(), ncomp=5), parents=[-1])
        b.insert(Leak())
        b.branch(0).comp(2).record("v")
        shape = jnp.where(jnp.arange(101) * 0.05 > 1.0, 0.1, 0.0)
        def fwd(amp):
            ds = b.branch(0).comp(2).data_stimulate(amp * shape)
            return jnp.max(jx.integrate(b, delta_t=0.05, t_max=5.0,
                                         data_stimuli=ds, solver="bwd_euler"))
        return jax.jit(fwd)

    for dev, name in [(cpu, 'CPU'), (metal, 'METAL')]:
        try:
            with jax.default_device(dev):
                fwd = make_small_forward()
                t0 = time.time(); v1 = float(fwd(0.5)); t1 = time.time() - t0
                t0 = time.time(); v2 = float(fwd(0.7)); t2 = time.time() - t0
            print(f"  {name:6s}  compile+run1={t1:.3f}s  run2={t2:.4f}s  peak={v1:.2f}", flush=True)
        except Exception as e:
            print(f"  {name:6s}  FAIL: {type(e).__name__}: {e}", flush=True)

    # --- 3. full MRG fiber forward
    banner("3. full MRG fiber forward (111 comps, dt=0.005, 5 ms)")
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from jaxfibers.fibers.mrg import build_mrg, node_indices
    from jaxfibers.stim.intracellular import rectangular_pulse

    def make_mrg_forward():
        cell, geom = build_mrg(diameter=10.0, n_nodes=11)
        nodes = node_indices(geom)
        mid = nodes[len(nodes) // 2]
        ts = np.arange(int(5.0 / 0.005) + 1) * 0.005
        shape = jnp.asarray(rectangular_pulse(ts, 1.0, 0.1, amp_nA=1.0))
        cell.branch(0).comp(mid).record("v")
        def fwd(amp):
            ds = cell.branch(0).comp(int(mid)).data_stimulate(amp * shape)
            return jnp.max(jx.integrate(cell, delta_t=0.005, t_max=5.0,
                                         data_stimuli=ds, solver="bwd_euler"))
        return jax.jit(fwd)

    for dev, name in [(cpu, 'CPU'), (metal, 'METAL')]:
        try:
            with jax.default_device(dev):
                fwd = make_mrg_forward()
                t0 = time.time(); v1 = float(fwd(1.0)); t1 = time.time() - t0
                t0 = time.time(); v2 = float(fwd(1.2)); t2 = time.time() - t0
            print(f"  {name:6s}  compile+run1={t1:.3f}s  run2={t2:.4f}s  peak={v1:.2f}", flush=True)
        except Exception as e:
            print(f"  {name:6s}  FAIL: {type(e).__name__}: {e}", flush=True)

    # --- 4. gradient through MRG fiber forward
    banner("4. gradient through MRG fiber forward")
    for dev, name in [(cpu, 'CPU'), (metal, 'METAL')]:
        try:
            with jax.default_device(dev):
                fwd = make_mrg_forward()
                gfwd = jax.jit(jax.grad(fwd))
                t0 = time.time(); g1 = float(gfwd(1.0)); t1 = time.time() - t0
                t0 = time.time(); g2 = float(gfwd(1.2)); t2 = time.time() - t0
            print(f"  {name:6s}  compile+run1={t1:.3f}s  run2={t2:.4f}s  grad={g1:+.3e}", flush=True)
        except Exception as e:
            print(f"  {name:6s}  FAIL: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
