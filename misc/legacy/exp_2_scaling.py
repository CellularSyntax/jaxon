"""Experiment 2: Computational scaling benchmark.

Simulate N independent 10 µm MRG fibers (each receiving a slightly different
intracellular pulse), N ∈ {1, 10, 100, 1000, 10000}. Compare wall-clock
execution time:
  * PyFibers (CPU, serial)               — one fiber at a time.
  * Jaxley   (CPU, batched via `jax.vmap`).
  * Jaxley   (GPU, batched via `jax.vmap`) — only if a CUDA / METAL device
    is visible to JAX. On a CUDA host the GPU line appears automatically;
    on an Apple-silicon host with `jax-metal` installed it is reported
    "best-effort" (jax-metal 0.1.1 is incompatible with jax >= 0.5).
    Otherwise the figure annotates the missing GPU line.

Produces:
  outputs/fig2_scaling.png   — log-log plot.

Note: The PyFibers "serial" line caps at N where wall-clock < ~120 s; larger N
is extrapolated linearly on a log-log fit and drawn dashed.

Run from project root:
    conda run -n jaxley_fibers python experiments/exp_2_scaling.py
"""

from __future__ import annotations

import sys, pathlib, time
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jaxley as jx

from jaxfibers.fibers.mrg import build_mrg, node_indices
from jaxfibers.stim.intracellular import rectangular_pulse, attach_intra_pulse
from jaxfibers.nrn_baseline import run_intracellular

OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

DIAM = 10.0
N_NODES = 11
TSTOP_MS = 3.0
DT_MS = 0.01
I_AMP_NA = 1.0
I_DELAY_MS = 1.0
I_DUR_MS = 0.1

# Caps to keep runtime sane on this machine:
PYFIBERS_MAX_N = 100         # PyFibers loop time scales linearly with N
JAXLEY_BATCH_NS = [1, 10, 100, 1000, 10000]  # 10000 fine on A16 (16 GB); may OOM on M1


# ------------------------------------------------------------------ #
# PyFibers serial timing                                              #
# ------------------------------------------------------------------ #

def time_pyfibers_serial(n: int) -> float:
    t0 = time.time()
    for _ in range(n):
        run_intracellular(
            diameter=DIAM, n_nodes=N_NODES, i_amp_nA=I_AMP_NA,
            i_dur_ms=I_DUR_MS, i_delay_ms=I_DELAY_MS,
            dt_ms=DT_MS, tstop_ms=TSTOP_MS,
        )
    return time.time() - t0


# ------------------------------------------------------------------ #
# Jaxley batched timing (single shared fiber, vmap over a per-fiber  #
# parameter — here the intracellular amplitude — for the scaling     #
# benchmark; each "fiber" has identical morphology + channel params, #
# differing only by its independent stimulus).                       #
# ------------------------------------------------------------------ #

def _build_jaxley_forward():
    """Construct the single-fiber forward fn (amp_nA -> peak Vm). Returns also (cell, mid_comp, shape)."""
    cell, geom = build_mrg(diameter=DIAM, n_nodes=N_NODES)
    nodes = node_indices(geom)
    mid_comp = nodes[len(nodes) // 2]
    n_steps = int(TSTOP_MS / DT_MS) + 1
    t = np.arange(n_steps) * DT_MS
    shape = jnp.asarray(rectangular_pulse(t, I_DELAY_MS, I_DUR_MS, amp_nA=1.0))
    cell.branch(0).comp(mid_comp).record("v")

    def _forward(amp):
        pulse = amp * shape                                 # (T,)
        ds = cell.branch(0).comp(int(mid_comp)).data_stimulate(pulse)
        v = jx.integrate(cell, delta_t=DT_MS, t_max=TSTOP_MS,
                         data_stimuli=ds, solver="bwd_euler")
        return jnp.max(v)

    return _forward, cell, mid_comp, shape


def time_jaxley_batched_cpu(n: int) -> tuple[float, float]:
    """Time Jaxley vmapped over `n` independent fibers (different stim amplitudes).

    Returns (compile_time_s, run_time_s).
    """
    forward, cell, mid_comp, _shape = _build_jaxley_forward()
    batched = jax.jit(jax.vmap(forward))
    # slightly different amps per "fiber" — keeps the work genuinely n-fold:
    amps = jnp.linspace(0.9 * I_AMP_NA, 1.1 * I_AMP_NA, n, dtype=jnp.float32)

    # warm-up / compile on actual shape:
    t0 = time.time()
    _ = batched(amps).block_until_ready()
    compile_t = time.time() - t0

    # second call: pure run time (graph already cached for this shape):
    t0 = time.time()
    out = batched(amps).block_until_ready()
    run_t = time.time() - t0
    return compile_t, run_t


def _gpu_devices():
    """Return list of JAX devices that are not the CPU (CUDA, ROCm, TPU, METAL...)."""
    return [d for d in jax.devices() if d.platform.lower() not in ("cpu", "tfrt_cpu")]


def time_jaxley_gpu(n: int) -> tuple[str | None, float, float]:
    """If any non-CPU device is visible, time the batched run on it.

    Returns:
        (platform_name_or_None, compile_time_s, run_time_s)
        platform_name is e.g. "CUDA", "ROCm", "METAL", or None if no GPU.
    """
    devs = _gpu_devices()
    if not devs:
        return None, 0.0, 0.0
    try:
        with jax.default_device(devs[0]):
            compile_t, run_t = time_jaxley_batched_cpu(n)
        return devs[0].platform.upper(), compile_t, run_t
    except Exception as e:
        print(f"  GPU attempt failed on {devs[0].platform}: {type(e).__name__}: {e}", flush=True)
        return None, 0.0, 0.0


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main():
    print(f"Jaxley devices: {jax.devices()}")

    print("\n--- PyFibers (CPU, serial) ---")
    pf_ns, pf_ts = [], []
    for n in [1, 10, 100]:
        if n > PYFIBERS_MAX_N:
            break
        print(f"  n = {n} ...", end=" ", flush=True)
        try:
            t = time_pyfibers_serial(n)
            print(f"{t:.2f} s")
            pf_ns.append(n); pf_ts.append(t)
        except Exception as e:
            print(f"FAILED {type(e).__name__}: {e}")
            break

    print("\n--- Jaxley (CPU, vmap) ---")
    jx_cpu_ns, jx_cpu_compile, jx_cpu_run = [], [], []
    for n in JAXLEY_BATCH_NS:
        print(f"  n = {n} ...", end=" ", flush=True)
        try:
            ct, rt = time_jaxley_batched_cpu(n)
            print(f"compile={ct:.2f} s | run={rt:.2f} s")
            jx_cpu_ns.append(n); jx_cpu_compile.append(ct); jx_cpu_run.append(rt)
        except Exception as e:
            print(f"FAILED {type(e).__name__}: {e}")
            break

    print("\n--- Jaxley (GPU, if visible) ---")
    jx_gpu_ns, jx_gpu_run, jx_gpu_platform = [], [], None
    for n in JAXLEY_BATCH_NS:
        print(f"  n = {n} ...", end=" ", flush=True)
        platform, ct, rt = time_jaxley_gpu(n)
        if platform is None:
            print("no GPU device visible to JAX; skipping further sizes.")
            break
        print(f"[{platform}]  compile={ct:.2f} s | run={rt:.2f} s")
        jx_gpu_ns.append(n); jx_gpu_run.append(rt); jx_gpu_platform = platform

    # ---------------------------------------------------------------- plot
    print("\nPlotting fig2 ...")
    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    if pf_ns:
        ax.loglog(pf_ns, pf_ts, "o-", color="C0", label="PyFibers (CPU serial)")
        # extrapolate (slope = 1 in log-log for serial)
        ratio = pf_ts[-1] / pf_ns[-1]
        ext_ns = [pf_ns[-1] * 10, pf_ns[-1] * 100]
        ax.loglog(ext_ns, [n * ratio for n in ext_ns], "o--", color="C0",
                  alpha=0.4, label="PyFibers (extrapolated, slope=1)")
    if jx_cpu_ns:
        ax.loglog(jx_cpu_ns, jx_cpu_run, "s-", color="C2", label="Jaxley (CPU vmap)")
    if jx_gpu_ns:
        ax.loglog(jx_gpu_ns, jx_gpu_run, "^-", color="C3",
                  label=f"Jaxley ({jx_gpu_platform} GPU)")
    else:
        import platform as _platform
        host_hint = f"this host: {_platform.system()} {_platform.machine()}"
        ax.text(0.97, 0.05,
                f"GPU line not shown: no GPU visible to JAX ({host_hint}).\n"
                "Install `jax[cuda12]` on a CUDA host and re-run to add this line.",
                ha="right", va="bottom", transform=ax.transAxes,
                fontsize=9, color="dimgray",
                bbox=dict(facecolor="white", edgecolor="lightgray", alpha=0.85))

    ax.set_xlabel("Number of fibers")
    ax.set_ylabel("Wall-clock execution time (s)")
    ax.set_title(f"Scaling: PyFibers vs Jaxley\n"
                 f"(MRG 10 µm, 11 nodes, intra 1 nA × 0.1 ms, {TSTOP_MS} ms sim, dt = {DT_MS} ms)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(OUT_DIR / "fig2_scaling.png", dpi=130)
    print(f"  -> {OUT_DIR/'fig2_scaling.png'}")


if __name__ == "__main__":
    main()
