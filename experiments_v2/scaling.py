"""Computational scaling benchmark — v2.

Compares wall-clock time for N independent fibers simulated via:
  * PyFibers (CPU, serial loop)
  * Jaxley   (CPU, batched vmap)
  * Jaxley   (GPU, batched vmap)

Model registry makes it trivial to add future fiber models.

Outputs (outputs/scaling/):
  fig_scaling.png
  data_scaling.json    ← all timing data for downstream postprocessing

Run from project root:
    python experiments_v2/scaling.py
"""

from __future__ import annotations

import os
import sys
import pathlib
import time
import json
from dataclasses import dataclass, field
from typing import Callable

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jaxley as jx

jax.config.update("jax_enable_x64", True)

from jaxfibers.fibers.mrg     import build_mrg,     node_indices as mrg_node_indices
from jaxfibers.fibers.mrg     import build_mrg_interp, node_indices as mrg_interp_node_indices
from jaxfibers.fibers.sundt   import build_sundt,   node_indices as sundt_node_indices
from jaxfibers.fibers.rattay  import build_rattay,  node_indices as rattay_node_indices
from jaxfibers.fibers.sweeney import build_sweeney, node_indices as sweeney_node_indices
from jaxfibers.stim.intracellular import rectangular_pulse, attach_intra_pulse
from jaxfibers.nrn_baseline  import (
    run_intracellular, run_intracellular_sundt,
    run_intracellular_rattay, run_intracellular_sweeney,
    run_intracellular_mrg_interp,
)

from experiments_v2.utils import ensure_dir, save_json

OUT = ensure_dir(ROOT / "outputs" / "scaling")

# ── timing budget for PyFibers serial ────────────────────────────────────────
# Stop adding N-values once cumulative time exceeds this; extrapolate instead.
# 2 hours per model covers PyFibers serial up to N=10^5 for the myelinated
# models (MRG, Sweeney, MRG_Interp) and up to N=10^4 for the unmyelinated
# C-fibers (Sundt, Rattay), with the figure extrapolating above the cap.
# Set via JAXLEY_FIBERS_PF_BUDGET_S env var to override per-run, e.g.
# `JAXLEY_FIBERS_PF_BUDGET_S=300 python experiments_v2/scaling.py` for a fast
# local smoke run with extrapolation past N=100.
PYFIBERS_BUDGET_S = float(os.environ.get("JAXLEY_FIBERS_PF_BUDGET_S", 7200.0))

N_FIBERS = [1, 10, 100, 1000, 10000, 100000]

# Cap N for the JAX runs if a particular (model, N) combination OOMs the
# available GPU (a16 has 16 GB; heavier models with large state at N=10^5
# may not fit). Failures are caught and reported as 'oom' in the JSON.
JAX_OOM_FALLBACK = True


# ── Model registry ────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    key: str
    name: str
    # Jaxley
    diameter: float
    n_nodes: int
    dt_ms: float
    tstop_ms: float
    i_amp_nA: float
    i_delay_ms: float
    i_dur_ms: float
    build_fn: Callable       # (diameter, n_nodes) -> (cell, geom)
    node_idx_fn: Callable    # (geom) -> list[int]
    # PyFibers
    pf_run_fn: Callable      # (**kwargs) -> NrnRunResult
    pf_kwargs: dict = field(default_factory=dict)


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "MRG": ModelConfig(
        key="MRG",
        name="MRG myelinated (D=10 µm, N=11 nodes)",
        diameter=10.0,
        n_nodes=11,
        dt_ms=0.01,
        tstop_ms=3.0,
        i_amp_nA=1.0,
        i_delay_ms=1.0,
        i_dur_ms=0.1,
        build_fn=build_mrg,
        node_idx_fn=mrg_node_indices,
        pf_run_fn=run_intracellular,
        pf_kwargs=dict(diameter=10.0, n_nodes=11, i_amp_nA=1.0,
                       i_delay_ms=1.0, i_dur_ms=0.1, dt_ms=0.01, tstop_ms=3.0),
    ),
    "Sundt": ModelConfig(
        key="Sundt",
        name="Sundt C-fiber (D=0.8 µm, N=51 compartments)",
        diameter=0.8,
        n_nodes=51,
        dt_ms=0.005,
        tstop_ms=10.0,
        i_amp_nA=0.5,
        i_delay_ms=1.0,
        i_dur_ms=0.5,
        build_fn=build_sundt,
        node_idx_fn=sundt_node_indices,
        pf_run_fn=run_intracellular_sundt,
        pf_kwargs=dict(diameter=0.8, n_nodes=51, i_amp_nA=0.5,
                       i_delay_ms=1.0, i_dur_ms=0.5, dt_ms=0.005, tstop_ms=10.0),
    ),
    "Rattay": ModelConfig(
        key="Rattay",
        name="Rattay C-fiber (D=0.8 µm, N=51 compartments)",
        diameter=0.8,
        n_nodes=51,
        dt_ms=0.005,
        tstop_ms=10.0,
        i_amp_nA=0.5,
        i_delay_ms=1.0,
        i_dur_ms=0.2,
        build_fn=build_rattay,
        node_idx_fn=rattay_node_indices,
        pf_run_fn=run_intracellular_rattay,
        pf_kwargs=dict(diameter=0.8, n_nodes=51, i_amp_nA=0.5,
                       i_delay_ms=1.0, i_dur_ms=0.2, dt_ms=0.005, tstop_ms=10.0),
    ),
    "Sweeney": ModelConfig(
        key="Sweeney",
        name="Sweeney myelinated (D=10 µm, N=21 nodes)",
        diameter=10.0,
        n_nodes=21,
        dt_ms=0.005,
        tstop_ms=5.0,
        i_amp_nA=1.0,
        i_delay_ms=1.0,
        i_dur_ms=0.1,
        build_fn=build_sweeney,
        node_idx_fn=sweeney_node_indices,
        pf_run_fn=run_intracellular_sweeney,
        pf_kwargs=dict(diameter=10.0, n_nodes=21, i_amp_nA=1.0,
                       i_delay_ms=1.0, i_dur_ms=0.1, dt_ms=0.005, tstop_ms=5.0),
    ),
    "MRG_Interp": ModelConfig(
        key="MRG_Interp",
        name="MRG interpolated myelinated (D=10 µm, N=11 nodes)",
        diameter=10.0,
        n_nodes=11,
        dt_ms=0.01,
        tstop_ms=3.0,
        i_amp_nA=1.0,
        i_delay_ms=1.0,
        i_dur_ms=0.1,
        build_fn=build_mrg_interp,
        node_idx_fn=mrg_interp_node_indices,
        pf_run_fn=run_intracellular_mrg_interp,
        pf_kwargs=dict(diameter=10.0, n_nodes=11, i_amp_nA=1.0,
                       i_delay_ms=1.0, i_dur_ms=0.1, dt_ms=0.01, tstop_ms=3.0),
    ),
}


# ── PyFibers serial timing ────────────────────────────────────────────────────

def time_pyfibers(cfg: ModelConfig, n: int, progress_every: int | None = None) -> float:
    """Time n serial pyfibers single-fibre runs.

    Prints periodic progress so long runs (e.g. n=10,000 on slow C-fibre
    models) show heartbeat / ETA in the SLURM log rather than going dark
    for an hour at a time.  ``progress_every`` defaults to max(1, n//20)
    so you get ~20 progress lines per N regardless of how big N is.
    """
    if progress_every is None:
        progress_every = max(1, n // 20)
    t0 = time.time()
    for i in range(n):
        cfg.pf_run_fn(**cfg.pf_kwargs)
        # Print after the iter so i=0 doesn't fire (avoid printing before
        # any iter ran).  Always print the last iter as a sanity flush.
        if (i + 1) % progress_every == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            done    = i + 1
            rate    = done / elapsed if elapsed > 0 else 0
            remaining = (n - done) / rate if rate > 0 else float("inf")
            print(f"      [pyfibers] {done:6d}/{n}  "
                  f"elapsed={elapsed:7.1f}s  rate={rate:6.1f}/s  "
                  f"ETA={remaining:6.0f}s",
                  flush=True)
    return time.time() - t0


# ── Jaxley forward function builder ──────────────────────────────────────────

def build_jaxley_forward(cfg: ModelConfig):
    """Construct a single-fiber forward fn: amp_nA -> peak Vm.

    Returns (forward_fn, shape_array).
    The function is vmapped over a batch of amplitudes.
    """
    cell, geom = cfg.build_fn(cfg.diameter, cfg.n_nodes)
    nodes = cfg.node_idx_fn(geom)
    mid_comp = nodes[len(nodes) // 2]

    n_steps = int(cfg.tstop_ms / cfg.dt_ms) + 1
    t_arr   = np.arange(n_steps) * cfg.dt_ms
    shape   = jnp.asarray(rectangular_pulse(t_arr, cfg.i_delay_ms, cfg.i_dur_ms, amp_nA=1.0))

    cell.branch(0).comp(mid_comp).record("v")

    def _single(amp):
        pulse = amp * shape
        ds = cell.branch(0).comp(int(mid_comp)).data_stimulate(pulse)
        v  = jx.integrate(cell, delta_t=cfg.dt_ms, t_max=cfg.tstop_ms,
                          data_stimuli=ds, solver="bwd_euler")
        return jnp.max(v)

    return _single, shape


# ── Jaxley batched timing ─────────────────────────────────────────────────────

def time_jaxley(cfg: ModelConfig, n: int, device_str: str) -> tuple[float, float]:
    """Returns (compile_s, run_s).

    device_str: "cpu" or "gpu" — selects the JAX default device.
    """
    try:
        target_device = jax.devices(device_str)[0]
    except RuntimeError:
        raise RuntimeError(f"No JAX device matching '{device_str}' found")

    with jax.default_device(target_device):
        fwd, _ = build_jaxley_forward(cfg)
        batch_fwd = jax.jit(jax.vmap(fwd))   # jit is required — vmap alone runs eager

        amps = jnp.linspace(0.95 * cfg.i_amp_nA, 1.05 * cfg.i_amp_nA, n,
                            dtype=jnp.float64)

        t0 = time.time()
        result = jax.block_until_ready(batch_fwd(amps))
        compile_s = time.time() - t0

        t0 = time.time()
        result = jax.block_until_ready(batch_fwd(amps))
        run_s = time.time() - t0

    return compile_s, run_s


# ── Run one model ─────────────────────────────────────────────────────────────

def run_model(model_key: str) -> dict:
    cfg = MODEL_REGISTRY[model_key]
    print(f"\n{'='*60}")
    print(f"Model: {cfg.name}")
    print(f"{'='*60}")

    try:
        jax.devices("gpu")
        has_gpu = True
    except RuntimeError:
        has_gpu = False
    results = {
        "model": model_key,
        "name": cfg.name,
        "diameter": cfg.diameter,
        "n_nodes": cfg.n_nodes,
        "dt_ms": cfg.dt_ms,
        "tstop_ms": cfg.tstop_ms,
        "N": N_FIBERS,
        "pyfibers": {},
        "jaxley_cpu": {"compile": {}, "run": {}},
        "jaxley_gpu": {"compile": {}, "run": {}} if has_gpu else None,
    }

    # ── PyFibers serial ───────────────────────────────────────────────────────
    print("\n--- PyFibers (CPU, serial) ---", flush=True)
    cumulative = 0.0
    for n in N_FIBERS:
        if cumulative > PYFIBERS_BUDGET_S:
            print(f"  n = {n} ... (budget exceeded, extrapolated)", flush=True)
            break
        print(f"  n = {n:6d} starting (cumulative {cumulative:.0f}s of "
              f"{PYFIBERS_BUDGET_S:.0f}s budget) ...", flush=True)
        t = time_pyfibers(cfg, n)
        cumulative += t
        results["pyfibers"][n] = t
        print(f"  n = {n:6d} ... {t:.2f} s  (cumulative {cumulative:.0f}s)",
              flush=True)
        _checkpoint_save(model_key, results)

    # ── Jaxley CPU ────────────────────────────────────────────────────────────
    print("\n--- Jaxley (CPU, vmap) ---", flush=True)
    for n in N_FIBERS:
        print(f"  n = {n:6d} starting (jaxley CPU) ...", flush=True)
        try:
            compile_s, run_s = time_jaxley(cfg, n, "cpu")
        except Exception as e:
            msg = str(e).splitlines()[0] if str(e) else type(e).__name__
            print(f"  n = {n:6d} ... FAILED: {msg}", flush=True)
            results["jaxley_cpu"]["compile"][n] = None
            results["jaxley_cpu"]["run"][n]     = None
            _checkpoint_save(model_key, results)
            if JAX_OOM_FALLBACK:
                continue
            raise
        results["jaxley_cpu"]["compile"][n] = compile_s
        results["jaxley_cpu"]["run"][n]     = run_s
        print(f"  n = {n:6d} ... compile={compile_s:.2f} s | run={run_s:.3f} s",
              flush=True)
        _checkpoint_save(model_key, results)

    # ── Jaxley GPU ────────────────────────────────────────────────────────────
    if has_gpu:
        print("\n--- Jaxley (GPU, vmap) ---", flush=True)
        for n in N_FIBERS:
            print(f"  n = {n:6d} starting (jaxley GPU) ...", flush=True)
            try:
                compile_s, run_s = time_jaxley(cfg, n, "gpu")
            except Exception as e:
                msg = str(e).splitlines()[0] if str(e) else type(e).__name__
                print(f"  n = {n:6d} ... FAILED: {msg}", flush=True)
                results["jaxley_gpu"]["compile"][n] = None
                results["jaxley_gpu"]["run"][n]     = None
                _checkpoint_save(model_key, results)
                if JAX_OOM_FALLBACK:
                    continue
                raise
            results["jaxley_gpu"]["compile"][n] = compile_s
            results["jaxley_gpu"]["run"][n]     = run_s
            print(f"  n = {n:6d} ... compile={compile_s:.2f} s | run={run_s:.3f} s",
                  flush=True)
            _checkpoint_save(model_key, results)
    else:
        print("\n[GPU not visible to JAX — skipping GPU timing]", flush=True)

    return results


def _checkpoint_save(model_key: str, results: dict) -> None:
    """Write partial results for one model to disk so a job kill
    (walltime, OOM, drain) doesn't lose everything done so far.

    Drops a model-specific shard alongside the final aggregated JSON.
    The shard at outputs/scaling/data_scaling.<model>.partial.json is
    overwritten after every (N, method) tuple completes.  When the
    full run finishes successfully we still write data_scaling.json
    in the usual place; the partial shards are kept as audit trail.
    """
    try:
        path = OUT / f"data_scaling.{model_key}.partial.json"
        save_json(results, path)
    except Exception as e:
        print(f"  [checkpoint save failed: {e}]", flush=True)


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(all_results: list[dict]) -> None:
    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5),
                             constrained_layout=True, sharey=False)
    if n_models == 1:
        axes = [axes]

    def _drop_none(ns, ts):
        """Strip (N, t) pairs where t is None (OOM/budget-skip)."""
        return zip(*[(n, t) for n, t in zip(ns, ts) if t is not None]) if any(t is not None for t in ts) else ([], [])

    for ax, res in zip(axes, all_results):
        has_gpu = res["jaxley_gpu"] is not None

        # PyFibers
        pf_ns_all = sorted(res["pyfibers"].keys())
        pf_ts_all = [res["pyfibers"][n] for n in pf_ns_all]
        pf_ns, pf_ts = _drop_none(pf_ns_all, pf_ts_all)
        if pf_ns:
            ax.loglog(pf_ns, pf_ts, "o-", color="C3", lw=2.0, ms=6, label="PyFibers (CPU serial)")

        # Jaxley CPU run
        cpu_ns_all = sorted(res["jaxley_cpu"]["run"].keys())
        cpu_ts_all = [res["jaxley_cpu"]["run"][n] for n in cpu_ns_all]
        cpu_ns, cpu_ts = _drop_none(cpu_ns_all, cpu_ts_all)
        if cpu_ns:
            ax.loglog(cpu_ns, cpu_ts, "s--", color="C0", lw=1.5, ms=5, label="Jaxley CPU (vmap)")

        # Jaxley GPU run
        if has_gpu:
            gpu_ns_all = sorted(res["jaxley_gpu"]["run"].keys())
            gpu_ts_all = [res["jaxley_gpu"]["run"][n] for n in gpu_ns_all]
            gpu_ns, gpu_ts = _drop_none(gpu_ns_all, gpu_ts_all)
            if gpu_ns:
                ax.loglog(gpu_ns, gpu_ts, "^--", color="C2", lw=1.5, ms=5, label="Jaxley GPU (vmap)")

        ax.set_xlabel("N fibers")
        ax.set_ylabel("Wall-clock time (s)")
        ax.set_title(res["name"])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Scaling benchmark: PyFibers (serial) vs Jaxley (batched vmap)", fontsize=11)
    path = OUT / "fig_scaling.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  → {path}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")

    all_results = []
    for key in MODEL_REGISTRY:
        res = run_model(key)
        all_results.append(res)
        # Incremental write of the aggregated dataset after every model
        # so an interrupted run still leaves a usable data_scaling.json
        # behind for whatever has been completed.
        save_json({"models": all_results}, OUT / "data_scaling.json")
        print(f"\n  [checkpoint] data_scaling.json updated "
              f"({len(all_results)}/{len(MODEL_REGISTRY)} models)",
              flush=True)

    make_figure(all_results)
    print("\nDone.", flush=True)
