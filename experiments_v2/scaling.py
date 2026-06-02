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
from jaxfibers.fibers.schild  import (
    build_schild94, node_indices as schild94_node_indices,
    build_schild97, node_indices as schild97_node_indices,
)
from jaxfibers.stim.intracellular import rectangular_pulse, attach_intra_pulse
from jaxfibers.nrn_baseline  import (
    run_intracellular, run_intracellular_sundt,
    run_intracellular_rattay, run_intracellular_sweeney,
    run_intracellular_mrg_interp,
    run_intracellular_schild94,
    run_intracellular_schild97,
)

from experiments_v2.utils import ensure_dir, save_json

OUT = ensure_dir(ROOT / "outputs" / "scaling")

# ── timing budget for PyFibers serial ────────────────────────────────────────
# Stop adding N-values once cumulative time exceeds this; extrapolate instead.
# 2 hours covers full N=10000 PyFibers serial for every model in the registry
# on the MedUni Vienna cluster (Sundt/Rattay ~30 min, Schild94/97 ~1 hour
# each). Set via JAXLEY_FIBERS_PF_BUDGET_S env var to override per-run, e.g.
# `JAXLEY_FIBERS_PF_BUDGET_S=300 python experiments_v2/scaling.py` for a fast
# local smoke run with extrapolation past N=100.
PYFIBERS_BUDGET_S = float(os.environ.get("JAXLEY_FIBERS_PF_BUDGET_S", 7200.0))

N_FIBERS = [1, 10, 100, 1000, 10000]


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
    "Schild94": ModelConfig(
        key="Schild94",
        name="Schild 1994 C-fiber (D=0.8 µm, N=51 compartments)",
        diameter=0.8,
        n_nodes=51,
        dt_ms=0.005,
        tstop_ms=10.0,
        i_amp_nA=0.5,
        i_delay_ms=1.0,
        i_dur_ms=0.5,
        build_fn=build_schild94,
        node_idx_fn=schild94_node_indices,
        pf_run_fn=run_intracellular_schild94,
        pf_kwargs=dict(diameter=0.8, n_nodes=51, i_amp_nA=0.5,
                       i_delay_ms=1.0, i_dur_ms=0.5, dt_ms=0.005, tstop_ms=10.0),
    ),
    "Schild97": ModelConfig(
        key="Schild97",
        name="Schild 1997 C-fiber (D=0.8 µm, N=51 compartments)",
        diameter=0.8,
        n_nodes=51,
        dt_ms=0.005,
        tstop_ms=10.0,
        i_amp_nA=0.5,
        i_delay_ms=1.0,
        i_dur_ms=0.5,
        build_fn=build_schild97,
        node_idx_fn=schild97_node_indices,
        pf_run_fn=run_intracellular_schild97,
        pf_kwargs=dict(diameter=0.8, n_nodes=51, i_amp_nA=0.5,
                       i_delay_ms=1.0, i_dur_ms=0.5, dt_ms=0.005, tstop_ms=10.0),
    ),
}


# ── PyFibers serial timing ────────────────────────────────────────────────────

def time_pyfibers(cfg: ModelConfig, n: int) -> float:
    t0 = time.time()
    for _ in range(n):
        cfg.pf_run_fn(**cfg.pf_kwargs)
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
    print("\n--- PyFibers (CPU, serial) ---")
    cumulative = 0.0
    for n in N_FIBERS:
        if cumulative > PYFIBERS_BUDGET_S:
            print(f"  n = {n} ... (budget exceeded, extrapolated)")
            break
        t = time_pyfibers(cfg, n)
        cumulative += t
        results["pyfibers"][n] = t
        print(f"  n = {n:6d} ... {t:.2f} s")

    # ── Jaxley CPU ────────────────────────────────────────────────────────────
    print("\n--- Jaxley (CPU, vmap) ---")
    for n in N_FIBERS:
        compile_s, run_s = time_jaxley(cfg, n, "cpu")
        results["jaxley_cpu"]["compile"][n] = compile_s
        results["jaxley_cpu"]["run"][n]     = run_s
        print(f"  n = {n:6d} ... compile={compile_s:.2f} s | run={run_s:.3f} s")

    # ── Jaxley GPU ────────────────────────────────────────────────────────────
    if has_gpu:
        print("\n--- Jaxley (GPU, vmap) ---")
        for n in N_FIBERS:
            compile_s, run_s = time_jaxley(cfg, n, "gpu")
            results["jaxley_gpu"]["compile"][n] = compile_s
            results["jaxley_gpu"]["run"][n]     = run_s
            print(f"  n = {n:6d} ... compile={compile_s:.2f} s | run={run_s:.3f} s")
    else:
        print("\n[GPU not visible to JAX — skipping GPU timing]")

    return results


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(all_results: list[dict]) -> None:
    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5),
                             constrained_layout=True, sharey=False)
    if n_models == 1:
        axes = [axes]

    for ax, res in zip(axes, all_results):
        has_gpu = res["jaxley_gpu"] is not None

        # PyFibers
        pf_ns = sorted(res["pyfibers"].keys())
        pf_ts = [res["pyfibers"][n] for n in pf_ns]
        ax.loglog(pf_ns, pf_ts, "o-", color="C3", lw=2.0, ms=6, label="PyFibers (CPU serial)")

        # Jaxley CPU run
        cpu_ns = sorted(res["jaxley_cpu"]["run"].keys())
        cpu_ts = [res["jaxley_cpu"]["run"][n] for n in cpu_ns]
        ax.loglog(cpu_ns, cpu_ts, "s--", color="C0", lw=1.5, ms=5, label="Jaxley CPU (vmap)")

        # Jaxley GPU run
        if has_gpu:
            gpu_ns = sorted(res["jaxley_gpu"]["run"].keys())
            gpu_ts = [res["jaxley_gpu"]["run"][n] for n in gpu_ns]
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

    save_json({"models": all_results}, OUT / "data_scaling.json")
    make_figure(all_results)
    print("\nDone.")
