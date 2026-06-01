"""Schild 1994 C-fiber validation suite — v2.

Validates the JAX (Jaxley bwd_euler) implementation of the Schild 1994
C-fiber model against PyFibers/NEURON via intracellular stimulation.

Tasks
-----
1. Vm + key gate + Ca traces at D=0.8 µm, 51 compartments
   Compare: Jaxley bwd_euler vs PyFibers/NEURON

Outputs (outputs/schild94_validation/)
---------------------------------------
  data_schild94_traces.json
  fig_schild94_traces.png

Run from project root:
    python experiments_v2/schild94_validation.py
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

jax.config.update("jax_enable_x64", True)

from jaxfibers.fibers.schild import (
    build_schild94, node_indices, section_centers_um,
    V_REST,
)
from jaxfibers.stim.intracellular import rectangular_pulse, attach_intra_pulse
from jaxfibers.nrn_baseline import run_intracellular_schild94

from experiments_v2.utils import ensure_dir, save_json

OUT = ensure_dir(ROOT / "outputs" / "schild94_validation")

# ── constants ──────────────────────────────────────────────────────────────────
CELSIUS      = 37.0
DT           = 0.005    # ms
TSTOP        = 10.0     # ms
DELAY        = 1.0      # ms
N_NODES      = 51       # compartments
INTRA_AMP_NA = 0.5      # nA
INTRA_PW_MS  = 0.5      # ms  (wider pulse for C-fiber)
TRACE_DIAM   = 0.8      # µm

# Jaxley state names for SchildCombined94
CH = "SchildCombined94"
JAX_STATES = {
    "m_naf":  f"{CH}_m_naf",
    "h_naf":  f"{CH}_h_naf",
    "l_naf":  f"{CH}_l_naf",
    "m_nas":  f"{CH}_m_nas",
    "h_nas":  f"{CH}_h_nas",
    "n_kd":   f"{CH}_n_kd",
    "cai":    f"{CH}_cai",
    "cao":    f"{CH}_cao",
}


# ── Task 1: Intracellular traces ───────────────────────────────────────────────

def task_traces():
    print("\n=== Task 1: Intracellular traces (D=0.8 µm) ===")
    D = TRACE_DIAM

    n_steps = int(TSTOP / DT) + 1
    t_arr   = np.arange(n_steps) * DT
    pulse   = rectangular_pulse(t_arr, DELAY, INTRA_PW_MS, INTRA_AMP_NA)

    # ── JAX (Jaxley bwd_euler) ────────────────────────────────────────────────
    print("  [JAX] building cell ...", flush=True)
    cell, geom = build_schild94(diameter=D, n_nodes=N_NODES, temperature=CELSIUS)
    nodes    = node_indices(geom)
    mid_comp = nodes[len(nodes) // 2]

    attach_intra_pulse(cell, mid_comp, pulse)
    cell.branch(0).comp(mid_comp).record("v")
    for jax_key in JAX_STATES.values():
        cell.branch(0).comp(mid_comp).record(jax_key)

    print("  [JAX] integrating ...", flush=True)
    t0_jax = time.time()
    rec    = np.asarray(jx.integrate(cell, delta_t=DT, t_max=TSTOP, solver="bwd_euler"))
    print(f"  [JAX] done in {time.time() - t0_jax:.1f} s", flush=True)

    jax_vm   = rec[0]
    jax_gate = {k: rec[i + 1] for i, k in enumerate(JAX_STATES)}

    # ── PyFibers/NEURON ───────────────────────────────────────────────────────
    print("  [PyFibers] running NEURON ...", flush=True)
    t0_pf = time.time()
    res   = run_intracellular_schild94(
        diameter=D, n_nodes=N_NODES, temperature=CELSIUS,
        i_delay_ms=DELAY, i_dur_ms=INTRA_PW_MS, i_amp_nA=INTRA_AMP_NA,
        dt_ms=DT, tstop_ms=TSTOP,
    )
    print(f"  [PyFibers] done in {time.time() - t0_pf:.1f} s | APs={res.n_aps_at_probe}",
          flush=True)

    pf_vm  = res.vm_mV[res.probe_node_idx]
    t_pf   = res.t_ms
    pf_gates = res.gates
    print(f"  PyFibers gates available: {sorted(pf_gates.keys())}")

    # Interpolate JAX to PyFibers time grid for RMSE
    vm_jax_on_pf = np.interp(t_pf, t_arr, jax_vm)
    rmse = float(np.sqrt(np.mean((vm_jax_on_pf - pf_vm)**2)))
    peak_jax = float(np.max(jax_vm))
    peak_pf  = float(np.max(pf_vm))
    print(f"  RMSE = {rmse:.3f} mV | peak JAX={peak_jax:.1f} mV | peak PF={peak_pf:.1f} mV")

    data = dict(
        diameter=D, celsius=CELSIUS,
        t_jax=t_arr.tolist(), vm_jax=jax_vm.tolist(),
        jax_gates={k: v.tolist() for k, v in jax_gate.items()},
        t_pf=t_pf.tolist(), vm_pf=pf_vm.tolist(),
        pf_gates={k: v.tolist() for k, v in pf_gates.items()},
        rmse_mV=rmse, peak_jax_mV=peak_jax, peak_pf_mV=peak_pf,
        n_aps_pf=res.n_aps_at_probe,
    )
    save_json(data, OUT / "data_schild94_traces.json")
    return data


# ── Figure ─────────────────────────────────────────────────────────────────────

def make_figure(data: dict):
    t_jax = np.array(data["t_jax"])
    t_pf  = np.array(data["t_pf"])
    vm_jax = np.array(data["vm_jax"])
    vm_pf  = np.array(data["vm_pf"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    # (A) Vm traces
    ax = axes[0, 0]
    ax.plot(t_pf,  vm_pf,  "k-",  lw=1.8, label="PyFibers/NEURON")
    ax.plot(t_jax, vm_jax, "r--", lw=1.2, label=f"JAX (RMSE={data['rmse_mV']:.2f} mV)")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("Vm (mV)")
    ax.set_title(f"(A) Vm  D={data['diameter']} µm")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (B) Fast Na gates (m_naf, h_naf, l_naf)
    ax = axes[0, 1]
    jax_gates = data["jax_gates"]
    pf_gates  = data["pf_gates"]
    colors = {"m_naf": "C0", "h_naf": "C2", "l_naf": "C4"}
    for gk, c in colors.items():
        if gk in jax_gates:
            ax.plot(t_jax, jax_gates[gk], "--", color=c, lw=1.0, label=f"JAX {gk}")
    for pf_key, pf_arr in pf_gates.items():
        c = next((v for k, v in colors.items() if k.split("_")[0] in pf_key), "gray")
        ax.plot(t_pf, pf_arr, "-", color=c, lw=1.5, alpha=0.7, label=f"PF {pf_key}")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("gate")
    ax.set_title("(B) naf gates  (m, h, l)")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)

    # (C) Slow Na and K gates
    ax = axes[0, 2]
    for gk, c in [("m_nas", "C1"), ("h_nas", "C3"), ("n_kd", "C5")]:
        if gk in jax_gates:
            ax.plot(t_jax, jax_gates[gk], "--", color=c, lw=1.0, label=f"JAX {gk}")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("gate")
    ax.set_title("(C) nas / kd gates  (JAX only)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (D) cai
    ax = axes[1, 0]
    if "cai" in jax_gates:
        ax.plot(t_jax, np.array(jax_gates["cai"]) * 1e3, "r--", lw=1.2, label="JAX cai")
    for pf_key, pf_arr in pf_gates.items():
        if "cai" in pf_key.lower():
            ax.plot(t_pf, np.array(pf_arr) * 1e3, "k-", lw=1.5, label=f"PF {pf_key}")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("cai (µM)")
    ax.set_title("(D) Intracellular Ca²⁺")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (E) cao
    ax = axes[1, 1]
    if "cao" in jax_gates:
        ax.plot(t_jax, jax_gates["cao"], "r--", lw=1.2, label="JAX cao")
    for pf_key, pf_arr in pf_gates.items():
        if "cao" in pf_key.lower():
            ax.plot(t_pf, pf_arr, "k-", lw=1.5, label=f"PF {pf_key}")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("cao (mM)")
    ax.set_title("(E) Periaxonal Ca²⁺")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (F) Summary text
    ax = axes[1, 2]
    lines = [
        f"Model: Schild 1994 C-fiber",
        f"Diameter: {data['diameter']} µm",
        f"N_comp: {N_NODES}",
        f"DT: {DT} ms | TSTOP: {TSTOP} ms",
        f"Stim: {INTRA_AMP_NA} nA × {INTRA_PW_MS} ms",
        f"",
        f"Vm RMSE: {data['rmse_mV']:.3f} mV",
        f"Peak JAX: {data['peak_jax_mV']:.1f} mV",
        f"Peak PF: {data['peak_pf_mV']:.1f} mV",
        f"PF APs detected: {data['n_aps_pf']}",
        f"",
        f"PF gates: {sorted(data['pf_gates'].keys())}",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=8, fontfamily="monospace")
    ax.set_axis_off()
    ax.set_title("(F) Summary")

    fig.suptitle(
        f"Schild 1994 C-fiber — JAX (Jaxley bwd_euler) vs PyFibers/NEURON\n"
        f"D={data['diameter']} µm, {N_NODES} compartments, T={CELSIUS}°C",
        fontsize=11,
    )
    path = OUT / "fig_schild94_traces.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")
    t0 = time.time()

    data = task_traces()
    make_figure(data)

    print(f"\nDone. Total wall time: {(time.time() - t0) / 60:.1f} min")
