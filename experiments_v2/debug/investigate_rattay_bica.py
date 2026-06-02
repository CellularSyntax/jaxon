"""Targeted investigation of Rattay bi_ca short-PW discrepancy.

The smoke test (smoke_test_sd_anomalies.py) compared JAX bi_ca vs PF mono_c
because pf_diagnostics always built a monophasic waveform.  This script uses
the correct bi_ca waveform in both models.

Investigation questions
-----------------------
1. Ve profile match: does PF compute the same extracellular potential per node?
2. With correct bi_ca waveform, do PF end nodes show the same anodal-break as JAX?
3. What are the actual JAX vs PF thresholds for bi_ca PW=0.05ms?
4. Does the discrepancy persist at PW=0.1ms and PW=0.2ms?

Run from project root:
    python experiments_v2/investigate_rattay_bica.py
"""

from __future__ import annotations
import sys, pathlib, dataclasses, time
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jaxley.solver_gate import solve_gate_exponential

jax.config.update("jax_enable_x64", True)

from jaxfibers.fibers.rattay import (
    build_rattay, node_indices, section_centers_um,
    V_REST, CM,
)
from jaxfibers.channels.rattay_channels import RattayHH
from jaxfibers.stim.extracellular import point_source_potentials_mV
from jaxfibers.stim.extracellular_coupled import arrays_from_geometry, integrate
from jaxfibers.nrn_baseline import build_rattay_pyfibers
from pyfibers.stimulation import ScaledStim

from experiments_v2.utils import (
    make_pulse_array, pulse_array_to_callable, PULSES,
    jax_bisect, pf_find_threshold, save_json,
)

OUT = pathlib.Path(ROOT / "outputs" / "investigate_rattay_bica")
OUT.mkdir(parents=True, exist_ok=True)

# ── shared constants ──────────────────────────────────────────────────────────
CELSIUS  = 37.0
DT       = 0.005   # ms
TSTOP    = 12.0    # ms
DELAY    = 1.0     # ms
N_NODES  = 51
N_STEPS  = int(TSTOP / DT)
SIGMA    = 0.3     # S/m
SRC_H    = 1000.0  # µm

GNABAR = 0.12; GKBAR = 0.036; GL = 3e-4
EL = -59.4; ENA = 45.0; EK = -82.0

D = 0.3   # µm — smallest diameter, largest discrepancy


# ── JAX setup ────────────────────────────────────────────────────────────────

def make_jax_setup(diameter: float = D):
    _, geom = build_rattay(diameter=diameter, n_nodes=N_NODES)
    nodes   = node_indices(geom)
    centers = np.array(section_centers_um(geom))
    mid     = nodes[len(nodes) // 2]

    geom_c = dataclasses.replace(geom, cm_uF_cm2=[CM] * geom.n_comp)
    static = arrays_from_geometry(geom_c, DT)

    Ve_unit = np.array(point_source_potentials_mV(
        list(centers), src_x_um=0., src_y_um=SRC_H,
        src_z_um=float(centers[mid]), i0_mA=-1.0,
    ))

    A_in = static["A_in_cm2"]
    is_node = static["is_node"]

    v0 = jnp.float64(V_REST)
    (am0, bm0), (ah0, bh0), (an0, bn0) = RattayHH._alpha_beta(v0, CELSIUS)
    m0 = float(am0 / (am0 + bm0))
    h0 = float(ah0 / (ah0 + bh0))
    n0 = float(an0 / (an0 + bn0))
    state0 = (
        jnp.full(geom.n_comp, m0, dtype=jnp.float64),
        jnp.full(geom.n_comp, h0, dtype=jnp.float64),
        jnp.full(geom.n_comp, n0, dtype=jnp.float64),
    )

    def membrane_fn(Vm, state, dt):
        M, H, N = state
        (am, bm), (ah, bh), (an, bn) = RattayHH._alpha_beta(Vm, CELSIUS)
        M2 = solve_gate_exponential(M, dt, am, bm)
        H2 = solve_gate_exponential(H, dt, ah, bh)
        N2 = solve_gate_exponential(N, dt, an, bn)
        g_na  = GNABAR * M2**3 * H2
        g_k   = GKBAR  * N2**4
        g_tot = (g_na + g_k + GL) * A_in * 1e6
        i_ion = (g_na*(Vm-ENA) + g_k*(Vm-EK) + GL*(Vm-EL)) * A_in * 1e6
        return g_tot, i_ion, (M2, H2, N2)

    return static, membrane_fn, state0, Ve_unit, nodes, centers, mid, geom


# ── JAX Vm trace helper ───────────────────────────────────────────────────────

def jax_vm_trace(static, mfn, state0, Ve_unit, amp_mA, pulse_arr):
    """Return (t_ms, Vm_nodes) where Vm_nodes is [n_steps, n_nodes]."""
    Ve_j = jnp.asarray(Ve_unit, dtype=jnp.float64)
    pm   = jnp.asarray(pulse_arr, dtype=jnp.float64)
    Ve   = Ve_j * (jnp.float64(amp_mA) / -1.0)

    (Vi_all, Vp_all), _ = integrate(static, mfn, state0, Ve, pm, DT,
                                    v_rest=V_REST, record="all")
    Vm_all = np.asarray(Vi_all - Vp_all)
    is_node = np.asarray(static["is_node"])
    node_idx = np.where(is_node)[0]
    Vm_nodes = Vm_all[:, node_idx]
    t_ms = (np.arange(Vm_all.shape[0]) + 1) * DT
    return t_ms, Vm_nodes, node_idx


# ── PF Vm trace helper ────────────────────────────────────────────────────────

def pf_vm_trace(amp_mA: float, pulse_arr: np.ndarray, diameter: float = D):
    """Run PF with the correct pulse waveform, return (t_ms, vm [n_nodes, n_t])."""
    fiber = build_rattay_pyfibers(diameter=diameter, n_nodes=N_NODES)
    fiber.record_vm()

    fiber.potentials = fiber.point_source_potentials(
        x=0.0, y=SRC_H, z=fiber.length / 2.0, i0=amp_mA, sigma=SIGMA,
    )
    wav = pulse_array_to_callable(pulse_arr, DT)
    stim = ScaledStim(waveform=wav, dt=DT, tstop=TSTOP)
    stim.run_sim(stimamp=1.0, fiber=fiber, ap_detect_location=0.5,
                 fail_on_end_excitation=False)

    vm = np.array([np.array(v) for v in fiber.vm])  # [n_nodes, n_t]
    t_ms = np.array(fiber.time)
    return t_ms, vm, fiber


# ── Ve profile comparison ─────────────────────────────────────────────────────

def compare_ve_profiles(static, Ve_unit, amp_mA: float, diameter: float = D):
    """Print JAX vs PF Ve at each node. Returns (jax_ve_nodes, pf_ve_nodes)."""
    _, geom = build_rattay(diameter=diameter, n_nodes=N_NODES)
    nodes = node_indices(geom)
    centers = np.array(section_centers_um(geom))
    mid = nodes[len(nodes) // 2]

    # JAX Ve at each compartment (already computed as Ve_unit * amp / -1)
    Ve_jax = np.array(Ve_unit) * (amp_mA / -1.0)  # mV at each comp
    is_node = np.asarray(static["is_node"])
    node_idx = np.where(is_node)[0]
    Ve_jax_nodes = Ve_jax[node_idx]

    # PF Ve: build fiber and compute potentials
    fiber = build_rattay_pyfibers(diameter=diameter, n_nodes=N_NODES)
    fiber.potentials = fiber.point_source_potentials(
        x=0.0, y=SRC_H, z=fiber.length / 2.0, i0=amp_mA, sigma=SIGMA,
    )
    Ve_pf_nodes = np.array(fiber.potentials)  # mV at each node

    print(f"\n{'─'*70}")
    print(f"Ve profile comparison at amp={amp_mA} mA (first/last 5 nodes + center)")
    print(f"{'node':>6}  {'JAX Ve (mV)':>13}  {'PF Ve (mV)':>12}  {'diff (mV)':>10}")
    print(f"{'─'*50}")
    show_idx = list(range(5)) + [mid // 1] + list(range(N_NODES - 5, N_NODES))
    # show_idx for nodes: use 0-4, center_node_local, 46-50
    center_node_local = int(np.searchsorted(node_idx, mid))
    show_node_local = list(range(5)) + [center_node_local] + list(range(N_NODES - 5, N_NODES))
    # deduplicate
    seen = set()
    show_node_local_uniq = [x for x in show_node_local if not (x in seen or seen.add(x))]

    for ni in show_node_local_uniq:
        if ni < len(Ve_jax_nodes) and ni < len(Ve_pf_nodes):
            diff = Ve_jax_nodes[ni] - Ve_pf_nodes[ni]
            print(f"  {ni:4d}  {Ve_jax_nodes[ni]:>13.4f}  {Ve_pf_nodes[ni]:>12.4f}  {diff:>10.4f}")

    max_diff = float(np.max(np.abs(Ve_jax_nodes - Ve_pf_nodes[:len(Ve_jax_nodes)])))
    rel_diff = max_diff / (np.abs(Ve_jax_nodes).mean() + 1e-9) * 100
    print(f"  max |diff| = {max_diff:.5f} mV  ({rel_diff:.2f}% of mean |Ve_jax|)")
    return Ve_jax_nodes, Ve_pf_nodes


# ── Bisection comparison ──────────────────────────────────────────────────────

def compare_thresholds(pw_ms: float, pulse_key: str = "bi_ca"):
    """Run JAX and PF bisection for given pulse, return (jax_thr, pf_thr)."""
    pulse_arr = make_pulse_array(pulse_key, pw_ms, N_STEPS, DT, DELAY)
    pm = jnp.asarray(pulse_arr, dtype=jnp.float64)

    static, mfn, state0, Ve_unit, nodes, centers, mid, geom = make_jax_setup()
    Ve_j = jnp.asarray(Ve_unit, dtype=jnp.float64)
    is_node = static["is_node"]

    @jax.jit
    def run(amp_mA, pulse_shape):
        Ve = Ve_j * (amp_mA / -1.0)
        (Vi_all, Vp_all), _ = integrate(static, mfn, state0, Ve, pulse_shape, DT,
                                        v_rest=V_REST, record="all")
        Vm_all   = Vi_all - Vp_all
        Vm_nodes = jnp.where(is_node[None, :], Vm_all, -jnp.inf)
        return jnp.max(Vm_nodes)

    spec = PULSES[pulse_key]
    print(f"  JAX bisect {pulse_key} PW={pw_ms}ms ...", flush=True)
    t0 = time.time()
    jax_thr = jax_bisect(run, pulse_arr, lo=spec.lo, hi=spec.hi, debug=True)
    print(f"  JAX threshold: {jax_thr:.4f} mA  ({time.time()-t0:.1f}s)", flush=True)

    print(f"  PF  bisect {pulse_key} PW={pw_ms}ms ...", flush=True)
    t0 = time.time()
    pf_thr = pf_find_threshold(
        "rattay", diameter=D, n_nodes=N_NODES,
        pulse_arr=pulse_arr, dt=DT, tstop=TSTOP,
        lo=spec.lo, hi=spec.hi,
    )
    print(f"  PF  threshold: {pf_thr:.4f} mA  ({time.time()-t0:.1f}s)", flush=True)

    err_pct = (jax_thr - pf_thr) / abs(pf_thr) * 100
    print(f"  Error (JAX-PF)/|PF| = {err_pct:+.1f}%", flush=True)
    return jax_thr, pf_thr, err_pct


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_vm_comparison(jax_t, jax_Vm_nodes, pf_t, pf_vm, pulse_arr,
                       amp_mA: float, pw_ms: float, tag: str):
    """Side-by-side Vm traces at end and center nodes for JAX and PF."""
    n_nodes = jax_Vm_nodes.shape[1]
    center_n = n_nodes // 2
    node_indices_to_plot = [0, 1, center_n, n_nodes - 2, n_nodes - 1]
    colors = ["tab:red", "tab:orange", "tab:green", "tab:blue", "tab:purple"]
    labels = [f"node 0 (end)", f"node 1", f"node {center_n} (ctr)",
              f"node {n_nodes-2}", f"node {n_nodes-1} (end)"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                             constrained_layout=True)

    # ── pulse waveform ────────────────────────────────────────────────────────
    t_pulse = (np.arange(len(pulse_arr)) + 1) * DT
    axes[0].plot(t_pulse, pulse_arr * amp_mA, "k-", lw=1.5)
    axes[0].axhline(0, color="gray", lw=0.5)
    axes[0].set_ylabel("Current (mA)")
    axes[0].set_title(f"Pulse: bi_ca PW={pw_ms}ms  amp={amp_mA:+.2f} mA")

    # ── JAX Vm ───────────────────────────────────────────────────────────────
    for i, (ni, col, lbl) in enumerate(zip(node_indices_to_plot, colors, labels)):
        axes[1].plot(jax_t, jax_Vm_nodes[:, ni], color=col, lw=1.5, label=lbl)
    axes[1].axhline(-30, color="k", lw=0.8, ls="--", label="AP thresh (-30 mV)")
    axes[1].axhline(V_REST, color="gray", lw=0.5, ls=":")
    axes[1].set_ylabel("Vm (mV)")
    axes[1].set_title(f"JAX — max over nodes: {jax_Vm_nodes.max():.1f} mV  "
                      f"fires={jax_Vm_nodes.max() > -30}")
    axes[1].legend(fontsize=7, loc="upper right", ncol=2)
    axes[1].set_ylim(-90, 70)

    # ── PF Vm ─────────────────────────────────────────────────────────────────
    n_pf_nodes = pf_vm.shape[0]
    pf_indices = [0, 1, n_pf_nodes // 2, n_pf_nodes - 2, n_pf_nodes - 1]
    for i, (ni, col, lbl) in enumerate(zip(pf_indices, colors, labels)):
        if ni < n_pf_nodes:
            axes[2].plot(pf_t, pf_vm[ni, :], color=col, lw=1.5, label=lbl)
    axes[2].axhline(-30, color="k", lw=0.8, ls="--", label="AP thresh (-30 mV)")
    axes[2].axhline(V_REST, color="gray", lw=0.5, ls=":")
    axes[2].set_ylabel("Vm (mV)")
    axes[2].set_xlabel("time (ms)")
    pf_max = pf_vm.max()
    axes[2].set_title(f"PF  — max over nodes: {pf_max:.1f} mV  fires={pf_max > -30}")
    axes[2].legend(fontsize=7, loc="upper right", ncol=2)
    axes[2].set_ylim(-90, 70)

    # zoom to first 5 ms
    axes[0].set_xlim(0, 5.0)

    path = OUT / f"vm_trace_{tag}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def plot_anodic_zoom(jax_t, jax_Vm_nodes, pf_t, pf_vm,
                     amp_mA: float, pw_ms: float, tag: str):
    """Zoom into anodic phase (1.0 + pw_ms to 1.0 + 2*pw_ms + 2ms) for end nodes."""
    n_jax = jax_Vm_nodes.shape[1]
    n_pf  = pf_vm.shape[0]
    t_start = DELAY + pw_ms - 0.05
    t_end   = DELAY + 2 * pw_ms + 2.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True,
                                   constrained_layout=True)
    end_nodes_jax = [0, 1, 2, n_jax - 3, n_jax - 2, n_jax - 1]
    end_nodes_pf  = [0, 1, 2, n_pf - 3, n_pf - 2, n_pf - 1]
    cmap = plt.cm.tab10

    for k, ni in enumerate(end_nodes_jax):
        mask = (jax_t >= t_start) & (jax_t <= t_end)
        ax1.plot(jax_t[mask], jax_Vm_nodes[mask, ni],
                 color=cmap(k), lw=1.5, label=f"node {ni}")
    ax1.axhline(-30, color="k", ls="--", lw=0.8, label="AP thresh")
    ax1.axvline(DELAY + pw_ms, color="gray", ls=":", lw=0.8, label="anodic start")
    ax1.axvline(DELAY + 2 * pw_ms, color="gray", ls="--", lw=0.8, label="anodic end")
    ax1.set_xlabel("time (ms)"); ax1.set_ylabel("Vm (mV)")
    ax1.set_title(f"JAX end nodes — amp={amp_mA:+.2f} mA PW={pw_ms}ms")
    ax1.legend(fontsize=8)

    for k, ni in enumerate(end_nodes_pf):
        if ni < n_pf:
            mask = (pf_t >= t_start) & (pf_t <= t_end)
            ax2.plot(pf_t[mask], pf_vm[ni, mask],
                     color=cmap(k), lw=1.5, label=f"node {ni}")
    ax2.axhline(-30, color="k", ls="--", lw=0.8, label="AP thresh")
    ax2.axvline(DELAY + pw_ms, color="gray", ls=":", lw=0.8, label="anodic start")
    ax2.axvline(DELAY + 2 * pw_ms, color="gray", ls="--", lw=0.8, label="anodic end")
    ax2.set_xlabel("time (ms)")
    ax2.set_title(f"PF  end nodes — amp={amp_mA:+.2f} mA PW={pw_ms}ms")
    ax2.legend(fontsize=8)

    path = OUT / f"anodic_zoom_{tag}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    results = {}

    print("\n" + "=" * 70)
    print("  Rattay bi_ca investigation — D=0.3 µm, N=51 nodes")
    print("=" * 70)

    static, mfn, state0, Ve_unit, nodes, centers, mid, geom = make_jax_setup()

    # ── Step 1: Ve profile match ───────────────────────────────────────────────
    print("\n[1] Comparing Ve profiles at -85.8 mA ...", flush=True)
    amp_probe = -85.8
    Ve_jax_nodes, Ve_pf_nodes = compare_ve_profiles(static, Ve_unit, amp_probe)
    results["ve_max_diff_mV"] = float(np.max(np.abs(Ve_jax_nodes - Ve_pf_nodes[:len(Ve_jax_nodes)])))

    # ── Step 2: Vm traces at -85.8 mA with CORRECT bi_ca waveform ─────────────
    print("\n[2] Vm traces at -85.8 mA with bi_ca PW=0.05ms (correct waveform) ...", flush=True)
    pw = 0.05
    pulse_arr = make_pulse_array("bi_ca", pw, N_STEPS, DT, DELAY)

    print("  JAX ...", flush=True)
    t0 = time.time()
    jax_t, jax_Vm_nodes, node_idx = jax_vm_trace(static, mfn, state0, Ve_unit, amp_probe, pulse_arr)
    print(f"  JAX done ({time.time()-t0:.1f}s) — max Vm = {jax_Vm_nodes.max():.2f} mV  "
          f"fires={jax_Vm_nodes.max() > -30}  "
          f"end0={jax_Vm_nodes[:, 0].max():.2f}  end-1={jax_Vm_nodes[:, -1].max():.2f}", flush=True)

    print("  PF (bi_ca waveform) ...", flush=True)
    t0 = time.time()
    pf_t, pf_vm, fiber = pf_vm_trace(amp_probe, pulse_arr)
    pf_fires = pf_vm.max() > -30
    print(f"  PF  done ({time.time()-t0:.1f}s) — max Vm = {pf_vm.max():.2f} mV  "
          f"fires={pf_fires}  "
          f"end0={pf_vm[0, :].max():.2f}  end-1={pf_vm[-1, :].max():.2f}", flush=True)

    # record any APs in PF
    n_aps_per_node = [int(fiber.apc[i].n) for i in range(fiber.nodecount)]
    print(f"  PF APs per node: end0={n_aps_per_node[0]}  ctr={n_aps_per_node[N_NODES//2]}  "
          f"end-1={n_aps_per_node[-1]}  any={sum(n_aps_per_node)}", flush=True)

    results["amp_probe_mA"] = amp_probe
    results["jax_maxVm_m85"] = float(jax_Vm_nodes.max())
    results["pf_maxVm_m85"]  = float(pf_vm.max())
    results["jax_end0_m85"]  = float(jax_Vm_nodes[:, 0].max())
    results["pf_end0_m85"]   = float(pf_vm[0, :].max())
    results["jax_fires_m85"] = bool(jax_Vm_nodes.max() > -30)
    results["pf_fires_m85"]  = bool(pf_fires)

    plot_vm_comparison(jax_t, jax_Vm_nodes, pf_t, pf_vm, pulse_arr,
                       amp_probe, pw, tag="m858_bica_005")
    plot_anodic_zoom(jax_t, jax_Vm_nodes, pf_t, pf_vm,
                     amp_probe, pw, tag="m858_bica_005")

    # ── Step 3: Scan over amplitudes (bracket threshold) ──────────────────────
    print("\n[3] Scanning amplitudes for bi_ca PW=0.05ms ...", flush=True)
    scan_amps = [-30.0, -50.0, -70.0, -85.8, -100.0, -120.0, -150.0, -200.0]
    scan_results = []
    for amp in scan_amps:
        _, Vm_jax_n, _ = jax_vm_trace(static, mfn, state0, Ve_unit, amp, pulse_arr)
        _, vm_pf, fib = pf_vm_trace(amp, pulse_arr)
        jax_max = float(Vm_jax_n.max())
        pf_max  = float(vm_pf.max())
        pf_aps  = sum(int(fib.apc[i].n) for i in range(fib.nodecount))
        scan_results.append({
            "amp": amp,
            "jax_max": jax_max,
            "pf_max": pf_max,
            "jax_fires": jax_max > -30,
            "pf_fires": pf_max > -30,
            "pf_total_aps": pf_aps,
        })
        print(f"  {amp:+7.1f} mA | JAX max={jax_max:>6.1f} mV fires={jax_max>-30} | "
              f"PF max={pf_max:>6.1f} mV fires={pf_max>-30} APs={pf_aps}", flush=True)
    results["scan_005ms"] = scan_results

    # ── Step 4: Repeat at PW=0.2ms (known good agreement) ────────────────────
    print("\n[4] Scanning amplitudes for bi_ca PW=0.20ms (control) ...", flush=True)
    pw_ctrl = 0.20
    pulse_ctrl = make_pulse_array("bi_ca", pw_ctrl, N_STEPS, DT, DELAY)
    scan_ctrl = []
    for amp in [-5.0, -8.0, -10.0, -12.0, -15.0, -20.0]:
        _, Vm_jax_n, _ = jax_vm_trace(static, mfn, state0, Ve_unit, amp, pulse_ctrl)
        _, vm_pf, fib = pf_vm_trace(amp, pulse_ctrl)
        jax_max = float(Vm_jax_n.max())
        pf_max  = float(vm_pf.max())
        pf_aps  = sum(int(fib.apc[i].n) for i in range(fib.nodecount))
        scan_ctrl.append({
            "amp": amp,
            "jax_max": jax_max,
            "pf_max": pf_max,
            "jax_fires": jax_max > -30,
            "pf_fires": pf_max > -30,
        })
        print(f"  {amp:+7.1f} mA | JAX max={jax_max:>6.1f} mV fires={jax_max>-30} | "
              f"PF max={pf_max:>6.1f} mV fires={pf_max>-30} APs={pf_aps}", flush=True)
    results["scan_020ms_ctrl"] = scan_ctrl

    # ── Step 5: Proper threshold bisections ───────────────────────────────────
    print("\n[5] Threshold bisections (bi_ca, two PWs) ...", flush=True)
    thr_results = {}
    for pw_ms in [0.05, 0.20]:
        print(f"\n  PW = {pw_ms} ms:", flush=True)
        jax_thr, pf_thr, err = compare_thresholds(pw_ms, "bi_ca")
        thr_results[f"pw_{pw_ms:.2f}".replace(".", "_")] = {
            "pw_ms": pw_ms, "jax": jax_thr, "pf": pf_thr, "err_pct": err,
        }
    results["thresholds"] = thr_results

    # ── Step 6: mono_c control thresholds ─────────────────────────────────────
    print("\n[6] Control: mono_c thresholds at PW=0.05ms ...", flush=True)
    jax_mono, pf_mono, err_mono = compare_thresholds(0.05, "mono_c")
    results["mono_c_005ms"] = {"jax": jax_mono, "pf": pf_mono, "err_pct": err_mono}

    # ── Save ──────────────────────────────────────────────────────────────────
    save_json(results, OUT / "investigation_results.json")

    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Ve profile max diff:  {results['ve_max_diff_mV']:.5f} mV")
    print(f"\nAt amp={amp_probe} mA, bi_ca PW=0.05ms:")
    print(f"  JAX max Vm = {results['jax_maxVm_m85']:.2f} mV  fires={results['jax_fires_m85']}")
    print(f"  PF  max Vm = {results['pf_maxVm_m85']:.2f} mV  fires={results['pf_fires_m85']}")
    print(f"\nThresholds:")
    for k, v in thr_results.items():
        print(f"  bi_ca PW={v['pw_ms']}ms:  JAX={v['jax']:.4f}  PF={v['pf']:.4f}  "
              f"err={v['err_pct']:+.1f}%")
    print(f"  mono_c PW=0.05ms: JAX={jax_mono:.4f}  PF={pf_mono:.4f}  "
          f"err={err_mono:+.1f}%")
    print(f"\nOutputs in: {OUT}")


if __name__ == "__main__":
    main()
