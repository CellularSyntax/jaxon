"""MRG_INTERPOLATION fiber validation suite — v2.

Validates the JAX coupled (Vi, Vpax) solver for the MRG_INTERPOLATION model
(polynomial-interpolated geometry, valid 2-16 µm) against PyFibers/NEURON.

Uses the identical double-cable solver and AxnodeMyel channel as MRG_DISCRETE;
only the geometry parameters differ (continuous polynomials vs. lookup table).

Tasks
-----
1. Vm + gate traces (intracellular) at D=10 µm
2. Strength-duration curves — 4 diameters × 8 pulse shapes × 6 PWs
3. Conduction velocity — 4 diameters

Outputs (outputs/mrg_interp_validation/)
------------------------------------------
  data_mrg_interp_traces.json
  data_mrg_interp_sd.json
  data_mrg_interp_cv.json
  fig_mrg_interp_traces.png
  fig_mrg_interp_sd_curves.png

Run from project root:
    python experiments_v2/mrg_interp_validation.py
"""

from __future__ import annotations

import sys, pathlib, time, dataclasses
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import jax
import jax.numpy as jnp
import jaxley as jx
from jaxley.solver_gate import solve_gate_exponential

jax.config.update("jax_enable_x64", True)

from jaxfibers.fibers.mrg import (
    build_mrg_interp, node_indices, section_centers_um,
    V_REST, CM_AXON,
)
from jaxfibers.channels.mrg_axnode import AxnodeMyel
from jaxfibers.stim.extracellular import point_source_potentials_mV
from jaxfibers.stim.intracellular import rectangular_pulse, attach_intra_pulse
from jaxfibers.nrn_baseline import run_intracellular_mrg_interp
from jaxfibers.stim.extracellular_coupled import arrays_from_geometry, integrate

from experiments_v2.utils import (
    PULSES, make_pulse_array, pf_find_threshold,
    jax_bisect, ensure_dir, save_json, GROUP_COLORS,
)

OUT = ensure_dir(ROOT / "outputs" / "mrg_interp_validation")

# ── constants ─────────────────────────────────────────────────────────────────
CELSIUS  = 37.0
DT       = 0.01       # ms  (MRG default)
TSTOP    = 5.0        # ms
DELAY    = 1.0        # ms
N_NODES  = 11         # compartments per fiber
SIGMA    = 0.3        # S/m
SRC_H    = 1000.0     # µm  point-source height

DIAMETERS  = [4.0, 7.0, 10.0, 14.0]      # µm  (all within [2,16] interpolation range)
SD_PWS     = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]   # ms
PULSE_KEYS = list(PULSES.keys())

TRACE_DIAM   = 10.0   # µm
INTRA_AMP_NA = 1.0
INTRA_PW_MS  = 0.1


# ── JAX coupled-solver setup ──────────────────────────────────────────────────

def _make_jax_setup(D: float):
    cell, geom = build_mrg_interp(diameter=D, n_nodes=N_NODES, temperature=CELSIUS)
    nodes   = node_indices(geom)
    centers = np.array(section_centers_um(geom))
    mid     = nodes[len(nodes) // 2]
    n_comp  = geom.n_comp

    geom_c = dataclasses.replace(geom, cm_uF_cm2=[CM_AXON] * n_comp)
    static  = arrays_from_geometry(geom_c, DT)

    Ve_unit = np.array(point_source_potentials_mV(
        list(centers), src_x_um=0., src_y_um=SRC_H,
        src_z_um=float(centers[mid]), i0_mA=-1.0,
    ))
    A_in = static["A_in_cm2"]

    # Gate steady-state seeds at V_REST
    state0 = dict(
        m  = jnp.full(n_comp, 0.074,  dtype=jnp.float64),
        h  = jnp.full(n_comp, 0.601,  dtype=jnp.float64),
        mp = jnp.full(n_comp, 0.00049,dtype=jnp.float64),
        s  = jnp.full(n_comp, 0.998,  dtype=jnp.float64),
    )

    celsius_arr = jnp.full(n_comp, CELSIUS, dtype=jnp.float64)
    is_node_arr = jnp.array(geom.is_node, dtype=bool)

    def membrane_fn(Vm, state, dt):
        m, h, mp, s = state["m"], state["h"], state["mp"], state["s"]
        is_node = is_node_arr

        # AxnodeMyel kinetics (replicate from AxnodeMyel channel)
        from jaxfibers.channels.mrg_axnode import AxnodeMyel as _A
        dummy_params = {
            "AxnodeMyel_celsius": celsius_arr,
        }
        dummy_states = {
            "AxnodeMyel_m": m, "AxnodeMyel_h": h,
            "AxnodeMyel_mp": mp, "AxnodeMyel_s": s,
        }
        new_states_raw = _A().update_states(dummy_states, dt, Vm, dummy_params)
        I_ion_node     = _A().compute_current(dummy_states, Vm, dummy_params)
        I_ion = jnp.where(is_node, I_ion_node, 0.0)

        new_state = {
            "m":  new_states_raw["AxnodeMyel_m"],
            "h":  new_states_raw["AxnodeMyel_h"],
            "mp": new_states_raw["AxnodeMyel_mp"],
            "s":  new_states_raw["AxnodeMyel_s"],
        }
        return I_ion, new_state

    return static, Ve_unit, state0, mid, nodes, n_comp, A_in, membrane_fn


def _jax_run_intra(D: float):
    """Intracellular JAX run: rectangular pulse at mid-node, return Vm traces."""
    static, Ve_unit, state0, mid, nodes, n_comp, A_in, membrane_fn = _make_jax_setup(D)
    n_steps = int(TSTOP / DT) + 1
    t_arr   = np.arange(n_steps) * DT
    pulse   = rectangular_pulse(t_arr, DELAY, INTRA_PW_MS, amp_nA=INTRA_AMP_NA)

    @jax.jit
    def run(I_intra_arr):
        def step(carry, I_intra_t):
            Vm, Vpax, state = carry
            # Add intracellular stimulus at mid node (convert nA to mA/cm²)
            stim = jnp.zeros(n_comp, dtype=jnp.float64)
            stim = stim.at[mid].add(I_intra_t * 1e-6 / A_in[mid])
            Vm_new, Vpax_new, state_new = integrate(
                static, membrane_fn, Vm, Vpax, state, DT,
                Ve=jnp.zeros(n_comp, dtype=jnp.float64),
                I_intra=stim,
            )
            return (Vm_new, Vpax_new, state_new), Vm_new[nodes]

        Vm0   = jnp.full(n_comp, V_REST,  dtype=jnp.float64)
        Vpax0 = jnp.full(n_comp, 0.0,    dtype=jnp.float64)
        _, vm_nodes = jax.lax.scan(step, (Vm0, Vpax0, state0), jnp.array(I_intra_arr))
        return vm_nodes  # (T, N_nodes)

    vm = np.array(run(pulse))
    return t_arr, vm.T  # (N_nodes, T)


# ── Task 1: Intracellular traces ──────────────────────────────────────────────

def task_traces():
    print("\n=== Task 1: Intracellular traces ===")
    D = TRACE_DIAM

    t0 = time.time()
    t_jax, vm_jax = _jax_run_intra(D)
    t1 = time.time()
    print(f"  JAX: {t1-t0:.1f} s")

    res_pf = run_intracellular_mrg_interp(
        diameter=D, n_nodes=N_NODES, temperature=CELSIUS,
        i_delay_ms=DELAY, i_dur_ms=INTRA_PW_MS, i_amp_nA=INTRA_AMP_NA,
        dt_ms=DT, tstop_ms=TSTOP,
    )
    mid_idx = N_NODES // 2
    t_pf  = res_pf.t_ms
    vm_pf = res_pf.vm_mV[mid_idx]

    # Interpolate JAX to PF time grid
    vm_jax_mid = np.interp(t_pf, t_jax, vm_jax[mid_idx])
    rmse = float(np.sqrt(np.mean((vm_jax_mid - vm_pf)**2)))
    print(f"  RMSE = {rmse:.3f} mV | PyFibers APs: {res_pf.n_aps_at_probe}")

    data = dict(
        t_jax=t_jax.tolist(), vm_jax_mid=vm_jax[mid_idx].tolist(),
        t_pf=t_pf.tolist(),   vm_pf_mid=vm_pf.tolist(),
        rmse_mV=rmse, diameter=D,
    )
    save_json(data, OUT / "data_mrg_interp_traces.json")

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(t_pf,  vm_pf,          "k-",  lw=1.5, label="PyFibers/NEURON")
    ax.plot(t_jax, vm_jax[mid_idx],"r--", lw=1.2, label=f"JAX (RMSE={rmse:.2f} mV)")
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Vm (mV)")
    ax.set_title(f"MRG_INTERP intracellular AP — D={D} µm")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(OUT / "fig_mrg_interp_traces.png", dpi=150)
    plt.close(fig)
    print(f"  → {OUT / 'fig_mrg_interp_traces.png'}")


# ── Task 2: Strength-duration curves ─────────────────────────────────────────

def task_sd():
    print("\n=== Task 2: Strength-duration curves ===")
    from experiments_v2.utils import pf_find_threshold, jax_bisect

    results = {"diameters": DIAMETERS, "pws_ms": SD_PWS, "pulse_keys": PULSE_KEYS, "data": {}}

    for D in DIAMETERS:
        print(f"  D={D} µm")
        results["data"][str(D)] = {}
        static, Ve_unit, state0, mid, nodes, n_comp, A_in, membrane_fn = _make_jax_setup(D)

        for pk in PULSE_KEYS:
            results["data"][str(D)][pk] = {}
            for pw in SD_PWS:
                # PyFibers threshold
                th_pf = pf_find_threshold(
                    "mrg_interp", D, N_NODES, pw, pk, TSTOP, DT, CELSIUS, SRC_H,
                )

                # JAX threshold
                n_steps = int(TSTOP / DT) + 1
                t_arr   = np.arange(n_steps) * DT
                Ve_unit_j = jnp.array(Ve_unit, dtype=jnp.float64)

                def jax_fires(amp):
                    pulse_arr = make_pulse_array(t_arr, DELAY, pw, pk)
                    Ve_seq = jnp.outer(jnp.array(pulse_arr, dtype=jnp.float64), Ve_unit_j) * amp
                    Vm0   = jnp.full(n_comp, V_REST, dtype=jnp.float64)
                    Vpax0 = jnp.full(n_comp, 0.0,   dtype=jnp.float64)

                    def step(carry, Ve_t):
                        Vm, Vpax, state = carry
                        Vm_new, Vpax_new, state_new = integrate(
                            static, membrane_fn, Vm, Vpax, state, DT,
                            Ve=Ve_t, I_intra=jnp.zeros(n_comp, dtype=jnp.float64),
                        )
                        return (Vm_new, Vpax_new, state_new), Vm_new[mid]

                    _, vm_mid = jax.lax.scan(step, (Vm0, Vpax0, state0), Ve_seq)
                    return jnp.max(vm_mid) > 0.0

                jax_fires_jit = jax.jit(jax_fires)
                th_jax = float(jax_bisect(jax_fires_jit, lo=-10.0, hi=-0.001, n_iter=18))

                err_pct = 100.0 * abs(th_jax - th_pf) / max(abs(th_pf), 1e-12)
                results["data"][str(D)][pk][str(pw)] = dict(
                    pf=th_pf, jax=th_jax, err_pct=err_pct,
                )
                print(f"    D={D} pw={pw} {pk}: PF={th_pf:.4f} JAX={th_jax:.4f} ({err_pct:.1f}%)")

    save_json(results, OUT / "data_mrg_interp_sd.json")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")
    task_traces()
    task_sd()
    print("\nDone.")
