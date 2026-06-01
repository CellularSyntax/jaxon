"""JAX coupled-solver thresholds for all MRG_DISCRETE fiber diameters.

Same electrode / waveform setup as pyfibers_mrg_thresholds.py.
Saves results to outputs/jax_thresholds.json.
"""
import sys, pathlib, dataclasses, json, time
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import jax.numpy as jnp
from jaxley.solver_gate import solve_gate_exponential

from jaxfibers.fibers.mrg import (
    build_mrg, section_centers_um, node_indices,
    V_REST, CM_AXON, G_PAS_MYSA, G_PAS_FLUT, G_PAS_STIN, _MRG_DISCRETE,
)
from jaxfibers.channels.mrg_axnode import AxnodeMyel
from jaxfibers.stim.extracellular import point_source_potentials_mV, rectangular_waveform
from mrg_extracellular_coupled import arrays_from_geometry, find_threshold

# ── shared parameters ──────────────────────────────────────────────────────
DIAMETERS = [5.7, 7.3, 8.7, 10.0, 11.5, 12.8, 14.0, 15.0, 16.0]  # skip 1/2 µm: impractical at 1 mm
N_NODES   = 21
DT        = 0.005; TSTOP = 5.0; PW = 0.1; DELAY = 1.0
CELSIUS   = 37.0

n_steps    = int(TSTOP / DT)
shape_arr  = rectangular_waveform(np.arange(n_steps + 1) * DT, DELAY, PW, amp=1.0)
pulse_mask = jnp.asarray(shape_arr[:n_steps] > 0.5)

# Resting gate steady-states (same for all diameters)
_v0 = jnp.float64(V_REST)
(a_m0,b_m0),(a_h0,b_h0),(a_mp0,b_mp0),(a_s0,b_s0) = AxnodeMyel._alpha_beta(_v0, CELSIUS)
m0  = float(a_m0  / (a_m0  + b_m0))
h0  = float(a_h0  / (a_h0  + b_h0))
mp0 = float(a_mp0 / (a_mp0 + b_mp0))
s0  = float(a_s0  / (a_s0  + b_s0))

_gnabar=3.0; _gnapbar=0.01; _gkbar=0.08; _gl=0.007
_ena=50.0;   _ek=-90.0;     _el=-90.0
_stype_gpas = {"node": 0.0, "mysa": G_PAS_MYSA, "flut": G_PAS_FLUT, "stin": G_PAS_STIN}

# ── per-diameter loop ──────────────────────────────────────────────────────
results = {}
print(f"{'D (µm)':>8}  {'threshold (mA)':>16}  {'time (s)':>8}")
print("-" * 38)

for D in DIAMETERS:
    t0 = time.time()
    _, geom   = build_mrg(diameter=D, n_nodes=N_NODES)
    nodes     = node_indices(geom)
    centers   = section_centers_um(geom)
    mid       = nodes[len(nodes) // 2]
    n_comp    = geom.n_comp

    geom_c  = dataclasses.replace(geom, cm_uF_cm2=[CM_AXON] * n_comp)
    static  = arrays_from_geometry(geom_c, DT)
    is_node = static["is_node"]
    A_in    = static["A_in_cm2"]

    g_pas_arr = jnp.asarray([_stype_gpas[s] for s in geom.section_type])
    Ve_unit   = np.array(point_source_potentials_mV(
        centers, src_x_um=0., src_y_um=1000., src_z_um=centers[mid], i0_mA=-1.0
    ))

    def membrane_fn_factory(is_node=is_node, A_in=A_in, g_pas_arr=g_pas_arr):
        state0 = (
            jnp.where(is_node, m0, 0.0), jnp.where(is_node, h0, 0.0),
            jnp.where(is_node, mp0, 0.0), jnp.where(is_node, s0, 0.0),
        )
        def membrane_fn(Vm, state, dt, is_node=is_node, A_in=A_in, g_pas_arr=g_pas_arr):
            M, H, MP, S = state
            (a_m,b_m),(a_h,b_h),(a_mp,b_mp),(a_s,b_s) = AxnodeMyel._alpha_beta(Vm, CELSIUS)
            M2  = solve_gate_exponential(M,  dt, a_m,  b_m)
            H2  = solve_gate_exponential(H,  dt, a_h,  b_h)
            MP2 = solve_gate_exponential(MP, dt, a_mp, b_mp)
            S2  = solve_gate_exponential(S,  dt, a_s,  b_s)
            g_na  = _gnabar  * M2**3 * H2
            g_nap = _gnapbar * MP2**3
            g_k   = _gkbar   * S2
            g_node   = (g_na+g_nap+g_k+_gl) * A_in * 1e6
            i_node   = ((g_na+g_nap)*(Vm-_ena)+g_k*(Vm-_ek)+_gl*(Vm-_el)) * A_in * 1e6
            g_pas_us = g_pas_arr * A_in * 1e6
            i_pas    = g_pas_us * (Vm - V_REST)
            return jnp.where(is_node, g_node, g_pas_us), jnp.where(is_node, i_node, i_pas), (M2,H2,MP2,S2)
        return membrane_fn, state0

    thr = find_threshold(
        static, membrane_fn_factory, Ve_unit, pulse_mask, DT,
        v_thresh=-30.0, lo=-50.0, hi=-0.001, tol=1e-4, v_rest=V_REST,
    )
    results[D] = float(thr)
    print(f"{D:>8.1f}  {thr:>16.5f}  {time.time()-t0:>8.1f}")

out = ROOT / "outputs" / "jax_thresholds.json"
out.parent.mkdir(exist_ok=True)
import json
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {out}")
