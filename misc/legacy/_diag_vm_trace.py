"""Diagnose Vm trace at center node for Vi-state integrator."""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import jax.numpy as jnp

from jaxfibers.fibers.mrg import build_mrg, section_centers_um, node_indices, V_REST
from jaxfibers.stim.extracellular import (
    point_source_potentials_mV, solve_vpax_static, rectangular_waveform,
)
from jaxfibers.stim.mrg_extracellular_solver import integrate_mrg_extracellular

D = 10.0; N = 21; DT = 0.005; TSTOP = 5.0; PW = 0.1; DELAY = 1.0
_, geom = build_mrg(diameter=D, n_nodes=N)
nodes = node_indices(geom)
mid = nodes[len(nodes) // 2]   # center node (110)
centers = section_centers_um(geom)

ve_unit = point_source_potentials_mV(centers, src_x_um=0., src_y_um=1000.,
                                     src_z_um=centers[mid], i0_mA=1.0)
vpax_unit = solve_vpax_static(ve_unit, geom.length_um, geom.xraxial_Mohm_cm,
                               geom.xg_myelin_S_cm2, geom.is_node, geom.diameter)

n_steps = int(TSTOP / DT)
t = np.arange(n_steps + 1) * DT
shape = rectangular_waveform(t, DELAY, PW, amp=1.0)

# Show vpax profile at a few nearby compartments
print("V_pax profile (unit, at 1 mA):")
print(f"  node-2  [{nodes[len(nodes)//2-2]}]: {vpax_unit[nodes[len(nodes)//2-2]]:.2f} mV")
print(f"  node-1  [{nodes[len(nodes)//2-1]}]: {vpax_unit[nodes[len(nodes)//2-1]]:.2f} mV")
print(f"  CENTER  [{mid}]:             {vpax_unit[mid]:.4f} mV")
print(f"  node+1  [{nodes[len(nodes)//2+1]}]: {vpax_unit[nodes[len(nodes)//2+1]]:.2f} mV")
print(f"  node+2  [{nodes[len(nodes)//2+2]}]: {vpax_unit[nodes[len(nodes)//2+2]]:.2f} mV")

# Check Cm values used in solver
from jaxfibers.stim.mrg_extracellular_solver import _build_geometry_arrays
Cm, Gax, Gpas, Epas, is_node_arr, A = _build_geometry_arrays(geom)
print(f"\nCm at center node [{mid}]: {float(Cm[mid]):.6f} nF")
print(f"Cm at MYSA [mid-1]: {float(Cm[mid-1]):.6e} nF")
print(f"Cm at FLUT [mid-2]: {float(Cm[mid-2]):.6e} nF")
print(f"Cm at STIN [mid-3]: {float(Cm[mid-3]):.6e} nF")
print(f"Gax[mid-1 (node-MYSA)]: {float(Gax[mid-1]):.4f} µS")
print(f"Gax[mid (MYSA-node??)]: wait, Gax is n_comp-1 — Gax[i] = between comp i and i+1")
print(f"Gax between mid-1 and mid: Gax[{mid-1}] = {float(Gax[mid-1]):.4f} µS")
print(f"Gax between mid and mid+1: Gax[{mid}] = {float(Gax[mid]):.4f} µS")

# Run at -0.183 (NEURON threshold) and -0.198 (our threshold with CM_AXON)
for amp in [-0.183, -0.195]:
    vpax_wave = np.outer(vpax_unit, shape * amp).astype(np.float32)
    rec = integrate_mrg_extracellular(
        geom=geom,
        vpax_waveform_mV=vpax_wave,
        stim_waveform_nA=None,
        dt_ms=DT, t_max_ms=TSTOP,
        record_comps=[mid],
        temperature=37.0,
    )
    Vm_trace = np.array(rec[0])
    onset_idx = int(DELAY / DT)
    print(f"\namp = {amp} mA:")
    print(f"  Vm at onset   t={DELAY:.3f} ms (step {onset_idx}): {Vm_trace[onset_idx]:.2f} mV")
    print(f"  Vm at onset+1 t={(DELAY+DT):.3f} ms: {Vm_trace[onset_idx+1]:.2f} mV")
    print(f"  Vm at onset+2 t={(DELAY+2*DT):.3f} ms: {Vm_trace[onset_idx+2]:.2f} mV")
    print(f"  Vm at onset+4: {Vm_trace[onset_idx+4]:.2f} mV")
    print(f"  max Vm: {Vm_trace.max():.2f} mV  (AP fires if > -20 mV)")
    print(f"  Vm at t=1.1 ms: {Vm_trace[int(1.1/DT)]:.2f} mV")
    print(f"  Vm at t=1.5 ms: {Vm_trace[int(1.5/DT)]:.2f} mV")
