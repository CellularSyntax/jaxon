"""Debug the V_m custom integrator step by step."""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import jax.numpy as jnp

from jaxfibers.fibers.mrg import build_mrg, section_centers_um, node_indices
from jaxfibers.stim.extracellular import (
    point_source_potentials_mV, solve_vpax_static, rectangular_waveform,
)
from jaxfibers.stim.extracellular_utils import axial_conductances_uS

D=10.0; N=21; DT=0.005; TSTOP=5.0; PW=0.1; DELAY=1.0
cell, geom = build_mrg(diameter=D, n_nodes=N)
nodes = node_indices(geom)
mid = nodes[len(nodes)//2]   # index 110 (11th node of 21)
centers = section_centers_um(geom)

ve_unit = point_source_potentials_mV(centers, src_x_um=0., src_y_um=1000.,
                                     src_z_um=centers[mid], i0_mA=1.0)
vpax_unit = solve_vpax_static(ve_unit, geom.length_um, geom.xraxial_Mohm_cm,
                               geom.xg_myelin_S_cm2, geom.is_node, geom.diameter)

# V_pax profile at threshold amplitude
amp = -0.183
vp_threshold = vpax_unit * amp

print(f"mid (node) comp index: {mid}")
print(f"V_pax at node:        {vp_threshold[mid]:.4f} mV")
print(f"V_pax at MYSA left:   {vp_threshold[mid-1]:.4f} mV")
print(f"V_pax at MYSA right:  {vp_threshold[mid+1]:.4f} mV (actually mid+1 is MYSA)")
print(f"Ve at node:           {ve_unit[mid]*amp:.4f} mV")

# Compute I_af at the node manually
Gax = axial_conductances_uS(geom)
G_left  = Gax[mid - 1]   # between comp mid-1 (MYSA) and mid (node)
G_right = Gax[mid]       # between comp mid (node) and mid+1 (MYSA)

I_af_node = (G_left  * (vp_threshold[mid-1] - vp_threshold[mid]) +
             G_right * (vp_threshold[mid+1] - vp_threshold[mid]))

print(f"\nGax[node-MYSA left]:  {G_left:.4f} µS")
print(f"Gax[node-MYSA right]: {G_right:.4f} µS")
print(f"dVp left  (MYSA-node): {vp_threshold[mid-1] - vp_threshold[mid]:.4f} mV")
print(f"dVp right (MYSA-node): {vp_threshold[mid+1] - vp_threshold[mid]:.4f} mV")
print(f"I_af at node:          {I_af_node:.4f} nA  (NEURON-like should be ~100-200 nA)")

# What V_m change does this produce in 1 timestep?
from jaxfibers.stim.mrg_extracellular_solver import _build_geometry_arrays
Cm, Gax_arr, Gpas, Epas, is_node, A = _build_geometry_arrays(geom)
Cm_node = float(Cm[mid])
diag_node = float(Cm[mid]/DT + Gax_arr[mid-1] + Gax_arr[mid] + Gpas[mid])

print(f"\nCm[node]={Cm_node:.6f} nF")
print(f"Cm/dt[node]={Cm_node/DT:.4f} µS")
print(f"diag[node]={diag_node:.4f} µS  (Cm/dt + 2*Gax + Gpas)")
print(f"\nExpected DeltaVm at node from I_af alone (ignoring axial drain):")
print(f"  DeltaVm = I_af * dt / Cm = {I_af_node * DT / Cm_node:.2f} mV")
print(f"  But with axial coupling: DeltaVm ≈ I_af / diag = {I_af_node / diag_node:.4f} mV (per step)")

# The tridiagonal solve for just node+MYSA (simplified 3-comp system)
Cm_mysa = float(Cm[mid-1])
Gax_nm = float(Gax_arr[mid-1])  # node-MYSA conductance

vp_mysa = vp_threshold[mid-1]
vp_node = vp_threshold[mid]
I_af_mysa_left = (float(Gax_arr[mid-2]) * (vp_threshold[mid-2] - vp_mysa) +
                  Gax_nm * (vp_node - vp_mysa))

print(f"\n--- MYSA left (comp {mid-1}) ---")
print(f"V_pax[MYSA]:  {vp_mysa:.4f} mV")
print(f"I_af[MYSA]:   {I_af_mysa_left:.4f} nA")
print(f"Cm[MYSA]:     {Cm_mysa:.6f} nF")
print(f"Cm/dt[MYSA]:  {Cm_mysa/DT:.4f} µS")

print("\n\n--- Summary: Is the V_m formulation physically correct? ---")
print(f"I_af at threshold ({amp} mA): {I_af_node:.1f} nA at node")
if abs(I_af_node) > 10:
    print("  ✓ I_af is significant → the activating function provides meaningful drive")
else:
    print("  ✗ I_af is too small → activating function insufficient")

print(f"\nDiag = {diag_node:.2f} µS, I_af/diag = {I_af_node/diag_node:.4f} mV change per step")
if abs(I_af_node/diag_node) > 0.5:
    print("  ✓ V_m change per step is significant")
else:
    print("  ✗ V_m change per step is very small — may take many steps to reach threshold")
