"""Quick smoke test for the Phase A.1 double-cable implementation."""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from jaxfibers.fibers.mrg import build_mrg, section_centers_um, node_indices
from jaxfibers.stim.extracellular import (
    point_source_potentials_mV, solve_vpax_static, activating_currents_nA,
)

cell, geom = build_mrg(diameter=10.0, n_nodes=11)
centers = section_centers_um(geom)
node_idx = node_indices(geom)
mid = node_idx[5]

# Verify series-formula effective parameters: node has full cm; myelinated sections are myelin-dominated
assert geom.cm_uF_cm2[node_idx[5]] == 2.0, f"node cm wrong: {geom.cm_uF_cm2[node_idx[5]]}"
cm_mysa = geom.cm_uF_cm2[node_idx[5]+1]
g_mysa  = geom.gleak_S_cm2[node_idx[5]+1]
assert cm_mysa < 1e-3, f"MYSA cm should be myelin-dominated (<1e-3 µF/cm²), got {cm_mysa}"
assert g_mysa  < 1e-5, f"MYSA g_pas should be myelin-dominated (<1e-5 S/cm²), got {g_mysa}"
print(f"cm[node]={geom.cm_uF_cm2[node_idx[5]]:.4f}  cm[MYSA]={cm_mysa:.4e}  gpas[MYSA]={g_mysa:.4e}")

# Verify double-cable parameters
print(f"xraxial[node]={geom.xraxial_Mohm_cm[node_idx[5]]:.1f} MΩ/cm  xg[node]={geom.xg_myelin_S_cm2[node_idx[5]]:.1e} S/cm²")
print(f"xraxial[MYSA]={geom.xraxial_Mohm_cm[node_idx[5]+1]:.1f} MΩ/cm  xg[MYSA]={geom.xg_myelin_S_cm2[node_idx[5]+1]:.2e} S/cm²")
print(f"xraxial[STIN]={geom.xraxial_Mohm_cm[node_idx[5]+4]:.1f} MΩ/cm  xg[STIN]={geom.xg_myelin_S_cm2[node_idx[5]+4]:.2e} S/cm²")

# Solve V_pax
ve = point_source_potentials_mV(centers, src_x_um=0., src_y_um=1000., src_z_um=centers[mid], i0_mA=1.0)
vpax = solve_vpax_static(ve, geom.length_um, geom.xraxial_Mohm_cm, geom.xg_myelin_S_cm2, geom.is_node, geom.diameter)

print(f"\nVe at nodes {node_idx[3:8]}:   {[f'{ve[i]:.4f}' for i in node_idx[3:8]]}")
print(f"Vpax at nodes {node_idx[3:8]}: {[f'{vpax[i]:.4f}' for i in node_idx[3:8]]}")

mysa = node_idx[5]+1
stin = node_idx[5]+4
print(f"\nMYSA: Ve={ve[mysa]:.5f}  Vpax={vpax[mysa]:.5f}  Vpax/Ve={vpax[mysa]/ve[mysa]:.4f}")
print(f"STIN: Ve={ve[stin]:.5f}  Vpax={vpax[stin]:.5f}  Vpax/Ve={vpax[stin]/ve[stin]:.4f}")

# V_pax at nodes must equal V_e (Dirichlet BC)
for i in node_idx:
    assert abs(vpax[i] - ve[i]) < 1e-10, f"Dirichlet BC violated at node {i}: vpax={vpax[i]:.6f} ve={ve[i]:.6f}"
print("\nDirichlet BC at all nodes: OK")

# Activating currents using V_pax
inj = activating_currents_nA(vpax, geom.length_um, geom.diam_um, geom.Ra_ohm_cm)
print(f"\nActivating currents at nodes (per 1 mA source, nA):")
for i in node_idx[3:8]:
    print(f"  node {i}: {inj[i]:.4f} nA")

print(f"\nPeak |inj|: {np.max(np.abs(inj)):.4f} nA")
print("\nAll checks passed.")
