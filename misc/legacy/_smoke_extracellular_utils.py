"""Smoke test for extracellular_utils.py activating function + Jaxley injection."""
import sys, pathlib, time
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import jax.numpy as jnp
import jaxley as jx

from jaxfibers.fibers.mrg import build_mrg, section_centers_um, node_indices
from jaxfibers.stim.extracellular import (
    point_source_potentials_mV, solve_vpax_static, rectangular_waveform,
)
from jaxfibers.stim.extracellular_utils import (
    axial_conductances_uS, compute_activating_currents_nA,
)

D = 10.0; N = 21; DT = 0.025; TSTOP = 5.0; PW = 0.1; DELAY = 1.0
cell, geom = build_mrg(diameter=D, n_nodes=N)
nodes = node_indices(geom)
mid = nodes[len(nodes) // 2]
centers = section_centers_um(geom)

# --- geometry check ---
Gax = axial_conductances_uS(geom)
print(f"n_comp={geom.n_comp}, n_Gax={len(Gax)}")
print(f"Gax[node-MYSA]={Gax[mid]:.4f} µS  (expect ~6.1 µS)")

# --- build V_pax waveform ---
ve_unit = point_source_potentials_mV(centers, src_x_um=0., src_y_um=1000.,
                                     src_z_um=centers[mid], i0_mA=1.0)
vpax_unit = solve_vpax_static(ve_unit, geom.length_um, geom.xraxial_Mohm_cm,
                               geom.xg_myelin_S_cm2, geom.is_node, geom.diameter)

n_steps = int(TSTOP / DT)
t = np.arange(n_steps + 1) * DT
shape = rectangular_waveform(t, DELAY, PW, amp=1.0)
amp = -0.183   # near NEURON threshold for d=10 µm

vpax_matrix = np.outer(vpax_unit, shape * amp)  # (n_comp, n_steps+1)

# --- activating currents ---
I_inj = compute_activating_currents_nA(vpax_matrix, geom)
print(f"\nI_inj shape: {I_inj.shape}  (expect ({geom.n_comp}, {n_steps+1}))")
print(f"Peak |I_inj| at mid-node: {float(jnp.abs(I_inj[mid]).max()):.4f} nA")
print(f"Peak |I_inj| total: {float(jnp.abs(I_inj).max()):.4f} nA")
print(f"I_inj at node during pulse (step {int(DELAY/DT)+1}): {float(I_inj[mid, int(DELAY/DT)+1]):.4f} nA")

# --- inject into Jaxley and integrate ---
print("\nInjecting into Jaxley cell and running integration ...")
# The activating function waveform (n_steps+1 points, index 0..n_steps)
# Jaxley step n uses stim[n], so we pass the full (n_comp, n_steps+1) array.
# Inject each compartment individually:
I_np = np.asarray(I_inj)
n_injected = 0
for i in range(geom.n_comp):
    if np.any(I_np[i] != 0.0):
        cell.branch(0).comp(i).stimulate(I_np[i])
        n_injected += 1
print(f"Stimuli inserted: {n_injected} compartments")

cell.branch(0).comp(mid).record()
cell.to_jax()

t0 = time.time()
v = jx.integrate(cell, delta_t=DT, t_max=TSTOP)
elapsed = time.time() - t0
print(f"Integration took {elapsed:.1f}s")
print(f"max V at node: {float(jnp.max(v)):.2f} mV  (expect ~+30 mV if AP fires)")
print(f"V at t=1.0 ms  (step {int(DELAY/DT)}): {float(v[0, int(DELAY/DT)]):.2f} mV")
print(f"V at t=1.5 ms  (step {int(1.5/DT)}):   {float(v[0, int(1.5/DT)]):.2f} mV")
