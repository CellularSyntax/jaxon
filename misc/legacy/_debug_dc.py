import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import jax.numpy as jnp

from jaxfibers.fibers.mrg import build_mrg, section_centers_um, node_indices
from jaxfibers.stim.extracellular import (
    point_source_potentials_mV, solve_vpax_static, rectangular_waveform,
)
from jaxfibers.stim.mrg_extracellular_solver import (
    integrate_mrg_extracellular, _build_geometry_arrays,
)

D=10.0; N=21; DT=0.005; TSTOP=5.0; PW=0.1; DELAY=1.0
cell, geom = build_mrg(diameter=D, n_nodes=N)
nodes = node_indices(geom)
mid = nodes[len(nodes)//2]
centers = section_centers_um(geom)

ve_unit = point_source_potentials_mV(centers, src_x_um=0., src_y_um=1000.,
                                     src_z_um=centers[mid], i0_mA=1.0)
vpax_unit = solve_vpax_static(ve_unit, geom.length_um, geom.xraxial_Mohm_cm,
                               geom.xg_myelin_S_cm2, geom.is_node, geom.diameter)

Cm, Gax, Gpas, Epas, is_node, A = _build_geometry_arrays(geom)
print(f"n_comp={geom.n_comp}")
print(f"Cm[node]={float(Cm[mid]):.6f} nF  (expect ~0.000208 nF = 0.208 pF)")
print(f"Gax[node-MYSA]={float(Gax[mid]):.4f} uS  (expect ~6 uS)")
print(f"Gpas[MYSA]={float(Gpas[mid+1]):.6f} uS")
print(f"Cm/dt at node: {float(Cm[mid])/DT:.6f} uS (expect ~0.0416 uS)")
print(f"vpax_unit[node]={vpax_unit[mid]:.2f} mV/mA")

# Test at amp = -1.0 mA (clearly suprathreshold in NEURON)
amp = -1.0
n_steps = int(TSTOP/DT)
t = np.arange(n_steps+1)*DT
shape = rectangular_waveform(t, DELAY, PW, amp=1.0)
vpax_wave = np.outer(vpax_unit, shape*amp).astype('float32')
print(f"\nV_pax at node at t=DELAY: {vpax_wave[mid, 200]:.2f} mV  (expect ~{vpax_unit[mid]*amp:.2f})")

rec = integrate_mrg_extracellular(geom, vpax_wave, None, DT, TSTOP, [mid])
v = np.array(rec[0])
print(f"\namp={amp} mA:")
print(f"  max V: {v.max():.2f} mV  (should be ~+30 mV if AP fires)")
print(f"  V at t=1.00 ms (step 200): {v[200]:.2f} mV")
print(f"  V at t=1.10 ms (step 220): {v[220]:.2f} mV")
print(f"  V at t=1.20 ms (step 240): {v[240]:.2f} mV")
