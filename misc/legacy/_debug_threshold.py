"""Test near-threshold amplitudes to verify AP firing."""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from jaxfibers.fibers.mrg import build_mrg, section_centers_um, node_indices
from jaxfibers.stim.extracellular import (
    point_source_potentials_mV, solve_vpax_static, rectangular_waveform,
)
from jaxfibers.stim.mrg_extracellular_solver import integrate_mrg_extracellular

D=10.0; N=21; DT=0.005; TSTOP=5.0; PW=0.1; DELAY=1.0
cell, geom = build_mrg(diameter=D, n_nodes=N)
nodes = node_indices(geom)
mid = nodes[len(nodes)//2]
centers = section_centers_um(geom)

ve_unit = point_source_potentials_mV(centers, src_x_um=0., src_y_um=1000.,
                                     src_z_um=centers[mid], i0_mA=1.0)
vpax_unit = solve_vpax_static(ve_unit, geom.length_um, geom.xraxial_Mohm_cm,
                               geom.xg_myelin_S_cm2, geom.is_node, geom.diameter)
print(f"vpax_unit[node]={vpax_unit[mid]:.2f} mV/mA")
print(f"Expected V_pax at NEURON threshold (-0.183 mA): {vpax_unit[mid]*-0.183:.2f} mV")

n_steps = int(TSTOP/DT)
t = np.arange(n_steps+1)*DT
shape = rectangular_waveform(t, DELAY, PW, amp=1.0)

for amp in [-0.15, -0.18, -0.183, -0.19, -0.20, -0.25, -0.30]:
    vpax_wave = np.outer(vpax_unit, shape*amp).astype('float32')
    rec = integrate_mrg_extracellular(geom, vpax_wave, None, DT, TSTOP, [mid])
    v = np.array(rec[0])
    fires = v.max() > -20.0
    print(f"amp={amp:.3f} mA: max V={v.max():.1f} mV  {'AP FIRES' if fires else 'no AP'}")
