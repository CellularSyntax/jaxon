"""Smoke test for the custom double-cable backward-Euler integrator."""
import sys, pathlib, time
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import jax.numpy as jnp

from jaxfibers.fibers.mrg import build_mrg, section_centers_um, node_indices
from jaxfibers.stim.extracellular import (
    point_source_potentials_mV, solve_vpax_static, rectangular_waveform,
)
from jaxfibers.stim.mrg_extracellular_solver import find_threshold_extracellular_dc

D = 10.0
N_NODES = 21
SRC_H = 1000.0
PW = 0.1
DELAY = 1.0
DT = 0.005
TSTOP = 5.0

cell, geom = build_mrg(diameter=D, n_nodes=N_NODES)
nodes = node_indices(geom)
mid_comp = nodes[len(nodes) // 2]
centers = section_centers_um(geom)

ve_unit = point_source_potentials_mV(centers, src_x_um=0., src_y_um=SRC_H,
                                     src_z_um=centers[mid_comp], i0_mA=1.0)
vpax_unit = solve_vpax_static(ve_unit, geom.length_um, geom.xraxial_Mohm_cm,
                               geom.xg_myelin_S_cm2, geom.is_node, geom.diameter)

n_steps = int(TSTOP / DT)
t = np.arange(n_steps + 1) * DT
shape = rectangular_waveform(t, DELAY, PW, amp=1.0)

print("Running DC integrator threshold bisection for d=10µm ...")
t0 = time.time()
th = find_threshold_extracellular_dc(
    geom=geom,
    vpax_unit_mV=vpax_unit,
    t_grid_ms=t,
    shape_waveform=shape,
    dt_ms=DT,
    t_max_ms=TSTOP,
    mid_comp=mid_comp,
)
elapsed = time.time() - t0
print(f"DC integrator threshold: {th:.5f} mA  ({elapsed:.1f}s)")
print(f"PyFibers (NEURON):       -0.18317 mA")
print(f"Error: {abs(th - (-0.18317)) / 0.18317 * 100:.1f}%")
