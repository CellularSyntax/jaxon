"""Threshold bisection using activating function + jx.integrate for d=10 µm."""
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
from jaxfibers.stim.extracellular_utils import compute_activating_currents_nA

D = 10.0; N = 21; DT = 0.025; TSTOP = 5.0; PW = 0.1; DELAY = 1.0
_, geom = build_mrg(diameter=D, n_nodes=N)
nodes = node_indices(geom)
mid = nodes[len(nodes) // 2]
centers = section_centers_um(geom)

ve_unit = point_source_potentials_mV(centers, src_x_um=0., src_y_um=1000.,
                                     src_z_um=centers[mid], i0_mA=1.0)
vpax_unit = solve_vpax_static(ve_unit, geom.length_um, geom.xraxial_Mohm_cm,
                               geom.xg_myelin_S_cm2, geom.is_node, geom.diameter)
n_steps = int(TSTOP / DT)
t = np.arange(n_steps + 1) * DT
shape = rectangular_waveform(t, DELAY, PW, amp=1.0)
vpax_matrix_unit = np.outer(vpax_unit, shape)  # (n_comp, n_steps+1) at 1 mA
I_unit = np.asarray(compute_activating_currents_nA(vpax_matrix_unit, geom))  # (n_comp, n_steps+1)


def fires(amp_mA: float) -> bool:
    """True if AP fires (max V > -20 mV) for given cathodic amplitude."""
    cell, _ = build_mrg(diameter=D, n_nodes=N)
    I_inj = I_unit * amp_mA
    for i in range(geom.n_comp):
        if np.any(I_inj[i] != 0.0):
            cell.branch(0).comp(i).stimulate(I_inj[i])
    cell.branch(0).comp(mid).record()
    cell.to_jax()
    v = jx.integrate(cell, delta_t=DT, t_max=TSTOP)
    return float(jnp.max(v)) > -20.0


# Bisection over cathodic magnitude.
# Work with positive magnitudes; actual amp = -mag_mA.
# Invariant: mag_lo = no fire, mag_hi = fires.
mag_lo, mag_hi = 0.01, 0.5
while fires(-mag_lo):     # ensure lo is truly subthreshold
    mag_lo /= 2.0
while not fires(-mag_hi): # ensure hi is truly suprathreshold
    mag_hi *= 2.0
print(f"Bracket: no-fire={-mag_lo:.4f} mA, fire={-mag_hi:.4f} mA")

t0 = time.time()
for i in range(12):
    mag_mid = (mag_lo + mag_hi) / 2.0
    if fires(-mag_mid):
        mag_hi = mag_mid   # suprathreshold → tighten upper bound
    else:
        mag_lo = mag_mid   # subthreshold   → tighten lower bound
    print(f"  iter {i+1:2d}: [{-mag_hi:.5f}, {-mag_lo:.5f}]  threshold≈{-mag_hi:.5f} mA")

elapsed = time.time() - t0
thr = -mag_hi
print(f"\nActivating-function threshold: {thr:.5f} mA  ({elapsed:.0f}s)")
print(f"NEURON (PyFibers) reference:  -0.18317 mA")
print(f"Error: {abs(thr - (-0.18317)) / 0.18317 * 100:.1f}%")
