"""Threshold bisection with Vi-state integrator (NEURON-equivalent direct coupling).

The Vi-state integrator uses V_i as the cable state variable, with channels
seeing V_m = V_i - V_pax. This replicates NEURON's extracellular mechanism:
at rectangular pulse onset, V_pax steps to ~-48 mV → channels instantly see
+48 mV depolarization via V_m = V_i - V_pax (direct coupling), unlike the
activating-function-only approach that misses the capacitive term.

Expected: threshold close to NEURON reference -0.18317 mA (<5% error).
"""
import sys, pathlib, time
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import jax.numpy as jnp

from jaxfibers.fibers.mrg import build_mrg, section_centers_um, node_indices
from jaxfibers.stim.extracellular import (
    point_source_potentials_mV, solve_vpax_static, rectangular_waveform,
)
from jaxfibers.stim.mrg_extracellular_solver import integrate_mrg_extracellular

D = 10.0; N = 21; DT = 0.005; TSTOP = 5.0; PW = 0.1; DELAY = 1.0
_, geom = build_mrg(diameter=D, n_nodes=N)
nodes = node_indices(geom)
mid = nodes[len(nodes) // 2]
centers = section_centers_um(geom)

# Precompute unit V_pax
ve_unit = point_source_potentials_mV(centers, src_x_um=0., src_y_um=1000.,
                                     src_z_um=centers[mid], i0_mA=1.0)
vpax_unit = solve_vpax_static(ve_unit, geom.length_um, geom.xraxial_Mohm_cm,
                               geom.xg_myelin_S_cm2, geom.is_node, geom.diameter)

n_steps = int(TSTOP / DT)
t = np.arange(n_steps + 1) * DT
shape = rectangular_waveform(t, DELAY, PW, amp=1.0)

print(f"V_pax at center node (unit, 1 mA): {vpax_unit[mid]:.4f} mV")
print(f"V_pax at center node (-0.183 mA):  {vpax_unit[mid]*-0.183:.4f} mV")
print(f"Expected Vm jump at onset:          {-vpax_unit[mid]*0.183:.4f} mV  (→ Vm = -80 + {-vpax_unit[mid]*0.183:.1f} = {-80 - vpax_unit[mid]*0.183:.1f} mV)")
print()


def fires(amp_mA: float) -> bool:
    vpax_wave = np.outer(vpax_unit, shape * amp_mA).astype(np.float32)
    rec = integrate_mrg_extracellular(
        geom=geom,
        vpax_waveform_mV=vpax_wave,
        stim_waveform_nA=None,
        dt_ms=DT,
        t_max_ms=TSTOP,
        record_comps=[mid],
        temperature=37.0,
    )
    return float(jnp.max(rec)) > -20.0


# Quick sanity check before bisection
print("Sanity checks:")
print(f"  fires(-0.10 mA): {fires(-0.10)}")
print(f"  fires(-0.18 mA): {fires(-0.18)}")
print(f"  fires(-0.25 mA): {fires(-0.25)}")
print()

# Bracket
mag_lo, mag_hi = 0.05, 0.50
while fires(-mag_lo):
    mag_lo /= 2.0
while not fires(-mag_hi):
    mag_hi *= 2.0
print(f"Bracket: no-fire={-mag_lo:.4f} mA, fire={-mag_hi:.4f} mA")

t0 = time.time()
for i in range(14):
    mag_mid = (mag_lo + mag_hi) / 2.0
    if fires(-mag_mid):
        mag_hi = mag_mid
    else:
        mag_lo = mag_mid
    print(f"  iter {i+1:2d}: [{-mag_hi:.5f}, {-mag_lo:.5f}]  threshold≈{-mag_hi:.5f} mA")

elapsed = time.time() - t0
thr = -mag_hi
neuron_ref = -0.18317
print(f"\nVi-state integrator threshold: {thr:.5f} mA  ({elapsed:.0f}s for 14 iters)")
print(f"NEURON (PyFibers) reference:   {neuron_ref:.5f} mA")
print(f"Error: {abs(thr - neuron_ref) / abs(neuron_ref) * 100:.1f}%")
