"""Quick threshold bisection for d=10µm to verify Phase A.1 improvement."""
import sys, pathlib, time
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import jax, jax.numpy as jnp
import jaxley as jx

from jaxfibers.fibers.mrg import build_mrg, section_centers_um, node_indices
from jaxfibers.stim.extracellular import (
    point_source_potentials_mV, solve_vpax_static, activating_currents_nA,
    rectangular_waveform,
)

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
inj_unit = activating_currents_nA(vpax_unit, geom.length_um, geom.diam_um, geom.Ra_ohm_cm)
active = np.where(np.abs(inj_unit) > 1e-12)[0]
inj_active = jnp.asarray(inj_unit[active])

n_steps = int(TSTOP / DT) + 1
t = np.arange(n_steps) * DT
shape = jnp.asarray(rectangular_waveform(t, DELAY, PW, amp=1.0))
cell.branch(0).comp(mid_comp).record("v")

def _forward(amp):
    currents = inj_active[:, None] * (shape * amp)[None, :]
    ds = None
    for k, ci in enumerate(active.tolist()):
        ds = cell.branch(0).comp(int(ci)).data_stimulate(currents[k], ds)
    v = jx.integrate(cell, delta_t=DT, t_max=TSTOP, data_stimuli=ds, solver="bwd_euler")
    return jnp.max(v)

forward_jit = jax.jit(_forward)
fires = lambda amp: float(forward_jit(jnp.float32(amp))) > -20.0

# warm up JIT
_ = fires(-0.5)

lo, hi = -15.0, -0.001
for _ in range(4):
    if fires(lo): break
    lo *= 2.0

t0 = time.time()
iters = 0
while abs(hi - lo) / abs(lo) > 1e-3:
    mid = (lo + hi) / 2.0
    if fires(mid): lo = mid
    else: hi = mid
    iters += 1

elapsed = time.time() - t0
print(f"d={D}µm | Jaxley double-cable threshold: {lo:.4f} mA  ({iters} iters, {elapsed:.1f}s)")
print(f"  PyFibers baseline (v1):  -0.183 mA")
print(f"  Jaxley single-cable (v1): -0.138 mA")
print(f"  New error vs PyFibers:  {abs(lo - (-0.183))/0.183*100:.1f}%")
