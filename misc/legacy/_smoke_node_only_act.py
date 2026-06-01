"""Try node-only activating function: in single-cable MRG, the myelin shields internodes,
so the standard simplification applies the activating function only at the nodes.

This is a quick test: does this reduce the NEURON vs Jaxley threshold gap?
"""
import sys, pathlib, time
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import jax
import jax.numpy as jnp
import jaxley as jx

from jaxfibers.fibers.mrg import build_mrg, node_indices, section_centers_um
from jaxfibers.stim.extracellular import (
    point_source_potentials_mV, activating_currents_nA, rectangular_waveform,
)
from jaxfibers.nrn_baseline import find_threshold_extracellular


def jx_threshold_node_only(diameter, n_nodes=21, src_height_um=1000.0,
                            pw_ms=0.1, delay_ms=1.0, dt_ms=0.005, tstop_ms=5.0,
                            rel_tol=5e-3, ap_threshold_mV=-20.0):
    cell, geom = build_mrg(diameter=diameter, n_nodes=n_nodes)
    nodes = node_indices(geom)
    mid_comp = nodes[len(nodes) // 2]

    centers = section_centers_um(geom)
    v_ext_unit_mV = point_source_potentials_mV(
        section_centers_um=centers, src_x_um=0.0, src_y_um=src_height_um,
        src_z_um=centers[mid_comp], i0_mA=1.0, sigma_S_m=0.3,
    )
    inj_unit_nA = activating_currents_nA(
        v_ext_mV=v_ext_unit_mV, length_um=geom.length_um,
        diam_um=geom.diam_um, Ra_ohm_cm=geom.Ra_ohm_cm,
    )
    # *** Mask: only apply activating function at nodes ***
    node_mask = np.array(geom.is_node, dtype=bool)
    inj_unit_nA = np.where(node_mask, inj_unit_nA, 0.0)
    active = np.where(np.abs(inj_unit_nA) > 1e-12)[0]
    inj_active = jnp.asarray(inj_unit_nA[active])
    print(f"  node-only active stimuli: {len(active)}/{len(inj_unit_nA)}")

    n_steps = int(tstop_ms / dt_ms) + 1
    t = np.arange(n_steps) * dt_ms
    shape = jnp.asarray(rectangular_waveform(t, delay_ms, pw_ms, amp=1.0))
    cell.branch(0).comp(mid_comp).record("v")

    def _forward(amp):
        currents = inj_active[:, None] * (shape * amp)[None, :]
        ds = None
        for k, ci in enumerate(active.tolist()):
            ds = cell.branch(0).comp(int(ci)).data_stimulate(currents[k], ds)
        v = jx.integrate(cell, delta_t=dt_ms, t_max=tstop_ms,
                         data_stimuli=ds, solver="bwd_euler")
        return jnp.max(v)

    forward = jax.jit(_forward)
    fires = lambda a: float(forward(jnp.float32(a))) > ap_threshold_mV

    lo, hi = -15.0, -0.001
    e = 0
    while not fires(lo) and e < 6:
        lo *= 2.0; e += 1
    if not fires(lo): raise RuntimeError("no fire")
    while abs(hi-lo)/abs(lo) > rel_tol:
        m = (lo+hi)/2.0
        if fires(m): lo = m
        else: hi = m
    return lo


for d in [5.7, 10.0, 14.0]:
    print(f"\n=== d = {d} µm ===", flush=True)
    t0 = time.time()
    jx_th = jx_threshold_node_only(d)
    print(f"  Jaxley (node-only act) = {jx_th:.5f} mA in {time.time()-t0:.1f}s", flush=True)
    nr_th = find_threshold_extracellular(diameter=d, n_nodes=21,
                                          src_height_um=1000.0, pw_ms=0.1,
                                          delay_ms=1.0, dt_ms=0.005,
                                          tstop_ms=5.0, rel_tol=5e-3)
    err = abs(jx_th - nr_th)/abs(nr_th)*100
    print(f"  NEURON                 = {nr_th:.5f} mA  |  err = {err:.2f}%", flush=True)
