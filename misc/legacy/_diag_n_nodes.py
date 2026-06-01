"""Check threshold vs N nodes to isolate end-effects from model error."""
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

D = 10.0; DT = 0.005; TSTOP = 5.0; PW = 0.1; DELAY = 1.0
NEURON_REF = -0.18317

for N in [11, 21, 31, 51]:
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

    def fires(amp_mA):
        vpax_wave = np.outer(vpax_unit, shape * amp_mA).astype(np.float32)
        rec = integrate_mrg_extracellular(
            geom=geom, vpax_waveform_mV=vpax_wave, stim_waveform_nA=None,
            dt_ms=DT, t_max_ms=TSTOP, record_comps=[mid], temperature=37.0,
        )
        return float(jnp.max(rec)) > -20.0

    mag_lo, mag_hi = 0.05, 0.50
    while fires(-mag_lo): mag_lo /= 2.0
    while not fires(-mag_hi): mag_hi *= 2.0
    for _ in range(14):
        mag_mid = (mag_lo + mag_hi) / 2.0
        if fires(-mag_mid): mag_hi = mag_mid
        else: mag_lo = mag_mid
    thr = -mag_hi
    err = abs(thr - NEURON_REF) / abs(NEURON_REF) * 100
    print(f"N={N:2d}: threshold={thr:.5f} mA  error={err:.1f}%")
