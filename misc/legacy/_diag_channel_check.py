"""Check pulse timing, channel kinetics, and Vm evolution at threshold."""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import jax.numpy as jnp

from jaxfibers.fibers.mrg import build_mrg, section_centers_um, node_indices, V_REST, CM_AXON
from jaxfibers.stim.extracellular import (
    point_source_potentials_mV, solve_vpax_static, rectangular_waveform,
)
from jaxfibers.stim.mrg_extracellular_solver import (
    integrate_mrg_extracellular, _build_geometry_arrays, _init_gates, _axnode_Eeff,
    _axnode_conductance_mS_cm2,
)

D = 10.0; N = 21; DT = 0.005; TSTOP = 5.0; PW = 0.1; DELAY = 1.0
_, geom = build_mrg(diameter=D, n_nodes=N)
nodes = node_indices(geom)
mid = nodes[len(nodes) // 2]
centers = section_centers_um(geom)

n_steps = int(TSTOP / DT)
t = np.arange(n_steps + 1) * DT
shape = rectangular_waveform(t, DELAY, PW, amp=1.0)

# 1. Pulse timing check
onset = int(DELAY / DT)
print("Pulse timing check (shape values around onset/offset):")
for k in [onset-1, onset, onset+1, onset+19, onset+20, onset+21]:
    print(f"  shape[{k}] = shape(t={k*DT:.3f} ms) = {shape[k]:.1f}")
n_pulse_steps = int(np.sum(shape > 0))
print(f"Total steps with shape>0: {n_pulse_steps}  (expect {int(PW/DT)} = {int(PW/DT)})")

# 2. Resting state gate values and node conductance
m0, h0, mp0, s0 = _init_gates(jnp.float32(V_REST), 37.0)
print(f"\nResting gates at V={V_REST}mV, T=37C:")
print(f"  m={float(m0):.4f}, h={float(h0):.4f}, mp={float(mp0):.4f}, s={float(s0):.4f}")
Cm, Gax, Gpas, Epas, is_node_arr, A = _build_geometry_arrays(geom)
G_rest = float(_axnode_conductance_mS_cm2(m0, h0, mp0, s0)) * float(A[mid]) * 1e6
print(f"  G_ion_rest[node] = {G_rest:.4f} µS")
E_rest = float(_axnode_Eeff(m0, h0, mp0, s0))
print(f"  E_eff_rest = {E_rest:.4f} mV  (should be close to {V_REST})")

# 3. First few steps of Vm at onset for our threshold and NEURON threshold
ve_unit = point_source_potentials_mV(centers, src_x_um=0., src_y_um=1000.,
                                     src_z_um=centers[mid], i0_mA=1.0)
vpax_unit = solve_vpax_static(ve_unit, geom.length_um, geom.xraxial_Mohm_cm,
                               geom.xg_myelin_S_cm2, geom.is_node, geom.diameter)

print(f"\nVm trace at center node (step-by-step around onset):")
for amp in [-0.170, -0.183]:
    vpax_wave = np.outer(vpax_unit, shape * amp).astype(np.float32)
    rec = integrate_mrg_extracellular(
        geom=geom, vpax_waveform_mV=vpax_wave, stim_waveform_nA=None,
        dt_ms=DT, t_max_ms=TSTOP, record_comps=[mid], temperature=37.0,
    )
    Vm = np.array(rec[0])
    print(f"\n  amp = {amp} mA:")
    for k in range(onset-1, onset+22):
        print(f"    t={k*DT:.3f} ms: Vm={Vm[k]:.2f} mV")
    print(f"    max Vm = {Vm.max():.2f} mV")

# 4. What Ve threshold PyFibers expects vs what our model has
print(f"\nKey geometry for D=10 µm (N=21):")
from jaxfibers.fibers.mrg import _MRG_DISCRETE, MYSA_LENGTH, NODE_LENGTH
p = _MRG_DISCRETE[D]
print(f"  axon_diam={p['axon_diam']}, node_diam={p['node_diam']}, nl={p['nl']}, delta_z={p['delta_z']}")
print(f"  MYSA diam in geom: {geom.diam_um[mid+1]:.2f} µm  (should be node_d={p['node_diam']})")
print(f"  FLUT diam in geom: {geom.diam_um[mid+2]:.2f} µm  (should be axon_d={p['axon_diam']})")
print(f"  A_node = {float(A[mid]):.4e} cm²")
print(f"  Cm_node = {float(Cm[mid]):.4e} nF")
print(f"  Cm_node_per_area = {float(Cm[mid])/float(A[mid])/1e3:.4f} µF/cm²  (expect {CM_AXON})")
