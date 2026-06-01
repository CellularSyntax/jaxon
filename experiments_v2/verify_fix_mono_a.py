"""Quick verification that PF threshold for mono_a now matches JAX after fix."""
import sys, pathlib, warnings
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import dataclasses
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jaxley.solver_gate import solve_gate_exponential

from jaxfibers.fibers.mrg import (
    build_mrg, node_indices, section_centers_um,
    V_REST, CM_AXON, G_PAS_MYSA, G_PAS_FLUT, G_PAS_STIN,
)
from jaxfibers.channels.mrg_axnode import AxnodeMyel
from jaxfibers.stim.extracellular import point_source_potentials_mV
from jaxfibers.stim.extracellular_coupled import arrays_from_geometry, integrate
from experiments_v2.utils import PULSES, make_pulse_array, pf_find_threshold, jax_bisect

DT=0.005; TSTOP=8.0; DELAY=1.0; N_NODES=21; SIGMA=0.3; SRC_H=1000.0; CELSIUS=37.0
GNABAR=3.0; GNAPBAR=0.01; GKBAR=0.08; GL=0.007; ENA=50.0; EK=-90.0; EL=-90.0
D = 5.7

_, geom = build_mrg(diameter=D, n_nodes=N_NODES)
nodes = node_indices(geom)
centers = np.array(section_centers_um(geom))
mid = nodes[len(nodes) // 2]
geom_c = dataclasses.replace(geom, cm_uF_cm2=[CM_AXON] * geom.n_comp)
static = arrays_from_geometry(geom_c, DT)
Ve_unit = np.array(point_source_potentials_mV(
    list(centers), 0., SRC_H, float(centers[mid]), -1.0))
is_node = static["is_node"]
A_in    = static["A_in_cm2"]
stype_gp = {"node": 0.0, "mysa": G_PAS_MYSA, "flut": G_PAS_FLUT, "stin": G_PAS_STIN}
g_pas_arr = jnp.asarray([stype_gp[s] for s in geom.section_type])

v0 = jnp.float64(V_REST)
(a_m0,b_m0),(a_h0,b_h0),(a_mp0,b_mp0),(a_s0,b_s0) = AxnodeMyel._alpha_beta(v0, CELSIUS)
m0 = float(a_m0/(a_m0+b_m0)); h0 = float(a_h0/(a_h0+b_h0))
mp0 = float(a_mp0/(a_mp0+b_mp0)); s0 = float(a_s0/(a_s0+b_s0))
state0 = (jnp.where(is_node,m0,0.), jnp.where(is_node,h0,0.),
          jnp.where(is_node,mp0,0.), jnp.where(is_node,s0,0.))

def mfn(Vm, state, dt):
    M,H,MP,S = state
    (am,bm),(ah,bh),(amp_,bmp),(as_,bs) = AxnodeMyel._alpha_beta(Vm, CELSIUS)
    M2 = solve_gate_exponential(M,dt,am,bm); H2 = solve_gate_exponential(H,dt,ah,bh)
    MP2 = solve_gate_exponential(MP,dt,amp_,bmp); S2 = solve_gate_exponential(S,dt,as_,bs)
    g_na=GNABAR*M2**3*H2; g_nap=GNAPBAR*MP2**3; g_k=GKBAR*S2
    g_node=(g_na+g_nap+g_k+GL)*A_in*1e6
    i_node=((g_na+g_nap)*(Vm-ENA)+g_k*(Vm-EK)+GL*(Vm-EL))*A_in*1e6
    g_pas_us=g_pas_arr*A_in*1e6; i_pas=g_pas_us*(Vm-V_REST)
    return jnp.where(is_node,g_node,g_pas_us), jnp.where(is_node,i_node,i_pas), (M2,H2,MP2,S2)

Ve_j = jnp.asarray(Ve_unit, dtype=jnp.float64)
@jax.jit
def run(amp_mA, pulse_shape):
    Ve = Ve_j * (amp_mA / -1.0)
    (Vi,Vp),_ = integrate(static, mfn, state0, Ve, pulse_shape, DT,
                          v_rest=V_REST, record="all")
    Vm_n = jnp.where(is_node[None,:], Vi-Vp, -jnp.inf)
    return jnp.max(Vm_n)

print("MRG D=5.7 mono_a — verifying fix (PF now uses any-node detection)")
print(f"{'PW (ms)':>8}  {'JAX (mA)':>10}  {'PF  (mA)':>10}  {'err %':>8}  {'old PF':>10}")
print("-" * 58)

for pw, pf_old in [(0.20, 19.33), (0.50, 4.71), (1.00, 4.11),
                   (0.02, None),   (0.05, None), (0.10, None)]:
    N = int(TSTOP / DT)
    arr = make_pulse_array("mono_a", pw, N, DT, DELAY)
    spec = PULSES["mono_a"]
    tstop_pf = max(TSTOP, DELAY + pw + 4.0)
    jax_thr = jax_bisect(run, arr, lo=spec.lo, hi=spec.hi)
    pf_thr  = pf_find_threshold("mrg", D, N_NODES, arr, DT, tstop_pf,
                                 lo=spec.lo, hi=spec.hi,
                                 src_height_um=SRC_H, sigma_S_m=SIGMA)
    err = (abs(jax_thr)-abs(pf_thr))/abs(pf_thr)*100
    old = f"{pf_old:.2f}" if pf_old else "new"
    print(f"  {pw:>6.2f}  {jax_thr:>10.4f}  {pf_thr:>10.4f}  {err:>+8.2f}%  {old:>10}")
