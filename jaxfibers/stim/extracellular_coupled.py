"""
Coupled 2N (V_i, V_pax) backward-Euler extracellular solver for the MRG double
cable -- the principled replacement for the quasi-static V_pax (Vi-state) solver.

WHY THIS EXISTS
---------------
The quasi-static solver sets V_pax purely from the external field through the
myelin conductance + periaxonal axial network (g_my*(V_p - V_e) + axial = 0).
That drops the dominant term: at internodes V_pax is capacitively anchored to V_i
through the FULL axolemmal capacitance C_m = CM_AXON * A_axon, whose adiabatic
elimination time constant (~hundreds of us) is far longer than the pulse, so V_pax
should stay near V_i, not collapse to the attenuated field. The error is ~18 mV on
internode V_pax and cannot be absorbed into any scalar effective capacitance
(the exact reduced capacitance is the dense matrix C_m*I + C_m*G_p^{-1}*L_i).

This module instead solves the full coupled (V_i, V_pax) system implicitly, which
is exactly NEURON's nlayer=1 `extracellular` mechanism. Verified against a
numpy dense-LU reference to 5.6e-11 mV; the 2x2 block-Thomas sweep matches dense
LU to 1.5e-11 mV.

UNITS  (identical to the validated reference)
  voltage mV | conductance & C/dt uS | capacitance nF | time ms | current nA
  C[uF/cm^2] * A[cm^2] * 1e3 -> nF ;  g[S/cm^2] * A[cm^2] * 1e6 -> uS
  xraxial[MOhm/cm] * L[cm] -> MOhm ; periaxonal G_p = 1/MOhm -> uS

DISCRETIZATION (per compartment k; unknown 2-vector [V_i, V_pax])
  a_k = Cm_mem/dt + g_eff_k                 membrane admittance
  c_k = Cmy/dt    + g_my_k                  myelin   admittance
  b_k = a_k*Vm_k^n - i_ion_k                membrane RHS source, Vm^n = Vi^n - Vp^n
  d_k = (Cmy/dt)*(Vp_k^n - Ve_k^{n-1})      myelin   RHS source
  D_k = [[a+Gi_diag, -a], [-a, a+c+Gp_diag]]            (internode)
  node periaxonal row PINNED -> [0, 1], RHS entry = Ve  (no myelin: V_pax = V_e)
  off-diagonals diagonal-2x2: -Gi (intra), -Gp (peri, zeroed on node peri rows)
  RHS r_k = [I_stim + b_k, -b_k + c_k*Ve_k + d_k]
"""
from __future__ import annotations
import jax, jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)   # conditioning: g_my spans ~1e-4..inf


# ============================================================ block-Thomas core
def _inv2(M):
    a, b = M[..., 0, 0], M[..., 0, 1]
    c, d = M[..., 1, 0], M[..., 1, 1]
    det = a * d - b * c
    return jnp.stack([jnp.stack([d, -b], -1),
                      jnp.stack([-c, a], -1)], -2) / det[..., None, None]


def block_thomas(D, Low, Up, R):
    """Block-tridiagonal solve, 2x2 blocks, via two lax.scans.
    D,Low,Up: [n,2,2] (Low[k]->k-1, Up[k]->k+1; Low[0],Up[n-1] unused). R: [n,2]."""
    n = D.shape[0]

    def fwd(carry, k):
        Dp_prev, Rp_prev = carry
        W = Low[k] @ _inv2(Dp_prev)
        Dp_k = D[k] - W @ Up[k - 1]
        Rp_k = R[k] - W @ Rp_prev
        return (Dp_k, Rp_k), (Dp_k, Rp_k)

    _, (Dp_tail, Rp_tail) = jax.lax.scan(fwd, (D[0], R[0]), jnp.arange(1, n))
    Dp = jnp.concatenate([D[0][None], Dp_tail], 0)
    Rp = jnp.concatenate([R[0][None], Rp_tail], 0)
    invDp = _inv2(Dp)

    def bwd(x_next, k):
        x_k = invDp[k] @ (Rp[k] - Up[k] @ x_next)
        return x_k, x_k

    x_last = invDp[n - 1] @ Rp[n - 1]
    _, x_rev = jax.lax.scan(bwd, x_last, jnp.arange(n - 2, -1, -1))
    return jnp.concatenate([x_rev[::-1], x_last[None]], 0)


# ============================================================ static assembly
def _build_static(Gi, Gp, is_node):
    """Constant matrix pieces (axial). Gi, Gp length n-1 (bond k joins comp k,k+1)."""
    n = is_node.shape[0]
    intern = 1.0 - is_node.astype(jnp.float64)
    Gi_diag = jnp.zeros(n).at[:-1].add(Gi).at[1:].add(Gi)
    Gp_diag = (jnp.zeros(n).at[:-1].add(Gp).at[1:].add(Gp)) * intern
    Up = jnp.zeros((n, 2, 2)); Low = jnp.zeros((n, 2, 2))
    Up = Up.at[:-1, 0, 0].set(-Gi);              Low = Low.at[1:, 0, 0].set(-Gi)
    Up = Up.at[:-1, 1, 1].set(-Gp * intern[:-1]); Low = Low.at[1:, 1, 1].set(-Gp * intern[1:])
    return Gi_diag, Gp_diag, Up, Low


def _be_step(Vi, Vp, ve, ve_prev, g_eff, i_ion,
             Cm_dt, Cmy_dt, gmy, Gi_diag, Gp_diag, Up, Low, is_node):
    a = Cm_dt + g_eff
    c = Cmy_dt + gmy
    Vm = Vi - Vp
    b = a * Vm - i_ion
    d = Cmy_dt * (Vp - ve_prev)
    D = jnp.stack([jnp.stack([a + Gi_diag, -a], -1),
                   jnp.stack([jnp.where(is_node, 0.0, -a),
                              jnp.where(is_node, 1.0, a + c + Gp_diag)], -1)], -2)
    R = jnp.stack([b, jnp.where(is_node, ve, -b + c * ve + d)], -1)
    X = block_thomas(D, Low, Up, R)
    return X[:, 0], X[:, 1]


# ============================================================ geometry adapter
def arrays_from_geometry(geom, dt):
    """Map a MrgGeometry dataclass to solver arrays (all length n_comp).

    Uses these MrgGeometry fields:
      length_um, diam_um (INNER axon diam), diameter (OUTER fiber diam),
      cm_uF_cm2, xc_myelin_uF_cm2, xg_myelin_S_cm2, Ra_ohm_cm,
      xraxial_Mohm_cm, is_node
    Returns a dict consumed by integrate(); reproduces the validated reference
    arrays to machine precision.
    """
    g = geom
    L_um  = np.asarray(g.length_um, float)
    din   = np.asarray(g.diam_um,   float)        # inner axon diameter
    dout  = np.asarray(g.diameter,  float)        # outer fiber diameter
    is_nd = np.asarray(g.is_node,   bool)
    A_in  = np.pi * din  * L_um * 1e-8            # cm^2 (um^2 -> cm^2 = 1e-8)
    A_out = np.pi * dout * L_um * 1e-8            # cm^2

    Cm_mem = np.asarray(g.cm_uF_cm2, float) * A_in  * 1e3                  # nF
    Cmy    = np.where(is_nd, 0.0,
                      np.asarray(g.xc_myelin_uF_cm2, float) * A_out * 1e3) # nF
    gmy    = np.where(is_nd, 0.0,
                      np.asarray(g.xg_myelin_S_cm2, float) * A_out * 1e6)  # uS

    # intracellular axial Gi (uS), bond k between comp k, k+1
    rin   = din / 2 * 1e-4                                                 # cm
    Lcm   = L_um * 1e-4
    Rhalf = np.asarray(g.Ra_ohm_cm, float) * (Lcm / 2) / (np.pi * rin ** 2)  # Ohm
    Gi    = 1e6 / (Rhalf[:-1] + Rhalf[1:])                                # uS

    # periaxonal axial Gp (uS)
    RhP   = np.asarray(g.xraxial_Mohm_cm, float) * (Lcm / 2)              # MOhm
    Gp    = 1.0 / (RhP[:-1] + RhP[1:])                                    # 1/MOhm = uS

    return dict(
        Cm_dt   = jnp.asarray(Cm_mem / dt),
        Cmy_dt  = jnp.asarray(Cmy / dt),
        gmy     = jnp.asarray(gmy),
        A_in_cm2= jnp.asarray(A_in),       # for converting channel S/cm^2 -> uS, mA/cm^2 -> nA
        is_node = jnp.asarray(is_nd),
        **dict(zip(("Gi_diag", "Gp_diag", "Up", "Low"),
                   _build_static(jnp.asarray(Gi), jnp.asarray(Gp), jnp.asarray(is_nd)))),
    )


# ============================================================ time integration
def integrate(static, membrane_fn, state0, Ve, pulse_mask, dt, v_rest=-80.0,
              record="center", field_on_before_pulse=False, center_comp=None,
              i_intra=None):
    """Backward-Euler integrate the coupled system.

    static       : dict from arrays_from_geometry(geom, dt)
    membrane_fn  : (Vm[n], state, dt) -> (g_eff[n] uS, i_ion[n] nA, new_state)
                   Wire this to your channels: nodes use AxnodeMyel, internodes
                   use the section pas leak. Multiply per-area g (S/cm^2) and
                   i (mA/cm^2) by static['A_in_cm2'] * 1e6 / 1e6 respectively
                   (S/cm^2 * cm^2 * 1e6 = uS ; mA/cm^2 * cm^2 * 1e6 = nA).
    state0       : initial channel/gate state (pytree)
    Ve           : [n] extracellular potential profile (mV) already scaled to the
                   test current. Multiplied element-wise by pulse_mask[s] each step.
    pulse_mask   : [nsteps] bool OR float. Bool: on/off. Float: signed shape
                   (e.g. +1 cathodic / -1 anodic for biphasic). Step s covers
                   time interval [s*dt, (s+1)*dt]; output index s = Vm at (s+1)*dt.
    record       : 'center' (return center-node Vm trace) or 'all' (full Vi,Vp)
    i_intra      : optional [nsteps, n_comp] intracellular injected current (nA).
                   Positive = depolarizing. Set non-injecting compartments to 0.

    Returns: trace of recorded quantity (and final state).
    """
    n = static["is_node"].shape[0]
    is_node = static["is_node"]
    if center_comp is None:                      # middle node; computed eagerly (jit-safe)
        nd = np.where(np.asarray(is_node))[0]
        center_comp = int(nd[len(nd) // 2])
    center = center_comp
    Ve = jnp.asarray(Ve)
    nsteps = pulse_mask.shape[0]
    # Float shape supports signed (biphasic) waveforms; bool mask still works (True=1, False=0).
    shape      = jnp.asarray(pulse_mask, dtype=jnp.float64)
    shape_prev = jnp.concatenate([
        jnp.array([1.0 if field_on_before_pulse else 0.0], dtype=jnp.float64),
        shape[:-1],
    ])
    i_intra_j = jnp.asarray(i_intra, dtype=jnp.float64) if i_intra is not None else None

    def step(carry, s):
        Vi, Vp, st = carry
        ve      = Ve * shape[s]
        ve_prev = Ve * shape_prev[s]
        g_eff, i_ion, st = membrane_fn(Vi - Vp, st, dt)
        if i_intra_j is not None:
            i_ion = i_ion - i_intra_j[s]   # positive i_intra depolarizes (subtracts from outward)
        Vi2, Vp2 = _be_step(Vi, Vp, ve, ve_prev, g_eff, i_ion,
                            static["Cm_dt"], static["Cmy_dt"], static["gmy"],
                            static["Gi_diag"], static["Gp_diag"],
                            static["Up"], static["Low"], is_node)
        out = (Vi2[center] - Vp2[center]) if record == "center" else (Vi2, Vp2)
        return (Vi2, Vp2, st), out

    Vi0 = jnp.full(n, v_rest); Vp0 = jnp.zeros(n)
    (Vi_f, Vp_f, st_f), trace = jax.lax.scan(step, (Vi0, Vp0, state0), jnp.arange(nsteps))
    return trace, st_f


# ============================================================ recording variant
def integrate_recording(static, membrane_fn, state0, Ve, pulse_mask, dt,
                        v_rest=-80.0, center_comp=None, i_intra=None):
    """Like integrate(record='center') but also records gate state (M, H, MP, S)
    at the center node at every time step.

    State tuple must be (M, H, MP, S) as returned by the AxnodeMyel membrane_fn.
    i_intra : optional [nsteps, n_comp] intracellular current (nA), positive = depolarizing.

    Returns:
        vm_trace  : [nsteps] Vm at center node (mV)
        m_trace   : [nsteps] M gate at center node
        h_trace   : [nsteps] H gate at center node
        mp_trace  : [nsteps] MP gate at center node
        s_trace   : [nsteps] S gate at center node
        final_state : final (M, H, MP, S) pytree
    """
    n = static["is_node"].shape[0]
    is_node = static["is_node"]
    if center_comp is None:
        nd = np.where(np.asarray(is_node))[0]
        center_comp = int(nd[len(nd) // 2])
    center = center_comp
    Ve = jnp.asarray(Ve)
    nsteps = pulse_mask.shape[0]
    shape      = jnp.asarray(pulse_mask, dtype=jnp.float64)
    shape_prev = jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), shape[:-1]])
    i_intra_j = jnp.asarray(i_intra, dtype=jnp.float64) if i_intra is not None else None

    def step(carry, s):
        Vi, Vp, st = carry
        ve      = Ve * shape[s]
        ve_prev = Ve * shape_prev[s]
        g_eff, i_ion, st2 = membrane_fn(Vi - Vp, st, dt)
        if i_intra_j is not None:
            i_ion = i_ion - i_intra_j[s]
        Vi2, Vp2 = _be_step(Vi, Vp, ve, ve_prev, g_eff, i_ion,
                            static["Cm_dt"], static["Cmy_dt"], static["gmy"],
                            static["Gi_diag"], static["Gp_diag"],
                            static["Up"], static["Low"], is_node)
        M2, H2, MP2, S2 = st2
        out = (Vi2[center] - Vp2[center], M2[center], H2[center], MP2[center], S2[center])
        return (Vi2, Vp2, st2), out

    Vi0 = jnp.full(n, v_rest); Vp0 = jnp.zeros(n)
    (_, _, st_f), (vm_t, m_t, h_t, mp_t, s_t) = jax.lax.scan(
        step, (Vi0, Vp0, state0), jnp.arange(nsteps)
    )
    return vm_t, m_t, h_t, mp_t, s_t, st_f


# ============================================================ threshold search
def find_threshold(static, membrane_fn_factory, Ve_unit, pulse_mask, dt,
                   v_thresh=-30.0, lo=-1.0, hi=-0.01, tol=1e-4, v_rest=-80.0):
    """Bisection on cathodic current amplitude (mA). Ve_unit is the extracellular
    profile at -1 mA; the trial Ve = Ve_unit * (amp / -1.0). A trial 'fires' if the
    center-node Vm crosses v_thresh. membrane_fn_factory() returns a fresh
    (membrane_fn, state0) per trial (so gate state resets)."""
    Ve_unit = np.asarray(Ve_unit)

    def fires(amp):
        membrane_fn, state0 = membrane_fn_factory()
        Ve = jnp.asarray(Ve_unit * (amp / -1.0))
        trace, _ = integrate(static, membrane_fn, state0, Ve, pulse_mask, dt,
                             v_rest=v_rest, record="center")
        return bool(np.max(np.asarray(trace)) > v_thresh)

    assert fires(hi) != fires(lo), "bracket does not straddle threshold"
    fire_hi = fires(hi)
    while abs(hi - lo) > tol:
        mid = 0.5 * (lo + hi)
        if fires(mid) == fire_hi:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)