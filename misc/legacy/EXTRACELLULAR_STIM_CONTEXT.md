# MRG Extracellular Stimulation in JAXley — Context for Next Agent

## Goal

Match the extracellular stimulation threshold of the NEURON/PyFibers MRG double-cable myelinated fiber model to **<5% error** across diameters D = 5.7, 10.0, and 14.0 µm.

**Setup**: Cathodic point-source electrode, 1 mm above fiber center, σ = 0.3 S/m, 0.1 ms rectangular pulse, dt = 0.005 ms, T = 37°C, N = 21 nodes, D = 10 µm.

**NEURON/PyFibers reference**: −0.18317 mA (threshold amplitude).

---

## Summary of Results So Far

| Approach | Threshold (D=10) | Error | Notes |
|---|---|---|---|
| Activating function (Vm-state, Jaxley API) | ~−0.22 mA | ~20% | misses capacitive coupling |
| Vi-state + CM_AXON everywhere | −0.198 mA | 8.3% | too hard to fire |
| **Vi-state + series-formula Cm (current best)** | **−0.170 mA** | **7.4%** | too easy to fire |
| Vi-state + CM_AXON + explicit cap. coupling | −6.41 mA | 3401% | approach is broken (see below) |

The target (<5%) has not yet been reached. The best working implementation is the "Vi-state + series-formula Cm" approach with 7.4% error.

---

## The MRG Double-Cable Model: NEURON Architecture

### What PyFibers (NEURON) Actually Does

The MRG fiber in PyFibers uses NEURON's **`extracellular` mechanism** to implement a full double-cable model. Each myelinated section (MYSA, FLUT, STIN) has:

- An axon membrane cable equation with `cm = CM_AXON × axon_diam/fiber_diam`
- A separate periaxonal space layer with `xraxial` (resistance), `xc` (capacitance), `xg` (conductance)
- External potentials set via `e_extracellular` at each section

The critical design is: **NEURON solves the axon membrane and periaxonal space simultaneously and coupled**.

### PyFibers MRG Source (exact code)

File: `reference_code/pyfibers/src/pyfibers/models/mrg.py`

```python
def create_node(self, index, node_type):
    rhoa = self.mrg_params["rhoa"]        # 0.7e6 Ω·µm = 70 Ω·cm
    node_diam = self.mrg_params["node_diam"]  # e.g. 3.3 µm for D=10
    nodelength = self.mrg_params["node_length"]  # 1.0 µm

    node = h.Section(name=f"{node_type} node {index}")
    node.nseg = 1
    node.diam = node_diam       # ← uses node_diam (3.3 µm), NOT fiber_diam
    node.L = nodelength
    node.Ra = rhoa / 10000      # = 70 Ω·cm
    node.cm = 2                 # CM_AXON = 2 µF/cm²

    space_p1 = 0.002  # µm, periaxonal gap at node/MYSA
    rpn0 = (rhoa * 0.01) / (math.pi * ((((node_diam/2) + space_p1)**2) - ((node_diam/2)**2)))

    node.insert("axnode_myel")
    node.insert("extracellular")
    node.xraxial[0] = rpn0    # MΩ/cm
    node.xc[0] = 0            # short circuit (no myelin at node)
    node.xg[0] = 1e10         # short circuit → V_pax = V_e at nodes

    return node


def create_mysa(self, i):
    rhoa = self.mrg_params["rhoa"]      # 0.7e6
    mycm = self.mrg_params["mycm"]      # 0.1 µF/cm² per myelin leaflet
    mygm = self.mrg_params["mygm"]      # 0.001 S/cm² per myelin leaflet
    nl   = self.mrg_params["nl"]        # 120 for D=10
    mysa_diam = self.mrg_params["node_diam"]       # 3.3 µm
    paralength1 = self.mrg_params["paranodal_length_1"]  # 3.0 µm

    space_p1 = 0.002
    rpn1 = (rhoa * 0.01) / (math.pi * ((((mysa_diam/2) + space_p1)**2) - ((mysa_diam/2)**2)))

    mysa = h.Section(name="mysa " + str(i))
    mysa.nseg = 1
    mysa.diam = self.diameter   # ← OUTER FIBER diameter (10 µm), NOT axon_diam!
    mysa.L = paralength1
    mysa.Ra = rhoa * (1 / (mysa_diam / self.diameter)**2) / 10000  # scaled Ra
    mysa.cm = 2 * mysa_diam / self.diameter   # scaled cm so area×cm = CM_AXON×axon_area
    mysa.insert("pas")
    mysa.g_pas = 0.001 * mysa_diam / self.diameter   # scaled g_pas
    mysa.e_pas = self.v_rest

    mysa.insert("extracellular")
    mysa.xraxial[0] = rpn1           # MΩ/cm
    mysa.xc[0] = mycm / (nl * 2)     # µF/cm², myelin sheath capacitance
    mysa.xg[0] = mygm / (nl * 2)     # S/cm², myelin sheath conductance

    return mysa


def create_stin(self, i):
    rhoa  = self.mrg_params["rhoa"]
    mycm  = self.mrg_params["mycm"]   # 0.1
    mygm  = self.mrg_params["mygm"]   # 0.001
    nl    = self.mrg_params["nl"]
    axon_diam  = self.mrg_params["axon_diam"]   # 6.9 µm for D=10
    flut_length = self.mrg_params["paranodal_length_2"]   # 46 µm

    interlength = (self.delta_z - nodelength - 2*paralength1 - 2*flut_length) / 6

    space_i = 0.004
    rpx = (rhoa * 0.01) / (math.pi * ((((axon_diam/2) + space_i)**2) - ((axon_diam/2)**2)))

    stin = h.Section(name="stin " + str(i))
    stin.nseg = 1
    stin.diam = self.diameter   # ← OUTER FIBER diameter again
    stin.L = interlength
    stin.Ra = rhoa * (1 / (axon_diam / self.diameter)**2) / 10000
    stin.cm = 2 * axon_diam / self.diameter
    stin.insert("pas")
    stin.g_pas = 0.0001 * axon_diam / self.diameter
    stin.e_pas = self.v_rest

    stin.insert("extracellular")
    stin.xraxial[0] = rpx
    stin.xc[0] = mycm / (nl * 2)
    stin.xg[0] = mygm / (nl * 2)

    return stin
```

### Key PyFibers Geometry for D=10 µm

| Parameter | Value |
|---|---|
| fiber_diam (outer) | 10.0 µm |
| node_diam | 3.3 µm |
| axon_diam | 6.9 µm |
| nl (myelin lamellae) | 120 |
| delta_z (internodal length) | 1150 µm |
| node length | 1.0 µm |
| MYSA length | 3.0 µm |
| FLUT length | 46.0 µm |
| STIN length | (1150 - 1 - 6 - 92) / 6 = 175.17 µm |
| rhoa | 0.7e6 Ω·µm = 70 Ω·cm |
| CM_AXON | 2.0 µF/cm² |
| mycm (per leaflet) | 0.1 µF/cm² |
| mygm (per leaflet) | 0.001 S/cm² |
| xc_myel = mycm/(2×nl) | 4.167e-4 µF/cm² |
| xg_myel = mygm/(2×nl) | 4.167e-6 S/cm² |

### Critical NEURON Geometry Trick

NEURON sections at MYSA/FLUT/STIN use the **outer fiber diameter** (`self.diameter = 10 µm`) for `diam`. This requires scaling all per-area quantities:

```
Ra_NEURON     = Ra_true × (fiber_diam / axon_diam)²    → preserves axial G
cm_NEURON     = CM_AXON × (axon_diam / fiber_diam)      → preserves membrane C
g_pas_NEURON  = g_pas_true × (axon_diam / fiber_diam)   → preserves membrane G
```

So when NEURON multiplies these by `π × fiber_diam × L`, it recovers the correct physical values based on the inner axon diameter.

---

## Our JAXley Implementation

### Architecture

We bypass Jaxley's native stimulation API entirely. Instead we have a custom backward-Euler integrator in `jaxfibers/stim/mrg_extracellular_solver.py` that:

1. Uses **V_i (intracellular voltage)** as the state variable
2. Precomputes the **quasi-static periaxonal voltage V_pax** from `solve_vpax_static()`
3. Integrates the cable equation: channels see `Vm = Vi - Vpax`, but axial currents are driven by `Vi`
4. Uses **Thomas algorithm** (JAX-compatible tridiagonal solve) via `jax.lax.scan`

### Why Vi-State (Not Vm-State)

The Vm-state formulation writes the cable equation in terms of `Vm = Vi - Vpax`:
```
Cm × dVm/dt = G_ax × d²Vm/dz² + (1/Ra) d²Vpax/dz² - I_ion(Vm) + I_stim
```

The second term is the **activating function** (Rattay 1986). This approach was tried first and gave ~20.5% error, because it misses the **capacitive coupling term** `-Cm × dVpax/dt`.

The Vi-state writes the cable equation in terms of `Vi`:
```
Cm × dVm/dt = Cm × d(Vi-Vpax)/dt = G_ax × d²Vi/dz² - I_ion(Vi-Vpax) + I_stim
→ Cm × dVi/dt = G_ax × d²Vi/dz² - I_ion(Vi-Vpax) + I_stim + Cm × dVpax/dt
```

This formulation **should** capture both the activating function (spatial term in Vpax) and the capacitive coupling (temporal term in Vpax) automatically. However, our implementation does NOT yet include the `Cm × dVpax/dt` term explicitly — see the unresolved issue section below.

### Geometry: `jaxfibers/fibers/mrg.py`

```python
NODE_LENGTH = 1.0       # µm
MYSA_LENGTH = 3.0       # µm
RHOA = 0.7e6            # Ω·µm = 70 Ω·cm
MYCM = 0.1              # µF/cm² per myelin leaflet
MYGM = 0.001            # S/cm² per myelin leaflet
CM_AXON = 2.0           # µF/cm²
SPACE_P1 = 0.002        # µm at node and MYSA
SPACE_P2 = 0.004        # µm at FLUT and STIN
G_PAS_MYSA = 1.0e-3     # S/cm²
G_PAS_FLUT = 1.0e-4     # S/cm²
G_PAS_STIN = 1.0e-4     # S/cm²

_MRG_DISCRETE = {
    10.0: dict(delta_z=1150, paranodal_length_2=46, axon_diam=6.9, node_diam=3.3, nl=120),
    ...
}

def _mrg_geometry(diameter, n_nodes):
    xc_myel = MYCM / (2 * nl)            # 4.167e-4 µF/cm²
    xg_myel = MYGM / (2 * nl)            # 4.167e-6 S/cm²
    cm_myel = 1.0 / (1.0/CM_AXON + 1.0/xc_myel)  # ≈ 4.166e-4 µF/cm² (series formula)
    g_mysa  = 1.0 / (1.0/G_PAS_MYSA + 1.0/xg_myel)  # ≈ 4.167e-6 S/cm²
    g_flut  = 1.0 / (1.0/G_PAS_FLUT  + 1.0/xg_myel)  # ≈ 4.167e-6 S/cm²
    g_stin  = 1.0 / (1.0/G_PAS_STIN  + 1.0/xg_myel)  # ≈ 4.167e-6 S/cm²

    # Period template: (type, L, diam_um, Ra, cm, gleak, is_node, xraxial, xc_myel, xg_myel)
    period_template = (
        ("node",  1.0,        node_d, 70, CM_AXON, 0.0,    True,  rpn, 0.0,    1e10),
        ("mysa",  3.0,        node_d, 70, cm_myel, g_mysa, False, rpn, xc_myel, xg_myel),
        ("flut",  flut_l,     axon_d, 70, cm_myel, g_flut, False, rpf, xc_myel, xg_myel),
        ("stin",  stin_l,     axon_d, 70, cm_myel, g_stin, False, rpf, xc_myel, xg_myel),
        # (×6 STINs)
        ("flut",  flut_l,     axon_d, 70, cm_myel, g_flut, False, rpf, xc_myel, xg_myel),
        ("mysa",  3.0,        node_d, 70, cm_myel, g_mysa, False, rpn, xc_myel, xg_myel),
    )
```

**Key difference from NEURON**: We store `diam_um = axon_diam` (inner axon diameter) for FLUT/STIN sections, and `diam_um = node_diam` for MYSA. NEURON uses `fiber_diam` (outer, = 10 µm) and compensates via scaling. This difference is **numerically equivalent for Ra and g_pas** but matters for capacitance (see below).

### Periaxonal Voltage: `jaxfibers/stim/extracellular.py`

```python
def solve_vpax_static(v_ext_mV, length_um, xraxial_Mohm_cm, xg_S_cm2, is_node, fiber_diam_um):
    """Quasi-static periaxonal voltage from the steady-state periaxonal cable equation.
    
    For each myelinated section:
        G_myelin × (V_pax - 0) + G_inter_left × (V_pax - V_pax_left) + G_inter_right × ... = G_myelin × V_ext
    
    Boundary conditions: V_pax = V_ext at nodes (xg=1e10 → short circuit).
    
    G_myelin [S] = xg_S_cm2 × π × fiber_diam [cm] × L [cm]
    G_inter  [S] = 1 / (sum of half-resistances × 1e6)
    """
    # (uses fiber_diam_um = OUTER fiber diameter for G_myelin area)
```

**Periaxonal RC time constants** are all << dt = 5 µs:
- τ_MYSA ≈ 8.7 ns
- τ_STIN ≈ 1.1 µs
→ Quasi-static is valid.

### Main Solver: `jaxfibers/stim/mrg_extracellular_solver.py` (current state)

```python
def _build_geometry_arrays(geom):
    n = geom.n_comp
    L = np.array(geom.length_um) * 1e-4   # cm
    r = np.array(geom.diam_um) / 2.0 * 1e-4   # cm (inner axon diam per section)
    Ra = np.array(geom.Ra_ohm_cm)

    A = np.pi * (2 * r) * L   # membrane area [cm²] using inner axon diam

    # *** CURRENT IMPLEMENTATION: series formula Cm ***
    cm_arr = np.array(geom.cm_uF_cm2)   # ≈ xc_myel=4.167e-4 for myelinated, 2.0 for nodes
    Cm_nF = cm_arr * A * 1e3            # gives tiny Cm at myelinated sections

    # Axial conductance using inner axon diam (= same as NEURON numerically)
    R_half_Ohm = Ra * (L / 2.0) / (np.pi * r**2)
    R_ax_Ohm = R_half_Ohm[:-1] + R_half_Ohm[1:]
    Gax_uS = 1e6 / R_ax_Ohm

    # G_pas using G_PAS_MYSA/FLUT/STIN × inner axon area (= same as NEURON numerically)
    Gpas_S_cm2 = np.zeros(n)
    for i, stype in enumerate(geom.section_type):
        if geom.is_node[i]: Gpas_S_cm2[i] = 0.0
        elif stype == "mysa": Gpas_S_cm2[i] = G_PAS_MYSA
        elif stype == "flut": Gpas_S_cm2[i] = G_PAS_FLUT
        else: Gpas_S_cm2[i] = G_PAS_STIN
    Gpas_uS = Gpas_S_cm2 * A * 1e6

    return jnp.array(Cm_nF), jnp.array(Gax_uS), jnp.array(Gpas_uS), ...


def integrate_mrg_extracellular(geom, vpax_waveform_mV, stim_waveform_nA, dt_ms, t_max_ms, record_comps, temperature=37.0):
    """Backward-Euler Vi-state integrator."""
    Cm, Gax, Gpas, Epas, is_node, A_cm2 = _build_geometry_arrays(geom)

    vpax = jnp.asarray(vpax_waveform_mV)  # (n_comp, n_steps+1)

    # Initial conditions: Vi = V_rest, gates at SS(V_rest)
    Vi0 = jnp.full((n_comp,), V_REST)   # -80 mV
    m0, h0, mp0, s0 = _init_gates(V_REST, temperature)

    # Constant tridiagonal parts (axial conductance)
    lower = -Gax
    upper = -Gax
    diag_ax = zeros; diag_ax[:-1] += Gax; diag_ax[1:] += Gax

    def step(carry, t_idx):
        Vi, M, H, MP, S = carry
        vp_old = vpax[:, t_idx]        # V_pax at t (for gate kinetics)
        vp_new = vpax[:, t_idx + 1]   # V_pax at t+dt (for RHS)
        I_s = stim[:, t_idx + 1]

        # Gates see membrane voltage Vm = Vi - Vpax
        Vm_old = Vi - vp_old
        M_n, H_n, MP_n, S_n = _update_gates(Vm_old[node_idx], M, H, MP, S, dt_ms, temperature)

        # Chord conductance and effective reversal (Vm frame)
        Gnode_uS = _axnode_conductance_mS_cm2(M_n, H_n, MP_n, S_n) * A[node_idx] * 1e6
        Enode_mV = _axnode_Eeff(M_n, H_n, MP_n, S_n)

        G_ch = zeros
        G_ch[node_idx] = Gnode_uS
        G_ch[myel_idx] = Gpas[myel_idx]

        diag = Cm / dt_ms + diag_ax + G_ch

        # RHS: the key Vi-frame transformation
        # G_ch × (E_eff + vp_new) converts reversal potentials from Vm to Vi frame.
        # This captures the spatial activating function through the Vpax gradient.
        E_eff = full(V_REST); E_eff[node_idx] = Enode_mV
        rhs = (Cm / dt_ms) * Vi + G_ch * (E_eff + vp_new) + I_s

        Vi_new = _tridiagonal_solve(lower, diag, upper, rhs)

        # Record Vm = Vi - Vpax at new time
        rec = Vi_new[record_idx] - vp_new[record_idx]
        return (Vi_new, ...), rec
```

### Gate Kinetics: `jaxfibers/channels/mrg_axnode.py`

The AxnodeMyel channel reproduces AXNODE_myel.mod with Q10-scaled kinetics:
```python
q10_1 = 2.2 ** ((celsius - 20.0) / 10.0)   # for m, mp gates
q10_2 = 2.9 ** ((celsius - 20.0) / 10.0)   # for h, s gates  
q10_3 = 3.0 ** ((celsius - 36.0) / 10.0)   # for s gate
```

At T=37°C, V=-80 mV (resting):
- m_inf = 0.0732 (confirmed correct analytically: α_m×Q10=1.23, β_m×Q10=15.6)
- h_inf = 0.6207
- mp_inf = 0.2026
- s_inf = 0.0430
- E_eff_rest = -79.88 mV ✓
- G_ion_rest = 0.0012 µS (tiny compared to Gax ≈ 6.11 µS node-MYSA) ✓

---

## What Has Been Ruled Out

1. **Gate kinetics errors**: Verified analytically. m=0.0732 at V=-80 is correct for McIntyre Q10-scaled model.
2. **Node geometry errors**: A_node = 1.037e-7 cm², Cm_node = 2.074e-4 nF, Cm_node/area = 2.0 µF/cm² ✓
3. **Pulse timing**: 20 steps during 0.1 ms pulse at dt=0.005 ms ✓
4. **End effects (finite fiber length)**: N=11, 21, 31, 51 all give same threshold (-0.170 mA) → not end effects
5. **Axial conductance**: Ra scaling formula verified: Gax(node-MYSA) ≈ 6.11 µS ✓
6. **G_pas correctness**: G_PAS_MYSA × node_area = NEURON's `g_pas × fiber_area` numerically ✓
7. **Periaxonal voltage at center node**: Vpax = Ve (Dirichlet) → center node sees full external potential ✓

---

## Numerical Diagnostics at Threshold

### At amp = −0.183 mA (NEURON threshold)
- V_pax at center node: −48.54 mV (for −0.183 mA, 1 mm height)
- Vm at onset (step 200): **−42.43 mV** (depolarized ~37.6 mV from rest)
- Vm traces to AP by t ≈ 1.095 ms, max Vm ≈ +11.26 mV ✓ fires

### At amp = −0.170 mA (our threshold)
- Vm at onset: −45.1 mV
- Fires (just barely), max Vm ≈ +9.53 mV

### Implication
Our model depolarizes the center node **more than NEURON does** for the same amplitude, hence fires at lower amplitude. The effective coupling from Vpax to Vm is too strong in our model.

---

## The Core Problem: Capacitance at Myelinated Sections

### Two Failed Extremes

**CM_AXON everywhere (threshold = −0.198 mA, 8.3% error — too hard to fire)**:
```python
Cm_nF = CM_AXON * A * 1e3   # 2.0 µF/cm² for all sections
```
Large Cm at internodal sections provides "inertia" — system resists change.

**Series formula (threshold = −0.170 mA, 7.4% error — too easy to fire)**:
```python
cm_myel = 1/(1/CM_AXON + 1/xc_myel) ≈ xc_myel = 4.167e-4 µF/cm²
Cm_nF = cm_arr * A * 1e3   # tiny Cm at myelinated sections
```
Tiny Cm → myelinated sections adjust instantly → stronger coupling → easier to fire.

**NEURON's actual value is between these two extremes** (−0.183 mA).

### Failed Attempt: Explicit Capacitive Coupling (threshold = −6.41 mA, BROKEN)

The physical equation is:
```
Cm_axon × d(Vi - Vpax)/dt = G_ax × d²Vi/dz² - I_ion(Vi-Vpax)
→ Cm_axon × dVi/dt = G_ax × d²Vi/dz² - I_ion(Vi-Vpax) + Cm_axon × dVpax/dt
```

The `Cm_axon × dVpax/dt` term (capacitive coupling) is missing from our RHS. Attempting to add it:
```python
# Changed Cm to CM_AXON everywhere, then added:
rhs = (Cm / dt_ms) * (Vi + vp_new - vp_old) + G_ch * (E_eff + vp_new) + I_s
```

This caused threshold to jump to −6.41 mA (3401% error).

**Why it fails**: Long STIN sections have `Cm_STIN = CM_AXON × π × 6.9e-4 cm × 175.17e-4 cm = 7.60e-3 nF` — about 37× larger than a node. With `(Cm/dt) × (vp_new - vp_old)` evaluated at STIN sections where Vpax ≈ −40 mV at threshold:
```
(7.60e-3 nF / 0.005 ms) × (−40 mV) = 1.52 µS × (−40 mV) = −60.8 nA
```
per STIN section! There are 60 STIN sections total. This overwhelms the center node's 2 nA signal, driving Vi at STIN sections strongly toward `−80 + Vpax_STIN`, which creates a hyperpolarizing axial current from internodal sections into the center node.

**In NEURON, this doesn't happen** because the myelin sheath limits how much charge can flow from periaxonal space to bath — the periaxonal space dynamics self-consistently regulate the charge redistribution. Our quasi-static Vpax bypasses this regulation.

### The Fundamental Tension

The quasi-static Vpax approximation is valid (τ_RC << dt). But there are two inconsistent choices for Cm:

1. **Cm = CM_AXON (axon membrane capacitance)**: physically correct for the cable equation, but then the capacitive coupling `Cm × dVpax/dt` must also be added — and this causes the blown-up threshold problem above.

2. **Cm = series formula ≈ xc_myel (effective capacitance from intracellular to bath)**: intuitively captures the fact that charge flowing from intracellular to bath passes through both CM_AXON and xc_myel in series. This approximation may implicitly capture the capacitive coupling. Gives 7.4% error, which is the best result so far.

The mismatch between NEURON (−0.183 mA) and our model (−0.170 mA) with series Cm means the series approximation is slightly wrong. NEURON is between the two Cm extremes:
- CM_AXON: −0.198 (too hard)
- series: −0.170 (too easy)
- NEURON: −0.183 (between)

---

## Key Unresolved Question

**What is the correct effective capacitance for the Vi-state cable equation under the quasi-static Vpax approximation?**

The answer must be something between `xc_myel × axon_area` and `CM_AXON × axon_area`. In NEURON's full dynamic simulation, the periaxonal voltage adjusts in response to BOTH the external field AND the charge flowing from the axon membrane. The quasi-static Vpax ignores the feedback from Vi onto Vpax. This feedback changes the effective Cm.

### Hypothesis

In NEURON, the effective capacitance for the cable equation at a myelinated section (accounting for the coupling between axon membrane and periaxonal space) is:

```
C_eff = CM_AXON × axon_area × C_xc / (CM_AXON × axon_area + C_xc)
```

where `C_xc = xc_myel × fiber_outer_area = xc_myel × π × fiber_diam × L`.

Note: **fiber_outer_area uses the OUTER diameter** (10 µm), NOT the inner axon diameter (6.9 µm).

For D=10 STIN:
- `CM_AXON × axon_area = 2.0 × π × 6.9e-4 × 175.17e-4 = 7.60e-6 µF`
- `C_xc = 4.167e-4 × π × 10e-4 × 175.17e-4 = 2.295e-9 µF` (xc dominates → series ≈ xc)

Our series formula uses `xc_myel × axon_area = xc_myel × π × axon_diam × L` but NEURON's xc capacitance uses `fiber_outer_area`:
- Our C_eff ≈ `4.167e-4 × π × 6.9e-4 × L` (axon diam)
- NEURON's actual series ≈ `4.167e-4 × π × 10e-4 × L` (outer fiber diam)

**Ratio = 10.0/6.9 = 1.449**. Our Cm at myelinated sections is 1.449× too small.

If we correct the area computation in Cm only (using outer fiber diameter for the xc part), the series formula becomes:
```
C_series = CM_AXON × axon_area × xc_myel × outer_area / (CM_AXON × axon_area + xc_myel × outer_area)
         ≈ xc_myel × outer_area     (since xc_myel << CM_AXON)
         = xc_myel × π × fiber_diam × L
```

This is **1.449× larger** than our current value. Since larger Cm → harder to fire → threshold moves toward −0.183 mA. This could close the 7.4% gap.

**To test**: In `_build_geometry_arrays`, replace:
```python
cm_arr = np.array(geom.cm_uF_cm2)
Cm_nF = cm_arr * A * 1e3   # uses axon_diam area
```
with:
```python
# Use outer fiber diameter for the periaxonal capacitance area
L_arr = np.array(geom.length_um) * 1e-4
fiber_diam_cm = geom.diameter * 1e-4
A_outer = np.pi * fiber_diam_cm * L_arr   # outer area for myelin Cm
xc_arr = np.array(geom.xc_myelin_uF_cm2)  # ≈ 4.167e-4 for myelinated, 0 for nodes
cm2_uF_cm2 = np.array(geom.cm_uF_cm2)     # CM_AXON for nodes
# For myelinated: use xc_myel × outer_area
Cm_nF = np.where(
    np.array(geom.is_node),
    CM_AXON * A * 1e3,           # nodes: CM_AXON × node_area
    xc_arr * A_outer * 1e3       # myelinated: xc_myel × outer_fiber_area
)
```

For D=10: change ratio of myelinated Cm = 10.0/6.9 = 1.449.
Expected threshold shift: 7.4% gap × partial correction → target < 5%.

---

## Code Organization

```
jaxley_fibers/
├── jaxfibers/
│   ├── fibers/
│   │   └── mrg.py              — MrgGeometry dataclass, _mrg_geometry(), build_mrg()
│   ├── channels/
│   │   └── mrg_axnode.py       — AxnodeMyel channel (all 4 gates)
│   └── stim/
│       ├── extracellular.py    — point_source_potentials_mV(), solve_vpax_static()
│       └── mrg_extracellular_solver.py  — integrate_mrg_extracellular() (the main solver)
├── reference_code/
│   └── pyfibers/
│       └── src/pyfibers/models/mrg.py   — PyFibers NEURON source (reference)
└── experiments/
    ├── _threshold_vi_state.py  — bisection test (gives −0.170 mA)
    ├── _diag_channel_check.py  — gate kinetics / geometry verification
    ├── _diag_n_nodes.py        — shows N-independence (end effects ruled out)
    └── _diag_vm_trace.py       — Vm trace at onset
```

---

## Running the Current Best

```bash
conda run -n jaxley_fibers python experiments/_threshold_vi_state.py
# Output:
# Vi-state integrator threshold: -0.16959 mA  (4s for 14 iters)
# NEURON (PyFibers) reference:   -0.18317 mA
# Error: 7.4%
```

---

## Where to Start

1. **Test the outer-area hypothesis**: Modify `_build_geometry_arrays` in `mrg_extracellular_solver.py` to use `xc_myel × π × fiber_diam × L` (outer fiber diameter) for myelinated section capacitance instead of `xc_myel × π × axon_diam × L` (inner axon diameter). Run `_threshold_vi_state.py` and check if threshold moves toward −0.183 mA.

2. **If that doesn't work**: Consider whether the quasi-static Vpax approximation itself introduces a systematic error. Compare the Vpax profile from `solve_vpax_static` against known NEURON outputs. Perhaps the myelin conductance at internodal sections (which determines Vpax there) differs between our model and NEURON.

3. **Check the exact Vm trace at onset vs NEURON**: Can we run PyFibers and record the Vm trace at the center node for amp = −0.183 mA? If NEURON gives Vm_onset ≈ −80 mV (slow rise) while our model gives −42 mV (instant jump), the issue is in the capacitive coupling mechanism, not Cm values.

4. **Alternative: solve the full dynamic periaxonal equations**: Instead of quasi-static Vpax, solve the periaxonal cable equation at each timestep. This would exactly replicate NEURON's extracellular mechanism and eliminate all quasi-static approximation errors.

---

## Important Notes for Next Agent

- The conda environment is `jaxley_fibers` (not `nerve_seg`).
- Run from project root: `conda run -n jaxley_fibers python experiments/_threshold_vi_state.py`
- The solver is JAX-based and JIT-compiled via `jax.lax.scan`. Numerical precision is float32.
- `jax.lax.scan` requires all array shapes to be static (no Python control flow inside scan body).
- The tridiagonal solver uses a custom Thomas algorithm scan (not `jnp.linalg.solve`).
- NEURON/PyFibers reference is available in the same environment if needed to extract exact Vm traces.
- `geom.xc_myelin_uF_cm2` stores the myelin sheath capacitance per unit area (µF/cm²).
- `geom.xg_myelin_S_cm2` stores the myelin sheath conductance per unit area (S/cm²).
- `geom.xraxial_Mohm_cm` stores periaxonal axial resistivity (MΩ/cm).
- `geom.diameter` is the OUTER fiber diameter (e.g., 10.0 µm).
- `geom.diam_um[i]` is the INNER axon diameter at compartment i (node_diam for node/MYSA, axon_diam for FLUT/STIN).
