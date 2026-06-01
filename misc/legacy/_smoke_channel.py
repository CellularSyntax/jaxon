"""Smoke test: cross-check Jaxley AxnodeMyel rates against NEURON's axnode_myel mechanism.

We build a single NEURON section, insert axnode_myel, set v to each of a sweep of voltages,
let NEURON evaluate the rate functions (via h.finitialize), and compare *_inf, tau_* against
our pure-JAX implementation.

Run from project root:
    conda run -n jaxley_fibers python experiments/_smoke_channel.py
"""
from __future__ import annotations

import sys
import pathlib

# Make the project package importable when run as a script.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from neuron import h

# Load pyfibers, which compiles & loads the axnode_myel mechanism.
import pyfibers  # noqa: F401  (side-effect: loads compiled MOD)

from jaxfibers.channels.mrg_axnode import AxnodeMyel

CELSIUS = 37.0


def neuron_rates(v_sweep_mv):
    """Build one section with axnode_myel, sweep voltage, read *_inf and tau_* back."""
    h.celsius = CELSIUS
    sec = h.Section(name="probe")
    sec.L = sec.diam = 10.0
    sec.insert("axnode_myel")
    seg = sec(0.5)

    out = {k: [] for k in ["m_inf", "h_inf", "mp_inf", "s_inf",
                            "tau_m", "tau_h", "tau_mp", "tau_s"]}
    for v in v_sweep_mv:
        # finitialize at v forces NMODL INITIAL to run, which calls evaluate_fct(v)
        # and populates *_inf and tau_* on the segment.
        h.finitialize(v)
        out["m_inf"].append(seg.m_inf_axnode_myel)
        out["h_inf"].append(seg.h_inf_axnode_myel)
        out["mp_inf"].append(seg.mp_inf_axnode_myel)
        out["s_inf"].append(seg.s_inf_axnode_myel)
        out["tau_m"].append(seg.tau_m_axnode_myel)
        out["tau_h"].append(seg.tau_h_axnode_myel)
        out["tau_mp"].append(seg.tau_mp_axnode_myel)
        out["tau_s"].append(seg.tau_s_axnode_myel)
    return {k: np.array(v) for k, v in out.items()}


def jaxley_rates(v_sweep_mv):
    import jax.numpy as jnp
    v = jnp.array(v_sweep_mv)
    (a_m, b_m), (a_h, b_h), (a_mp, b_mp), (a_s, b_s) = AxnodeMyel._alpha_beta(v, CELSIUS)

    def inf_tau(a, b):
        return np.asarray(a / (a + b)), np.asarray(1.0 / (a + b))

    m_inf,  tau_m  = inf_tau(a_m,  b_m)
    h_inf,  tau_h  = inf_tau(a_h,  b_h)
    mp_inf, tau_mp = inf_tau(a_mp, b_mp)
    s_inf,  tau_s  = inf_tau(a_s,  b_s)
    return dict(m_inf=m_inf, h_inf=h_inf, mp_inf=mp_inf, s_inf=s_inf,
                tau_m=tau_m, tau_h=tau_h, tau_mp=tau_mp, tau_s=tau_s)


def main():
    v_sweep = np.linspace(-120.0, 60.0, 19)  # mV
    n = neuron_rates(v_sweep)
    j = jaxley_rates(v_sweep)

    print(f"{'v (mV)':>8} | {'m_inf N':>10} {'m_inf J':>10} | {'h_inf N':>10} {'h_inf J':>10} | "
          f"{'tau_m N':>10} {'tau_m J':>10}")
    for i, vv in enumerate(v_sweep):
        print(f"{vv:8.1f} | {n['m_inf'][i]:10.6f} {j['m_inf'][i]:10.6f} | "
              f"{n['h_inf'][i]:10.6f} {j['h_inf'][i]:10.6f} | "
              f"{n['tau_m'][i]:10.6f} {j['tau_m'][i]:10.6f}")

    print()
    max_err = {}
    for key in n.keys():
        rel = np.abs(n[key] - j[key]) / (np.abs(n[key]) + 1e-12)
        max_err[key] = (rel.max(), v_sweep[rel.argmax()])
    print("Max relative error per quantity (across sweep):")
    for key, (err, vv) in max_err.items():
        flag = "OK " if err < 1e-5 else "** "
        print(f"  {flag}{key:8s}  rel_err={err: .3e}  (at v={vv:.1f} mV)")


if __name__ == "__main__":
    main()
