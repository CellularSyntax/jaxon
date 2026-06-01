"""Experiment 3: Gradient-based stimulation-waveform optimization.

NEURON's NMODL backend is not differentiable end-to-end. Jaxley is — every
operation in the channel update + cable solve is `jax.numpy`, so we can take
gradients of any scalar function of the membrane trace w.r.t. the waveform
parameters and run plain Adam.

Task formulation:
  * Parameterize the extracellular stimulation waveform as a 20-bin piecewise-
    constant signal over 0.5 ms (so 25 µs per bin).
  * Hold the spatial extracellular potential profile fixed (point source 1 mm
    above mid node of a 10 µm MRG fiber).
  * Loss = energy_proxy + λ · hinge(V_thresh − max V_m(mid node))²
    where energy_proxy = sum(amp_bin^2) * dt and the hinge enforces a successful AP.
  * Initial waveform: a flat cathodic square pulse at exactly threshold amplitude.
  * Optimizer: Adam (optax) for ~200 steps, lr 5e-3.

Produces:
  outputs/fig3_optimization.png
    (A) initial vs final waveform overlay
    (B) loss curve (and inset: peak V_m per epoch)

Run from project root:
    conda run -n jaxley_fibers python experiments/exp_3_optimization.py
"""

from __future__ import annotations

import sys, pathlib, time
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax
import jaxley as jx
import pandas as pd

from jaxfibers.fibers.mrg import build_mrg, node_indices, section_centers_um
from jaxfibers.stim.extracellular import point_source_potentials_mV, activating_currents_nA

OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------------ config
DIAM = 10.0
N_NODES = 11
TEMPERATURE = 37.0
DT_MS = 0.005
PULSE_MS = 2.0          # longer pulse window so the optimizer can exploit timing,
                        # not just amplitude (energy ∝ ∫I² dt, so spreading reduces energy
                        # ONLY if the AP-threshold constraint binds before fully spread).
N_BINS = 20             # so each bin is 100 µs
TSTOP_MS = 5.0
SRC_HEIGHT_UM = 1000.0
SIGMA_S_M = 0.3
V_THRESH_MV = -20.0     # AP criterion
PENALTY_LAMBDA = 1e4    # weight on AP-fail hinge term (raised so AP constraint binds)
SEED_AMP_MA = -0.40     # near-threshold flat start — encourages the optimizer
                        # to reshape rather than just scale.
LR = 5e-3
N_EPOCHS = 400


def main():
    print(f"JAX devices: {jax.devices()}")
    cell, geom = build_mrg(diameter=DIAM, n_nodes=N_NODES, temperature=TEMPERATURE)
    nodes = node_indices(geom)
    mid_comp = nodes[len(nodes) // 2]

    centers = section_centers_um(geom)
    v_ext_unit_mV = point_source_potentials_mV(
        section_centers_um=centers, src_x_um=0.0, src_y_um=SRC_HEIGHT_UM,
        src_z_um=centers[mid_comp], i0_mA=1.0, sigma_S_m=SIGMA_S_M,
    )
    inj_unit_nA = activating_currents_nA(
        v_ext_mV=v_ext_unit_mV, length_um=geom.length_um,
        diam_um=geom.diam_um, Ra_ohm_cm=geom.Ra_ohm_cm,
    )
    inj_unit_nA_jax = jnp.asarray(inj_unit_nA)
    active_inds = np.where(np.abs(inj_unit_nA) > 1e-12)[0]
    inj_active = inj_unit_nA_jax[active_inds]   # (n_active,)
    print(f"  {len(active_inds)}/{len(inj_unit_nA)} compartments receive nonzero activating current")

    # Time grid: pulse occupies the first PULSE_MS, with PULSE_DELAY_MS lead-in:
    n_steps = int(TSTOP_MS / DT_MS) + 1
    t_ms = np.arange(n_steps) * DT_MS
    bin_dt_ms = PULSE_MS / N_BINS
    bin_starts_ms = 1.0 + np.arange(N_BINS) * bin_dt_ms   # 1 ms lead-in then pulse
    # Build a (T, N_BINS) one-hot map that says which bin (if any) each time step belongs to:
    in_bin = np.zeros((n_steps, N_BINS), dtype=np.float32)
    for k, t0 in enumerate(bin_starts_ms):
        mask = (t_ms >= t0) & (t_ms < t0 + bin_dt_ms)
        in_bin[mask, k] = 1.0
    in_bin_jax = jnp.asarray(in_bin)                       # (T, N_BINS)

    # Record at mid node only:
    cell.branch(0).comp(mid_comp).record("v")

    def forward(theta: jnp.ndarray) -> jnp.ndarray:
        """Return scalar loss for a given waveform `theta` (mA per bin)."""
        # waveform(T,) = in_bin @ theta
        wav = in_bin_jax @ theta                           # (T,)
        # per-compartment per-timestep current = inj_active(:,None) * wav(None,:):
        currents = inj_active[:, None] * wav[None, :]      # (n_active, T)
        # Build data_stimuli via chained data_stimulate calls:
        ds = None
        for k, ci in enumerate(active_inds.tolist()):
            ds = cell.branch(0).comp(int(ci)).data_stimulate(currents[k], ds)
        v = jx.integrate(
            cell, delta_t=DT_MS, t_max=TSTOP_MS,
            data_stimuli=ds, solver="bwd_euler",
        )
        v_mid = v[0]                                       # (T,)
        peak = jnp.max(v_mid)
        energy = jnp.sum(theta ** 2) * bin_dt_ms           # ∫ I^2 dt proxy (mA²·ms)
        # Hinge: penalty only if peak fails to exceed V_thresh
        deficit = jax.nn.relu(V_THRESH_MV - peak)
        loss = energy + PENALTY_LAMBDA * deficit ** 2
        return loss, (peak, energy)

    value_and_grad = jax.jit(jax.value_and_grad(forward, has_aux=True))

    # Initial waveform: flat at SEED_AMP_MA
    theta0 = jnp.full((N_BINS,), SEED_AMP_MA, dtype=jnp.float32)
    theta = theta0

    opt = optax.adam(LR)
    state = opt.init(theta)

    losses, peaks, energies = [], [], []
    print(f"Optimizing {N_BINS}-bin waveform for {N_EPOCHS} epochs, lr={LR} ...")
    t0 = time.time()
    for epoch in range(N_EPOCHS):
        (loss, (peak, energy)), grads = value_and_grad(theta)
        updates, state = opt.update(grads, state, theta)
        theta = optax.apply_updates(theta, updates)
        losses.append(float(loss))
        peaks.append(float(peak))
        energies.append(float(energy))
        if epoch % 20 == 0 or epoch == N_EPOCHS - 1:
            print(f"  epoch {epoch:3d} | loss={float(loss):.4f} | peak Vm = {float(peak):+6.2f} mV "
                  f"| energy = {float(energy):.4f}")
    print(f"Optimization wall-time: {time.time() - t0:.1f} s")

    theta_final = np.asarray(theta)
    theta_init = np.asarray(theta0)

    # --- plot ----------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    ax = axs[0]
    bin_centers = bin_starts_ms + bin_dt_ms / 2.0
    ax.step(np.concatenate([bin_centers, [bin_centers[-1] + bin_dt_ms]]),
            np.concatenate([theta_init, [theta_init[-1]]]),
            where="post", color="C0", lw=2.0, label="initial square (flat)")
    ax.step(np.concatenate([bin_centers, [bin_centers[-1] + bin_dt_ms]]),
            np.concatenate([theta_final, [theta_final[-1]]]),
            where="post", color="C3", lw=2.0, label="optimized")
    ax.axhline(0, color="k", lw=0.5, alpha=0.6)
    ax.set_xlabel("time (ms)"); ax.set_ylabel("waveform amplitude (mA)")
    ax.set_title("(A) Starting vs optimized waveform\n"
                 f"({N_BINS}-bin piecewise constant over {PULSE_MS:.1f} ms; "
                 f"AP retained at peak Vm = {peaks[-1]:.1f} mV)")
    ax.legend(loc="best", fontsize=9); ax.grid(alpha=0.3)

    ax = axs[1]
    ax.plot(losses, color="C3", lw=2.0)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss = energy + λ·hinge²")
    ax.set_title(f"(B) Optimization curve (Adam, lr={LR}, λ={PENALTY_LAMBDA:g})")
    ax.set_yscale("log"); ax.grid(alpha=0.3)

    # Inset: energy and peak Vm
    ins = ax.inset_axes([0.43, 0.40, 0.55, 0.45])
    ins.plot(energies, color="C2", lw=1.5, label="energy proxy")
    ins.set_ylabel("energy (mA²·ms)", color="C2", fontsize=8)
    ins.tick_params(axis="y", labelsize=8, colors="C2")
    ins.tick_params(axis="x", labelsize=8)
    ins2 = ins.twinx()
    ins2.plot(peaks, color="C0", lw=1.5, label="peak Vm")
    ins2.axhline(V_THRESH_MV, color="C0", ls="--", lw=0.8, alpha=0.7)
    ins2.set_ylabel("peak Vm (mV)", color="C0", fontsize=8)
    ins2.tick_params(axis="y", labelsize=8, colors="C0")
    ins.set_xlabel("epoch", fontsize=8)
    ins.set_title("energy vs peak Vm", fontsize=8)

    fig.suptitle(
        "Gradient-based stimulation waveform optimization through a Jaxley MRG fiber\n"
        "(intractable in pure NEURON — backpropagation through the cable solve)",
        fontsize=11,
    )
    fig.savefig(OUT_DIR / "fig3_optimization.png", dpi=130)
    print(f"  -> {OUT_DIR/'fig3_optimization.png'}")


if __name__ == "__main__":
    main()
