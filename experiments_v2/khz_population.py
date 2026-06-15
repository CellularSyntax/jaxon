"""Kilohertz frequency population response (Hussain 2024 Fig 3b equivalent).

For each combination of:
  - fiber diameter D in {5.7, 8.7, 14.0} µm           (columns in figure)
  - kHz frequency  f in {1, 2, 5, 10} kHz              (rows in figure)
  - kHz amplitude swept over N_AMPS values

Runs both the JAX coupled solver and PyFibers, counts APs at a distal
detection node, and averages across N_FIBERS radial positions (src_y_um)
to produce population mean ± 95 % CI bands.

Setup matches khz_block.py:
  - MRG_INTERPOLATION, N_NODES = 25
  - Point source at varying height above fiber midpoint, sigma = 10 S/m
  - Intrinsic pacing at loc = 0.1 (100 Hz, 14 pulses from t = 15 ms)
  - kHz signal on during t = 50–100 ms
  - AP detection at loc = 0.9

Outputs (outputs/khz_population/)
-----------------------------------
  data_khz_population.json   — per-(D, freq) AP count mean / CI arrays
  fig_khz_population.png     — standalone 4-row × 3-col grid

Run from project root:
    python experiments_v2/khz_population.py
"""
from __future__ import annotations

import sys
import pathlib
import dataclasses
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jaxley.solver_gate import solve_gate_exponential

jax.config.update("jax_enable_x64", True)

from jaxfibers.fibers.mrg import (
    build_mrg_interp, node_indices, section_centers_um,
    V_REST, CM_AXON, G_PAS_MYSA, G_PAS_FLUT, G_PAS_STIN,
)
from jaxfibers.channels.mrg_axnode import AxnodeMyel
from jaxfibers.stim.extracellular import point_source_potentials_mV
from jaxfibers.stim.extracellular_coupled import arrays_from_geometry, integrate
from pyfibers import build_fiber, FiberModel, ScaledStim

from experiments_v2.utils import ensure_dir, save_json

OUT = ensure_dir(ROOT / "outputs" / "khz_population")

# ── Experiment parameters ──────────────────────────────────────────────────
DIAMETERS        = [5.7, 8.7, 14.0]         # µm
KHZ_FREQS        = [1.0, 2.0, 5.0, 10.0]   # kHz
# Population: fibers at different radial offsets from the electrode.
SRC_H_FIBERS_UM  = [250, 500, 1000]         # µm
N_AMPS           = 8
# Amplitude range per diameter (positive; negated internally for cathodic convention).
AMP_MAX_MA       = {5.7: 10.0, 8.7: 5.0, 14.0: 2.0}

# ── Simulation constants ───────────────────────────────────────────────────
N_NODES       = 25
CELSIUS       = 37.0
DT            = 0.005    # ms  — 20 samples/period at 10 kHz; accurate for sinusoid
TSTOP         = 100.0    # ms  — Hussain Fig 3b: 100 ms total simulation
N_STEPS       = int(TSTOP / DT)

KHZ_START     = 0.5      # ms  — Hussain: kHz stimulation starts at t = 0.5 ms
SRC_X         = 0.0      # µm  (electrode at fiber midpoint longitudinally)
SIGMA         = 10.0     # S/m
SRC_I0        = 1.0      # mA  (reference current for Ve_unit)

PACE_LOC          = 0.1
PACE_START        = 50.0    # ms  — Hussain: pacing starts at t = 50 ms
PACE_INTERVAL     = 10.0    # ms  — 100 Hz intrinsic rate
PACE_N            = 5       # pulses at t = 50, 60, 70, 80, 90 ms
PACE_PW           = 0.1     # ms
PACE_AMP_NA       = 2.0     # nA  — Hussain: 2 nA intracellular anodic pulse
AP_DETECT_LOC     = 0.9
AP_THRESH_MV      = -20.0
AP_COUNT_AFTER_MS = 50.0    # ms  — count only APs arriving after t = 50 ms

# ── MRG channel constants ──────────────────────────────────────────────────
GNABAR = 3.0;  GNAPBAR = 0.01; GKBAR = 0.08; GL = 0.007
ENA    = 50.0; EK      = -90.0; EL    = -90.0


def _build_membrane_fn(static, geom):
    is_node   = static["is_node"]
    A_in      = static["A_in_cm2"]
    stype_gp  = {"node": 0.0, "mysa": G_PAS_MYSA, "flut": G_PAS_FLUT, "stin": G_PAS_STIN}
    g_pas_arr = jnp.asarray([stype_gp[s] for s in geom.section_type])

    v0 = jnp.float64(V_REST)
    (a_m0, b_m0), (a_h0, b_h0), (a_mp0, b_mp0), (a_s0, b_s0) = \
        AxnodeMyel._alpha_beta(v0, CELSIUS)
    state0 = (
        jnp.where(is_node, float(a_m0  / (a_m0  + b_m0)),  0.0),
        jnp.where(is_node, float(a_h0  / (a_h0  + b_h0)),  0.0),
        jnp.where(is_node, float(a_mp0 / (a_mp0 + b_mp0)), 0.0),
        jnp.where(is_node, float(a_s0  / (a_s0  + b_s0)),  0.0),
    )

    def membrane_fn(Vm, state, dt):
        M, H, MP, S = state
        (a_m, b_m), (a_h, b_h), (a_mp, b_mp), (a_s, b_s) = \
            AxnodeMyel._alpha_beta(Vm, CELSIUS)
        M2  = solve_gate_exponential(M,  dt, a_m,  b_m)
        H2  = solve_gate_exponential(H,  dt, a_h,  b_h)
        MP2 = solve_gate_exponential(MP, dt, a_mp, b_mp)
        S2  = solve_gate_exponential(S,  dt, a_s,  b_s)
        g_na   = GNABAR  * M2**3 * H2
        g_nap  = GNAPBAR * MP2**3
        g_k    = GKBAR   * S2
        g_node = (g_na + g_nap + g_k + GL) * A_in * 1e6
        i_node = ((g_na + g_nap) * (Vm - ENA)
                    + g_k * (Vm - EK)
                    + GL  * (Vm - EL)) * A_in * 1e6
        g_pas_us = g_pas_arr * A_in * 1e6
        i_pas    = g_pas_us  * (Vm - V_REST)
        g_eff = jnp.where(is_node, g_node,  g_pas_us)
        i_ion = jnp.where(is_node, i_node,  i_pas)
        return g_eff, i_ion, (M2, H2, MP2, S2)

    return membrane_fn, state0


def _count_aps(vm: np.ndarray) -> int:
    """Count threshold crossings at detection node only after t = 50 ms (Hussain protocol)."""
    t = (np.arange(len(vm)) + 1) * DT
    vm_after = vm[t >= AP_COUNT_AFTER_MS]
    if len(vm_after) < 2:
        return 0
    rises = np.where((vm_after[:-1] <= AP_THRESH_MV) & (vm_after[1:] > AP_THRESH_MV))[0]
    return int(len(rises))


def _make_pulse_mask(freq_khz: float) -> np.ndarray:
    """Sinusoidal kHz extracellular signal starting at t = 0.5 ms (Hussain Fig 3b)."""
    t = (np.arange(N_STEPS) + 1) * DT
    mask = np.zeros(N_STEPS, dtype=np.float64)
    on = t >= KHZ_START
    # angle = 2π · f_kHz [kHz] · t [ms]  →  correct units since kHz·ms = 1
    mask[on] = np.sin(2.0 * np.pi * freq_khz * t[on])
    return mask


def _make_i_intra(n_comp: int, pace_node: int) -> np.ndarray:
    t = (np.arange(N_STEPS) + 1) * DT
    ii = np.zeros((N_STEPS, n_comp), dtype=np.float64)
    for ps in PACE_START + np.arange(PACE_N) * PACE_INTERVAL:
        on = (t >= ps) & (t < ps + PACE_PW)
        ii[on, pace_node] = PACE_AMP_NA
    return ii


def _build_jax_runner(diameter: float, src_y_um: float, freq_khz: float):
    """Return a jit'd callable ``run(amp_mA_positive) -> vm_far [N_STEPS]``.

    Amplitude is taken as a positive magnitude and negated internally
    (cathodic convention, matching khz_block.py)."""
    _, geom   = build_mrg_interp(diameter=diameter, n_nodes=N_NODES)
    nodes     = node_indices(geom)
    centers   = np.array(section_centers_um(geom))
    n_comp    = geom.n_comp
    z_mid     = float(centers[len(centers) // 2])

    Ve_unit = np.asarray(point_source_potentials_mV(
        list(centers),
        src_x_um=SRC_X, src_y_um=src_y_um, src_z_um=z_mid,
        i0_mA=SRC_I0, sigma_S_m=SIGMA,
    ))
    geom_c = dataclasses.replace(geom, cm_uF_cm2=[CM_AXON] * n_comp)
    static = arrays_from_geometry(geom_c, DT)
    membrane_fn, state0 = _build_membrane_fn(static, geom)

    pace_node = nodes[int(round(PACE_LOC      * (len(nodes) - 1)))]
    far_node  = nodes[int(round(AP_DETECT_LOC * (len(nodes) - 1)))]
    i_intra   = _make_i_intra(n_comp, int(pace_node))
    pulse_mask = _make_pulse_mask(freq_khz)

    Ve_j  = jnp.asarray(Ve_unit,   dtype=jnp.float64)
    pm_j  = jnp.asarray(pulse_mask)
    ii_j  = jnp.asarray(i_intra)
    fn    = int(far_node)

    @jax.jit
    def run(amp_positive):
        Ve = Ve_j * (-amp_positive)   # cathodic
        vm_far, _ = integrate(
            static, membrane_fn, state0, Ve, pm_j, DT,
            v_rest=V_REST, record="center", center_comp=fn,
            i_intra=ii_j,
        )
        return vm_far

    return run


def _pf_run_one(diameter: float, src_y_um: float, freq_khz: float,
                 amp_positive: float) -> int:
    fiber = build_fiber(FiberModel.MRG_INTERPOLATION, diameter=diameter,
                         n_nodes=N_NODES)
    fiber.potentials = fiber.point_source_potentials(
        SRC_X, src_y_um, fiber.length / 2.0, SRC_I0, SIGMA,
    )
    fiber.add_intrinsic_activity(
        loc=PACE_LOC, start_time=PACE_START,
        avg_interval=PACE_INTERVAL, num_stims=PACE_N,
    )
    fiber.record_vm()  # record all nodes for manual AP counting

    def wav(t):
        if t >= KHZ_START:
            return float(np.sin(2.0 * np.pi * freq_khz * t))
        return 0.0

    stim = ScaledStim(waveform=wav, dt=DT, tstop=TSTOP)
    stim.run_sim(-amp_positive, fiber,
                  ap_detect_location=AP_DETECT_LOC,
                  fail_on_end_excitation=False)

    # Count APs at detection node after t = 50 ms (Hussain protocol)
    far_idx = int(round(AP_DETECT_LOC * (len(fiber.vm) - 1)))
    vm_far = np.array(fiber.vm[far_idx])
    t_pf = np.array(fiber.time)
    vm_after = vm_far[t_pf >= AP_COUNT_AFTER_MS]
    if len(vm_after) < 2:
        return 0
    rises = np.where((vm_after[:-1] <= AP_THRESH_MV) & (vm_after[1:] > AP_THRESH_MV))[0]
    return int(len(rises))


def _ci95(arr: np.ndarray):
    """Mean and 95 % CI half-width over axis 0 (population axis)."""
    mean = arr.mean(axis=0)
    n    = arr.shape[0]
    ci   = 1.96 * arr.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 \
           else np.zeros_like(mean)
    return mean, ci


def main():
    print("=== kHz population response (Hussain 2024 Fig 3b) ===")
    print(f"Diameters: {DIAMETERS} µm   freqs: {KHZ_FREQS} kHz")
    print(f"Population heights: {SRC_H_FIBERS_UM} µm   N_AMPS={N_AMPS}")
    n_cells = len(DIAMETERS) * len(KHZ_FREQS) * len(SRC_H_FIBERS_UM)
    print(f"Total (D, freq, height) cells: {n_cells} "
          f"× {N_AMPS} amps × 2 solvers = {n_cells * N_AMPS * 2} runs\n")

    results      = {}
    amp_range_out = {}

    for D in DIAMETERS:
        amps = np.linspace(0.0, AMP_MAX_MA[D], N_AMPS)
        amp_range_out[str(D)] = amps.tolist()

        for freq_khz in KHZ_FREQS:
            key = f"D{D}_f{freq_khz}kHz"
            print(f"  {key} ...", flush=True)
            t0 = time.time()

            n_pop  = len(SRC_H_FIBERS_UM)
            jax_ap = np.zeros((n_pop, N_AMPS), dtype=float)
            pf_ap  = np.zeros((n_pop, N_AMPS), dtype=float)

            for fi, src_y in enumerate(SRC_H_FIBERS_UM):
                print(f"    src_y={src_y} µm", flush=True)

                # ── JAX ──────────────────────────────────────────────────────
                runner = _build_jax_runner(D, float(src_y), freq_khz)
                _ = np.asarray(runner(jnp.float64(amps[0])))   # warm-up / JIT
                for ai, amp in enumerate(amps):
                    vm = np.asarray(runner(jnp.float64(float(amp))))
                    jax_ap[fi, ai] = _count_aps(vm)

                # ── PyFibers ─────────────────────────────────────────────────
                for ai, amp in enumerate(amps):
                    pf_ap[fi, ai] = _pf_run_one(D, float(src_y), freq_khz,
                                                  float(amp))

            mean_jax, ci_jax = _ci95(jax_ap)
            mean_pf,  ci_pf  = _ci95(pf_ap)
            results[key] = dict(
                amps_mA  = amps.tolist(),
                mean_jax = mean_jax.tolist(),
                ci95_jax = ci_jax.tolist(),
                mean_pf  = mean_pf.tolist(),
                ci95_pf  = ci_pf.tolist(),
            )
            print(f"    -> {time.time() - t0:.0f} s", flush=True)

    save_json({
        "diameters_um":    DIAMETERS,
        "freqs_khz":       KHZ_FREQS,
        "src_h_fibers_um": SRC_H_FIBERS_UM,
        "n_amps":          N_AMPS,
        "amp_range_mA":    amp_range_out,
        "results":         results,
    }, OUT / "data_khz_population.json")

    _make_figure(results, DIAMETERS, KHZ_FREQS, OUT / "fig_khz_population.png")
    print(f"\n  -> {OUT / 'data_khz_population.json'}")
    print(f"  -> {OUT / 'fig_khz_population.png'}")
    print("\n=== Done ===")


def _make_figure(results, diameters, freqs_khz, out_path):
    n_rows = len(freqs_khz)
    n_cols = len(diameters)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.8 * n_cols, 3.0 * n_rows),
                              constrained_layout=True)
    axes = np.atleast_2d(axes)

    for r, f in enumerate(freqs_khz):
        for c, D in enumerate(diameters):
            ax  = axes[r, c]
            key = f"D{D}_f{f}kHz"
            if key not in results:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                         transform=ax.transAxes, fontsize=9)
                continue
            cell = results[key]
            amps = np.array(cell["amps_mA"])
            mj, cj = np.array(cell["mean_jax"]), np.array(cell["ci95_jax"])
            mp, cp = np.array(cell["mean_pf"]),  np.array(cell["ci95_pf"])

            ax.plot(amps, mp, color="C0", lw=1.6,
                     label="NEURON" if (r == 0 and c == 0) else None)
            ax.fill_between(amps, mp - cp, mp + cp, color="C0", alpha=0.18)
            ax.plot(amps, mj, color="C1", lw=1.4,
                     label="S-MF" if (r == 0 and c == 0) else None)
            ax.fill_between(amps, mj - cj, mj + cj, color="C1", alpha=0.18)
            ax.set_ylim(bottom=0)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.3, lw=0.4)
            if r == 0:
                ax.set_title(f"{D} µm", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{f} kHz\n# APs", fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel("stimulus amplitude (mA)", fontsize=9)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Kilohertz frequency population response", fontsize=12,
                  fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
