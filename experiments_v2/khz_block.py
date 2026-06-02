"""Kilohertz frequency block — verbatim PyFibers tutorial reproduction.

Mirrors `pyfibers/tutorials/5_block_threshold` exactly:
  * MRG_INTERPOLATION, D=10 µm, N=25 nodes
  * 20 kHz square wave, on from t=50 to t=100 ms
  * Intrinsic activity via add_intrinsic_activity (loc=0.1, every 10 ms, 14
    stims, starting at t=15 ms)
  * ScaledStim.run_sim returns (n_aps, ap_time) for several amplitudes
  * AP detection at loc=0.9

This is a clean PyFibers-only reproduction.  Once it works, we layer a
jaxfibers comparison on top.

Outputs (outputs/khz_block/)
----------------------------
  data_khz_block.json       — per-amplitude n_aps + traces
  fig_khz_block_traces.png  — Vm(t) at loc=0.9 for each amplitude
  fig_khz_block.png         — kept for fig3_combined compatibility (TBD)

Run from project root:
    python experiments_v2/khz_block.py
"""

from __future__ import annotations

import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.signal as sg

# Trigger our nrn_baseline path setup (prepends c:/nrn826/lib/python on Windows)
from jaxfibers.nrn_baseline import build_mrg_pyfibers   # noqa: F401  (side-effect import)
from pyfibers import build_fiber, FiberModel, ScaledStim

from experiments_v2.utils import ensure_dir, save_json

OUT = ensure_dir(ROOT / "outputs" / "khz_block")

# ── tutorial parameters (verbatim) ────────────────────────────────────────────
N_NODES   = 25
DIAMETER  = 10.0
DT        = 0.001       # ms
TSTOP     = 150.0       # ms
KHZ_FREQ  = 20.0        # kHz
KHZ_ON    = 50.0        # ms
KHZ_OFF   = 100.0       # ms
# Point source coordinates from the tutorial: (0, 250, fiber.length/2, i0=1, sigma=10)
SRC_X     = 0.0
SRC_Y     = 250.0
SRC_I0    = 1.0
SIGMA     = 10.0
# Intrinsic activity
PACE_LOC      = 0.1
PACE_START    = 15.0
PACE_INTERVAL = 10.0
PACE_N        = 14
AP_DETECT_LOC = 0.9
# Amplitudes shown in the tutorial
AMPLITUDES = [-0.5, -1.5, -2.5, -3.0]


def waveform(t: float) -> float:
    """20 kHz square wave, on for t in [KHZ_ON, KHZ_OFF], zero elsewhere."""
    if KHZ_ON < t < KHZ_OFF:
        return float(sg.square(2 * np.pi * KHZ_FREQ * t))
    return 0.0


def main():
    print("=== kHz block (verbatim PyFibers tutorial reproduction) ===")
    print(f"Fiber: MRG_INTERPOLATION  D={DIAMETER} µm  N={N_NODES} nodes")
    print(f"Sim: dt={DT} ms  tstop={TSTOP} ms")
    print(f"kHz: {KHZ_FREQ} kHz square wave, on {KHZ_ON}-{KHZ_OFF} ms")
    print(f"Pacing: loc={PACE_LOC}  start={PACE_START} ms  interval={PACE_INTERVAL} ms  "
          f"n={PACE_N}\n")

    results = []
    for amp in AMPLITUDES:
        print(f"--- amp = {amp} mA ---", flush=True)
        # Fresh fiber per amplitude (cleanest — avoids accumulated state).
        fiber = build_fiber(FiberModel.MRG_INTERPOLATION,
                             diameter=DIAMETER, n_nodes=N_NODES)
        fiber.potentials = fiber.point_source_potentials(
            SRC_X, SRC_Y, fiber.length / 2.0, SRC_I0, SIGMA,
        )
        fiber.record_vm()
        fiber.add_intrinsic_activity(
            loc=PACE_LOC,
            start_time=PACE_START,
            avg_interval=PACE_INTERVAL,
            num_stims=PACE_N,
        )
        blockstim = ScaledStim(waveform=waveform, dt=DT, tstop=TSTOP)
        n_aps, ap_time = blockstim.run_sim(amp, fiber)
        print(f"  -> {n_aps} APs, last AP time = {ap_time}", flush=True)

        # Save Vm at detection node + a few representative nodes
        vm_idx = fiber.loc_index(AP_DETECT_LOC)
        results.append({
            "amp_mA":     amp,
            "n_aps":      int(n_aps),
            "ap_time_ms": float(ap_time) if ap_time is not None else None,
            "t_ms":       np.array(blockstim.time).tolist(),
            "vm_90_mV":   np.array(fiber.vm[vm_idx]).tolist(),
        })

    # ── plot Vm at loc=0.9 for every amplitude ───────────────────────────────
    fig, axes = plt.subplots(len(AMPLITUDES), 1,
                              figsize=(10, 2.5 * len(AMPLITUDES)),
                              sharex=True, constrained_layout=True)
    for ax, res in zip(axes, results):
        t = np.array(res["t_ms"])
        vm = np.array(res["vm_90_mV"])
        ax.plot(t, vm, color="C0", lw=1.0, label=r"$V_m(t)$ at 90% length")
        ax.axvspan(KHZ_ON, KHZ_OFF, alpha=0.25, color="red", label="kHz on")
        for s in PACE_START + np.arange(PACE_N) * PACE_INTERVAL:
            ax.axvline(s, color="k", ls="--", lw=0.5, alpha=0.5)
        ax.set_ylabel(f"{res['amp_mA']} mA\nVm (mV)")
        ax.set_ylim(-90, 30)
        ax.set_title(f"amp = {res['amp_mA']} mA, {res['n_aps']} APs",
                     fontsize=10, loc="left")
        ax.grid(alpha=0.3, lw=0.4)
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle(f"PyFibers kHz block tutorial reproduction — "
                  f"MRG_INTERPOLATION D={DIAMETER} µm, {int(KHZ_FREQ)} kHz",
                  fontsize=11)
    path = OUT / "fig_khz_block_traces.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  -> {path}")

    save_json({
        "fiber_model":   "MRG_INTERPOLATION",
        "diameter_um":   DIAMETER,
        "n_nodes":       N_NODES,
        "dt_ms":         DT,
        "tstop_ms":      TSTOP,
        "khz_freq":      KHZ_FREQ,
        "khz_on_ms":     KHZ_ON,
        "khz_off_ms":    KHZ_OFF,
        "src_y_um":      SRC_Y,
        "sigma_S_m":     SIGMA,
        "pace":          {"loc": PACE_LOC, "start_ms": PACE_START,
                            "interval_ms": PACE_INTERVAL, "n_pulses": PACE_N},
        "ap_detect_loc": AP_DETECT_LOC,
        "results":       results,
    }, OUT / "data_khz_block.json")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
