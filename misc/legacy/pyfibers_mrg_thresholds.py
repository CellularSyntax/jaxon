"""NEURON/PyFibers reference thresholds for MRG_DISCRETE fiber diameters.

Same electrode / waveform setup as experiments/_threshold_coupled.py:
  - point source 1000 µm above fiber center, sigma = 0.3 S/m
  - monophasic cathodic rectangular pulse, 0.1 ms PW
  - dt = 0.005 ms, TSTOP = 5.0 ms, DELAY = 1.0 ms
  - N = 21 nodes

Saves results to outputs/pyfibers_thresholds.json.
"""
import sys, pathlib, json, time
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from pyfibers import FiberModel, build_fiber, ScaledStim

DIAMETERS = [5.7, 7.3, 8.7, 10.0, 11.5, 12.8, 14.0, 15.0, 16.0]  # skip 1/2 µm: impractical at 1 mm
N_NODES   = 21
DT        = 0.005   # ms
TSTOP     = 5.0     # ms
DELAY     = 1.0     # ms
PW        = 0.1     # ms
SIGMA     = 0.3     # S/m (matches point_source_potentials_mV default)

# Callable waveform (PyFibers >= 0.8 requires callable, not array)
def waveform(t):
    return np.where((t >= DELAY) & (t < DELAY + PW), 1.0, 0.0)

results = {}
print(f"{'D (µm)':>8}  {'threshold (mA)':>16}  {'time (s)':>8}")
print("-" * 38)

for D in DIAMETERS:
    t0 = time.time()
    fiber = build_fiber(
        diameter=D, fiber_model=FiberModel.MRG_DISCRETE,
        temperature=37, n_nodes=N_NODES,
        passive_end_nodes=False,   # match JAX: all nodes active
    )
    fiber.potentials = fiber.point_source_potentials(
        0, 1000, fiber.length / 2, 1, SIGMA
    )
    stim = ScaledStim(waveform=waveform, dt=DT, tstop=TSTOP)
    amp, _ = stim.find_threshold(fiber, stimamp_top=-1.0)
    results[D] = float(amp)
    print(f"{D:>8.1f}  {amp:>16.5f}  {time.time()-t0:>8.1f}")

out = ROOT / "outputs" / "pyfibers_thresholds.json"
out.parent.mkdir(exist_ok=True)
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {out}")
