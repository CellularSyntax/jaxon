"""Quick standalone timing test of the refactored Jaxley bisection."""
import sys, pathlib, time
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Use the refactored function:
from experiments.exp_1_validation import jaxley_find_threshold_extra
from jaxfibers.nrn_baseline import find_threshold_extracellular

for d in [10.0]:
    print(f"\n=== d = {d} µm ===", flush=True)
    t0 = time.time()
    th = jaxley_find_threshold_extra(diameter=d, n_nodes=21,
                                     src_height_um=1000.0, pw_ms=0.1,
                                     delay_ms=1.0, dt_ms=0.005,
                                     tstop_ms=5.0, rel_tol=5e-3)
    print(f"Jaxley threshold = {th:.4f} mA in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    th_n = find_threshold_extracellular(diameter=d, n_nodes=21,
                                        src_height_um=1000.0, pw_ms=0.1,
                                        delay_ms=1.0, dt_ms=0.005,
                                        tstop_ms=5.0, rel_tol=5e-3)
    print(f"NEURON threshold = {th_n:.4f} mA in {time.time()-t0:.1f}s", flush=True)
    print(f"|err| = {abs(th-th_n)/abs(th_n)*100:.2f}%", flush=True)
