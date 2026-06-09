"""Audit FEM Ve_unit z-variation across the entire duke_Ves dataset.

Background.  The smart-init tripolar patterns work by subtracting the
cathode contribution at a column's middle-z contact from the anodic
guard contributions at the column's top/bottom-z contacts.  When the
three contacts in a column produce nearly identical Ve fields along z
(insufficient FEM mesh resolution between contact z-levels), that
subtraction collapses to ~0 and tripolar firing rate is zero at every
magnitude tested.  Bipolar patterns are unaffected because they exploit
the angular gradient (cross-cuff dipole), not the axial gradient.

The defining quality metric is therefore the per-channel mean of the
per-fiber std along z:

    z_std_chan[k] = mean over fibers of  std over compartments of Ve_unit[k, f, :]

Working sample (sub-54_sam-2) has mean z_std ~ 248 mV/mA.
Broken sample (sub-53_sam-2) has mean z_std ~  61 mV/mA, and tripolar
fires nothing at +/- 1.5 mA.

We use 150 mV/mA as the threshold (~ 2x sub-54-level safety margin,
~ 2.5x sub-53-level failure threshold).  Samples below this should be
excluded from the cohort or have their FEM re-meshed with finer z
resolution between contact heights.

Usage:
    python -m experiments_v2.duke_fem_quality <duke_ves_root> \
        [--threshold 150.0] [--out-json path]

Prints a per-sample table and writes a JSON pass/fail summary that the
selectivity sweep can consult to auto-skip bad anatomies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Make project importable when invoked from the project root.
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from experiments_v2.duke_loader import load_duke_sample  # noqa: E402


def channel_z_std(Ve_unit: np.ndarray) -> np.ndarray:
    """Per-channel mean of per-fiber std along z.

    Returns shape [K].  A channel with strong z-variation (i.e. the
    FEM resolved the field distinctly at different fiber compartments)
    gets a large value; a channel where Ve is nearly constant along z
    gets a small value.
    """
    return Ve_unit.std(axis=2).mean(axis=1)   # [K]


def audit_sample(sample_dir: Path) -> dict:
    sample = load_duke_sample(sample_dir, fiber_diam_um=10.0,
                                 n_nodes=21, verbose=False)
    Ve_unit = sample["Ve_unit"]
    if Ve_unit.ndim != 3:
        return dict(name=sample_dir.name, ok=False,
                      error=f"Ve_unit ndim={Ve_unit.ndim}")
    K, F, N = Ve_unit.shape
    z_std = channel_z_std(Ve_unit)
    return dict(
        name=sample_dir.name,
        K=int(K), n_fibers=int(F), n_comp=int(N),
        z_std_per_chan=z_std.tolist(),
        z_std_mean=float(z_std.mean()),
        z_std_min=float(z_std.min()),
        z_std_max=float(z_std.max()),
        Ve_global_max=float(Ve_unit.max()),
        Ve_frac_zero=float((Ve_unit == 0.0).mean()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("duke_ves_root", type=str,
                          help="Directory containing per-sample subdirectories")
    parser.add_argument("--threshold", type=float, default=150.0,
                          help="Minimum acceptable per-channel mean z-std "
                                "(mV/mA).  Samples below this fail.")
    parser.add_argument("--out-json", type=str, default=None,
                          help="Write pass/fail summary to this JSON path "
                                "(consumed by selectivity_sweep_duke).")
    args = parser.parse_args()

    root = Path(args.duke_ves_root).resolve()
    if not root.is_dir():
        print(f"[error] {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    candidates = sorted(p for p in root.iterdir()
                         if p.is_dir() and (p / "paths_Ve.npz").exists())
    print(f"[scan] {root}: {len(candidates)} candidate samples")
    print()

    results: list[dict] = []
    for sd in candidates:
        try:
            r = audit_sample(sd)
        except Exception as e:
            r = dict(name=sd.name, ok=False, error=str(e))
        results.append(r)

    # Flag pass/fail.
    threshold = float(args.threshold)
    for r in results:
        if "z_std_mean" in r:
            r["passes"] = bool(r["z_std_mean"] >= threshold)
        else:
            r["passes"] = False

    # Print table.
    print(f"{'sample':>32}  {'K':>3}  {'F':>5}  {'z_std_mean':>12}  "
          f"{'z_std_min':>11}  {'z_std_max':>11}  {'Ve_max':>10}  pass?")
    print("-" * 110)
    n_pass = 0
    for r in results:
        if "z_std_mean" in r:
            flag = "PASS" if r["passes"] else "FAIL"
            print(f"{r['name']:>32}  {r['K']:>3}  {r['n_fibers']:>5}  "
                  f"{r['z_std_mean']:>12.1f}  "
                  f"{r['z_std_min']:>11.1f}  {r['z_std_max']:>11.1f}  "
                  f"{r['Ve_global_max']:>10.1f}  {flag}")
            if r["passes"]:
                n_pass += 1
        else:
            print(f"{r['name']:>32}  ERROR  {r.get('error', '')}")
    print()
    print(f"[summary] {n_pass}/{len(results)} samples pass "
          f"(threshold = {threshold} mV/mA per-channel z-std mean)")

    fails = [r["name"] for r in results if not r["passes"]]
    if fails:
        print(f"[excluded] {len(fails)} samples: {fails}")

    out_json = (Path(args.out_json) if args.out_json is not None
                else root / "duke_fem_quality.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(dict(
            threshold=threshold,
            duke_ves_root=str(root),
            n_total=len(results),
            n_pass=n_pass,
            results=results,
        ), f, indent=2)
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()
