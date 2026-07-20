"""Analyze sparse fiber sampling sweep results.

For each (sample, seed), joins:
  - data_seed_NNNN.json        -> dense optimization SI (SI_dense_opt)
  - sparse_sampling_seed_NNNN.json -> SI per strategy (SI_sparse_reported)

Prints a summary table comparing SI_dense_opt vs SI_sparse_reported for each
strategy: centroid, random-1/fasc, random-3/fasc, random-10/fasc.

The gap (SI_dense_opt - SI_sparse_reported) is the key §3.4 metric.

Run from project root:

    python -m experiments_v2.analyze_sparse_sampling
    python -m experiments_v2.analyze_sparse_sampling --csv sparse_summary.csv
    python -m experiments_v2.analyze_sparse_sampling --per-seed
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT  = Path(__file__).resolve().parent.parent
# Corrected sweep tree; override with DUKE_SWEEP_ROOT (relative to repo root).
SWEEP = ROOT / (os.environ.get("DUKE_SWEEP_ROOT", "").strip() or "outputs/duke_sweeps_fixed")

STRATEGIES = [
    ("centroid",  1, "centroid"),
    ("random",    1, "rand-1/fasc"),
    ("random",    3, "rand-3/fasc"),
    ("random",   10, "rand-10/fasc"),
]


def _species_of(sample: str) -> str:
    return "human" if sample.startswith("human") else "swine"


def _load_rows() -> list[dict]:
    rows = []
    for sample_dir in sorted(SWEEP.iterdir()):
        if not sample_dir.is_dir():
            continue
        sample = sample_dir.name
        # Find all seeds that have both dense and sparse JSONs
        for sparse_path in sorted(sample_dir.glob("sparse_sampling_seed_*.json")):
            seed = int(sparse_path.stem.split("_")[-1])
            dense_path = sample_dir / f"data_seed_{seed:04d}.json"
            if not dense_path.exists():
                continue
            try:
                dense = json.loads(dense_path.read_text())
                sparse = json.loads(sparse_path.read_text())
            except Exception as e:
                print(f"[warn] {sample} seed {seed}: {e}", file=sys.stderr)
                continue

            rect = dense.get("rect") or {}
            dense_si = float(rect.get("achievable_si", abs(rect.get("final_si", float("nan")))))
            n_fascicles = int(sparse.get("dense_n_fibers", 0))  # proxy; actual fasc count not stored

            # Index sparse results by (strategy, n_per_fascicle)
            sparse_by_key: dict[tuple, float] = {}
            for r in sparse.get("results", []):
                if r.get("skipped"):
                    continue
                key = (r["strategy"], int(r["n_per_fascicle"]))
                sparse_by_key[key] = float(r["si"])

            row = dict(
                sample=sample,
                species=_species_of(sample),
                seed=seed,
                dense_si=dense_si,
                dense_n_fibers=int(sparse.get("dense_n_fibers", 0)),
                dense_n_target=int(sparse.get("dense_n_target", 0)),
            )
            for strat, n_per_fasc, label in STRATEGIES:
                si = sparse_by_key.get((strat, n_per_fasc), float("nan"))
                row[label] = si
                row[f"gap_{label}"] = dense_si - si
            rows.append(row)
    return rows


def _strategy_labels() -> list[str]:
    return [label for _, _, label in STRATEGIES]


def _print_summary(rows: list[dict]) -> None:
    labels = _strategy_labels()
    print()
    print(f"{'':>8s}  {'n':>4s}  {'dense':>8s}  " +
          "  ".join(f"{lb:>12s}" for lb in labels))
    print(f"{'':>8s}  {'':>4s}  {'SI_opt':>8s}  " +
          "  ".join(f"{'SI / gap':>12s}" for _ in labels))
    print("-" * (24 + 14 * len(labels)))
    for species in ("swine", "human"):
        sp_rows = [r for r in rows if r["species"] == species]
        if not sp_rows:
            continue
        dense = np.array([r["dense_si"] for r in sp_rows])
        print(f"\n{species:>8s}  {len(sp_rows):>4d}  {dense.mean():>7.3f}   ", end="")
        parts = []
        for label in labels:
            vals = np.array([r[label] for r in sp_rows if np.isfinite(r[label])])
            gaps = np.array([r[f"gap_{label}"] for r in sp_rows if np.isfinite(r[label])])
            if len(vals) == 0:
                parts.append(f"{'n/a':>12s}")
            else:
                parts.append(f"{vals.mean():>5.3f}/{gaps.mean():>+5.3f}")
        print("  ".join(parts))
        # Also print median row
        print(f"{'(median)':>8s}  {'':>4s}  {np.median(dense):>7.3f}   ", end="")
        parts = []
        for label in labels:
            vals = np.array([r[label] for r in sp_rows if np.isfinite(r[label])])
            gaps = np.array([r[f"gap_{label}"] for r in sp_rows if np.isfinite(r[label])])
            if len(vals) == 0:
                parts.append(f"{'n/a':>12s}")
            else:
                parts.append(f"{np.median(vals):>5.3f}/{np.median(gaps):>+5.3f}")
        print("  ".join(parts))
    print()
    print("[format] SI_sparse_reported / gap=(SI_dense_opt - SI_sparse_reported)")
    print("[note]   positive gap = dense optimizer outperforms sparse")
    print("[note]   negative gap = sparse SI_reported > dense (overestimation artifact)")
    print()


def _print_per_seed(rows: list[dict]) -> None:
    labels = _strategy_labels()
    hdr = (f"{'species':>6s}  {'sample':<24s}  {'seed':>4s}  "
           f"{'dense':>7s}  " +
           "  ".join(f"{lb:>12s}" for lb in labels))
    print()
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: (x["species"], x["sample"], x["seed"])):
        line = (f"{r['species']:>6s}  {r['sample']:<24s}  {r['seed']:>4d}  "
                f"{r['dense_si']:>6.3f}   ")
        parts = []
        for label in labels:
            si = r[label]
            gap = r[f"gap_{label}"]
            if np.isfinite(si):
                parts.append(f"{si:>5.3f}/{gap:>+5.3f}")
            else:
                parts.append(f"{'---':>12s}")
        print(line + "  ".join(parts))
    print()


def _write_csv(rows: list[dict], path: Path) -> None:
    import csv
    labels = _strategy_labels()
    fields = (["species", "sample", "seed", "dense_si", "dense_n_fibers",
               "dense_n_target"] +
              labels + [f"gap_{lb}" for lb in labels])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["species"], x["sample"], x["seed"])):
            w.writerow(r)
    print(f"[csv]  -> {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-seed", action="store_true",
                    help="print one line per (sample, seed)")
    ap.add_argument("--csv", type=Path, default=None,
                    help="write per-seed rows to CSV")
    args = ap.parse_args(argv)

    rows = _load_rows()
    if not rows:
        print("[analyze_sparse_sampling] no paired sparse+dense JSONs found", file=sys.stderr)
        return 1

    n_samples = len({r["sample"] for r in rows})
    print(f"[analyze_sparse_sampling] {len(rows)} paired (sample×seed) records "
          f"across {n_samples} samples")

    _print_summary(rows)
    if args.per_seed:
        _print_per_seed(rows)
    if args.csv is not None:
        _write_csv(rows, args.csv.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
