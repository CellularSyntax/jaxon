"""Aggregate the Duke FEM-nerve sweep into a per-species summary.

Reads every ``outputs/duke_sweeps/<sample>/data_seed_*.json`` (plus the
legacy ``outputs/selectivity_sweep_duke_*/`` layout) and prints SI
statistics broken out by species (swine vs human).  Convention: a
sample whose directory name starts with ``human-`` is human; everything
else is swine.

Optionally also re-builds the four manuscript figures (cross-section
gallery per species, summary violin, convergence) by invoking
``make_figures.py`` with the Duke targets.

Run from the project root:

    python -m experiments_v2.analyze_duke_results
    python -m experiments_v2.analyze_duke_results --rebuild-figures
    python -m experiments_v2.analyze_duke_results --csv duke_si.csv
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTROOT_LOCAL = ROOT / "manuscript" / "outputs"
OUTROOT_REPO  = ROOT / "outputs"


def _species_of(sample: str) -> str:
    return "human" if sample.startswith("human") else "swine"


def _discover_duke_jsons() -> list[tuple[str, Path]]:
    """Return list of (sample_name, json_path) covering both output layouts."""
    found: list[tuple[str, Path]] = []
    seen_pairs: set[tuple[str, int]] = set()
    for root in (OUTROOT_LOCAL, OUTROOT_REPO):
        if not root.exists():
            continue
        # New layout
        duke_root = root / "duke_sweeps"
        if duke_root.exists():
            for d in sorted(duke_root.iterdir()):
                if not d.is_dir():
                    continue
                for j in sorted(d.glob("data_seed_*.json")):
                    seed = int(j.stem.split("_")[-1])
                    key = (d.name, seed)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    found.append((d.name, j))
        # Legacy layout
        for d in sorted(root.glob("selectivity_sweep_duke_*")):
            sample = d.name[len("selectivity_sweep_duke_"):]
            for j in sorted(d.glob("data_seed_*.json")):
                seed = int(j.stem.split("_")[-1])
                key = (sample, seed)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                found.append((sample, j))
    return found


def _summarise(jsons: list[tuple[str, Path]]) -> dict:
    """Bucket records by species and return per-species statistics."""
    by_species: dict[str, list[dict]] = {"swine": [], "human": []}
    for sample, j in jsons:
        try:
            d = json.loads(j.read_text())
        except Exception as e:
            print(f"[warn] cannot read {j}: {e}", file=sys.stderr)
            continue
        rect_signed = float(d["rect"]["final_si"])
        wave_signed = float(d.get("waveform", {}).get("final_si", rect_signed))
        # achievable_si = |signed_si| — backward-compatible with old JSONs
        # that pre-date the autoflip bookkeeping.
        rect_ach = float(d["rect"].get("achievable_si", abs(rect_signed)))
        wave_ach = float(d.get("waveform", {}).get(
            "achievable_si", abs(wave_signed)))
        flipped_rect = bool(d["rect"].get(
            "target_flipped", rect_signed < 0))
        flipped_wave = bool(d.get("waveform", {}).get(
            "target_flipped", wave_signed < 0))
        rect_loss = float(d["rect"]["final_loss"])
        divider  = float(d.get("divider_deg", 0.0))
        seed     = int(d.get("seed", 0))
        # Multi-start Adam-FD bookkeeping (absent on pre-multistart JSONs).
        restart_mags = d["rect"].get("restart_mags", [])
        best_restart = int(d["rect"].get("best_restart", 0))
        n_restarts   = int(d["rect"].get("n_restarts", 1))
        winning_mag  = (float(restart_mags[best_restart])
                        if restart_mags and 0 <= best_restart < len(restart_mags)
                        else float("nan"))
        by_species[_species_of(sample)].append({
            "sample":           sample,
            "seed":             seed,
            "divider":          divider,
            "rect_si":          rect_signed,
            "wave_si":          wave_signed,
            "rect_achievable":  rect_ach,
            "wave_achievable":  wave_ach,
            "rect_flipped":     flipped_rect,
            "wave_flipped":     flipped_wave,
            "rect_loss":        rect_loss,
            "best_si":          max(rect_signed, wave_signed),
            "best_achievable":  max(rect_ach, wave_ach),
            "n_restarts":       n_restarts,
            "best_restart":     best_restart,
            "winning_mag":      winning_mag,
        })
    return by_species


def _print_table(by_species: dict) -> None:
    print()
    print(f"{'species':>8s}  {'n':>4s}  "
          f"{'rect mean':>10s}  {'rect med':>9s}  "
          f"{'%>=0.95':>8s}  {'%=1.0':>7s}  "
          f"{'wave mean':>10s}  {'wave med':>9s}  "
          f"{'best mean':>10s}  {'#flipped':>9s}")
    print("-" * 100)
    for species in ("swine", "human"):
        rows = by_species.get(species, [])
        if not rows:
            continue
        rect = np.array([r["rect_achievable"] for r in rows])
        wave = np.array([r["wave_achievable"] for r in rows])
        best = np.array([r["best_achievable"] for r in rows])
        n_flipped = sum(1 for r in rows if r["rect_flipped"] or r["wave_flipped"])
        print(f"{species:>8s}  {len(rows):>4d}  "
              f"{rect.mean():>9.3f}   {np.median(rect):>8.3f}   "
              f"{100*(rect>=0.95).mean():>7.1f}%  "
              f"{100*(rect>=0.999).mean():>6.1f}%  "
              f"{wave.mean():>9.3f}   {np.median(wave):>8.3f}   "
              f"{best.mean():>9.3f}   {n_flipped:>4d}/{len(rows):<4d}")
    print()
    print("[note] all SI values are |signed_si| (achievable selectivity); "
          "#flipped is the number of seeds where the optimiser landed in "
          "the anti-selective basin, equivalent to a positive-SI solution "
          "with target↔non-target swapped.")


def _print_per_nerve(by_species: dict) -> None:
    print()
    print(f"{'species':>6s}  {'sample':<24s}  {'seed':>4s}  "
          f"{'divider':>7s}  {'rect_si':>8s}  {'wave_si':>8s}  {'best':>7s}")
    print("-" * 80)
    for species in ("swine", "human"):
        for r in sorted(by_species.get(species, []), key=lambda x: x["sample"]):
            print(f"{species:>6s}  {r['sample']:<24s}  {r['seed']:>4d}  "
                  f"{r['divider']:>6.1f}°  "
                  f"{r['rect_si']:>+7.3f}   {r['wave_si']:>+7.3f}   "
                  f"{r['best_si']:>+6.3f}")
    print()


def _write_csv(by_species: dict, csv_path: Path) -> None:
    import csv
    rows = sum(by_species.values(), [])
    if not rows:
        print(f"[warn] no records; csv {csv_path} not written", file=sys.stderr)
        return
    keys = ["species", "sample", "seed", "divider", "rect_si", "wave_si",
            "rect_loss", "best_si"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for species in ("swine", "human"):
            for r in sorted(by_species.get(species, []), key=lambda x: x["sample"]):
                w.writerow({"species": species, **{k: r[k] for k in keys[1:]}})
    print(f"[csv]  -> {csv_path}")


def _rebuild_figures() -> int:
    """Run make_figures.py for the four Duke figure targets."""
    cmd = [sys.executable, str(ROOT / "manuscript" / "make_figures.py"),
           "duke_xsections", "duke_xsections_hard",
           "duke_selectivity", "duke_convergence"]
    print(f"\n[rebuild] {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=ROOT)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-nerve", action="store_true",
                    help="print one line per nerve as well as the species summary")
    ap.add_argument("--csv", type=Path, default=None,
                    help="write all per-nerve rows to this CSV file")
    ap.add_argument("--rebuild-figures", action="store_true",
                    help="invoke make_figures.py for the four Duke figure targets")
    args = ap.parse_args(argv)

    jsons = _discover_duke_jsons()
    if not jsons:
        print("[analyze_duke_results] no Duke sweep JSONs found under outputs/",
              file=sys.stderr)
        return 1
    print(f"[analyze_duke_results] found {len(jsons)} per-seed JSONs across "
          f"{len({s for s, _ in jsons})} samples")

    by_species = _summarise(jsons)
    _print_table(by_species)
    if args.per_nerve:
        _print_per_nerve(by_species)
    if args.csv is not None:
        _write_csv(by_species, args.csv.resolve())

    if args.rebuild_figures:
        return _rebuild_figures()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
