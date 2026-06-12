"""Process Duke selectivity sweep results into:

- a per-sample CSV (outputs/duke_sweeps/duke_si.csv)
- a printed console table by species
- a LaTeX snippet (manuscript/figures/duke/duke_tables.tex) with the
  cohort-summary and probe-vs-L1 agreement tables in the same shape as
  the tab:duke-summary and tab:duke-probe-vs-l1 placeholders in
  03_results.tex

Run from project root::

    python -m experiments_v2.process_duke_results
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median

import numpy as np

ROOT     = Path(__file__).resolve().parent.parent
SWEEP    = ROOT / "outputs" / "duke_sweeps"
OUT_TBL  = ROOT / "manuscript" / "figures" / "duke" / "duke_tables.tex"
OUT_CSV  = SWEEP / "duke_si.csv"

AGREEMENT_TOL = 0.05   # |Delta SI| threshold for "agree within noise"


def _species_of(sample: str) -> str:
    return "human" if sample.startswith("human") else "swine"


def _n_active(amps_mA) -> int:
    if amps_mA is None:
        return 0
    return int(np.sum(np.abs(np.asarray(amps_mA, dtype=float)) > 1e-9))


def _firing(d: dict) -> dict:
    """Return target/off-target fired counts from the 'firing' dict if
    present, else compute from final_acts."""
    f = d.get("firing")
    if isinstance(f, dict):
        return {k: int(v) for k, v in f.items()
                  if isinstance(v, (int, float))}
    return {}


def _load_one(jpath: Path) -> dict:
    raw = json.loads(jpath.read_text())
    sample = raw.get("sample", jpath.parent.name)
    rect = raw.get("rect") or {}
    cluster = raw.get("cluster") or {}

    rect_si = float(rect.get("achievable_si", abs(rect.get("final_si", 0.0))))
    rect_n_active = _n_active(rect.get("amps_mA"))

    return dict(
        sample=sample,
        species=_species_of(sample),
        n_target_fibers=int(cluster.get("n_target_fibers", 0)),
        sector_start=cluster.get("window_start_deg"),
        sector_end=cluster.get("window_end_deg"),
        rect_si=rect_si,
        rect_n_active=rect_n_active,
        rect_time=float(rect.get("time_s", float("nan"))),
    )


def _discover() -> list[dict]:
    """Load every data_seed_NNNN.json across all sample dirs.

    With the 4-position sweep each sample directory contains up to
    CLUSTER_N_POSITIONS (default 4) JSON files — one per angular window.
    Each is treated as an independent row so the summary statistics cover
    all positions, not just seed 0.
    """
    if not SWEEP.exists():
        return []
    out = []
    for d in sorted(SWEEP.iterdir()):
        if not d.is_dir():
            continue
        for j in sorted(d.glob("data_seed_*.json")):
            try:
                out.append(_load_one(j))
            except Exception as e:
                print(f"[warn] failed to parse {j}: {e}")
    return out


# ── Console output ──────────────────────────────────────────────────────────

def _print_per_sample(rows: list[dict]) -> None:
    print()
    print(f"{'sample':<22} {'species':<6} {'n_tgt':>5} {'probe SI':>8}")
    print("-" * 46)
    for r in sorted(rows, key=lambda x: (x["species"], x["sample"])):
        print(f"{r['sample']:<22} {r['species']:<6} "
              f"{r['n_target_fibers']:>5d} "
              f"{r['rect_si']:>8.3f}")


def _summary_by_species(rows: list[dict]) -> dict:
    out = {}
    for sp in ("swine", "human", "all"):
        if sp == "all":
            sub = rows
        else:
            sub = [r for r in rows if r["species"] == sp]
        if not sub:
            continue
        rect_si = np.array([r["rect_si"] for r in sub])
        out[sp] = dict(
            n=len(sub),
            rect_mean=float(rect_si.mean()),
            rect_median=float(np.median(rect_si)),
            rect_pct_90=float(100 * (rect_si >= 0.90).mean()),
            rect_active_med=float(np.median([r["rect_n_active"] for r in sub])),
        )
    return out


def _print_summary(s: dict) -> None:
    print()
    print(f"{'species':<8} {'n':>3} | {'probe mean/med/%>=90':>22}")
    print("-" * 40)
    for sp in ("swine", "human", "all"):
        if sp not in s:
            continue
        r = s[sp]
        print(f"{sp:<8} {r['n']:>3d} | "
              f"{r['rect_mean']:>6.3f} {r['rect_median']:>6.3f} {r['rect_pct_90']:>5.1f}%")
    print()


# ── CSV ────────────────────────────────────────────────────────────────────

def _write_csv(rows: list[dict]) -> None:
    keys = ["species", "sample", "n_target_fibers",
            "rect_si", "rect_n_active", "rect_time"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["species"], x["sample"])):
            w.writerow({k: r.get(k) for k in keys})
    print(f"[csv]  -> {OUT_CSV}")


# ── LaTeX snippet ──────────────────────────────────────────────────────────

def _tex_table(s: dict) -> str:
    def cell(v, fmt="{:.3f}"):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "--"
        return fmt.format(v)

    tbl_rows = []
    for sp in ("swine", "human", "all"):
        if sp not in s:
            continue
        tbl_rows.append((sp, s[sp]))

    parts: list[str] = []
    parts.append("% Auto-generated by experiments_v2/process_duke_results.py")
    parts.append("% Replaces \\needdata{} placeholder in tab:duke-summary.")
    parts.append("% Re-run after every cluster sweep.")
    parts.append("")
    parts.append("\\begin{table}[h]")
    parts.append("  \\centering")
    parts.append("  \\caption{Achievable selectivity index $\\mathrm{SI}$ on the Duke "
                  "microCT-derived vagus nerve cohort (probe Adam-FD), by species.  "
                  "Cells: mean / median / \\% reaching SI $\\geq 0.90$.  "
                  "$n_{\\mathrm{active}}$ is the median number of non-zero contacts.}")
    parts.append("  \\label{tab:duke-summary}")
    parts.append("  \\begin{tabular}{lrll}")
    parts.append("    \\toprule")
    parts.append("    Species & $n$ & Probe SI (mean / med / \\%$\\geq$0.90) & $n_{\\mathrm{active}}$ \\\\")
    parts.append("    \\midrule")
    for sp, r in tbl_rows:
        trio = (cell(r["rect_mean"]) + " / "
                + cell(r["rect_median"]) + " / "
                + cell(r["rect_pct_90"], "{:.0f}\\%"))
        parts.append("    " + sp + " & " + str(r["n"]) + " & "
                      + trio + " & "
                      + cell(r["rect_active_med"], "{:.0f}") + " \\\\")
    parts.append("    \\bottomrule")
    parts.append("  \\end{tabular}")
    parts.append("\\end{table}")
    return "\n".join(parts) + "\n"


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    rows = _discover()
    print(f"[discover] {len(rows)} samples with completed seed_0000 in "
          f"{SWEEP}")
    if not rows:
        raise SystemExit("Nothing to process.")
    _print_per_sample(rows)
    s = _summary_by_species(rows)
    _print_summary(s)
    _write_csv(rows)
    OUT_TBL.parent.mkdir(parents=True, exist_ok=True)
    OUT_TBL.write_text(_tex_table(s))
    print(f"[tex]  -> {OUT_TBL}")


if __name__ == "__main__":
    main()
