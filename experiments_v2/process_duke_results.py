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
    l1   = raw.get("rect_l1") or {}
    wav  = raw.get("waveform") or {}
    cluster = raw.get("cluster") or {}

    rect_si = float(rect.get("achievable_si", abs(rect.get("final_si", 0.0))))
    wave_si = float(wav .get("achievable_si", abs(wav .get("final_si", 0.0))))
    if l1:
        l1_si = float(l1.get("achievable_si", abs(l1.get("final_si", 0.0))))
        l1_n_active = int(l1.get("n_active_contacts",
                                    _n_active(l1.get("amps_mA"))))
        l1_time = float(l1.get("time_s", float("nan")))
    else:
        l1_si = float("nan"); l1_n_active = 0; l1_time = float("nan")

    rect_n_active = _n_active(rect.get("amps_mA"))
    wave_n_active = _n_active(np.array(wav.get("u_mA", [])).flatten())

    return dict(
        sample=sample,
        species=_species_of(sample),
        n_target_fibers=int(cluster.get("n_target_fibers", 0)),
        sector_start=cluster.get("window_start_deg"),
        sector_end=cluster.get("window_end_deg"),
        rect_si=rect_si, rect_n_active=rect_n_active,
        rect_time=float(rect.get("time_s", float("nan"))),
        l1_si=l1_si, l1_n_active=l1_n_active, l1_time=l1_time,
        wave_si=wave_si, wave_n_active=wave_n_active,
        wave_time=float(wav.get("time_s", float("nan"))),
        best_si=float(max(rect_si, l1_si if np.isfinite(l1_si) else 0.0,
                            wave_si)),
        agree_within_tol=(np.isfinite(l1_si)
                            and abs(rect_si - l1_si) <= AGREEMENT_TOL),
        delta_si=float(rect_si - l1_si) if np.isfinite(l1_si) else float("nan"),
    )


def _discover() -> list[dict]:
    if not SWEEP.exists():
        return []
    out = []
    for d in sorted(SWEEP.iterdir()):
        if not d.is_dir():
            continue
        j = d / "data_seed_0000.json"
        if not j.exists():
            continue
        try:
            out.append(_load_one(j))
        except Exception as e:
            print(f"[warn] failed to parse {j}: {e}")
    return out


# ── Console output ──────────────────────────────────────────────────────────

def _print_per_sample(rows: list[dict]) -> None:
    print()
    print(f"{'sample':<22} {'species':<6} {'n_tgt':>5} {'rect':>6} "
          f"{'L1':>6} {'wave':>6} {'best':>6}  {'|d|<=0.05':>9}")
    print("-" * 80)
    for r in sorted(rows, key=lambda x: (x["species"], x["sample"])):
        print(f"{r['sample']:<22} {r['species']:<6} "
              f"{r['n_target_fibers']:>5d} "
              f"{r['rect_si']:>6.3f} "
              f"{r['l1_si']:>6.3f} "
              f"{r['wave_si']:>6.3f} "
              f"{r['best_si']:>6.3f}  "
              f"{('Y' if r['agree_within_tol'] else 'N'):>9}")


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
        l1_si   = np.array([r["l1_si"]   for r in sub if np.isfinite(r["l1_si"])])
        wave_si = np.array([r["wave_si"] for r in sub])
        delta   = np.array([abs(r["delta_si"]) for r in sub
                              if np.isfinite(r["delta_si"])])
        agree   = sum(r["agree_within_tol"] for r in sub)
        n       = len(sub)
        n_l1    = sum(np.isfinite(r["l1_si"]) for r in sub)
        out[sp] = dict(
            n=n,
            rect_mean=float(rect_si.mean()),
            rect_median=float(np.median(rect_si)),
            rect_pct_95=float(100 * (rect_si >= 0.95).mean()),
            l1_n=n_l1,
            l1_mean=float(l1_si.mean()) if l1_si.size else float("nan"),
            l1_median=float(np.median(l1_si)) if l1_si.size else float("nan"),
            l1_pct_95=float(100 * (l1_si >= 0.95).mean()) if l1_si.size else float("nan"),
            wave_mean=float(wave_si.mean()),
            wave_median=float(np.median(wave_si)),
            wave_pct_95=float(100 * (wave_si >= 0.95).mean()),
            agree_frac=100.0 * agree / max(n_l1, 1),
            delta_mean_abs=float(delta.mean()) if delta.size else float("nan"),
            rect_active_med=float(np.median([r["rect_n_active"]
                                                for r in sub])),
            l1_active_med=float(np.median([r["l1_n_active"]
                                              for r in sub
                                              if r["l1_n_active"] > 0]))
                          if any(r["l1_n_active"] > 0 for r in sub)
                          else float("nan"),
        )
    return out


def _print_summary(s: dict) -> None:
    print()
    print(f"{'species':<8} {'n':>3} | "
          f"{'rect mean/med/%>=95':>22} | "
          f"{'L1 mean/med/%>=95':>22} | "
          f"{'wave mean/med/%>=95':>22} | "
          f"{'agree %':>8} {'<|d|>':>7}")
    print("-" * 110)
    for sp in ("swine", "human", "all"):
        if sp not in s:
            continue
        r = s[sp]
        print(f"{sp:<8} {r['n']:>3d} | "
              f"{r['rect_mean']:>6.3f} {r['rect_median']:>6.3f} {r['rect_pct_95']:>5.1f}% | "
              f"{r['l1_mean']:>6.3f} {r['l1_median']:>6.3f} {r['l1_pct_95']:>5.1f}% | "
              f"{r['wave_mean']:>6.3f} {r['wave_median']:>6.3f} {r['wave_pct_95']:>5.1f}% | "
              f"{r['agree_frac']:>7.1f}% {r['delta_mean_abs']:>7.3f}")
    print()


# ── CSV ────────────────────────────────────────────────────────────────────

def _write_csv(rows: list[dict]) -> None:
    keys = ["species", "sample", "n_target_fibers",
            "rect_si", "rect_n_active", "rect_time",
            "l1_si", "l1_n_active", "l1_time",
            "wave_si", "wave_n_active", "wave_time",
            "best_si", "delta_si", "agree_within_tol"]
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

    def trio(prefix):
        return (cell(s[sp][f"{prefix}_mean"]) + " / "
                + cell(s[sp][f"{prefix}_median"]) + " / "
                + cell(s[sp][f"{prefix}_pct_95"], "{:.0f}\\%"))

    rows = []
    for sp in ("swine", "human", "all"):
        if sp not in s:
            continue
        rows.append((sp, s[sp]))

    parts: list[str] = []
    parts.append("% Auto-generated by experiments_v2/process_duke_results.py")
    parts.append("% Replaces \\needdata{} placeholders in tab:duke-summary and")
    parts.append("% tab:duke-probe-vs-l1.  Re-run after every cluster sweep.")
    parts.append("")
    parts.append("\\begin{table}[h]")
    parts.append("  \\centering")
    parts.append("  \\caption{Achievable selectivity index $\\mathrm{SI}$ on the Duke "
                  "microCT-derived vagus nerve cohort, by species and "
                  "optimisation path.  ``Probe'' is the probe-based smart-init "
                  "Adam-FD; ``L1'' is the random-init L1-discovery validation "
                  "path.  Cells: mean / median / \\% reaching SI $\\geq 0.95$.}")
    parts.append("  \\label{tab:duke-summary}")
    parts.append("  \\begin{tabular}{lrlll}")
    parts.append("    \\toprule")
    parts.append("    Species & $n$ & Probe SI & L1 SI & Wave SI \\\\")
    parts.append("    \\midrule")
    for sp, r in rows:
        parts.append("    " + sp + " & " + str(r["n"]) + " & "
                      + trio("rect") + " & " + trio("l1") + " & "
                      + trio("wave") + " \\\\")
    parts.append("    \\bottomrule")
    parts.append("  \\end{tabular}")
    parts.append("\\end{table}")
    parts.append("")
    parts.append("\\begin{table}[h]")
    parts.append("  \\centering")
    parts.append("  \\caption{Probe-based smart init vs random-init L1-discovery "
                  "agreement on the Duke cohort.  Two paths are said to "
                  "\\emph{agree} if $|\\Delta \\mathrm{SI}| \\leq 0.05$ "
                  "(experimental noise floor).  $n_{\\mathrm{active}}$ is the "
                  "median number of non-zero contacts in the final amplitude "
                  "vector.}")
    parts.append("  \\label{tab:duke-probe-vs-l1}")
    parts.append("  \\begin{tabular}{lrrrr}")
    parts.append("    \\toprule")
    parts.append("    & mean $|\\Delta \\mathrm{SI}|$ & \\% agree "
                  "& probe $n_{\\mathrm{active}}$ & L1 $n_{\\mathrm{active}}$ \\\\")
    parts.append("    \\midrule")
    for sp, r in rows:
        parts.append("    " + sp + " & "
                      + cell(r["delta_mean_abs"]) + " & "
                      + cell(r["agree_frac"], "{:.0f}\\%") + " & "
                      + cell(r["rect_active_med"], "{:.0f}") + " & "
                      + cell(r["l1_active_med"], "{:.0f}") + " \\\\")
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
