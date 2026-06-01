"""Aggregate analysis for selectivity_sweep outputs.

Loads all data_seed_*.json files from outputs/selectivity_sweep/, computes
population-level SI statistics, and produces paper-quality figures.

Run from project root after all array tasks have completed:
    python experiments_v2/analyze_selectivity_sweep.py
"""
from __future__ import annotations

import sys
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments_v2.utils import ensure_dir, save_json

OUT = ensure_dir(ROOT / "outputs" / "selectivity_sweep")


def load_results(out_dir: pathlib.Path) -> list[dict]:
    files = sorted(out_dir.glob("data_seed_*.json"))
    if not files:
        raise FileNotFoundError(f"No seed JSON files found in {out_dir}")
    results = []
    for f in files:
        with open(f) as fh:
            results.append(json.load(fh))
    print(f"Loaded {len(results)} seed files.")
    return results


def main():
    results = load_results(OUT)
    n = len(results)

    si_base = np.array([r["si_baseline"] for r in results])
    si_rect = np.array([r["rect"]["final_si"] for r in results])
    si_wave = np.array([r["waveform"]["final_si"] for r in results])

    # ── summary stats ──────────────────────────────────────────────────────────
    def stats(x):
        return {
            "n": int(len(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "median": float(np.median(x)),
            "q25": float(np.percentile(x, 25)),
            "q75": float(np.percentile(x, 75)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        }

    summary = {
        "n_seeds": n,
        "si_baseline": stats(si_base),
        "si_rect": stats(si_rect),
        "si_waveform": stats(si_wave),
    }
    save_json(summary, OUT / "data_sweep_summary.json")

    print(f"\nPopulation summary (n={n} seeds):")
    for label, arr in [("Baseline", si_base), ("Rect", si_rect), ("Waveform", si_wave)]:
        q25, med, q75 = np.percentile(arr, [25, 50, 75])
        print(f"  {label:10s}  median={med:+.3f}  IQR=[{q25:+.3f}, {q75:+.3f}]  "
              f"min={np.min(arr):+.3f}  max={np.max(arr):+.3f}")

    # ── violin plot ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    vp = ax.violinplot([si_base, si_rect, si_wave], showmedians=True, showextrema=True)
    for pc, c in zip(vp["bodies"], ["C7", "C0", "C2"]):
        pc.set_facecolor(c)
        pc.set_alpha(0.7)
    vp["cmedians"].set_color("k")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Baseline", "Rect", "Waveform"])
    ax.set_ylabel("Selectivity Index (SI)")
    ax.set_title(f"SI distribution — {n} nerve realisations")
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sweep_si_violin.png", dpi=150)
    plt.close(fig)
    print(f"  → {OUT / 'fig_sweep_si_violin.png'}")

    # ── CDF plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    for arr, label, color in [
        (si_base, "Baseline", "C7"),
        (si_rect, "Rect",     "C0"),
        (si_wave, "Waveform", "C2"),
    ]:
        sorted_si = np.sort(arr)
        cdf = np.arange(1, n + 1) / n
        ax.step(sorted_si, cdf, label=label, color=color, where="post")
    ax.set_xlabel("Selectivity Index (SI)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(f"SI CDF — {n} nerve realisations")
    ax.legend()
    ax.set_xlim(-1.05, 1.05)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sweep_si_cdf.png", dpi=150)
    plt.close(fig)
    print(f"  → {OUT / 'fig_sweep_si_cdf.png'}")

    # ── loss curves (mean ± 1σ) ────────────────────────────────────────────────
    n_iter_rect = min(len(r["rect"]["loss_history"]) for r in results)
    n_iter_wave = min(len(r["waveform"]["loss_history"]) for r in results)
    loss_rect = np.array([r["rect"]["loss_history"][:n_iter_rect] for r in results])
    loss_wave = np.array([r["waveform"]["loss_history"][:n_iter_wave] for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for data, ax, title, color in [
        (loss_rect, axes[0], "Rectangular", "C0"),
        (loss_wave, axes[1], "Arbitrary waveform", "C2"),
    ]:
        mu = data.mean(0)
        sig = data.std(0)
        x = np.arange(len(mu))
        ax.plot(x, mu, color=color)
        ax.fill_between(x, mu - sig, mu + sig, alpha=0.2, color=color)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("WQ loss")
        ax.set_title(f"{title}\nmean ± 1σ  (n={n})")
    fig.tight_layout()
    fig.savefig(OUT / "fig_sweep_loss_curves.png", dpi=150)
    plt.close(fig)
    print(f"  → {OUT / 'fig_sweep_loss_curves.png'}")

    print(f"\nAll outputs saved to {OUT}")


if __name__ == "__main__":
    main()
