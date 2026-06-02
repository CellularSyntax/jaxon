"""Aggregate analysis for selectivity_sweep outputs.

Loads all data_seed_*.json files from outputs/selectivity_sweep/, computes
population-level SI statistics, and produces paper-quality figures.

Outputs (in outputs/selectivity_sweep/)
---------------------------------------
  data_sweep_summary.json    — numeric stats for manuscript text
  fig_sweep_si_violin.png    — SI distributions (baseline / rect / waveform)
  fig_sweep_si_cdf.png       — SI CDF for the same three stages
  fig_sweep_loss_curves.png  — loss vs iteration (mean ± 1σ)
  fig_sweep_pair_scatter.png — rect SI vs waveform SI per seed
  fig_sweep_examples.png     — activation maps for low / median / high SI seeds
  fig_sweep_restart_hist.png — distribution of winning LBFGS restart index

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


# ── data loading ──────────────────────────────────────────────────────────────

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


# ── summary stats ─────────────────────────────────────────────────────────────

def _stats(x: np.ndarray) -> dict:
    return {
        "n":      int(len(x)),
        "mean":   float(np.mean(x)),
        "std":    float(np.std(x)),
        "median": float(np.median(x)),
        "q25":    float(np.percentile(x, 25)),
        "q75":    float(np.percentile(x, 75)),
        "min":    float(np.min(x)),
        "max":    float(np.max(x)),
        "frac_ge_0.5": float((x >= 0.5).mean()),
        "frac_ge_0.9": float((x >= 0.9).mean()),
    }


# ── figures ──────────────────────────────────────────────────────────────────

def fig_violin(si_base, si_rect, si_wave, out_path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    vp = ax.violinplot([si_base, si_rect, si_wave],
                        showmedians=True, showextrema=True)
    for pc, c in zip(vp["bodies"], ["C7", "C0", "C2"]):
        pc.set_facecolor(c); pc.set_alpha(0.7)
    vp["cmedians"].set_color("k")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Baseline", "Rect", "Waveform"])
    ax.set_ylabel("Selectivity Index (SI)")
    ax.set_title(f"SI distribution — {len(si_base)} nerve realisations")
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_ylim(-1.05, 1.05)
    for i, vals in enumerate([si_base, si_rect, si_wave]):
        ax.text(i + 1, 1.02, f"med={np.median(vals):+.2f}",
                 ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")


def fig_cdf(si_base, si_rect, si_wave, out_path) -> None:
    n = len(si_base)
    fig, ax = plt.subplots(figsize=(5, 4))
    for arr, label, color in [
        (si_base, "Baseline", "C7"),
        (si_rect, "Rect",     "C0"),
        (si_wave, "Waveform", "C2"),
    ]:
        sorted_si = np.sort(arr)
        cdf = np.arange(1, n + 1) / n
        ax.step(sorted_si, cdf, label=label, color=color, where="post", lw=1.6)
    ax.set_xlabel("Selectivity Index (SI)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(f"SI CDF — {n} nerve realisations")
    ax.legend()
    ax.set_xlim(-1.05, 1.05)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")


def fig_loss_curves(results, out_path) -> None:
    n = len(results)
    n_iter_rect = min(len(r["rect"]["loss_history"]) for r in results)
    n_iter_wave = min(len(r["waveform"]["loss_history"]) for r in results)
    loss_rect = np.array([r["rect"]["loss_history"][:n_iter_rect]     for r in results])
    loss_wave = np.array([r["waveform"]["loss_history"][:n_iter_wave] for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for data, ax, title, color, xlabel in [
        (loss_rect, axes[0], "Rect (LBFGS, best restart)", "C0", "LBFGS iteration"),
        (loss_wave, axes[1], "Waveform (Adam, warm-start)", "C2", "Adam iteration"),
    ]:
        mu = data.mean(0)
        sig = data.std(0)
        x = np.arange(len(mu))
        ax.plot(x, mu, color=color, lw=1.6, label=f"mean (n={n})")
        ax.fill_between(x, mu - sig, mu + sig, alpha=0.25, color=color, label="± 1σ")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("WQ loss")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")


def fig_pair_scatter(si_rect, si_wave, out_path) -> None:
    n = len(si_rect)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(si_rect, si_wave, c="C0", s=28, alpha=0.6, edgecolors="none")
    lo = min(si_rect.min(), si_wave.min()) - 0.05
    hi = max(si_rect.max(), si_wave.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "k:", lw=0.8, label="rect = wave")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Rect SI (LBFGS)")
    ax.set_ylabel("Waveform SI (Adam, warm-started)")
    n_better = int((si_wave > si_rect + 0.01).sum())
    ax.set_title(f"Waveform vs rect SI per seed\n({n_better}/{n} seeds: "
                 f"waveform > rect by >0.01)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")


def fig_examples(results, out_path) -> None:
    """Activation maps for low / median / high waveform SI seeds (3 columns each)."""
    si_wave = np.array([r["waveform"]["final_si"] for r in results])
    if len(results) < 3:
        print(f"  Skipping example maps: only {len(results)} seeds")
        return

    sorted_idx = np.argsort(si_wave)
    picks = [
        ("low",    sorted_idx[0]),
        ("median", sorted_idx[len(results) // 2]),
        ("high",   sorted_idx[-1]),
    ]

    fig, axes = plt.subplots(len(picks), 3, figsize=(11, 3.4 * len(picks)),
                              constrained_layout=True)
    if len(picks) == 1:
        axes = axes[None, :]

    for row_i, (label, seed_i) in enumerate(picks):
        s = results[seed_i]
        x = np.array(s["nerve"]["fiber_x_um"])
        y = np.array(s["nerve"]["fiber_y_um"])
        diam = np.array(s["nerve"]["fiber_diam"])
        target = np.array(s["nerve"]["target_mask"], dtype=bool)
        rect_acts = np.array(s["rect"]["final_acts"])
        wave_acts = np.array(s["waveform"]["final_acts"])

        cols = [
            (None,      "Target classes"),
            (rect_acts, f"Rect SI={s['rect']['final_si']:+.2f}"),
            (wave_acts, f"Waveform SI={s['waveform']['final_si']:+.2f}"),
        ]
        for col_i, (acts_arr, col_label) in enumerate(cols):
            ax = axes[row_i, col_i]
            # Nerve outline
            nerve_circle = plt.Circle((0, 0), 500, fill=False,
                                       edgecolor="k", lw=0.8, ls=":")
            ax.add_patch(nerve_circle)
            if acts_arr is None:
                ax.scatter(x[target],  y[target],  c="C0", s=diam[target]  * 10,
                            label="target", edgecolors="k", lw=0.5)
                ax.scatter(x[~target], y[~target], c="C3", s=diam[~target] * 10,
                            label="off-target", edgecolors="k", lw=0.5)
                ax.legend(fontsize=8, loc="upper right")
            else:
                sc = ax.scatter(x, y, c=acts_arr, cmap="RdYlGn",
                                 vmin=0, vmax=1, s=diam * 10,
                                 edgecolors="k", lw=0.5)
                ax.scatter(x[target], y[target], facecolors="none",
                            edgecolors="C0", linewidths=1.2, s=diam[target] * 12)
                if col_i == 2:
                    plt.colorbar(sc, ax=ax, label="m gate proxy", fraction=0.046)
            ax.set_xlim(-600, 600); ax.set_ylim(-600, 600)
            ax.set_aspect("equal")
            ax.set_xlabel("x (µm)", fontsize=9)
            if col_i == 0:
                ax.set_ylabel(f"{label} SI seed {s['seed']}\ny (µm)", fontsize=9)
            ax.set_title(col_label, fontsize=10)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.3, lw=0.4)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def fig_restart_hist(results, out_path) -> None:
    """Histogram of which LBFGS restart won per seed.

    Restart 0 is the Ve-weighted deterministic init; 1..M-1 are random.
    If restart 0 always wins, multi-restart adds little; if random restarts
    frequently win, multi-restart is paying for itself.
    """
    best_restarts = [r["rect"].get("best_restart", 0) for r in results]
    n_restarts = max(best_restarts) + 1 if best_restarts else 1
    counts = np.bincount(best_restarts, minlength=n_restarts)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(range(n_restarts), counts, color="C0", alpha=0.8)
    ax.set_xlabel("Winning LBFGS restart index")
    ax.set_ylabel("# seeds")
    ax.set_title(f"LBFGS multi-restart: which restart wins per seed (n={len(results)})\n"
                  f"Restart 0 = Ve-weighted deterministic; 1..M-1 = random")
    ax.set_xticks(range(n_restarts))
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                str(int(c)), ha="center", fontsize=9)
    ax.grid(alpha=0.3, lw=0.4, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    results = load_results(OUT)
    n = len(results)

    si_base = np.array([r["si_baseline"]          for r in results])
    si_rect = np.array([r["rect"]["final_si"]     for r in results])
    si_wave = np.array([r["waveform"]["final_si"] for r in results])

    summary = {
        "n_seeds":            n,
        "si_baseline":        _stats(si_base),
        "si_rect":            _stats(si_rect),
        "si_waveform":        _stats(si_wave),
        "wave_minus_rect":    _stats(si_wave - si_rect),
        "rect_time_s":        _stats(np.array([r["rect"]["time_s"]     for r in results])),
        "waveform_time_s":    _stats(np.array([r["waveform"]["time_s"] for r in results])),
    }
    save_json(summary, OUT / "data_sweep_summary.json")

    print(f"\nPopulation summary (n={n} seeds):")
    for label, arr in [("Baseline", si_base), ("Rect", si_rect), ("Waveform", si_wave)]:
        q25, med, q75 = np.percentile(arr, [25, 50, 75])
        print(f"  {label:10s}  median={med:+.3f}  IQR=[{q25:+.3f}, {q75:+.3f}]  "
              f"frac>=0.5={float((arr>=0.5).mean()):.0%}  "
              f"frac>=0.9={float((arr>=0.9).mean()):.0%}")
    print(f"  Wave-rect:  median={np.median(si_wave-si_rect):+.3f}, "
          f"frac waveform better = {float((si_wave > si_rect + 0.01).mean()):.0%}")

    # Figures
    fig_violin(si_base, si_rect, si_wave, OUT / "fig_sweep_si_violin.png")
    fig_cdf(si_base, si_rect, si_wave,    OUT / "fig_sweep_si_cdf.png")
    fig_loss_curves(results,              OUT / "fig_sweep_loss_curves.png")
    fig_pair_scatter(si_rect, si_wave,    OUT / "fig_sweep_pair_scatter.png")
    fig_examples(results,                 OUT / "fig_sweep_examples.png")
    fig_restart_hist(results,             OUT / "fig_sweep_restart_hist.png")

    print(f"\nAll outputs saved to {OUT}")


if __name__ == "__main__":
    main()
