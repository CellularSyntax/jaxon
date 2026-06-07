"""Composite multi-panel manuscript figures from outputs/*/data_*.json.

Each function below builds ONE composite figure used in the manuscript.
All read directly from outputs/, all save to figures/.  Re-run any time
new sweep data lands; only the figures whose source files have moved
will get re-generated (cheap to call from the Makefile).

Run:
    python make_figures.py            # build all figures
    python make_figures.py --force    # rebuild even if up to date
    python make_figures.py validation # build just this one
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUTROOT = HERE.parent / "outputs"
FIGDIR = HERE / "figures"
FIGDIR.mkdir(exist_ok=True)

MODELS = ["mrg", "sundt", "sweeney", "rattay"]
MODEL_LABEL = {
    "mrg": "MRG",
    "sundt": "Sundt",
    "sweeney": "Sweeney",
    "rattay": "Rattay",
}
MODEL_NOMINAL_D = {"mrg": "10.0", "sundt": "1.0", "sweeney": "10.0", "rattay": "1.0"}
COLORS = {"mrg": "#1f77b4", "sundt": "#d62728", "sweeney": "#2ca02c", "rattay": "#9467bd"}

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ─── 1. validation: all 4 models, SD curves + CV + error histogram ───────────
def fig_validation_4models():
    """3-row × 4-column panel: SD curves, CV vs diameter, error histogram."""
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.5))
    for col, model in enumerate(MODELS):
        d_sd = json.loads((OUTROOT / f"{model}_validation/data_{model}_sd.json").read_text())
        d_cv = json.loads((OUTROOT / f"{model}_validation/data_{model}_cv.json").read_text())
        clr = COLORS[model]

        # Row 0: SD curves at nominal diameter (mono cathodic), jax vs pyfibers
        Dnom = MODEL_NOMINAL_D[model]
        D_avail = sorted(d_sd.keys(), key=float)
        if Dnom not in d_sd:
            Dnom = D_avail[len(D_avail) // 2]
        sd = d_sd[Dnom]["mono_c"]
        pws = sorted(sd.keys(), key=float)
        pw_arr = np.array([float(p) for p in pws])
        jax = np.array([sd[p]["jax"] for p in pws])
        pyf = np.array([sd[p]["pyfibers"] for p in pws])
        ax = axes[0, col]
        ax.plot(pw_arr, np.abs(jax), "-", color=clr, lw=1.8, label="jaxon", zorder=2)
        ax.plot(pw_arr, np.abs(pyf), "o", mfc="none", mec="k", mew=1.0, ms=6,
                label="pyfibers (NEURON)", zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("pulse width (ms)"); ax.set_ylabel("|threshold| (mA)")
        ax.set_title(f"{MODEL_LABEL[model]}  $D={Dnom}\\,\\mu$m", color=clr)
        if col == 0:
            ax.legend(loc="upper right", frameon=False)

        # Row 1: CV vs diameter
        ax = axes[1, col]
        D_cv = sorted(d_cv.keys(), key=float)
        D_arr = np.array([float(D) for D in D_cv])
        jax_cv = np.array([d_cv[D]["jax"] for D in D_cv])
        pyf_cv = np.array([d_cv[D]["pyfibers"] for D in D_cv])
        ax.plot(D_arr, jax_cv, "-o", color=clr, lw=1.5, mfc=clr, mec=clr, ms=5, label="jaxon")
        ax.plot(D_arr, pyf_cv, "s", mfc="none", mec="k", mew=1.0, ms=6, label="pyfibers (NEURON)")
        ax.set_xlabel("diameter ($\\mu$m)"); ax.set_ylabel("CV (m/s)")
        ax.set_title("conduction velocity")

        # Row 2: error histogram across ALL SD configs
        errs = []
        for D in d_sd:
            for wf in d_sd[D]:
                for pw in d_sd[D][wf]:
                    e = d_sd[D][wf][pw].get("err_pct")
                    if e is not None and np.isfinite(e):
                        errs.append(abs(e))
        errs = np.array(errs)
        ax = axes[2, col]
        ax.hist(errs, bins=np.linspace(0, errs.max() * 1.05, 25),
                color=clr, alpha=0.85, edgecolor="black", lw=0.4)
        ax.axvline(np.median(errs), color="k", lw=1.2, ls="--",
                   label=f"median {np.median(errs):.2f}\\%")
        ax.axvline(1.0, color="red", lw=1.0, ls=":", label="1\\% tol")
        ax.set_xlabel("|threshold error| (%)"); ax.set_ylabel("count")
        ax.set_title(f"$n={len(errs)}$  peak {errs.max():.2f}\\%")
        ax.legend(loc="upper right", frameon=False)
        ax.set_xlim(0, max(errs.max() * 1.05, 1.05))

    plt.suptitle("Validation of jaxon against pyfibers-wrapped NEURON "
                 "across all four fiber models", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIGDIR / "fig_validation_4models.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


# ─── 2. intracellular traces: all 4 models, jax vs NEURON overlay ─────────────
def fig_traces_4models():
    """1-row × 4-column trace overlay at the centre node."""
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8))
    for col, model in enumerate(MODELS):
        p = OUTROOT / f"{model}_validation/data_{model}_traces.json"
        if not p.exists():
            continue
        tr = json.loads(p.read_text())
        clr = COLORS[model]
        t_jax = np.array(tr["t_intra_jax"])
        vm_jax = np.array(tr["vm_intra_jax"])
        t_nrn = np.array(tr["t_intra_nrn"])
        vm_nrn = np.array(tr["vm_intra_nrn"])
        # Pick centre node trace (last axis index ~middle)
        c = vm_jax.shape[1] // 2 if vm_jax.ndim > 1 else None
        v_jax = vm_jax[:, c] if c is not None else vm_jax
        c2 = vm_nrn.shape[1] // 2 if vm_nrn.ndim > 1 else None
        v_nrn = vm_nrn[:, c2] if c2 is not None else vm_nrn
        ax = axes[col]
        ax.plot(t_nrn, v_nrn, "k--", lw=1.5, alpha=0.8, label="NEURON")
        ax.plot(t_jax, v_jax, "-", color=clr, lw=1.4, label="jaxon")
        ax.set_xlabel("time (ms)"); ax.set_ylabel("$V_m$ (mV)" if col == 0 else "")
        ax.set_title(f"{MODEL_LABEL[model]}", color=clr)
        ax.legend(loc="upper right", frameon=False, fontsize=7)
        peak_diff_mV = float(np.max(np.abs(np.interp(t_jax, t_nrn, v_nrn) - v_jax)))
        ax.text(0.04, 0.05, f"$\\Delta V_m^{{peak}}={peak_diff_mV:.2f}$ mV",
                transform=ax.transAxes, fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey"))
    plt.suptitle("Intracellular membrane-potential traces: jaxon vs NEURON",
                 fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = FIGDIR / "fig_traces_4models.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


# ─── 3. propagation phenomena: collision, kHz block, DC block ─────────────────
def fig_phenomena():
    """2×2: AP collision, kHz block, DC block, jax vs pyfibers."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))

    # AP collision: snapshot Vm along the fiber at the moment two APs meet
    ax = axes[0, 0]
    d = json.loads((OUTROOT / "ap_collision/data_ap_collision.json").read_text())
    r = d["results"][2]  # pick D=10 µm
    snap_t = d["snapshot_t_ms"]
    snaps_jax = np.array(r["snapshots_nodes_mV"])  # [n_snap, n_nodes]
    snaps_pf = np.array(r["snapshots_nodes_pf_mV"])
    n_nodes = r["n_nodes"]
    nodes = np.arange(n_nodes)
    for i, t in enumerate(snap_t):
        ax.plot(nodes, snaps_pf[i], "k--", lw=0.9, alpha=0.5)
        ax.plot(nodes, snaps_jax[i], "-", color=plt.cm.viridis(i / len(snap_t)),
                lw=1.5, label=f"t={t:.2f} ms")
    ax.set_xlabel("node index"); ax.set_ylabel("$V_m$ (mV)")
    ax.set_title(f"AP collision (MRG $D={r['diameter_um']}\\,\\mu$m): "
                 f"two cathodic pulses from opposite ends annihilate on meeting")
    ax.legend(loc="upper right", frameon=False, fontsize=7, ncol=2)
    ax.text(0.97, 0.05,
            f"peak $|\\Delta V_m|$ jax/NEURON = {abs(r['vm_peak_mV']-r['vm_peak_pf_mV'])*1000:.2f} $\\mu$V",
            transform=ax.transAxes, ha="right", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey"))

    # kHz block: distal-node Vm trace; pace pulses are blocked by 20 kHz hold
    ax = axes[0, 1]
    d = json.loads((OUTROOT / "khz_block/data_khz_block.json").read_text())
    pace = d["pace"]
    for r in d["results"]:
        t = np.array(r["t_pf_ms"])
        v_pf = np.array(r["vm_pf_90"])
        v_jx = np.array(r["vm_jax_90"])
        t_jx = np.linspace(t[0], t[-1], len(v_jx))
        ax.plot(t, v_pf, "k--", lw=0.6, alpha=0.5)
        ax.plot(t_jx, v_jx, "-", lw=1.0,
                label=f"block $A$={r['amp_mA']:.2f} mA: {r['n_aps_jax']}/{pace['n_pulses']} APs")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("$V_m$ at distal node (mV)")
    ax.set_title(f"kHz conduction block ({d['khz_freq']} kHz, MRG $D={d['diameter_um']}\\,\\mu$m)")
    ax.legend(loc="upper right", frameon=False, fontsize=7)

    # DC block: distal-node Vm trace; cathodic DC of varying amplitude
    ax = axes[1, 0]
    d = json.loads((OUTROOT / "dc_block/data_dc_block.json").read_text())
    for r in d["results"]:
        t_jx = np.array(r["jax_t_ms"]); v_jx = np.array(r["jax_vm_nodes"])
        t_pf = np.array(r["pf_t_ms"]);   v_pf = np.array(r["pf_vm_nodes"])
        v_jx_d = v_jx[:, -1] if v_jx.ndim > 1 else v_jx
        v_pf_d = v_pf[:, -1] if v_pf.ndim > 1 else v_pf
        ax.plot(t_pf, v_pf_d, "k--", lw=0.6, alpha=0.5)
        ax.plot(t_jx, v_jx_d, "-", lw=1.2,
                label=f"$\\alpha$={r['amp_factor']:.2f}, $A$={r['amp_mA']*1000:.0f} $\\mu$A")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("$V_m$ at distal node (mV)")
    ax.set_title(f"DC block (MRG $D={d['diameter_um']}\\,\\mu$m): "
                 f"hyperpolarising block prevents AP propagation past source")
    ax.legend(loc="upper right", frameon=False, fontsize=7)

    # Spike desync: sync index vs cathodic block amplitude, one curve per IFR
    ax = axes[1, 1]
    d = json.loads((OUTROOT / "spike_desync/data_spike_desync.json").read_text())
    ifrs = sorted(d["results"].keys(), key=int)
    cols = plt.cm.plasma(np.linspace(0.15, 0.85, len(ifrs)))
    for i, ifr in enumerate(ifrs):
        rr = d["results"][ifr]
        amps = np.array(rr["amps_mA"])
        sync = np.array(rr["sync"])
        ax.plot(amps, sync, "-o", color=cols[i], lw=1.6, ms=5,
                label=f"{ifr} Hz IFR")
    ax.axhline(1.0, color="grey", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlabel("cathodic perturbation amplitude (mA)")
    ax.set_ylabel("sync index  (1 = locked, 0 = no spikes)")
    ax.set_title(f"Spike desynchronisation (MRG $D={d['diameter_um']}\\,\\mu$m, "
                 f"{d['n_ref_spikes']} reference spikes)")
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.set_ylim(-0.05, 1.1)

    plt.suptitle("Propagation phenomena reproduced under jaxon (lines) "
                 "vs NEURON (dashed)", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIGDIR / "fig_phenomena.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


# ─── 4. scaling: all 7 models on log-log ──────────────────────────────────────
def fig_scaling_allmodels():
    """log-log of wall time vs N for all benchmarked models."""
    d = json.loads((OUTROOT / "scaling/data_scaling.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    cmap = plt.cm.tab10
    for i, m in enumerate(d["models"]):
        clr = cmap(i % 10)
        N = np.array(m["N"])
        py = np.array([m["pyfibers"].get(str(n), np.nan) for n in N])
        jx = np.array([m["jaxley_gpu"]["run"].get(str(n), np.nan) for n in N])
        # Panel A: absolute times
        axes[0].plot(N, py, "--o", color=clr, lw=1.2, mfc="none", ms=5,
                     label=f"pyfibers  {m['model']}", alpha=0.65)
        axes[0].plot(N, jx, "-",  color=clr, lw=2.0,
                     label=f"jaxon     {m['model']}")
        # Panel B: speedup
        with np.errstate(divide="ignore", invalid="ignore"):
            sp = py / jx
        axes[1].plot(N, sp, "-o", color=clr, lw=1.6, ms=5, label=m["model"])
    for ax in axes:
        ax.set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("number of fibres $N$")
    axes[0].set_ylabel("wall time per forward sim (s)")
    axes[0].set_title("Absolute throughput")
    axes[0].legend(loc="upper left", fontsize=6, ncol=2, frameon=False)
    axes[1].set_yscale("log")
    axes[1].axhline(1.0, color="grey", lw=0.8, ls="--", alpha=0.5)
    axes[1].set_xlabel("number of fibres $N$")
    axes[1].set_ylabel("jaxon speedup vs pyfibers ($\\times$)")
    axes[1].set_title("Speedup")
    axes[1].legend(loc="upper left", fontsize=7, frameon=False)
    plt.suptitle("Scaling on a single NVIDIA A16 GPU: jaxon vs pyfibers "
                 "(8-thread CPU baseline)", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIGDIR / "fig_scaling_allmodels.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


# ─── 5. selectivity: per-seed summary across all completed Phase 3 seeds ──────
def fig_selectivity_summary():
    """Per-seed SI bar chart from completed Phase 3 seeds."""
    seeds = sorted((OUTROOT / "selectivity_sweep_phase3_manuscript").glob("data_seed_*.json"))
    if not seeds:
        return
    data = []
    for p in seeds:
        d = json.loads(p.read_text())
        data.append({
            "seed": d["seed"],
            "si_baseline": d["si_baseline"],
            "rect": d["rect"]["final_si"],
            "wave": d["waveform"]["final_si"],
            "loss": d["rect"]["final_loss"],
        })
    n = len(data)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    seeds = [r["seed"] for r in data]
    rect = np.array([r["rect"] for r in data])
    wave = np.array([r["wave"] for r in data])
    base = np.array([r["si_baseline"] for r in data])

    # Panel A: per-seed bar comparison
    x = np.arange(n)
    w = 0.28
    axes[0].bar(x - w, base, w, color="#cccccc", label=f"baseline (mean {base.mean():.2f})")
    axes[0].bar(x,     rect, w, color="#1f77b4", label=f"rect LBFGS (mean {rect.mean():.3f})")
    axes[0].bar(x + w, wave, w, color="#2ca02c", label=f"wave Adam (mean {wave.mean():.3f})")
    axes[0].axhline(0.95, color="red", lw=0.8, ls="--", alpha=0.6, label="SI=0.95")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"s{s}" for s in seeds], fontsize=8)
    axes[0].set_xlabel("Phase 3 seed")
    axes[0].set_ylabel("selectivity index")
    axes[0].set_title(f"Per-seed selectivity ({n}/100 completed)")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc="lower right", fontsize=7, frameon=False)

    # Panel B: rect SI distribution + LBFGS loss trajectories
    axes[1].hist(rect, bins=np.linspace(0.85, 1.02, 18), color="#1f77b4", alpha=0.85,
                 edgecolor="black", lw=0.4)
    axes[1].axvline(rect.mean(), color="red", lw=1.2, ls="--",
                    label=f"mean {rect.mean():.3f}")
    axes[1].axvline(np.median(rect), color="black", lw=1.0, ls=":",
                    label=f"median {np.median(rect):.3f}")
    axes[1].set_xlabel("rect SI (LBFGS multi-start)")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"rect SI distribution (n={n} seeds)")
    axes[1].legend(loc="upper left", fontsize=8, frameon=False)
    axes[1].set_xlim(0.85, 1.02)

    plt.suptitle(f"Selectivity optimisation across {n} Hussain-style nerve "
                 "realisations (single-diameter $D=5.7$ $\\mu$m)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = FIGDIR / "fig_selectivity_summary.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


# ─── orchestration ────────────────────────────────────────────────────────────
ALL = {
    "validation": fig_validation_4models,
    "traces":     fig_traces_4models,
    "phenomena":  fig_phenomena,
    "scaling":    fig_scaling_allmodels,
    "selectivity": fig_selectivity_summary,
}

def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", nargs="*", default=[],
                        help="Figures to build (default: all). "
                             f"Choices: {list(ALL)} or 'all'.")
    args = parser.parse_args(argv)
    if not args.targets or "all" in args.targets:
        targets = list(ALL)
    else:
        bad = [t for t in args.targets if t not in ALL]
        if bad:
            parser.error(f"unknown target(s): {bad}; choose from {list(ALL)} or 'all'")
        targets = args.targets
    for t in targets:
        print(f"[make_figures] {t}")
        try:
            ALL[t]()
        except Exception as e:
            print(f"  ERROR in {t}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
