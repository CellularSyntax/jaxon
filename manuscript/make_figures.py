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
# Output search path: when sweep data is refreshed from the cluster we
# drop the new JSONs into manuscript/outputs/ so the manuscript figures
# pick them up without overwriting the original repo outputs/.
# Look in manuscript/outputs/<subdir>/ first; fall back to ../outputs/.
_LOCAL_OUTROOT = HERE / "outputs"
_REPO_OUTROOT  = HERE.parent / "outputs"
def _outdir(subdir: str) -> Path:
    """Resolve outputs subdir, preferring the manuscript-local copy."""
    local = _LOCAL_OUTROOT / subdir
    repo  = _REPO_OUTROOT  / subdir
    return local if local.exists() else repo

OUTROOT = _REPO_OUTROOT   # kept for back-compat with old call sites
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
        valid = np.isfinite(jax_cv) | np.isfinite(pyf_cv)
        if valid.any():
            ax.plot(D_arr, jax_cv, "-o", color=clr, lw=1.5, mfc=clr, mec=clr,
                    ms=5, label="jaxon")
            ax.plot(D_arr, pyf_cv, "s", mfc="none", mec="k", mew=1.0, ms=6,
                    label="pyfibers (NEURON)")
            ax.set_xlabel("diameter ($\\mu$m)"); ax.set_ylabel("CV (m/s)")
            ax.set_title("conduction velocity")
        else:
            # All-NaN: don't render an empty subplot; show a placeholder note.
            ax.text(0.5, 0.5,
                    "CV measurement\nunavailable\n(re-run validation)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, style="italic", color="grey",
                    bbox=dict(boxstyle="round,pad=0.6", fc="white",
                              ec="lightgrey", lw=0.6))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("conduction velocity")
            for spine in ax.spines.values():
                spine.set_visible(False)

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


# ─── 2. trace agreement: extracellular AND intracellular, all 4 models ────────
def fig_traces_4models():
    """4-row × 4-column: stim + Vm for extracellular and intracellular.

    The manuscript ultimately targets extracellular PNS, so the
    extracellular case is the validation that matters for the
    selectivity-optimisation story.  We also show the intracellular
    case to document that the integrator reproduces NEURON in the
    simpler regime where there is no extracellular forcing.

    Layout per column (= one fibre model):
      Row 0:  extracellular cathodic stimulus I_e(t) [mA]
      Row 1:  V_m response to extracellular stim, jaxon vs NEURON
      Row 2:  intracellular current injection I_i(t) [nA]
      Row 3:  V_m response to intracellular injection, jaxon vs NEURON

    Per-model pulse widths and amplitudes match validate_*.py.
    """
    DELAY_MS = 1.0
    # Extracellular pulse widths
    PW_E_MS = {"mrg": 0.1, "sweeney": 0.1, "sundt": 0.2, "rattay": 0.2}
    # Intracellular: pulse widths + amplitudes (nA)
    PW_I_MS = {"mrg": 0.1, "sweeney": 0.1, "sundt": 0.2, "rattay": 0.2}
    AMP_I_NA = {"mrg": 1.0, "sweeney": 3.0, "sundt": 0.5, "rattay": 0.5}

    fig, axes = plt.subplots(
        4, 4, figsize=(12, 8.0),
        gridspec_kw={"height_ratios": [1, 3, 1, 3],
                     "hspace": 0.18, "wspace": 0.30},
        sharex="col",
    )

    for col, model in enumerate(MODELS):
        p = _outdir(f"{model}_validation") / f"data_{model}_traces.json"
        if not p.exists():
            continue
        tr = json.loads(p.read_text())
        clr = COLORS[model]
        D = float(tr["diameter"])

        # ───── Extracellular block ────────────────────────────────────────────
        t_jax_e  = np.array(tr["t_extra_jax"])
        vm_jax_e = np.array(tr["vm_extra_jax"])
        t_nrn_e  = np.array(tr["t_extra_nrn"])
        vm_nrn_e = np.array(tr["vm_extra_nrn"])
        amp_e = float(tr["amp_extra_mA"])
        thr_e = float(tr["threshold_extra_mA"])
        v_jax_e = vm_jax_e[:, vm_jax_e.shape[1] // 2] if vm_jax_e.ndim > 1 else vm_jax_e
        v_nrn_e = vm_nrn_e[:, vm_nrn_e.shape[1] // 2] if vm_nrn_e.ndim > 1 else vm_nrn_e

        ax_s = axes[0, col]
        pw_e = PW_E_MS[model]
        t_stim_e = np.linspace(0, t_jax_e[-1], 2000)
        stim_e = np.where((t_stim_e >= DELAY_MS) & (t_stim_e < DELAY_MS + pw_e),
                          amp_e, 0.0)
        ax_s.fill_between(t_stim_e, 0, stim_e, color=clr, alpha=0.75, lw=0)
        ax_s.axhline(0,      color="black", lw=0.4, alpha=0.6)
        ax_s.axhline(thr_e,  color="grey",  lw=0.8, ls=":", alpha=0.8)
        ax_s.set_title(f"{MODEL_LABEL[model]}  $D={D:g}\\,\\mu$m",
                       color=clr, fontsize=10, pad=4)
        ymin = min(amp_e, thr_e) * 1.15
        ax_s.set_ylim(ymin, abs(ymin) * 0.12)
        if col == 0:
            ax_s.set_ylabel("$I_e$ (mA)\nextracellular", fontsize=8)
        ax_s.tick_params(axis="y", labelsize=7)
        ax_s.text(0.97, 0.08,
                  f"amp = {amp_e:.2f} mA\nthr = {thr_e:.2f} mA",
                  transform=ax_s.transAxes, ha="right", va="bottom",
                  fontsize=6, family="DejaVu Sans Mono",
                  bbox=dict(boxstyle="round,pad=0.18", fc="white",
                            ec="grey", lw=0.4))

        ax_v = axes[1, col]
        ax_v.plot(t_nrn_e, v_nrn_e, "k--", lw=1.4, alpha=0.85, label="NEURON")
        ax_v.plot(t_jax_e, v_jax_e, "-",   color=clr, lw=1.4, label="jaxon")
        if col == 0:
            ax_v.set_ylabel("$V_m$ at centre node (mV)", fontsize=8)
        ax_v.tick_params(axis="both", labelsize=7)
        ax_v.legend(loc="upper right", frameon=False, fontsize=7)
        peak_diff_e = float(
            np.max(np.abs(np.interp(t_jax_e, t_nrn_e, v_nrn_e) - v_jax_e))
        )
        ax_v.text(0.04, 0.05, f"$\\Delta V_m^{{peak}}={peak_diff_e:.2f}$ mV",
                  transform=ax_v.transAxes, fontsize=7,
                  bbox=dict(boxstyle="round,pad=0.2", fc="white",
                            ec="grey", lw=0.4))

        # ───── Intracellular block ────────────────────────────────────────────
        t_jax_i  = np.array(tr["t_intra_jax"])
        vm_jax_i = np.array(tr["vm_intra_jax"])
        t_nrn_i  = np.array(tr["t_intra_nrn"])
        vm_nrn_i = np.array(tr["vm_intra_nrn"])
        v_jax_i = vm_jax_i[:, vm_jax_i.shape[1] // 2] if vm_jax_i.ndim > 1 else vm_jax_i
        v_nrn_i = vm_nrn_i[:, vm_nrn_i.shape[1] // 2] if vm_nrn_i.ndim > 1 else vm_nrn_i

        ax_s2 = axes[2, col]
        pw_i = PW_I_MS[model]
        amp_i = AMP_I_NA[model]
        t_stim_i = np.linspace(0, t_jax_i[-1], 2000)
        stim_i = np.where((t_stim_i >= DELAY_MS) & (t_stim_i < DELAY_MS + pw_i),
                          amp_i, 0.0)
        ax_s2.fill_between(t_stim_i, 0, stim_i, color=clr, alpha=0.55, lw=0)
        ax_s2.axhline(0, color="black", lw=0.4, alpha=0.6)
        ax_s2.set_ylim(-amp_i * 0.12, amp_i * 1.15)
        if col == 0:
            ax_s2.set_ylabel("$I_i$ (nA)\nintracellular", fontsize=8)
        ax_s2.tick_params(axis="y", labelsize=7)
        ax_s2.text(0.97, 0.92,
                   f"amp = {amp_i:.1f} nA",
                   transform=ax_s2.transAxes, ha="right", va="top",
                   fontsize=6, family="DejaVu Sans Mono",
                   bbox=dict(boxstyle="round,pad=0.18", fc="white",
                             ec="grey", lw=0.4))

        ax_v2 = axes[3, col]
        ax_v2.plot(t_nrn_i, v_nrn_i, "k--", lw=1.4, alpha=0.85, label="NEURON")
        ax_v2.plot(t_jax_i, v_jax_i, "-",   color=clr, lw=1.4, label="jaxon")
        ax_v2.set_xlabel("time (ms)", fontsize=8)
        if col == 0:
            ax_v2.set_ylabel("$V_m$ at centre node (mV)", fontsize=8)
        ax_v2.tick_params(axis="both", labelsize=7)
        ax_v2.legend(loc="upper right", frameon=False, fontsize=7)
        peak_diff_i = float(
            np.max(np.abs(np.interp(t_jax_i, t_nrn_i, v_nrn_i) - v_jax_i))
        )
        ax_v2.text(0.04, 0.05, f"$\\Delta V_m^{{peak}}={peak_diff_i:.2f}$ mV",
                   transform=ax_v2.transAxes, fontsize=7,
                   bbox=dict(boxstyle="round,pad=0.2", fc="white",
                             ec="grey", lw=0.4))

    fig.suptitle(
        "Response to extracellular cathodic stimulation (rows 1–2) and "
        "intracellular current injection (rows 3–4): jaxon vs NEURON",
        fontsize=11, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIGDIR / "fig_traces_4models.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


# ─── 3. propagation phenomena: collision, kHz block, DC block ─────────────────
def fig_phenomena():
    """2×2 panel: AP collision, kHz block, DC block, spike desync.

    Short axes titles + (a)/(b)/(c)/(d) panel labels in the top-left
    corner of each axes — the full descriptive title for each panel
    lives in the figure caption.  Generous wspace / hspace so the
    panels never touch, regardless of how long the legends are.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.93,
                        wspace=0.28, hspace=0.38)
    PANEL_LBL = dict(fontsize=12, fontweight="bold")

    def _label_panel(ax, letter):
        ax.text(-0.10, 1.06, f"({letter})", transform=ax.transAxes,
                ha="left", va="bottom", **PANEL_LBL)

    # ── (a) AP collision: snapshot Vm along the fibre ────────────────────────
    ax = axes[0, 0]
    d = json.loads((OUTROOT / "ap_collision/data_ap_collision.json").read_text())
    r = d["results"][2]  # D=10 µm
    snap_t = d["snapshot_t_ms"]
    snaps_jax = np.array(r["snapshots_nodes_mV"])      # [n_snap, n_nodes]
    snaps_pf  = np.array(r["snapshots_nodes_pf_mV"])
    n_nodes   = r["n_nodes"]
    nodes     = np.arange(n_nodes)
    for i, t in enumerate(snap_t):
        ax.plot(nodes, snaps_pf[i], "k--", lw=0.9, alpha=0.5)
        ax.plot(nodes, snaps_jax[i], "-",
                color=plt.cm.viridis(i / len(snap_t)), lw=1.5,
                label=f"t = {t:.2f} ms")
    ax.set_xlabel("node index")
    ax.set_ylabel("$V_m$ (mV)")
    ax.set_title(f"AP collision  (MRG $D={r['diameter_um']:g}\\,\\mu$m)",
                 fontsize=10, pad=6)
    ax.legend(loc="upper right", frameon=False, fontsize=7, ncol=2,
              columnspacing=0.6, handlelength=1.2)
    ax.text(0.97, 0.04,
            f"peak $|\\Delta V_m|$ jax/NEURON = "
            f"{abs(r['vm_peak_mV']-r['vm_peak_pf_mV'])*1000:.2f} $\\mu$V",
            transform=ax.transAxes, ha="right", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey",
                      lw=0.4))
    _label_panel(ax, "a")

    # ── (b) kHz block: distal-node Vm trace, 4 amplitudes ────────────────────
    ax = axes[0, 1]
    d = json.loads((OUTROOT / "khz_block/data_khz_block.json").read_text())
    pace = d["pace"]
    for r in d["results"]:
        t   = np.array(r["t_pf_ms"])
        v_pf = np.array(r["vm_pf_90"])
        v_jx = np.array(r["vm_jax_90"])
        t_jx = np.linspace(t[0], t[-1], len(v_jx))
        ax.plot(t,    v_pf, "k--", lw=0.6, alpha=0.5)
        ax.plot(t_jx, v_jx, "-",   lw=1.0,
                label=f"$|A|$={abs(r['amp_mA']):.2f} mA  "
                      f"({r['n_aps_jax']}/{pace['n_pulses']} APs)")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("$V_m$ at distal node (mV)")
    ax.set_title(f"kHz block  ({d['khz_freq']:g} kHz, MRG "
                 f"$D={d['diameter_um']:g}\\,\\mu$m)",
                 fontsize=10, pad=6)
    ax.legend(loc="upper right", frameon=False, fontsize=7,
              handlelength=1.2)
    _label_panel(ax, "b")

    # ── (c) DC block: distal-node Vm trace, 4 amplitudes ─────────────────────
    ax = axes[1, 0]
    d = json.loads((OUTROOT / "dc_block/data_dc_block.json").read_text())
    for r in d["results"]:
        t_jx = np.array(r["jax_t_ms"]); v_jx = np.array(r["jax_vm_nodes"])
        t_pf = np.array(r["pf_t_ms"]);   v_pf = np.array(r["pf_vm_nodes"])
        v_jx_d = v_jx[:, -1] if v_jx.ndim > 1 else v_jx
        v_pf_d = v_pf[:, -1] if v_pf.ndim > 1 else v_pf
        ax.plot(t_pf, v_pf_d, "k--", lw=0.6, alpha=0.5)
        ax.plot(t_jx, v_jx_d, "-",   lw=1.2,
                label=f"$\\alpha$={r['amp_factor']:.2f}  "
                      f"(${r['amp_mA']*1000:.0f}\\,\\mu$A)")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("$V_m$ at distal node (mV)")
    ax.set_title(f"DC block  (MRG $D={d['diameter_um']:g}\\,\\mu$m)",
                 fontsize=10, pad=6)
    ax.legend(loc="upper right", frameon=False, fontsize=7,
              handlelength=1.2)
    _label_panel(ax, "c")

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
    ax.set_title(f"Spike desync.\\  (MRG $D={d['diameter_um']:g}\\,\\mu$m, "
                 f"{d['n_ref_spikes']} ref. spikes)",
                 fontsize=10, pad=6)
    ax.legend(loc="lower left", frameon=False, fontsize=7,
              handlelength=1.2)
    ax.set_ylim(-0.05, 1.1)
    _label_panel(ax, "d")

    fig.suptitle("Propagation phenomena reproduced under jaxon (solid) "
                 "vs NEURON (dashed)",
                 fontsize=12, fontweight="bold", y=0.985)
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


# ─── 5. selectivity: violin distribution across all completed Phase 3 seeds ───
def fig_selectivity_summary():
    """Cross-model selectivity violin (rect + wave SI per model), plus a
    rect-vs-wave per-seed scatter coloured by model.

    Reads every selectivity_sweep_phase3_*/ sub-directory that has at
    least one data_seed_*.json file (manuscript/outputs/ preferred, falls
    back to ../outputs/).  Each per-model directory becomes one group of
    violins on the x-axis.  When only the original MRG @ 5.7 µm sweep
    exists, the figure degrades gracefully to a single-model view.
    """
    # Per-model display labels + colours, in the canonical order they
    # should appear left-to-right on the figure.
    MODEL_DISPLAY = [
        ("selectivity_sweep_phase3_manuscript", "MRG\n5.7 µm",  "#1f77b4"),
        ("selectivity_sweep_phase3_mrg10um",    "MRG\n10 µm",   "#4a90c4"),
        ("selectivity_sweep_phase3_sweeney_10um", "Sweeney\n10 µm", "#2ca02c"),
        ("selectivity_sweep_phase3_sundt_1um",  "Sundt\n1 µm",  "#d62728"),
        ("selectivity_sweep_phase3_rattay_1um", "Rattay\n1 µm", "#9467bd"),
    ]

    # Collect all sweeps that exist on disk.
    models = []
    for subdir, label, colour in MODEL_DISPLAY:
        sweep_dir = _outdir(subdir)
        seeds = sorted(sweep_dir.glob("data_seed_*.json"))
        if not seeds:
            continue
        rows = []
        for p in seeds:
            d = json.loads(p.read_text())
            rows.append({
                "seed":     d["seed"],
                "baseline": d["si_baseline"],
                "rect":     d["rect"]["final_si"],
                "wave":     d["waveform"]["final_si"],
            })
        rect = np.array([r["rect"] for r in rows])
        wave = np.array([r["wave"] for r in rows])
        base = np.array([r["baseline"] for r in rows])
        models.append(dict(label=label, colour=colour, n=len(rows),
                           rect=rect, wave=wave, base=base))

    if not models:
        return
    n_models = len(models)

    fig, axes = plt.subplots(1, 2, figsize=(max(11, 2.0 * n_models + 4), 4.6))

    # ── Panel A: per-model paired violins (rect + wave side by side) ────────
    ax = axes[0]
    rng = np.random.default_rng(0)
    centres = np.arange(n_models) * 2.0   # one cluster per model
    half = 0.40                           # half-width of paired violins
    for i, m in enumerate(models):
        x_rect = centres[i] - half * 0.55
        x_wave = centres[i] + half * 0.55
        for x, vals, edge_alpha in [(x_rect, m["rect"], 1.0),
                                    (x_wave, m["wave"], 0.7)]:
            parts = ax.violinplot([vals], positions=[x], widths=half,
                                  showmeans=False, showmedians=False,
                                  showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(m["colour"])
                body.set_edgecolor("black")
                body.set_alpha(0.55 * edge_alpha)
                body.set_linewidth(0.6)
            jitter = rng.uniform(-half*0.18, half*0.18, size=len(vals))
            ax.scatter(x + jitter, vals, s=10, color="black", alpha=0.45,
                       edgecolors="none", zorder=3)
            ax.hlines(float(np.median(vals)), x - half*0.4, x + half*0.4,
                      color="red", lw=1.6, zorder=4)
    ax.axhline(0.95, color="red", lw=0.8, ls="--", alpha=0.55,
               label="acceptance criterion (SI = 0.95)")
    ax.set_xticks(centres)
    ax.set_xticklabels(
        [f"{m['label']}\n(n={m['n']})" for m in models],
        fontsize=8,
    )
    # Sub-tick annotation for rect / wave
    for cx in centres:
        ax.text(cx - half*0.55, -0.115, "rect", ha="center", va="top",
                fontsize=7, color="grey", transform=ax.get_xaxis_transform())
        ax.text(cx + half*0.55, -0.115, "wave", ha="center", va="top",
                fontsize=7, color="grey", transform=ax.get_xaxis_transform())
    ax.set_ylabel("selectivity index")
    ax.set_ylim(-0.05, 1.10)
    ax.set_title("Per-model selectivity distribution  "
                 "(rect LBFGS-FD,  warm-started wave Adam)",
                 fontsize=10)
    ax.legend(loc="lower right", fontsize=7, frameon=False)

    # ── Panel B: rect-vs-wave scatter coloured by model ────────────────────
    ax2 = axes[1]
    ax2.plot([0, 1.05], [0, 1.05], "k:", lw=0.8, alpha=0.5,
             label="wave = rect (no change)")
    for m in models:
        ax2.scatter(m["rect"], m["wave"], s=26, c=m["colour"],
                    edgecolors="black", lw=0.4, alpha=0.85, zorder=3,
                    label=f"{m['label'].replace(chr(10), ' ')}  "
                          f"(n={m['n']})")
    ax2.set_xlabel("rect SI")
    ax2.set_ylabel("warm-started wave SI")
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.10)
    # Aggregate improved / preserved / regressed across all models for the
    # title (a single global summary is more readable than per-model labels).
    all_rect = np.concatenate([m["rect"] for m in models])
    all_wave = np.concatenate([m["wave"] for m in models])
    n_improved  = int(np.sum(all_wave > all_rect + 0.005))
    n_preserved = int(np.sum(np.abs(all_wave - all_rect) <= 0.005))
    n_regressed = int(np.sum(all_wave < all_rect - 0.005))
    ax2.set_title(
        f"wave vs rect per seed (all models)   "
        f"improved: {n_improved}, preserved: {n_preserved}, "
        f"regressed: {n_regressed}",
        fontsize=9,
    )
    ax2.legend(loc="upper left", fontsize=7, frameon=False)
    ax2.set_aspect("equal", adjustable="box")

    n_total = sum(m["n"] for m in models)
    fig.suptitle(
        f"Selectivity optimisation across {n_total} randomised "
        f"Hussain-style nerve realisations spanning {n_models} fibre "
        "model" + ("s" if n_models > 1 else ""),
        fontsize=11, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIGDIR / "fig_selectivity_summary.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


# ─── 6. cross-section activation gallery ──────────────────────────────────────
def fig_selectivity_xsections():
    """Gallery of per-seed cross-section activation plots.

    For each per-model sweep directory, picks the first ``n_seeds`` of
    its completed seeds and arranges the corresponding
    ``fig_seed_XXXX_rect_xsection.png`` (pre-rendered by the sweep
    pipeline) into a (n_models × n_seeds) grid.  Different seeds use
    different random divider angles, so the gallery shows how the
    optimised cuff adapts to different on/off-target fascicle layouts.
    """
    import matplotlib.image as mpimg

    # Reuse the same model-display ordering as fig_selectivity_summary.
    MODEL_DISPLAY = [
        ("selectivity_sweep_phase3_manuscript",   "MRG  5.7 µm"),
        ("selectivity_sweep_phase3_mrg10um",      "MRG  10 µm"),
        ("selectivity_sweep_phase3_sweeney_10um", "Sweeney  10 µm"),
        ("selectivity_sweep_phase3_sundt_1um",    "Sundt  1 µm"),
        ("selectivity_sweep_phase3_rattay_1um",   "Rattay  1 µm"),
    ]
    n_seeds_per_model = 4   # how many cross-sections per row

    # Collect (label, [(seed_num, png_path, json_path), ...]) per model
    # that has at least one seed.
    rows = []
    for subdir, label in MODEL_DISPLAY:
        sweep_dir = _outdir(subdir)
        seed_jsons = sorted(sweep_dir.glob("data_seed_*.json"))
        if not seed_jsons:
            continue
        entries = []
        for j in seed_jsons[:n_seeds_per_model]:
            # Filename pattern: data_seed_NNNN.json → seed number NNNN
            seed_num = int(j.stem.split("_")[-1])
            png = sweep_dir / f"fig_seed_{seed_num:04d}_rect_xsection.png"
            if png.exists():
                entries.append((seed_num, png, j))
        if entries:
            rows.append((label, entries))

    if not rows:
        return

    n_rows = len(rows)
    n_cols = max(len(r[1]) for r in rows)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.6 * n_cols, 2.6 * n_rows),
        squeeze=False,
    )
    for r, (label, entries) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c >= len(entries):
                continue
            seed_num, png, json_path = entries[c]
            ax.imshow(mpimg.imread(png))
            # Per-panel SI annotation from the JSON.
            try:
                d = json.loads(json_path.read_text())
                rect_si = float(d["rect"]["final_si"])
                ax.set_title(f"seed {seed_num}  SI={rect_si:+.2f}",
                             fontsize=8, pad=2)
            except Exception:
                ax.set_title(f"seed {seed_num}", fontsize=8, pad=2)
        # Row label on the left
        axes[r, 0].set_ylabel(label, fontsize=10, fontweight="bold",
                              rotation=90, labelpad=10)

    fig.suptitle("Cross-section activation pattern at the rect-LBFGS optimum, "
                 "per model × seed   (random divider angle per seed)",
                 fontsize=11, fontweight="bold", y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIGDIR / "fig_selectivity_xsections.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


# ─── orchestration ────────────────────────────────────────────────────────────
ALL = {
    "validation":   fig_validation_4models,
    "traces":       fig_traces_4models,
    "phenomena":    fig_phenomena,
    "scaling":      fig_scaling_allmodels,
    "selectivity":  fig_selectivity_summary,
    "xsections":    fig_selectivity_xsections,
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
