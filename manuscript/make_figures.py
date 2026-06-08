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
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
# All sweep data lives in the repo-level outputs/ at the project root.
OUTROOT = HERE.parent / "outputs"
def _outdir(subdir: str) -> Path:
    """Resolve ``outputs/<subdir>/`` at the project root."""
    return OUTROOT / subdir

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
    "axes.formatter.useoffset": False,
    "font.size": 13,
    "axes.titlesize": 13,
    "axes.labelsize": 14,
    "axes.labelweight": "normal",
    "legend.fontsize": 11,
    "legend.frameon": False,
    "legend.borderpad": 0.3,
    "legend.handletextpad": 0.5,
    "legend.handlelength": 1.6,
    "legend.labelspacing": 0.35,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 2.0,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ─── 0. overview: pipeline + cuff + cable circuits (conceptual Fig 1) ────────
def fig_overview():
    """Conceptual overview figure for the manuscript front matter.

    Four panels:
      (a) Nerve cross-section + 6-contact cuff schematic
          (illustrative: not from sweep data).
      (b) Forward pipeline: boxes + arrows  V_e --> cable solver
          --> activation proxy --> selectivity loss.
      (c) Single-cable equivalent circuit (Sundt/Sweeney/Rattay axons):
          one membrane R||C compartment with axial coupling to its
          two neighbours, plus the bath-V_e contact.
      (d) Double-cable equivalent circuit (MRG axon): nodal membrane
          R||C, periaxonal layer above it (between axon and myelin),
          myelin C||G, and bath-V_e contact, with axial coupling on
          both the intracellular and periaxonal layers.
    """
    from matplotlib.patches import (
        Circle as _Circle, Wedge as _Wedge, FancyBboxPatch,
        FancyArrowPatch, Rectangle as _Rectangle,
    )

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.0, 1.4], height_ratios=[1.0, 1.0],
        hspace=0.32, wspace=0.20,
        left=0.04, right=0.98, top=0.96, bottom=0.04,
    )

    # ── Panel (a): cuff + nerve cross-section ────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    R_NERVE = 500.0      # µm
    R_CUFF  = 1500.0     # µm
    # Nerve outline
    ax.add_patch(_Circle((0, 0), R_NERVE, fc="0.95", ec="0.5",
                         lw=1.4, ls="--"))
    # Fascicles: 4 target (upper half), 4 off-target (lower half) — illustrative.
    rng = np.random.default_rng(7)
    angles_t = np.linspace(15, 165, 4)
    angles_o = np.linspace(195, 345, 4)
    for a_deg in angles_t:
        x = (R_NERVE * 0.55) * np.cos(np.deg2rad(a_deg))
        y = (R_NERVE * 0.55) * np.sin(np.deg2rad(a_deg))
        ax.add_patch(_Circle((x, y), 90, fc="#9bd09b", ec="#2ca02c",
                             lw=1.0, alpha=0.9))
    for a_deg in angles_o:
        x = (R_NERVE * 0.55) * np.cos(np.deg2rad(a_deg))
        y = (R_NERVE * 0.55) * np.sin(np.deg2rad(a_deg))
        ax.add_patch(_Circle((x, y), 90, fc="0.82", ec="0.55",
                             lw=1.0, alpha=0.9))
    # Divider line
    ax.plot([-R_NERVE * 1.05, R_NERVE * 1.05], [0, 0],
            color="black", lw=2.4, zorder=4)
    # 6 cuff contacts (ring at 1.18 R for visual proximity)
    angles_c = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    contact_r = R_NERVE * 1.20
    polarities = [-1, +1, +1, -1, +1, +1]   # illustrative bipolar
    for i, ang in enumerate(angles_c):
        cx = contact_r * np.cos(ang)
        cy = contact_r * np.sin(ang)
        clr = "#1f77b4" if polarities[i] < 0 else "#ff7f0e"
        ax.add_patch(_Circle((cx, cy), 70, fc=clr, ec="black", lw=0.8,
                             zorder=6))
        ax.text(cx * 1.28, cy * 1.28, f"C{i}",
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=clr)
    # Target / off-target labels
    ax.text(0, R_NERVE * 1.45, "target", ha="center", va="center",
            fontsize=12, color="#2ca02c", style="italic", fontweight="bold")
    ax.text(0, -R_NERVE * 1.45, "off-target", ha="center", va="center",
            fontsize=12, color="0.45", style="italic")
    ax.set_xlim(-R_NERVE * 1.7, R_NERVE * 1.7)
    ax.set_ylim(-R_NERVE * 1.7, R_NERVE * 1.7)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(-0.06, 1.00, "(a)", transform=ax.transAxes,
            ha="left", va="top", fontsize=16, fontweight="bold")

    # ── Panel (b): forward pipeline (boxes + arrows) ─────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    boxes = [
        # (x_centre, y_centre, w, h, label, sub-label, facecolor)
        (1.4, 8.0, 2.0, 1.4, "Cuff geometry",  r"$K$ contacts on a ring", "#cfe2f3"),
        (4.8, 8.0, 2.0, 1.4, r"$V_e$ field",   r"per-contact: $V_e^{(k)}(\mathbf{x})$", "#cfe2f3"),
        (8.2, 8.0, 2.0, 1.4, "Per-contact $\\mathbf{I}\\in\\mathbb{R}^{K}$", "design variable", "#fde9b3"),
        (3.1, 5.0, 2.0, 1.4, r"Coupled cable solver",  r"backward-Euler", "#d5e8d4"),
        (6.5, 5.0, 2.0, 1.4, "Activation proxy", r"$a_f \in [0,1]$ per fibre", "#d5e8d4"),
        (4.8, 2.0, 4.0, 1.4, "Selectivity loss $\\mathcal{L}(\\mathbf{I})$",
         "gradient via " + r"$\partial \mathcal{L} / \partial \mathbf{I}$", "#f4cccc"),
    ]
    for cx, cy, w, h, lab, sub, fc in boxes:
        ax.add_patch(FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            fc=fc, ec="black", lw=1.0,
        ))
        ax.text(cx, cy + 0.20, lab, ha="center", va="center",
                fontsize=11, fontweight="bold")
        ax.text(cx, cy - 0.34, sub, ha="center", va="center",
                fontsize=9, style="italic", color="0.3")

    def _arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>", mutation_scale=18,
            lw=1.4, color="0.25",
        ))

    _arrow(2.4, 8.0, 3.8, 8.0)   # cuff → Ve
    _arrow(5.8, 8.0, 7.2, 8.0)   # Ve → I
    _arrow(4.8, 7.3, 4.0, 5.7)   # Ve → solver
    _arrow(8.2, 7.3, 7.3, 5.7)   # I → solver / activation pipeline
    _arrow(4.1, 5.0, 5.5, 5.0)   # solver → activation
    _arrow(6.5, 4.3, 5.4, 2.7)   # activation → loss
    # Gradient back-arrow (downstream loss → upstream design var)
    ax.add_patch(FancyArrowPatch(
        (4.8, 2.7), (8.2, 7.3),
        arrowstyle="-|>", mutation_scale=18, lw=1.4, color="#d62728",
        connectionstyle="arc3,rad=-0.35",
    ))
    ax.text(8.8, 4.0, "autodiff\n$\\nabla_{\\mathbf{I}}\\mathcal{L}$",
            ha="left", va="center", fontsize=10,
            color="#d62728", fontweight="bold")
    ax.text(-0.04, 1.00, "(b)", transform=ax.transAxes,
            ha="left", va="top", fontsize=16, fontweight="bold")

    # ── Panel (c): single-cable equivalent circuit ───────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    _draw_single_cable_circuit(ax)
    ax.text(-0.06, 1.00, "(c)", transform=ax.transAxes,
            ha="left", va="top", fontsize=16, fontweight="bold")

    # ── Panel (d): double-cable equivalent circuit (MRG) ─────────────────────
    ax = fig.add_subplot(gs[1, 1])
    _draw_double_cable_circuit(ax)
    ax.text(-0.04, 1.00, "(d)", transform=ax.transAxes,
            ha="left", va="top", fontsize=16, fontweight="bold")

    out = FIGDIR / "fig_overview.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


def _draw_single_cable_circuit(ax):
    """Schematic single-cable: three compartments connected by R_axial.
    Each compartment has membrane R||C between V_i and V_e (bath)."""
    from matplotlib.patches import Rectangle as _R, FancyArrowPatch
    ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # Top rail: bath / extracellular V_e
    ax.plot([0.5, 9.5], [6.0, 6.0], color="0.4", lw=1.4, ls="--")
    ax.text(0.3, 6.0, r"$V_e$", ha="right", va="center", fontsize=12,
            color="0.4")
    # Bottom rail: intracellular V_i (cable)
    ax.plot([0.5, 9.5], [2.0, 2.0], color="black", lw=1.6)
    ax.text(0.3, 2.0, r"$V_i$", ha="right", va="center", fontsize=12)

    # Three compartments at x = 2.5, 5.0, 7.5
    xs = [2.5, 5.0, 7.5]
    for i, x in enumerate(xs):
        # Resistor (membrane R) — rectangle
        ax.add_patch(_R((x - 0.55, 3.6), 0.50, 1.1, fc="white",
                       ec="black", lw=1.1))
        ax.text(x - 0.30, 4.15, r"$R_m$", ha="center", va="center",
                fontsize=11)
        # Capacitor (membrane C) — two parallel lines
        ax.plot([x + 0.20, x + 0.50], [3.85, 3.85], color="black", lw=1.6)
        ax.plot([x + 0.20, x + 0.50], [4.45, 4.45], color="black", lw=1.6)
        ax.plot([x + 0.35, x + 0.35], [4.45, 6.0], color="black", lw=1.2)
        ax.plot([x + 0.35, x + 0.35], [2.0, 3.85], color="black", lw=1.2)
        ax.text(x + 0.80, 4.15, r"$C_m$", ha="center", va="center",
                fontsize=11)
        # Wires from R top/bottom to rails
        ax.plot([x - 0.30, x - 0.30], [4.70, 6.0], color="black", lw=1.2)
        ax.plot([x - 0.30, x - 0.30], [2.0, 3.60], color="black", lw=1.2)

        # Compartment label below
        ax.text(x + 0.05, 1.55,
                f"node {i+1}", ha="center", va="center",
                fontsize=9, color="0.3", style="italic")

    # Axial resistors between compartments (on V_i rail)
    for x0, x1 in [(xs[0], xs[1]), (xs[1], xs[2])]:
        midx = 0.5 * (x0 + x1)
        ax.add_patch(_R((midx - 0.30, 1.65), 0.60, 0.7, fc="white",
                       ec="black", lw=1.1))
        ax.text(midx, 1.40, r"$R_a$", ha="center", va="center",
                fontsize=10)

    # Ionic currents annotation (just on the middle compartment)
    ax.annotate(r"$I_{\mathrm{ion}}$  (Na, K, leak)",
                xy=(5.0 + 0.04, 4.15),
                xytext=(5.6, 5.30), fontsize=10,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))

    # Caption-like header below axes (NOT a title — just a tag)
    ax.text(5.0, 6.5,
            "Single-cable (Sundt, Sweeney, Rattay)",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
            color="0.15")


def _draw_double_cable_circuit(ax):
    """Schematic double-cable: nodal compartments with separate
    intracellular V_i, periaxonal V_px and bath V_e.  Axial coupling
    on both V_i and V_px rails."""
    from matplotlib.patches import Rectangle as _R
    ax.set_xlim(0, 12); ax.set_ylim(0, 7)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # Top rail: bath / extracellular V_e
    ax.plot([0.5, 11.5], [6.4, 6.4], color="0.4", lw=1.4, ls="--")
    ax.text(0.3, 6.4, r"$V_e$", ha="right", va="center", fontsize=12,
            color="0.4")
    # Middle rail: periaxonal V_px
    ax.plot([0.5, 11.5], [4.4, 4.4], color="#8e6ec7", lw=1.6)
    ax.text(0.3, 4.4, r"$V_{\mathrm{px}}$", ha="right", va="center",
            fontsize=12, color="#8e6ec7")
    # Bottom rail: intracellular V_i
    ax.plot([0.5, 11.5], [1.7, 1.7], color="black", lw=1.6)
    ax.text(0.3, 1.7, r"$V_i$", ha="right", va="center", fontsize=12)

    # Two periods: node + paranode + internode + paranode  (simplified)
    period_centres = [3.0, 9.0]
    labels = ["node",  "internode"]
    for ci, (cx, lab) in enumerate(zip(period_centres, labels)):
        # ── Membrane (V_i ↔ V_px) ─ R||C ─────────────────────────────────
        # R_m
        ax.add_patch(_R((cx - 0.60, 2.6), 0.55, 1.1, fc="white",
                       ec="black", lw=1.1))
        ax.text(cx - 0.33, 3.15, r"$R_m$", ha="center", va="center",
                fontsize=11)
        # C_m
        ax.plot([cx + 0.20, cx + 0.50], [2.85, 2.85], color="black", lw=1.6)
        ax.plot([cx + 0.20, cx + 0.50], [3.45, 3.45], color="black", lw=1.6)
        ax.plot([cx + 0.35, cx + 0.35], [3.45, 4.40], color="black", lw=1.2)
        ax.plot([cx + 0.35, cx + 0.35], [1.70, 2.85], color="black", lw=1.2)
        ax.text(cx + 0.80, 3.15, r"$C_m$", ha="center", va="center",
                fontsize=11)
        # wires R↔rails
        ax.plot([cx - 0.33, cx - 0.33], [3.70, 4.40], color="black", lw=1.2)
        ax.plot([cx - 0.33, cx - 0.33], [1.70, 2.60], color="black", lw=1.2)

        # ── Myelin (V_px ↔ V_e) ─ G||C ───────────────────────────────────
        # G_m (myelin shunt)
        ax.add_patch(_R((cx - 0.60, 5.05), 0.55, 1.0, fc="white",
                       ec="#8e6ec7", lw=1.1))
        ax.text(cx - 0.33, 5.55, r"$G_{\mathrm{my}}$", ha="center",
                va="center", fontsize=10, color="#8e6ec7")
        # C_my (myelin capacitance)
        ax.plot([cx + 0.20, cx + 0.50], [5.30, 5.30], color="#8e6ec7", lw=1.6)
        ax.plot([cx + 0.20, cx + 0.50], [5.85, 5.85], color="#8e6ec7", lw=1.6)
        ax.plot([cx + 0.35, cx + 0.35], [5.85, 6.40], color="#8e6ec7", lw=1.2)
        ax.plot([cx + 0.35, cx + 0.35], [4.40, 5.30], color="#8e6ec7", lw=1.2)
        ax.text(cx + 0.85, 5.55, r"$C_{\mathrm{my}}$", ha="center",
                va="center", fontsize=10, color="#8e6ec7")
        # wires G↔rails
        ax.plot([cx - 0.33, cx - 0.33], [6.05, 6.40], color="#8e6ec7", lw=1.2)
        ax.plot([cx - 0.33, cx - 0.33], [4.40, 5.05], color="#8e6ec7", lw=1.2)

        # Compartment label
        ax.text(cx + 0.05, 0.9, lab, ha="center", va="center",
                fontsize=10, color="0.3", style="italic", fontweight="bold")

    # Axial resistors on V_i and V_px (between the two compartments)
    midx = 0.5 * (period_centres[0] + period_centres[1])
    # R_a (intracellular axial)
    ax.add_patch(_R((midx - 0.30, 1.35), 0.60, 0.7, fc="white",
                   ec="black", lw=1.1))
    ax.text(midx, 1.10, r"$R_a$", ha="center", va="center", fontsize=10)
    # R_px (periaxonal axial)
    ax.add_patch(_R((midx - 0.30, 4.05), 0.60, 0.7, fc="white",
                   ec="#8e6ec7", lw=1.1))
    ax.text(midx, 3.80, r"$R_{\mathrm{px}}$", ha="center", va="center",
            fontsize=10, color="#8e6ec7")

    # Annotation: nodal ionic currents (left compartment)
    ax.annotate(r"$I_{\mathrm{Naf}} + I_{\mathrm{Nap}}$"
                "\n" + r"$+ I_{\mathrm{Ks}} + I_L$",
                xy=(3.0 - 0.05, 3.15),
                xytext=(0.7, 0.10), fontsize=10,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))

    # Header
    ax.text(6.0, 6.65,
            "Double-cable (MRG)",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
            color="0.15")


# ─── 1. validation: all 4 models, SD curves + CV + error histogram ───────────
def fig_validation_4models():
    """3-row × 4-column panel: SD curves, CV vs diameter, error histogram."""
    fig, axes = plt.subplots(3, 4, figsize=(13, 9.5))
    # Single shared legend across the whole figure for the jaxon-vs-NEURON
    # series; placed under the figure so it never overlaps the data.
    from matplotlib.lines import Line2D
    shared_handles = [
        Line2D([0], [0], color="0.3", lw=2.0, label="jaxon"),
        Line2D([0], [0], marker="o", color="none", mec="k", mew=1.0,
               ms=7, ls="", label="pyfibers (NEURON)"),
        Line2D([0], [0], color="k", lw=1.4, ls="--", label="median"),
        Line2D([0], [0], color="red", lw=1.2, ls=":", label="1\\% tol"),
    ]
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
        ax.plot(pw_arr, np.abs(jax), "-", color=clr, lw=2.0, zorder=2)
        ax.plot(pw_arr, np.abs(pyf), "o", mfc="none", mec="k", mew=1.2, ms=7,
                zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("pulse width (ms)")
        if col == 0:
            ax.set_ylabel("|threshold| (mA)")
        # Column header: model + diameter ABOVE the axes (not over data).
        ax.set_title(f"{MODEL_LABEL[model]}  $D={Dnom}\\,\\mu$m",
                     color=clr, fontsize=14, fontweight="bold", pad=6)

        # Row 1: CV vs diameter
        ax = axes[1, col]
        D_cv = sorted(d_cv.keys(), key=float)
        D_arr = np.array([float(D) for D in D_cv])
        jax_cv = np.array([d_cv[D]["jax"] for D in D_cv])
        pyf_cv = np.array([d_cv[D]["pyfibers"] for D in D_cv])
        valid = np.isfinite(jax_cv) | np.isfinite(pyf_cv)
        if valid.any():
            ax.plot(D_arr, jax_cv, "-o", color=clr, lw=2.0, mfc=clr, mec=clr,
                    ms=7)
            ax.plot(D_arr, pyf_cv, "s", mfc="none", mec="k", mew=1.2, ms=8)
            ax.set_xlabel("diameter ($\\mu$m)")
            if col == 0:
                ax.set_ylabel("CV (m/s)")
            # Single-point sweeps (Sweeney) otherwise get a degenerate y-axis
            # with absurd tick precision.  Centre on the value with a wide
            # padding so the single point sits in the middle of a readable
            # range.
            if len(D_arr) == 1 and np.isfinite(jax_cv[0]):
                v = float(jax_cv[0])
                pad = max(abs(v) * 0.10, 1.0)
                ax.set_ylim(v - pad, v + pad)
                ax.set_xlim(D_arr[0] - 1.0, D_arr[0] + 1.0)
        else:
            ax.text(0.5, 0.5,
                    "CV measurement\nunavailable\n(re-run validation)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, style="italic", color="grey",
                    bbox=dict(boxstyle="round,pad=0.6", fc="white",
                              ec="lightgrey", lw=0.6))
            ax.set_xticks([]); ax.set_yticks([])
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
        ax.axvline(np.median(errs), color="k", lw=1.4, ls="--")
        ax.axvline(1.0, color="red", lw=1.2, ls=":")
        ax.set_xlabel("|threshold error| (%)")
        if col == 0:
            ax.set_ylabel("count")
        # Single compact stats block (replaces the in-axes legend).
        ax.text(0.97, 0.96,
                f"$n={len(errs)}$\nmed.\\ {np.median(errs):.2f}\\%\n"
                f"peak {errs.max():.2f}\\%",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=11, family="DejaVu Sans Mono",
                bbox=dict(boxstyle="round,pad=0.30", fc="white",
                          ec="0.7", lw=0.6))
        ax.set_xlim(0, max(errs.max() * 1.05, 1.05))

    # Figure-level legend at the bottom: jaxon (solid line) vs pyfibers (open circles).
    fig.legend(handles=shared_handles, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.01),
               fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out = FIGDIR / "fig_validation_4models.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


# ─── 1b. per-waveform validation breakdown (2×8 grid) ─────────────────────────
def fig_validation_waveforms():
    """2-row × 8-column per-waveform validation figure.

    Top row: strength--duration curves at MRG $D=10\\,\\mu$m for each
    of the eight stimulus waveforms (monophasic cathodic / anodic,
    biphasic cathodic-first / anodic-first, sinusoidal, sawtooth,
    exponentially decaying, Gaussian).  jaxon as solid coloured line,
    pyfibers (NEURON) as open circles.

    Bottom row: histogram of |relative threshold error| for the same
    waveform but aggregated across all four fiber models and their
    diameter ranges (~120 configurations per histogram).  Median + 1\\%
    tolerance verticals are drawn for reference.
    """
    WF_ORDER = [
        ("mono_c",  "Mono.\\\ncathodic"),
        ("mono_a",  "Mono.\\\nanodic"),
        ("bi_ca",   "Biph.\\\ncath-first"),
        ("bi_ac",   "Biph.\\\nanod-first"),
        ("sine",    "Sinusoidal"),
        ("sawtooth","Sawtooth"),
        ("exp",     "Exp.\\\ndecay"),
        ("gaussian","Gaussian"),
    ]
    # SD-curve reference cell: MRG @ 10 µm.
    REF_MODEL = "mrg"
    REF_D     = "10.0"
    ref_clr   = COLORS[REF_MODEL]

    # Pre-load all four model SD JSONs once.
    sd_all = {m: json.loads((OUTROOT / f"{m}_validation/data_{m}_sd.json").read_text())
              for m in MODELS}

    fig, axes = plt.subplots(2, len(WF_ORDER), figsize=(20, 7.5),
                             gridspec_kw={"hspace": 0.40, "wspace": 0.30})

    for col, (wf, lab) in enumerate(WF_ORDER):
        # ── Row 0: SD curves at MRG 10 µm ────────────────────────────────────
        ax = axes[0, col]
        ref_sd = sd_all[REF_MODEL][REF_D].get(wf)
        if ref_sd is None:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
        else:
            pws = sorted(ref_sd.keys(), key=float)
            pw_arr = np.array([float(p) for p in pws])
            jax = np.array([ref_sd[p]["jax"] for p in pws])
            pyf = np.array([ref_sd[p]["pyfibers"] for p in pws])
            ax.plot(pw_arr, np.abs(jax), "-", color=ref_clr, lw=2.0,
                    zorder=2)
            ax.plot(pw_arr, np.abs(pyf), "o", mfc="none", mec="k",
                    mew=1.2, ms=7, zorder=3)
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(lab, fontsize=12, fontweight="bold", color=ref_clr,
                     pad=4)
        if col == 0:
            ax.set_ylabel("|threshold|\n(mA)", fontsize=12)
        ax.set_xlabel("PW (ms)", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)

        # ── Row 1: error histogram aggregated across all models ──────────────
        ax = axes[1, col]
        errs = []
        for m in MODELS:
            d = sd_all[m]
            for D in d:
                if wf not in d[D]:
                    continue
                for pw, cell in d[D][wf].items():
                    e = cell.get("err_pct")
                    if e is not None and np.isfinite(e):
                        errs.append(abs(e))
        errs = np.array(errs)
        if len(errs) == 0:
            ax.set_xticks([]); ax.set_yticks([])
            continue
        # Shared x-axis range (0 - 2.5 %) so panels are directly
        # visually comparable; only biphasic cathodic-first exceeds 1%.
        ax.hist(errs, bins=np.linspace(0, 2.5, 26),
                color="0.4", alpha=0.85, edgecolor="black", lw=0.4)
        ax.axvline(np.median(errs), color="k", lw=1.4, ls="--")
        ax.axvline(1.0, color="red", lw=1.2, ls=":")
        # Stats annotation: n, median, peak.
        ax.text(0.97, 0.96,
                f"$n={len(errs)}$\nmed.\\ {np.median(errs):.2f}\\%\n"
                f"peak {errs.max():.2f}\\%",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, family="DejaVu Sans Mono",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="0.7", lw=0.5))
        ax.set_xlim(0, 2.5)
        ax.set_xlabel("|err.| (%)", fontsize=11)
        if col == 0:
            ax.set_ylabel("count", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)

    # Shared figure-level legend (top row) + line key (bottom row).
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="0.3", lw=2.0, label="jaxon (MRG $D=10\\,\\mu$m)"),
        Line2D([0], [0], marker="o", color="none", mec="k", mew=1.2,
               ms=7, ls="", label="pyfibers (NEURON)"),
        Line2D([0], [0], color="k", lw=1.4, ls="--", label="median err."),
        Line2D([0], [0], color="red", lw=1.2, ls=":", label="1\\% tol"),
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.01), fontsize=11)
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    out = FIGDIR / "fig_validation_waveforms.png"
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
        4, 4, figsize=(14, 9.5),
        gridspec_kw={"height_ratios": [1, 3, 1, 3],
                     "hspace": 0.22, "wspace": 0.30},
        sharex="col",
    )
    from matplotlib.lines import Line2D
    shared_handles = [
        Line2D([0], [0], color="0.3", lw=2.0, label="jaxon"),
        Line2D([0], [0], color="k", lw=2.0, ls="--", label="NEURON"),
    ]

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
        # Column header (model + diameter) as a proper axes title.
        ax_s.set_title(f"{MODEL_LABEL[model]}  $D={D:g}\\,\\mu$m",
                       color=clr, fontsize=14, fontweight="bold", pad=8)
        ymin = min(amp_e, thr_e) * 1.15
        ax_s.set_ylim(ymin, abs(ymin) * 0.12)
        if col == 0:
            ax_s.set_ylabel("$I_e$ (mA)", fontsize=13)
        ax_s.text(0.97, 0.10,
                  f"amp = {amp_e:.2f} mA\nthr = {thr_e:.2f} mA",
                  transform=ax_s.transAxes, ha="right", va="bottom",
                  fontsize=9, family="DejaVu Sans Mono",
                  bbox=dict(boxstyle="round,pad=0.22", fc="white",
                            ec="grey", lw=0.5))

        ax_v = axes[1, col]
        ax_v.plot(t_nrn_e, v_nrn_e, "k--", lw=1.6, alpha=0.85)
        ax_v.plot(t_jax_e, v_jax_e, "-",   color=clr, lw=1.6)
        if col == 0:
            ax_v.set_ylabel("$V_m$ (mV)", fontsize=13)
        peak_diff_e = float(
            np.max(np.abs(np.interp(t_jax_e, t_nrn_e, v_nrn_e) - v_jax_e))
        )
        ax_v.text(0.04, 0.05, f"$\\Delta V_m^{{peak}}={peak_diff_e:.2f}$ mV",
                  transform=ax_v.transAxes, fontsize=9,
                  bbox=dict(boxstyle="round,pad=0.22", fc="white",
                            ec="grey", lw=0.5))

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
            ax_s2.set_ylabel("$I_i$ (nA)", fontsize=13)
        ax_s2.text(0.97, 0.92,
                   f"amp = {amp_i:.1f} nA",
                   transform=ax_s2.transAxes, ha="right", va="top",
                   fontsize=9, family="DejaVu Sans Mono",
                   bbox=dict(boxstyle="round,pad=0.22", fc="white",
                             ec="grey", lw=0.5))

        ax_v2 = axes[3, col]
        ax_v2.plot(t_nrn_i, v_nrn_i, "k--", lw=1.6, alpha=0.85)
        ax_v2.plot(t_jax_i, v_jax_i, "-",   color=clr, lw=1.6)
        ax_v2.set_xlabel("time (ms)", fontsize=13)
        if col == 0:
            ax_v2.set_ylabel("$V_m$ (mV)", fontsize=13)
        peak_diff_i = float(
            np.max(np.abs(np.interp(t_jax_i, t_nrn_i, v_nrn_i) - v_jax_i))
        )
        ax_v2.text(0.04, 0.05, f"$\\Delta V_m^{{peak}}={peak_diff_i:.2f}$ mV",
                   transform=ax_v2.transAxes, fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.22", fc="white",
                             ec="grey", lw=0.5))

    fig.legend(handles=shared_handles, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.01), fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
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
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.96,
                        wspace=0.28, hspace=0.30)
    PANEL_LBL = dict(fontsize=15, fontweight="bold")

    def _label_panel(ax, letter):
        ax.text(-0.13, 1.04, f"({letter})", transform=ax.transAxes,
                ha="left", va="bottom", **PANEL_LBL)

    # ── (a) AP collision: snapshot Vm along the fibre ────────────────────────
    ax = axes[0, 0]
    d = json.loads((OUTROOT / "ap_collision/data_ap_collision.json").read_text())
    r = d["results"][2]  # D=10 µm
    snap_t = d["snapshot_t_ms"]
    snaps_jax = np.array(r["snapshots_nodes_mV"])
    snaps_pf  = np.array(r["snapshots_nodes_pf_mV"])
    n_nodes   = r["n_nodes"]
    nodes     = np.arange(n_nodes)
    for i, t in enumerate(snap_t):
        ax.plot(nodes, snaps_pf[i], "k--", lw=1.0, alpha=0.5)
        ax.plot(nodes, snaps_jax[i], "-",
                color=plt.cm.viridis(i / len(snap_t)), lw=2.0,
                label=f"t = {t:.2f} ms")
    ax.set_xlabel("node index")
    ax.set_ylabel("$V_m$ (mV)")
    # Compact legend just outside upper-right corner of axes.
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=10, ncol=1, handlelength=1.4, borderaxespad=0.0)
    ax.text(0.97, 0.04,
            f"peak $|\\Delta V_m|$ jax/NEURON = "
            f"{abs(r['vm_peak_mV']-r['vm_peak_pf_mV'])*1000:.2f} $\\mu$V",
            transform=ax.transAxes, ha="right", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey",
                      lw=0.5))
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
        ax.plot(t,    v_pf, "k--", lw=0.7, alpha=0.5)
        ax.plot(t_jx, v_jx, "-",   lw=1.3,
                label=f"$|A|$={abs(r['amp_mA']):.2f} mA  "
                      f"({r['n_aps_jax']}/{pace['n_pulses']} APs)")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("$V_m$ at distal node (mV)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=10, handlelength=1.4, borderaxespad=0.0)
    _label_panel(ax, "b")

    # ── (c) DC block: distal-node Vm trace, 4 amplitudes ─────────────────────
    ax = axes[1, 0]
    d = json.loads((OUTROOT / "dc_block/data_dc_block.json").read_text())
    for r in d["results"]:
        t_jx = np.array(r["jax_t_ms"]); v_jx = np.array(r["jax_vm_nodes"])
        t_pf = np.array(r["pf_t_ms"]);   v_pf = np.array(r["pf_vm_nodes"])
        v_jx_d = v_jx[:, -1] if v_jx.ndim > 1 else v_jx
        v_pf_d = v_pf[:, -1] if v_pf.ndim > 1 else v_pf
        ax.plot(t_pf, v_pf_d, "k--", lw=0.7, alpha=0.5)
        ax.plot(t_jx, v_jx_d, "-",   lw=1.6,
                label=f"$\\alpha$={r['amp_factor']:.2f}  "
                      f"(${r['amp_mA']*1000:.0f}\\,\\mu$A)")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("$V_m$ at distal node (mV)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=10, handlelength=1.4, borderaxespad=0.0)
    _label_panel(ax, "c")

    # Spike desync: sync index vs cathodic block amplitude.  The new
    # spike_desync.py reports results keyed as `D{D}_ifr{IFR}_f{F}` for
    # a 3-D parameter grid (diameter × IFR × stim-frequency).  For this
    # panel we pick one representative diameter + stim frequency and
    # show one curve per IFR.
    ax = axes[1, 1]
    d = json.loads((OUTROOT / "spike_desync/data_spike_desync.json").read_text())
    D_ref = 8.7      # closest available diameter to the legacy 10 µm pick
    F_ref = 50       # representative stim frequency
    ifrs = sorted(d["ifr_hz"])
    cols = plt.cm.plasma(np.linspace(0.15, 0.85, len(ifrs)))
    for i, ifr in enumerate(ifrs):
        key = f"D{D_ref}_ifr{ifr}_f{F_ref}"
        rr = d["results"].get(key)
        if rr is None:
            continue
        amps = np.array(rr["amps_mA"])
        sync = np.array(rr["mean_jax"])
        ci   = np.array(rr["ci95_half_jax"])
        ax.plot(amps, sync, "-o", color=cols[i], lw=2.0, ms=6,
                label=f"{ifr:g} Hz IFR")
        ax.fill_between(amps, sync - ci, sync + ci,
                        color=cols[i], alpha=0.15, lw=0)
    ax.axhline(1.0, color="grey", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlabel("cathodic perturbation amplitude (mA)")
    ax.set_ylabel("sync index")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=10, handlelength=1.4, borderaxespad=0.0,
              title=f"MRG $D={D_ref}\\,\\mu$m, $f_{{\\mathrm{{stim}}}}={F_ref}$ Hz",
              title_fontsize=9)
    ax.set_ylim(-0.05, 1.1)
    _label_panel(ax, "d")
    out = FIGDIR / "fig_phenomena.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


# ─── 4. scaling: all 7 models on log-log ──────────────────────────────────────
def fig_scaling_allmodels():
    """log-log of wall time vs N for all benchmarked models."""
    d = json.loads((OUTROOT / "scaling/data_scaling.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
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
        ax.grid(True, which="major", alpha=0.3, lw=0.5)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("number of fibres $N$")
    axes[0].set_ylabel("wall time per forward sim (s)")
    # Move the per-model series legend outside the axes on the right.
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                   fontsize=10, ncol=1, borderaxespad=0.0)
    axes[0].text(-0.16, 1.04, "(a)", transform=axes[0].transAxes,
                 ha="left", va="bottom", fontsize=15, fontweight="bold")
    axes[1].set_yscale("log")
    axes[1].axhline(1.0, color="grey", lw=0.8, ls="--", alpha=0.5)
    axes[1].set_xlabel("number of fibres $N$")
    axes[1].set_ylabel("jaxon speedup vs pyfibers ($\\times$)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                   fontsize=11, borderaxespad=0.0)
    axes[1].text(-0.16, 1.04, "(b)", transform=axes[1].transAxes,
                 ha="left", va="bottom", fontsize=15, fontweight="bold")
    plt.tight_layout()
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

    fig, axes = plt.subplots(1, 2, figsize=(max(14, 2.0 * n_models + 7), 5.4))

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
        fontsize=11,
    )
    # Sub-tick annotation for rect / wave
    for cx in centres:
        ax.text(cx - half*0.55, -0.135, "rect", ha="center", va="top",
                fontsize=10, color="grey", transform=ax.get_xaxis_transform())
        ax.text(cx + half*0.55, -0.135, "wave", ha="center", va="top",
                fontsize=10, color="grey", transform=ax.get_xaxis_transform())
    ax.set_ylabel("selectivity index")
    ax.set_ylim(-0.05, 1.10)
    ax.text(-0.10, 1.04, "(a)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)

    # ── Panel B: rect-vs-wave scatter coloured by model ────────────────────
    ax2 = axes[1]
    ax2.plot([0, 1.05], [0, 1.05], "k:", lw=0.8, alpha=0.5,
             label="wave = rect")
    for m in models:
        ax2.scatter(m["rect"], m["wave"], s=30, c=m["colour"],
                    edgecolors="black", lw=0.4, alpha=0.85, zorder=3,
                    label=f"{m['label'].replace(chr(10), ' ')}  "
                          f"(n={m['n']})")
    ax2.set_xlabel("rect SI")
    ax2.set_ylabel("warm-started wave SI")
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.10)
    all_rect = np.concatenate([m["rect"] for m in models])
    all_wave = np.concatenate([m["wave"] for m in models])
    n_improved  = int(np.sum(all_wave > all_rect + 0.005))
    n_preserved = int(np.sum(np.abs(all_wave - all_rect) <= 0.005))
    n_regressed = int(np.sum(all_wave < all_rect - 0.005))
    # Improvement counts as a compact annotation in the lower-right corner.
    ax2.text(0.98, 0.03,
             f"improved: {n_improved}\n"
             f"preserved: {n_preserved}\n"
             f"regressed: {n_regressed}",
             transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=10, family="DejaVu Sans Mono",
             bbox=dict(boxstyle="round,pad=0.28", fc="white",
                       ec="0.7", lw=0.5))
    ax2.text(-0.10, 1.04, "(b)", transform=ax2.transAxes,
             ha="left", va="bottom", fontsize=15, fontweight="bold")
    # Legend OUTSIDE the axes on the right so it doesn't cover scatter points.
    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
               fontsize=10, borderaxespad=0.0)
    ax2.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    out = FIGDIR / "fig_selectivity_summary.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


# ─── 5b. optimisation convergence curves ──────────────────────────────────────
def fig_optimization_convergence():
    """Per-model loss-vs-epoch curves across all 25 seeds.

    Two panels: (a) rectangular LBFGS-FD stage, (b) warm-started
    waveform-Adam stage.  For each panel + each model we plot the
    median (solid) and IQR (shaded) of the loss across seeds at each
    iteration index, padding shorter (early-stopped) seeds with their
    final loss value so the curve is well-defined out to the longest
    run.  This makes the spread of optimiser difficulty per model
    visible without overplotting 125 raw curves.
    """
    MODEL_DISPLAY = [
        ("selectivity_sweep_phase3_manuscript",   "MRG  5.7 µm",   "#1f77b4"),
        ("selectivity_sweep_phase3_mrg10um",      "MRG  10 µm",    "#4a90c4"),
        ("selectivity_sweep_phase3_sweeney_10um", "Sweeney  10 µm","#2ca02c"),
        ("selectivity_sweep_phase3_sundt_1um",    "Sundt  1 µm",   "#d62728"),
        ("selectivity_sweep_phase3_rattay_1um",   "Rattay  1 µm",  "#9467bd"),
    ]

    def _stack_padded(curves: list[np.ndarray]) -> np.ndarray:
        """Pad each curve to the max length with its final value, then
        stack into a [n_seeds, max_len] matrix."""
        L = max(len(c) for c in curves)
        out = np.empty((len(curves), L), dtype=np.float64)
        for i, c in enumerate(curves):
            out[i, :len(c)] = c
            if len(c) < L:
                out[i, len(c):] = c[-1]
        return out

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0),
                             gridspec_kw={"wspace": 0.22})

    for subdir, label, clr in MODEL_DISPLAY:
        sweep_dir = _outdir(subdir)
        files = sorted(sweep_dir.glob("data_seed_*.json"))
        if not files:
            continue
        rect_lh, wave_lh = [], []
        for fp in files:
            d = json.loads(fp.read_text())
            rect_lh.append(np.asarray(d["rect"]["loss_history"], dtype=float))
            wh = d.get("waveform", {}).get("loss_history") or []
            wave_lh.append(np.asarray(wh, dtype=float))

        # Panel A: rect stage
        ax = axes[0]
        R = _stack_padded(rect_lh)
        x = np.arange(R.shape[1])
        med = np.nanmedian(R, axis=0)
        q1  = np.nanpercentile(R, 25, axis=0)
        q3  = np.nanpercentile(R, 75, axis=0)
        ax.fill_between(x, q1, q3, color=clr, alpha=0.18, lw=0)
        ax.plot(x, med, "-", color=clr, lw=1.8, label=label)

        # Panel B: wave-stage SI evolution.  Warm-started wave often
        # leaves the loss nearly unchanged but can either preserve
        # SI=1 or destabilise the warm-start; SI-vs-iter shows that
        # directly.  Read si_history (saved alongside loss_history for
        # wave stage only).
        ax = axes[1]
        wave_si = []
        for fp in files:
            d = json.loads(fp.read_text())
            sh = d.get("waveform", {}).get("si_history") or []
            if len(sh) > 1:
                wave_si.append(np.asarray(sh, dtype=float))
        if not wave_si:
            continue
        S = _stack_padded(wave_si)
        xw = np.arange(S.shape[1])
        med_s = np.nanmedian(S, axis=0)
        q1_s  = np.nanpercentile(S, 25, axis=0)
        q3_s  = np.nanpercentile(S, 75, axis=0)
        ax.fill_between(xw, q1_s, q3_s, color=clr, alpha=0.18, lw=0)
        ax.plot(xw, med_s, "-", color=clr, lw=1.8,
                label=f"{label}  (n={len(wave_si)})")

    # Panel A formatting (log y, loss).
    ax = axes[0]
    ax.set_yscale("log")
    ax.set_xlabel("optimiser iteration")
    ax.set_ylabel("WQ loss")
    ax.text(-0.14, 1.03, "(a)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=15, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3, lw=0.5)
    # Capture per-model legend handles from panel A; share at figure bottom.
    panel_a_handles, panel_a_labels = ax.get_legend_handles_labels()

    # Panel B formatting (linear y, SI in [0, 1]).
    ax = axes[1]
    accept_line = ax.axhline(
        0.95, color="red", lw=0.9, ls="--", alpha=0.7,
        label="acceptance (SI=0.95)",
    )
    ax.axhline(1.0, color="black", lw=0.6, ls=":", alpha=0.6)
    ax.set_xlabel("optimiser iteration")
    ax.set_ylabel("selectivity index")
    ax.set_ylim(-0.05, 1.08)
    ax.text(-0.14, 1.03, "(b)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=15, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3, lw=0.5)

    # Single shared legend below both panels (5 model curves + acceptance).
    fig.legend(
        panel_a_handles + [accept_line],
        panel_a_labels + ["acceptance (SI=0.95)"],
        loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.02),
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out = FIGDIR / "fig_optimization_convergence.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.name}")


# ─── 5c. mixed-diameter loss landscape ────────────────────────────────────────
def fig_mixed_diameter_landscape():
    """Bistability of the mixed-diameter loss landscape.

    Reads outputs/mixed_diameter_landscape/data_landscape.json (produced
    by experiments_v2/mixed_diameter_landscape.py) and renders two panels:

    (a) Activation rate (target / off-target) and selectivity index SI
        as a function of the bipolar amplitude scaling alpha.  Because
        the off-target population has a larger diameter (lower
        chronaxie), it fires first as alpha increases — the SI(alpha)
        curve is negative (anti-selective) across the entire
        intermediate-alpha regime; there is no positive-SI window.

    (b) Regularised WQ loss vs alpha at several lambda values.  For
        lambda=0 the loss has two local minima: alpha=0 (silent /
        trivial) and alpha-large (saturated, all-fire).  As lambda
        grows the all-fire minimum is penalised and the alpha=0
        attractor wins; no intermediate alpha produces a low-loss,
        high-SI optimum.  This is the loss-landscape obstruction
        discussed in Section~4.2: the gradient-informative forward
        solver still cannot reach SI>0 because the WQ landscape simply
        does not have a selective minimum in this regime.
    """
    p = _outdir("mixed_diameter_landscape") / "data_landscape.json"
    if not p.exists():
        print(f"  skip: {p} not found  (run experiments_v2/mixed_diameter_landscape.py)")
        return
    d = json.loads(p.read_text())
    alpha = np.asarray(d["alpha_mA"])
    si    = np.asarray(d["si"])
    tgt   = np.asarray(d["target_act_rate"])
    off   = np.asarray(d["offtarget_act_rate"])
    wq    = np.asarray(d["wq_loss"])
    losses_per_l = d["losses_per_lambda"]
    cfg = d["config"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0),
                             gridspec_kw={"wspace": 0.30})

    # ── Panel A: activation rates ───────────────────────────────────────────
    ax = axes[0]
    ax.plot(alpha, tgt, "-o", color="#2ca02c", lw=1.8, ms=5,
            label=f"target  ($D={cfg['target_D_um']}\\,\\mu$m)")
    ax.plot(alpha, off, "-s", color="#d62728", lw=1.8, ms=5, mfc="none",
            label=f"off-target  ($D={cfg['offtarget_D_um']}\\,\\mu$m)")
    ax.set_xlabel(r"bipolar amplitude scale $\alpha$ (mA)")
    ax.set_ylabel("fraction of fibres fired")
    ax.set_ylim(-0.05, 1.08)
    ax.text(-0.14, 1.03, "(a)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3, lw=0.5)

    # ── Panel B: SI(alpha) ───────────────────────────────────────────────────
    ax = axes[1]
    ax.fill_between(alpha, 0.0, np.minimum(si, 0.0),
                    color="#d62728", alpha=0.18, lw=0,
                    label="anti-selective  (SI<0)")
    ax.plot(alpha, si, "-", color="black", lw=2.2)
    ax.axhline(0.0, color="black", lw=0.6, ls=":", alpha=0.6)
    ax.axhline(1.0, color="grey", lw=0.6, ls="--", alpha=0.4)
    si_min_idx = int(np.argmin(si))
    ax.annotate(
        fr"$\min\,\mathrm{{SI}} = {si[si_min_idx]:+.2f}$",
        xy=(alpha[si_min_idx], si[si_min_idx]),
        xytext=(alpha[si_min_idx] + 0.8, si[si_min_idx] - 0.05),
        ha="left", va="top", fontsize=11,
        arrowprops=dict(arrowstyle="->", lw=0.9, color="#d62728"),
    )
    ax.set_xlabel(r"bipolar amplitude scale $\alpha$ (mA)")
    ax.set_ylabel(r"selectivity index $\mathrm{SI}(\alpha)$")
    ax.set_ylim(-1.05, 1.08)
    ax.text(-0.14, 1.03, "(b)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=15, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3, lw=0.5)

    # ── Panel C: regularised loss landscape ─────────────────────────────────
    ax = axes[2]
    lambdas = cfg["lambdas"]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(lambdas)))
    for i, lam in enumerate(lambdas):
        L = np.asarray(losses_per_l[f"{lam:g}"])
        ax.plot(alpha, L, "-", color=cmap[i], lw=2.0,
                label=fr"$\lambda = {lam:g}$")
    ax.set_xlabel(r"bipolar amplitude scale $\alpha$ (mA)")
    ax.set_ylabel(r"WQ loss $+\, \lambda\, \overline{I^{\,2}}$")
    ax.text(-0.14, 1.03, "(c)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=15, fontweight="bold")
    L_strong = np.asarray(losses_per_l[f"{lambdas[-1]:g}"])
    ax.annotate("silent / trivial",
                xy=(alpha[0], L_strong[0]),
                xytext=(alpha[0] + 0.5, L_strong[0] + 0.10),
                fontsize=11, ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.9, color="0.3"))
    # Legend outside (right) so the four lambda lines remain visible.
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=11, borderaxespad=0.0)
    ax.grid(True, alpha=0.3, lw=0.5)
    out = FIGDIR / "fig_mixed_diameter_landscape.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.name}")


# ─── 6. cross-section activation gallery ──────────────────────────────────────
def _reproduce_nerve_for_seed(seed: int, fiber_diam_um: float,
                              n_fascicles: int = 8,
                              n_fibers_per_fascicle: int = 25,
                              nerve_radius_um: float = 500.0,
                              randomize_divider: bool = True,
                              default_divider_deg: float = 0.0):
    """Re-run the deterministic Hussain-style nerve generator for a given
    sweep seed, returning ``(nerve, divider_deg)``.  Mirrors what
    ``experiments_v2/selectivity_sweep.py`` does on the cluster, so we
    can recover the fascicle outlines + divider angle that the saved
    ``data_seed_NNNN.json`` does not include."""
    from jaxfibers.nerve.geometry import make_hussain_style_nerve
    if randomize_divider:
        divider_deg = float(
            np.random.default_rng(seed + 1_000_003).uniform(0.0, 180.0)
        )
    else:
        divider_deg = default_divider_deg
    nerve = make_hussain_style_nerve(
        n_fascicles=n_fascicles,
        n_fibers_per_fascicle=n_fibers_per_fascicle,
        nerve_radius_um=nerve_radius_um,
        divider_angle_deg=divider_deg,
        diameters=[fiber_diam_um],
        seed=seed,
    )
    return nerve, divider_deg


def _draw_cross_section_cell(ax_xsec, ax_pulse_grid, seed_num: int,
                             json_path, fiber_diam_um: float):
    """Draw one (cross-section + pulse-strip) cell into the two supplied
    axes / sub-grid.  ``ax_xsec`` is a single Axes; ``ax_pulse_grid`` is
    a list of 6 Axes laid out left-to-right for the 6 cuff contacts."""
    from matplotlib.patches import Circle as _Circle, Wedge as _Wedge

    d = json.loads(json_path.read_text())
    amps = np.asarray(d["rect"]["amps_mA"], dtype=float)
    acts = np.asarray(d["rect"]["final_acts"], dtype=float)
    rect_si = float(d["rect"]["final_si"])
    fired = acts >= 0.5

    nerve, divider_deg = _reproduce_nerve_for_seed(
        seed=int(seed_num), fiber_diam_um=fiber_diam_um,
    )
    # If the saved target_mask disagrees with the regenerated one, prefer
    # the saved one (more authoritative) and skip drawing the divider's
    # target/off-target arrow heads if they would be misleading.
    saved_tgt = np.asarray(d["nerve"]["target_mask"], dtype=bool)
    regen_tgt = nerve.target_mask.astype(bool)
    if saved_tgt.shape == regen_tgt.shape and bool((saved_tgt == regen_tgt).all()):
        target_mask = regen_tgt
    else:
        target_mask = saved_tgt

    # ── Cross-section axes ───────────────────────────────────────────────────
    R = 500.0    # NERVE_RADIUS_UM
    cuff_r = 1500.0

    # Nerve outline
    ax_xsec.add_patch(_Circle((0, 0), R, fill=False,
                              ec="0.55", lw=0.8, ls="--"))
    # Fascicle disks (light grey wash)
    for fc in nerve.fascicles:
        ax_xsec.add_patch(_Circle((fc.cx_um, fc.cy_um), fc.r_um,
                                  fc=(0.93, 0.93, 0.93), ec="0.65", lw=0.5))

    # Divider line through the centre, spanning the full nerve diameter
    th = np.deg2rad(divider_deg)
    # Line direction (along the divider) = (cos θ, sin θ); normal = (-sin θ, cos θ).
    cosT, sinT = np.cos(th), np.sin(th)
    line_len = 1.05 * R
    ax_xsec.plot([-line_len * cosT, line_len * cosT],
                 [-line_len * sinT, line_len * sinT],
                 color="black", lw=2.4, solid_capstyle="round", zorder=4)

    # 'target' / 'off' labels placed perpendicular to the divider, just
    # outside the contact ring so they never overlap the contacts.  No
    # arrow — keeps the cell clean.
    nrm_x, nrm_y = -sinT, cosT       # unit normal, points into the target half-plane

    # Fibers (4 styles)
    fx = np.asarray(nerve.fiber_x_um if saved_tgt is None else d["nerve"]["fiber_x_um"])
    fy = np.asarray(nerve.fiber_y_um if saved_tgt is None else d["nerve"]["fiber_y_um"])
    # Prefer saved positions (always available).
    fx = np.asarray(d["nerve"]["fiber_x_um"], dtype=float)
    fy = np.asarray(d["nerve"]["fiber_y_um"], dtype=float)

    is_tgt = target_mask
    tgt_fired   = is_tgt &  fired
    tgt_silent  = is_tgt & ~fired
    off_fired   = (~is_tgt) &  fired
    off_silent  = (~is_tgt) & ~fired

    # Off-target silent: small grey dots (background)
    ax_xsec.scatter(fx[off_silent], fy[off_silent],
                    s=8, c="0.55", edgecolors="none", alpha=0.7, zorder=2)
    # Target silent: open green circles
    ax_xsec.scatter(fx[tgt_silent], fy[tgt_silent],
                    s=22, facecolors="none", edgecolors="#2ca02c", lw=1.1, zorder=3)
    # Target fired: filled green
    ax_xsec.scatter(fx[tgt_fired], fy[tgt_fired],
                    s=26, c="#2ca02c", edgecolors="black", lw=0.3, zorder=4)
    # Off-target fired (LEAKAGE): red ✕
    ax_xsec.scatter(fx[off_fired], fy[off_fired],
                    s=44, marker="x", c="#d62728", lw=1.6, zorder=5)

    # Contacts at ±60°, sized by |amp|, coloured by polarity.  Drawn
    # on a smaller inscribed ring (just outside the nerve outline) so
    # they sit close enough to the cross-section to read at a glance.
    contact_ring_r = R * 1.18  # just outside the nerve outline
    K = 6
    angles = np.linspace(0.0, 2 * np.pi, K, endpoint=False)
    amp_max = max(float(np.max(np.abs(amps))), 1e-9)
    for k, ang in enumerate(angles):
        a_k = float(amps[k])
        cx_k = contact_ring_r * np.cos(ang)
        cy_k = contact_ring_r * np.sin(ang)
        if a_k < 0:
            clr = "#1f77b4"  # cathodic = blue
        else:
            clr = "#ff7f0e"  # anodic = orange
        # Size scales with |amp| / amp_max in [0.4, 1.0]
        s_k = 40 + 220 * (abs(a_k) / amp_max)
        ax_xsec.scatter(cx_k, cy_k, s=s_k, c=clr, edgecolors="black",
                        lw=0.5, marker="o", zorder=6)

    # 'target' / 'off-target' labels just outside the contact ring,
    # perpendicular to the divider line.  Drawn last so they sit on top.
    lbl_r = R * 1.42
    ax_xsec.text(
        +nrm_x * lbl_r, +nrm_y * lbl_r, "target",
        ha="center", va="center", fontsize=7, color="0.10",
        style="italic", fontweight="bold", zorder=7,
        bbox=dict(boxstyle="round,pad=0.18", fc="white",
                  ec="0.6", lw=0.4, alpha=0.95),
    )
    ax_xsec.text(
        -nrm_x * lbl_r, -nrm_y * lbl_r, "off-target",
        ha="center", va="center", fontsize=7, color="0.40",
        style="italic", zorder=7,
        bbox=dict(boxstyle="round,pad=0.18", fc="white",
                  ec="0.6", lw=0.4, alpha=0.95),
    )

    # Aspect / limits — pad enough to fit the label ring + bboxes.
    pad = R * 0.62
    ax_xsec.set_xlim(-R - pad, R + pad)
    ax_xsec.set_ylim(-R - pad, R + pad)
    ax_xsec.set_aspect("equal", adjustable="box")
    ax_xsec.set_xticks([]); ax_xsec.set_yticks([])
    for spine in ax_xsec.spines.values():
        spine.set_visible(False)

    # Per-cell annotation: seed + SI in the top-right corner
    ax_xsec.text(0.98, 0.97, f"seed {int(seed_num)}\nSI = {rect_si:+.2f}",
                 transform=ax_xsec.transAxes, ha="right", va="top",
                 fontsize=7, family="DejaVu Sans Mono",
                 bbox=dict(boxstyle="round,pad=0.20", fc="white",
                           ec="0.7", lw=0.5))

    # ── Pulse strip ──────────────────────────────────────────────────────────
    # 6 mini-panels, one per contact, showing the monophasic rectangular
    # pulse (delay 1 ms, pulse-width 0.1 ms, total span 1.5 ms shown).
    DELAY_MS = 1.0
    PW_MS    = 0.1
    T_VIEW   = 1.5
    t_grid = np.linspace(0.0, T_VIEW, 256)
    pulse_mask = ((t_grid >= DELAY_MS) & (t_grid < DELAY_MS + PW_MS)).astype(float)
    y_lim = max(amp_max * 1.25, 1e-3)
    for k, ax_p in enumerate(ax_pulse_grid):
        a_k = float(amps[k])
        clr = "#1f77b4" if a_k < 0 else "#ff7f0e"
        ax_p.fill_between(t_grid, 0.0, a_k * pulse_mask,
                          color=clr, alpha=0.85, lw=0)
        ax_p.axhline(0.0, color="0.4", lw=0.5)
        ax_p.set_ylim(-y_lim, y_lim)
        ax_p.set_xlim(0.0, T_VIEW)
        ax_p.set_xticks([])
        ax_p.set_yticks([])
        for spine in ax_p.spines.values():
            spine.set_visible(False)
        # Tiny label inside each pulse panel: "C{k}" top-left, "±A.AA"
        # bottom-right of the same panel.
        sign_chr = "−" if a_k < 0 else "+"
        ax_p.text(0.05, 0.96, f"C{k}",
                  transform=ax_p.transAxes, ha="left", va="top",
                  fontsize=7, color="0.20", family="DejaVu Sans Mono")
        ax_p.text(0.97, 0.04, f"{sign_chr}{abs(a_k):.2f}",
                  transform=ax_p.transAxes, ha="right", va="bottom",
                  fontsize=7, color="0.20", family="DejaVu Sans Mono")


def _make_xsection_gallery(seed_picker, out_name: str, n_seeds_per_model: int = 2):
    """Shared gallery renderer: build (n_models × n_seeds_per_model) grid
    of cross-section + pulse-strip cells.  ``seed_picker(scored)`` takes
    the list of ``(rect_si, seed_num, json_path)`` for one model
    (sorted ascending by seed) and returns the seeds it wants picked,
    in left-to-right column order."""
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.lines import Line2D

    MODEL_DISPLAY = [
        ("selectivity_sweep_phase3_manuscript",   "MRG\n5.7 µm",    5.7),
        ("selectivity_sweep_phase3_mrg10um",      "MRG\n10 µm",    10.0),
        ("selectivity_sweep_phase3_sweeney_10um", "Sweeney\n10 µm",10.0),
        ("selectivity_sweep_phase3_sundt_1um",    "Sundt\n1 µm",    1.0),
        ("selectivity_sweep_phase3_rattay_1um",   "Rattay\n1 µm",   1.0),
    ]

    rows: list[tuple[str, float, list]] = []
    for subdir, label, diam in MODEL_DISPLAY:
        sweep_dir = _outdir(subdir)
        seed_jsons = sorted(sweep_dir.glob("data_seed_*.json"))
        if not seed_jsons:
            continue
        scored = []
        for j in seed_jsons:
            try:
                rect_si = float(json.loads(j.read_text())["rect"]["final_si"])
            except Exception:
                continue
            seed_num = int(j.stem.split("_")[-1])
            scored.append((rect_si, seed_num, j))
        picked = seed_picker(scored)[:n_seeds_per_model]
        if picked:
            rows.append((label, diam, picked))

    if not rows:
        return

    n_rows = len(rows)
    n_cols = n_seeds_per_model

    fig = plt.figure(figsize=(4.6 * n_cols + 0.7, 4.4 * n_rows))
    outer = GridSpec(
        n_rows, n_cols + 1,
        width_ratios=[0.10] + [1.0] * n_cols,
        wspace=0.18, hspace=0.20,
        left=0.02, right=0.985, top=0.95, bottom=0.02,
    )

    for r, (label, diam, picked) in enumerate(rows):
        ax_lbl = fig.add_subplot(outer[r, 0])
        ax_lbl.axis("off")
        ax_lbl.text(
            0.5, 0.5, label,
            ha="center", va="center", rotation=90,
            fontsize=11, fontweight="bold",
            transform=ax_lbl.transAxes,
        )
        for c, (_si, seed_num, json_path) in enumerate(picked):
            inner = GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[r, c + 1],
                height_ratios=[2.6, 1.0], hspace=0.12,
            )
            ax_xsec = fig.add_subplot(inner[0, 0])
            pulse_grid_spec = GridSpecFromSubplotSpec(
                1, 6, subplot_spec=inner[1, 0], wspace=0.18,
            )
            ax_pulses = [fig.add_subplot(pulse_grid_spec[0, k]) for k in range(6)]
            _draw_cross_section_cell(
                ax_xsec, ax_pulses, seed_num=seed_num,
                json_path=json_path, fiber_diam_um=diam,
            )

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", mfc="#2ca02c",
               mec="black", mew=0.3, ms=7, label="target fired"),
        Line2D([0], [0], marker="o", color="none", mfc="none",
               mec="#2ca02c", mew=1.2, ms=7, label="target silent"),
        Line2D([0], [0], marker="x", color="#d62728",
               mew=1.6, ms=8, ls="", label="off-target fired (leak)"),
        Line2D([0], [0], marker="o", color="none", mfc="0.55",
               mec="none", ms=5, label="off-target silent"),
        Line2D([0], [0], marker="o", color="none", mfc="#1f77b4",
               mec="black", mew=0.4, ms=8, label="contact (cathodic)"),
        Line2D([0], [0], marker="o", color="none", mfc="#ff7f0e",
               mec="black", mew=0.4, ms=8, label="contact (anodic)"),
    ]
    fig.legend(
        handles=legend_handles, loc="upper center",
        bbox_to_anchor=(0.52, 0.995), ncol=6, frameon=False, fontsize=8.5,
    )

    out = FIGDIR / out_name
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


def fig_selectivity_xsections():
    """Cross-section + per-contact pulse gallery, top-SI seeds per model.

    Two seeds per model (top SI), 5 models, 5 rows × 2 cols = 10 cells.
    Each cell stacks a cross-section over a 6-mini-panel pulse strip
    (one panel per cuff contact).  Marker scheme: target-fired = filled
    green, target-silent = open green, off-target-fired = red ✕
    (leakage), off-target-silent = small grey dot.  Cuff contacts are
    coloured by polarity (cathodic blue / anodic orange) and sized by
    |amplitude|.  The thick black line is the target / off-target
    divider (random per seed).  Row labels (model + diameter) sit once
    on the left margin per row.
    """
    def _top_si(scored):
        # Highest SI first; tie-break by seed asc; then re-order picked
        # subset by seed asc for canonical left-to-right reading.
        scored = sorted(scored, key=lambda r: (-r[0], r[1]))
        return sorted(scored[:2], key=lambda r: r[1])
    _make_xsection_gallery(_top_si, "fig_selectivity_xsections.png")


def fig_selectivity_xsections_hard():
    """Companion to ``fig_selectivity_xsections``: shows the per-model
    worst-SI seeds (the imperfect cases) using the same cell layout.

    The top-SI gallery (Fig. 6) shows what the optimiser achieves when
    the random anatomy is favourable; this figure shows the residual
    failure modes — seeds with $\\mathrm{SI}<1.0$ where either some
    target fibres remain silent (open green) or some off-target fibres
    are recruited (red ✕).  Same five fibre models × two seeds layout."""
    def _bottom_si(scored):
        # Lowest SI first; tie-break by seed asc; then re-order picked
        # subset by seed asc for canonical left-to-right reading.
        scored = sorted(scored, key=lambda r: (r[0], r[1]))
        return sorted(scored[:2], key=lambda r: r[1])
    _make_xsection_gallery(_bottom_si, "fig_selectivity_xsections_hard.png")


# ============================================================================
# Duke FEM-nerve figures (parallel to the synthetic Hussain-style nerves)
# ============================================================================
# These read JSONs written by experiments_v2/selectivity_sweep_duke.py from
# outputs/selectivity_sweep_duke_<sample>/ and render selectivity-summary,
# cross-section gallery, hard-case gallery, and convergence figures using
# the realistic polygonal fascicle geometry from duke_Ves/<sample>/nerve_xsec.json.
# Coexists with — does not replace — the synthetic-nerve figures above.

# Per-sample cache: nerve outline polygon + fascicle polygons + electrode metadata.
_DUKE_XSEC_CACHE: dict[str, dict] = {}
# Where the Duke FEM bundles live on disk.  ``DUKE_VES_ROOT`` env var
# wins; otherwise prefer the new ``duke_meshes/`` (swine + human) bundle
# at the project root, fall back to the legacy ``duke_Ves/`` name.  Only
# roots that exist on disk are kept; the cross-section drawer iterates
# them when resolving a sample directory.
_DUKE_ROOTS: list[Path] = []
_env_root = os.environ.get("DUKE_VES_ROOT", "").strip()
if _env_root:
    _DUKE_ROOTS.append(Path(_env_root))
_DUKE_ROOTS.extend([HERE.parent / "duke_meshes", HERE.parent / "duke_Ves"])
_DUKE_ROOTS = [p for p in _DUKE_ROOTS if p.exists()]
_DUKE_ROOT = _DUKE_ROOTS[0] if _DUKE_ROOTS else (HERE.parent / "duke_meshes")


def _load_duke_xsec(sample: str) -> dict | None:
    """Load + cache duke_Ves/<sample>/nerve_xsec.json (and electrode_config).

    Returns dict with keys ``outline`` (Nx2 µm), ``fascicles`` (list of
    dicts), ``contact_phi_deg`` (12,), ``contact_z_m`` (12,), or ``None``
    if the sample directory is not present locally."""
    if sample in _DUKE_XSEC_CACHE:
        return _DUKE_XSEC_CACHE[sample]
    # Try every configured root in order (env override, duke_meshes/, duke_Ves/).
    sample_dir = None
    for root in _DUKE_ROOTS or [_DUKE_ROOT]:
        cand = root / sample
        if (cand / "nerve_xsec.json").exists():
            sample_dir = cand
            break
    if sample_dir is None:
        return None
    nxsec_path = sample_dir / "nerve_xsec.json"
    nx = json.loads(nxsec_path.read_text())
    fasc = []
    for fc in nx["fascicles"]:
        fasc.append({
            "id":       int(fc["id"]),
            "centroid": tuple(fc["centroid_xy_um"]),
            "radius":   float(fc["radius_um"]),
            "polygon":  np.asarray(fc["polygon_xy_um"], dtype=np.float64),
        })
    # Electrode metadata — best-effort, may be missing.
    contact_phi_deg = None
    contact_z_m = None
    ec_path = sample_dir / "electrode_config.json"
    if ec_path.exists():
        try:
            ec = json.loads(ec_path.read_text())
            # Try both schema variants seen in the FEM bundles.
            patches = ec.get("patches") or ec.get("contacts") or []
            if patches:
                contact_phi_deg = np.array([float(p.get("phi_deg",
                                          np.rad2deg(p.get("phi", 0.0))))
                                            for p in patches])
                contact_z_m = np.array([float(p.get("z", 0.0)) for p in patches])
        except Exception:
            pass
    # Fall back to canonical 3 axial rows × 4 angles if needed.
    if contact_phi_deg is None or len(contact_phi_deg) != 12:
        contact_phi_deg = np.repeat([0.0, 90.0, 180.0, 270.0], 3)
        contact_z_m     = np.tile([-1.35e-3, 0.0, +1.35e-3], 4)
    out = {
        "outline":         np.asarray(nx["nerve_outline_xy_um"], dtype=np.float64),
        "fascicles":       fasc,
        "contact_phi_deg": contact_phi_deg,
        "contact_z_m":     contact_z_m,
    }
    _DUKE_XSEC_CACHE[sample] = out
    return out


def _duke_sweep_dirs() -> list[Path]:
    """Discover Duke sweep output directories under ``outputs/``.

    Searches both the new layout
    (``outputs/duke_sweeps/<sample>/data_seed_*.json``) and the legacy
    layout (``outputs/selectivity_sweep_duke_<sample>/...``).  Returns
    a deduplicated list of output dirs, one per Duke sample with at
    least one JSON inside.
    """
    found: list[Path] = []
    if not OUTROOT.exists():
        return found
    # New layout: outputs/duke_sweeps/<sample>/
    duke_root = OUTROOT / "duke_sweeps"
    if duke_root.exists():
        for d in sorted(duke_root.iterdir()):
            if d.is_dir() and any(d.glob("data_seed_*.json")):
                found.append(d)
    # Legacy layout: outputs/selectivity_sweep_duke_<sample>/
    for d in sorted(OUTROOT.glob("selectivity_sweep_duke_*")):
        if any(d.glob("data_seed_*.json")):
            found.append(d)
    # De-duplicate by sample name (first occurrence wins).
    seen = set()
    uniq = []
    for d in found:
        name = _duke_sample_name_from_dir(d)
        if name not in seen:
            seen.add(name)
            uniq.append(d)
    return uniq


def _duke_sample_name_from_dir(sweep_dir: Path) -> str:
    """Extract the sample bundle name from either layout's dir name.

    New layout: ``outputs/duke_sweeps/sub-10_sam-1`` -> ``sub-10_sam-1``.
    Legacy:    ``outputs/selectivity_sweep_duke_sub-10_sam-1`` -> ``sub-10_sam-1``.
    """
    name = sweep_dir.name
    return name[len("selectivity_sweep_duke_"):] if name.startswith("selectivity_sweep_duke_") else name


def _duke_species_of(sample_name: str) -> str:
    """Infer species from a Duke sample folder name.

    Convention: ``human-sub-*`` -> human; everything else -> swine.
    """
    return "human" if sample_name.startswith("human") else "swine"


def _draw_duke_cross_section_cell(ax_xsec, ax_pulse_grid,
                                   json_path: Path, sample: str):
    """Cross-section + 12-contact pulse strip for one Duke seed.  Handles
    polygonal fascicles + the 12-contact MultiContact cuff (3 axial rows
    × 4 azimuthal angles).  Mirrors ``_draw_cross_section_cell`` for
    synthetic nerves but reads the irregular geometry from the cached
    ``_load_duke_xsec`` output and the per-seed JSON.
    """
    from matplotlib.patches import Polygon as _Polygon
    d = json.loads(json_path.read_text())
    amps = np.asarray(d["rect"]["amps_mA"], dtype=float)
    acts = np.asarray(d["rect"]["final_acts"], dtype=float)
    rect_si  = float(d["rect"]["final_si"])
    seed_num = int(d["seed"])
    divider_deg = float(d.get("divider_deg", 0.0))
    fired = acts >= 0.5
    target_mask = np.asarray(d["nerve"]["target_mask"], dtype=bool)
    fx = np.asarray(d["nerve"]["fiber_x_um"], dtype=float)
    fy = np.asarray(d["nerve"]["fiber_y_um"], dtype=float)

    geom = _load_duke_xsec(sample)
    if geom is None:
        ax_xsec.text(0.5, 0.5,
                     f"Duke geometry not found for\n{sample}\n"
                     "(check duke_Ves/<sample>/nerve_xsec.json)",
                     ha="center", va="center", transform=ax_xsec.transAxes,
                     fontsize=9, style="italic", color="grey")
        ax_xsec.set_xticks([]); ax_xsec.set_yticks([])
        return

    outline = geom["outline"]
    # Outer half-extent in µm — used for the contact ring and label radii.
    R_ext = float(np.max(np.abs(outline)) * 1.05)

    # Nerve outer outline (polygon)
    ax_xsec.add_patch(_Polygon(outline, closed=True, fill=False,
                                ec="0.55", lw=1.0, ls="--"))
    # Fascicle polygons — classify target vs off-target by divider line
    th = np.deg2rad(divider_deg)
    cosT, sinT = np.cos(th), np.sin(th)
    nrm_x, nrm_y = -sinT, cosT
    for fc in geom["fascicles"]:
        cx, cy = fc["centroid"]
        side = nrm_x * cx + nrm_y * cy
        is_tgt_fasc = side > 0.0
        face = (0.78, 0.93, 0.78) if is_tgt_fasc else (0.88, 0.88, 0.88)
        edge = "#2ca02c" if is_tgt_fasc else "0.5"
        ax_xsec.add_patch(_Polygon(fc["polygon"], closed=True,
                                    fc=face, ec=edge, lw=0.6, alpha=0.85))

    # Divider line through (0, 0) spanning the outline
    line_len = R_ext * 1.10
    ax_xsec.plot([-line_len * cosT, line_len * cosT],
                 [-line_len * sinT, line_len * sinT],
                 color="black", lw=2.4, solid_capstyle="round", zorder=4)

    # Fibers (4 styles, identical to synthetic)
    tgt_fired   = target_mask &  fired
    tgt_silent  = target_mask & ~fired
    off_fired   = (~target_mask) &  fired
    off_silent  = (~target_mask) & ~fired
    ax_xsec.scatter(fx[off_silent], fy[off_silent],
                    s=8, c="0.55", edgecolors="none", alpha=0.7, zorder=2)
    ax_xsec.scatter(fx[tgt_silent], fy[tgt_silent],
                    s=22, facecolors="none", edgecolors="#2ca02c", lw=1.1, zorder=3)
    ax_xsec.scatter(fx[tgt_fired], fy[tgt_fired],
                    s=26, c="#2ca02c", edgecolors="black", lw=0.3, zorder=4)
    ax_xsec.scatter(fx[off_fired], fy[off_fired],
                    s=44, marker="x", c="#d62728", lw=1.6, zorder=5)

    # 12 contacts on an outer ring, sized by |amp|, coloured by polarity
    contact_ring_r = R_ext * 1.18
    amp_max = max(float(np.max(np.abs(amps))), 1e-9)
    phi_deg = geom["contact_phi_deg"]
    # The MultiContact array stacks 3 axial contacts at each of 4 angles;
    # to distinguish them visually within one azimuthal "station" we
    # nudge them tangentially by a small offset proportional to their z.
    z_m = geom["contact_z_m"]
    z_min, z_max = float(z_m.min()), float(z_m.max())
    z_range = max(z_max - z_min, 1e-9)
    NUDGE_DEG = 8.0
    for k in range(len(amps)):
        a_k = float(amps[k])
        # Tangential nudge in degrees: z=-z_max → -NUDGE, z=+z_max → +NUDGE
        nudge = ((z_m[k] - 0.5 * (z_min + z_max)) / (0.5 * z_range)) * NUDGE_DEG
        ang_deg = phi_deg[k] + nudge
        ang = np.deg2rad(ang_deg)
        cx_k = contact_ring_r * np.cos(ang)
        cy_k = contact_ring_r * np.sin(ang)
        clr = "#1f77b4" if a_k < 0 else "#ff7f0e"
        s_k = 38 + 200 * (abs(a_k) / amp_max)
        ax_xsec.scatter(cx_k, cy_k, s=s_k, c=clr, edgecolors="black",
                        lw=0.5, marker="o", zorder=6)

    # 'target' / 'off-target' labels perpendicular to the divider
    lbl_r = R_ext * 1.42
    ax_xsec.text(
        +nrm_x * lbl_r, +nrm_y * lbl_r, "target",
        ha="center", va="center", fontsize=8, color="0.10",
        style="italic", fontweight="bold", zorder=7,
        bbox=dict(boxstyle="round,pad=0.18", fc="white",
                  ec="0.6", lw=0.4, alpha=0.95),
    )
    ax_xsec.text(
        -nrm_x * lbl_r, -nrm_y * lbl_r, "off-target",
        ha="center", va="center", fontsize=8, color="0.40",
        style="italic", zorder=7,
        bbox=dict(boxstyle="round,pad=0.18", fc="white",
                  ec="0.6", lw=0.4, alpha=0.95),
    )

    # Aspect / limits — keep symmetric around (0,0).
    pad = R_ext * 0.62
    ax_xsec.set_xlim(-R_ext - pad, R_ext + pad)
    ax_xsec.set_ylim(-R_ext - pad, R_ext + pad)
    ax_xsec.set_aspect("equal", adjustable="box")
    ax_xsec.set_xticks([]); ax_xsec.set_yticks([])
    for spine in ax_xsec.spines.values():
        spine.set_visible(False)
    # Per-cell annotation
    ax_xsec.text(0.98, 0.97,
                 f"seed {seed_num}\nSI = {rect_si:+.2f}",
                 transform=ax_xsec.transAxes, ha="right", va="top",
                 fontsize=8, family="DejaVu Sans Mono",
                 bbox=dict(boxstyle="round,pad=0.20", fc="white",
                           ec="0.7", lw=0.5))

    # ── 12-contact pulse strip ──────────────────────────────────────────────
    DELAY_MS = 1.0
    PW_MS    = 0.1
    T_VIEW   = 1.5
    t_grid = np.linspace(0.0, T_VIEW, 256)
    pulse_mask = ((t_grid >= DELAY_MS) & (t_grid < DELAY_MS + PW_MS)).astype(float)
    y_lim = max(amp_max * 1.25, 1e-3)
    for k, ax_p in enumerate(ax_pulse_grid):
        a_k = float(amps[k])
        clr = "#1f77b4" if a_k < 0 else "#ff7f0e"
        ax_p.fill_between(t_grid, 0.0, a_k * pulse_mask,
                          color=clr, alpha=0.85, lw=0)
        ax_p.axhline(0.0, color="0.4", lw=0.5)
        ax_p.set_ylim(-y_lim, y_lim)
        ax_p.set_xlim(0.0, T_VIEW)
        ax_p.set_xticks([]); ax_p.set_yticks([])
        for spine in ax_p.spines.values():
            spine.set_visible(False)
        sign_chr = "−" if a_k < 0 else "+"
        ax_p.text(0.05, 0.96, f"C{k}",
                  transform=ax_p.transAxes, ha="left", va="top",
                  fontsize=7, color="0.20", family="DejaVu Sans Mono")
        ax_p.text(0.97, 0.04, f"{sign_chr}{abs(a_k):.2f}",
                  transform=ax_p.transAxes, ha="right", va="bottom",
                  fontsize=7, color="0.20", family="DejaVu Sans Mono")


def _make_duke_xsection_gallery(seed_picker, out_name_template: str,
                                  n_seeds_per_sample: int = 1,
                                  n_grid_cols: int = 4):
    """Build species-grouped Duke cross-section galleries.

    Produces TWO files: ``<out_name_template>_swine.png`` and
    ``<out_name_template>_human.png`` (one per species detected in
    ``_duke_sweep_dirs()``).  Within each species the samples are
    arranged in a ``n_grid_cols``-wide grid of cells; each cell holds
    a cross-section + 12-contact pulse strip for the seed(s) picked by
    ``seed_picker``.  With the default ``n_seeds_per_sample=1`` (the
    standard 1-divider-per-nerve sweep) the cell count equals the
    sample count per species.
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.lines import Line2D

    sweep_dirs = _duke_sweep_dirs()
    if not sweep_dirs:
        print(f"  no Duke sweep dirs found under outputs/ — skipping {out_name_template}")
        return

    # Bucket sweep_dirs by species first.
    by_species: dict[str, list[tuple[str, list]]] = {"swine": [], "human": []}
    for sweep_dir in sweep_dirs:
        sample = _duke_sample_name_from_dir(sweep_dir)
        species = _duke_species_of(sample)
        seed_jsons = sorted(sweep_dir.glob("data_seed_*.json"))
        scored = []
        for j in seed_jsons:
            try:
                rect_si = float(json.loads(j.read_text())["rect"]["final_si"])
            except Exception:
                continue
            seed_num = int(j.stem.split("_")[-1])
            scored.append((rect_si, seed_num, j))
        picked = seed_picker(scored)[:n_seeds_per_sample]
        if picked:
            by_species[species].append((sample, picked))

    for species, samples in by_species.items():
        if not samples:
            continue
        _make_duke_xsection_grid(samples, n_grid_cols, species,
                                   f"{out_name_template}_{species}.png")


def _make_duke_xsection_grid(samples: list, n_grid_cols: int, species: str,
                              out_name: str):
    """Render the per-species cross-section grid."""
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.lines import Line2D

    # Each sample = one "cell" (cross-section + pulse strip).  With one
    # seed per sample there is exactly one cell per sample.  Multi-seed
    # samples expand to consecutive cells along the row-major sweep.
    cells: list[tuple[str, Path]] = []
    for sample, picked in samples:
        for _si, _seed_num, json_path in picked:
            cells.append((sample, json_path))

    n_cells = len(cells)
    n_cols = n_grid_cols
    n_rows = (n_cells + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(5.5 * n_cols, 5.6 * n_rows + 0.5))
    outer = GridSpec(
        n_rows, n_cols,
        wspace=0.22, hspace=0.30,
        left=0.03, right=0.985, top=0.96, bottom=0.04,
    )

    for i, (sample, json_path) in enumerate(cells):
        r, c = divmod(i, n_cols)
        inner = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[r, c],
            height_ratios=[2.6, 1.0], hspace=0.14,
        )
        ax_xsec = fig.add_subplot(inner[0, 0])
        pulse_spec = GridSpecFromSubplotSpec(
            1, 12, subplot_spec=inner[1, 0], wspace=0.20,
        )
        ax_pulses = [fig.add_subplot(pulse_spec[0, k]) for k in range(12)]
        _draw_duke_cross_section_cell(
            ax_xsec, ax_pulses, json_path=json_path, sample=sample,
        )
        # Sample tag at top-left of cell (above the cross-section).
        ax_xsec.text(0.02, 1.04, sample,
                     transform=ax_xsec.transAxes, ha="left", va="bottom",
                     fontsize=11, fontweight="bold", color="0.20")

    # Species header at top of figure.
    fig.text(0.5, 0.985,
             f"{species.capitalize()}  (n={n_cells})",
             ha="center", va="top", fontsize=15, fontweight="bold")

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", mfc="#2ca02c",
               mec="black", mew=0.3, ms=7, label="target fired"),
        Line2D([0], [0], marker="o", color="none", mfc="none",
               mec="#2ca02c", mew=1.2, ms=7, label="target silent"),
        Line2D([0], [0], marker="x", color="#d62728",
               mew=1.6, ms=8, ls="", label="off-target fired (leak)"),
        Line2D([0], [0], marker="o", color="none", mfc="0.55",
               mec="none", ms=5, label="off-target silent"),
        Line2D([0], [0], marker="o", color="none", mfc="#1f77b4",
               mec="black", mew=0.4, ms=8, label="contact (cathodic)"),
        Line2D([0], [0], marker="o", color="none", mfc="#ff7f0e",
               mec="black", mew=0.4, ms=8, label="contact (anodic)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.52, 0.995), ncol=6,
               frameon=False, fontsize=9)

    out = FIGDIR / out_name
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


def fig_duke_selectivity_xsections():
    """Duke gallery: best seed per nerve, grouped by species.

    With the default sweep (SEED_END=1 → 1 seed per nerve) ``_top`` just
    returns that single seed.  Produces two output files:
    ``fig_duke_selectivity_xsections_swine.png`` and ``_human.png``.
    """
    def _top(scored):
        scored = sorted(scored, key=lambda r: (-r[0], r[1]))
        return scored[:1]
    _make_duke_xsection_gallery(_top, "fig_duke_selectivity_xsections")


def fig_duke_selectivity_xsections_hard():
    """Duke gallery: worst seed per nerve, grouped by species.

    With 1 seed per nerve this is equivalent to the top-SI gallery
    (same single seed).  Kept as a separate entry point for symmetry
    with the synthetic figures and for the case where the user opts
    back into multi-seed mode (SEED_END>1)."""
    def _bot(scored):
        scored = sorted(scored, key=lambda r: (r[0], r[1]))
        return scored[:1]
    _make_duke_xsection_gallery(_bot, "fig_duke_selectivity_xsections_hard")


_DUKE_SPECIES_COLOURS = {"swine": "#3a6ec1", "human": "#d96b2c"}


def fig_duke_selectivity_summary():
    """Cross-anatomy SI distribution for the Duke sweep, grouped by species.

    Panel (a): two paired violins (rect / wave) — one for swine, one for
    human — pooled over every nerve in the species.  Per-nerve SI is
    overlaid as a jittered dot.
    Panel (b): rect-vs-wave scatter, points coloured by species, showing
    whether the wave stage improved on the rect optimum nerve-by-nerve.

    Reads every ``outputs/duke_sweeps/<sample>/data_seed_*.json`` (plus
    legacy ``outputs/selectivity_sweep_duke_*/...``) and infers species
    from the sample-name prefix (``human-*`` -> human, else swine).
    """
    sweep_dirs = _duke_sweep_dirs()
    if not sweep_dirs:
        print("  no Duke sweep dirs found under outputs/ — skipping")
        return

    # Collect per-nerve final SI's, bucketed by species.
    by_species: dict[str, dict] = {
        "swine": {"rect": [], "wave": [], "labels": []},
        "human": {"rect": [], "wave": [], "labels": []},
    }
    for sweep_dir in sweep_dirs:
        sample = _duke_sample_name_from_dir(sweep_dir)
        species = _duke_species_of(sample)
        seed_rects, seed_waves = [], []
        for j in sorted(sweep_dir.glob("data_seed_*.json")):
            try:
                dd = json.loads(j.read_text())
            except Exception:
                continue
            seed_rects.append(float(dd["rect"]["final_si"]))
            seed_waves.append(float(dd.get("waveform", {}).get("final_si",
                                                              dd["rect"]["final_si"])))
        if seed_rects:
            # Per-nerve SI = the median across that nerve's seeds (collapses
            # to the single value when SEED_END=1).
            by_species[species]["rect"].append(float(np.median(seed_rects)))
            by_species[species]["wave"].append(float(np.median(seed_waves)))
            by_species[species]["labels"].append(sample)

    species_present = [s for s in ("swine", "human") if by_species[s]["rect"]]
    if not species_present:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4),
                             gridspec_kw={"wspace": 0.22})

    # ── Panel A: per-species paired violins (rect / wave) ──────────────────
    ax = axes[0]
    rng = np.random.default_rng(0)
    centres = np.arange(len(species_present)) * 2.0
    half = 0.42
    for i, species in enumerate(species_present):
        rect = np.array(by_species[species]["rect"])
        wave = np.array(by_species[species]["wave"])
        clr = _DUKE_SPECIES_COLOURS[species]
        for x, vals, edge_alpha in [(centres[i] - half * 0.55, rect, 1.0),
                                    (centres[i] + half * 0.55, wave, 0.7)]:
            if len(vals) >= 2:
                parts = ax.violinplot([vals], positions=[x], widths=half,
                                      showmeans=False, showmedians=False,
                                      showextrema=False)
                for body in parts["bodies"]:
                    body.set_facecolor(clr)
                    body.set_edgecolor("black")
                    body.set_alpha(0.55 * edge_alpha)
                    body.set_linewidth(0.6)
            jitter = rng.uniform(-half * 0.18, half * 0.18, size=len(vals))
            ax.scatter(x + jitter, vals, s=22, color=clr, alpha=0.85,
                       edgecolors="black", lw=0.4, zorder=3)
            ax.hlines(float(np.median(vals)), x - half * 0.4, x + half * 0.4,
                      color="red", lw=1.8, zorder=4)
    ax.axhline(0.95, color="red", lw=0.8, ls="--", alpha=0.55,
               label="acceptance (SI=0.95)")
    ax.set_xticks(centres)
    ax.set_xticklabels(
        [f"{s.capitalize()}\n(n={len(by_species[s]['rect'])})"
         for s in species_present],
        fontsize=12, fontweight="bold",
    )
    for cx in centres:
        ax.text(cx - half * 0.55, -0.135, "rect", ha="center", va="top",
                fontsize=10, color="grey", transform=ax.get_xaxis_transform())
        ax.text(cx + half * 0.55, -0.135, "wave", ha="center", va="top",
                fontsize=10, color="grey", transform=ax.get_xaxis_transform())
    ax.set_ylabel("selectivity index")
    ax.set_ylim(-0.05, 1.10)
    ax.text(-0.08, 1.04, "(a)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)

    # ── Panel B: rect-vs-wave scatter, coloured by species ────────────────
    ax2 = axes[1]
    ax2.plot([0, 1.05], [0, 1.05], "k:", lw=0.8, alpha=0.5,
             label="wave = rect")
    for species in species_present:
        rect = np.array(by_species[species]["rect"])
        wave = np.array(by_species[species]["wave"])
        clr = _DUKE_SPECIES_COLOURS[species]
        ax2.scatter(rect, wave, s=44, c=clr,
                    edgecolors="black", lw=0.4, alpha=0.85, zorder=3,
                    label=f"{species.capitalize()}  (n={len(rect)})")
    ax2.set_xlabel("rect SI (per nerve)")
    ax2.set_ylabel("warm-started wave SI (per nerve)")
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.10)
    # Counts pooled across both species.
    all_rect = np.concatenate([np.array(by_species[s]["rect"])
                                for s in species_present])
    all_wave = np.concatenate([np.array(by_species[s]["wave"])
                                for s in species_present])
    n_improved  = int(np.sum(all_wave > all_rect + 0.005))
    n_preserved = int(np.sum(np.abs(all_wave - all_rect) <= 0.005))
    n_regressed = int(np.sum(all_wave < all_rect - 0.005))
    ax2.text(0.98, 0.03,
             f"improved:  {n_improved}\n"
             f"preserved: {n_preserved}\n"
             f"regressed: {n_regressed}",
             transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=10, family="DejaVu Sans Mono",
             bbox=dict(boxstyle="round,pad=0.28", fc="white",
                       ec="0.7", lw=0.5))
    ax2.text(-0.10, 1.04, "(b)", transform=ax2.transAxes,
             ha="left", va="bottom", fontsize=15, fontweight="bold")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
               fontsize=10, borderaxespad=0.0)
    ax2.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    out = FIGDIR / "fig_duke_selectivity_summary.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


def fig_duke_optimization_convergence():
    """Per-sample loss / SI convergence curves for the Duke sweep."""
    sweep_dirs = _duke_sweep_dirs()
    if not sweep_dirs:
        print("  no Duke sweep dirs found — skipping")
        return

    def _stack_padded(curves):
        L = max(len(c) for c in curves)
        out = np.empty((len(curves), L), dtype=np.float64)
        for i, c in enumerate(curves):
            out[i, :len(c)] = c
            if len(c) < L:
                out[i, len(c):] = c[-1]
        return out

    # Bucket rect loss + wave SI trajectories by species so we can show
    # one median + IQR band per species (instead of one line per nerve,
    # which would be 38 overlapping curves).
    by_species: dict[str, dict] = {
        "swine": {"rect": [], "wave_si": []},
        "human": {"rect": [], "wave_si": []},
    }
    for sweep_dir in sweep_dirs:
        sample = _duke_sample_name_from_dir(sweep_dir)
        species = _duke_species_of(sample)
        for j in sorted(sweep_dir.glob("data_seed_*.json")):
            try:
                dd = json.loads(j.read_text())
            except Exception:
                continue
            by_species[species]["rect"].append(
                np.asarray(dd["rect"]["loss_history"], float))
            sh = dd.get("waveform", {}).get("si_history") or []
            if len(sh) > 1:
                by_species[species]["wave_si"].append(np.asarray(sh, float))

    species_present = [s for s in ("swine", "human")
                       if by_species[s]["rect"]]
    if not species_present:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0),
                             gridspec_kw={"wspace": 0.22})

    for species in species_present:
        clr = _DUKE_SPECIES_COLOURS[species]
        rect_lh = by_species[species]["rect"]
        wave_si = by_species[species]["wave_si"]
        R = _stack_padded(rect_lh)
        x = np.arange(R.shape[1])
        med = np.nanmedian(R, axis=0)
        q1, q3 = np.nanpercentile(R, [25, 75], axis=0)
        axes[0].fill_between(x, q1, q3, color=clr, alpha=0.20, lw=0)
        axes[0].plot(x, med, "-", color=clr, lw=2.2,
                     label=f"{species.capitalize()}  (n={len(rect_lh)})")
        if wave_si:
            S = _stack_padded(wave_si)
            xw = np.arange(S.shape[1])
            med_s = np.nanmedian(S, axis=0)
            q1_s, q3_s = np.nanpercentile(S, [25, 75], axis=0)
            axes[1].fill_between(xw, q1_s, q3_s, color=clr, alpha=0.20, lw=0)
            axes[1].plot(xw, med_s, "-", color=clr, lw=2.2,
                         label=f"{species.capitalize()}  (n={len(wave_si)})")

    ax = axes[0]
    ax.set_yscale("log")
    ax.set_xlabel("optimiser iteration")
    ax.set_ylabel("WQ loss")
    ax.text(-0.14, 1.03, "(a)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=15, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3, lw=0.5)
    panel_a_handles, panel_a_labels = ax.get_legend_handles_labels()

    ax = axes[1]
    accept_line = ax.axhline(
        0.95, color="red", lw=0.9, ls="--", alpha=0.7,
        label="acceptance (SI=0.95)")
    ax.axhline(1.0, color="black", lw=0.6, ls=":", alpha=0.6)
    ax.set_xlabel("optimiser iteration")
    ax.set_ylabel("selectivity index")
    ax.set_ylim(-0.05, 1.08)
    ax.text(-0.14, 1.03, "(b)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=15, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3, lw=0.5)

    fig.legend(
        panel_a_handles + [accept_line],
        panel_a_labels + ["acceptance (SI=0.95)"],
        loc="lower center", ncol=len(species_present) + 1,
        bbox_to_anchor=(0.5, -0.02), fontsize=11,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out = FIGDIR / "fig_duke_optimization_convergence.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.name}")


# ─── orchestration ────────────────────────────────────────────────────────────
ALL = {
    "overview":       fig_overview,
    "validation":     fig_validation_4models,
    "waveforms":      fig_validation_waveforms,
    "traces":         fig_traces_4models,
    "phenomena":      fig_phenomena,
    "scaling":        fig_scaling_allmodels,
    "selectivity":    fig_selectivity_summary,
    "convergence":    fig_optimization_convergence,
    "mixed_landscape":fig_mixed_diameter_landscape,
    "xsections":      fig_selectivity_xsections,
    "xsections_hard": fig_selectivity_xsections_hard,
    # Duke (FEM-Ve) figures — coexist with the synthetic ones above.
    "duke_xsections":      fig_duke_selectivity_xsections,
    "duke_xsections_hard": fig_duke_selectivity_xsections_hard,
    "duke_selectivity":    fig_duke_selectivity_summary,
    "duke_convergence":    fig_duke_optimization_convergence,
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
