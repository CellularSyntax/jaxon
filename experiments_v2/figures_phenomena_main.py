"""Figure 2: Propagation phenomena — PyFibers vs JAXON.

One row per fiber model (MRG, Sweeney, Sundt, Rattay), three panels each:
  AP propagation     — waterfall line-plot (y=position, x=time, one trace per node)
  kHz frequency block — Vm vs time at distal node, 4 amplitude levels
  AP collision        — Vm vs node # at 5 time points × 4 diameters

Models are configured in MODELS; each reads outputs/<phenomenon><suffix>/.
Rows whose data is missing render as placeholders, so the figure builds before
every model's sims have been generated.

Run from project root:
    python -m experiments_v2.figures_phenomena_main
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "manuscript" / "figures" / "main"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_PF   = "#0072B2"   # blue       — PyFibers (original; matches NEURON blue in fig1/3)
C_JAX  = "#D55E00"  # vermillion — JAXON   (Wong palette; replaces shrill amber)
C_REST = "#555555"
V_REST = -80.0
VM_LIM = (-92, 55)

FS    = 9
FS_SM = 8

mpl.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size":          FS,
    "axes.labelsize":     FS,
    "axes.titlesize":     FS_SM,
    "xtick.labelsize":    FS_SM,
    "ytick.labelsize":    FS_SM,
    "legend.fontsize":    FS_SM,
    "axes.linewidth":     0.5,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "xtick.major.size":   2.5,
    "ytick.major.size":   2.5,
    "lines.linewidth":    0.8,
    "patch.linewidth":    0.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          False,
    "savefig.dpi":        600,
    "savefig.bbox":       "tight",
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "pdf.fonttype":       42,
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(path: Path) -> dict | None:
    if not path.exists():
        print(f"[fig2] missing: {path}", file=sys.stderr)
        return None
    with open(path) as f:
        return json.load(f)


def _heading(fig, x: float, y: float, letter: str, title: str) -> None:
    """Panel heading at absolute figure-fraction coordinates."""
    fig.text(x, y, letter, transform=fig.transFigure,
             fontsize=15, fontweight="bold", va="bottom", ha="left", clip_on=False)
    fig.text(x + 0.022, y, title, transform=fig.transFigure,
             fontsize=11, va="bottom", ha="left", clip_on=False)


# ── Panel a: AP propagation waterfall ─────────────────────────────────────────

_WATERFALL_T_MAX = 3.5   # ms


def _panel_a(gs_cell, fig, dc_data: dict | None,
             v_rest: float = V_REST, t_max: float = _WATERFALL_T_MAX,
             stride: int = 2) -> plt.Axes:
    ax = fig.add_subplot(gs_cell)

    if dc_data is None:
        ax.text(0.5, 0.5, "data missing", ha="center", va="center",
                transform=ax.transAxes, fontsize=FS_SM, color="#AAAAAA")
        return ax

    res = next((r for r in dc_data["results"]
                if abs(r["amp_factor"] - 1.1) < 0.05), dc_data["results"][1])

    t_jax  = np.array(res["jax_t_ms"])
    t_pf   = np.array(res["pf_t_ms"])
    vm_jax = np.array(res["jax_vm_nodes"])   # [n_t, n_nodes]
    vm_pf  = np.array(res["pf_vm_nodes"])
    pos    = np.array(dc_data["node_pos_mm"])

    jm = t_jax <= t_max
    pm = t_pf  <= t_max
    t_jax,  vm_jax = t_jax[jm],  vm_jax[jm]
    t_pf,   vm_pf  = t_pf[pm],   vm_pf[pm]

    n_nodes     = len(pos)
    plot_idx    = list(range(0, n_nodes, stride))
    n_plot      = len(plot_idx)
    # Scale so AP peak occupies ~50% of the inter-plotted-trace spacing,
    # regardless of how densely the nodes are physically packed.
    plot_spacing = (pos[-1] - pos[0]) / max(n_plot - 1, 1)
    y_scale      = plot_spacing * 0.5 / (40.0 - v_rest)

    for i in plot_idx:
        y_off     = pos[i]
        trace_pf  = (vm_pf[:,  i] - v_rest) * y_scale
        trace_jax = (vm_jax[:, i] - v_rest) * y_scale
        ax.plot(t_pf,  y_off + trace_pf,  color=C_PF,  lw=0.7, alpha=0.9,
                label="PyFibers" if i == 0 else None)
        ax.plot(t_jax, y_off + trace_jax, color=C_JAX, lw=0.7, ls="--", alpha=0.9,
                label="JAXON"    if i == 0 else None)

    ax.set_xlabel("time (ms)")
    ax.set_ylabel("position (mm)")
    ax.set_xlim(0, t_max)
    ax.set_ylim(pos[0] - plot_spacing * 0.5, pos[-1] + plot_spacing * 0.5)

    # Short horizontal arrow pointing right to the stimulation site
    stim_pos = pos[len(pos) // 2]
    stim_t   = float(dc_data.get("delay_ms", 0.5))
    ax.annotate("",
                xy=(stim_t, stim_pos),
                xytext=(stim_t - t_max * 0.07, stim_pos),
                arrowprops=dict(arrowstyle="-|>", color="black",
                                lw=1.6, mutation_scale=16))

    # Legend above the axes, right-aligned, single row
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02),
              ncol=2, frameon=False, fontsize=FS_SM, borderaxespad=0)
    return ax


# ── Panel b: DC depolarization block ──────────────────────────────────────────

WIN_ON  = 50.0     # ms — block field on  (shaded band)
WIN_OFF = 100.0    # ms — block field off
_SKIP   = 20


def _panel_b(gs_cell, fig, khz_data: dict | None,
             v_rest: float = V_REST) -> plt.Axes:
    # Invisible outer axis so we can return a reference that spans the full panel
    ax_outer = fig.add_subplot(gs_cell)
    ax_outer.axis("off")

    gs_inner = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs_cell, hspace=0.08)
    axes = [fig.add_subplot(gs_inner[r, 0]) for r in range(4)]

    if khz_data is None:
        for ax in axes:
            ax.axis("off")
        return ax_outer

    vm_lo = round(v_rest / 10) * 10 - 12   # headroom below rest
    vm_lim = (vm_lo, 55)
    ytick_rest = round(v_rest / 10) * 10    # nearest 10 mV for tick label

    for r, res in enumerate(khz_data["results"]):
        ax   = axes[r]
        _t   = np.array(res["t_pf_ms"])
        _vpf = np.array(res["vm_pf_90"])
        _vjx = np.array(res["vm_jax_90"])
        _n   = min(len(_t), len(_vpf), len(_vjx))
        t    = _t[:_n:_SKIP]
        vpf  = _vpf[:_n:_SKIP]
        vjx  = _vjx[:_n:_SKIP]

        ax.axvspan(WIN_ON, WIN_OFF, color="#FFCCCC", alpha=0.45, zorder=0, lw=0)
        ax.plot(t, vpf, color=C_PF,  lw=1.0)
        ax.plot(t, vjx, color=C_JAX, lw=0.9, ls="--")
        ax.axhline(v_rest, color=C_REST, lw=0.3, ls=":", zorder=0)

        ax.set_ylim(*vm_lim)
        ax.set_xlim(0, 150)
        ax.set_yticks([ytick_rest, 0])

        # Amplitude as in-plot annotation
        ax.text(0.03, 0.94, f"{res['amp_mA']:.1f} mA",
                transform=ax.transAxes, ha="left", va="top", fontsize=FS_SM)

        ax.set_ylabel("V$_m$ (mV)")
        if r < 3:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("time (ms)")

    # Single-row legend placed just above the top subplot
    handles = [
        Line2D([0], [0], color=C_PF,  lw=1.4, ls="-",  label="PyFibers"),
        Line2D([0], [0], color=C_JAX, lw=1.4, ls="--", label="JAXON"),
    ]
    # bbox y=1.00 → legend sits flush above axes[0] top, below the panel heading
    axes[0].legend(handles=handles, loc="lower left",
                   bbox_to_anchor=(0.0, 1.00), ncol=2, frameon=False,
                   fontsize=FS_SM, borderaxespad=0)

    return ax_outer


# ── Panel c: AP collision ──────────────────────────────────────────────────────

def _panel_c(gs_cell, fig, coll_data: dict | None,
             v_rest: float = V_REST) -> plt.Axes:
    ax_outer = fig.add_subplot(gs_cell)
    ax_outer.axis("off")

    if coll_data is None:
        return ax_outer

    snap_ts = list(coll_data["snapshot_t_ms"])
    results = coll_data["results"]
    n_rows  = len(results)
    n_cols  = len(snap_ts)

    gs_inner = gridspec.GridSpecFromSubplotSpec(
        n_rows, n_cols, subplot_spec=gs_cell,
        hspace=0.08, wspace=0.04,
    )

    for r, res in enumerate(results):
        D         = res["diameter_um"]
        snaps_jax = np.array(res["snapshots_nodes_mV"])
        pf_raw    = res.get("snapshots_nodes_pf_mV", [])
        snaps_pf  = np.array(pf_raw) if pf_raw else None
        nodes     = np.arange(snaps_jax.shape[1])

        for c, t_snap in enumerate(snap_ts):
            ax = fig.add_subplot(gs_inner[r, c])

            # Full box + square aspect for collision plots
            for spine in ax.spines.values():
                spine.set_visible(True)
            ax.set_box_aspect(1)

            if snaps_pf is not None:
                ax.plot(nodes, snaps_pf[c], color=C_PF,  lw=1.0)
            ax.plot(nodes, snaps_jax[c],    color=C_JAX, lw=0.9, ls="--")
            ax.axhline(v_rest, color=C_REST, lw=0.3, ls=":", zorder=0)
            ax.set_ylim(round(v_rest / 10) * 10 - 12, 55)
            ax.set_xlim(0, nodes[-1])
            ax.set_yticks([round(v_rest / 10) * 10, 0])

            # Column header inside the box (frees space above for the legend)
            if r == 0:
                ax.text(0.5, 0.97, f"t = {t_snap} ms",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=FS_SM)
            if c == 0:
                ax.set_ylabel(f"{D} µm")
            else:
                ax.set_yticklabels([])
            if r == n_rows - 1:
                ax.set_xlabel("node #")
            else:
                ax.set_xticklabels([])


    # Legend above the panel, top-right — same style as panels a and b
    leg_handles = [
        Line2D([0], [0], color=C_PF,  lw=1.0, label="PyFibers"),
        Line2D([0], [0], color=C_JAX, lw=0.9, ls="--", label="JAXON"),
    ]
    ax_outer.legend(handles=leg_handles, loc="lower right",
                    bbox_to_anchor=(1.0, 1.02), ncol=2, frameon=False,
                    fontsize=FS_SM, borderaxespad=0)

    return ax_outer


# ── Main ──────────────────────────────────────────────────────────────────────

# One row per fiber model.  ``suffix`` selects the outputs/<phenomenon><suffix>/
# directories; ``v_rest``/``t_max``/``stride`` tune the per-model panel display.
# A-fibers (myelinated) first, then C-fibers (unmyelinated), matching fig1.
MODELS = [
    dict(key="mrg",     label="MRG\n(A-fiber)",     suffix="",         v_rest=-80.0, t_max=3.5, stride=2),
    dict(key="sweeney", label="Sweeney\n(A-fiber)", suffix="_sweeney", v_rest=-80.0, t_max=3.5, stride=2),
    dict(key="sundt",   label="Sundt\n(C-fiber)",   suffix="_sundt",   v_rest=-60.0, t_max=8.0, stride=2),
    dict(key="rattay",  label="Rattay\n(C-fiber)",  suffix="_rattay",  v_rest=-70.0, t_max=8.0, stride=2),
]


def _load_model(suffix: str) -> tuple:
    """Load (propagation, DC-block, collision) JSONs for one model suffix.

    Columns: AP propagation (legacy 'dc_block' dir), DC depolarization block
    ('depol_block'), AP collision.  (kHz block moved to a supplementary figure.)
    """
    def _p(stem: str) -> Path:
        name = f"{stem}{suffix}"
        return ROOT / "outputs" / name / f"data_{name}.json"
    return (_load(_p("dc_block")), _load(_p("depol_block")), _load(_p("ap_collision")))


def main() -> int:
    # ── Column layout (identical for every row) ───────────────────────────────
    FIG_W   = 14.0
    L, R    = 0.06, 0.98
    col_gap = 0.05
    panel_w = (R - L) - 2 * col_gap

    a_w = panel_w * 0.35
    b_w = panel_w * 0.23

    a_l = L;              a_r = a_l + a_w
    b_l = a_r + col_gap;  b_r = b_l + b_w
    c_l = b_r + col_gap;  c_r = R
    c_width = c_r - c_l

    # ── Vertical layout: stack one band per model ─────────────────────────────
    # Panel c (4 rows × 5 cols of square subplots) fixes the physical row
    # height; derived in inches so adding rows just grows the figure rather
    # than squashing the panels.
    n_rows       = len(MODELS)
    row_panel_in = c_width * 4.24 * FIG_W / 5.16   # square-subplot constraint
    row_head_in  = 1.05                            # heading + legend band per row
    top_in, bot_in = 0.30, 0.55
    FIG_H = bot_in + top_in + n_rows * (row_panel_in + row_head_in)

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    def _gs(left, right, bot, top):
        return gridspec.GridSpec(1, 1, figure=fig,
                                 left=left, right=right, bottom=bot, top=top)

    letters = "abcdefghijklmnopqrstuvwxyz"
    label_x = L - 0.04

    for i, m in enumerate(MODELS):
        dc_data, khz_data, coll_data = _load_model(m["suffix"])

        # Band i from the top: heading band sits above the panel band.
        block_top_in = FIG_H - top_in - i * (row_panel_in + row_head_in)
        panel_top_in = block_top_in - row_head_in
        panel_bot_in = panel_top_in - row_panel_in
        T = panel_top_in / FIG_H
        B = panel_bot_in / FIG_H

        _panel_a(_gs(a_l, a_r, B, T)[0, 0], fig, dc_data,
                 v_rest=m["v_rest"], t_max=m["t_max"], stride=m["stride"])
        _panel_b(_gs(b_l, b_r, B, T)[0, 0], fig, khz_data, v_rest=m["v_rest"])
        _panel_c(_gs(c_l, c_r, B, T)[0, 0], fig, coll_data, v_rest=m["v_rest"])

        # Headings for this row (a/b/c, d/e/f, ...).
        y = T + 0.025 * (12.0 / FIG_H)   # keep ~constant physical offset
        for col_l, lett, title in [
            (a_l, letters[3 * i + 0], "AP propagation"),
            (b_l, letters[3 * i + 1], "DC block"),
            (c_l, letters[3 * i + 2], "AP collision"),
        ]:
            _heading(fig, col_l, y, lett, title)

        # Row label at the left margin.
        fig.text(label_x, (B + T) / 2, m["label"],
                 transform=fig.transFigure, fontsize=FS_SM,
                 va="center", ha="right", rotation=90,
                 color="#555555", style="italic", clip_on=False)

    for ext in (".png", ".svg"):
        p = OUT_DIR / f"fig2_phenomena{ext}"
        fig.savefig(p, dpi=600 if ext == ".png" else None)
        print(f"  -> {p}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
