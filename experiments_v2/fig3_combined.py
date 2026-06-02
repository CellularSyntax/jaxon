"""Compose the four emergent-phenomena panels into a single Fig-3 figure.

Reads the JSON outputs from `dc_block.py`, `khz_block.py`, `ap_collision.py`,
and `spike_desync.py`, and produces a single 2×2 figure that matches the
layout of Hussain et al. *Nat. Commun.* 15:7597 (2024) Fig 3:

    +------------------------------+------------------------------+
    | a   Multiple conduction      | b   Kilohertz frequency      |
    |     responses to cathodic    |     population response      |
    |     stimulation              |                              |
    +------------------------------+------------------------------+
    | c   AP collision             | d   State-dependent effects  |
    |                              |     of stimulation           |
    +------------------------------+------------------------------+

Each panel is itself a sub-grid (rows × cols depend on the data).
Missing JSONs are skipped gracefully.

Run from project root after the four phenomena scripts have produced data:
    python experiments_v2/fig3_combined.py
"""

from __future__ import annotations

import sys
import pathlib
import json

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = ROOT / "outputs" / "fig3_combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── data loaders ─────────────────────────────────────────────────────────────

def _load_json(path: pathlib.Path) -> dict | None:
    if not path.exists():
        print(f"[fig3] WARNING: missing {path} — skipping that panel")
        return None
    with open(path) as f:
        return json.load(f)


def _panel_a_dc_block(subfig):
    """Panel a — Multiple conduction responses to cathodic stimulation."""
    data = _load_json(ROOT / "outputs" / "dc_block" / "data_dc_block.json")
    if data is None:
        subfig.suptitle("a — DC block (no data)", fontsize=12, fontweight="bold",
                         x=0.02, ha="left", y=0.98)
        return
    snap_ts   = data["snapshot_t_ms"]
    n_rows    = len(data["results"])
    n_cols    = len(snap_ts)
    axes = subfig.subplots(n_rows, n_cols, sharey=True, sharex=True)

    for r, res in enumerate(data["results"]):
        snaps = np.array(res["snapshots_nodes_mV"])
        amp   = res["amp_mA"]; fac = res["amp_factor"]
        for c, t in enumerate(snap_ts):
            ax = axes[r, c]
            n_nodes_r = snaps.shape[1]
            # Convert node # to mm (approximate via fiber length / n_nodes).
            # Hussain uses "position along axon (mm)".  Without geometry here
            # we plot vs node # but label it accordingly.
            ax.plot(np.arange(n_nodes_r), snaps[c], color="C0", lw=1.4)
            ax.axhline(-80, color="gray", lw=0.4, ls=":")
            ax.set_ylim(-110, 130)
            if r == 0:
                ax.set_title(f"t = {t} ms", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{amp:.3f} mA\nV$_m$ (mV)", fontsize=8)
            if r == n_rows - 1:
                ax.set_xlabel("node #", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3, lw=0.4)
    subfig.suptitle("a   Multiple conduction responses to cathodic stimulation",
                    fontsize=11, fontweight="bold", x=0.02, ha="left", y=1.0)


def _panel_b_khz_block(subfig):
    """Panel b — Kilohertz frequency population response."""
    data = _load_json(ROOT / "outputs" / "khz_block" / "data_khz_block.json")
    if data is None:
        subfig.suptitle("b — kHz block (no data)", fontsize=12, fontweight="bold",
                         x=0.02, ha="left", y=0.98)
        return

    # Detect format: new (per_diameter) vs legacy (results)
    if "per_diameter" not in data:
        subfig.suptitle("b   kHz block (legacy single-diameter format)",
                         fontsize=11, fontweight="bold", x=0.02, ha="left", y=1.0)
        ax = subfig.subplots(1, 1)
        ax.text(0.5, 0.5, "Re-run khz_block.py to get the\n3-diameter population view",
                 ha="center", va="center", fontsize=10, transform=ax.transAxes)
        return

    diams = data["diameters_um"]
    freqs = data["freqs_khz"]
    amp_max = data["amp_max_mA"]
    n_rows, n_cols = len(freqs), len(diams)
    axes = subfig.subplots(n_rows, n_cols, sharey=True)

    for c, D in enumerate(diams):
        dia_data = next(d for d in data["per_diameter"] if d["diameter_um"] == D)
        baseline_mean = float(np.mean(dia_data["n_aps_baseline"]))
        for r, f_khz in enumerate(freqs):
            ax = axes[r, c]
            d = dia_data["results_by_freq"][str(f_khz) if str(f_khz) in dia_data["results_by_freq"]
                                            else float(f_khz)]
            amps = np.array(d["amps_mA"])
            mean = np.array(d["mean"])
            ci   = np.array(d["ci95_half"])
            ax.plot(amps, mean, color="C1", lw=1.4)
            ax.fill_between(amps, mean - ci, mean + ci, color="C1", alpha=0.25)
            ax.axhline(baseline_mean, color="gray", lw=0.5, ls="--")
            if r == 0:
                ax.set_title(f"{D} µm", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{int(f_khz)} kHz\n# APs", fontsize=8)
            if r == n_rows - 1:
                ax.set_xlabel("stim amp (mA)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3, lw=0.4)
            ax.set_ylim(bottom=-1)
            ax.set_xlim(0, amp_max[str(D)] if isinstance(amp_max, dict) and str(D) in amp_max
                          else amp_max[D] if isinstance(amp_max, dict) and D in amp_max
                          else max(amps))
    subfig.suptitle("b   Kilohertz frequency population response",
                    fontsize=11, fontweight="bold", x=0.02, ha="left", y=1.0)


def _panel_c_ap_collision(subfig):
    """Panel c — AP collision."""
    data = _load_json(ROOT / "outputs" / "ap_collision" / "data_ap_collision.json")
    if data is None:
        subfig.suptitle("c — AP collision (no data)", fontsize=12, fontweight="bold",
                         x=0.02, ha="left", y=0.98)
        return
    snap_ts = data["snapshot_t_ms"]
    n_rows = len(data["results"])
    n_cols = len(snap_ts)
    axes = subfig.subplots(n_rows, n_cols, sharey=True, sharex=True)

    for r, res in enumerate(data["results"]):
        D = res["diameter_um"]
        snaps = np.array(res["snapshots_nodes_mV"])
        n_nodes_r = snaps.shape[1]
        for c, t in enumerate(snap_ts):
            ax = axes[r, c]
            ax.plot(np.arange(n_nodes_r), snaps[c], color="C0", lw=1.4)
            ax.axhline(-80, color="gray", lw=0.4, ls=":")
            ax.set_xlim(0, n_nodes_r - 1)
            ax.set_ylim(-90, 50)
            if r == 0:
                ax.set_title(f"t = {t} ms", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{D} µm\nV$_m$ (mV)", fontsize=8)
            if r == n_rows - 1:
                ax.set_xlabel("node #", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3, lw=0.4)
    subfig.suptitle("c   AP collision", fontsize=12, fontweight="bold",
                    x=0.02, ha="left", y=0.98)


def _panel_d_spike_desync(subfig):
    """Panel d — State-dependent effects of stimulation."""
    data = _load_json(ROOT / "outputs" / "spike_desync" / "data_spike_desync.json")
    if data is None:
        subfig.suptitle("d — Spike desync (no data)", fontsize=12, fontweight="bold",
                         x=0.02, ha="left", y=0.98)
        return

    # Current spike_desync.py is single-diameter; collapse to 1-row plot.
    ax = subfig.subplots(1, 1)
    cmap = plt.get_cmap("viridis")
    for i, (f_str, r) in enumerate(data["results"].items()):
        c = cmap(i / max(len(data["results"]) - 1, 1))
        ax.plot(r["amps_mA"], r["sync"], "o-", color=c, lw=1.5, ms=5,
                 label=f"{f_str} Hz stim")
    ax.set_xlabel("stim amplitude (mA)", fontsize=9)
    ax.set_ylabel("SPIKE-synchronization", fontsize=9)
    ax.set_title(f"D = {data['diameter_um']} µm, IFR = {data['ifr_hz']} Hz",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, lw=0.4)
    ax.set_ylim(-0.05, 1.05)
    subfig.suptitle("d   State-dependent effects of stimulation",
                    fontsize=11, fontweight="bold", x=0.02, ha="left", y=1.0)


def main():
    print("[fig3] Composing Fig 3 from panel data ...")
    fig = plt.figure(figsize=(18, 16), constrained_layout=True)
    subfigs = fig.subfigures(2, 2, wspace=0.04, hspace=0.04)

    _panel_a_dc_block(   subfigs[0, 0])
    _panel_b_khz_block(  subfigs[0, 1])
    _panel_c_ap_collision(subfigs[1, 0])
    _panel_d_spike_desync(subfigs[1, 1])
    path = OUT_DIR / "fig3_combined.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
