"""Compose the four emergent-phenomena panels into a single Fig-3 figure.

Reads the JSON outputs from `dc_block.py`, `khz_block.py`, `ap_collision.py`,
and `spike_desync.py`, and produces a single 2×2 figure that matches the
layout of Hussain et al. *Nat. Commun.* 15:7597 (2024) Fig 3.

In all panels: PyFibers/NEURON is blue solid, jaxfibers is orange dashed
(matching Hussain's NEURON / S-MF colour convention).

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

V_REST = -80.0


def _load_json(path: pathlib.Path) -> dict | None:
    if not path.exists():
        print(f"[fig3] WARNING: missing {path} - skipping that panel")
        return None
    with open(path) as f:
        return json.load(f)


def _panel_a_dc_block(subfig):
    """Panel a — Multiple conduction responses (waterfall, jax + pyfibers)."""
    data = _load_json(ROOT / "outputs" / "dc_block" / "data_dc_block.json")
    if data is None:
        subfig.suptitle("a   DC block (no data)", fontsize=11, fontweight="bold",
                         x=0.02, ha="left", y=1.0)
        return
    n_amps = len(data["results"])
    axes = subfig.subplots(1, n_amps, sharey=True)
    if n_amps == 1:
        axes = [axes]
    npos = np.array(data["node_pos_mm"])
    inter_node = npos[1] - npos[0]
    v_range = 150.0
    scale = (inter_node * 0.7) / v_range

    for c, r in enumerate(data["results"]):
        ax = axes[c]
        jax_t  = np.array(r["jax_t_ms"])
        jax_vm = np.array(r["jax_vm_nodes"])     # [N_STEPS, n_nodes]
        pf_t   = np.array(r["pf_t_ms"])
        pf_vm  = np.array(r["pf_vm_nodes"])      # [n_t, n_nodes]
        for i, node_y in enumerate(npos):
            ax.plot(pf_t,  node_y + (pf_vm[:, i] - V_REST) * scale,
                    color="C0", lw=0.8, alpha=0.9,
                    label="PyFibers" if (c == 0 and i == 0) else None)
            ax.plot(jax_t, node_y + (jax_vm[:, i] - V_REST) * scale,
                    color="C1", lw=0.7, ls="--", alpha=0.85,
                    label="jaxfibers" if (c == 0 and i == 0) else None)
        ax.set_xlim(0, jax_t[-1])
        ax.set_xlabel("time (ms)", fontsize=8)
        ax.set_title(f"{r['amp_mA']:.3f} mA  ({r['amp_factor']}× thr)",
                      fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, lw=0.4)
        if c == 0:
            ax.set_ylabel("node position  (mm)", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
    subfig.suptitle("a   Multiple conduction responses to cathodic stimulation",
                    fontsize=11, fontweight="bold", x=0.02, ha="left", y=1.0)


def _panel_b_khz_block(subfig):
    """Panel b — kHz block, jax + pyfibers."""
    data = _load_json(ROOT / "outputs" / "khz_block" / "data_khz_block.json")
    if data is None or "per_diameter" not in data:
        subfig.suptitle("b   kHz block (no/legacy data)",
                         fontsize=11, fontweight="bold", x=0.02, ha="left", y=1.0)
        return
    diams = data["diameters_um"]
    freqs = data["freqs_khz"]
    amp_max = data["amp_max_mA"]
    axes = subfig.subplots(len(freqs), len(diams), sharey=True)
    for c, D in enumerate(diams):
        dia_data = next(d for d in data["per_diameter"] if d["diameter_um"] == D)
        for r, f_khz in enumerate(freqs):
            ax = axes[r, c]
            key = str(f_khz) if str(f_khz) in dia_data["results_by_freq"] else float(f_khz)
            d = dia_data["results_by_freq"][key]
            amps = np.array(d["amps_mA"])
            mp = np.array(d.get("mean_pf",  d.get("mean", [])))
            cp = np.array(d.get("ci95_half_pf",  d.get("ci95_half", [])))
            mj = np.array(d.get("mean_jax", d.get("mean", [])))
            cj = np.array(d.get("ci95_half_jax", d.get("ci95_half", [])))
            if len(mp):
                ax.plot(amps, mp, color="C0", lw=1.3,
                         label="PyFibers" if (r == 0 and c == 0) else None)
                if len(cp):
                    ax.fill_between(amps, mp - cp, mp + cp, color="C0", alpha=0.2)
            if len(mj):
                ax.plot(amps, mj, color="C1", lw=1.0, ls="--",
                         label="jaxfibers" if (r == 0 and c == 0) else None)
                if len(cj):
                    ax.fill_between(amps, mj - cj, mj + cj, color="C1", alpha=0.2)
            if r == 0:
                ax.set_title(f"{D} µm", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{int(f_khz)} kHz\n# APs", fontsize=8)
            if r == len(freqs) - 1:
                ax.set_xlabel("stim amp (mA)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3, lw=0.4)
            ax.set_ylim(bottom=-1)
            x_max = amp_max[str(D)] if isinstance(amp_max, dict) and str(D) in amp_max \
                      else (amp_max[D] if isinstance(amp_max, dict) and D in amp_max
                              else max(amps))
            ax.set_xlim(0, x_max)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="upper right")
    subfig.suptitle("b   Kilohertz frequency population response",
                    fontsize=11, fontweight="bold", x=0.02, ha="left", y=1.0)


def _panel_c_ap_collision(subfig):
    """Panel c — AP collision (jax + pyfibers overlaid)."""
    data = _load_json(ROOT / "outputs" / "ap_collision" / "data_ap_collision.json")
    if data is None:
        subfig.suptitle("c   AP collision (no data)",
                         fontsize=11, fontweight="bold", x=0.02, ha="left", y=1.0)
        return
    snap_ts = data["snapshot_t_ms"]
    n_rows = len(data["results"])
    n_cols = len(snap_ts)
    axes = subfig.subplots(n_rows, n_cols, sharey=True, sharex=True)
    for r, res in enumerate(data["results"]):
        D = res["diameter_um"]
        snaps_jax = np.array(res["snapshots_nodes_mV"])
        snaps_pf  = np.array(res.get("snapshots_nodes_pf_mV", []))
        n_nodes_r = snaps_jax.shape[1]
        for c, t in enumerate(snap_ts):
            ax = axes[r, c]
            if len(snaps_pf):
                ax.plot(np.arange(n_nodes_r), snaps_pf[c], color="C0", lw=1.3,
                         label="PyFibers" if (r == 0 and c == 0) else None)
            ax.plot(np.arange(n_nodes_r), snaps_jax[c], color="C1", lw=1.0, ls="--",
                     label="jaxfibers" if (r == 0 and c == 0) else None)
            ax.axhline(V_REST, color="gray", lw=0.4, ls=":")
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
            if r == 0 and c == 0 and len(snaps_pf):
                ax.legend(fontsize=7, loc="upper right")
    subfig.suptitle("c   AP collision",
                    fontsize=11, fontweight="bold", x=0.02, ha="left", y=1.0)


def _panel_d_spike_desync(subfig):
    """Panel d — State-dependent effects (3×3 grid, line style = stim freq)."""
    data = _load_json(ROOT / "outputs" / "spike_desync" / "data_spike_desync.json")
    if data is None:
        subfig.suptitle("d   Spike desync (no data)",
                         fontsize=11, fontweight="bold", x=0.02, ha="left", y=1.0)
        return
    # Old single-panel format: detect and skip.
    if "results" in data and isinstance(data["results"], dict) and \
       not any(k.startswith("D") for k in data["results"]):
        # Legacy single-diameter format
        subfig.suptitle("d   Spike desync (legacy single-diameter)",
                         fontsize=11, fontweight="bold", x=0.02, ha="left", y=1.0)
        ax = subfig.subplots(1, 1)
        ax.text(0.5, 0.5, "Re-run spike_desync.py for the\n3×3 grid + pyfibers comparison",
                 ha="center", va="center", transform=ax.transAxes)
        return

    diams = data["diameters_um"]
    ifrs  = data["ifr_hz"]
    fstims = data["f_stim_hz"]
    amp_range = data["amp_range_mA"]
    linestyles = {30: "-", 50: "--", 100: ":"}

    axes = subfig.subplots(len(ifrs), len(diams))
    for c, D in enumerate(diams):
        for r, ifr in enumerate(ifrs):
            ax = axes[r, c]
            for f in fstims:
                key = f"D{D}_ifr{ifr}_f{f}"
                if key not in data["results"]:
                    continue
                cell = data["results"][key]
                amps = np.array(cell["amps_mA"])
                mp = np.array(cell["mean_pf"]);   cp = np.array(cell["ci95_half_pf"])
                mj = np.array(cell["mean_jax"]);  cj = np.array(cell["ci95_half_jax"])
                ax.plot(amps, mp, color="C0", lw=1.2, ls=linestyles[f],
                         label=f"PF {f} Hz" if (r == 0 and c == 0) else None)
                ax.fill_between(amps, mp - cp, mp + cp, color="C0", alpha=0.12)
                ax.plot(amps, mj, color="C1", lw=0.9, ls=linestyles[f],
                         label=f"jax {f} Hz" if (r == 0 and c == 0) else None)
                ax.fill_between(amps, mj - cj, mj + cj, color="C1", alpha=0.12)
            ax.set_ylim(0, 1.05)
            xr = amp_range[str(D)] if isinstance(amp_range, dict) and str(D) in amp_range \
                 else amp_range[D] if isinstance(amp_range, dict) and D in amp_range \
                 else (0, 1)
            ax.set_xlim(*xr)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3, lw=0.4)
            if r == 0:
                ax.set_title(f"{D} µm", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{int(ifr)} Hz\nSPIKE-sync", fontsize=8)
            if r == len(ifrs) - 1:
                ax.set_xlabel("stim amp (mA)", fontsize=8)
            if r == 0 and c == 0:
                ax.legend(fontsize=6, loc="lower left", ncol=2)
    subfig.suptitle("d   State-dependent effects of stimulation",
                    fontsize=11, fontweight="bold", x=0.02, ha="left", y=1.0)


def main():
    print("[fig3] Composing Fig 3 from panel data ...")
    fig = plt.figure(figsize=(18, 16), constrained_layout=True)
    subfigs = fig.subfigures(2, 2, wspace=0.04, hspace=0.04)
    _panel_a_dc_block(    subfigs[0, 0])
    _panel_b_khz_block(   subfigs[0, 1])
    _panel_c_ap_collision(subfigs[1, 0])
    _panel_d_spike_desync(subfigs[1, 1])
    path = OUT_DIR / "fig3_combined.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
