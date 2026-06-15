"""Compose the four emergent-phenomena panels into a single Fig-3 figure.

Reads the JSON outputs produced by:
  dc_block.py       → Panel a  (multiple conduction responses, spatial snapshots)
  khz_population.py → Panel b  (kHz frequency population response)
  ap_collision.py   → Panel c  (AP collision spatial snapshots)
  spike_desync.py   → Panel d  (state-dependent effects, SPIKE synchronisation)

In all panels NEURON / PyFibers is blue solid and jaxfibers / S-MF is orange
dashed (matching Hussain et al. Nat. Commun. 15:7597 (2024) Fig 3 colour
convention).

Missing JSONs are skipped gracefully (placeholder text in that panel).

Run from project root after the four data scripts have run:
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

# Colourblind-safe palette (Wong / Okabe — matches Hussain Fig 3)
C_NEURON  = "#0072B2"   # blue  — NEURON / PyFibers
C_SMF     = "#E69F00"   # amber — S-MF   / jaxfibers
C_GREY    = "#555555"


def _load_json(path: pathlib.Path) -> dict | None:
    if not path.exists():
        print(f"[fig3] WARNING: missing {path} — skipping that panel")
        return None
    with open(path) as f:
        return json.load(f)


# ── Panel a — Multiple conduction responses (spatial snapshots) ─────────────

# Time points at which to extract spatial Vm snapshots (absolute time in ms).
# Change these if your dc_block.py uses a different TSTOP or DELAY.
_SNAP_TS_MS = [1.0, 1.5, 2.0, 2.5]


def _panel_a_dc_block(subfig):
    data = _load_json(ROOT / "outputs" / "dc_block" / "data_dc_block.json")
    if data is None:
        ax = subfig.subplots(1, 1)
        ax.text(0.5, 0.5, "dc_block data missing\nRun dc_block.py first",
                 ha="center", va="center", transform=ax.transAxes)
        return ax

    results   = data["results"]
    node_pos  = np.array(data["node_pos_mm"])
    snap_ts   = _SNAP_TS_MS
    n_rows    = len(results)
    n_cols    = len(snap_ts)
    axes = subfig.subplots(n_rows, n_cols, sharex=True, sharey=True,
                            gridspec_kw={"hspace": 0.08, "wspace": 0.04})

    for r, res in enumerate(results):
        jax_t  = np.array(res["jax_t_ms"])
        jax_vm = np.array(res["jax_vm_nodes"])   # [N_STEPS, n_nodes]
        pf_t   = np.array(res["pf_t_ms"])
        pf_vm  = np.array(res["pf_vm_nodes"])    # [n_t, n_nodes]

        for c, ts in enumerate(snap_ts):
            ax = axes[r, c]
            ji = int(np.argmin(np.abs(jax_t - ts)))
            pi = int(np.argmin(np.abs(pf_t  - ts)))

            ax.plot(node_pos, pf_vm[pi, :],
                     color=C_NEURON, lw=1.5,
                     label="PyFibers" if (r == 0 and c == 0) else None)
            ax.plot(node_pos, jax_vm[ji, :],
                     color=C_SMF, lw=1.2, ls="--",
                     label="JAXON" if (r == 0 and c == 0) else None)
            ax.axhline(V_REST, color=C_GREY, lw=0.4, ls=":", zorder=0)
            ax.set_ylim(-90, 50)
            ax.grid(alpha=0.25, lw=0.4)

            if r == 0:
                ax.set_title(f"t = {ts} ms")
            if c == 0:
                amp_label = f"{res['amp_mA']:.3f} mA"
                ax.set_ylabel(f"{amp_label}\nV$_m$ (mV)")
            if r == n_rows - 1:
                ax.set_xlabel("position along axon (mm)")

    axes[0, 0].legend(loc="upper right", frameon=False)
    return axes[0, 0]


# ── Panel b — kHz frequency population response ──────────────────────────────

def _panel_b_khz_population(subfig):
    # Prefer the new population data; fall back to the simple trace file.
    data = _load_json(ROOT / "outputs" / "khz_population" / "data_khz_population.json")
    if data is None:
        ax = subfig.subplots(1, 1)
        ax.text(0.5, 0.5,
                 "khz_population data missing\nRun khz_population.py first",
                 ha="center", va="center", transform=ax.transAxes)
        return ax

    diameters = data["diameters_um"]
    freqs_khz = data["freqs_khz"]
    results   = data["results"]
    n_rows    = len(freqs_khz)
    n_cols    = len(diameters)
    axes = subfig.subplots(n_rows, n_cols, sharex="col",
                            gridspec_kw={"hspace": 0.08, "wspace": 0.08})
    axes = np.atleast_2d(axes)

    for r, f in enumerate(freqs_khz):
        for c, D in enumerate(diameters):
            ax  = axes[r, c]
            key = f"D{D}_f{f}kHz"
            if key not in results:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                         transform=ax.transAxes)
                continue
            cell = results[key]
            amps = np.array(cell["amps_mA"])
            mj, cj = np.array(cell["mean_jax"]), np.array(cell["ci95_jax"])
            mp, cp = np.array(cell["mean_pf"]),  np.array(cell["ci95_pf"])

            ax.plot(amps, mp, color=C_NEURON, lw=1.5,
                     label="PyFibers" if (r == 0 and c == 0) else None)
            ax.fill_between(amps, mp - cp, mp + cp, color=C_NEURON, alpha=0.18)
            ax.plot(amps, mj, color=C_SMF, lw=1.4,
                     label="JAXON" if (r == 0 and c == 0) else None)
            ax.fill_between(amps, mj - cj, mj + cj, color=C_SMF, alpha=0.18)

            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.25, lw=0.4)

            if r == 0:
                ax.set_title(f"{D} µm")
            if c == 0:
                ax.set_ylabel(f"{f} kHz\n# APs")
            if r == n_rows - 1:
                ax.set_xlabel("stimulus amplitude (mA)")

    axes[0, 0].legend(loc="upper right", frameon=False)
    return axes[0, 0]


# ── Panel c — AP collision (spatial snapshots) ───────────────────────────────

def _panel_c_ap_collision(subfig):
    data = _load_json(ROOT / "outputs" / "ap_collision" / "data_ap_collision.json")
    if data is None:
        ax = subfig.subplots(1, 1)
        ax.text(0.5, 0.5, "ap_collision data missing\nRun ap_collision.py first",
                 ha="center", va="center", transform=ax.transAxes)
        return ax

    snap_ts = data["snapshot_t_ms"]
    n_rows  = len(data["results"])
    n_cols  = len(snap_ts)
    axes = subfig.subplots(n_rows, n_cols, sharey=True, sharex=True,
                            gridspec_kw={"hspace": 0.08, "wspace": 0.04})

    for r, res in enumerate(data["results"]):
        D             = res["diameter_um"]
        snaps_jax     = np.array(res["snapshots_nodes_mV"])   # [n_snap, n_nodes]
        snaps_pf_raw  = res.get("snapshots_nodes_pf_mV", [])
        snaps_pf      = np.array(snaps_pf_raw) if snaps_pf_raw else None
        n_nodes       = snaps_jax.shape[1]

        for c, t in enumerate(snap_ts):
            ax = axes[r, c]
            if snaps_pf is not None:
                ax.plot(np.arange(n_nodes), snaps_pf[c],
                         color=C_NEURON, lw=1.5,
                         label="PyFibers" if (r == 0 and c == 0) else None)
            ax.plot(np.arange(n_nodes), snaps_jax[c],
                     color=C_SMF, lw=1.2, ls="--",
                     label="JAXON" if (r == 0 and c == 0) else None)
            ax.axhline(V_REST, color=C_GREY, lw=0.4, ls=":", zorder=0)
            ax.set_xlim(0, n_nodes - 1)
            ax.set_ylim(-90, 50)
            ax.grid(alpha=0.25, lw=0.4)

            if r == 0:
                ax.set_title(f"t = {t} ms")
            if c == 0:
                ax.set_ylabel(f"{D} µm\nV$_m$ (mV)")
            if r == n_rows - 1:
                ax.set_xlabel("position along axon (node #)")

    if data["results"] and data.get("snapshot_t_ms"):
        axes[0, 0].legend(loc="upper right", frameon=False)
    return axes[0, 0]


# ── Panel d — State-dependent effects (SPIKE synchronisation) ────────────────

def _panel_d_spike_desync(subfig):
    data = _load_json(ROOT / "outputs" / "spike_desync" / "data_spike_desync.json")
    if data is None:
        ax = subfig.subplots(1, 1)
        ax.text(0.5, 0.5, "spike_desync data missing\nRun spike_desync.py first",
                 ha="center", va="center", transform=ax.transAxes)
        subfig.suptitle("d   State-dependent effects (no data)",
                         fontsize=11, fontweight="bold", x=0.02, ha="left")
        return

    # Guard against legacy single-diameter format.
    if "results" in data and isinstance(data["results"], dict) and \
       not any(k.startswith("D") for k in data["results"]):
        ax = subfig.subplots(1, 1)
        ax.text(0.5, 0.5,
                 "Legacy single-diameter format\nRe-run spike_desync.py",
                 ha="center", va="center", transform=ax.transAxes)
        subfig.suptitle("d   State-dependent effects (legacy data)",
                         fontsize=11, fontweight="bold", x=0.02, ha="left")
        return

    diams   = data["diameters_um"]
    ifrs    = data["ifr_hz"]
    fstims  = data["f_stim_hz"]
    amp_range = data["amp_range_mA"]
    ls_map  = {30: "-", 50: "--", 100: ":"}

    n_rows  = len(ifrs)
    n_cols  = len(diams)
    axes = subfig.subplots(n_rows, n_cols, sharex="col",
                            gridspec_kw={"hspace": 0.08, "wspace": 0.08})

    for c, D in enumerate(diams):
        for r, ifr in enumerate(ifrs):
            ax = axes[r, c]
            for f in fstims:
                key = f"D{D}_ifr{ifr}_f{f}"
                if key not in data["results"]:
                    continue
                cell = data["results"][key]
                amps = np.array(cell["amps_mA"])
                mp   = np.array(cell["mean_pf"]);  cp = np.array(cell["ci95_half_pf"])
                mj   = np.array(cell["mean_jax"]); cj = np.array(cell["ci95_half_jax"])
                ls   = ls_map.get(f, "-")
                ax.plot(amps, mp, color=C_NEURON, lw=1.3, ls=ls,
                         label=f"PF {f} Hz" if (r == 0 and c == 0) else None)
                ax.fill_between(amps, mp - cp, mp + cp, color=C_NEURON, alpha=0.12)
                ax.plot(amps, mj, color=C_SMF,    lw=1.0, ls=ls,
                         label=f"JAX {f} Hz" if (r == 0 and c == 0) else None)
                ax.fill_between(amps, mj - cj, mj + cj, color=C_SMF, alpha=0.12)

            ax.set_ylim(0, 1.05)
            # amp_range keys may be int or str depending on json serialisation
            xr = (amp_range.get(str(D)) or amp_range.get(D) or [None, None])
            if xr[0] is not None:
                ax.set_xlim(xr[0], xr[1])
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25, lw=0.4)

            if r == 0:
                ax.set_title(f"{D} µm", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{int(ifr)} Hz\nSPIKE sync.", fontsize=8)
            if r == n_rows - 1:
                ax.set_xlabel("stim amplitude (mA)", fontsize=8)

    axes[0, 0].legend(fontsize=6, loc="lower left", ncol=2, frameon=False)
    subfig.suptitle("d   State-dependent effects of stimulation",
                    fontsize=11, fontweight="bold", x=0.02, ha="left")


# ── Shared legend strips ─────────────────────────────────────────────────────

def _add_panel_legend(subfig, entries):
    """Add a compact bottom-of-panel legend using invisible dummy lines."""
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=col, lw=1.8, ls=ls, label=lab)
               for col, ls, lab in entries]
    subfig.legend(handles=handles, loc="lower center", ncol=len(entries),
                   frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("[fig3] Composing Fig 3 from panel data ...")
    fig = plt.figure(figsize=(22, 20), constrained_layout=True)

    # 2 × 2 outer grid; height_ratios give more room to panels with more rows.
    subfigs = fig.subfigures(2, 2, wspace=0.06, hspace=0.06,
                               height_ratios=[1.0, 1.0],
                               width_ratios=[1.0, 1.0])

    _panel_a_dc_block(    subfigs[0, 0])
    _panel_b_khz_population(subfigs[0, 1])
    _panel_c_ap_collision(subfigs[1, 0])
    _panel_d_spike_desync(subfigs[1, 1])

    # Shared legend at the very bottom of the figure.
    from matplotlib.lines import Line2D
    leg_handles = [
        Line2D([0], [0], color=C_NEURON, lw=2.0, ls="-",  label="PyFibers"),
        Line2D([0], [0], color=C_SMF,    lw=2.0, ls="--", label="JAXON"),
    ]
    fig.legend(handles=leg_handles, loc="lower center", ncol=2,
                frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.005))

    path = OUT_DIR / "fig3_combined.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
