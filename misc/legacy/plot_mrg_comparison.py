"""MRG threshold vs fiber diameter — PyFibers (NEURON) vs JAX coupled solver.

Reads:
  outputs/pyfibers_thresholds.json
  outputs/jax_thresholds.json

Saves:
  outputs/fig_mrg_comparison.pdf  (+ .png)
"""
import sys, pathlib, json
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

OUT = ROOT / "outputs"

# ── load data ──────────────────────────────────────────────────────────────
with open(OUT / "pyfibers_thresholds.json") as f:
    pf_raw = json.load(f)
with open(OUT / "jax_thresholds.json") as f:
    jx_raw = json.load(f)

# Convert string keys (JSON) back to float, sort by diameter
# JSON keys are always strings; rebuild with float keys
pf = {float(k): v for k, v in pf_raw.items()}
jx = {float(k): v for k, v in jx_raw.items()}
# Only plot diameters present in both result files
diams  = sorted(set(pf) & set(jx))
pf_thr = np.array([pf[d] for d in diams])
jx_thr = np.array([jx[d] for d in diams])

# Work with absolute (cathodic) amplitudes for plotting
pf_abs = np.abs(pf_thr)
jx_abs = np.abs(jx_thr)
err_pct = (jx_abs - pf_abs) / pf_abs * 100  # signed % error

# ── figure ─────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6),
                                gridspec_kw=dict(height_ratios=[3, 1.2], hspace=0.08),
                                sharex=True)

# --- top panel: threshold vs diameter ---
ax1.plot(diams, pf_abs, "o-",  color="#1f77b4", lw=2,   ms=6, label="PyFibers / NEURON")
ax1.plot(diams, jx_abs, "s--", color="#d62728", lw=1.8, ms=5, label="JAX coupled solver")
ax1.set_ylabel("|Threshold| (mA)", fontsize=12)
ax1.legend(fontsize=10, framealpha=0.9)
ax1.set_title("MRG activation threshold vs fiber diameter\n"
              "(point source, 1 mm, 0.1 ms monophasic cathodic, N=21 nodes)",
              fontsize=10)
ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax1.grid(True, which="major", alpha=0.3)
ax1.grid(True, which="minor", alpha=0.1)

# --- bottom panel: % error ---
ax2.axhline(0, color="k", lw=0.8, ls="--")
ax2.axhline(5,  color="gray", lw=0.6, ls=":")
ax2.axhline(-5, color="gray", lw=0.6, ls=":")
ax2.plot(diams, err_pct, "^-", color="#2ca02c", lw=1.5, ms=5)
ax2.set_ylabel("Error (%)", fontsize=11)
ax2.set_xlabel("Fiber diameter (µm)", fontsize=12)
ax2.set_ylim(-10, 10)
ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax2.grid(True, which="major", alpha=0.3)

ax1.set_xlim(diams[0] - 0.5, diams[-1] + 0.5)

fig.tight_layout()
for ext in (".pdf", ".png"):
    fig.savefig(OUT / f"fig_mrg_comparison{ext}", dpi=180, bbox_inches="tight")
    print(f"Saved → {OUT / ('fig_mrg_comparison' + ext)}")

# ── print table ────────────────────────────────────────────────────────────
print(f"\n{'D (µm)':>8}  {'PyFibers':>12}  {'JAX':>12}  {'error':>8}")
print("-" * 46)
for d, pf, jx, e in zip(diams, pf_abs, jx_abs, err_pct):
    print(f"{d:>8.1f}  {-pf:>12.5f}  {-jx:>12.5f}  {e:>7.1f}%")
