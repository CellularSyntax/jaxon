"""Supplement figure: optimizer & gradient validation (reviewer #4ii, #6).
(a) exact autodiff gradient vs central finite differences on the real
    selectivity objective (pooled over points along a trajectory);
(b) selectivity optimization driven purely by the analytic gradient;
(c) loss-convergence trajectories for representative nerves (existing runs).
Run (jaxley_fibers env): python -m experiments_v2.fig_optimizer_validation
"""
import os, sys
os.environ.setdefault("JAXLEY_FIBERS_SOFT_TEMPERATURE", "0.1")
os.environ.setdefault("JAXLEY_FIBERS_ENERGY_LAMBDA", "1e-3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import json, glob
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt

from jaxfibers.stim.batch_solve import (stack_fiber_statics, initial_states_batch,
    batch_integrate_m_max_fd)
from jaxfibers.optim.optimizer import (selectivity_loss_from_mmax,
    run_rect_optimization_autodiff)
from jaxfibers.optim.losses import activation_proxy_batch, selectivity_index
from experiments_v2.duke_loader import (load_duke_sample, select_random_cluster_targets,
    cluster_target_mask)

DT = 0.005
PAL = {"swine": "#D6604D", "human": "#4393C3"}

def pulse(tstop=4.0, delay=0.5, pw=0.2, ratio=4.0):
    n = int(tstop/DT); t = (np.arange(n)+1)*DT; p = np.zeros(n)
    p[(t>=delay)&(t<delay+pw)] = 1.0
    p[(t>=delay+pw)&(t<delay+pw+pw*ratio)] = -1.0/ratio
    return jnp.asarray(p)

# ---------- gradient data (subsampled, CPU-cheap) ----------
print("loading nerve for gradient check ...", flush=True)
duke = load_duke_sample(os.path.join(ROOT, "duke_Ves", "human_sub-50_sam-2"),
                        fiber_diam_um=5.7, n_nodes=21, verbose=False)
ci = select_random_cluster_targets(duke["fasc_meta"], duke["fasc_id"],
        duke["nerve_outline_xy_um"], n_positions=4, rng_seed=0,
        radius_quantile=0.32, angular_window_deg=90.0, n_min_target_fibers=50)[0]
tm = cluster_target_mask(duke["nerve_geom"], duke["fasc_id"], duke["fasc_meta"], ci["target_ids"])
rng = np.random.default_rng(0)
ti, oi = np.where(tm)[0], np.where(~tm)[0]
sel = np.sort(np.concatenate([rng.choice(ti, 24, False), rng.choice(oi, 24, False)]))
geoms = [duke["geoms"][i] for i in sel]
fs = stack_fiber_statics(geoms, DT); s0 = initial_states_batch(geoms)
Ve = jnp.asarray(duke["Ve_unit"][:, sel, :]); nidx = jnp.asarray(duke["node_indices"][sel], jnp.int32)
tgt = jnp.asarray(tm[sel], float)
w = jnp.where(tgt > 0, 0.5/float(tgt.sum()), 0.5/float((1-tgt).sum()))
PM = pulse(); PP = jnp.concatenate([jnp.zeros(1), PM[:-1]])
K = Ve.shape[0]

def loss(amps):
    Vc = jnp.einsum("k,kfn->fn", -amps, Ve)
    m = batch_integrate_m_max_fd(fs, s0, Vc, PM, PP, DT)
    return selectivity_loss_from_mmax(m, nidx, tgt, w, amps)
gfun = jax.jit(jax.grad(loss))

print("autodiff optimization (analytic gradient) ...", flush=True)
res = run_rect_optimization_autodiff(fs, s0, Ve, PM, nidx, np.asarray(tgt > 0),
        weights=np.asarray(w), dt=DT, n_steps=30, amp_init_mA=-0.3,
        amp_clip=(-2.0, 2.0), lr=0.02, early_stop_si=2.0,
        early_stop_patience=10**9, early_stop_si_patience=10**9, verbose=False)
si_hist = np.asarray(res["history"]["si"])
amps_hist = res["history"]["amps"]

print("pooling autodiff-vs-FD gradient components along the trajectory ...", flush=True)
eps = 5e-4
g_ad_all, g_fd_all = [], []
for it in (0, len(amps_hist)//2, len(amps_hist)-1):
    a = jnp.asarray(amps_hist[it])
    g_ad = np.asarray(gfun(a))
    g_fd = np.array([float(loss(a.at[k].add(eps)) - loss(a.at[k].add(-eps)))/(2*eps)
                     for k in range(K)])
    g_ad_all.append(g_ad); g_fd_all.append(g_fd)
g_ad_all = np.concatenate(g_ad_all); g_fd_all = np.concatenate(g_fd_all)
cos = float(g_ad_all @ g_fd_all / (np.linalg.norm(g_ad_all)*np.linalg.norm(g_fd_all)+1e-30))
rel = float(np.linalg.norm(g_ad_all - g_fd_all)/(np.linalg.norm(g_fd_all)+1e-30))

# ---------- convergence data (existing runs) ----------
REPR = [("swine", "sub-13_sam-3"), ("swine", "sub-10_sam-1"), ("swine", "sub-8_sam-1"),
        ("human", "human_sub-54_sam-3"), ("human", "human_sub-50_sam-2"), ("human", "human_sub-53_sam-2")]
conv = []
for sp, samp in REPR:
    try:
        d = json.load(open(os.path.join(ROOT, "outputs/duke_sweeps", samp, "data_seed_0000.json")))
        lh = np.asarray(d["rect"]["loss_history"], float)
        conv.append((sp, samp, lh, float(d["rect"]["achievable_si"])))
    except Exception as e:
        print("skip", samp, e)

# ---------- plot ----------
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.dpi": 600})
fig, ax = plt.subplots(1, 3, figsize=(9.2, 2.9))

lim = max(np.abs(g_ad_all).max(), np.abs(g_fd_all).max()) * 1.15
ax[0].plot([-lim, lim], [-lim, lim], color="0.6", lw=0.8, ls="--", zorder=0)
ax[0].scatter(g_fd_all, g_ad_all, s=16, color="#333333", alpha=0.8, edgecolors="none")
ax[0].set_xlim(-lim, lim); ax[0].set_ylim(-lim, lim); ax[0].set_aspect("equal")
ax[0].set_xlabel(r"finite-difference $\partial\mathcal{L}/\partial I_k$")
ax[0].set_ylabel(r"autodiff $\partial\mathcal{L}/\partial I_k$")
ax[0].set_title("(a) gradient correctness", loc="left", fontsize=8, weight="bold")
ax[0].text(0.05, 0.93, f"cosine = {cos:.4f}\nrel-L2 = {rel:.1e}", transform=ax[0].transAxes,
           va="top", ha="left", fontsize=7)

ax[1].plot(np.arange(si_hist.size), si_hist, "-o", ms=3, color="#1f77b4", lw=1.0)
ax[1].axhline(si_hist.max(), color="0.6", ls=":", lw=0.8)
ax[1].set_xlabel("iteration"); ax[1].set_ylabel("selectivity index")
ax[1].set_title("(b) analytic-gradient optimization", loc="left", fontsize=8, weight="bold")
ax[1].text(0.95, 0.1, f"{si_hist[0]:+.2f} $\\rightarrow$ {si_hist.max():+.2f}",
           transform=ax[1].transAxes, va="bottom", ha="right", fontsize=7)

for sp, samp, lh, si in conv:
    best = np.minimum.accumulate(lh)        # running-best (warm-started) loss
    x = np.arange(best.size)
    ax[2].plot(x, best, color=PAL[sp], lw=1.1, alpha=0.85)
    ax[2].text(x[-1], best[-1], f" SI={si:.2f}", fontsize=6, color=PAL[sp], va="center")
ax[2].set_xlabel("iteration"); ax[2].set_ylabel("best loss so far")
ax[2].set_title("(c) convergence (representative nerves)", loc="left", fontsize=8, weight="bold")
ax[2].plot([], [], color=PAL["swine"], label="swine"); ax[2].plot([], [], color=PAL["human"], label="human")
ax[2].legend(frameon=False, fontsize=7, loc="upper right")

fig.tight_layout()
out = os.path.join(ROOT, "manuscript/figures/supp/figS_optimizer_validation.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
print("saved", out)
print(f"gradient cosine={cos:.4f} relL2={rel:.2e}  SI {si_hist[0]:+.3f}->{si_hist.max():+.3f}")
