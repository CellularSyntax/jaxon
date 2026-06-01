"""Autodiff gradient validation — Task 4 (Phase 3, migration_plan.md v3)

Validates that jax.grad through the MRG coupled (Vi, Vpax) solver matches
central finite differences for three classes of parameter:

  4.1  Stimulation amplitude  amp_mA (scalar)
       d(V_peak)/d(amp) compared against FD across a range from deep
       sub-threshold to well suprathreshold.

  4.2  Waveform bins  w[0..K-1] (K free amplitudes, one per time-bin)
       d(V_peak)/d(w[k]) for all k simultaneously via one jax.grad call.
       Timing comparison: autodiff O(1) vs FD O(K) for K = 1..200.

  4.3  Electrode height  src_y_um (scalar)
       d(V_peak)/d(height) using a JAX-native point-source formula —
       demonstrates differentiability w.r.t. electrode geometry.

Model: MRG D=10 µm, 21 nodes, 0.1 ms mono pulse, 5 ms sim, dt=0.005 ms,
       1 mm above centre, σ=0.3 S/m, T=37 °C.

Output:
  outputs/fig_autodiff_validation.png  (4-panel)
  outputs/data_autodiff_validation.json

Run from project root:
    conda run -n jaxley_fibers python experiments/exp_autodiff_validation.py
"""

from __future__ import annotations

import sys, pathlib, json, time, dataclasses
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import jax
import jax.numpy as jnp
from jaxley.solver_gate import solve_gate_exponential

from jaxfibers.fibers.mrg import (
    build_mrg, node_indices, section_centers_um,
    V_REST, CM_AXON, G_PAS_MYSA, G_PAS_FLUT, G_PAS_STIN,
)
from jaxfibers.channels.mrg_axnode import AxnodeMyel
from mrg_extracellular_coupled import arrays_from_geometry, integrate

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# ── simulation constants ──────────────────────────────────────────────────────
CELSIUS   = 37.0
DT        = 0.005      # ms
TSTOP     = 5.0        # ms
DELAY     = 1.0        # ms
PW        = 0.1        # ms
SIGMA     = 0.3        # S/m
HEIGHT    = 1000.0     # µm
DIAM      = 10.0       # µm
N_NODES   = 21
N_STEPS   = int(TSTOP / DT)    # 1000

# MRG channel constants (from validation suite)
GNABAR = 3.0; GNAPBAR = 0.01; GKBAR = 0.08; GL = 0.007
ENA = 50.0; EK = -90.0; EL = -90.0

# FD step size (central differences)
H_AMP = 1e-5     # mA  — well within the smooth regime
H_HEIGHT = 0.1   # µm  — 0.1 µm perturbation of 1000 µm height


# ── JAX setup (verbatim from exp_mrg_validation_suite) ────────────────────────

def _build_setup():
    _, geom   = build_mrg(diameter=DIAM, n_nodes=N_NODES)
    centers   = np.array(section_centers_um(geom))
    n_comp    = geom.n_comp
    mid       = node_indices(geom)[len(node_indices(geom)) // 2]

    geom_c = dataclasses.replace(geom, cm_uF_cm2=[CM_AXON] * n_comp)
    static = arrays_from_geometry(geom_c, DT)
    is_node   = static["is_node"]
    A_in      = static["A_in_cm2"]

    stype_gp  = {"node": 0.0, "mysa": G_PAS_MYSA, "flut": G_PAS_FLUT, "stin": G_PAS_STIN}
    g_pas_arr = jnp.asarray([stype_gp[s] for s in geom.section_type])

    # Resting gates
    v0 = jnp.float64(V_REST)
    (am,bm),(ah,bh),(amp_,bmp),(as_,bs) = AxnodeMyel._alpha_beta(v0, CELSIUS)
    state0 = (
        jnp.where(is_node, float(am/(am+bm)),   0.0),
        jnp.where(is_node, float(ah/(ah+bh)),   0.0),
        jnp.where(is_node, float(amp_/(amp_+bmp)), 0.0),
        jnp.where(is_node, float(as_/(as_+bs)), 0.0),
    )

    def membrane_fn(Vm, state, dt_):
        M, H, MP, S = state
        (am,bm),(ah,bh),(amp_,bmp),(as_,bs) = AxnodeMyel._alpha_beta(Vm, CELSIUS)
        M2  = solve_gate_exponential(M,  dt_, am,  bm)
        H2  = solve_gate_exponential(H,  dt_, ah,  bh)
        MP2 = solve_gate_exponential(MP, dt_, amp_, bmp)
        S2  = solve_gate_exponential(S,  dt_, as_,  bs)
        gna = GNABAR * M2**3 * H2; gnap = GNAPBAR * MP2**3; gk = GKBAR * S2
        g_node = (gna + gnap + gk + GL) * A_in * 1e6
        i_node = ((gna + gnap) * (Vm - ENA) + gk * (Vm - EK) + GL * (Vm - EL)) * A_in * 1e6
        g_pas  = g_pas_arr * A_in * 1e6
        i_pas  = g_pas * (Vm - V_REST)
        return (jnp.where(is_node, g_node, g_pas),
                jnp.where(is_node, i_node, i_pas),
                (M2, H2, MP2, S2))

    # Standard unit-amplitude Ve at HEIGHT above fibre centre
    t_vec  = (np.arange(N_STEPS) + 1) * DT
    shape0 = jnp.asarray(((t_vec >= DELAY) & (t_vec < DELAY + PW)).astype(np.float64))

    def Ve_from_height_jax(src_y_um):
        """Differentiable point-source profile (mV) for variable electrode height."""
        dy  = src_y_um * 1e-6                              # m
        dz  = (centers[mid] - centers) * 1e-6             # m (src_z fixed at fibre centre)
        r   = jnp.sqrt(dy**2 + dz**2)
        return -1.0 / (4.0 * jnp.pi * SIGMA * r)          # mV (i0=-1 mA)

    Ve_unit = np.array(Ve_from_height_jax(jnp.float64(HEIGHT)))

    return static, membrane_fn, state0, Ve_unit, shape0, centers, mid, Ve_from_height_jax, geom


# ── core loss functions ───────────────────────────────────────────────────────

def make_loss_amp(static, mfn, state0, Ve_unit, shape):
    """V_peak as a function of scalar amplitude (cathodic negative)."""
    Ve_j = jnp.asarray(Ve_unit, dtype=jnp.float64)
    @jax.jit
    def loss(amp_mA: jnp.ndarray) -> jnp.ndarray:
        Ve = Ve_j * (amp_mA / -1.0)
        trace, _ = integrate(static, mfn, state0, Ve, shape, DT, v_rest=V_REST)
        return jnp.max(trace)
    return loss


def make_loss_waveform(static, mfn, state0, Ve_unit, amp_ref, K: int):
    """V_peak as a function of K waveform-bin amplitudes (mA/bin).

    Bins uniformly tile the pulse window [DELAY, DELAY+PW].
    waveform_bins[k] = scalar multiplier on Ve_unit during bin k.
    Total active steps = int(PW/DT) = 20; K divides that range.
    """
    Ve_j    = jnp.asarray(Ve_unit * (amp_ref / -1.0), dtype=jnp.float64)
    n_steps = N_STEPS
    t_vec   = (np.arange(n_steps) + 1) * DT
    # Bin assignment: which bin does each step belong to?
    active_start = int(np.round(DELAY / DT))
    active_end   = int(np.round((DELAY + PW) / DT))
    n_active     = active_end - active_start           # = 20 steps
    bins_per_step = np.zeros(n_steps, dtype=int)       # 0 = "off" bin (unused)
    for s in range(active_start, active_end):
        bins_per_step[s] = 1 + (s - active_start) * K // n_active
    # B[step, bin_idx] — one-hot (K+1 cols, col 0 = inactive)
    B = np.zeros((n_steps, K + 1), dtype=np.float64)
    for s in range(n_steps):
        B[s, bins_per_step[s]] = 1.0
    B_j = jnp.asarray(B[:, 1:], dtype=jnp.float64)  # (n_steps, K), drop col 0

    @jax.jit
    def loss(waveform_bins: jnp.ndarray) -> jnp.ndarray:
        mask = B_j @ waveform_bins                    # (n_steps,) ← differentiable
        trace, _ = integrate(static, mfn, state0, Ve_j, mask, DT, v_rest=V_REST)
        return jnp.max(trace)
    return loss, B_j


def make_loss_height(static, mfn, state0, Ve_from_height_jax, amp_ref, shape):
    """V_peak as a function of electrode height (µm)."""
    @jax.jit
    def loss(src_y_um: jnp.ndarray) -> jnp.ndarray:
        Ve = Ve_from_height_jax(src_y_um) * (amp_ref / -1.0)
        trace, _ = integrate(static, mfn, state0, Ve, shape, DT, v_rest=V_REST)
        return jnp.max(trace)
    return loss


# ── finite-difference helpers ─────────────────────────────────────────────────

def fd_scalar(fn, x, h):
    """Central FD for scalar x."""
    return (fn(jnp.float64(x + h)) - fn(jnp.float64(x - h))) / (2.0 * h)


def fd_vector(fn, x, h):
    """Central FD for vector x, returning gradient of same shape."""
    g = np.zeros_like(np.asarray(x))
    for i in range(len(g)):
        xp = np.array(x, dtype=np.float64); xp[i] += h
        xm = np.array(x, dtype=np.float64); xm[i] -= h
        g[i] = (fn(jnp.asarray(xp)) - fn(jnp.asarray(xm))) / (2.0 * h)
    return g


# ── Task 4.1 — amplitude gradient ─────────────────────────────────────────────

def run_amplitude_gradient(static, mfn, state0, Ve_unit, shape):
    print("[4.1] Amplitude gradient ...", flush=True)
    loss_amp = make_loss_amp(static, mfn, state0, Ve_unit, shape)
    grad_amp = jax.jit(jax.grad(loss_amp))

    # Find threshold via bisection
    def fires(a): return bool(loss_amp(jnp.float64(a)) > -30.0)
    lo, hi = -5.0, -0.001
    for _ in range(10):
        if fires(lo): break
        lo *= 2.0
    while abs(hi - lo) > 1e-4:
        mid = 0.5*(lo+hi)
        if fires(mid) == fires(hi): hi = mid
        else: lo = mid
    thr = 0.5*(lo+hi)
    print(f"  threshold ≈ {thr:.5f} mA")

    # Sample amplitudes: sub-threshold and suprathreshold (skip near-threshold band)
    sub_amps = np.linspace(-0.05, thr * 0.90, 20)   # clearly sub-threshold
    supra_amps = np.linspace(thr * 1.10, -0.50, 12)  # clearly supra-threshold
    amps = np.concatenate([sub_amps, supra_amps])

    # Warm up JIT
    _ = grad_amp(jnp.float64(amps[0])).block_until_ready()
    _ = loss_amp(jnp.float64(amps[0])).block_until_ready()

    ad_grads, fd_grads, vpeak_vals = [], [], []
    for a in amps:
        ad = float(grad_amp(jnp.float64(a)))
        fd = float(fd_scalar(loss_amp, a, H_AMP))
        vp = float(loss_amp(jnp.float64(a)))
        ad_grads.append(ad); fd_grads.append(fd); vpeak_vals.append(vp)

    ad_grads   = np.array(ad_grads)
    fd_grads   = np.array(fd_grads)
    vpeak_vals = np.array(vpeak_vals)
    rel_err    = np.abs(ad_grads - fd_grads) / (np.abs(fd_grads) + 1e-12)
    print(f"  median |rel err| = {np.median(rel_err)*100:.4f}%  "
          f"max |rel err| = {np.max(rel_err)*100:.4f}%", flush=True)

    return dict(amps=amps.tolist(), ad=ad_grads.tolist(), fd=fd_grads.tolist(),
                vpeak=vpeak_vals.tolist(), rel_err=rel_err.tolist(),
                threshold_mA=float(thr), sub_n=len(sub_amps))


# ── Task 4.2 — waveform bins gradient ─────────────────────────────────────────

def run_waveform_gradient(static, mfn, state0, Ve_unit, thr_mA):
    print("[4.2] Waveform bin gradient ...", flush=True)
    amp_ref = thr_mA * 1.3
    K = 20
    loss_wav, B_j = make_loss_waveform(static, mfn, state0, Ve_unit, amp_ref, K)
    grad_wav = jax.jit(jax.grad(loss_wav))

    # Reference waveform: uniform rectangular (each bin = 1.0)
    w0 = jnp.ones(K, dtype=jnp.float64)

    # Warm up
    _ = grad_wav(w0).block_until_ready()

    t0 = time.perf_counter()
    ad_grads = np.array(grad_wav(w0).block_until_ready())
    t_ad = time.perf_counter() - t0
    print(f"  autodiff K={K}: {t_ad*1e3:.1f} ms", flush=True)

    t0 = time.perf_counter()
    fd_grads = fd_vector(loss_wav, w0, 1e-4)
    t_fd = time.perf_counter() - t0
    print(f"  FD       K={K}: {t_fd*1e3:.0f} ms (= {2*K} forward passes)", flush=True)

    rel_err = np.abs(ad_grads - fd_grads) / (np.abs(fd_grads) + 1e-12)
    print(f"  median |rel err| = {np.median(rel_err)*100:.4f}%  "
          f"max = {np.max(rel_err)*100:.4f}%", flush=True)

    return dict(K=K, ad=ad_grads.tolist(), fd=fd_grads.tolist(),
                rel_err=rel_err.tolist(), t_ad_ms=t_ad*1e3, t_fd_ms=t_fd*1e3)


# ── Task 4.2b — runtime scaling with K ────────────────────────────────────────

def run_timing_vs_K(static, mfn, state0, Ve_unit, thr_mA):
    print("[4.2b] Runtime scaling vs K ...", flush=True)
    amp_ref = thr_mA * 1.3
    Ks = [1, 5, 10, 20, 50, 100, 200]
    t_ad_list, t_fd_list = [], []
    N_REP = 3

    for K in Ks:
        loss_k, _ = make_loss_waveform(static, mfn, state0, Ve_unit, amp_ref, K)
        grad_k    = jax.jit(jax.grad(loss_k))
        w0 = jnp.ones(K, dtype=jnp.float64)

        # Warm up
        _ = grad_k(w0).block_until_ready()
        _ = loss_k(w0).block_until_ready()

        # Autodiff timing
        t0 = time.perf_counter()
        for _ in range(N_REP):
            grad_k(w0).block_until_ready()
        t_ad = (time.perf_counter() - t0) / N_REP

        # FD timing (2K forward passes)
        t0 = time.perf_counter()
        fd_vector(loss_k, w0, 1e-4)
        t_fd = time.perf_counter() - t0

        t_ad_list.append(t_ad); t_fd_list.append(t_fd)
        print(f"  K={K:3d}: autodiff {t_ad*1e3:.1f} ms | FD {t_fd*1e3:.0f} ms "
              f"| speedup {t_fd/t_ad:.1f}×", flush=True)

    return dict(Ks=Ks, t_ad_ms=[t*1e3 for t in t_ad_list],
                t_fd_ms=[t*1e3 for t in t_fd_list])


# ── Task 4.3 — electrode height gradient ──────────────────────────────────────

def run_height_gradient(static, mfn, state0, Ve_from_height_jax, shape, thr_mA):
    print("[4.3] Electrode height gradient ...", flush=True)
    amp_ref = thr_mA * 1.3
    loss_h = make_loss_height(static, mfn, state0, Ve_from_height_jax, amp_ref, shape)
    grad_h = jax.jit(jax.grad(loss_h))

    # Compute at several heights (sub-threshold for given amp_ref if height is larger)
    heights = np.array([500., 750., 1000., 1500., 2000., 3000.])

    _ = grad_h(jnp.float64(heights[0])).block_until_ready()

    ad_list, fd_list = [], []
    for h in heights:
        ad = float(grad_h(jnp.float64(h)))
        fd = float(fd_scalar(loss_h, h, H_HEIGHT))
        ad_list.append(ad); fd_list.append(fd)
        print(f"  height={h:.0f} µm: autodiff={ad:.4f} mV/µm  FD={fd:.4f} mV/µm  "
              f"|rel err|={abs(ad-fd)/(abs(fd)+1e-12)*100:.4f}%", flush=True)

    rel_err = np.abs(np.array(ad_list) - np.array(fd_list)) / \
              (np.abs(np.array(fd_list)) + 1e-12)
    return dict(heights=heights.tolist(), ad=ad_list, fd=fd_list,
                rel_err=rel_err.tolist())


# ── figure ────────────────────────────────────────────────────────────────────

def make_figure(r41, r42, r42b, r43):
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    # ── (A) Amplitude gradient: autodiff vs FD ────────────────────────────────
    ax = axs[0, 0]
    amps   = np.array(r41["amps"])
    ad     = np.array(r41["ad"])
    fd     = np.array(r41["fd"])
    thr    = r41["threshold_mA"]
    sub_n  = r41["sub_n"]

    ax.plot(amps[:sub_n], ad[:sub_n],  "o-",  color="C2", ms=4, lw=1.5, label="autodiff (sub-thr)")
    ax.plot(amps[:sub_n], fd[:sub_n],  "x--", color="C0", ms=4, lw=1.0, label="FD (sub-thr)")
    ax.plot(amps[sub_n:], ad[sub_n:],  "o-",  color="C3", ms=4, lw=1.5, label="autodiff (supra-thr)")
    ax.plot(amps[sub_n:], fd[sub_n:],  "x--", color="C1", ms=4, lw=1.0, label="FD (supra-thr)")
    ax.axvline(thr, color="gray", ls=":", lw=1.0, label=f"threshold ({thr:.3f} mA)")
    ax.set_xlabel("Amplitude (mA)"); ax.set_ylabel("dV_peak / d(amp)  (mV/mA)")
    ax.set_title(f"(A) Amplitude gradient: autodiff vs FD\nMRG D={DIAM}µm, "
                 f"median |err| = {np.median(np.abs(np.array(r41['rel_err'])))*100:.3f}%")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.25)

    # ── (B) Waveform bin gradients ────────────────────────────────────────────
    ax = axs[0, 1]
    K   = r42["K"]
    ad2 = np.array(r42["ad"])
    fd2 = np.array(r42["fd"])
    x   = np.arange(K)
    w   = 0.35
    ax.bar(x - w/2, np.abs(fd2), w, color="C0", alpha=0.8, label=f"FD ({2*K} passes)")
    ax.bar(x + w/2, np.abs(ad2), w, color="C2", alpha=0.8, label=f"autodiff (1 pass)")
    ax.set_xlabel(f"Waveform bin (k)"); ax.set_ylabel("|dV_peak / dw_k|  (mV per unit)")
    ax.set_title(f"(B) Waveform bin gradients  K={K}\n"
                 f"t_autodiff={r42['t_ad_ms']:.1f} ms  t_FD={r42['t_fd_ms']:.0f} ms  "
                 f"speedup={r42['t_fd_ms']/r42['t_ad_ms']:.1f}×")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)

    # ── (C) Runtime scaling vs K ──────────────────────────────────────────────
    ax = axs[1, 0]
    Ks     = r42b["Ks"]
    t_ad   = np.array(r42b["t_ad_ms"])
    t_fd   = np.array(r42b["t_fd_ms"])
    ax.loglog(Ks, t_ad, "s-",  color="C2", lw=2.0, ms=7, label="autodiff (reverse-mode)")
    ax.loglog(Ks, t_fd, "o--", color="C0", lw=2.0, ms=6, label="FD (central, 2K passes)")
    # O(K) reference line
    ref_t  = t_fd[0] / Ks[0]
    ax.loglog(Ks, [ref_t * K for K in Ks], ":", color="gray", lw=1.0, label="O(K) slope")
    ax.set_xlabel("Number of parameters K")
    ax.set_ylabel("Wall-clock time (ms)")
    ax.set_title("(C) Runtime scaling: autodiff O(1) vs FD O(K)")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.25)
    # Annotate speedup at largest K
    spd_max = t_fd[-1] / t_ad[-1]
    ax.text(Ks[-1], t_ad[-1], f" {spd_max:.0f}×", va="center", fontsize=9, color="C2")

    # ── (D) Error summary ─────────────────────────────────────────────────────
    ax = axs[1, 1]
    all_rel_err = np.concatenate([
        np.array(r41["rel_err"]) * 100,
        np.array(r42["rel_err"]) * 100,
        np.array(r43["rel_err"]) * 100,
    ])
    labels = (
        [f"amp\n{a:.2f}mA" for a in r41["amps"]] +
        [f"w[{k}]" for k in range(r42["K"])] +
        [f"h={h:.0f}µm" for h in r43["heights"]]
    )
    colors = (["C2"] * len(r41["amps"]) +
              ["C3"] * r42["K"] +
              ["C4"] * len(r43["heights"]))
    x_pos  = np.arange(len(all_rel_err))
    ax.bar(x_pos, all_rel_err, color=colors, alpha=0.8)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="1 % guideline")
    # x-tick labels: too many — just show category boundaries
    n_amp  = len(r41["amps"])
    n_wav  = r42["K"]
    n_hgt  = len(r43["heights"])
    ax.set_xticks([n_amp//2, n_amp + n_wav//2, n_amp + n_wav + n_hgt//2])
    ax.set_xticklabels(["amplitude\n(4.1)", "waveform bins\n(4.2)", "height\n(4.3)"])
    ax.set_ylabel("|relative error|  (%)")
    ax.set_title(f"(D) Autodiff vs FD error summary\n"
                 f"median = {np.median(all_rel_err):.4f}%  max = {np.max(all_rel_err):.4f}%")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)
    # Colour legend patches
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="C2", alpha=0.8, label="amplitude (4.1)"),
        Patch(color="C3", alpha=0.8, label="waveform bins (4.2)"),
        Patch(color="C4", alpha=0.8, label="electrode height (4.3)"),
        plt.Line2D([0],[0], color="gray", ls="--", lw=0.8, label="1% guideline"),
    ], fontsize=8)

    fig.suptitle(
        "Autodiff validation: jax.grad vs central finite differences\n"
        f"MRG D={DIAM}µm, {N_NODES} nodes, PW={PW}ms, 1mm above centre, σ={SIGMA} S/m, T={CELSIUS}°C",
        fontsize=10,
    )
    out = OUT / "fig_autodiff_validation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  → {out}")


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_wall = time.time()
    static, mfn, state0, Ve_unit, shape0, centers, mid, Ve_from_height_jax, geom = \
        _build_setup()

    r41  = run_amplitude_gradient(static, mfn, state0, Ve_unit, shape0)
    thr  = r41["threshold_mA"]

    r42  = run_waveform_gradient(static, mfn, state0, Ve_unit, thr)
    r42b = run_timing_vs_K(static, mfn, state0, Ve_unit, thr)
    r43  = run_height_gradient(static, mfn, state0, Ve_from_height_jax, shape0, thr)

    print("\n=== Generating figure ===", flush=True)
    make_figure(r41, r42, r42b, r43)

    data = {"4.1_amplitude": r41, "4.2_waveform": r42,
            "4.2b_timing": r42b, "4.3_height": r43}
    with open(OUT / "data_autodiff_validation.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → {OUT / 'data_autodiff_validation.json'}")
    print(f"\nDone. Total wall time: {(time.time()-t_wall)/60:.1f} min")
