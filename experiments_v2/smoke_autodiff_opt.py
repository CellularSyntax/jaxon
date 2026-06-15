"""Smoke test for the autodiff / gradient-based optimization path.

Run ONE Duke nerve/seed through the differentiable optimizer to confirm,
before launching a full cohort sweep, that:

  A. autodiff gradients are CORRECT     (vs finite differences, at a firing
                                         operating point where the gradient
                                         carries signal);
  B. the SAME 12-amplitude problem      reaches the SAME selectivity under
                                         Adam-FD (cohort method) and exact-
                                         gradient LBFGS (autodiff);
  C. the autodiff WAVEFORM optimizer    (K x T DOF -- where FD would need
                                         K*T+1 forwards, i.e. infeasible) runs
                                         AND fits GPU memory at the nerve scale.

Key design point: the optimizer must START IN THE FIRING REGIME.  At a small
init amplitude a Duke nerve can be entirely sub-threshold, giving a ~zero
gradient where no optimizer (FD or autodiff) can move -- so we first run a
magnitude probe to find a firing start, then run A/B/C from there.

Usage (cluster, inside the container after slurm/setup_env.sh):

    DUKE_SAMPLE_DIR=duke_Ves/sub-11_sam-3 \
    python -m experiments_v2.smoke_autodiff_opt

Useful overrides:
    MAX_FIBERS=300       cap fibers (memory headroom check)
    T_STOP=1.5           shorter window -> smaller T for the waveform test
    SMOKE_RECT_STEPS=20  optimizer iterations for the head-to-head
    SMOKE_WAVE_ITERS=5   waveform optimizer iterations (slow; keep small)
"""
from __future__ import annotations

import os
import sys
import time

# small-run defaults, set BEFORE importing the sweep module (reads them at import)
os.environ.setdefault("DUKE_SAMPLE_DIR", "duke_Ves/sub-11_sam-3")
os.environ.setdefault("RECT_OPTIMIZER", "lbfgs")
os.environ.setdefault("N_OPT_RECT", os.environ.get("SMOKE_RECT_STEPS", "20"))
os.environ.setdefault("N_RESTARTS_RECT", "1")

import numpy as np
import jax
import jax.numpy as jnp

from experiments_v2 import selectivity_sweep_duke as S
from jaxfibers.optim.optimizer import (
    run_rect_optimization,
    run_rect_optimization_lbfgs,
    run_waveform_optimization,
    _build_rect_loss_fn,
    _initial_amps_for_seed,
)
from jaxfibers.optim.losses import selectivity_index

WAVE_ITERS = int(os.environ.get("SMOKE_WAVE_ITERS", "5"))
# Ve-weighted pattern is scaled to each of these peak magnitudes (mA) to find
# a firing start.  Clipped to AMP_CLIP.
PROBE_MAGS = [-0.1, -0.3, -0.6, -1.0, -1.5, -2.0]


def gpu_peak_mb() -> float:
    try:
        st = jax.devices()[0].memory_stats() or {}
        b = st.get("peak_bytes_in_use", st.get("bytes_in_use", 0))
        return float(b) / 1e6
    except Exception:
        return float("nan")


def banner(msg: str) -> None:
    print("\n" + "=" * 72 + f"\n{msg}\n" + "=" * 72, flush=True)


def main() -> int:
    results: dict[str, bool] = {}

    banner(f"SMOKE: autodiff optimization on {S.SAMPLE_NAME}")
    print(f"device: {jax.devices()[0]}  | default float = {jnp.zeros(1).dtype}",
          flush=True)

    duke = S.load_duke_sample(
        S.SAMPLE_PATH, fiber_diam_um=S.FIBER_DIAMETER_UM, n_nodes=S.N_NODES,
        max_fibers=(S.MAX_FIBERS if S.MAX_FIBERS > 0 else None),
        subsample_seed=S.SUBSAMPLE_SEED, verbose=True,
    )
    seed_in = S._build_seed(duke, seed=0, verbose=True)
    if seed_in is None:
        print("[smoke] seed 0 produced no valid target.", file=sys.stderr)
        return 2

    Ve_unit = np.asarray(seed_in["Ve_unit"])
    K, N, ncomp = Ve_unit.shape
    T = int(seed_in["N_STEPS"])
    tgt = np.asarray(seed_in["target_mask"], dtype=bool)
    pulse = np.asarray(seed_in["pulse_mask"], float)
    pulse_prev = np.concatenate([[0.0], pulse[:-1]])
    w = np.asarray(seed_in["weights"], float)
    print(f"[smoke] N_fibers={N}  K_contacts={K}  n_comp={ncomp}  T_steps={T}  "
          f"targets={tgt.sum()}/{N}  baseline_SI={seed_in['si_baseline']:+.3f}",
          flush=True)

    Ve_j  = jnp.asarray(Ve_unit, jnp.float64)
    tgt_j = jnp.asarray(tgt, jnp.float64)
    loss_fn = _build_rect_loss_fn(
        seed_in["fs_batch"], seed_in["s0_batch"], Ve_j,
        jnp.asarray(pulse, jnp.float64), jnp.asarray(pulse_prev, jnp.float64),
        jnp.asarray(seed_in["node_indices"], jnp.int32),
        tgt_j, jnp.asarray(w, jnp.float64), S.DT,
    )

    def amps_at_mag(mag):
        return np.asarray(_initial_amps_for_seed(
            Ve_j, tgt_j, mag, S.AMP_CLIP, 1, jax.random.PRNGKey(0))[0])

    # ── 0. magnitude probe: find a FIRING start (otherwise gradient ~ 0) ─────
    banner("0. magnitude probe -- find a firing start")
    mags = [m for m in PROBE_MAGS if S.AMP_CLIP[0] <= m <= S.AMP_CLIP[1]]
    probe = [(m, float(loss_fn(amps_at_mag(m)))) for m in mags]
    for m, L in probe:
        print(f"  mag={m:+.2f} mA   loss={L:.4f}", flush=True)
    best_mag, best_L = min(probe, key=lambda t: t[1])
    amps_start = amps_at_mag(best_mag)
    firing = best_L < 0.49
    print(f"  chosen start: mag={best_mag:+.2f} mA  loss={best_L:.4f}  "
          f"{'(firing)' if firing else '(STILL SUB-THRESHOLD!)'}", flush=True)
    if not firing:
        print("  WARN: no firing start found in the probe range -- the checks "
              "below will be uninformative.  Try another nerve/seed, drop "
              "MAX_FIBERS, or widen AMP_CLIP.", flush=True)

    # ── A. gradient correctness at the firing start ─────────────────────────
    banner("A. gradient correctness (autodiff vs finite difference)")
    amps0 = jnp.asarray(amps_start, jnp.float64)
    grad = jax.grad(loss_fn)
    jax.block_until_ready(grad(amps0))                    # warm-up (compile)
    t0 = time.time(); g_ad = np.asarray(grad(amps0)); jax.block_until_ready(g_ad)
    t_ad = time.time() - t0
    jax.block_until_ready(loss_fn(amps0))                 # warm-up forward
    eps = 1e-3
    g_fd = np.zeros(K)
    t1 = time.time()
    for k in range(K):
        ap = amps0.at[k].add(eps); am = amps0.at[k].add(-eps)
        g_fd[k] = (float(loss_fn(ap)) - float(loss_fn(am))) / (2 * eps)
    t_fd = time.time() - t1
    denom = (np.linalg.norm(g_ad) * np.linalg.norm(g_fd)) or 1.0
    cos = float(g_ad @ g_fd / denom)
    rel = float(np.max(np.abs(g_ad - g_fd))) / (float(np.max(np.abs(g_fd))) or 1.0)
    print(f"  ||g_autodiff|| = {np.linalg.norm(g_ad):.4e}  ({t_ad:.1f}s, 1 backward)")
    print(f"  ||g_finitediff||= {np.linalg.norm(g_fd):.4e}  ({t_fd:.1f}s, {2*K} forwards)")
    print(f"  cosine(ad, fd) = {cos:.5f}   max rel err = {rel:.2%}", flush=True)
    results["A_grad_correct"] = bool(np.all(np.isfinite(g_ad)) and cos > 0.98
                                     and rel < 0.10)
    print(f"  --> {'PASS' if results['A_grad_correct'] else 'FAIL'}", flush=True)

    # ── B. same 12-amplitude problem: Adam-FD vs autodiff-LBFGS ─────────────
    banner("B. head-to-head, identical 12-amplitude problem "
           "(Adam-FD vs autodiff-LBFGS)")
    budget = S.N_OPT_RECT
    NEVER = 10 ** 9
    common = dict(
        fiber_statics_batch=seed_in["fs_batch"], state0_batch=seed_in["s0_batch"],
        Ve_unit=seed_in["Ve_unit"], pulse_mask=seed_in["pulse_mask"],
        node_indices=seed_in["node_indices"], target_mask=seed_in["target_mask"],
        weights=seed_in["weights"], dt=S.DT, amp_clip=S.AMP_CLIP, verbose=False,
    )
    row, best_amps, best_loss = {}, None, np.inf

    try:
        t0 = time.time()
        fd = run_rect_optimization(
            **common, n_steps=budget, amps_init_vector=amps_start,
            lr=S.ADAM_LR_MA, fd_eps=S.FD_EPS_MA,
            early_stop_si=2.0, early_stop_patience=NEVER, early_stop_si_patience=0,
        )
        jax.block_until_ready(fd["amps"]); t_fd_opt = time.time() - t0
        hl = np.asarray(fd["history"]["loss"]); hs = np.asarray(fd["history"]["si"])
        i = int(np.argmin(hl))
        row["Adam-FD"] = (len(hl), float(hl[i]), float(hs[i]), t_fd_opt, gpu_peak_mb())
        si0 = float(hs[0])
        if hl[i] < best_loss:
            best_loss = float(hl[i]); best_amps = np.asarray(fd["amps"])
    except Exception as e:               # noqa: BLE001
        row["Adam-FD"] = None; si0 = float("nan")
        print(f"  Adam-FD FAILED ({type(e).__name__}: {str(e)[:140]})", flush=True)

    try:
        t0 = time.time()
        lb = run_rect_optimization_lbfgs(
            **common, n_restarts=1, n_steps=budget,
            amp_init_mA=best_mag, rng_seed=0,
        )
        jax.block_until_ready(lb["final_acts"]); t_lb_opt = time.time() - t0
        si_lb = float(selectivity_index(np.asarray(lb["final_acts"]), tgt))
        row["LBFGS-autodiff"] = (budget, float(lb["final_loss"]), si_lb,
                                 t_lb_opt, gpu_peak_mb())
        if lb["final_loss"] < best_loss:
            best_loss = float(lb["final_loss"]); best_amps = np.asarray(lb["amps"])
    except Exception as e:               # noqa: BLE001
        row["LBFGS-autodiff"] = None
        msg = str(e); oom = "RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower()
        print(f"  LBFGS-autodiff FAILED ({type(e).__name__}"
              f"{' [OOM]' if oom else ''}: {msg[:140]})", flush=True)

    print(f"  start SI = {si0:+.3f}   (budget = {budget} iters, identical firing "
          f"start; wall-clock includes LBFGS line search; peak MB cumulative)",
          flush=True)
    print(f"  {'optimizer':16s} {'iters':>5} {'best loss':>10} "
          f"{'best SI':>8} {'wall (s)':>9} {'peak MB':>8}", flush=True)
    sis = {}
    for name in ("Adam-FD", "LBFGS-autodiff"):
        r = row.get(name)
        if r is None:
            print(f"  {name:16s}   ---  (failed)", flush=True); continue
        it, ls, si, t, mb = r; sis[name] = si
        print(f"  {name:16s} {it:>5d} {ls:>10.4f} {si:>+8.3f} {t:>9.1f} {mb:>8.0f}",
              flush=True)
    both = ("Adam-FD" in sis and "LBFGS-autodiff" in sis)
    moved = both and (sis["Adam-FD"] > si0 + 0.02 or sis["LBFGS-autodiff"] > si0 + 0.02)
    agree = both and abs(sis["Adam-FD"] - sis["LBFGS-autodiff"]) <= 0.10
    if both:
        print(f"  equivalence: |dSI| = {abs(sis['Adam-FD']-sis['LBFGS-autodiff']):.3f}"
              f"  ({'agree' if agree else 'DIFFER'}); progress from start "
              f"{'yes' if moved else 'NO'}", flush=True)
    results["B_headtohead"] = bool(both and moved and agree)
    print(f"  --> {'PASS' if results['B_headtohead'] else 'FAIL'} "
          f"(both reached SI>start and agree within 0.10)", flush=True)

    # ── C. autodiff WAVEFORM optimization (K x T DOF; FD infeasible) ────────
    banner("C. autodiff waveform optimization (K x T DOF) -- memory/timing gate")
    u_init = (best_amps[:, None] * pulse[None, :]) if best_amps is not None else None
    print(f"  waveform DOF = K*T = {K}*{T} = {K*T}  "
          f"(FD would need {K*T+1} forwards/gradient -- infeasible)", flush=True)
    try:
        t0 = time.time()
        wres = run_waveform_optimization(
            fiber_statics_batch=seed_in["fs_batch"], state0_batch=seed_in["s0_batch"],
            Ve_unit=seed_in["Ve_unit"], node_indices=seed_in["node_indices"],
            target_mask=seed_in["target_mask"], weights=seed_in["weights"],
            dt=S.DT, T=T, n_steps=WAVE_ITERS, lr=1e-3, u_init=u_init,
            early_stop_patience=3, verbose=False,
        )
        jax.block_until_ready(wres["best_acts"]); dt_c = time.time() - t0
        print(f"  best loss={min(wres['history']['loss']):.4f}  "
              f"best SI={wres['best_si']:+.3f}", flush=True)
        print(f"  ran {WAVE_ITERS} iters in {dt_c:.1f}s "
              f"({dt_c/max(WAVE_ITERS,1):.1f}s/iter), peak {gpu_peak_mb():.0f} MB",
              flush=True)
        results["C_waveform_autodiff"] = bool(np.isfinite(min(wres["history"]["loss"])))
        print(f"  --> {'PASS' if results['C_waveform_autodiff'] else 'FAIL'} "
              f"(ran + finite at N={N}, T={T})", flush=True)
    except Exception as e:               # noqa: BLE001
        results["C_waveform_autodiff"] = False
        msg = str(e); oom = "RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower()
        print(f"  --> FAIL ({type(e).__name__}{' [OOM]' if oom else ''}: {msg[:200]})",
              flush=True)
        if oom:
            print("  HINT: autodiff backprop OOM. Re-run with MAX_FIBERS=<smaller> "
                  "and/or T_STOP=<smaller>, or use gradient checkpointing.",
                  flush=True)

    banner("SUMMARY")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
    all_ok = all(results.values())
    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'} "
          f"(peak GPU {gpu_peak_mb():.0f} MB)", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
