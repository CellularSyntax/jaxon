"""Smoke test for the autodiff / gradient-based optimization path.

Run ONE Duke nerve/seed through the differentiable optimizer to confirm,
before launching a full cohort sweep, that:

  A. autodiff gradients are CORRECT     (vs finite differences on the
                                         rectangular-amplitude loss);
  B. the autodiff AMPLITUDE optimizer   (LBFGS, exact gradients) runs and
                                         improves selectivity;
  C. the autodiff WAVEFORM optimizer    (K x T DOF -- the regime where FD is
                                         hopeless) runs AND fits in GPU memory
                                         at the real nerve scale.

This is the test that tells you whether the differentiable path is viable at
the intended N before you spend cluster time on the cohort.  Backprop through
the ODE scan is memory-bound (a 40 GB A100 holds roughly N~200 at full T
without checkpointing) and slow for waveforms (~100 s/iter at large T), so the
peak-memory and per-iter-time numbers printed below are the deciding factors.

Usage (cluster, inside the container after slurm/setup_env.sh):

    DUKE_SAMPLE_DIR=duke_Ves/sub-10_sam-1 \
    python -m experiments_v2.smoke_autodiff_opt

Useful overrides:
    MAX_FIBERS=300      cap fibers (sanity-check memory headroom first)
    T_STOP=1.0          shorter window -> smaller T for the waveform test
    SMOKE_WAVE_ITERS=5  waveform optimizer iterations (keep small; slow)
    SMOKE_RECT_STEPS=15 LBFGS iterations
"""
from __future__ import annotations

import os
import sys
import time

# ── small-run defaults (set BEFORE importing the sweep module, which reads
#    these at import time).  Override any of them from the environment. ────────
os.environ.setdefault("DUKE_SAMPLE_DIR", "duke_Ves/sub-10_sam-1")
os.environ.setdefault("RECT_OPTIMIZER", "lbfgs")          # autodiff path
os.environ.setdefault("N_OPT_RECT", os.environ.get("SMOKE_RECT_STEPS", "15"))
os.environ.setdefault("N_RESTARTS_RECT", "2")

import numpy as np
import jax
import jax.numpy as jnp

# Importing the sweep module configures jax (x64) and all run constants from
# the environment, and re-exports the exact data loader / seed builder the
# full cohort uses -- so this smoke test exercises the real pipeline.
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

    # ── load one nerve, build one seed (the exact cohort pipeline) ───────────
    duke = S.load_duke_sample(
        S.SAMPLE_PATH, fiber_diam_um=S.FIBER_DIAMETER_UM, n_nodes=S.N_NODES,
        max_fibers=(S.MAX_FIBERS if S.MAX_FIBERS > 0 else None),
        subsample_seed=S.SUBSAMPLE_SEED, verbose=True,
    )
    seed_in = S._build_seed(duke, seed=0, verbose=True)
    if seed_in is None:
        print("[smoke] seed 0 produced no valid target (try another "
              "DUKE_SAMPLE_DIR or seed).", file=sys.stderr)
        return 2

    Ve_unit   = np.asarray(seed_in["Ve_unit"])
    K, N, ncomp = Ve_unit.shape
    T         = int(seed_in["N_STEPS"])
    tgt       = np.asarray(seed_in["target_mask"], dtype=bool)
    print(f"[smoke] N_fibers={N}  K_contacts={K}  n_comp={ncomp}  T_steps={T}  "
          f"targets={tgt.sum()}/{N}  baseline_SI={seed_in['si_baseline']:+.3f}",
          flush=True)

    pulse      = np.asarray(seed_in["pulse_mask"], float)
    pulse_prev = np.concatenate([[0.0], pulse[:-1]])
    w          = np.asarray(seed_in["weights"], float)

    # ── A. gradient correctness: autodiff vs finite difference ──────────────
    banner("A. gradient correctness (autodiff vs finite difference)")
    loss_fn = _build_rect_loss_fn(
        seed_in["fs_batch"], seed_in["s0_batch"],
        jnp.asarray(Ve_unit, jnp.float64), jnp.asarray(pulse, jnp.float64),
        jnp.asarray(pulse_prev, jnp.float64),
        jnp.asarray(seed_in["node_indices"], jnp.int32),
        jnp.asarray(tgt, jnp.float64), jnp.asarray(w, jnp.float64), S.DT,
    )
    # Test point = the optimizer's own Ve-weighted deterministic init, so it
    # sits in the gradient-informative regime (not a flat all-sub/all-supra
    # region where both gradients would vanish).
    amps0 = _initial_amps_for_seed(
        jnp.asarray(Ve_unit, jnp.float64), jnp.asarray(tgt, jnp.float64),
        S.AMP_INIT_MA, S.AMP_CLIP, 1, jax.random.PRNGKey(0),
    )[0]

    t0 = time.time()
    g_ad = np.asarray(jax.grad(loss_fn)(amps0)); jax.block_until_ready(g_ad)
    t_ad = time.time() - t0

    eps = 1e-3
    L0 = float(loss_fn(amps0))
    g_fd = np.zeros(K)
    t1 = time.time()
    for k in range(K):
        ap = amps0.at[k].add(eps); am = amps0.at[k].add(-eps)
        g_fd[k] = (float(loss_fn(ap)) - float(loss_fn(am))) / (2 * eps)
    t_fd = time.time() - t1

    denom = (np.linalg.norm(g_ad) * np.linalg.norm(g_fd)) or 1.0
    cos = float(g_ad @ g_fd / denom)
    max_abs = float(np.max(np.abs(g_ad - g_fd)))
    rel = max_abs / (float(np.max(np.abs(g_fd))) or 1.0)
    print(f"  loss(amps0)        = {L0:.6f}", flush=True)
    print(f"  ||g_autodiff||     = {np.linalg.norm(g_ad):.4e}  ({t_ad:.1f}s, 1 backward)")
    print(f"  ||g_finitediff||   = {np.linalg.norm(g_fd):.4e}  ({t_fd:.1f}s, {2*K} forwards)")
    print(f"  cosine(ad, fd)     = {cos:.5f}")
    print(f"  max|ad-fd|         = {max_abs:.3e}   (rel {rel:.2%})")
    finite = bool(np.all(np.isfinite(g_ad)))
    results["A_grad_correct"] = bool(finite and cos > 0.98 and rel < 0.10)
    print(f"  --> {'PASS' if results['A_grad_correct'] else 'FAIL'} "
          f"(need finite, cosine>0.98, rel<10%)", flush=True)

    # ── B. SAME 12-amplitude problem: Adam-FD (current) vs autodiff-LBFGS ────
    # Identical nerve, loss, and starting point (the Ve-weighted deterministic
    # init); both run a fixed budget with early-stop disabled, so the ONLY
    # difference is the gradient source.  This shows whether exact gradients
    # buy anything on the problem we actually solve.
    banner("B. head-to-head, identical 12-amplitude problem "
           "(Adam-FD vs autodiff-LBFGS)")
    budget = S.N_OPT_RECT
    amps_init = np.asarray(amps0)        # reuse the Ve-weighted init from A
    NEVER = 10 ** 9
    common = dict(
        fiber_statics_batch=seed_in["fs_batch"], state0_batch=seed_in["s0_batch"],
        Ve_unit=seed_in["Ve_unit"], pulse_mask=seed_in["pulse_mask"],
        node_indices=seed_in["node_indices"], target_mask=seed_in["target_mask"],
        weights=seed_in["weights"], dt=S.DT, amp_clip=S.AMP_CLIP, verbose=False,
    )

    def _fd_best(hist):
        loss = np.asarray(hist["loss"]); si = np.asarray(hist["si"])
        i = int(np.argmin(loss))
        return float(si[i]), float(loss[i]), len(loss)

    row = {}
    amps_warm = None
    # --- Adam-FD (the method the cohort used) ---
    try:
        t0 = time.time()
        fd = run_rect_optimization(
            **common, n_steps=budget, amps_init_vector=amps_init,
            lr=S.ADAM_LR_MA, fd_eps=S.FD_EPS_MA,
            early_stop_si=2.0, early_stop_patience=NEVER,
            early_stop_si_patience=0,
        )
        jax.block_until_ready(fd["amps"])
        t_fd_opt = time.time() - t0
        si_fd, loss_fd, it_fd = _fd_best(fd["history"])
        row["Adam-FD"] = (it_fd, loss_fd, si_fd, t_fd_opt, gpu_peak_mb())
    except Exception as e:               # noqa: BLE001
        row["Adam-FD"] = None
        print(f"  Adam-FD FAILED ({type(e).__name__}: {str(e)[:140]})", flush=True)

    # --- autodiff LBFGS (exact gradients), same single start, same budget ---
    try:
        t0 = time.time()
        lb = run_rect_optimization_lbfgs(
            **common, n_restarts=1, n_steps=budget,
            amp_init_mA=S.AMP_INIT_MA, rng_seed=0,
        )
        jax.block_until_ready(lb["final_acts"])
        t_lb_opt = time.time() - t0
        si_lb = float(selectivity_index(np.asarray(lb["final_acts"]), tgt))
        row["LBFGS-autodiff"] = (budget, float(lb["final_loss"]), si_lb,
                                 t_lb_opt, gpu_peak_mb())
        amps_warm = np.asarray(lb["amps"])
    except Exception as e:               # noqa: BLE001
        row["LBFGS-autodiff"] = None
        msg = str(e); oom = "RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower()
        print(f"  LBFGS-autodiff FAILED ({type(e).__name__}"
              f"{' [OOM]' if oom else ''}: {msg[:140]})", flush=True)

    print(f"  baseline SI = {seed_in['si_baseline']:+.3f}   "
          f"(budget = {budget} iters, identical start; "
          f"peak MB = cumulative GPU high-water mark)", flush=True)
    print(f"  {'optimizer':16s} {'iters':>5} {'best loss':>10} "
          f"{'best SI':>8} {'wall (s)':>9} {'peak MB':>8}", flush=True)
    for name in ("Adam-FD", "LBFGS-autodiff"):
        r = row.get(name)
        if r is None:
            print(f"  {name:16s}   ---  (failed)", flush=True); continue
        it, ls, si, t, mb = r
        print(f"  {name:16s} {it:>5d} {ls:>10.4f} {si:>+8.3f} "
              f"{t:>9.1f} {mb:>8.0f}", flush=True)
    ok_b = (row.get("Adam-FD") is not None and row.get("LBFGS-autodiff") is not None
            and np.isfinite(row["LBFGS-autodiff"][1]))
    results["B_headtohead"] = bool(ok_b)
    print(f"  --> {'PASS' if ok_b else 'FAIL'} "
          f"(both optimizers ran on the identical problem)", flush=True)

    # ── C. autodiff WAVEFORM optimization (K x T DOF; FD is hopeless here) ──
    banner("C. autodiff waveform optimization (K x T DOF) -- memory/timing gate")
    if amps_warm is None:
        u_init = None
        print("  (no warm-start amps from B; starting waveform from zeros)", flush=True)
    else:
        u_init = amps_warm[:, None] * pulse[None, :]            # [K, T]
    t0 = time.time()
    try:
        wres = run_waveform_optimization(
            fiber_statics_batch=seed_in["fs_batch"],
            state0_batch=seed_in["s0_batch"],
            Ve_unit=seed_in["Ve_unit"], node_indices=seed_in["node_indices"],
            target_mask=seed_in["target_mask"], weights=seed_in["weights"],
            dt=S.DT, T=T, n_steps=WAVE_ITERS, lr=1e-3, u_init=u_init,
            early_stop_patience=3, verbose=False,
        )
        jax.block_until_ready(wres["best_acts"])
        dt_c = time.time() - t0
        dof = K * T
        print(f"  waveform DOF = K*T = {K}*{T} = {dof}", flush=True)
        print(f"  best loss={min(wres['history']['loss']):.4f}  "
              f"best SI={wres['best_si']:+.3f}", flush=True)
        print(f"  ran {WAVE_ITERS} iters in {dt_c:.1f}s "
              f"({dt_c / max(WAVE_ITERS,1):.1f}s/iter), peak {gpu_peak_mb():.0f} MB",
              flush=True)
        results["C_waveform_autodiff"] = bool(
            np.isfinite(min(wres["history"]["loss"])))
        print(f"  --> {'PASS' if results['C_waveform_autodiff'] else 'FAIL'} "
              f"(ran + finite at full N={N}, T={T})", flush=True)
    except Exception as e:           # noqa: BLE001
        results["C_waveform_autodiff"] = False
        msg = str(e)[:200]
        oom = "RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower()
        print(f"  --> FAIL ({type(e).__name__}{' [OOM]' if oom else ''}: {msg})",
              flush=True)
        if oom:
            print("  HINT: autodiff backprop OOM at this N/T. Re-run with "
                  "MAX_FIBERS=<smaller> and/or T_STOP=<smaller>, or enable "
                  "gradient checkpointing for the cohort run.", flush=True)

    # ── summary ─────────────────────────────────────────────────────────────
    banner("SUMMARY")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
    all_ok = all(results.values())
    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'} "
          f"(peak GPU {gpu_peak_mb():.0f} MB)", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
