"""Reviewer #1: validate the population-level selectivity metric against NEURON.

For an optimized config (saved amps), re-score recruitment of the FULL fiber
population in NEURON (pyfibers) under the IDENTICAL extracellular field, and
compare the NEURON selectivity index to jaxon's.

jaxon and NEURON are fed the same per-compartment extracellular potential
  Ve(t,x) = -sum_k amps_k * pulse(t) * Ve_unit[k,fiber,x]
(the convention jaxon's solver integrates).  A fiber is 'fired' when an action
potential reaches BOTH end nodes -- the NEURON analogue of jaxon's
min(end-node) activation proxy.  SI = frac_fired_target - frac_fired_offtarget.

Run (PowerShell, jaxon env with c:\\nrn826 on PYTHONPATH/PATH):
  python -m experiments_v2.neuron_pop_validate --sample human_sub-50_sam-2 --seed 0 --maxfib 80
  python -m experiments_v2.neuron_pop_validate --sample human_sub-50_sam-2 --seed 0   # full pop
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# NEURON 8.2 needs the current hoc build on the path (Windows).
if sys.platform == "win32":
    _N = r"c:\nrn826\lib\python"
    if os.path.isdir(_N) and _N not in sys.path[:3]:
        sys.path.insert(0, _N)

DT = 0.005
DELAY, PW, RATIO = 1.0, 0.5, 4.0     # biphasic-asym (matches the run)
TSTOP = 6.0                           # > pulse end (3.5) + propagation margin


def biphasic_asym(t):
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    cath = (t >= DELAY) & (t < DELAY + PW)
    anod = (t >= DELAY + PW) & (t < DELAY + PW + PW * RATIO)
    out[cath] = 1.0
    out[anod] = -1.0 / RATIO
    return out


def jaxon_si(duke, amps, target_mask, node_indices):
    """Recompute jaxon recruitment on this exact (fresh) population."""
    os.environ.setdefault("JAXON_SOFT_TEMPERATURE", "0.1")
    import jax; jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jaxon.stim.batch_solve import (
        stack_fiber_statics, initial_states_batch, batch_integrate_m_max)
    from jaxon.optim.losses import activation_proxy_batch, selectivity_index
    geoms = duke["geoms"]
    fs = stack_fiber_statics(geoms, DT); s0 = initial_states_batch(geoms)
    n = int(TSTOP / DT); t = (np.arange(n) + 1) * DT
    pulse = jnp.asarray(biphasic_asym(t))
    amps_j = jnp.asarray(amps); Ve = jnp.asarray(duke["Ve_unit"])
    u = amps_j[:, None] * pulse[None, :]
    Ve_seq = jnp.einsum("kt,kfn->ftn", -u, Ve)
    m_max = batch_integrate_m_max(fs, s0, Ve_seq, DT)
    acts = np.asarray(activation_proxy_batch(m_max, jnp.asarray(node_indices, jnp.int32)))
    return acts, float(selectivity_index(acts, target_mask))


def neuron_fired_population(Ve_unit, amps, idx, diam=5.7, n_nodes=21,
                            vm_thresh_mV=0.0, verbose=True):
    """Per-fiber NEURON recruitment under the optimized field. Returns bool[len(idx)].

    'fired' = action potential (Vm > vm_thresh_mV) reaches BOTH end nodes -- the
    NEURON analogue of jaxon's min(end-node) activation proxy.
    """
    from jaxon.nrn_baseline import build_mrg_pyfibers
    from pyfibers import ScaledStim
    fiber = build_mrg_pyfibers(diameter=diam, n_nodes=n_nodes, temperature=37.0)
    fiber.record_vm()
    npot = len(fiber.potentials); ncomp = Ve_unit.shape[2]
    if verbose:
        print(f"  pyfibers sections={npot}  jaxon n_comp={ncomp}  "
              f"nodecount={fiber.nodecount}  vm_nodes={len(fiber.vm)}", flush=True)
    if npot != ncomp:
        raise RuntimeError(f"compartment mismatch: pyfibers {npot} vs jaxon {ncomp}")
    stim = ScaledStim(waveform=biphasic_asym, dt=DT, tstop=TSTOP)
    fired = np.zeros(len(idx), dtype=bool)
    t0 = time.time()
    for i, f in enumerate(idx):
        pot = -(amps[:, None] * Ve_unit[:, f, :]).sum(0)     # [n_comp] mV
        fiber.potentials = pot
        stim.run_sim(stimamp=1.0, fiber=fiber, ap_detect_location=0.5,
                     fail_on_end_excitation=False)
        v_first = np.asarray(fiber.vm[0]).max()
        v_last  = np.asarray(fiber.vm[-1]).max()
        fired[i] = (v_first > vm_thresh_mV) and (v_last > vm_thresh_mV)
        if verbose and (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(idx)} fibers  ({time.time()-t0:.0f}s)", flush=True)
    return fired


def si_from_fired(fired, target):
    target = np.asarray(target, bool)
    nt, no = target.sum(), (~target).sum()
    tr = fired[target].sum() / max(nt, 1)
    orate = fired[~target].sum() / max(no, 1)
    return float(tr - orate), float(tr), float(orate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maxfib", type=int, default=0, help="0=full population")
    ap.add_argument("--sweep_root", default="outputs/duke_sweeps")
    ap.add_argument("--ves_root", default="duke_Ves")
    ap.add_argument("--out", default="outputs/reviewer_analyses/neuron_pop")
    args = ap.parse_args()

    from experiments_v2.duke_loader import load_duke_sample, cluster_target_mask
    jp = os.path.join(ROOT, args.sweep_root, args.sample, f"data_seed_{args.seed:04d}.json")
    d = json.load(open(jp))
    amps = np.asarray(d["rect"]["amps_mA"], float)
    target_ids = (d.get("cluster") or {}).get("target_ids") or []
    json_si = float(d["rect"]["achievable_si"])
    print(f"[{args.sample} seed {args.seed}] K={amps.size} amps; "
          f"target_ids={target_ids}; JSON achievable_si={json_si:.3f}", flush=True)

    sample_dir = os.path.join(ROOT, args.ves_root, args.sample)
    duke = load_duke_sample(sample_dir, fiber_diam_um=5.7, n_nodes=21, verbose=False)
    tmask = cluster_target_mask(duke["nerve_geom"], duke["fasc_id"],
                                duke["fasc_meta"], target_ids)
    n_fib = duke["Ve_unit"].shape[1]
    print(f"  fresh load: n_fibers={n_fib}  n_target={int(tmask.sum())}", flush=True)

    print("  computing jaxon SI on fresh population ...", flush=True)
    acts, sij = jaxon_si(duke, amps, tmask, duke["node_indices"])
    jfired = acts > 0.5
    sij2, jt, jo = si_from_fired(jfired, tmask)
    print(f"  jaxon SI (fresh) = {sij:.3f}  [tgt {jt:.2f} off {jo:.2f}]  "
          f"(JSON {json_si:.3f})", flush=True)

    idx = np.arange(n_fib)
    if args.maxfib and args.maxfib < n_fib:
        rng = np.random.default_rng(0)
        tgt_i = np.where(tmask)[0]; off_i = np.where(~tmask)[0]
        k = args.maxfib // 2
        idx = np.sort(np.concatenate([rng.choice(tgt_i, min(k, tgt_i.size), False),
                                      rng.choice(off_i, min(k, off_i.size), False)]))
        print(f"  TEST subsample: {idx.size} fibers", flush=True)

    print("  running NEURON population ...", flush=True)
    nfired = neuron_fired_population(duke["Ve_unit"], amps, idx)
    sin, nt_, no_ = si_from_fired(nfired, tmask[idx])
    agree = float(np.mean(nfired == jfired[idx]))
    print(f"\n  === RESULT ({idx.size} fibers) ===", flush=True)
    print(f"  NEURON SI = {sin:.3f}  [tgt {nt_:.2f} off {no_:.2f}]", flush=True)
    print(f"  jaxon  SI = {si_from_fired(jfired[idx], tmask[idx])[0]:.3f}", flush=True)
    print(f"  per-fiber fired agreement: {100*agree:.1f}%", flush=True)

    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
    op = os.path.join(ROOT, args.out, f"{args.sample}_seed{args.seed}.json")
    json.dump(dict(sample=args.sample, seed=args.seed, n_eval=int(idx.size),
                   json_si=json_si, jaxon_si_full=sij,
                   jaxon_si_eval=si_from_fired(jfired[idx], tmask[idx])[0],
                   neuron_si=sin, agreement=agree,
                   neuron_tgt=nt_, neuron_off=no_), open(op, "w"), indent=2)
    print(f"  saved {op}", flush=True)


if __name__ == "__main__":
    main()
