"""Sparse-to-dense transfer evaluation — cluster script.

Loads an existing data_seed_NNNN.json (dense optimisation result) to
obtain the target fascicle configuration, then for each sparse fibre-
sampling strategy (centroid, 1/fasc, 3/fasc, 10/fasc):

  1. Subsamples the Duke nerve to the sparse fibre set.
  2. Runs rect optimisation on the sparse set (smart-init Adam-FD,
     same hyperparameters as the original cluster run).
  3. Evaluates the resulting amps on the FULL dense population →
     SI_transfer.

Outputs one file per seed:
  outputs/duke_sweeps/<sample>/transfer_eval_seed_NNNN.json

Dense data_seed_NNNN.json must already exist (run
selectivity_sweep_duke.py first).  Seeds for which transfer_eval_
seed_NNNN.json already exists are skipped.

Env-var interface (same pattern as selectivity_sweep_duke):
  DUKE_SAMPLE_DIR   path to the Duke bundle, e.g. duke_Ves/sub-10_sam-1
  SEED_START        first seed (inclusive), default 0
  SEED_END          last seed (exclusive), default 25
  N_OPT_ITERS       Adam-FD iterations per strategy, default 90

All other optimiser knobs default to the same values as
selectivity_sweep_duke.py:
  FIBER_DIAMETER_UM  N_NODES  DT  T_STOP  DELAY_MS  PW_MS
  AMP_CLIP_LO/HI  ADAM_LR_MA  FD_EPS_MA  FD_EPS_SMART_MA
  PROBE_MAGS_MA  SPARSE_N_PER_FASCICLE_LIST  SPARSE_SWEEP_RNG_SEED
  EARLY_STOP_SI

Example SLURM array submission:
  sbatch --array=0-28 --partition=a16 \\
    --wrap='python -m experiments_v2.sparse_transfer_sweep' \\
    --export=DUKE_SAMPLE_DIR=duke_Ves/sub-10_sam-1,SEED_START=$SLURM_ARRAY_TASK_ID,SEED_END=$((SLURM_ARRAY_TASK_ID+1))
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from pathlib import Path

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jaxon.stim.batch_solve import (
    stack_fiber_statics, initial_states_batch, batch_integrate_m_max,
)
from jaxon.optim.optimizer import run_rect_optimization
from jaxon.optim.losses import activation_proxy_batch, selectivity_index
from jaxon.nerve.geometry import NerveGeometry
from experiments_v2.utils import ensure_dir, save_json
from experiments_v2.duke_loader import load_duke_sample, cluster_target_mask

# ── sample location ────────────────────────────────────────────────────────────
DUKE_SAMPLE_DIR = os.environ.get("DUKE_SAMPLE_DIR", "").strip()
if not DUKE_SAMPLE_DIR:
    raise RuntimeError(
        "sparse_transfer_sweep: set DUKE_SAMPLE_DIR "
        "(e.g. DUKE_SAMPLE_DIR=duke_Ves/sub-10_sam-1)"
    )
SAMPLE_PATH = Path(DUKE_SAMPLE_DIR)
if not SAMPLE_PATH.is_absolute():
    SAMPLE_PATH = ROOT / SAMPLE_PATH
SAMPLE_NAME = SAMPLE_PATH.name

_OUT_OVERRIDE = os.environ.get("JAXON_OUTPUT_DIR", "").strip()
if _OUT_OVERRIDE:
    _p = pathlib.Path(_OUT_OVERRIDE)
    OUT = ensure_dir(_p if _p.is_absolute() else ROOT / _p)
else:
    OUT = ensure_dir(ROOT / "outputs" / "duke_sweeps" / SAMPLE_NAME)


def _env_int(name, default): return int(os.environ.get(name, default))
def _env_flt(name, default): return float(os.environ.get(name, default))

FIBER_DIAMETER_UM = _env_flt("FIBER_DIAMETER_UM", 5.7)
N_NODES           = _env_int("N_NODES", 21)
DT                = _env_flt("DT", 0.005)
T_STOP            = _env_flt("T_STOP", 3.0)
DELAY_MS          = _env_flt("DELAY_MS", 1.0)
PW_MS             = _env_flt("PW_MS", 1.0)
SEED_START        = _env_int("SEED_START", 0)
SEED_END          = _env_int("SEED_END", 25)
N_OPT_ITERS       = _env_int("N_OPT_ITERS", 90)   # match cluster: max(N_OPT_RECT*3, 30)
_AMP_CLIP_LO      = _env_flt("AMP_CLIP_LO", -2.00)
_AMP_CLIP_HI      = _env_flt("AMP_CLIP_HI",  2.00)
AMP_CLIP          = (_AMP_CLIP_LO, _AMP_CLIP_HI)
ADAM_LR_MA        = _env_flt("ADAM_LR_MA",        0.005)
FD_EPS_MA         = _env_flt("FD_EPS_MA",         0.030)
FD_EPS_SMART_MA   = _env_flt("FD_EPS_SMART_MA",   0.010)
EARLY_STOP_SI     = _env_flt("EARLY_STOP_SI",      0.95)
SPARSE_N_PER_FASCICLE_LIST = [
    int(x.strip()) for x in
    os.environ.get("SPARSE_N_PER_FASCICLE_LIST", "1,3,10").split(",")
    if x.strip()
]
SPARSE_SWEEP_RNG_SEED = _env_int("SPARSE_SWEEP_RNG_SEED", 42)
PROBE_MAGS_MA = [
    float(x.strip()) for x in
    os.environ.get("PROBE_MAGS_MA",
                   "-0.02,-0.05,-0.10,-0.20,-0.40,-0.80,-1.20,-1.50").split(",")
    if x.strip()
]


# ── helper functions (adapted from selectivity_sweep_duke.py) ──────────────────

def _build_pulse_mask(t_grid: np.ndarray, delay_ms: float, pw_ms: float,
                      shape: str, asym_ratio: float = 4.0) -> np.ndarray:
    shape = shape.strip().lower()
    pulse = np.zeros_like(t_grid, dtype=np.float64)
    if shape == "monophasic":
        m = (t_grid >= delay_ms) & (t_grid < delay_ms + pw_ms)
        pulse[m] = 1.0
    elif shape == "biphasic_sym":
        half = pw_ms / 2.0
        pulse[(t_grid >= delay_ms) & (t_grid < delay_ms + half)]          =  1.0
        pulse[(t_grid >= delay_ms + half) & (t_grid < delay_ms + pw_ms)]  = -1.0
    elif shape == "biphasic_asym":
        cath_dur = pw_ms
        anod_dur = pw_ms * float(asym_ratio)
        pulse[(t_grid >= delay_ms) & (t_grid < delay_ms + cath_dur)]                       = 1.0
        pulse[(t_grid >= delay_ms + cath_dur) & (t_grid < delay_ms + cath_dur + anod_dur)] = -1.0 / float(asym_ratio)
    else:
        raise ValueError(f"Unknown pulse shape {shape!r}")
    return pulse


def _class_balanced_weights(target_mask: np.ndarray) -> np.ndarray:
    target_mask = np.asarray(target_mask, dtype=bool)
    n_t  = int(target_mask.sum())
    n_nt = int((~target_mask).sum())
    n    = int(target_mask.size)
    if n_t == 0 or n_nt == 0:
        return np.ones(n, dtype=np.float64) / max(n, 1)
    return np.where(target_mask, 0.5 / n_t, 0.5 / n_nt).astype(np.float64)


def _sparse_subsample_duke(duke: dict, n_per_fascicle: int | str,
                            rng_seed: int = 0) -> dict:
    fasc_id   = duke["fasc_id"]
    fasc_meta = duke["fasc_meta"]
    nerve     = duke["nerve_geom"]
    fiber_x   = nerve.fiber_x_um
    fiber_y   = nerve.fiber_y_um

    rng = np.random.default_rng(int(rng_seed))
    picked: list[int] = []
    for fid in np.unique(fasc_id):
        idx_f = np.where(fasc_id == fid)[0]
        if len(idx_f) == 0:
            continue
        if n_per_fascicle == "centroid":
            meta = next((m for m in fasc_meta if m["id"] == fid), None)
            if meta is not None:
                cx = float(meta["centroid_xy_um"][0])
                cy = float(meta["centroid_xy_um"][1])
            else:
                cx = float(fiber_x[idx_f].mean())
                cy = float(fiber_y[idx_f].mean())
            dists = np.hypot(fiber_x[idx_f] - cx, fiber_y[idx_f] - cy)
            picked.append(int(idx_f[int(np.argmin(dists))]))
        else:
            n = min(int(n_per_fascicle), len(idx_f))
            picked.extend(int(i) for i in rng.choice(idx_f, size=n, replace=False))

    keep = np.sort(np.array(picked, dtype=int))
    n_k  = len(keep)
    sparse_nerve = NerveGeometry(
        n_fibers=n_k,
        fiber_x_um=fiber_x[keep].copy(),
        fiber_y_um=fiber_y[keep].copy(),
        fiber_diam=np.full(n_k, float(nerve.fiber_diam[0]), dtype=np.float64),
        target_mask=np.zeros(n_k, dtype=bool),
        fascicles=nerve.fascicles,
        divider_angle_deg=nerve.divider_angle_deg,
    )
    return {
        **duke,
        "nerve_geom":   sparse_nerve,
        "fasc_id":      fasc_id[keep].copy(),
        "Ve_unit":      duke["Ve_unit"][:, keep, :],
        "node_indices": duke["node_indices"][keep].copy(),
        "geoms":        [duke["geoms"][i] for i in keep],
    }


def _smart_spatial_pattern(Ve_unit: np.ndarray,
                            target_mask: np.ndarray) -> np.ndarray:
    Ve_unit     = np.asarray(Ve_unit)
    target_mask = np.asarray(target_mask, dtype=bool)
    K = Ve_unit.shape[0]
    if int(target_mask.sum()) == 0 or int((~target_mask).sum()) == 0:
        return np.zeros(K, dtype=np.float64)
    Ve_abs_max  = np.max(np.abs(Ve_unit), axis=2)
    mean_tgt    = Ve_abs_max[:, target_mask].mean(axis=1)
    mean_off    = Ve_abs_max[:, ~target_mask].mean(axis=1)
    raw_contrast = (mean_tgt - mean_off) / (mean_tgt + mean_off + 1e-10)
    rng = float(raw_contrast.max() - raw_contrast.min())
    if rng > 1e-10:
        contrast = 2.0 * (raw_contrast - raw_contrast.min()) / rng - 1.0
    else:
        contrast = np.zeros_like(raw_contrast)
    return contrast.astype(np.float64)


def _phi_to_compass(phi_deg: float) -> str:
    phi = float(phi_deg) % 360.0
    compass = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    return compass[int(np.round(phi / 45.0)) % 8]


def _angular_column_groups(contact_xyz_um: np.ndarray,
                            xy_tol_um: float = 50.0) -> list[list[int]]:
    K = int(contact_xyz_um.shape[0])
    contact_xy = np.asarray(contact_xyz_um, dtype=np.float64)[:, :2]
    assigned = np.zeros(K, dtype=bool)
    groups: list[list[int]] = []
    for i in range(K):
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True
        for j in range(i + 1, K):
            if assigned[j]:
                continue
            if float(np.hypot(contact_xy[i, 0] - contact_xy[j, 0],
                               contact_xy[i, 1] - contact_xy[j, 1])) < xy_tol_um:
                group.append(j)
                assigned[j] = True
        groups.append(group)
    return groups


def _tripolar_pattern_for_column(column: list[int], n_contacts: int,
                                  contact_z_um: np.ndarray) -> np.ndarray:
    col_sorted = sorted(column, key=lambda k: float(contact_z_um[k]))
    n_col = len(col_sorted)
    pattern = np.zeros(n_contacts, dtype=np.float64)
    if n_col == 1:
        pattern[col_sorted[0]] = -1.0
    elif n_col == 2:
        pattern[col_sorted[0]] =  1.0
        pattern[col_sorted[1]] = -1.0
    else:
        mid = n_col // 2
        outer_amp = 1.0 / float(n_col - 1)
        for idx, k in enumerate(col_sorted):
            pattern[k] = -1.0 if idx == mid else outer_amp
    return pattern


def _sparse_tripolar_patterns_per_column(
        spatial_contrast: np.ndarray,
        contact_xyz_um: np.ndarray) -> dict[str, np.ndarray]:
    K = int(len(spatial_contrast))
    contact_z = np.asarray(contact_xyz_um, dtype=np.float64)[:, 2]
    contact_xy = np.asarray(contact_xyz_um, dtype=np.float64)[:, :2]
    groups = _angular_column_groups(contact_xyz_um)
    patterns: dict[str, np.ndarray] = {}
    used: dict[str, int] = {}
    for grp in groups:
        phi_deg = float(np.degrees(np.arctan2(
            np.mean([contact_xy[k, 1] for k in grp]),
            np.mean([contact_xy[k, 0] for k in grp]),
        )))
        base = f"tripolar_{_phi_to_compass(phi_deg)}"
        name = base
        if name in patterns:
            used[base] = used.get(base, 1) + 1
            name = f"{base}{used[base]}"
        patterns[name] = _tripolar_pattern_for_column(grp, K, contact_z)
    return patterns


def _magnitude_probe(spatial_patterns: dict,
                     probe_mags_mA: list,
                     target_mask: np.ndarray,
                     fs_batch, s0_batch,
                     Ve_unit_j, pulse_mask_j, node_idx_j,
                     dt: float) -> tuple:
    """Return (best_amps_vec, best_mag, best_score, best_acts, history, best_pattern_name)."""
    target_mask_bool = np.asarray(target_mask, dtype=bool)
    n_t   = int(target_mask_bool.sum())
    n_off = int((~target_mask_bool).sum())

    @jax.jit
    def _fwd(amps):
        u      = amps[:, None] * pulse_mask_j[None, :]
        Ve_seq = jnp.einsum("kt,kfn->ftn", -u, Ve_unit_j)
        m_max  = batch_integrate_m_max(fs_batch, s0_batch, Ve_seq, dt)
        return activation_proxy_batch(m_max, node_idx_j)

    unique_abs  = sorted({abs(float(m)) for m in probe_mags_mA})
    signed_mags = []
    for m_abs in unique_abs:
        signed_mags.extend([-m_abs, +m_abs])

    active_patterns = {n: p for n, p in spatial_patterns.items()
                       if np.any(np.asarray(p) != 0)}

    best_score = -float("inf")
    best_mag   = float(signed_mags[0]) if signed_mags else 0.0
    best_name  = next(iter(active_patterns.keys())) if active_patterns else "none"
    best_amps: np.ndarray | None = None
    best_acts: np.ndarray | None = None
    history: list[dict] = []
    for pat_name, spatial in active_patterns.items():
        for mag in signed_mags:
            amps_vec = float(mag) * np.asarray(spatial, dtype=np.float64)
            acts_np  = np.asarray(_fwd(jnp.asarray(amps_vec, dtype=jnp.float64)))
            fired    = acts_np > 0.5
            nft  = int(np.sum(fired & target_mask_bool))
            nfn  = int(np.sum(fired & ~target_mask_bool))
            score = nft / max(n_t, 1) - nfn / max(n_off, 1)
            history.append({"pattern_name": pat_name, "mag_mA": float(mag),
                             "score": float(score)})
            if score > best_score:
                best_score = score
                best_mag   = float(mag)
                best_name  = pat_name
                best_amps  = amps_vec.copy()
                best_acts  = acts_np
    return (best_amps, best_mag, float(best_score), best_acts, history, best_name)


def _eval_amps_on_dense(amps_mA: np.ndarray,
                        Ve_unit: np.ndarray,
                        fs_batch, s0_batch,
                        pulse_mask: np.ndarray,
                        node_indices: np.ndarray,
                        target_mask: np.ndarray,
                        dt: float) -> float:
    """One forward pass: sparse amps applied to full dense population → SI."""
    amps_j     = jnp.asarray(amps_mA, dtype=jnp.float64)
    Ve_unit_j  = jnp.asarray(Ve_unit, dtype=jnp.float64)
    pulse_j    = jnp.asarray(pulse_mask, dtype=jnp.float64)
    node_idx_j = jnp.asarray(node_indices, dtype=jnp.int32)
    u      = amps_j[:, None] * pulse_j[None, :]
    Ve_seq = jnp.einsum("kt,kfn->ftn", -u, Ve_unit_j)
    m_max  = batch_integrate_m_max(fs_batch, s0_batch, Ve_seq, dt)
    acts   = np.asarray(activation_proxy_batch(m_max, node_idx_j))
    return float(selectivity_index(acts, target_mask))


# ── per-seed transfer evaluation ───────────────────────────────────────────────

def _run_transfer_eval(duke: dict, seed: int, dense: dict,
                       dense_fs_batch, dense_s0_batch,
                       dense_target_mask: np.ndarray,
                       dense_pulse_mask: np.ndarray) -> dict:
    label      = f"[{SAMPLE_NAME} | seed {seed:4d}]"
    target_ids = list(dense["cluster"]["target_ids"])
    nerve      = duke["nerve_geom"]

    strategies = [("centroid", "centroid")] + [
        ("random", n) for n in SPARSE_N_PER_FASCICLE_LIST
    ]
    results = []
    for strat_name, n_per_fasc in strategies:
        lbl = "centroid" if strat_name == "centroid" else f"{n_per_fasc}/fasc"
        t0 = time.time()
        try:
            sd = _sparse_subsample_duke(duke, n_per_fasc,
                                         rng_seed=SPARSE_SWEEP_RNG_SEED)
        except Exception as e:
            print(f"{label} [{lbl}] subsample failed: {e}", flush=True)
            continue

        target_mask_sp = cluster_target_mask(
            sd["nerve_geom"], sd["fasc_id"], sd["fasc_meta"], target_ids,
        )
        n_tot = int(sd["nerve_geom"].n_fibers)
        n_tgt = int(target_mask_sp.sum())
        if n_tgt == 0:
            print(f"{label} [{lbl}] 0 target fibers — skip", flush=True)
            results.append({
                "strategy": strat_name,
                "n_per_fascicle": 1 if strat_name == "centroid" else int(n_per_fasc),
                "n_fibers_sparse": n_tot, "n_target_sparse": 0,
                "si_sparse": None, "si_transfer": None, "amps_mA": None,
                "wall_s": 0.0, "skipped": True,
            })
            continue

        sparse_fs  = stack_fiber_statics(sd["geoms"], DT)
        sparse_s0  = initial_states_batch(sd["geoms"])
        weights_sp = _class_balanced_weights(target_mask_sp)

        Ve_unit_j_sp  = jnp.asarray(sd["Ve_unit"], dtype=jnp.float64)
        pulse_j       = jnp.asarray(dense_pulse_mask, dtype=jnp.float64)
        node_idx_j_sp = jnp.asarray(sd["node_indices"], dtype=jnp.int32)

        # Smart init: spatial contrast + magnitude probe
        spatial_bipolar = _smart_spatial_pattern(sd["Ve_unit"], target_mask_sp)
        tripolar        = _sparse_tripolar_patterns_per_column(
            spatial_bipolar, duke["contact_xyz_um"])
        spatial_patterns = {"bipolar": spatial_bipolar}
        spatial_patterns.update(tripolar)

        (amps_init, best_mag, best_score, _probe_acts,
         _probe_hist, best_pat) = _magnitude_probe(
            spatial_patterns, PROBE_MAGS_MA, target_mask_sp,
            sparse_fs, sparse_s0,
            Ve_unit_j_sp, pulse_j, node_idx_j_sp, DT,
        )
        print(f"{label} [{lbl:10s}] probe winner: {best_pat} "
              f"mag={best_mag:+.3f} mA  score={best_score:+.3f}", flush=True)

        freeze_mask = np.abs(amps_init) < 1e-9
        adam_res = run_rect_optimization(
            fiber_statics_batch=sparse_fs,
            state0_batch=sparse_s0,
            Ve_unit=sd["Ve_unit"],
            pulse_mask=dense_pulse_mask,
            node_indices=sd["node_indices"],
            target_mask=target_mask_sp,
            weights=weights_sp,
            dt=DT, n_steps=N_OPT_ITERS,
            amps_init_vector=amps_init,
            amp_clip=AMP_CLIP,
            lr=ADAM_LR_MA, fd_eps=FD_EPS_SMART_MA,
            freeze_zero_mask=freeze_mask,
            early_stop_si=EARLY_STOP_SI,
            verbose=False,
        )
        loss_hist = np.asarray(adam_res["history"]["loss"])
        best_iter = int(np.argmin(loss_hist))
        best_amps = np.asarray(adam_res["history"]["amps"][best_iter])
        best_acts = np.asarray(adam_res["history"]["acts"][best_iter])
        si_sparse = float(selectivity_index(best_acts, target_mask_sp))

        si_transfer = _eval_amps_on_dense(
            best_amps,
            duke["Ve_unit"], dense_fs_batch, dense_s0_batch,
            dense_pulse_mask, duke["node_indices"],
            dense_target_mask, DT,
        )
        wall_s = time.time() - t0
        print(f"{label} [{lbl:10s}] n={n_tot:4d}  n_tgt={n_tgt:4d}  "
              f"SI_sparse={si_sparse:.3f}  SI_transfer={si_transfer:.3f}  "
              f"({wall_s:.0f}s)", flush=True)

        results.append({
            "strategy":        strat_name,
            "n_per_fascicle":  1 if strat_name == "centroid" else int(n_per_fasc),
            "n_fibers_sparse": n_tot,
            "n_target_sparse": n_tgt,
            "si_sparse":       si_sparse,
            "si_transfer":     si_transfer,
            "amps_mA":         best_amps.tolist(),
            "wall_s":          wall_s,
            "skipped":         False,
        })

    return {
        "sample":          SAMPLE_NAME,
        "seed":            seed,
        "dense_si":        float(dense["rect"]["achievable_si"]),
        "n_fibers_dense":  int(nerve.n_fibers),
        "n_target_dense":  int(dense_target_mask.sum()),
        "target_ids":      target_ids,
        "n_opt_iters":     N_OPT_ITERS,
        "results":         results,
    }


def main():
    print(f"[transfer sweep] {SAMPLE_NAME}  seeds {SEED_START}-{SEED_END-1}  "
          f"n_opt_iters={N_OPT_ITERS}", flush=True)
    duke = load_duke_sample(
        SAMPLE_PATH, fiber_diam_um=FIBER_DIAMETER_UM, n_nodes=N_NODES, verbose=True,
    )
    print(f"[transfer sweep] loaded {SAMPLE_NAME}: "
          f"{duke['nerve_geom'].n_fibers} fibers, "
          f"{len(duke['fasc_meta'])} fascicles", flush=True)

    # Pre-build dense solver statics (shared across seeds)
    dense_fs_batch = stack_fiber_statics(duke["geoms"], DT)
    dense_s0_batch = initial_states_batch(duke["geoms"])

    N_STEPS = int(T_STOP / DT)
    t_grid  = (np.arange(N_STEPS) + 1) * DT

    for s in range(SEED_START, SEED_END):
        dense_path = OUT / f"data_seed_{s:04d}.json"
        out_path   = OUT / f"transfer_eval_seed_{s:04d}.json"

        if not dense_path.exists():
            print(f"[seed {s}] dense JSON missing ({dense_path}) — skip", flush=True)
            continue
        if out_path.exists():
            print(f"[seed {s}] already done ({out_path}) — skip", flush=True)
            continue

        try:
            dense = json.loads(dense_path.read_text())
        except Exception as e:
            print(f"[seed {s}] failed to load dense JSON: {e} — skip", flush=True)
            continue

        cluster_snap = dense.get("cluster")
        if not cluster_snap or not cluster_snap.get("target_ids"):
            print(f"[seed {s}] no cluster info in dense JSON — skip", flush=True)
            continue

        target_ids = list(cluster_snap["target_ids"])
        dense_target_mask = cluster_target_mask(
            duke["nerve_geom"], duke["fasc_id"], duke["fasc_meta"], target_ids,
        )
        pulse_shape = dense.get("pulse_shape", "biphasic_sym")
        pulse_pw    = float(dense.get("pulse_pw_ms", PW_MS))
        dense_pulse_mask = _build_pulse_mask(t_grid, DELAY_MS, pulse_pw, pulse_shape)

        result = _run_transfer_eval(
            duke, s, dense,
            dense_fs_batch, dense_s0_batch,
            dense_target_mask, dense_pulse_mask,
        )
        save_json(result, out_path)
        print(f"[seed {s}] -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
