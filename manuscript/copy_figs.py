"""Sync figures into manuscript/figures/.

Two-step pipeline:

  1. Run make_figures.py to (re)generate the composite multi-panel
     figures that compose data from multiple outputs/*/data_*.json
     files.  These are the figures the manuscript actually cites.

  2. Copy any single-source figures (e.g. selectivity cross-section
     PNGs that aren't composed but are referenced as-is) from
     ../outputs/ into figures/.

Missing source files are a warning, not an error, so the build
proceeds with \figmaybe placeholders for figures that don't yet exist
(e.g. while Phase 3 sweep is still running on the cluster).

Usage:
    python copy_figs.py
"""
from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Search both locations: manuscript-local outputs/ (fresh sweep data
# downloaded from the cluster), and the repo outputs/ (older /
# canonical artifacts).  Prefer the local copy when both exist.
_LOCAL_OUTROOT = HERE / "outputs"
_REPO_OUTROOT  = HERE.parent / "outputs"
OUTROOT = _REPO_OUTROOT   # kept for back-compat
FIGDIR = HERE / "figures"

def _resolve(rel: str) -> Path:
    """Resolve outputs/<rel>, preferring manuscript/outputs/ if present."""
    local = _LOCAL_OUTROOT / rel
    repo  = _REPO_OUTROOT  / rel
    return local if local.exists() else repo

FIGURES = {
    # Selectivity exemplar (seed 0 rect cross-section).  This figure is
    # referenced as-is rather than composed from JSON, so it lives in
    # the copy step, not in make_figures.py.  _resolve() picks the
    # manuscript-local outputs/ copy if present, otherwise the repo's.
    _resolve("selectivity_sweep_phase3_manuscript/fig_seed_0000_rect_xsection.png"):
        FIGDIR / "fig_seed_0000_rect_xsection.png",
}

def _run_make_figures() -> int:
    """Step 1: regenerate composite multi-panel figures from JSON sources."""
    print("[figs] step 1: building composite figures via make_figures.py")
    script = HERE / "make_figures.py"
    if not script.exists():
        print(f"[figs] WARN: {script} not found, skipping composite build")
        return 0
    rc = subprocess.call([sys.executable, str(script)])
    if rc != 0:
        print(f"[figs] make_figures.py exited {rc} -- continuing anyway")
    return rc


def _copy_single_source() -> tuple[int, int]:
    """Step 2: copy single-source figures from outputs/ into figures/."""
    print("[figs] step 2: copying single-source figures")
    copied = missing = 0
    for src, dst in FIGURES.items():
        if not src.exists():
            print(f"[figs] WARN: source not found {src} -- skipping")
            missing += 1
            continue
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            continue
        shutil.copy2(src, dst)
        print(f"[figs] copied {src.name}")
        copied += 1
    return copied, missing


def main() -> int:
    FIGDIR.mkdir(exist_ok=True)
    _run_make_figures()
    copied, missing = _copy_single_source()
    print(f"[figs] done: {copied} single-source copied, {missing} missing")
    return 0  # missing files are not an error

if __name__ == "__main__":
    sys.exit(main())
