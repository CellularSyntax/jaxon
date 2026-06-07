"""Copy locked figures from ../outputs/ into manuscript/figures/.

Portable replacement for the Unix `cp -u` calls that don't work under
plain Windows cmd.exe.  Missing source files are a warning, not an
error, so the build proceeds with `\figmaybe` placeholders for figures
that don't yet exist (e.g. Phase 3 sweep still running on the cluster).

Usage:
    python copy_figs.py
"""
from __future__ import annotations
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTROOT = HERE.parent / "outputs"
FIGDIR = HERE / "figures"

FIGURES = {
    OUTROOT / "scaling" / "fig_scaling.png":
        FIGDIR / "fig_scaling.png",
    OUTROOT / "selectivity_sweep_phase3_manuscript" / "fig_seed_0000_rect_xsection.png":
        FIGDIR / "fig_seed_0000_rect_xsection.png",
}

def main() -> int:
    FIGDIR.mkdir(exist_ok=True)
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
    print(f"[figs] done: {copied} copied, {missing} missing")
    return 0  # missing files are not an error

if __name__ == "__main__":
    sys.exit(main())
