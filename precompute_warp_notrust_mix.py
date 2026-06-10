"""Shim -> scripts/training/precompute_warp_notrust_mix.py"""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "scripts/training/precompute_warp_notrust_mix.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")

