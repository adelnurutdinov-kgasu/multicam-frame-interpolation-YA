"""Shim -> scripts/training/eval_color_recon_psnr.py"""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "scripts/training/eval_color_recon_psnr.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")

