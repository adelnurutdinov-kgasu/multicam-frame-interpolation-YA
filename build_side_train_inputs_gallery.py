"""Shim → scripts/training/build_side_train_inputs_gallery.py"""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "scripts/training/build_side_train_inputs_gallery.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
