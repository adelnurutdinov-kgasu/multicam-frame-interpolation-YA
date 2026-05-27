"""Launcher -> scripts/inference/run_test_blend_blur.py"""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "inference" / "run_test_blend_blur.py"), run_name="__main__")
