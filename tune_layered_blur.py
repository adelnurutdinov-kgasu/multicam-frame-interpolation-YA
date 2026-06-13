"""Launcher -> scripts/tuning/tune_layered_blur.py"""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "tuning" / "tune_layered_blur.py"), run_name="__main__")
