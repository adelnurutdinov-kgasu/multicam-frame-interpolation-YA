"""Launcher -> scripts/inference/precompute_cv_split.py"""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "inference" / "precompute_cv_split.py"), run_name="__main__")
