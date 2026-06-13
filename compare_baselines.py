"""Launcher -> scripts/baselines/compare_baselines.py"""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "baselines" / "compare_baselines.py"), run_name="__main__")
