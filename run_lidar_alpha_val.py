"""Launcher -> scripts/inference/run_lidar_alpha_val.py"""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "inference" / "run_lidar_alpha_val.py"), run_name="__main__")
