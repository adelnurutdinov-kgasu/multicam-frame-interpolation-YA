"""Launcher -> scripts/inference/flip_mirror_lidar_masks.py"""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "inference" / "flip_mirror_lidar_masks.py"), run_name="__main__")
