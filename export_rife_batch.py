"""Launcher -> scripts/baselines/export_rife_batch.py"""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "baselines" / "export_rife_batch.py"), run_name="__main__")
