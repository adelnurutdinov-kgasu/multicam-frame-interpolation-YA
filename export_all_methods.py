"""Launcher -> scripts/baselines/export_all_methods.py"""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "baselines" / "export_all_methods.py"), run_name="__main__")
