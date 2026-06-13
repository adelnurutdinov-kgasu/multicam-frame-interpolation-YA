"""Launcher -> scripts/inference/export_submission_blend50.py"""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "inference" / "export_submission_blend50.py"), run_name="__main__")
