"""Launcher -> scripts/inference/build_test_outputs_gallery.py"""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "inference" / "build_test_outputs_gallery.py"), run_name="__main__")
