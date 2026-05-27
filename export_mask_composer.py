"""Launcher -> scripts/ego/export_mask_composer.py"""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "ego" / "export_mask_composer.py"), run_name="__main__")
