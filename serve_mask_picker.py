"""Launcher -> scripts/ego/serve_mask_picker.py"""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "ego" / "serve_mask_picker.py"), run_name="__main__")
