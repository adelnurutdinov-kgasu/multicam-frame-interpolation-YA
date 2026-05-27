"""Launcher -> scripts/ego/import_mask_picker_selections.py"""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "ego" / "import_mask_picker_selections.py"), run_name="__main__")
