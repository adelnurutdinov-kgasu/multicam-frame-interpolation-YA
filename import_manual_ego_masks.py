"""Launcher -> scripts/ego/import_manual_ego_masks.py"""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "ego" / "import_manual_ego_masks.py"), run_name="__main__")
