"""Launcher -> scripts/inference/build_blend_preview_gallery.py"""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "inference" / "build_blend_preview_gallery.py"), run_name="__main__")
