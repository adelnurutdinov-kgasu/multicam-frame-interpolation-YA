"""Launcher -> scripts/ego/rebuild_composer_html.py"""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "ego" / "rebuild_composer_html.py"), run_name="__main__")
