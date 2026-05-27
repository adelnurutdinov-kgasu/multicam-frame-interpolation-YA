"""Launcher -> scripts/inference/run_test_consensus_inference.py"""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "inference" / "run_test_consensus_inference.py"), run_name="__main__")
