"""
Import user-composed masks from masks_composed/ into masks_approved.

Expected filenames:
  {vehicle}_{camera}.png
  {vehicle}_{camera}_composed.png

Usage:
  python scripts/ego/import_composed_masks.py
  python scripts/ego/import_composed_masks.py --also-update-selected
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
import sys
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


import cv2
import numpy as np
from PIL import Image

from ego_mask_extract import save_binary_mask
from import_manual_ego_masks import APPROVED_DIR, MEANS_DIR, ROOT, load_index

COMPOSED_DIR = ROOT / "masks_composed"
SELECTED_DIR = ROOT / "masks_selected"
LOG_PATH = ROOT / "composed_masks_log.json"

NAME_RE = re.compile(
    r"^(.+)_(front|rear|left_fwd|right_fwd|left_bwd|right_bwd)(?:_composed)?\.png$",
    re.I,
)


def parse_name(path: Path) -> tuple[str, str] | None:
    m = NAME_RE.match(path.name)
    if not m:
        return None
    return m.group(1), m.group(2)


def resize_to_mean(mask: np.ndarray, vehicle: str, camera: str) -> np.ndarray:
    mean_path = MEANS_DIR / f"{vehicle}_{camera}.png"
    if mean_path.is_file():
        hw = np.array(Image.open(mean_path)).shape[:2]
        if mask.shape[:2] != hw:
            mask = cv2.resize(mask.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
            return mask.astype(bool)
    return mask.astype(bool)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--also-update-selected", action="store_true")
    args = ap.parse_args()

    if not COMPOSED_DIR.is_dir():
        print(f"Create {COMPOSED_DIR} and add composed PNGs first.")
        return 1

    index = load_index()
    log: list[dict] = []

    for path in sorted(COMPOSED_DIR.glob("*.png")):
        parsed = parse_name(path)
        if not parsed:
            continue
        vehicle, camera = parsed
        mask = np.array(Image.open(path).convert("L")) > 127
        mask = resize_to_mean(mask, vehicle, camera)
        if mask.sum() < 32:
            print(f"Skip empty: {path.name}")
            continue

        out = APPROVED_DIR / f"{vehicle}_{camera}.png"
        save_binary_mask(mask, out)
        if args.also_update_selected:
            SELECTED_DIR.mkdir(parents=True, exist_ok=True)
            save_binary_mask(mask, SELECTED_DIR / f"{vehicle}_{camera}.png")

        log.append({
            "source": str(path.resolve()),
            "vehicle": vehicle,
            "camera": camera,
            "mask_pct": float(mask.mean()),
            "approved_png": str(out.resolve()),
        })
        print(f"Imported {vehicle}/{camera}  coverage={100*mask.mean():.1f}%")

    summary = {"n_imported": len(log), "items": log}
    LOG_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done: {len(log)} masks -> {APPROVED_DIR}")
    print(f"Log: {LOG_PATH}")
    return 0 if log else 1


if __name__ == "__main__":
    raise SystemExit(main())
