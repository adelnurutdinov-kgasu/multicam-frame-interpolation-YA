"""
Apply mask_picker_selections.json -> masks_approved.

Selections are stored per frame:
  {group_id}|{frame_id} -> chosen mask candidate

For masks_approved (one mask per vehicle+camera) we pick:
  1) selection on __mean__ frame, else
  2) legacy group-only entry, else
  3) most common candidate across saved frames in the group

ZERO_MASK_GROUPS (ego_mask_policy): пустая маска, picker игнорируется.

Usage:
  python scripts/ego/import_mask_picker_selections.py
  python scripts/ego/import_mask_picker_selections.py --dry-run
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
import sys
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


import cv2
import numpy as np
from PIL import Image

from ego_mask_extract import save_binary_mask
from ego_mask_policy import (
    ZERO_MASK_GROUP_IDS,
    is_zero_mask_group,
    write_zero_approved_mask,
)
from import_manual_ego_masks import APPROVED_DIR, MEANS_DIR, ROOT

SELECTIONS = ROOT / "mask_picker_selections.json"
BANK = ROOT / "browser_gallery"


def load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 127


def resize_to_mean(mask: np.ndarray, vehicle: str, camera: str) -> np.ndarray:
    mean_path = MEANS_DIR / f"{vehicle}_{camera}.png"
    if mean_path.is_file():
        hw = np.array(Image.open(mean_path)).shape[:2]
        if mask.shape[:2] != hw:
            u8 = cv2.resize(mask.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
            return u8.astype(bool)
    return mask.astype(bool)


def parse_group_id(group_id: str) -> tuple[str, str] | None:
    for cam in ("left_fwd", "right_fwd", "left_bwd", "right_bwd", "front", "rear"):
        if group_id.endswith("_" + cam):
            return group_id[: -(len(cam) + 1)], cam
    return None


def parse_selection_key(key: str, row: dict) -> tuple[str, str | None]:
    if "|" in key:
        group_id, frame_id = key.split("|", 1)
        return group_id, frame_id
    group_id = str(row.get("group") or key)
    frame_id = row.get("frame_id")
    return group_id, frame_id


def group_selections(items: dict) -> dict[str, list[tuple[str | None, dict]]]:
    grouped: dict[str, list[tuple[str | None, dict]]] = defaultdict(list)
    for key, row in items.items():
        if not isinstance(row, dict):
            continue
        group_id, frame_id = parse_selection_key(str(key), row)
        grouped[group_id].append((frame_id, row))
    return grouped


def pick_row_for_group(rows: list[tuple[str | None, dict]]) -> tuple[str | None, dict] | None:
    if not rows:
        return None
    for frame_id, row in rows:
        if frame_id == "__mean__":
            return frame_id, row
    legacy = [(fid, row) for fid, row in rows if fid is None or row.get("frame_id") in (None, "")]
    if legacy:
        return legacy[0]
    counts = Counter(row.get("candidate_id") or row.get("mask_rel") for _, row in rows)
    best = counts.most_common(1)[0][0]
    for frame_id, row in rows:
        key = row.get("candidate_id") or row.get("mask_rel")
        if key == best:
            return frame_id, row
    return rows[0]


def resolve_mask_path(rel: str) -> Path | None:
    src = BANK / rel.replace("\\", "/")
    if src.is_file():
        return src
    src = ROOT / rel.replace("\\", "/")
    return src if src.is_file() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    grouped: dict[str, list] = {}
    if SELECTIONS.is_file():
        data = json.loads(SELECTIONS.read_text(encoding="utf-8"))
        items = data.get("selections", data) if isinstance(data, dict) else {}
        grouped = group_selections(items)

    # Zero-группы только если в picker нет выбора (test picker может переопределить)
    n_zero = 0
    for group_id in sorted(ZERO_MASK_GROUP_IDS):
        if group_id in grouped:
            continue
        parsed = parse_group_id(group_id)
        if not parsed:
            continue
        vehicle, camera = parsed
        print(f"{group_id} <- ZERO (без маскирования)")
        if not args.dry_run:
            write_zero_approved_mask(APPROVED_DIR, vehicle, camera)
        n_zero += 1

    if not grouped and n_zero == 0:
        print(f"No selections file: {SELECTIONS}")
        return 1
    if not grouped:
        print(f"{'Would apply' if args.dry_run else 'Applied'} zero-only: {n_zero}")
        return 0 if n_zero else 1
    n = 0
    for group_id in sorted(grouped):
        parsed = parse_group_id(group_id)
        if not parsed:
            print(f"Skip bad id: {group_id}")
            continue
        vehicle, camera = parsed

        picked = pick_row_for_group(grouped[group_id])
        if not picked:
            continue
        frame_id, row = picked
        n_frames = len(grouped[group_id])
        cand_id = row.get("candidate_id", "")
        rel = row.get("mask_rel") or row.get("rel")
        frame_note = f" frame={frame_id}" if frame_id else ""
        out = APPROVED_DIR / f"{vehicle}_{camera}.png"

        if cand_id == "zero:none":
            print(f"{group_id}{frame_note} <- ZERO (без маскирования)  [{n_frames} saved frame(s)]")
            if not args.dry_run:
                write_zero_approved_mask(APPROVED_DIR, vehicle, camera)
            n += 1
            continue

        if not rel:
            print(f"Skip {group_id}: no mask_rel")
            continue
        src = resolve_mask_path(rel)
        if src is None:
            print(f"Skip {group_id}: missing {rel}")
            continue

        mask = load_mask(src)
        mask = resize_to_mean(mask, vehicle, camera)
        if mask.sum() < 32:
            print(f"{group_id}{frame_note} <- empty mask -> ZERO")
            if not args.dry_run:
                write_zero_approved_mask(APPROVED_DIR, vehicle, camera)
            n += 1
            continue

        print(
            f"{group_id}{frame_note} <- {row.get('label', rel)}  "
            f"({100 * mask.mean():.1f}%)  [{n_frames} saved frame(s)]"
        )
        if not args.dry_run:
            APPROVED_DIR.mkdir(parents=True, exist_ok=True)
            save_binary_mask(mask, out)
        n += 1

    print(
        f"{'Would apply' if args.dry_run else 'Applied'}: "
        f"{n} from picker + {n_zero} zero -> {APPROVED_DIR}"
    )
    return 0 if (n + n_zero) else 1


if __name__ == "__main__":
    raise SystemExit(main())
