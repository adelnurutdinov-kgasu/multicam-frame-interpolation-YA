"""
Export YOLO-seg dataset for ego/camera artifacts (hood, mirror, body in frame).

Sources:
  1) mask_picker_selections.json — per-frame image + chosen mask
  2) ZERO_MASK_GROUPS — пустая маска (crozby_right_fwd и др.)
  3) Extra frames from cv_dataset train/ using group-level mask

Output (Ultralytics format):
  <out>/images/{train,val}/*.jpg
  <out>/labels/{train,val}/*.txt   — class 0 polygon, normalized
  <out>/data.yaml

Usage:
  python yolo/export_ego_artifact_seg.py
  python yolo/export_ego_artifact_seg.py --max-per-group 12
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_EGO = REPO / "scripts" / "ego"
if str(_EGO) not in sys.path:
    sys.path.append(str(_EGO))

from import_manual_ego_masks import APPROVED_DIR, MEANS_DIR, ROOT as EGO_ROOT, collect_sample_paths
from import_mask_picker_selections import (
    BANK,
    FORCED_GROUP_MASKS,
    SELECTIONS,
    group_selections,
    load_mask,
    parse_group_id,
    parse_selection_key,
    pick_row_for_group,
    resolve_mask_path,
    resize_to_mean,
)
from test_ego_consistent_edges import DEFAULT_DATASET

CLASS_NAME = "ego_artifact"
SAMPLE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_[a-zA-Z]+_\d+__\d{3})$")


def mask_to_yolo_lines(mask: np.ndarray, cls_id: int = 0, min_area: int = 64, epsilon: float = 2.0) -> list[str]:
    h, w = mask.shape
    u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines: list[str] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        parts = [str(cls_id)]
        for x, y in approx:
            parts.append(f"{x / w:.6f}")
            parts.append(f"{y / h:.6f}")
        lines.append(" ".join(parts))
    return lines


def resize_mask_to_hw(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    if mask.shape[:2] == (h, w):
        return mask.astype(bool)
    u8 = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return u8.astype(bool)


def group_mask_path(group_id: str, grouped: dict, vehicle: str, camera: str) -> tuple[Path | None, str]:
    if group_id in FORCED_GROUP_MASKS:
        rel = FORCED_GROUP_MASKS[group_id]["mask_rel"]
        p = resolve_mask_path(rel)
        return p, FORCED_GROUP_MASKS[group_id].get("label", rel)

    approved = APPROVED_DIR / f"{vehicle}_{camera}.png"
    if approved.is_file():
        return approved, "masks_approved"

    rows = grouped.get(group_id, [])
    picked = pick_row_for_group(rows)
    if not picked:
        return None, ""
    _, row = picked
    rel = row.get("mask_rel") or row.get("rel")
    if not rel:
        return None, ""
    return resolve_mask_path(rel), row.get("label", rel)


def add_sample(
    records: list[dict],
    img_path: Path,
    mask_path: Path,
    group_id: str,
    source: str,
) -> None:
    if not img_path.is_file() or not mask_path.is_file():
        return
    rgb = np.array(Image.open(img_path).convert("RGB"))
    h, w = rgb.shape[:2]
    mask = load_mask(mask_path)
    mask = resize_mask_to_hw(mask, h, w)
    if mask.sum() < 32:
        return
    lines = mask_to_yolo_lines(mask.astype(np.uint8))
    if not lines:
        return
    records.append(
        {
            "group": group_id,
            "source": source,
            "image": img_path,
            "label_lines": lines,
            "stem": f"{group_id}__{img_path.stem}",
        }
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out", type=Path, default=EGO_ROOT / "yolo_ego_artifact")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--max-per-group", type=int, default=24, help="extra dataset frames per group")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not SELECTIONS.is_file():
        print(f"Missing {SELECTIONS}")
        return 1

    data = json.loads(SELECTIONS.read_text(encoding="utf-8"))
    items = data.get("selections", data)
    grouped = group_selections(items)

    records: list[dict] = []

    for key, row in items.items():
        if not isinstance(row, dict):
            continue
        group_id, frame_id = parse_selection_key(str(key), row)
        if frame_id == "__mean__":
            continue
        rel = row.get("mask_rel") or row.get("rel")
        if not rel:
            continue
        mask_p = resolve_mask_path(rel)
        frame_rel = row.get("frame_rel")
        img_p = BANK / frame_rel.replace("\\", "/") if frame_rel else None
        if img_p is None or not img_p.is_file():
            sid = row.get("frame_id") or frame_id
            parsed = parse_group_id(group_id)
            if parsed and sid and sid != "__mean__":
                vehicle, camera = parsed
                cand = args.dataset / sid / "target" / f"{camera}.jpg"
                if cand.is_file():
                    img_p = cand
        if mask_p and img_p and img_p.is_file():
            add_sample(records, img_p, mask_p, group_id, "picker")

    all_groups = sorted(set(grouped) | set(FORCED_GROUP_MASKS))
    rng = random.Random(args.seed)
    for group_id in all_groups:
        parsed = parse_group_id(group_id)
        if not parsed:
            continue
        vehicle, camera = parsed
        mask_p, label = group_mask_path(group_id, grouped, vehicle, camera)
        if mask_p is None:
            continue
        mask_bool = resize_to_mean(load_mask(mask_p), vehicle, camera)
        tmp = args.out / "_tmp_masks" / f"{group_id}.png"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((mask_bool.astype(np.uint8) * 255)).save(tmp)

        paths = collect_sample_paths(args.dataset, vehicle, camera, max_scenes=args.max_per_group)
        rng.shuffle(paths)
        for sp in paths[: args.max_per_group]:
            img = sp / "target" / f"{camera}.jpg"
            add_sample(records, img, tmp, group_id, f"dataset+{label}")

    if not records:
        print("No training samples exported")
        return 1

    if args.out.exists():
        shutil.rmtree(args.out)
    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng.shuffle(records)
    n_val = max(1, int(len(records) * args.val_frac))
    val_set = {id(r) for r in records[:n_val]}

    for rec in tqdm(records, desc="write yolo"):
        split = "val" if id(rec) in val_set else "train"
        stem = rec["stem"]
        dst_img = args.out / "images" / split / f"{stem}.jpg"
        dst_lbl = args.out / "labels" / split / f"{stem}.txt"
        shutil.copy2(rec["image"], dst_img)
        dst_lbl.write_text("\n".join(rec["label_lines"]) + "\n", encoding="utf-8")

    tmp_dir = args.out / "_tmp_masks"
    if tmp_dir.is_dir():
        shutil.rmtree(tmp_dir)

    yaml_path = args.out / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(args.out.resolve()).replace("\\", "/"),
                "train": "images/train",
                "val": "images/val",
                "names": {0: CLASS_NAME},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    meta = {
        "n_samples": len(records),
        "n_train": len(records) - n_val,
        "n_val": n_val,
        "n_groups": len(all_groups),
        "forced": list(FORCED_GROUP_MASKS.keys()),
    }
    (args.out / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Exported {len(records)} samples -> {args.out}")
    print(f"  train={meta['n_train']}  val={meta['n_val']}  data.yaml ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
