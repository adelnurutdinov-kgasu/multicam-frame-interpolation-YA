"""
Семантическая сегментация: road, sky, vegetation, pole, building… на весь кадр.

YOLO26-sem — Cityscapes, 19 классов. Сохраняет контуры в JSON (без дублирования картинок).

Примеры:
  python yolo/sem_dataset.py --config yolo/config_cv_dataset_sem.yaml --dry-run
  python yolo/sem_dataset.py --config yolo/config_cv_dataset_sem.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch  # noqa: F401 — до pandas на Windows

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from yolo.cityscapes_classes import CITYSCAPES_COLORS, CITYSCAPES_NAMES, NAME_TO_ID
from yolo.pipeline_common import (
    base_arg_parser,
    prepare_run,
    save_config_copy,
    write_run_meta,
)


def load_class_filter(class_names: list[str]) -> set[int] | None:
    if not class_names:
        return None
    ids = set()
    missing = []
    for name in class_names:
        if name in NAME_TO_ID:
            ids.add(NAME_TO_ID[name])
        else:
            missing.append(name)
    if missing:
        raise ValueError(f"Неизвестные классы Cityscapes: {missing}")
    return ids


def regions_from_semantic_map(
    sem_map: np.ndarray,
    names: dict[int, str],
    allowed_ids: set[int] | None,
    min_area: int,
    epsilon: float,
) -> list[dict]:
    regions: list[dict] = []
    for cls_id in np.unique(sem_map):
        cls_id = int(cls_id)
        if allowed_ids is not None and cls_id not in allowed_ids:
            continue
        if cls_id not in names:
            continue

        binary = (sem_map == cls_id).astype(np.uint8)
        total_area = int(binary.sum())
        if total_area < min_area:
            continue

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = int(cv2.contourArea(cnt))
            if area < min_area:
                continue
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            poly = approx.reshape(-1, 2)
            if len(poly) < 3:
                continue
            regions.append(
                {
                    "class_id": cls_id,
                    "class_name": names[cls_id],
                    "area_pixels": area,
                    "polygon_xy": [[float(x), float(y)] for x, y in poly],
                }
            )
    return regions


def main() -> int:
    args = base_arg_parser("YOLO semantic segmentation (Cityscapes)").parse_args()
    cfg, config_path, dataset_dir, run_dir, images, code = prepare_run(
        args, "yolo/config_cv_dataset_sem.yaml", "runs/semantic"
    )
    if code >= 0:
        return code

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Установите зависимости: pip install -r requirements-yolo.txt")
        return 1

    model_name = cfg.get("model", "yolo26s-sem.pt")
    if "-sem" not in model_name.lower():
        print(f"Внимание: для семантики нужна модель с суффиксом -sem, сейчас: {model_name}")

    class_names = cfg.get("classes") or []
    allowed_ids = load_class_filter(class_names)
    min_area = int(cfg.get("min_region_area", 200))
    epsilon = float(cfg.get("contour_epsilon", 2.0))

    save_config_copy(config_path, run_dir)
    vis_dir = run_dir / "images"
    mask_dir = run_dir / "masks"
    if cfg.get("save_images", False):
        vis_dir.mkdir(parents=True, exist_ok=True)
    if cfg.get("save_mask_png", False):
        mask_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)
    device = cfg.get("device") or None
    imgsz = int(cfg.get("imgsz", 1024))

    rows: list[dict] = []
    jsonl_path = run_dir / "semantic.jsonl"
    if cfg.get("save_json", True) and jsonl_path.exists():
        jsonl_path.unlink()

    for img_path in tqdm(images, desc="YOLO-sem"):
        rel = img_path.relative_to(dataset_dir) if img_path.is_relative_to(dataset_dir) else Path(img_path.name)
        rel_str = rel.as_posix()

        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            device=device,
            half=bool(cfg.get("half", False)),
            verbose=False,
        )
        r = results[0]
        h, w = r.orig_shape

        sem_map = None
        if hasattr(r, "semantic_mask") and r.semantic_mask is not None and r.semantic_mask.data is not None:
            sem_map = r.semantic_mask.data.cpu().numpy().astype(np.int32)
            if sem_map.shape != (h, w):
                sem_map = cv2.resize(sem_map.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(
                    np.int32
                )

        names = r.names if isinstance(r.names, dict) else CITYSCAPES_NAMES
        regions = regions_from_semantic_map(sem_map, names, allowed_ids, min_area, epsilon) if sem_map is not None else []

        present_classes = sorted({reg["class_name"] for reg in regions})
        record = {
            "image": rel_str,
            "width": w,
            "height": h,
            "classes_present": present_classes,
            "regions": regions,
        }

        if cfg.get("save_json", True):
            with jsonl_path.open("a", encoding="utf-8") as jf:
                jf.write(json.dumps(record, ensure_ascii=False) + "\n")

        for reg in regions:
            rows.append(
                {
                    "image": rel_str,
                    "class_name": reg["class_name"],
                    "class_id": reg["class_id"],
                    "area_pixels": reg["area_pixels"],
                    "width": w,
                    "height": h,
                }
            )

        if cfg.get("save_mask_png", False) and sem_map is not None:
            out_m = mask_dir / f"{Path(rel).with_suffix('')}.png"
            out_m.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_m), sem_map.astype(np.uint8))

        if cfg.get("save_images", False) and sem_map is not None:
            color = np.zeros((h, w, 3), dtype=np.uint8)
            for cls_id in np.unique(sem_map):
                cls_id = int(cls_id)
                c = CITYSCAPES_COLORS.get(names.get(cls_id, ""), (128, 128, 128))
                color[sem_map == cls_id] = c
            base = cv2.imread(str(img_path))
            if base is not None:
                blend = cv2.addWeighted(base, 0.5, color, 0.5, 0)
                out_img = vis_dir / f"{Path(rel).with_suffix('')}.jpg"
                out_img.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_img), blend)

    if cfg.get("save_csv", True) and rows:
        df = pd.DataFrame(rows)
        df.to_csv(run_dir / "semantic_regions.csv", index=False, encoding="utf-8-sig")
        summary = (
            df.groupby("class_name")
            .agg(count=("class_name", "size"), mean_area=("area_pixels", "mean"))
            .reset_index()
            .sort_values("count", ascending=False)
        )
        summary.to_csv(run_dir / "summary_by_class.csv", index=False, encoding="utf-8-sig")

    write_run_meta(
        run_dir,
        {
            "task": "semantic_segmentation",
            "dataset": "cityscapes",
            "dataset_dir": str(dataset_dir),
            "images_processed": len(images),
            "model": model_name,
            "classes_filter": class_names,
            "class_names": list(CITYSCAPES_NAMES.values()),
            "run_dir": str(run_dir),
        },
    )
    print(f"Готово. Семантика: {run_dir}")
    print(f"  semantic.jsonl  semantic_regions.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
