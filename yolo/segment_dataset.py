"""
Пакетная instance segmentation: маски объектов (не только боксы).

YOLO26-seg — актуальные веса Ultralytics с суффиксом -seg.

Примеры:
  python yolo/segment_dataset.py
  python yolo/segment_dataset.py --config yolo/config_seg.yaml --model yolo26m-seg.pt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Windows/colabus: torch до pandas, иначе WinError 1114 на c10.dll
import torch  # noqa: F401

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from yolo.pipeline_common import (
    base_arg_parser,
    prepare_run,
    save_config_copy,
    write_run_meta,
)
from yolo.utils import load_class_ids


def polygon_to_yolo_line(cls_id: int, polygon: np.ndarray, w: int, h: int) -> str:
    """Полигон в пикселях -> строка YOLO-seg (нормализованные x,y)."""
    pts = polygon.astype(np.float64)
    if pts.ndim == 1:
        return ""
    parts = [str(cls_id)]
    for x, y in pts:
        parts.append(f"{x / w:.6f}")
        parts.append(f"{y / h:.6f}")
    return " ".join(parts)


def main() -> int:
    args = base_arg_parser("YOLO instance segmentation (маски)").parse_args()
    cfg, config_path, dataset_dir, run_dir, images, code = prepare_run(
        args, "yolo/config_seg.yaml", "runs/segment"
    )
    if code >= 0:
        return code

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Установите зависимости: pip install -r requirements-yolo.txt")
        return 1

    model_name = cfg.get("model", "yolo26s-seg.pt")
    if "-seg" not in model_name.lower():
        print(f"Внимание: для масок нужна модель с суффиксом -seg, сейчас: {model_name}")

    class_names = cfg.get("classes") or []
    class_ids = load_class_ids(class_names) if class_names else None

    save_config_copy(config_path, run_dir)
    vis_dir = run_dir / "images"
    mask_dir = run_dir / "masks"
    overlay_dir = run_dir / "overlay"
    label_dir = run_dir / "labels_seg"
    for d, flag in [
        (vis_dir, cfg.get("save_images", True)),
        (mask_dir, cfg.get("save_masks", True)),
        (overlay_dir, cfg.get("save_overlay", True)),
        (label_dir, cfg.get("save_labels", True)),
    ]:
        if flag:
            d.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)
    device = cfg.get("device") or None
    conf = float(cfg.get("conf", 0.25))
    iou = float(cfg.get("iou", 0.45))
    imgsz = int(cfg.get("imgsz", 640))

    rows: list[dict] = []
    jsonl_path = run_dir / "segments.jsonl"
    if cfg.get("save_json", True) and jsonl_path.exists():
        jsonl_path.unlink()

    for img_path in tqdm(images, desc="YOLO-seg"):
        rel = img_path.relative_to(dataset_dir) if img_path.is_relative_to(dataset_dir) else Path(img_path.name)
        rel_str = rel.as_posix()
        stem = Path(rel).with_suffix("")

        results = model.predict(
            source=str(img_path),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            classes=class_ids,
            max_det=int(cfg.get("max_det", 300)),
            half=bool(cfg.get("half", False)),
            verbose=False,
        )
        r = results[0]
        h, w = r.orig_shape

        instances = []
        label_lines: list[str] = []
        masks_data = None
        if r.masks is not None and r.masks.data is not None:
            masks_data = r.masks.data.cpu().numpy()

        if r.boxes is not None and len(r.boxes) and r.masks is not None:
            for i, box in enumerate(r.boxes):
                cls_id = int(box.cls.item())
                name = r.names[cls_id]
                conf_v = float(box.conf.item())
                xyxy = box.xyxy[0].tolist()

                polygon = None
                if r.masks.xy is not None and i < len(r.masks.xy):
                    polygon = r.masks.xy[i]
                    if hasattr(polygon, "tolist"):
                        polygon_list = polygon.tolist()
                    else:
                        polygon_list = [list(map(float, p)) for p in polygon]
                else:
                    polygon_list = []

                area_px = 0
                if masks_data is not None and i < len(masks_data):
                    m = (masks_data[i] > 0.5).astype(np.uint8) * 255
                    area_px = int(np.count_nonzero(m))
                    if cfg.get("save_masks", True):
                        out_m = mask_dir / str(stem) / f"{i:03d}_{name}.png"
                        out_m.parent.mkdir(parents=True, exist_ok=True)
                        m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                        cv2.imwrite(str(out_m), m_resized)

                instances.append(
                    {
                        "instance_id": i,
                        "class_id": cls_id,
                        "class_name": name,
                        "confidence": round(conf_v, 4),
                        "bbox_xyxy": [round(x, 2) for x in xyxy],
                        "area_pixels": area_px,
                        "polygon_xy": polygon_list,
                    }
                )
                rows.append(
                    {
                        "image": rel_str,
                        "instance_id": i,
                        "class_name": name,
                        "class_id": cls_id,
                        "confidence": conf_v,
                        "area_pixels": area_px,
                        "width": w,
                        "height": h,
                    }
                )

                if cfg.get("save_labels", True) and polygon is not None and len(polygon):
                    line = polygon_to_yolo_line(cls_id, np.array(polygon), w, h)
                    if line:
                        label_lines.append(line)

        if label_lines and cfg.get("save_labels", True):
            label_file = label_dir / f"{stem}.txt"
            label_file.parent.mkdir(parents=True, exist_ok=True)
            label_file.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

        record = {"image": rel_str, "width": w, "height": h, "instances": instances}
        if cfg.get("save_json", True):
            with jsonl_path.open("a", encoding="utf-8") as jf:
                jf.write(json.dumps(record, ensure_ascii=False) + "\n")

        if cfg.get("save_images", True):
            plotted = r.plot()
            out_img = vis_dir / f"{stem}.jpg"
            out_img.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_img), plotted)

        if cfg.get("save_overlay", True) and instances:
            base = cv2.imread(str(img_path))
            if base is not None:
                blend = base.copy().astype(np.float32)
                alpha = 0.45
                colors = [
                    (0, 255, 0),
                    (255, 128, 0),
                    (0, 128, 255),
                    (255, 0, 255),
                    (255, 255, 0),
                ]
                if masks_data is not None:
                    for i in range(min(len(masks_data), len(instances))):
                        m = cv2.resize(
                            (masks_data[i] > 0.5).astype(np.uint8),
                            (w, h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        c = np.array(colors[i % len(colors)], dtype=np.float32)
                        mask3 = np.stack([m, m, m], axis=-1)
                        blend = np.where(
                            mask3 > 0,
                            blend * (1 - alpha) + c * alpha,
                            blend,
                        )
                out_ov = overlay_dir / f"{stem}.jpg"
                out_ov.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_ov), blend.astype(np.uint8))

    if cfg.get("save_csv", True) and rows:
        df = pd.DataFrame(rows)
        df.to_csv(run_dir / "segments.csv", index=False, encoding="utf-8-sig")
        summary = (
            df.groupby("class_name")
            .agg(
                count=("class_name", "size"),
                mean_conf=("confidence", "mean"),
                mean_area=("area_pixels", "mean"),
            )
            .reset_index()
            .sort_values("count", ascending=False)
        )
        summary.to_csv(run_dir / "summary_by_class.csv", index=False, encoding="utf-8-sig")

    write_run_meta(
        run_dir,
        {
            "task": "instance_segmentation",
            "dataset_dir": str(dataset_dir),
            "images_processed": len(images),
            "model": model_name,
            "classes_filter": class_names,
            "run_dir": str(run_dir),
        },
    )
    print(f"Готово. Сегменты: {run_dir}")
    print(f"  labels_seg/  segments.jsonl  segments.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
