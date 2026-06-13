"""
Пакетный прогон датасета фото через YOLO (Ultralytics).

Примеры:
  python yolo/detect_dataset.py --config yolo/config_cv_dataset.yaml --dry-run
  python yolo/detect_dataset.py --config yolo/config_cv_dataset.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Windows/colabus: torch до pandas, иначе WinError 1114 на c10.dll
import torch  # noqa: F401

import cv2
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


def main() -> int:
    args = base_arg_parser("YOLO-детекция на датасете фото").parse_args()
    cfg, config_path, dataset_dir, run_dir, images, code = prepare_run(
        args, "yolo/config.yaml", "runs/detect"
    )
    if code >= 0:
        return code

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Установите зависимости: pip install -r requirements-yolo.txt")
        return 1

    model_name = cfg.get("model", "yolo26s.pt")
    class_names = cfg.get("classes") or []
    class_ids = load_class_ids(class_names) if class_names else None

    save_config_copy(config_path, run_dir)
    vis_dir = run_dir / "images"
    label_dir = run_dir / "labels"
    if cfg.get("save_images", True):
        vis_dir.mkdir(parents=True, exist_ok=True)
    if cfg.get("save_labels", True):
        label_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)
    device = cfg.get("device") or None
    conf = float(cfg.get("conf", 0.25))
    iou = float(cfg.get("iou", 0.45))
    imgsz = int(cfg.get("imgsz", 640))

    rows: list[dict] = []
    jsonl_path = run_dir / "detections.jsonl"
    if cfg.get("save_json", True) and jsonl_path.exists():
        jsonl_path.unlink()

    for img_path in tqdm(images, desc="YOLO"):
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

        detections = []
        if r.boxes is not None and len(r.boxes):
            for box in r.boxes:
                cls_id = int(box.cls.item())
                name = r.names[cls_id]
                xyxy = box.xyxy[0].tolist()
                conf_v = float(box.conf.item())
                detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": name,
                        "confidence": round(conf_v, 4),
                        "bbox_xyxy": [round(x, 2) for x in xyxy],
                    }
                )
                rows.append(
                    {
                        "image": rel_str,
                        "class_name": name,
                        "class_id": cls_id,
                        "confidence": conf_v,
                        "x1": xyxy[0],
                        "y1": xyxy[1],
                        "x2": xyxy[2],
                        "y2": xyxy[3],
                        "width": w,
                        "height": h,
                    }
                )

        record = {"image": rel_str, "width": w, "height": h, "detections": detections}
        if cfg.get("save_json", True):
            with jsonl_path.open("a", encoding="utf-8") as jf:
                jf.write(json.dumps(record, ensure_ascii=False) + "\n")

        if cfg.get("save_labels", True) and detections:
            label_file = label_dir / f"{stem}.txt"
            label_file.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for d in detections:
                x1, y1, x2, y2 = d["bbox_xyxy"]
                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{d['class_id']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            label_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        if cfg.get("save_images", True):
            plotted = r.plot()
            out_img = vis_dir / f"{stem}.jpg"
            out_img.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_img), plotted)

    if cfg.get("save_csv", True) and rows:
        df = pd.DataFrame(rows)
        df.to_csv(run_dir / "detections.csv", index=False, encoding="utf-8-sig")
        summary = (
            df.groupby("class_name")
            .agg(count=("class_name", "size"), mean_conf=("confidence", "mean"))
            .reset_index()
            .sort_values("count", ascending=False)
        )
        summary.to_csv(run_dir / "summary_by_class.csv", index=False, encoding="utf-8-sig")

    write_run_meta(
        run_dir,
        {
            "task": "detection",
            "dataset_dir": str(dataset_dir),
            "images_processed": len(images),
            "model": model_name,
            "classes_filter": class_names,
            "run_dir": str(run_dir),
        },
    )
    print(f"Готово. Метки: {run_dir}")
    print(f"  labels/  detections.jsonl  detections.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
