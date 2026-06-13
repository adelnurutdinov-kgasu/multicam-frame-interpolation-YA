"""
Fine-tune YOLO-seg on ego/camera artifact masks.

  1) python yolo/export_ego_artifact_seg.py
  2) python yolo/train_ego_artifact_seg.py

Usage:
  python yolo/train_ego_artifact_seg.py
  python yolo/train_ego_artifact_seg.py --epochs 80 --model yolo26s-seg.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_DATA = REPO / "methods_gallery/_ego_manual_masks/yolo_ego_artifact/data.yaml"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--model", default="yolo26s-seg.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="")
    ap.add_argument("--project", type=Path, default=REPO / "runs/ego_artifact_seg")
    ap.add_argument("--name", default="train")
    args = ap.parse_args()

    if not args.data.is_file():
        print(f"Missing dataset yaml: {args.data}")
        print("Run first: python yolo/export_ego_artifact_seg.py")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("pip install -r requirements-yolo.txt")
        return 1

    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
        patience=15,
        mosaic=0.3,
        copy_paste=0.1,
    )
    print(f"Done. Weights: {args.project / args.name / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
