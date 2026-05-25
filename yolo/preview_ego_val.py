"""Visualize YOLO ego_artifact predictions on val set (GT vs pred)."""

from __future__ import annotations

import argparse
import sys
from html import escape
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_DATA = REPO / "methods_gallery/_ego_manual_masks/yolo_ego_artifact"
DEFAULT_WEIGHTS = REPO / "runs/ego_artifact_seg/train/weights/best.pt"
OUT_DIR = REPO / "methods_gallery/_ego_manual_masks/yolo_ego_artifact/val_preview"


def load_gt_polygons(label_path: Path, w: int, h: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if not label_path.is_file():
        return mask
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        pts = []
        for i in range(1, len(parts), 2):
            x = float(parts[i]) * w
            y = float(parts[i + 1]) * h
            pts.append([x, y])
        if len(pts) >= 3:
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    return mask


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(255, 48, 48), alpha=0.52) -> np.ndarray:
    out = rgb.copy().astype(np.float32)
    m = mask > 127
    if not m.any():
        return rgb
    c = np.array(color, dtype=np.float32)
    out[m] = out[m] * (1 - alpha) + c * alpha
    return out.astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    if not args.weights.is_file():
        print(f"Missing weights: {args.weights}")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("pip install -r requirements-yolo.txt")
        return 1

    val_img = args.data / "images" / "val"
    val_lbl = args.data / "labels" / "val"
    imgs = sorted(val_img.glob("*.jpg"))
    if not imgs:
        print(f"No val images in {val_img}")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    cards_dir = args.out / "cards"
    cards_dir.mkdir(exist_ok=True)

    model = YOLO(str(args.weights))
    rows: list[str] = []

    for img_path in imgs:
        stem = img_path.stem
        rgb = np.array(Image.open(img_path).convert("RGB"))
        h, w = rgb.shape[:2]

        gt = load_gt_polygons(val_lbl / f"{stem}.txt", w, h)
        gt_vis = overlay_mask(rgb, gt)

        results = model.predict(source=str(img_path), conf=args.conf, verbose=False)
        r = results[0]
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        if r.masks is not None and r.masks.data is not None:
            for i, md in enumerate(r.masks.data.cpu().numpy()):
                m = (md > 0.5).astype(np.uint8) * 255
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                pred_mask = np.maximum(pred_mask, m)

        pred_vis = overlay_mask(rgb, pred_mask, color=(48, 128, 255))
        n_pred = int((pred_mask > 127).sum())
        n_gt = int((gt > 127).sum())
        inter = int(((gt > 127) & (pred_mask > 127)).sum())
        union = int(((gt > 127) | (pred_mask > 127)).sum())
        iou = inter / union if union else 0.0

        combo = np.concatenate([gt_vis, pred_vis], axis=1)
        card_name = f"{stem}.jpg"
        cv2.imwrite(str(cards_dir / card_name), cv2.cvtColor(combo, cv2.COLOR_RGB2BGR))

        rows.append(
            f'<div class="card">'
            f'<div class="meta"><b>{escape(stem)}</b><br>'
            f'GT px: {n_gt:,} · Pred px: {n_pred:,} · IoU: {iou:.2f}</div>'
            f'<img src="cards/{escape(card_name)}" loading="lazy">'
            f'<div class="hint">слева GT · справа YOLO</div></div>'
        )

    html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<title>YOLO ego_artifact — val preview</title>
<style>
body {{ background:#111; color:#eee; font-family:system-ui,sans-serif; margin:0; padding:16px; }}
h1 {{ font-size:1.2rem; }}
.sub {{ color:#888; font-size:0.85rem; margin-bottom:16px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(520px,1fr)); gap:12px; }}
.card {{ background:#1a1a1a; border:1px solid #333; border-radius:8px; overflow:hidden; }}
.card img {{ width:100%; display:block; }}
.meta {{ padding:8px 10px; font-size:0.78rem; line-height:1.4; }}
.hint {{ padding:0 10px 8px; color:#666; font-size:0.72rem; }}
</style></head><body>
<h1>YOLO ego_artifact — val ({len(imgs)} кадров)</h1>
<p class="sub">weights: {escape(str(args.weights))} · conf={args.conf}</p>
<div class="grid">{''.join(rows)}</div>
</body></html>"""
    (args.out / "index.html").write_text(html, encoding="utf-8")
    print(f"Preview: {args.out / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
