"""Загрузка аннотаций и отрисовка поверх оригинала (без сохранения на диск)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "person": (0, 255, 0),
    "bicycle": (255, 128, 0),
    "car": (0, 128, 255),
    "motorcycle": (255, 0, 255),
    "bus": (255, 255, 0),
    "truck": (128, 0, 255),
    "traffic light": (0, 255, 255),
    "stop sign": (255, 0, 0),
    "fire hydrant": (0, 165, 255),
    "parking meter": (147, 20, 255),
}


def color_for(name: str) -> tuple[int, int, int]:
    if name in CLASS_COLORS:
        return CLASS_COLORS[name]
    h = abs(hash(name)) % 180
    return cv2.cvtColor(np.uint8([[[h, 220, 220]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()


def load_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["image"]] = rec
    return out


@dataclass
class AnnotationStore:
    dataset_dir: Path
    detect_jsonl: Path
    segment_jsonl: Path
    detections: dict[str, dict] = field(default_factory=dict)
    segments: dict[str, dict] = field(default_factory=dict)
    images: list[str] = field(default_factory=list)

    @classmethod
    def open(cls, dataset_dir: Path, detect_jsonl: Path, segment_jsonl: Path) -> "AnnotationStore":
        store = cls(
            dataset_dir=dataset_dir.resolve(),
            detect_jsonl=detect_jsonl.resolve(),
            segment_jsonl=segment_jsonl.resolve(),
        )
        store.detections = load_jsonl(store.detect_jsonl)
        store.segments = load_jsonl(store.segment_jsonl)
        keys = set(store.detections) | set(store.segments)
        if keys:
            store.images = sorted(keys)
        else:
            store.images = [
                p.relative_to(store.dataset_dir).as_posix()
                for p in sorted(store.dataset_dir.rglob("*.jpg"))
            ]
        return store

    def filter_images(
        self,
        path_substr: str = "",
        only_with_boxes: bool = False,
        only_with_segments: bool = False,
    ) -> list[str]:
        items = self.images
        if path_substr.strip():
            needle = path_substr.strip().replace("\\", "/").lower()
            items = [i for i in items if needle in i.lower()]
        if only_with_boxes:
            items = [i for i in items if self.detections.get(i, {}).get("detections")]
        if only_with_segments:
            items = [i for i in items if self.segments.get(i, {}).get("instances")]
        return items


def render_overlay(
    store: AnnotationStore,
    rel_path: str,
    show_boxes: bool,
    show_segments: bool,
    conf_min: float,
    class_names: list[str] | None,
) -> tuple[np.ndarray | None, str]:
    img_path = store.dataset_dir / rel_path.replace("/", "\\")
    if not img_path.is_file():
        img_path = store.dataset_dir / rel_path
    if not img_path.is_file():
        return None, f"Файл не найден: {rel_path}"

    img = cv2.imread(str(img_path))
    if img is None:
        return None, f"Не удалось прочитать: {rel_path}"

    allowed = {c.lower() for c in class_names} if class_names else None
    det_n = 0
    seg_n = 0
    overlay = img.copy()

    if show_segments and rel_path in store.segments:
        for inst in store.segments[rel_path].get("instances", []):
            if inst.get("confidence", 0) < conf_min:
                continue
            name = inst.get("class_name", "")
            if allowed and name.lower() not in allowed:
                continue
            poly = inst.get("polygon_xy") or []
            if len(poly) < 3:
                continue
            pts = np.array(poly, dtype=np.int32)
            color = color_for(name)
            mask = np.zeros(overlay.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            tint = np.zeros_like(overlay)
            tint[:] = color
            alpha = 0.4
            m = mask > 0
            overlay[m] = cv2.addWeighted(overlay, 1 - alpha, tint, alpha, 0)[m]
            cv2.polylines(overlay, [pts], True, color, 2, cv2.LINE_AA)
            seg_n += 1

    if show_boxes and rel_path in store.detections:
        for d in store.detections[rel_path].get("detections", []):
            if d.get("confidence", 0) < conf_min:
                continue
            name = d.get("class_name", "")
            if allowed and name.lower() not in allowed:
                continue
            x1, y1, x2, y2 = map(int, d["bbox_xyxy"])
            color = color_for(name)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            label = f"{name} {d['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(overlay, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            det_n += 1

    rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    info = (
        f"{rel_path}\n"
        f"боксов: {det_n} | сегментов: {seg_n} | conf>={conf_min:.2f}"
    )
    return rgb, info
