"""
Просмотр боксов и сегментов поверх оригиналов (ничего не копирует на диск).

Запуск:
  pip install -r requirements-viewer.txt
  python viewer/app.py

Откройте http://127.0.0.1:7860
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from viewer.render import AnnotationStore, render_overlay

DEFAULT_DATASET = Path(r"C:/Users/adel/Downloads/cv_dataset/final_dataset_v5_participants")
DEFAULT_DETECT = Path(r"C:/Users/adel/Downloads/cv_dataset/annotations/detect/detections.jsonl")
DEFAULT_SEGMENT = Path(r"C:/Users/adel/Downloads/cv_dataset/annotations/segment/segments.jsonl")

_store: AnnotationStore | None = None
_filtered: list[str] = []


def get_store(dataset: str, detect: str, segment: str) -> AnnotationStore:
    global _store
    ds = Path(dataset)
    det = Path(detect)
    seg = Path(segment)
    if _store is None or _store.dataset_dir != ds.resolve():
        _store = AnnotationStore.open(ds, det, seg)
    return _store


def reload_store(dataset: str, detect: str, segment: str) -> tuple[str, gr.Slider]:
    store = get_store(dataset, detect, segment)
    global _filtered
    _filtered = store.images
    msg = (
        f"Загружено: {len(store.images)} кадров | "
        f"detect jsonl: {len(store.detections)} | "
        f"segment jsonl: {len(store.segments)}"
    )
    max_idx = max(0, len(_filtered) - 1)
    return msg, gr.Slider(minimum=0, maximum=max_idx, value=0, step=1, label="Индекс кадра")


def apply_filter(
    dataset: str,
    detect: str,
    segment: str,
    path_filter: str,
    only_boxes: bool,
    only_seg: bool,
) -> tuple[str, gr.Slider]:
    store = get_store(dataset, detect, segment)
    global _filtered
    _filtered = store.filter_images(path_filter, only_boxes, only_seg)
    max_idx = max(0, len(_filtered) - 1)
    msg = f"После фильтра: {len(_filtered)} кадров"
    return msg, gr.Slider(minimum=0, maximum=max_idx, value=0, step=1, label="Индекс кадра")


def show_frame(
    dataset: str,
    detect: str,
    segment: str,
    index: int,
    show_boxes: bool,
    show_segments: bool,
    conf: float,
    path_filter: str,
    only_boxes: bool,
    only_seg: bool,
    class_filter: str,
) -> tuple:
    store = get_store(dataset, detect, segment)
    global _filtered
    _filtered = store.filter_images(path_filter, only_boxes, only_seg)
    if not _filtered:
        return None, "Нет кадров по фильтру"

    idx = int(min(max(0, index), len(_filtered) - 1))
    rel = _filtered[idx]
    classes = [c.strip() for c in class_filter.split(",") if c.strip()] or None
    rgb, info = render_overlay(store, rel, show_boxes, show_segments, conf, classes)
    nav = f"[{idx + 1}/{len(_filtered)}]"
    return rgb, f"{nav}\n{info}"


def step(index: int, delta: int) -> int:
    global _filtered
    if not _filtered:
        return 0
    return int(min(max(0, index + delta), len(_filtered) - 1))


def build_ui(default_dataset: str, default_detect: str, default_segment: str) -> gr.Blocks:
    with gr.Blocks(title="CV Dataset Viewer") as demo:
        gr.Markdown(
            "# Просмотр YOLO: боксы + сегменты\n"
            "Оригиналы читаются с диска, overlay рисуется в памяти — **без копирования картинок**."
        )
        with gr.Row():
            dataset_in = gr.Textbox(label="Папка датасета", value=default_dataset)
            detect_in = gr.Textbox(label="detections.jsonl", value=default_detect)
            segment_in = gr.Textbox(label="segments.jsonl", value=default_segment)
        reload_btn = gr.Button("Перезагрузить аннотации")
        status = gr.Textbox(label="Статус", interactive=False)

        with gr.Row():
            path_filter = gr.Textbox(label="Фильтр пути (подстрока)", placeholder="target, left_fwd, test/...")
            only_boxes = gr.Checkbox(label="Только с боксами")
            only_seg = gr.Checkbox(label="Только с сегментами")
            filter_btn = gr.Button("Применить фильтр")

        with gr.Row():
            show_boxes = gr.Checkbox(label="Боксы", value=True)
            show_segments = gr.Checkbox(label="Сегменты", value=True)
            conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Мин. confidence")
            class_filter = gr.Textbox(
                label="Классы (через запятую, пусто = все)",
                placeholder="person, car, truck",
            )

        index = gr.Slider(0, 0, value=0, step=1, label="Индекс кадра")
        with gr.Row():
            prev_btn = gr.Button("← Назад")
            next_btn = gr.Button("Вперёд →")

        image_out = gr.Image(label="Кадр", type="numpy")
        info_out = gr.Textbox(label="Инфо", lines=4, interactive=False)

        inputs = [
            dataset_in,
            detect_in,
            segment_in,
            index,
            show_boxes,
            show_segments,
            conf,
            path_filter,
            only_boxes,
            only_seg,
            class_filter,
        ]

        reload_btn.click(reload_store, [dataset_in, detect_in, segment_in], [status, index])
        filter_btn.click(
            apply_filter,
            [dataset_in, detect_in, segment_in, path_filter, only_boxes, only_seg],
            [status, index],
        )
        for ctrl in [index, show_boxes, show_segments, conf, class_filter]:
            ctrl.change(show_frame, inputs, [image_out, info_out])
        path_filter.change(show_frame, inputs, [image_out, info_out])
        only_boxes.change(show_frame, inputs, [image_out, info_out])
        only_seg.change(show_frame, inputs, [image_out, info_out])

        prev_btn.click(lambda i: step(i, -1), index, index).then(show_frame, inputs, [image_out, info_out])
        next_btn.click(lambda i: step(i, 1), index, index).then(show_frame, inputs, [image_out, info_out])

        demo.load(reload_store, [dataset_in, detect_in, segment_in], [status, index]).then(
            show_frame, inputs, [image_out, info_out]
        )

    return demo


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--detect", default=str(DEFAULT_DETECT))
    p.add_argument("--segment", default=str(DEFAULT_SEGMENT))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    demo = build_ui(args.dataset, args.detect, args.segment)
    demo.launch(server_name=args.host, server_port=args.port, inbrowser=True)


if __name__ == "__main__":
    main()
