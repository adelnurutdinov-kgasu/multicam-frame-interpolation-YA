"""
Просмотр боксов, instance-seg и semantic-seg поверх оригиналов.

Запуск:
  python viewer/app.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gradio as gr

from viewer.render import AnnotationStore, render_overlay

DEFAULT_DATASET = Path(r"C:/Users/adel/Downloads/cv_dataset/final_dataset_v5_participants")
DEFAULT_DETECT = Path(r"C:/Users/adel/Downloads/cv_dataset/annotations/detect/detections.jsonl")
DEFAULT_SEGMENT = Path(r"C:/Users/adel/Downloads/cv_dataset/annotations/segment/segments.jsonl")
DEFAULT_SEMANTIC = Path(r"C:/Users/adel/Downloads/cv_dataset/annotations/semantic/semantic.jsonl")

_store: AnnotationStore | None = None
_store_key: tuple[str, str, str, str] | None = None
_filtered: list[str] = []


def get_store(dataset: str, detect: str, segment: str, semantic: str) -> AnnotationStore:
    global _store, _store_key
    key = (str(Path(dataset).resolve()), str(Path(detect).resolve()), str(Path(segment).resolve()), str(Path(semantic).resolve()))
    if _store is None or _store_key != key:
        _store = AnnotationStore.open(Path(dataset), Path(detect), Path(segment), Path(semantic))
        _store_key = key
    return _store


def reload_store(dataset: str, detect: str, segment: str, semantic: str) -> tuple[str, gr.Slider]:
    store = get_store(dataset, detect, segment, semantic)
    global _filtered
    _filtered = store.images
    msg = (
        f"Кадров: {len(store.images)} | "
        f"detect: {len(store.detections)} | "
        f"inst-seg: {len(store.segments)} | "
        f"semantic: {len(store.semantics)}"
    )
    max_idx = max(0, len(_filtered) - 1)
    return msg, gr.Slider(minimum=0, maximum=max_idx, value=0, step=1, label="Индекс кадра")


def apply_filter(
    dataset: str,
    detect: str,
    segment: str,
    semantic: str,
    path_filter: str,
    only_boxes: bool,
    only_seg: bool,
    only_sem: bool,
) -> tuple[str, gr.Slider]:
    store = get_store(dataset, detect, segment, semantic)
    global _filtered
    _filtered = store.filter_images(path_filter, only_boxes, only_seg, only_sem)
    max_idx = max(0, len(_filtered) - 1)
    return f"После фильтра: {len(_filtered)} кадров", gr.Slider(minimum=0, maximum=max_idx, value=0, step=1, label="Индекс кадра")


def show_frame(
    dataset: str,
    detect: str,
    segment: str,
    semantic: str,
    index: int,
    show_boxes: bool,
    show_segments: bool,
    show_semantic: bool,
    conf: float,
    path_filter: str,
    only_boxes: bool,
    only_seg: bool,
    only_sem: bool,
    class_filter: str,
) -> tuple:
    store = get_store(dataset, detect, segment, semantic)
    global _filtered
    _filtered = store.filter_images(path_filter, only_boxes, only_seg, only_sem)
    if not _filtered:
        return None, "Нет кадров по фильтру"

    idx = int(min(max(0, index), len(_filtered) - 1))
    rel = _filtered[idx]
    classes = [c.strip() for c in class_filter.split(",") if c.strip()] or None
    rgb, info = render_overlay(store, rel, show_boxes, show_segments, show_semantic, conf, classes)
    return rgb, f"[{idx + 1}/{len(_filtered)}]\n{info}"


def step(index: int, delta: int) -> int:
    global _filtered
    if not _filtered:
        return 0
    return int(min(max(0, index + delta), len(_filtered) - 1))


def build_ui(default_dataset: str, default_detect: str, default_segment: str, default_semantic: str) -> gr.Blocks:
    with gr.Blocks(title="CV Dataset Viewer") as demo:
        gr.Markdown(
            "# Просмотр: боксы + объекты + семантика сцены\n"
            "**Семантика** — road, sky, vegetation, pole, building… (Cityscapes). Overlay в памяти."
        )
        with gr.Row():
            dataset_in = gr.Textbox(label="Датасет", value=default_dataset)
            detect_in = gr.Textbox(label="detections.jsonl", value=default_detect)
        with gr.Row():
            segment_in = gr.Textbox(label="segments.jsonl (inst)", value=default_segment)
            semantic_in = gr.Textbox(label="semantic.jsonl (сцена)", value=default_semantic)
        reload_btn = gr.Button("Перезагрузить")
        status = gr.Textbox(label="Статус", interactive=False)

        with gr.Row():
            path_filter = gr.Textbox(label="Фильтр пути", placeholder="left_fwd, target, test/...")
            only_boxes = gr.Checkbox(label="Только с боксами")
            only_seg = gr.Checkbox(label="Только inst-seg")
            only_sem = gr.Checkbox(label="Только semantic")
            filter_btn = gr.Button("Фильтр")

        with gr.Row():
            show_boxes = gr.Checkbox(label="Боксы (detect)", value=False)
            show_segments = gr.Checkbox(label="Объекты (inst-seg)", value=False)
            show_semantic = gr.Checkbox(label="Сцена (semantic)", value=True)
            conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="conf (для detect/inst)")
        class_filter = gr.Textbox(
            label="Классы через запятую (пусто = все)",
            placeholder="road, vegetation, sky, pole, car",
        )

        index = gr.Slider(0, 0, value=0, step=1, label="Индекс")
        with gr.Row():
            prev_btn = gr.Button("←")
            next_btn = gr.Button("→")
        image_out = gr.Image(label="Кадр", type="numpy")
        info_out = gr.Textbox(label="Инфо", lines=5, interactive=False)

        inputs = [
            dataset_in, detect_in, segment_in, semantic_in, index,
            show_boxes, show_segments, show_semantic, conf,
            path_filter, only_boxes, only_seg, only_sem, class_filter,
        ]
        reload_args = [dataset_in, detect_in, segment_in, semantic_in]

        reload_btn.click(reload_store, reload_args, [status, index])
        filter_btn.click(apply_filter, reload_args + [path_filter, only_boxes, only_seg, only_sem], [status, index])
        for ctrl in [index, show_boxes, show_segments, show_semantic, conf, class_filter]:
            ctrl.change(show_frame, inputs, [image_out, info_out])
        for ctrl in [path_filter, only_boxes, only_seg, only_sem]:
            ctrl.change(show_frame, inputs, [image_out, info_out])
        prev_btn.click(lambda i: step(i, -1), index, index).then(show_frame, inputs, [image_out, info_out])
        next_btn.click(lambda i: step(i, 1), index, index).then(show_frame, inputs, [image_out, info_out])
        demo.load(reload_store, reload_args, [status, index]).then(show_frame, inputs, [image_out, info_out])

    return demo


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--detect", default=str(DEFAULT_DETECT))
    p.add_argument("--segment", default=str(DEFAULT_SEGMENT))
    p.add_argument("--semantic", default=str(DEFAULT_SEMANTIC))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    build_ui(args.dataset, args.detect, args.segment, args.semantic).launch(
        server_name=args.host, server_port=args.port, inbrowser=True
    )


if __name__ == "__main__":
    main()
