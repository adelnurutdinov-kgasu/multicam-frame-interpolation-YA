"""
Сравнение стратегий topology match на одном сэмпле.

  python mega_parallax_topology_compare.py --sample-dir <path>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from layered_parallax import get_lidar_depth, intrinsics_to_K, load_rgb, psnr
from mega_parallax import (
    TOPOLOGY_MATCH_MODES,
    MegaParallaxConfig,
    load_semantic_regions,
    mega_parallax_predict_source_match,
    rasterize_semantic_map,
    topology_map_to_rgb,
    warp_topology_by_flow,
)


@dataclass
class MatchEval:
    mode: str
    psnr: float
    label_iou: float
    class_iou: float
    target_topo_iou: float
    n_pairs: int
    n_topo_pairs: int
    unmatched_t0: int
    unmatched_t1: int
    fallback_px: int
    mean_pair_iou: float
    pred: np.ndarray
    agree: np.ndarray


def _make_panel(cells: list[np.ndarray], labels: list[str], cols: int = 3, pad: int = 8) -> np.ndarray:
    h, w = cells[0].shape[:2]
    rows = (len(cells) + cols - 1) // cols
    label_h = 28
    canvas_h = rows * (h + label_h + pad) + pad
    canvas_w = cols * (w + pad) + pad
    canvas = np.full((canvas_h, canvas_w, 3), 32, dtype=np.uint8)
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    for i, (img, label) in enumerate(zip(cells, labels)):
        r, c = divmod(i, cols)
        y0 = pad + r * (h + label_h + pad)
        x0 = pad + c * (w + pad)
        cell = img if img.shape[:2] == (h, w) else cv2.resize(img, (w, h))
        if cell.ndim == 2:
            cell = cv2.cvtColor(cell, cv2.COLOR_GRAY2RGB)
        pil.paste(Image.fromarray(cell), (x0, y0 + label_h))
        draw.text((x0 + 4, y0 + 4), label[:80], fill=(240, 240, 240))
    return np.array(pil)


def _agree_map(topo_result, regions_t0) -> np.ndarray:
    h, w = topo_result.topo_t0.shape
    topo_warp = warp_topology_by_flow(topo_result.topo_t0, topo_result.flow_refined)
    agree = np.zeros((h, w, 3), dtype=np.uint8)
    ok = (topo_warp > 0) & (topo_result.topo_t1 > 0) & (topo_warp == topo_result.topo_t1)
    bad = (topo_warp > 0) & (topo_result.topo_t1 > 0) & (topo_warp != topo_result.topo_t1)
    agree[ok] = (0, 200, 0)
    agree[bad] = (220, 0, 0)
    return agree


def evaluate_mode(
    img_t0: np.ndarray,
    img_t1: np.ndarray,
    gt: np.ndarray,
    c2w_t0: np.ndarray,
    c2w_t1: np.ndarray,
    c2w_tgt: np.ndarray,
    K: np.ndarray,
    depth_t0: np.ndarray,
    depth_t1: np.ndarray,
    depth_tgt: np.ndarray,
    sem_t0: np.ndarray,
    sem_t1: np.ndarray,
    alpha: float,
    mode: str,
    base_cfg: MegaParallaxConfig,
) -> MatchEval:
    if mode == "iou_only":
        cfg = replace(base_cfg, topology_align=False, topology_match_mode="none")
    else:
        cfg = replace(base_cfg, topology_align=True, topology_match_mode=mode)

    pred, regions_t0, regions_t1, pairs, seg_ids, _labels, topo_result = mega_parallax_predict_source_match(
        img_t0, img_t1, c2w_t0, c2w_t1, c2w_tgt, K,
        depth_t0, depth_t1, sem_t0, sem_t1, alpha, cfg, track_segments=True, depth_tgt=depth_tgt,
    )
    assert seg_ids is not None

    matched_t0 = {id(p.reg_t0) for p in pairs}
    matched_t1 = {id(p.reg_t1) for p in pairs}
    unmatched_t0 = sum(1 for r in regions_t0 if id(r) not in matched_t0)
    unmatched_t1 = sum(1 for r in regions_t1 if id(r) not in matched_t1)
    fallback_px = int((seg_ids == 9999).sum())
    mean_iou = float(np.mean([p.iou for p in pairs])) if pairs else 0.0

    if topo_result is not None:
        agree = _agree_map(topo_result, regions_t0)
        label_iou = topo_result.topo_iou
        class_iou = topo_result.class_iou
        target_topo_iou = topo_result.target_topo_iou
        n_topo = len(topo_result.coarse_pairs)
    else:
        agree = np.zeros_like(pred)
        label_iou = 0.0
        class_iou = 0.0
        target_topo_iou = 0.0
        n_topo = 0

    return MatchEval(
        mode=mode,
        psnr=psnr(pred, gt),
        label_iou=label_iou,
        class_iou=class_iou,
        target_topo_iou=target_topo_iou,
        n_pairs=len(pairs),
        n_topo_pairs=n_topo,
        unmatched_t0=unmatched_t0,
        unmatched_t1=unmatched_t1,
        fallback_px=fallback_px,
        mean_pair_iou=mean_iou,
        pred=pred,
        agree=agree,
    )


def compare_sample(
    sample_dir: Path,
    split: str,
    semantic_jsonl: Path,
    out_dir: Path,
    cfg: MegaParallaxConfig | None = None,
) -> Path:
    cfg = cfg or MegaParallaxConfig(zoning_frame="source_match")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
    cam = meta["target_camera"]
    ts = meta["timestamps_ns"]
    alpha = float((ts["target"] - ts["t0"]) / (ts["t1"] - ts["t0"]))
    sid = sample_dir.name[:40]

    img_t0 = load_rgb(sample_dir / "input" / "t0" / f"{cam}.jpg")
    img_t1 = load_rgb(sample_dir / "input" / "t1" / f"{cam}.jpg")
    gt = load_rgb(sample_dir / "target" / f"{cam}.jpg")
    K = intrinsics_to_K(meta["intrinsics"][cam])
    c2w_t0 = np.array(meta["poses_c2w"]["t0"][cam], dtype=np.float64)
    c2w_t1 = np.array(meta["poses_c2w"]["t1"][cam], dtype=np.float64)
    c2w_tgt = np.array(meta["poses_c2w"]["target"][cam], dtype=np.float64)
    depth_t0 = get_lidar_depth(sample_dir, cam, "t0")
    depth_t1 = get_lidar_depth(sample_dir, cam, "t1")
    depth_tgt = get_lidar_depth(sample_dir, cam, "target")
    sem_t0 = rasterize_semantic_map(
        load_semantic_regions(semantic_jsonl, f"{split}/{meta['sample_id']}/input/t0/{cam}.jpg"),
        img_t0.shape[0], img_t0.shape[1],
    )
    sem_t1 = rasterize_semantic_map(
        load_semantic_regions(semantic_jsonl, f"{split}/{meta['sample_id']}/input/t1/{cam}.jpg"),
        img_t0.shape[0], img_t0.shape[1],
    )

    modes = list(TOPOLOGY_MATCH_MODES) + ["iou_only"]
    results: list[MatchEval] = []
    for mode in modes:
        print(f"  {mode}...", flush=True)
        results.append(evaluate_mode(
            img_t0, img_t1, gt, c2w_t0, c2w_t1, c2w_tgt, K,
            depth_t0, depth_t1, depth_tgt, sem_t0, sem_t1, alpha, mode, cfg,
        ))

    results.sort(key=lambda r: (-r.psnr, -r.target_topo_iou, -r.class_iou))

    cells: list[np.ndarray] = [gt]
    labels: list[str] = ["GT"]
    for r in results:
        cells.append(r.agree)
        labels.append(f"{r.mode} agree cls={r.class_iou:.2f}")
        cells.append(r.pred)
        labels.append(f"{r.mode} pred PSNR={r.psnr:.2f}")

    panel = _make_panel(cells, labels, cols=4)
    panel_path = out_dir / f"{sid}_topology_compare.jpg"
    Image.fromarray(panel).save(panel_path, quality=92)

    lines = [
        f"sample: {sample_dir.name}",
        f"sorted by PSNR (best first)",
        "",
        f"{'mode':<22} {'PSNR':>6} {'tgt':>5} {'cls':>5} {'pairs':>5} "
        f"{'coarse':>6} {'u0':>3} {'u1':>3} {'fallback':>9} {'mean_iou':>8}",
        "-" * 98,
    ]
    for r in results:
        lines.append(
            f"{r.mode:<22} {r.psnr:6.2f} {r.target_topo_iou:5.3f} {r.class_iou:5.3f} "
            f"{r.n_pairs:5d} {r.n_topo_pairs:6d} {r.unmatched_t0:3d} {r.unmatched_t1:3d} "
            f"{r.fallback_px:9d} {r.mean_pair_iou:8.3f}"
        )
    report_path = out_dir / f"{sid}_topology_compare.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved: {panel_path}")
    print(f"Saved: {report_path}")
    return panel_path


def main() -> int:
    p = argparse.ArgumentParser(description="Compare topology matching strategies")
    p.add_argument("--sample-dir", type=Path, required=True)
    p.add_argument("--split", default="train")
    p.add_argument(
        "--semantic-jsonl",
        default=r"C:/Users/adel/Downloads/cv_dataset/annotations/semantic/semantic.jsonl",
    )
    p.add_argument("--out-dir", type=Path, default=Path("methods_gallery/_preview/zones"))
    p.add_argument("--region-mode", default="object", choices=["object", "grid"])
    p.add_argument("--n-depth-layers", type=int, default=8)
    args = p.parse_args()

    cfg = MegaParallaxConfig(
        region_mode=args.region_mode,
        n_depth_layers=args.n_depth_layers,
        zoning_frame="source_match",
    )
    compare_sample(
        args.sample_dir.resolve(),
        args.split,
        Path(args.semantic_jsonl),
        args.out_dir.resolve(),
        cfg,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
