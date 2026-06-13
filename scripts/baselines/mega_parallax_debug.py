"""
Debug-визуализация mega parallax: зоны на target и откуда они семплируют t0/t1.

Пример:
  python mega_parallax_debug.py --sample-dir <path> --focus "traffic sign"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
import sys
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from layered_parallax import (
    AlphaTuneConfig,
    BlurTuneConfig,
    backwarp_map,
    get_lidar_depth,
    intrinsics_to_K,
    layer_alpha,
    load_rgb,
    remap_rgb,
)
from mega_parallax import (
    MegaParallaxConfig,
    MegaRegion,
    load_semantic_regions,
    make_mega_regions,
    mega_parallax_predict,
    rasterize_semantic_map,
)
from yolo.cityscapes_classes import CITYSCAPES_COLORS, CITYSCAPES_NAMES


def _load_tune_configs(root: Path) -> tuple[AlphaTuneConfig, BlurTuneConfig]:
    alpha_cfg = AlphaTuneConfig()
    blur_cfg = BlurTuneConfig()
    ap, bp = root / "alpha_tune_best.json", root / "blur_tune_best.json"
    if ap.is_file():
        bc = json.loads(ap.read_text(encoding="utf-8")).get("best_config", {})
        alpha_cfg = AlphaTuneConfig(**{k: bc[k] for k in alpha_cfg.__dataclass_fields__ if k in bc})
    if bp.is_file():
        bc = json.loads(bp.read_text(encoding="utf-8")).get("best_blur_config", {})
        blur_cfg = BlurTuneConfig(
            sigma_near=bc.get("sigma_near", 1.5),
            sigma_far=bc.get("sigma_far", 2.0),
            flow_k=bc.get("flow_k", 0.05),
            comp_k=bc.get("comp_k", 0.0),
        )
    return alpha_cfg, blur_cfg


def _region_color(i: int, cls_name: str) -> tuple[int, int, int]:
    base = CITYSCAPES_COLORS.get(cls_name, (128, 128, 128))
    rng = np.random.default_rng(i * 9973 + hash(cls_name) % 10000)
    jitter = rng.integers(-35, 36, size=3)
    return tuple(int(np.clip(c + j, 0, 255)) for c, j in zip(base, jitter))


def _overlay_mask(img: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = img.astype(np.float32).copy()
    c = np.array(color, dtype=np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * c
    return out.clip(0, 255).astype(np.uint8)


def _draw_sample_points(
    src: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    step: int = 6,
    radius: int = 2,
) -> np.ndarray:
    out = src.copy()
    ys, xs = np.where(mask)
    if ys.size == 0:
        return out
    idx = np.arange(0, ys.size, max(1, ys.size // 400))
    for y, x in zip(ys[idx], xs[idx]):
        sx = int(round(float(map_x[y, x])))
        sy = int(round(float(map_y[y, x])))
        if 0 <= sx < out.shape[1] and 0 <= sy < out.shape[0]:
            cv2.circle(out, (sx, sy), radius, color, -1, lineType=cv2.LINE_AA)
    return out


def _region_warp_parts(
    img_t0: np.ndarray,
    img_t1: np.ndarray,
    mean: np.ndarray,
    reg: MegaRegion,
    c2w_t0: np.ndarray,
    c2w_t1: np.ndarray,
    c2w_tgt: np.ndarray,
    K: np.ndarray,
    alpha: float,
    z_ref: float,
    n_layers: int,
    alpha_cfg: AlphaTuneConfig,
) -> dict[str, np.ndarray]:
    a_blend, _ = layer_alpha(reg.z_median, reg.depth_layer_idx, n_layers, alpha, z_ref, alpha_cfg)
    mw = alpha_cfg.mean_w if alpha_cfg.mean_w > 0 else 0.0

    mx0, my0, _ = backwarp_map(c2w_tgt, c2w_t0, K, reg.z_median, img_t0.shape[0], img_t0.shape[1])
    mx1, my1, _ = backwarp_map(c2w_tgt, c2w_t1, K, reg.z_median, img_t0.shape[0], img_t0.shape[1])
    w0_full = remap_rgb(img_t0, mx0, my0).astype(np.float32)
    w1_full = remap_rgb(img_t1, mx1, my1).astype(np.float32)
    warp_blend = (1 - a_blend) * w0_full + a_blend * w1_full
    final = (1 - mw) * warp_blend + mw * mean.astype(np.float32) if mw > 0 else warp_blend

    h, w = img_t0.shape[:2]
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    disp_t0 = np.stack([mx0 - u, my0 - v], axis=-1)
    disp_t1 = np.stack([mx1 - u, my1 - v], axis=-1)

    m = reg.mask
    black = np.zeros_like(img_t0)
    parts = {
        "w0_on_target": np.where(m[..., None], w0_full, 0).astype(np.uint8),
        "w1_on_target": np.where(m[..., None], w1_full, 0).astype(np.uint8),
        "mean_on_target": np.where(m[..., None], mean, 0),
        "blend_on_target": np.where(m[..., None], final, 0).clip(0, 255).astype(np.uint8),
        "disp_t0": disp_t0,
        "disp_t1": disp_t1,
        "map_x_t0": mx0,
        "map_y_t0": my0,
        "map_x_t1": mx1,
        "map_y_t1": my1,
        "a_blend": np.full((), a_blend),
        "mean_w": np.full((), mw),
    }
    return parts


def _disp_vis(disp: np.ndarray, mask: np.ndarray, scale: float = 4.0) -> np.ndarray:
    mag = np.sqrt(disp[..., 0] ** 2 + disp[..., 1] ** 2)
    out = np.zeros((*mag.shape, 3), dtype=np.uint8)
    if not np.any(mask):
        return out
    m = mag[mask]
    lo, hi = np.percentile(m, 5), np.percentile(m, 95)
    norm = np.clip((mag - lo) / max(hi - lo, 1e-6), 0, 1)
    hue = ((norm * 180).astype(np.uint8))  # 0..180 for cv2 HSV
    hsv = np.zeros((*mag.shape, 3), dtype=np.uint8)
    hsv[..., 0] = hue
    hsv[..., 1] = np.where(mask, 220, 0).astype(np.uint8)
    hsv[..., 2] = np.where(mask, (norm * 255).astype(np.uint8), 0)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return out


def _make_panel(cells: list[np.ndarray], labels: list[str], cols: int = 3, pad: int = 8) -> np.ndarray:
    if not cells:
        raise ValueError("empty panel")
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
        draw.text((x0 + 4, y0 + 4), label, fill=(240, 240, 240))
    return np.array(pil)


def visualize_sample(
    sample_dir: Path,
    split: str,
    semantic_jsonl: Path,
    out_dir: Path,
    cfg: MegaParallaxConfig,
    focus_class: str | None = None,
) -> Path:
    root = Path(__file__).resolve().parent
    alpha_cfg, blur_cfg = _load_tune_configs(root)
    cfg = cfg or MegaParallaxConfig()
    cfg.alpha_cfg = alpha_cfg
    cfg.blur_cfg = blur_cfg

    meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
    cam = meta["target_camera"]
    ts = meta["timestamps_ns"]
    alpha = float((ts["target"] - ts["t0"]) / (ts["t1"] - ts["t0"]))

    img_t0 = load_rgb(sample_dir / "input" / "t0" / f"{cam}.jpg")
    img_t1 = load_rgb(sample_dir / "input" / "t1" / f"{cam}.jpg")
    gt = load_rgb(sample_dir / "target" / f"{cam}.jpg")
    depth = get_lidar_depth(sample_dir, cam, "target")
    rel = f"{split}/{meta['sample_id']}/target/{cam}.jpg"
    sem = rasterize_semantic_map(load_semantic_regions(semantic_jsonl, rel), depth.shape[0], depth.shape[1])

    K = intrinsics_to_K(meta["intrinsics"][cam])
    c2w_t0 = np.array(meta["poses_c2w"]["t0"][cam], dtype=np.float64)
    c2w_t1 = np.array(meta["poses_c2w"]["t1"][cam], dtype=np.float64)
    c2w_tgt = np.array(meta["poses_c2w"]["target"][cam], dtype=np.float64)

    regions = make_mega_regions(depth, sem, cfg.n_depth_layers, cfg.min_region_pixels, cfg)
    pred, _, _ = mega_parallax_predict(
        img_t0, img_t1, c2w_t0, c2w_t1, c2w_tgt, K, depth, alpha, sem, cfg,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    sid = sample_dir.name[:40]
    valid = np.isfinite(depth) & (depth > 0)
    z_ref = float(np.median(depth[valid])) if np.any(valid) else 10.0
    mean = ((img_t0.astype(np.float32) + img_t1.astype(np.float32)) * 0.5).astype(np.uint8)

    # --- 1) карта всех зон на target ---
    zone_rgb = np.zeros_like(gt)
    zone_id = np.full(depth.shape, -1, dtype=np.int16)
    legend: list[dict] = []
    for i, reg in enumerate(regions):
        color = _region_color(i, reg.class_name)
        zone_rgb[reg.mask] = color
        zone_id[reg.mask] = i
        legend.append({
            "id": i,
            "class": reg.class_name,
            "z_m": round(reg.z_median, 2),
            "pixels": int(reg.mask.sum()),
            "layer_idx": reg.depth_layer_idx,
            "rgb": color,
        })

    zones_overlay = _overlay_mask(gt, zone_id >= 0, (255, 255, 255), 0.0)
    zones_overlay = (0.55 * gt.astype(np.float32) + 0.45 * zone_rgb.astype(np.float32)).astype(np.uint8)
    Image.fromarray(zones_overlay).save(out_dir / f"{sid}_zones_on_target.jpg", quality=92)
    Image.fromarray(zone_rgb).save(out_dir / f"{sid}_zones_flat.jpg", quality=92)

    # --- 2) семантика sign/pole на target + depth ---
    from yolo.cityscapes_classes import NAME_TO_ID
    sign_sem = sem == NAME_TO_ID["traffic sign"]
    pole_sem = sem == NAME_TO_ID["pole"]
    sign_depth = depth.copy()
    sign_depth[~sign_sem] = np.nan

    report_lines = [
        f"sample: {sample_dir.name}",
        f"camera: {cam}  alpha: {alpha:.3f}",
        f"regions: {len(regions)}  mode: {cfg.region_mode}",
        "",
        "=== Все зоны (target grid) ===",
    ]
    for item in legend:
        report_lines.append(
            f"#{item['id']:2d} {item['class']:14s} z={item['z_m']:6.1f}m  px={item['pixels']:6d}  layer={item['layer_idx']}"
        )

    focus_regs = [r for r in regions if focus_class and r.class_name == focus_class]
    if focus_class and not focus_regs:
        focus_regs = [r for r in regions if focus_class in r.class_name]

    # also auto-focus sign+pole if user asked sign
    if focus_class and "sign" in focus_class.lower():
        focus_regs = [r for r in regions if r.class_name in {"traffic sign", "pole"}]

    report_lines += ["", f"=== Зоны с классом '{focus_class or 'traffic sign/pole'}' ==="]
    for reg in focus_regs:
        report_lines.append(
            f"  {reg.class_name} z={reg.z_median:.1f}m px={int(reg.mask.sum())} "
            f"overlap_sign_sem={int((reg.mask & sign_sem).sum())}"
        )

    # --- 3) per-focus region matching panels ---
    panel_cells: list[np.ndarray] = []
    panel_labels: list[str] = []

    for idx, reg in enumerate(focus_regs or regions[:6]):
        parts = _region_warp_parts(
            img_t0, img_t1, mean, reg, c2w_t0, c2w_t1, c2w_tgt, K,
            alpha, z_ref, cfg.n_depth_layers, alpha_cfg,
        )
        m = reg.mask
        rid = next(i for i, r in enumerate(regions) if r is reg)
        tag = f"#{rid} {reg.class_name} z={reg.z_median:.1f}m"

        t0_pts = _draw_sample_points(img_t0, parts["map_x_t0"], parts["map_y_t0"], m, (255, 80, 80))
        t1_pts = _draw_sample_points(img_t1, parts["map_x_t1"], parts["map_y_t1"], m, (80, 160, 255))

        disp0 = _disp_vis(parts["disp_t0"], m)
        disp1 = _disp_vis(parts["disp_t1"], m)

        report_lines += [
            "",
            f"--- Region {tag} ---",
            f"  alpha_blend={parts['a_blend']:.3f}  mean_w={parts['mean_w']:.3f}",
            f"  disp_t0 px: mean={np.mean(np.abs(parts['disp_t0'][m])):.1f}  max={np.max(np.abs(parts['disp_t0'][m])):.1f}",
            f"  disp_t1 px: mean={np.mean(np.abs(parts['disp_t1'][m])):.1f}  max={np.max(np.abs(parts['disp_t1'][m])):.1f}",
        ]

        sub = _make_panel(
            [
                _overlay_mask(gt, m, _region_color(rid, reg.class_name)),
                parts["w0_on_target"],
                parts["w1_on_target"],
                parts["mean_on_target"],
                parts["blend_on_target"],
                np.where(m[..., None], pred, 0),
                t0_pts,
                t1_pts,
                disp0,
                disp1,
            ],
            [
                f"{tag} mask on GT",
                "w0 warp -> target",
                "w1 warp -> target",
                "mean (no warp)",
                f"blend a={parts['a_blend']:.2f} mw={parts['mean_w']:.2f}",
                "final pred (region)",
                "t0 + sample points",
                "t1 + sample points",
                "disp t0->target",
                "disp t1->target",
            ],
            cols=5,
        )
        out_reg = out_dir / f"{sid}_region_{rid:02d}_{reg.class_name.replace(' ', '_')}.jpg"
        Image.fromarray(sub).save(out_reg, quality=90)
        panel_cells.append(sub)
        panel_labels.append(tag)

    # --- 4) sign semantic vs which zones claim those pixels ---
    if np.any(sign_sem):
        claimed = np.zeros_like(sign_sem, dtype=np.int16) - 1
        for i, reg in enumerate(regions):
            claimed[reg.mask & sign_sem] = i
        conflict = sign_sem & (claimed < 0)
        multi = sign_sem.copy()
        report_lines += [
            "",
            "=== Семантика traffic sign vs зоны mega ===",
            f"  sign pixels (semantic): {int(sign_sem.sum())}",
            f"  sign covered by some region: {int((claimed >= 0).sum())}",
            f"  sign NOT in any mega region: {int(conflict.sum())}",
        ]
        for i, reg in enumerate(regions):
            ov = int((reg.mask & sign_sem).sum())
            if ov > 0:
                report_lines.append(
                    f"  zone #{i} ({reg.class_name}, z={reg.z_median:.1f}m) covers {ov} sign px"
                )

        sign_vis = gt.copy()
        sign_vis[conflict] = (255, 0, 255)
        for i, reg in enumerate(regions):
            ov = reg.mask & sign_sem
            if np.any(ov):
                sign_vis = _overlay_mask(sign_vis, ov, _region_color(i, reg.class_name), 0.5)
        Image.fromarray(sign_vis).save(out_dir / f"{sid}_sign_zone_overlap.jpg", quality=92)

    report_path = out_dir / f"{sid}_zones_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    # summary panel: t0 | zones | t1 | pred | gt
    summary = _make_panel(
        [img_t0, zones_overlay, img_t1, pred, gt],
        ["t0", "zones on target", "t1", "mega pred", "GT target"],
        cols=5,
    )
    summary_path = out_dir / f"{sid}_zones_summary.jpg"
    Image.fromarray(summary).save(summary_path, quality=92)
    return summary_path


def _seg_color(seg_id: int) -> tuple[int, int, int]:
    if seg_id < 0:
        return (32, 32, 32)
    if seg_id == 9999:
        return (255, 0, 255)
    return _region_color(int(seg_id), str(seg_id))


def _seg_rgb(seg_ids: np.ndarray) -> np.ndarray:
    h, w = seg_ids.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for sid in np.unique(seg_ids):
        if sid < 0:
            continue
        out[seg_ids == sid] = _seg_color(int(sid))
    return out


def _seg_boundaries(seg_ids: np.ndarray, thickness: int = 1) -> np.ndarray:
    edges = np.zeros(seg_ids.shape, dtype=bool)
    edges[:-1, :] |= seg_ids[:-1, :] != seg_ids[1:, :]
    edges[1:, :] |= seg_ids[1:, :] != seg_ids[:-1, :]
    edges[:, :-1] |= seg_ids[:, :-1] != seg_ids[:, 1:]
    edges[:, 1:] |= seg_ids[:, 1:] != seg_ids[:, :-1]
    if thickness > 1:
        edges = cv2.dilate(edges.astype(np.uint8), np.ones((thickness, thickness), np.uint8)).astype(bool)
    return edges


def visualize_source_match_segments(
    sample_dir: Path,
    split: str,
    semantic_jsonl: Path,
    out_dir: Path,
    cfg: MegaParallaxConfig | None = None,
) -> Path:
    from layered_parallax import get_lidar_depth, intrinsics_to_K, load_rgb
    from mega_parallax import (
        build_geo_match_context,
        build_t0_topology_on_target,
        load_semantic_regions,
        mega_parallax_predict_source_match,
        project_region_to_target,
        rasterize_semantic_map,
        target_topology_to_rgb,
        topology_map_to_rgb,
        warp_topology_by_flow,
    )

    cfg = cfg or MegaParallaxConfig(zoning_frame="source_match")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
    cam = meta["target_camera"]
    ts = meta["timestamps_ns"]
    alpha = float((ts["target"] - ts["t0"]) / (ts["t1"] - ts["t0"]))
    sid = meta["sample_id"]

    img_t0 = load_rgb(sample_dir / "input" / "t0" / f"{cam}.jpg")
    img_t1 = load_rgb(sample_dir / "input" / "t1" / f"{cam}.jpg")
    gt = load_rgb(sample_dir / "target" / f"{cam}.jpg")
    h, w = img_t0.shape[:2]

    K = intrinsics_to_K(meta["intrinsics"][cam])
    c2w_t0 = np.array(meta["poses_c2w"]["t0"][cam], dtype=np.float64)
    c2w_t1 = np.array(meta["poses_c2w"]["t1"][cam], dtype=np.float64)
    c2w_tgt = np.array(meta["poses_c2w"]["target"][cam], dtype=np.float64)

    depth_t0 = get_lidar_depth(sample_dir, cam, "t0")
    depth_t1 = get_lidar_depth(sample_dir, cam, "t1")
    depth_tgt = get_lidar_depth(sample_dir, cam, "target")
    sem_t0 = rasterize_semantic_map(
        load_semantic_regions(semantic_jsonl, f"{split}/{sid}/input/t0/{cam}.jpg"), h, w,
    )
    sem_t1 = rasterize_semantic_map(
        load_semantic_regions(semantic_jsonl, f"{split}/{sid}/input/t1/{cam}.jpg"), h, w,
    )

    pred, regions_t0, regions_t1, pairs, seg_ids, labels, topo_result = mega_parallax_predict_source_match(
        img_t0, img_t1, c2w_t0, c2w_t1, c2w_tgt, K,
        depth_t0, depth_t1, sem_t0, sem_t1, alpha, cfg, track_segments=True, depth_tgt=depth_tgt,
    )
    topo_target, topo_target_labels = build_t0_topology_on_target(regions_t0, c2w_t0, c2w_tgt, K, (h, w))

    assert seg_ids is not None and topo_result is not None
    seg_rgb = _seg_rgb(seg_ids)

    rgb_t0 = topology_map_to_rgb(topo_result.topo_t0, regions_t0)
    rgb_t1 = topology_map_to_rgb(topo_result.topo_t1, regions_t1)
    topo_warp = warp_topology_by_flow(topo_result.topo_t0, topo_result.flow_refined)
    rgb_warp = topology_map_to_rgb(topo_warp, regions_t0)
    agree = np.zeros((h, w, 3), dtype=np.uint8)
    ok = (topo_warp > 0) & (topo_result.topo_t1 > 0) & (topo_warp == topo_result.topo_t1)
    bad = (topo_warp > 0) & (topo_result.topo_t1 > 0) & (topo_warp != topo_result.topo_t1)
    agree[ok] = (0, 200, 0)
    agree[bad] = (220, 0, 0)

    geo = build_geo_match_context(c2w_t0, c2w_t1, c2w_tgt, K, depth_tgt, alpha, h, w, cfg.n_depth_layers)
    rgb_tgt_topo = target_topology_to_rgb(
        topo_result.target_topo if topo_result.target_topo is not None else np.full((h, w), -1, dtype=np.int16),
        geo.target_layer_z,
    )
    tgt_agree = np.zeros((h, w, 3), dtype=np.uint8)
    topo0_tgt = np.full((h, w), -1, dtype=np.int16)
    topo1_tgt = np.full((h, w), -1, dtype=np.int16)
    for i, pair in enumerate(pairs):
        lid = i + 1
        m0 = project_region_to_target(pair.reg_t0, c2w_t0, c2w_tgt, K, h, w)
        m1 = project_region_to_target(pair.reg_t1, c2w_t1, c2w_tgt, K, h, w)
        topo0_tgt[m0] = lid
        topo1_tgt[m1] = lid
    tok = (topo0_tgt > 0) & (topo1_tgt > 0) & (topo0_tgt == topo1_tgt)
    tbad = (topo0_tgt > 0) & (topo1_tgt > 0) & (topo0_tgt != topo1_tgt)
    tgt_agree[tok] = (0, 200, 0)
    tgt_agree[tbad] = (220, 0, 0)

    edges = _seg_boundaries(seg_ids, thickness=2)
    pred_edges = pred.copy()
    pred_edges[edges] = (255, 40, 40)
    seg_on_pred = (0.45 * pred.astype(np.float32) + 0.55 * seg_rgb.astype(np.float32)).astype(np.uint8)

    panel = _make_panel(
        [rgb_t0, rgb_t1, rgb_warp, agree, seg_on_pred, pred_edges, gt],
        [
            "t0 zones",
            "t1 zones",
            f"t0 aligned lbl={topo_result.topo_iou:.2f} tgt={topo_result.target_topo_iou:.2f}",
            "green=match red=mismatch",
            "pred + segments",
            "segment seams",
            "GT",
        ],
        cols=4,
    )

    stem = sample_dir.name[:40]
    panel_path = out_dir / f"{stem}_source_match_segments.jpg"
    Image.fromarray(panel).save(panel_path, quality=92)

    pred_path = out_dir / f"{stem}_source_match_pred.jpg"
    Image.fromarray(pred).save(pred_path, quality=92)

    topo_panel = _make_panel(
        [rgb_tgt_topo, rgb_t0, rgb_t1, tgt_agree],
        [
            "target LiDAR layers (ref)",
            "t0 sem zones",
            "t1 sem zones",
            f"t0|t1 on target tgt_iou={topo_result.target_topo_iou:.2f}",
        ],
        cols=2,
    )
    topo_path = out_dir / f"{stem}_topology.jpg"
    Image.fromarray(topo_panel).save(topo_path, quality=92)

    matched_t0 = {id(p.reg_t0) for p in pairs}
    unmatched_t0 = [r for r in regions_t0 if id(r) not in matched_t0]
    used_t1 = {id(p.reg_t1) for p in pairs}
    unmatched_t1 = [r for r in regions_t1 if id(r) not in used_t1]

    def _name_for_seg(seg_id: int) -> str:
        if seg_id == 9999:
            return "fallback farneback"
        if 1 <= seg_id <= len(pairs):
            p = pairs[seg_id - 1]
            return (
                f"P{seg_id - 1} {p.reg_t0.class_name}|{p.reg_t1.class_name} "
                f"z0={p.reg_t0.z_median:.0f} z1={p.reg_t1.z_median:.0f} iou={p.iou:.2f}"
            )
        u0_base = len(pairs) + 1
        if u0_base <= seg_id < u0_base + len(unmatched_t0):
            r = unmatched_t0[seg_id - u0_base]
            return f"U0 {r.class_name} z={r.z_median:.0f}m (t0 only)"
        u1_base = u0_base + len(unmatched_t0)
        if u1_base <= seg_id < u1_base + len(unmatched_t1):
            r = unmatched_t1[seg_id - u1_base]
            return f"U1 {r.class_name} z={r.z_median:.0f}m (t1 only)"
        return f"id={seg_id}"

    report = [
        f"sample: {sample_dir.name}",
        f"topology label iou after align: {topo_result.topo_iou:.3f}",
        f"topology class iou: {topo_result.class_iou:.3f}",
        f"target depth topology iou: {topo_result.target_topo_iou:.3f}",
        f"coarse topology pairs: {len(topo_result.coarse_pairs)}",
        f"matched pairs: {len(pairs)} (topology-confirmed: {sum(1 for p in pairs if p.from_topology)})",
        f"t0 zones: {len(regions_t0)}  t1 zones: {len(regions_t1)}",
        f"unmatched t0: {len(unmatched_t0)}  unmatched t1: {len(unmatched_t1)}",
        "",
        "=== Coarse topology (class + depth rank) ===",
    ]
    for i, (r0, r1) in enumerate(topo_result.coarse_pairs):
        report.append(
            f"  C{i} {r0.class_name} z0={r0.z_median:.0f} <-> {r1.class_name} z1={r1.z_median:.0f}"
        )
    report += ["", "=== Final matched pairs ==="]
    for i, p in enumerate(pairs):
        tag = "topo" if p.from_topology else "iou"
        report.append(
            f"  P{i} [{tag}] {p.reg_t0.class_name}|{p.reg_t1.class_name} "
            f"iou={p.iou:.2f} flow=({p.flow_du:+.1f},{p.flow_dv:+.1f})"
        )
    report += ["", "=== Сегменты результата ==="]
    for seg_id in sorted(int(x) for x in np.unique(seg_ids) if int(x) >= 0):
        report.append(
            f"  id={seg_id:4d}  px={int((seg_ids == seg_id).sum()):7d}  {_name_for_seg(seg_id)}"
        )

    report += ["", "=== Expected on target pose (t0→target) ==="]
    for i, lab in enumerate(topo_target_labels[1:], start=1):
        report.append(f"  id={i:4d}  px={int((topo_target == i).sum()):7d}  {lab}")

    (out_dir / f"{stem}_source_match_segments.txt").write_text("\n".join(report), encoding="utf-8")
    return panel_path, pred_path, topo_path


def main() -> int:
    p = argparse.ArgumentParser(description="Mega parallax zone / t0-t1 matching debug")
    p.add_argument("--sample-dir", type=Path, required=True)
    p.add_argument("--split", default="train")
    p.add_argument(
        "--semantic-jsonl",
        default=r"C:/Users/adel/Downloads/cv_dataset/annotations/semantic/semantic.jsonl",
    )
    p.add_argument("--out-dir", type=Path, default=Path("methods_gallery/_preview/zones"))
    p.add_argument("--focus", default="traffic sign", help="Класс для детальной панели")
    p.add_argument("--region-mode", default="object", choices=["object", "grid"])
    p.add_argument("--n-depth-layers", type=int, default=8)
    p.add_argument("--mode", default="source_match", choices=["target", "source_match"])
    args = p.parse_args()

    cfg = MegaParallaxConfig(
        region_mode=args.region_mode,
        n_depth_layers=args.n_depth_layers,
        zoning_frame="source_match" if args.mode == "source_match" else "target",
    )

    if args.mode == "source_match":
        panel_path, pred_path, topo_path = visualize_source_match_segments(
            args.sample_dir.resolve(),
            args.split,
            Path(args.semantic_jsonl),
            args.out_dir.resolve(),
            cfg,
        )
        print(f"Saved: {panel_path}")
        print(f"Saved: {pred_path}")
        print(f"Saved: {topo_path}")
        return 0

    out = visualize_sample(
        args.sample_dir.resolve(),
        args.split,
        Path(args.semantic_jsonl),
        args.out_dir.resolve(),
        cfg,
        args.focus,
    )
    print(f"Saved: {out}")
    print(f"Report: {args.out_dir / (args.sample_dir.name[:40] + '_zones_report.txt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
