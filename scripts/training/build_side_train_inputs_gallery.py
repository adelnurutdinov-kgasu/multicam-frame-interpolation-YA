"""HTML-галерея: только инпуты U-Net + LiDAR r3+blur для post-blend (side train).

Пример:
  python scripts/training/build_side_train_inputs_gallery.py
  python scripts/training/build_side_train_inputs_gallery.py --num 10 --seed 7
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from html import escape
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from consensus_kit import (  # noqa: E402
    ConsensusConfig,
    _edge_contrast_mask,
    _expand_warp_nearest,
    _chw_npy_to_hwc_u8,
    _load_ego_mask,
    _local_median_rgb01,
    _norm_depth,
    _resize_hw,
    _resize_maps,
    build_base_init,
    discover_samples,
    selective_median_outlier_fix,
    telea_postprocess_variants,
)
from lib.lidar_density_mask import lidar_rife_blend_maps  # noqa: E402
from ya_paths import BAKED_ROOT, CV_ROOT, EGO_MASKS_APPROVED, RIFE_ROOT, TRAIN_SPLIT, WARPS_ROOT  # noqa: E402

SIDE_CAMERAS = ("left_fwd", "right_fwd", "left_bwd", "right_bwd")
MIRROR_CAMERAS = ("left_fwd", "right_bwd")
DEFAULT_OUT = REPO / "methods_gallery" / "_side_train_inputs_preview"

# Только то, что идёт в model(inputs, effective_mask, base_init)
UNET_COLUMNS = (
    ("ch0–2 warp_rgb", "inp_warp.jpg"),
    ("warp expand (smart)", "inp_warp_expand.jpg"),
    ("warp+mean in no-trust (50/50)", "inp_warp_notrust_mix.jpg"),
    ("|delta| warp→expand", "inp_warp_expand_delta.jpg"),
    ("ch3 coverage", "inp_coverage.png"),
    ("ch4 depth_norm", "inp_depth.png"),
    ("ch9 ego_mask", "inp_ego.png"),
    ("ch10–12 mean×(1−ego)", "inp_mean_scene.jpg"),
    ("base_init (legacy)", "base_init_legacy.jpg"),
    ("base_init (expand+blur)", "base_init.jpg"),
    ("|delta| legacy→new", "base_init_delta.jpg"),
    ("effective_mask", "eff_mask.png"),
)

# Post-inference blend (image space камеры, без mirror flip)
BLEND_COLUMNS = (
    ("GT (cam space)", "ref_gt.jpg"),
    ("r3+blur density_fine", "lidar_density_fine.png"),
    ("lidar_trust", "lidar_trust.png"),
    ("blend_mask ≥0.12", "lidar_blend_mask.png"),
    ("trust overlay", "lidar_overlay.jpg"),
)
EXPAND_METHODS_DEFAULT = ("nearest", "dilate", "inpaint_telea", "inpaint_ns")
FIXED_TELEA_VARIANT = "telea_med5_outlier_soft_prefill_n030"


def side_cfg(warp_expand_px: float, post_blur_sigma: float, expand_method: str) -> ConsensusConfig:
    cfg = ConsensusConfig(
        dataset_root=str(TRAIN_SPLIT),
        consensus_root=str(WARPS_ROOT / "train"),
        baked_root=str(BAKED_ROOT / "train"),
        rife_root=str(RIFE_ROOT / "train"),
        ego_masks_root=str(EGO_MASKS_APPROVED),
        allowed_cameras=SIDE_CAMERAS,
        image_h=544,
        image_w=1024,
        use_static_far=False,
        use_artifact_mask=True,
        use_anchor_frames=True,
    )
    cfg.base_warp_expand_px = float(warp_expand_px)
    cfg.base_post_blur_sigma = float(post_blur_sigma)
    cfg.base_expand_method = str(expand_method)
    cfg.no_trust_mix_alpha = 0.5
    cfg.no_trust_thr = 1e-4
    cfg.no_trust_feather_px = 2.0
    return cfg


def _flip_h(*arrays):
    return tuple(a[:, ::-1].copy() for a in arrays)


def pick_diverse(baked_dirs: list[Path], n: int, seed: int) -> list[Path]:
    by_cam: dict[str, list[Path]] = defaultdict(list)
    for p in baked_dirs:
        meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
        by_cam[meta["camera"]].append(p)
    rng = np.random.RandomState(seed)
    for cam in by_cam:
        rng.shuffle(by_cam[cam])

    picked: list[Path] = []
    cams = [c for c in SIDE_CAMERAS if by_cam[c]]
    idx = {c: 0 for c in cams}
    while len(picked) < n and any(idx[c] < len(by_cam[c]) for c in cams):
        for c in cams:
            if len(picked) >= n:
                break
            if idx[c] < len(by_cam[c]):
                picked.append(by_cam[c][idx[c]])
                idx[c] += 1
    return picked[:n]


def _f01_u8(a: np.ndarray) -> np.ndarray:
    return (np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)


def _scalar_map(m: np.ndarray, cmap: int = cv2.COLORMAP_VIRIDIS) -> np.ndarray:
    u8 = (np.clip(m, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(u8, cmap)


def _overlay_mask(rgb_u8: np.ndarray, mask: np.ndarray, color=(255, 180, 40), alpha=0.42) -> np.ndarray:
    out = rgb_u8.astype(np.float32).copy()
    m = np.clip(mask, 0.0, 1.0)[..., None]
    col = np.array(color, dtype=np.float32)
    out = out * (1.0 - m * alpha) + col * (m * alpha)
    return out.astype(np.uint8)


def _local_bilateral_rgb01(rgb01: np.ndarray, d: int = 5, sigma_color: float = 25.0, sigma_space: float = 3.0) -> np.ndarray:
    u8 = _f01_u8(rgb01)
    out = cv2.bilateralFilter(u8, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    return out.astype(np.float32) / 255.0


def _prefill_holes_mean_noise(
    base: np.ndarray,
    cov: np.ndarray,
    noise_std: float = 0.012,
    black_thr: float = 0.02,
    black_lift_floor: float = 0.16,
    seed: int = 42,
) -> np.ndarray:
    """
    Заполняем дырки + абсолютно чёрные пиксели: локальный mean 5x5 + очень слабый шум.
    black_thr — порог "абсолютно чёрного" по grayscale в [0..1].
    black_lift_floor — минимальная яркость (gray) для forced-black области.
    """
    holes = (np.clip(cov, 0.0, 1.0) <= 1e-4)
    gray = np.mean(np.clip(base, 0.0, 1.0), axis=-1)
    black = gray <= float(black_thr)
    fill_mask = holes | black
    if not np.any(fill_mask):
        return base
    out = base.copy()
    rng = np.random.RandomState(seed)

    # Локальный цветовой центр (RGB, то же пространство, что у исходника).
    mu = cv2.blur(np.clip(base, 0.0, 1.0).astype(np.float32), (5, 5))

    # Коррелированный RGB-шум из статистики валидной части текущего кадра.
    valid = np.clip(cov, 0.0, 1.0) > 1e-4
    ref = base[valid]
    if ref.shape[0] < 64:
        ref = base.reshape(-1, 3)
    ref = np.clip(ref.astype(np.float32), 0.0, 1.0)
    ref_centered = ref - ref.mean(axis=0, keepdims=True)
    cov_rgb = (ref_centered.T @ ref_centered) / max(1, ref_centered.shape[0] - 1)
    # Нормируем общую силу и масштабируем noise_std.
    tr = float(np.trace(cov_rgb))
    if tr > 1e-8:
        cov_rgb = cov_rgb / tr
    cov_rgb = cov_rgb * (float(noise_std) ** 2) * 3.0
    cov_rgb += np.eye(3, dtype=np.float32) * 1e-6

    idx = np.argwhere(fill_mask)
    if idx.size > 0:
        n = idx.shape[0]
        z = rng.multivariate_normal(
            mean=np.zeros(3, dtype=np.float32),
            cov=cov_rgb.astype(np.float64),
            size=n,
        ).astype(np.float32)
        yy, xx = idx[:, 0], idx[:, 1]
        out[yy, xx] = np.clip(mu[yy, xx] + z, 0.0, 1.0)

    # Жесткий anti-black: поднимаем только абсолютно черные области до минимального floor.
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    out_gray = np.mean(out, axis=-1)
    force = black & (out_gray < float(black_lift_floor))
    if np.any(force):
        scale = np.ones_like(out_gray, dtype=np.float32)
        scale[force] = float(black_lift_floor) / np.clip(out_gray[force], 1e-4, None)
        out = np.clip(out * scale[..., None], 0.0, 1.0)
    return out.astype(np.float32)


def telea_variants(
    warp: np.ndarray,
    cov: np.ndarray,
    radius_px: float,
    contrast_thr: float,
    outlier_thr: float,
    prefill_noise_std: float,
    prefill_black_thr: float,
    prefill_black_lift_floor: float,
    prefill_noise_sweep: tuple[float, ...],
) -> list[dict]:
    """Focused variants around telea_med5_outlier_soft with minimal clutter."""
    base, _ = expand_warp_method(warp, cov, radius_px, "inpaint_telea")
    base_post = telea_postprocess_variants(
        base, cov, contrast_thr=contrast_thr, outlier_thr=outlier_thr
    )
    out = [{"name": "telea_raw", "img": base}]
    for item in base_post:
        if item["name"] == "telea_med5_outlier_soft":
            out.append({"name": "telea_med5_outlier_soft", "img": item["img"]})
            break

    # Эксперименты: один и тот же telea_med5_outlier_soft после prefill с разной силой шума.
    sweep = tuple(float(x) for x in prefill_noise_sweep) if prefill_noise_sweep else (0.0, prefill_noise_std)
    for ns in sweep:
        pref = _prefill_holes_mean_noise(
            base,
            cov,
            noise_std=float(ns),
            black_thr=float(prefill_black_thr),
            black_lift_floor=float(prefill_black_lift_floor),
        )
        pref_post = telea_postprocess_variants(
            pref, cov, contrast_thr=contrast_thr, outlier_thr=outlier_thr
        )
        out.append({"name": f"telea_prefill_n{int(round(ns * 1000)):03d}", "img": pref})
        for item in pref_post:
            if item["name"] == "telea_med5_outlier_soft":
                out.append(
                    {"name": f"telea_med5_outlier_soft_prefill_n{int(round(ns * 1000)):03d}", "img": item["img"]}
                )
                break
    return out


def _hole_mask_with_radius(cov: np.ndarray, radius_px: float) -> np.ndarray:
    valid = (cov > 1e-4).astype(np.uint8)
    if valid.max() == 0:
        return np.ones_like(valid, dtype=np.uint8)
    if valid.min() == 1:
        return np.zeros_like(valid, dtype=np.uint8)
    inv = (valid == 0).astype(np.uint8)
    if radius_px <= 0:
        return inv
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    return ((inv == 1) & (dist <= float(radius_px))).astype(np.uint8)


def expand_warp_method(warp: np.ndarray, cov: np.ndarray, radius_px: float, method: str) -> tuple[np.ndarray, np.ndarray]:
    m = method.strip().lower()
    if m == "nearest":
        return _expand_warp_nearest(warp, cov, radius_px)

    valid = (cov > 1e-4).astype(np.uint8)
    fill_mask = _hole_mask_with_radius(cov, radius_px)
    if fill_mask.max() == 0:
        return warp, cov

    warp_u8 = _f01_u8(warp)
    out_u8 = warp_u8.copy()

    if m == "dilate":
        k = int(max(1, round(float(radius_px)))) * 2 + 1
        ker = np.ones((k, k), np.uint8)
        for c in range(3):
            d = cv2.dilate(warp_u8[..., c], ker, iterations=1)
            out_u8[..., c][fill_mask == 1] = d[fill_mask == 1]
    elif m in ("inpaint_telea", "inpaint_ns"):
        algo = cv2.INPAINT_TELEA if m == "inpaint_telea" else cv2.INPAINT_NS
        inpr = max(1.0, float(radius_px))
        for c in range(3):
            out_u8[..., c] = cv2.inpaint(warp_u8[..., c], fill_mask, inpr, algo)
    else:
        # fallback: nearest
        return _expand_warp_nearest(warp, cov, radius_px)

    cov_out = cov.copy()
    cov_out[fill_mask == 1] = np.maximum(cov_out[fill_mask == 1], 1e-3)
    return out_u8.astype(np.float32) / 255.0, cov_out


def build_unet_tensors(baked_dir: Path, cfg: ConsensusConfig) -> dict:
    """Тензоры как в consensus_training_side / run_test_consensus_inference."""
    meta = json.loads((baked_dir / "meta.json").read_text(encoding="utf-8"))
    sid = meta["sample_id"]
    cam = meta["camera"]
    src = Path(meta.get("source_dir", Path(cfg.dataset_root) / sid))
    warp_dir = Path(cfg.consensus_root) / sid
    hw = (cfg.image_h, cfg.image_w)

    warp_rgb = _chw_npy_to_hwc_u8(warp_dir / cfg.consensus_file)
    coverage = np.load(warp_dir / "coverage.npy").astype(np.float32)
    depth_raw = np.load(baked_dir / "d1.npy").astype(np.float32)
    depth_raw = np.where(np.isfinite(depth_raw), depth_raw, 0.0).astype(np.float32)
    warp_rgb, coverage, depth_raw = _resize_maps(warp_rgb, coverage, depth_raw, hw)

    t0 = _resize_hw(np.array(Image.open(src / "input" / "t0" / f"{cam}.jpg").convert("RGB")), hw, cv2.INTER_LINEAR)
    t1 = _resize_hw(np.array(Image.open(src / "input" / "t1" / f"{cam}.jpg").convert("RGB")), hw, cv2.INTER_LINEAR)
    mean_u8 = ((t0.astype(np.float32) + t1.astype(np.float32)) * 0.5).astype(np.uint8)
    target_rgb = _resize_hw(np.array(Image.open(src / "target" / f"{cam}.jpg").convert("RGB")), hw, cv2.INTER_LINEAR)
    art_mask = _load_ego_mask(cfg, sid, cam, hw)

    # LiDAR — camera image space (как inference v6, без flip)
    lidar_maps = lidar_rife_blend_maps(
        src,
        cam,
        hw,
        spread_radius_fine=cfg.lidar_spread_radius_fine,
        spread_blur_fine=cfg.lidar_spread_blur_fine,
        zone_min=cfg.lidar_zone_min,
    )

    mirrored = cam in MIRROR_CAMERAS
    if mirrored:
        warp_rgb, coverage, depth_raw, art_mask, mean_u8 = _flip_h(
            warp_rgb, coverage, depth_raw, art_mask, mean_u8
        )

    warp = warp_rgb.astype(np.float32) / 255.0
    cov = np.clip(coverage, 0.0, 1.0)
    radius = float(getattr(cfg, "base_warp_expand_px", 0.0))
    selected_method = str(getattr(cfg, "base_expand_method", "nearest"))
    telea_compare = telea_variants(
        warp,
        cov,
        radius,
        float(getattr(cfg, "telea_contrast_thr", 0.14)),
        float(getattr(cfg, "telea_outlier_thr", 0.06)),
        float(getattr(cfg, "telea_prefill_noise_std", 0.012)),
        float(getattr(cfg, "telea_prefill_black_thr", 0.02)),
        float(getattr(cfg, "telea_prefill_black_lift_floor", 0.16)),
        tuple(getattr(cfg, "telea_prefill_noise_sweep", (0.0, 0.008, 0.016, 0.03))),
    )
    if selected_method == FIXED_TELEA_VARIANT:
        sel = next((x for x in telea_compare if x["name"] == FIXED_TELEA_VARIANT), None)
        if sel is None:
            warp_expand, cov_expand = expand_warp_method(warp, cov, radius, "inpaint_telea")
        else:
            warp_expand = np.clip(sel["img"], 0.0, 1.0).astype(np.float32)
            cov_expand = cov.copy()
    else:
        warp_expand, cov_expand = expand_warp_method(warp, cov, radius, selected_method)
    methods = getattr(cfg, "compare_expand_methods", EXPAND_METHODS_DEFAULT)
    expand_compare = []
    for method in methods:
        wm, _ = expand_warp_method(warp, cov, radius, method)
        expand_compare.append({
            "method": method,
            "warp_u8": _f01_u8(wm),
            "delta_u8": _f01_u8(np.abs(wm - warp) * 8.0),
        })
    depth_n = _norm_depth(depth_raw)
    art = np.clip(art_mask, 0.0, 1.0)
    mean_f = mean_u8.astype(np.float32) / 255.0
    warp_expand_pre_mix = warp_expand.copy()
    trust_for_mix = np.clip(lidar_maps["lidar_trust"].astype(np.float32), 0.0, 1.0)
    if mirrored:
        trust_for_mix = trust_for_mix[:, ::-1].copy()
    no_trust = (trust_for_mix <= float(getattr(cfg, "no_trust_thr", 1e-4))).astype(np.float32)
    feather_px = float(getattr(cfg, "no_trust_feather_px", 0.0))
    if feather_px > 0:
        no_trust = cv2.GaussianBlur(no_trust.astype(np.float32), (0, 0), sigmaX=feather_px, sigmaY=feather_px)
        no_trust = np.clip(no_trust, 0.0, 1.0)
    mix_alpha = float(getattr(cfg, "no_trust_mix_alpha", 0.5))
    warp_expand = (
        warp_expand * (1.0 - mix_alpha * no_trust[..., None])
        + mean_f * (mix_alpha * no_trust[..., None])
    )
    # Добавляем артефакт камеры поверх (как в старом варианте): зона ego берётся из mean_t0t1.
    warp_expand = warp_expand * (1.0 - art[..., None]) + mean_f * art[..., None]
    sf_rgb = np.zeros_like(warp)
    sf_mask = np.zeros_like(cov)

    base_init_legacy, _base_mask = build_base_init(
        warp, cov, sf_rgb, sf_mask, art, mean_f, cfg.use_anchor_frames
    )
    base_init, _ = build_base_init(
        warp_expand,
        cov_expand,
        sf_rgb,
        sf_mask,
        art,
        mean_f,
        cfg.use_anchor_frames,
        warp_expand_px=0.0,
        post_blur_sigma=getattr(cfg, "base_post_blur_sigma", 0.0),
    )
    mean_in = mean_f * (1.0 - art[..., None])
    eff_mask = np.ones_like(_base_mask) if cfg.use_anchor_frames else _base_mask

    return {
        "sid": sid,
        "camera": cam,
        "mirrored": mirrored,
        "ref_gt": target_rgb,
        "inp_warp_u8": warp_rgb,
        "inp_warp_expand_u8": _f01_u8(warp_expand_pre_mix),
        "inp_warp_notrust_mix_u8": _f01_u8(warp_expand),
        "inp_warp_expand_delta_u8": _f01_u8(np.abs(warp_expand - warp) * 6.0),
        "inp_coverage": cov,
        "inp_coverage_expand": cov_expand,
        "inp_depth_n": depth_n,
        "inp_ego": art,
        "inp_mean_scene_u8": _f01_u8(mean_in),
        "base_init_legacy_u8": _f01_u8(base_init_legacy),
        "base_init_u8": _f01_u8(base_init),
        "base_init_delta_u8": _f01_u8(np.abs(base_init - base_init_legacy) * 4.0),
        "eff_mask": eff_mask,
        "lidar_density_fine": lidar_maps["density_fine"],
        "lidar_trust": lidar_maps["lidar_trust"],
        "lidar_blend_mask": lidar_maps["blend_mask"],
        "in_channels": 13,
        "static_far_off": not cfg.use_static_far,
        "base_warp_expand_px": float(getattr(cfg, "base_warp_expand_px", 0.0)),
        "base_post_blur_sigma": float(getattr(cfg, "base_post_blur_sigma", 0.0)),
        "base_expand_method": selected_method,
        "no_trust_mix_alpha": mix_alpha,
        "no_trust_thr": float(getattr(cfg, "no_trust_thr", 1e-4)),
        "no_trust_feather_px": feather_px,
        "expand_compare": expand_compare,
        "telea_compare": telea_compare,
    }


def export_sample(view: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "sample_id": view["sid"],
        "camera": view["camera"],
        "mirrored_unet_input": view["mirrored"],
        "unet_in_channels": view["in_channels"],
        "static_far_ch5_8": "zeros (OFF)" if view["static_far_off"] else "active",
        "base_init_legacy": "no warp expansion, no blur",
        "base_init_new": (
            f"warp {view['base_expand_method']} expand {view['base_warp_expand_px']:.1f}px -> compose -> "
            f"gaussian blur sigma={view['base_post_blur_sigma']:.1f}"
        ),
        "no_trust_mix": (
            f"if lidar_trust<= {view['no_trust_thr']:.4f}: "
            f"warp = (1-{view['no_trust_mix_alpha']:.2f})*warp + {view['no_trust_mix_alpha']:.2f}*mean_t0t1 "
            f"(feather σ={view['no_trust_feather_px']:.1f}); then ego area <- mean_t0t1"
        ),
        "warp_expand_methods_compare": [m["method"] for m in view["expand_compare"]],
        "telea_variants_compare": [m["name"] for m in view["telea_compare"]],
        "lidar_blend": "spread r3 + blur σ=1, fine_only, zone_min=0.12",
        "lidar_space": "camera image space (no mirror flip)",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    Image.fromarray(view["inp_warp_u8"]).save(out_dir / "inp_warp.jpg", quality=92)
    Image.fromarray(view["inp_warp_expand_u8"]).save(out_dir / "inp_warp_expand.jpg", quality=92)
    Image.fromarray(view["inp_warp_notrust_mix_u8"]).save(out_dir / "inp_warp_notrust_mix.jpg", quality=92)
    Image.fromarray(view["inp_warp_expand_delta_u8"]).save(out_dir / "inp_warp_expand_delta.jpg", quality=92)
    for m in view["expand_compare"]:
        name = m["method"]
        Image.fromarray(m["warp_u8"]).save(out_dir / f"warp_m_{name}.jpg", quality=92)
        Image.fromarray(m["delta_u8"]).save(out_dir / f"warp_m_{name}_delta.jpg", quality=92)
    warp = view["inp_warp_u8"].astype(np.float32) / 255.0
    for m in view["telea_compare"]:
        name = m["name"]
        img = np.clip(m["img"], 0.0, 1.0)
        Image.fromarray(_f01_u8(img)).save(out_dir / f"telea_v_{name}.jpg", quality=92)
        Image.fromarray(_f01_u8(np.abs(img - warp) * 8.0)).save(out_dir / f"telea_v_{name}_delta.jpg", quality=92)
    Image.fromarray(_scalar_map(view["inp_coverage"])).save(out_dir / "inp_coverage.png")
    Image.fromarray(_scalar_map(view["inp_depth_n"], cv2.COLORMAP_INFERNO)).save(out_dir / "inp_depth.png")
    Image.fromarray(_scalar_map(view["inp_ego"], cv2.COLORMAP_BONE)).save(out_dir / "inp_ego.png")
    Image.fromarray(view["inp_mean_scene_u8"]).save(out_dir / "inp_mean_scene.jpg", quality=92)
    Image.fromarray(view["base_init_legacy_u8"]).save(out_dir / "base_init_legacy.jpg", quality=92)
    Image.fromarray(view["base_init_u8"]).save(out_dir / "base_init.jpg", quality=92)
    Image.fromarray(view["base_init_delta_u8"]).save(out_dir / "base_init_delta.jpg", quality=92)
    Image.fromarray(_scalar_map(view["eff_mask"], cv2.COLORMAP_BONE)).save(out_dir / "eff_mask.png")

    ref = view["ref_gt"]
    Image.fromarray(ref).save(out_dir / "ref_gt.jpg", quality=92)
    fine = view["lidar_density_fine"]
    trust = view["lidar_trust"]
    blend = view["lidar_blend_mask"]
    Image.fromarray(_scalar_map(fine, cv2.COLORMAP_HOT)).save(out_dir / "lidar_density_fine.png")
    Image.fromarray(_scalar_map(trust)).save(out_dir / "lidar_trust.png")
    Image.fromarray(_scalar_map(blend, cv2.COLORMAP_BONE)).save(out_dir / "lidar_blend_mask.png")
    Image.fromarray(_overlay_mask(ref, trust)).save(out_dir / "lidar_overlay.jpg", quality=92)


def write_html(rows: list[dict], out_path: Path, title: str, note: str) -> None:
    parts = [
        "<!doctype html><html lang='ru'><head><meta charset='utf-8'>",
        f"<title>{escape(title)}</title>",
        "<style>",
        "body{font-family:Segoe UI,system-ui,sans-serif;margin:20px;background:#0f1115;color:#e8eaed;}",
        "h1{font-size:22px;margin:0 0 8px;} .note{color:#9aa0a6;font-size:13px;line-height:1.5;max-width:1200px;}",
        "section{border:1px solid #2a2f3a;border-radius:10px;padding:14px;margin:18px 0;background:#171a21;}",
        "h2{font-size:15px;margin:0 0 6px;} h3{font-size:12px;color:#9aa0a6;font-weight:600;margin:12px 0 8px;text-transform:uppercase;letter-spacing:.04em;}",
        ".tags{font-size:12px;color:#8ab4f8;margin-left:8px;}",
        ".grid{display:grid;grid-template-columns:repeat(7,minmax(130px,1fr));gap:8px;}",
        ".grid-methods{display:grid;grid-template-columns:repeat(8,minmax(120px,1fr));gap:8px;}",
        ".cell{text-align:center;} .cell img{width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:4px;border:1px solid #333;}",
        ".lbl{font-size:11px;color:#bdc1c6;margin:4px 0 6px;min-height:32px;}",
        "code{background:#252830;padding:1px 5px;border-radius:3px;}",
        "a{color:#8ab4f8;}",
        "</style></head><body>",
        f"<h1>{escape(title)}</h1>",
        f"<p class='note'>{note}</p>",
    ]
    for r in rows:
        sid = escape(r["sid"])
        cam = escape(r["camera"])
        tag = " <span class='tags'>U-Net: mirror H</span>" if r["mirrored"] else ""
        sub = r["subdir"]
        parts.append(f"<section><h2>{cam} — <code>{sid}</code>{tag}</h2>")
        parts.append("<h3>U-Net — model(inputs, effective_mask, base_init)</h3><div class='grid'>")
        for label, fn in UNET_COLUMNS:
            rel = f"{sub}/{fn}"
            parts.append(
                f"<div class='cell'><div class='lbl'>{escape(label)}</div>"
                f"<a href='{rel}' target='_blank'><img src='{rel}' loading='lazy'></a></div>"
            )
        parts.append("</div>")
        methods = r.get("expand_methods", [])
        parts.append("<h3>Warp expansion compare</h3><div class='grid-methods'>")
        for method in methods:
            m = escape(method)
            rel_w = f"{sub}/warp_m_{method}.jpg"
            rel_d = f"{sub}/warp_m_{method}_delta.jpg"
            parts.append(
                f"<div class='cell'><div class='lbl'>{m}</div>"
                f"<a href='{rel_w}' target='_blank'><img src='{rel_w}' loading='lazy'></a></div>"
            )
            parts.append(
                f"<div class='cell'><div class='lbl'>|delta| {m}</div>"
                f"<a href='{rel_d}' target='_blank'><img src='{rel_d}' loading='lazy'></a></div>"
            )
        parts.append("</div>")
        telea_methods = r.get("telea_methods", [])
        parts.append("<h3>Telea variants (anti-grain, selective)</h3><div class='grid-methods'>")
        for method in telea_methods:
            m = escape(method)
            rel_w = f"{sub}/telea_v_{method}.jpg"
            rel_d = f"{sub}/telea_v_{method}_delta.jpg"
            parts.append(
                f"<div class='cell'><div class='lbl'>{m}</div>"
                f"<a href='{rel_w}' target='_blank'><img src='{rel_w}' loading='lazy'></a></div>"
            )
            parts.append(
                f"<div class='cell'><div class='lbl'>|delta| {m}</div>"
                f"<a href='{rel_d}' target='_blank'><img src='{rel_d}' loading='lazy'></a></div>"
            )
        parts.append("</div>")
        parts.append("<h3>Post-blend LiDAR (camera space, r3+blur)</h3><div class='grid'>")
        for label, fn in BLEND_COLUMNS:
            rel = f"{sub}/{fn}"
            parts.append(
                f"<div class='cell'><div class='lbl'>{escape(label)}</div>"
                f"<a href='{rel}' target='_blank'><img src='{rel}' loading='lazy'></a></div>"
            )
        parts.append("</div></section>")
    parts.append("</body></html>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="HTML: U-Net inputs + LiDAR r3+blur (side train)")
    p.add_argument("--num", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--warp-expand-px", type=float, default=4.0, help="smart warp expansion radius in pixels")
    p.add_argument(
        "--expand-method",
        type=str,
        default=FIXED_TELEA_VARIANT,
        choices=["nearest", "dilate", "inpaint_telea", "inpaint_ns", FIXED_TELEA_VARIANT],
        help="method used for base_init(new)",
    )
    p.add_argument(
        "--expand-methods",
        type=str,
        default="nearest,dilate,inpaint_telea,inpaint_ns",
        help="comma-separated methods for compare block",
    )
    p.add_argument(
        "--telea-contrast-thr",
        type=float,
        default=0.14,
        help="edge contrast threshold for selective Telea cleanup",
    )
    p.add_argument(
        "--telea-outlier-thr",
        type=float,
        default=0.06,
        help="outlier threshold vs local median (anti-grain)",
    )
    p.add_argument(
        "--telea-prefill-noise-std",
        type=float,
        default=0.012,
        help="std of weak noise in hole prefill mean+noise",
    )
    p.add_argument(
        "--telea-prefill-black-thr",
        type=float,
        default=0.02,
        help="absolute-black threshold for forced prefill in [0..1]",
    )
    p.add_argument(
        "--telea-prefill-black-lift-floor",
        type=float,
        default=0.16,
        help="minimum grayscale floor for forced-black prefill in [0..1]",
    )
    p.add_argument(
        "--telea-prefill-noise-sweep",
        type=str,
        default="0,0.008,0.016,0.03",
        help="comma-separated noise std sweep for focused telea_med5_outlier_soft experiments",
    )
    p.add_argument("--base-post-blur-sigma", type=float, default=1.0, help="post blur sigma for base_init")
    p.add_argument("--sample-ids", type=str, default="", help="comma-separated sample_id override")
    args = p.parse_args()

    methods = tuple(
        m.strip() for m in args.expand_methods.split(",")
        if m.strip() in {"nearest", "dilate", "inpaint_telea", "inpaint_ns"}
    )
    if not methods:
        methods = EXPAND_METHODS_DEFAULT
    cfg = side_cfg(args.warp_expand_px, args.base_post_blur_sigma, args.expand_method)
    cfg.compare_expand_methods = methods
    cfg.telea_contrast_thr = float(args.telea_contrast_thr)
    cfg.telea_outlier_thr = float(args.telea_outlier_thr)
    cfg.telea_prefill_noise_std = float(args.telea_prefill_noise_std)
    cfg.telea_prefill_black_thr = float(args.telea_prefill_black_thr)
    cfg.telea_prefill_black_lift_floor = float(args.telea_prefill_black_lift_floor)
    sweep_vals = []
    for tok in args.telea_prefill_noise_sweep.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            sweep_vals.append(float(tok))
        except ValueError:
            continue
    cfg.telea_prefill_noise_sweep = tuple(sweep_vals) if sweep_vals else (0.0, 0.008, 0.016, 0.03)
    all_ready = discover_samples(cfg)
    if not all_ready:
        print("Нет готовых side-сэмплов (bake + warps + target). Проверьте прекомпьют train.")
        return 1

    if args.sample_ids.strip():
        want = {s.strip() for s in args.sample_ids.split(",") if s.strip()}
        baked_dirs = []
        for pth in all_ready:
            meta = json.loads((pth / "meta.json").read_text(encoding="utf-8"))
            if meta["sample_id"] in want:
                baked_dirs.append(pth)
        if len(baked_dirs) < len(want):
            print(f"warning: найдено {len(baked_dirs)}/{len(want)} из --sample-ids")
    else:
        baked_dirs = pick_diverse(all_ready, args.num, args.seed)

    out_root = args.out_dir.resolve()
    if out_root.exists():
        for child in out_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)

    rows = []
    print(f"Side train ready: {len(all_ready)} | gallery: {len(baked_dirs)} samples -> {out_root}")
    for baked_dir in baked_dirs:
        view = build_unet_tensors(baked_dir, cfg)
        sub = f"{view['camera']}_{view['sid'][:48]}"
        export_sample(view, out_root / sub)
        rows.append(
            {
                "sid": view["sid"],
                "camera": view["camera"],
                "mirrored": view["mirrored"],
                "subdir": sub,
                "expand_methods": [m["method"] for m in view["expand_compare"]],
                "telea_methods": [m["name"] for m in view["telea_compare"]],
            }
        )
        print(f"  {view['camera']:10s}  mirror={view['mirrored']}  {view['sid']}")

    note = (
        "<b>U-Net (13 каналов):</b> ch0–2 warp · ch3 coverage · ch4 depth_norm · "
        "ch5–8 static_far <i>(нули, OFF)</i> · ch9 ego · ch10–12 mean×(1−ego). "
        "Плюс отдельно <code>warp expand</code>, <code>base_init legacy</code>, <code>base_init new</code>, "
        "<code>|delta|</code> и <code>effective_mask</code>. "
        f"<b>Base-init new:</b> расширение warp методом <b>{cfg.base_expand_method}</b> "
        f"на <b>{cfg.base_warp_expand_px:.1f}px</b>, потом стандартная сборка и финальный blur "
        f"<b>σ={cfg.base_post_blur_sigma:.1f}</b>. "
        f"Сравнение методов: <code>{', '.join(cfg.compare_expand_methods)}</code>. "
        "Telea anti-grain focused: показываем только <code>telea_med5_outlier_soft</code> "
        "и его варианты после prefill дыр/черных зон. "
        "Noise sweep: "
        f"<code>{', '.join(f'{v:.3f}' for v in cfg.telea_prefill_noise_sweep)}</code>. "
        "Для <code>left_fwd</code>/<code>right_bwd</code> U-Net-инпуты с H-flip.<br>"
        "<b>LiDAR blend</b> (после сети, как test inference): spread r=3 + blur σ=1, fine_only → "
        "<code>density_fine</code>; затем spike + zone_min≥0.12 → <code>lidar_trust</code> и бинарная "
        "<code>blend_mask</code>. LiDAR в <b>camera space</b> (без flip). "
        f"Train side ready: <b>{len(all_ready)}</b> · CV_ROOT: <code>{CV_ROOT}</code>"
    )
    write_html(
        rows,
        out_root / "index.html",
        "Side — U-Net inputs + LiDAR r3+blur",
        note,
    )
    print(f"\nОткройте в браузере:\n  {out_root / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
