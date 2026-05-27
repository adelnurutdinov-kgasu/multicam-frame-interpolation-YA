"""LiDAR point density → зоны доверия в image plane целевой камеры."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from lib.lidar_depth_map import intrinsics_to_K, project_to_camera, rasterize_depth

EPS = 1e-6
ZoneMethod = Literal["spread", "gaussian", "dilate"]
SpreadFalloff = Literal["linear", "hard", "cosine"]


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if cv2 is None or sigma <= 0:
        return img.astype(np.float32, copy=True)
    k = int(max(3, round(sigma * 4)) | 1)
    return cv2.GaussianBlur(img.astype(np.float32), (k, k), sigma)


def _norm01(arr: np.ndarray, peak_percentile: float = 99.0) -> np.ndarray:
    nz = arr[arr > 0]
    if nz.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    peak = float(np.percentile(nz, peak_percentile))
    return np.clip(arr / (peak + EPS), 0.0, 1.0).astype(np.float32)


def _hit_mask(hits: np.ndarray, min_hits_pixel: int) -> np.ndarray:
    return (hits >= float(min_hits_pixel)).astype(np.uint8)


def _spread_trust(
    hit_mask: np.ndarray,
    spread_radius: float,
    falloff: SpreadFalloff = "linear",
) -> np.ndarray:
    """
    Распространение зоны от каждого hit без смешивания интенсивностей.

    Каждый пиксель: trust = f(расстояние до **ближайшего** hit), 0 за пределами radius.
    Не как blur — соседние кластеры не «чернеют» в общую массу.
    """
    if hit_mask.max() == 0 or spread_radius <= 0:
        return np.zeros_like(hit_mask, dtype=np.float32)
    if cv2 is None:
        return hit_mask.astype(np.float32)

    # distanceTransform считает расстояние до ближайшего НУЛЕВОГО пикселя.
    # Нам нужно расстояние до ближайшего hit, поэтому инвертируем маску:
    # hit=1 -> 0, background=0 -> 1.
    inv = (hit_mask == 0).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    r = float(spread_radius)
    if falloff == "hard":
        return (dist <= r).astype(np.float32)
    if falloff == "cosine":
        t = np.clip(dist / r, 0.0, 1.0)
        out = np.zeros_like(dist, dtype=np.float32)
        m = dist <= r
        out[m] = (0.5 * (1.0 + np.cos(np.pi * t[m]))).astype(np.float32)
        return out
    # linear
    return np.clip(1.0 - dist / r, 0.0, 1.0).astype(np.float32)


def _dilate_trust(hit_mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Бинарное расширение (ровная зона без градиента)."""
    if hit_mask.max() == 0 or radius_px <= 0 or cv2 is None:
        return hit_mask.astype(np.float32)
    k = int(max(1, radius_px)) * 2 + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(hit_mask, ker, iterations=1).astype(np.float32)


def _zones_from_hits(
    hits_f: np.ndarray,
    *,
    zone_method: ZoneMethod,
    spread_radius_fine: float,
    spread_radius: float,
    spread_falloff: SpreadFalloff,
    zone_sigma_fine: float,
    zone_sigma: float,
    min_hits_pixel: int,
    preserve_fine: bool,
    peak_percentile: float,
    spread_blur_fine: float = 0.0,
    fine_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает (density_fine, density_zone, lidar_trust)."""
    mask = _hit_mask(hits_f, min_hits_pixel)

    if zone_method == "gaussian":
        density_fine = _gaussian_blur(hits_f, zone_sigma_fine)
        density_zone = _gaussian_blur(hits_f, zone_sigma)
        fine_n = _norm01(density_fine, peak_percentile)
        coarse_n = _norm01(density_zone, peak_percentile)
        trust = np.maximum(fine_n, coarse_n) if preserve_fine else coarse_n
        return density_fine, density_zone, trust

    if zone_method == "dilate":
        density_fine = _dilate_trust(mask, int(round(spread_radius_fine)))
        density_zone = _dilate_trust(mask, int(round(spread_radius)))
        trust = np.maximum(density_fine, density_zone) if preserve_fine else density_zone
        return density_fine, density_zone, trust

    # spread (default) — distance transform + опционально лёгкий blur только на fine
    density_fine = _spread_trust(mask, spread_radius_fine, spread_falloff)
    if spread_blur_fine > 0:
        density_fine = _gaussian_blur(density_fine, spread_blur_fine)
    if fine_only or spread_radius <= 0:
        density_zone = np.zeros_like(density_fine)
        trust = density_fine
    else:
        density_zone = _spread_trust(mask, spread_radius, spread_falloff)
        trust = np.maximum(density_fine, density_zone) if preserve_fine else density_zone
    return density_fine, density_zone, trust.astype(np.float32)


def build_lidar_density(
    sample_dir: Path,
    camera: str,
    timestep: str = "target",
    *,
    splat_radius: int = 2,
    zone_method: ZoneMethod = "spread",
    spread_radius_fine: float = 3.0,
    spread_radius: float = 14.0,
    spread_falloff: SpreadFalloff = "linear",
    spread_blur_fine: float = 1.0,
    fine_only: bool = True,
    zone_sigma: float = 14.0,
    zone_sigma_fine: float = 4.0,
    zone_min: float = 0.12,
    min_hits_pixel: int = 1,
    preserve_fine: bool = True,
    peak_percentile: float = 99.0,
    spike_trust: float = 1.0,
    target_hw: Optional[Tuple[int, int]] = None,
) -> dict:
    """
    Проецирует LiDAR в камеру. Маска доверия по умолчанию — **spread** (не blur).

    spread: trust(p) = f(dist(p, ближайший hit)); лёгкое расширение зоны покрытия.
    spread_blur_fine: σ≈1 px Gaussian **только** на fine-слой после spread (сгладить зубцы).
    gaussian: старый режим (размывает плотность hits).
    dilate: бинарное расширение hit-маски.
    """
    sample_dir = Path(sample_dir)
    meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
    intr = meta["intrinsics"][camera]
    K = intrinsics_to_K(intr)
    W, H = int(intr["width"]), int(intr["height"])
    c2w = np.array(meta["poses_c2w"][timestep][camera], dtype=np.float64)

    empty = {
        "hits": np.zeros((H, W), np.float32),
        "density_fine": np.zeros((H, W), np.float32),
        "density_zone": np.zeros((H, W), np.float32),
        "lidar_trust": np.zeros((H, W), np.float32),
        "depth_lidar": np.full((H, W), np.nan, np.float32),
        "n_projected": 0,
        "camera": camera,
        "timestep": timestep,
        "zone_method": zone_method,
    }

    npz_path = sample_dir / "input" / "lidar.npz"
    if not npz_path.is_file():
        if target_hw and (target_hw[0], target_hw[1]) != (H, W):
            return _resize_lidar_pack(empty, target_hw)
        return empty

    xyz = np.load(npz_path)["xyz"].astype(np.float64)
    u, v, z, _ = project_to_camera(xyz, c2w, K, W, H)
    depth_map, hits = rasterize_depth(u, v, z, H, W, splat_radius=max(0, splat_radius))
    hits_f = hits.astype(np.float32)

    density_fine, density_zone, lidar_trust = _zones_from_hits(
        hits_f,
        zone_method=zone_method,
        spread_radius_fine=spread_radius_fine,
        spread_radius=spread_radius,
        spread_falloff=spread_falloff,
        zone_sigma_fine=zone_sigma_fine,
        zone_sigma=zone_sigma,
        min_hits_pixel=min_hits_pixel,
        preserve_fine=preserve_fine,
        peak_percentile=peak_percentile,
        spread_blur_fine=spread_blur_fine,
        fine_only=fine_only,
    )

    # на самих hits — полный trust (столб = 1.0)
    if spike_trust > 0 and min_hits_pixel > 0:
        spike = _hit_mask(hits_f, min_hits_pixel).astype(np.float32)
        lidar_trust = np.maximum(lidar_trust, spike * float(spike_trust))

    if zone_min > 0:
        lidar_trust = np.where(lidar_trust >= zone_min, lidar_trust, 0.0).astype(np.float32)

    depth_out = depth_map.copy()
    depth_out[~np.isfinite(depth_out) | (depth_out <= 0)] = np.nan

    pack = {
        "hits": hits_f,
        "density_fine": density_fine.astype(np.float32),
        "density_zone": density_zone.astype(np.float32),
        "lidar_trust": lidar_trust.astype(np.float32),
        "depth_lidar": depth_out.astype(np.float32),
        "n_projected": int(len(u)),
        "camera": camera,
        "timestep": timestep,
        "zone_method": zone_method,
        "spread_radius_fine": float(spread_radius_fine),
        "spread_radius": float(spread_radius),
        "spread_blur_fine": float(spread_blur_fine),
    }

    if target_hw is not None and (target_hw[0], target_hw[1]) != (H, W):
        pack = _resize_lidar_pack(pack, target_hw)
    return pack


def _resize_lidar_pack(pack: dict, target_hw: Tuple[int, int]) -> dict:
    th, tw = target_hw
    if cv2 is None:
        raise ImportError("opencv-python нужен для resize lidar maps")
    hits_f = cv2.resize(pack["hits"], (tw, th), interpolation=cv2.INTER_NEAREST)
    density_fine = cv2.resize(pack["density_fine"], (tw, th), interpolation=cv2.INTER_NEAREST)
    density_zone = cv2.resize(pack["density_zone"], (tw, th), interpolation=cv2.INTER_NEAREST)
    lidar_trust = cv2.resize(pack["lidar_trust"], (tw, th), interpolation=cv2.INTER_NEAREST)
    depth_out = pack["depth_lidar"]
    d_valid = np.isfinite(depth_out) & (depth_out > 0)
    depth_out = cv2.resize(
        np.where(d_valid, depth_out, 0).astype(np.float32), (tw, th), interpolation=cv2.INTER_NEAREST
    )
    depth_out[depth_out <= 0] = np.nan
    return {**pack, "hits": hits_f, "density_fine": density_fine, "density_zone": density_zone,
            "lidar_trust": lidar_trust, "depth_lidar": depth_out.astype(np.float32)}


def lidar_rife_blend_maps(
    sample_dir: Path,
    camera: str,
    target_hw: Tuple[int, int],
    *,
    spread_radius_fine: float = 3.0,
    spread_blur_fine: float = 1.0,
    zone_min: float = 0.12,
    mask_thr: Optional[float] = None,
) -> dict:
    """
    Как в lidar_rife_blend.ipynb:
    - density_fine — r3+blur (spread fine + лёгкий blur)
    - lidar_trust — trust после spike + zone_min
    - blend_mask — бинарная зона (lidar_trust >= mask_thr)
    """
    thr = zone_min if mask_thr is None else mask_thr
    lm = build_lidar_density(
        sample_dir,
        camera,
        target_hw=target_hw,
        zone_method="spread",
        spread_radius_fine=spread_radius_fine,
        spread_blur_fine=spread_blur_fine,
        spread_radius=0,
        fine_only=True,
        zone_min=zone_min,
        spike_trust=1.0,
    )
    trust = lm["lidar_trust"]
    return {
        **lm,
        "blend_mask": (trust >= thr).astype(np.float32),
        "mask_thr": float(thr),
    }


def lidar_trust_for_sample(
    sample_dir: Path,
    camera: str,
    target_hw: Tuple[int, int],
    *,
    spread_radius_fine: float = 3.0,
    spread_blur_fine: float = 1.0,
    zone_min: float = 0.12,
    mask_thr: Optional[float] = None,
) -> np.ndarray:
    """Маска r3+blur (fine_only) для одного sample_dir."""
    return lidar_rife_blend_maps(
        sample_dir,
        camera,
        target_hw,
        spread_radius_fine=spread_radius_fine,
        spread_blur_fine=spread_blur_fine,
        zone_min=zone_min,
        mask_thr=mask_thr,
    )["lidar_trust"]


def blur_rgb_soft(rgb: np.ndarray, sigma: float) -> np.ndarray:
    """Лёгкий Gaussian blur на RGB [0..1], HWC."""
    if sigma <= 0:
        return rgb.astype(np.float32, copy=False)
    if rgb.ndim == 2:
        return _gaussian_blur(rgb.astype(np.float32), sigma)
    out = np.empty_like(rgb, dtype=np.float32)
    for c in range(rgb.shape[-1]):
        out[..., c] = _gaussian_blur(rgb[..., c].astype(np.float32), sigma)
    return np.clip(out, 0.0, 1.0)


def _blend_mask_weight(
    lidar_trust: np.ndarray,
    mask_thr: float,
    feather_px: float = 0.0,
    erode_px: float = 0.0,
) -> np.ndarray:
    """Бинарная маска; erode_px — сжать границу внутрь; feather — мягкий ramp от края."""
    w = np.clip(lidar_trust.astype(np.float32), 0.0, 1.0)
    if w.ndim == 3:
        w = w[..., 0]
    bin_m = (w >= mask_thr).astype(np.uint8)
    if bin_m.max() == 0:
        return bin_m.astype(np.float32)
    if cv2 is None:
        return bin_m.astype(np.float32)
    if erode_px > 0:
        dist_in = cv2.distanceTransform(bin_m, cv2.DIST_L2, 3)
        bin_m = (dist_in > float(erode_px)).astype(np.uint8)
        if bin_m.max() == 0:
            return bin_m.astype(np.float32)
    if feather_px <= 0:
        return bin_m.astype(np.float32)
    dist_in = cv2.distanceTransform(bin_m, cv2.DIST_L2, 3)
    return np.clip(dist_in / float(feather_px), 0.0, 1.0).astype(np.float32)


def feather_binary_mask(mask: np.ndarray, feather_px: float = 0.0) -> np.ndarray:
    """Мягкий вес 0..1 внутри бинарной маски, линейный ramp от края внутрь."""
    bin_m = (mask.astype(np.float32) > 0.5).astype(np.uint8)
    if feather_px <= 0 or bin_m.max() == 0:
        return bin_m.astype(np.float32)
    if cv2 is None:
        return bin_m.astype(np.float32)
    dist_in = cv2.distanceTransform(bin_m, cv2.DIST_L2, 3)
    return np.clip(dist_in / float(feather_px), 0.0, 1.0).astype(np.float32)


def blend_ego_mean(
    base: np.ndarray,
    mean_t0t1: np.ndarray,
    art_mask: np.ndarray,
    feather_px: float = 3.0,
    alpha: float = 1.0,
) -> np.ndarray:
    """Подмешать mean(t0,t1) в зонах ego-артефактов с мягким краем."""
    w = feather_binary_mask(art_mask, feather_px) * float(alpha)
    w3 = w[..., None]
    return np.clip((1.0 - w3) * base + w3 * mean_t0t1, 0.0, 1.0).astype(np.float32)


def _lidar_binary_and_dist(
    lidar_trust: np.ndarray,
    mask_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Бинарная LiDAR-маска и расстояние от внешней границы внутрь (px)."""
    w = np.clip(lidar_trust.astype(np.float32), 0.0, 1.0)
    if w.ndim == 3:
        w = w[..., 0]
    bin_m = (w >= mask_thr).astype(np.uint8)
    if bin_m.max() == 0 or cv2 is None:
        return bin_m, np.zeros_like(bin_m, dtype=np.float32)
    dist_in = cv2.distanceTransform(bin_m, cv2.DIST_L2, 3)
    return bin_m, dist_in


def darkest_rgb(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Поканально более тёмный из двух RGB [0..1]."""
    return np.minimum(a.astype(np.float32), b.astype(np.float32))


def blend_model_rife_lidar(
    model: np.ndarray,
    rife: np.ndarray,
    lidar_trust: np.ndarray,
    alpha: float = 0.5,
    mask_thr: float = 0.12,
    model_blur_sigma: float = 0.0,
    rife_blur_sigma: float = 0.0,
    mask_feather_px: float = 0.0,
    mask_erode_px: float = 0.0,
    boundary_dark_px: float = 0.0,
) -> np.ndarray:
    """
    Consensus и RIFE блюрятся отдельно (model_blur_sigma / rife_blur_sigma).

    - За маской LiDAR — только RIFE.
    - Внутри маски, dist ≤ erode — только RIFE (сжатие границы).
    - Полоса границы (erode < dist ≤ boundary_dark_px) — min(consensus, RIFE) поканально.
    - Ядро (dist > boundary_dark_px) — α·consensus + (1−α)·RIFE.
    """
    model_in = blur_rgb_soft(model, model_blur_sigma) if model_blur_sigma > 0 else model.astype(np.float32)
    rife_in = blur_rgb_soft(rife, rife_blur_sigma) if rife_blur_sigma > 0 else rife.astype(np.float32)
    mixed = np.clip(alpha * model_in + (1.0 - alpha) * rife_in, 0.0, 1.0)
    dark = darkest_rgb(model_in, rife_in)

    bin_m, dist_in = _lidar_binary_and_dist(lidar_trust, mask_thr)
    if bin_m.max() == 0:
        return rife_in.astype(np.float32)

    erode = float(mask_erode_px)
    band = float(boundary_dark_px)
    inside = bin_m.astype(bool)
    core = inside & (dist_in > max(erode, band))
    if band > erode:
        boundary = inside & (dist_in > erode) & (dist_in <= band)
    else:
        boundary = np.zeros_like(inside, dtype=bool)

    if mask_feather_px > 0 and cv2 is not None:
        core_w = np.zeros_like(dist_in, dtype=np.float32)
        if core.any():
            core_bin = core.astype(np.uint8)
            d_core = cv2.distanceTransform(core_bin, cv2.DIST_L2, 3)
            core_w = np.clip(d_core / float(mask_feather_px), 0.0, 1.0) * core.astype(np.float32)
        out = rife_in.copy()
        bw = boundary.astype(np.float32)[..., None]
        cw = core_w[..., None]
        out = out * (1.0 - bw) + dark * bw
        out = out * (1.0 - cw) + mixed * cw
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    out = rife_in.copy()
    out[boundary] = dark[boundary]
    out[core] = mixed[core]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def plot_lidar_density_debug(
    maps: dict,
    rgb: Optional[np.ndarray] = None,
    art: Optional[np.ndarray] = None,
    depth_baked: Optional[np.ndarray] = None,
    title: str = "",
):
    import matplotlib.pyplot as plt

    hits = maps["hits"]
    fine = maps.get("density_fine", maps.get("density_zone"))
    zone = maps["density_zone"]
    trust = maps["lidar_trust"]
    zm = maps.get("zone_method", "spread")

    ncols = 6 if depth_baked is not None else 5
    fig, ax = plt.subplots(1, ncols, figsize=(2.8 * ncols, 3.2))
    fig.suptitle(
        (title or f"{maps.get('camera')}  n={maps.get('n_projected')}") + f"  [{zm}]",
        fontsize=10,
    )

    def _show(i, arr, cmap, t, vmin=None, vmax=None):
        ax[i].imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax[i].set_title(t, fontsize=8)
        ax[i].axis("off")

    _show(0, np.log1p(hits), "hot", "hits")
    blur_f = maps.get("spread_blur_fine", 0)
    _show(1, fine, "viridis", f"fine r={maps.get('spread_radius_fine', '?')} blur={blur_f}", 0, 1)
    _show(2, zone, "viridis", f"zone r={maps.get('spread_radius', '?')}", 0, 1)
    _show(3, trust, "viridis", "trust", 0, 1)
    col = 4
    if art is not None:
        _show(col, art, "gray", "ego", 0, 1)
        col += 1
    elif rgb is not None:
        ax[col].imshow(np.clip(rgb, 0, 1) if rgb.max() <= 1.01 else rgb / 255.0)
        ax[col].set_title("RGB", fontsize=8)
        ax[col].axis("off")
        col += 1
    if depth_baked is not None and col < ncols:
        d = depth_baked.copy()
        valid = np.isfinite(d) & (d > 0)
        dn = np.zeros_like(d)
        if valid.any():
            dn[valid] = d[valid] / (d[valid].max() + EPS)
        _show(col, dn, "plasma", "depth baked", 0, 1)

    plt.tight_layout()
    print(f"lidar_trust>0.25: {100 * (trust > 0.25).mean():.1f}%")
    return fig
