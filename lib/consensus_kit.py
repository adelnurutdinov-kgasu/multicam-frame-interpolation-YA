"""Shared inference utilities for consensus U-Net (front/rear training)."""
from __future__ import annotations

# Увеличь при изменении API — в ноутбуке: importlib.reload(consensus_kit)
CONSENSUS_KIT_VERSION = 5

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

try:
    import cv2
except ImportError:
    cv2 = None

EPS = 1e-6
VEHICLE_RE = re.compile(r"_([a-zA-Z]+)_\d+__\d{3}$")


@dataclass
class ConsensusConfig:
    dataset_root: str
    consensus_root: str
    baked_root: str
    rife_root: str
    ego_masks_root: str
    consensus_file: str = "consensus_raw.npy"
    use_static_far: bool = False
    static_far_root: str = ""
    use_artifact_mask: bool = True
    use_anchor_frames: bool = True
    allowed_cameras: Tuple[str, ...] = ("front", "rear")
    image_h: int = 544
    image_w: int = 1024
    in_channels: int = 13
    base_channels: int = 32
    seed: int = 42
    val_fraction: float = 0.16
    max_samples: int = 0
    # LiDAR density zones (маска доверия вместо depth_valid)
    use_lidar_trust: bool = True
    lidar_timestep: str = "target"
    lidar_splat_radius: int = 2
    lidar_zone_method: str = "spread"
    lidar_spread_radius_fine: float = 3.0
    lidar_spread_radius: float = 0.0
    lidar_spread_falloff: str = "linear"
    lidar_spread_blur_fine: float = 1.0
    lidar_fine_only: bool = True
    lidar_zone_sigma: float = 14.0
    lidar_zone_sigma_fine: float = 4.0
    lidar_zone_min: float = 0.12
    lidar_min_hits_pixel: int = 1
    lidar_preserve_fine: bool = True
    lidar_peak_percentile: float = 99.0
    lidar_spike_trust: float = 1.0


# --- model ---


class PartialConv2d(nn.Conv2d):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__(in_ch, out_ch, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        ker = self.kernel_size
        self.register_buffer("mask_kernel", torch.ones(1, 1, ker[0], ker[1]))
        self.slide_winsize = float(ker[0] * ker[1])

    def forward(self, x, mask):
        with torch.no_grad():
            updated_mask = F.conv2d(
                mask, self.mask_kernel, bias=None,
                stride=self.stride, padding=self.padding, dilation=self.dilation,
            )
            mask_ratio = self.slide_winsize / (updated_mask + 1e-8)
            updated_mask = torch.clamp(updated_mask, 0.0, 1.0)
            mask_ratio = mask_ratio * updated_mask
        x_masked = x * mask
        out = F.conv2d(x_masked, self.weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation)
        if self.bias is not None:
            bias = self.bias.view(1, -1, 1, 1)
            out = (out - bias) * mask_ratio + bias
            out = out * updated_mask
        else:
            out = out * mask_ratio
        return out, updated_mask


class PartialDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pc1 = PartialConv2d(in_ch, out_ch, 3, padding=1)
        self.pc2 = PartialConv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x, mask):
        x, mask = self.pc1(x, mask)
        x = self.act(x)
        x, mask = self.pc2(x, mask)
        x = self.act(x)
        return x, mask


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ConsensusUNet(nn.Module):
    def __init__(self, in_ch=13, base=32, predict_confidence=True):
        super().__init__()
        self.predict_confidence = predict_confidence
        self.penc1 = PartialDoubleConv(in_ch, base)
        self.penc2 = PartialDoubleConv(base, base * 2)
        self.penc3 = PartialDoubleConv(base * 2, base * 4)
        self.penc4 = PartialDoubleConv(base * 4, base * 8)
        self.bottleneck = ConvBlock(base * 8, base * 8)
        self.dec4 = ConvBlock(base * 8 + base * 8 + 1, base * 4)
        self.dec3 = ConvBlock(base * 4 + base * 4 + 1, base * 2)
        self.dec2 = ConvBlock(base * 2 + base * 2 + 1, base)
        self.dec1 = ConvBlock(base + base + 1, base)
        self.out_residual = nn.Conv2d(base, 3, 3, padding=1)
        nn.init.zeros_(self.out_residual.weight)
        nn.init.zeros_(self.out_residual.bias)
        if predict_confidence:
            self.out_conf = nn.Conv2d(base, 1, 3, padding=1)
            nn.init.zeros_(self.out_conf.weight)
            nn.init.zeros_(self.out_conf.bias)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def _pool_with_mask(self, x, m):
        return F.avg_pool2d(x, 2), F.max_pool2d(m, 2)

    def forward(self, x, eff_mask, base_init):
        e1, m1 = self.penc1(x, eff_mask)
        x2, m2 = self._pool_with_mask(e1, m1)
        e2, m2 = self.penc2(x2, m2)
        x3, m3 = self._pool_with_mask(e2, m2)
        e3, m3 = self.penc3(x3, m3)
        x4, m4 = self._pool_with_mask(e3, m3)
        e4, m4 = self.penc4(x4, m4)
        xb, mb = self._pool_with_mask(e4, m4)
        b = self.bottleneck(xb)
        d4 = self.dec4(torch.cat([self.up(b), e4, m4], dim=1))
        d3 = self.dec3(torch.cat([self.up(d4), e3, m3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2, m2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1, m1], dim=1))
        residual = self.out_residual(d1)
        pred = (base_init + residual).clamp(0.0, 1.0)
        out = {"pred": pred, "residual": residual}
        if self.predict_confidence:
            out["confidence"] = torch.sigmoid(self.out_conf(d1))
        return out


def load_consensus_model(ckpt_path: Path, device: torch.device, use_ema: bool = True, cfg: Optional[ConsensusConfig] = None):
    cfg = cfg or default_config()
    model = ConsensusUNet(in_ch=cfg.in_channels, base=cfg.base_channels)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("ema") if use_ema and ckpt.get("ema") else ckpt["model"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model, ckpt


def default_config() -> ConsensusConfig:
    from ya_paths import (
        BAKED_ROOT,
        DATASET_ROOT,
        EGO_MASKS_APPROVED,
        RIFE_ROOT,
        STATIC_FAR_ROOT,
        TRAIN_SPLIT,
        WARPS_ROOT,
    )

    return ConsensusConfig(
        dataset_root=str(TRAIN_SPLIT),
        consensus_root=str(WARPS_ROOT / "train"),
        baked_root=str(BAKED_ROOT / "train"),
        rife_root=str(RIFE_ROOT / "train"),
        ego_masks_root=str(EGO_MASKS_APPROVED),
        static_far_root=str(STATIC_FAR_ROOT),
    )


# --- metrics ---


def psnr_uint8_np(pred: np.ndarray, gt: np.ndarray) -> float:
    p = (np.clip(pred, 0, 1) * 255.0).round().astype(np.uint8)
    g = (np.clip(gt, 0, 1) * 255.0).round().astype(np.uint8)
    mse = np.mean((p.astype(np.float64) - g.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 99.0
    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def psnr_uint8_torch(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred_u8 = (pred.clamp(0, 1) * 255.0).round()
    gt_u8 = (gt.clamp(0, 1) * 255.0).round()
    mse = ((pred_u8 - gt_u8) ** 2).mean().item()
    if mse < 1e-10:
        return 99.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def tensor_to_hwc_rgb(t) -> np.ndarray:
    """Tensor CHW (3,H,W), NCHW (1,3,H,W) или numpy → float32 HWC [0..1]."""
    if isinstance(t, np.ndarray):
        if t.ndim == 3 and t.shape[-1] == 3:
            out = t.astype(np.float32)
        elif t.ndim == 3 and t.shape[0] == 3:
            out = np.transpose(t, (1, 2, 0)).astype(np.float32)
        else:
            raise ValueError(f"numpy rgb shape {t.shape}")
        return out / 255.0 if out.max() > 1.01 else out
    x = t.detach().cpu()
    if x.dim() == 4:
        x = x[0]
    if x.dim() == 3 and x.shape[0] in (1, 3):
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        return x.permute(1, 2, 0).numpy().astype(np.float32)
    raise ValueError(f"ожидали CHW rgb, получили shape {tuple(t.shape)}")


# --- data ---


def _parse_vehicle(sample_id: str) -> Optional[str]:
    m = VEHICLE_RE.search(sample_id)
    return m.group(1) if m else None


def _resize_hw(img: np.ndarray, target_hw, interp) -> np.ndarray:
    H, W = target_hw
    if img.shape[:2] == (H, W):
        return img
    if cv2 is None:
        raise ImportError("opencv-python required")
    return cv2.resize(img, (W, H), interpolation=interp)


def _chw_npy_to_hwc_u8(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.clip(arr.transpose(1, 2, 0), 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def _resize_maps(warp_rgb, coverage, depth, target_hw):
    warp_rgb = _resize_hw(warp_rgb, target_hw, cv2.INTER_LINEAR)
    coverage = _resize_hw(coverage, target_hw, cv2.INTER_LINEAR)
    depth = _resize_hw(depth, target_hw, cv2.INTER_NEAREST)
    return warp_rgb, coverage, depth


def _load_ego_mask(cfg: ConsensusConfig, sample_id: str, camera: str, target_hw) -> np.ndarray:
    H, W = target_hw
    if not cfg.use_artifact_mask:
        return np.zeros((H, W), dtype=np.float32)
    vehicle = _parse_vehicle(sample_id)
    if not vehicle:
        return np.zeros((H, W), dtype=np.float32)
    path = Path(cfg.ego_masks_root) / f"{vehicle}_{camera}.png"
    if not path.is_file():
        return np.zeros((H, W), dtype=np.float32)
    m = np.array(Image.open(path).convert("L"))
    m = _resize_hw(m, target_hw, cv2.INTER_NEAREST)
    return (m > 127).astype(np.float32)


def _norm_depth(d: np.ndarray) -> np.ndarray:
    d = d.astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if valid.sum() == 0:
        return np.zeros_like(d, dtype=np.float32)
    dmax = float(d[valid].max())
    out = np.zeros_like(d, dtype=np.float32)
    out[valid] = d[valid] / (dmax + EPS)
    return out


def _blur_rgb01(rgb: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or cv2 is None:
        return rgb.astype(np.float32, copy=False)
    out = np.empty_like(rgb, dtype=np.float32)
    for c in range(rgb.shape[-1]):
        out[..., c] = cv2.GaussianBlur(
            rgb[..., c].astype(np.float32), (0, 0), sigmaX=float(sigma), sigmaY=float(sigma)
        )
    return np.clip(out, 0.0, 1.0)


def _edge_contrast_mask(rgb01: np.ndarray, thr: float = 0.14) -> np.ndarray:
    if cv2 is None:
        return np.zeros(rgb01.shape[:2], dtype=np.float32)
    gray = cv2.cvtColor((np.clip(rgb01, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(
        np.float32
    ) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return (mag >= float(thr)).astype(np.float32)


def _local_median_rgb01(rgb01: np.ndarray, ksize: int = 3) -> np.ndarray:
    if cv2 is None:
        return rgb01.astype(np.float32, copy=True)
    u8 = (np.clip(rgb01, 0.0, 1.0) * 255.0).astype(np.uint8)
    out = np.empty_like(u8)
    k = int(ksize) | 1
    for c in range(3):
        out[..., c] = cv2.medianBlur(u8[..., c], k)
    return out.astype(np.float32) / 255.0


def selective_median_outlier_fix(
    base: np.ndarray,
    med: np.ndarray,
    region: np.ndarray,
    *,
    outlier_thr: float = 0.06,
    min_med_gray: float = 0.07,
    never_darken: bool = True,
    cov: Optional[np.ndarray] = None,
    min_cov: float = 0.12,
) -> np.ndarray:
    """
    Сглаживание зернистости: правим только выбросы относительно local median.
    never_darken=True — не затемняем пиксель (чёрные прострелы не раздуваются).
    """
    base_g = np.mean(base, axis=-1)
    med_g = np.mean(med, axis=-1)
    diff = base_g - med_g
    bright_out = diff > float(outlier_thr)
    # тёмный выброс только если фон локально не чёрная дыра
    dark_out = (diff < -float(outlier_thr)) & (med_g > float(min_med_gray))
    apply = (bright_out | dark_out).astype(np.float32)
    if cov is not None:
        apply *= (np.clip(cov, 0.0, 1.0) > float(min_cov)).astype(np.float32)
    if region.ndim == 3:
        region = region[..., 0]
    apply = np.clip(apply * np.clip(region, 0.0, 1.0), 0.0, 1.0)
    apply3 = apply[..., None]
    out = base * (1.0 - apply3) + med * apply3
    if never_darken:
        out = np.maximum(out, base)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def telea_postprocess_variants(
    base: np.ndarray,
    cov: np.ndarray,
    *,
    contrast_thr: float = 0.12,
    outlier_thr: float = 0.06,
) -> List[dict]:
    """Варианты anti-grain поверх inpaint Telea (для визуального выбора)."""
    edge = _edge_contrast_mask(base, contrast_thr)
    valid = (cov > 1e-4).astype(np.float32)
    # только края на валидном warp — без заливки дыр median'ом
    region_edge = np.clip(0.85 * edge * valid, 0.0, 1.0)
    if cv2 is not None:
        region_edge = cv2.GaussianBlur(region_edge.astype(np.float32), (0, 0), sigmaX=0.8, sigmaY=0.8)
    region_edge = np.clip(region_edge, 0.0, 1.0)

    med5 = _local_median_rgb01(base, 5)
    med5_full_blend = base * (1.0 - region_edge[..., None]) + med5 * region_edge[..., None]
    med5_outlier = selective_median_outlier_fix(
        base, med5, region_edge, outlier_thr=outlier_thr, never_darken=True, cov=cov
    )
    # чуть мягче порог — меньше агрессии на границах теней
    med5_outlier_soft = selective_median_outlier_fix(
        base,
        med5,
        region_edge,
        outlier_thr=outlier_thr * 1.35,
        never_darken=True,
        cov=cov,
        min_cov=0.08,
    )
    return [
        {"name": "telea_raw", "img": base},
        {"name": "telea_med5_sel_full", "img": med5_full_blend},
        {"name": "telea_med5_sel", "img": med5_outlier},
        {"name": "telea_med5_outlier_soft", "img": med5_outlier_soft},
    ]


def _expand_warp_nearest(warp: np.ndarray, cov: np.ndarray, radius_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Expand warp into nearby uncovered pixels via nearest valid sample (not blur)."""
    r = float(radius_px)
    if r <= 0 or cv2 is None:
        return warp, cov
    valid = (cov > 1e-4).astype(np.uint8)
    if valid.max() == 0 or valid.min() == 1:
        return warp, cov
    inv = (valid == 0).astype(np.uint8)
    dist, labels = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    fill = (inv == 1) & (dist <= r)
    if not np.any(fill):
        return warp, cov

    h, w = cov.shape
    label_ids = labels[fill].astype(np.int64) - 1
    ys = np.clip(label_ids // w, 0, h - 1)
    xs = np.clip(label_ids % w, 0, w - 1)

    warp_out = warp.copy()
    cov_out = cov.copy()
    warp_out[fill] = warp[ys, xs]
    cov_out[fill] = cov[ys, xs]
    return warp_out, cov_out


def build_base_init(
    warp,
    cov,
    sf_rgb,
    sf_mask,
    art_mask,
    mean_t0t1,
    use_anchor: bool,
    warp_expand_px: float = 0.0,
    post_blur_sigma: float = 0.0,
):
    warp_smart, cov_smart = _expand_warp_nearest(warp, cov, warp_expand_px)
    warp_w = cov_smart[..., None]
    sf_w = sf_mask[..., None] * (1.0 - warp_w)
    base_init = warp_smart * warp_w + sf_rgb * sf_w
    base_mask = np.clip(cov_smart + sf_mask * (1.0 - cov_smart), 0.0, 1.0)
    base_init = base_init * (1.0 - art_mask[..., None])
    base_mask = base_mask * (1.0 - art_mask)
    if use_anchor:
        hole = (1.0 - base_mask)[..., None]
        mean_scene = mean_t0t1 * (1.0 - art_mask[..., None])
        mean_art = mean_t0t1 * art_mask[..., None]
        base_init = base_init + mean_scene * hole + mean_art * art_mask[..., None]
    if post_blur_sigma > 0:
        base_init = _blur_rgb01(base_init, post_blur_sigma)
    return base_init, base_mask


class ConsensusDataset(Dataset):
    def __init__(self, sample_dirs: List[Path], cfg: ConsensusConfig, augment: bool = False):
        self.dirs = [Path(p) for p in sample_dirs]
        self.cfg = cfg
        self.augment = augment

    def __len__(self):
        return len(self.dirs)

    def _load_raw(self, baked_dir: Path) -> dict:
        meta = json.loads((baked_dir / "meta.json").read_text(encoding="utf-8"))
        sid = meta["sample_id"]
        cam = meta["camera"]
        src = Path(meta.get("source_dir", Path(self.cfg.dataset_root) / sid))
        warp_dir = Path(self.cfg.consensus_root) / sid
        target_hw = (self.cfg.image_h, self.cfg.image_w)

        warp_rgb = _chw_npy_to_hwc_u8(warp_dir / self.cfg.consensus_file)
        coverage = np.load(warp_dir / "coverage.npy").astype(np.float32)
        depth_raw = np.load(baked_dir / "d1.npy").astype(np.float32)
        depth_valid = (np.isfinite(depth_raw) & (depth_raw > 0)).astype(np.float32)
        depth = np.where(np.isfinite(depth_raw), depth_raw, 0.0).astype(np.float32)
        target_rgb = np.array(Image.open(src / "target" / f"{cam}.jpg").convert("RGB"))
        warp_rgb, coverage, depth = _resize_maps(warp_rgb, coverage, depth, target_hw)
        target_rgb = _resize_hw(target_rgb, target_hw, cv2.INTER_LINEAR)

        t0 = _resize_hw(np.array(Image.open(src / "input" / "t0" / f"{cam}.jpg").convert("RGB")), target_hw, cv2.INTER_LINEAR)
        t1 = _resize_hw(np.array(Image.open(src / "input" / "t1" / f"{cam}.jpg").convert("RGB")), target_hw, cv2.INTER_LINEAR)
        mean_t0t1_rgb = ((t0.astype(np.float32) + t1.astype(np.float32)) * 0.5).astype(np.uint8)

        H, W = target_hw
        sf_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        sf_mask = np.zeros((H, W), dtype=np.float32)
        art_mask = _load_ego_mask(self.cfg, sid, cam, target_hw)

        rife_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        rife_path = Path(self.cfg.rife_root) / f"{sid}.jpg"
        if rife_path.is_file():
            rife_rgb = _resize_hw(np.array(Image.open(rife_path).convert("RGB")), target_hw, cv2.INTER_LINEAR)

        lidar_trust = np.zeros((H, W), dtype=np.float32)
        if getattr(self.cfg, "use_lidar_trust", True):
            from lib.lidar_density_mask import build_lidar_density

            lm = build_lidar_density(
                src,
                cam,
                timestep=getattr(self.cfg, "lidar_timestep", "target"),
                splat_radius=getattr(self.cfg, "lidar_splat_radius", 2),
                zone_method=getattr(self.cfg, "lidar_zone_method", "spread"),
                spread_radius_fine=getattr(self.cfg, "lidar_spread_radius_fine", 3.0),
                spread_radius=getattr(self.cfg, "lidar_spread_radius", 0.0),
                spread_falloff=getattr(self.cfg, "lidar_spread_falloff", "linear"),
                spread_blur_fine=getattr(self.cfg, "lidar_spread_blur_fine", 1.0),
                fine_only=getattr(self.cfg, "lidar_fine_only", True),
                zone_sigma=getattr(self.cfg, "lidar_zone_sigma", 14.0),
                zone_sigma_fine=getattr(self.cfg, "lidar_zone_sigma_fine", 4.0),
                zone_min=getattr(self.cfg, "lidar_zone_min", 0.12),
                min_hits_pixel=getattr(self.cfg, "lidar_min_hits_pixel", 1),
                preserve_fine=getattr(self.cfg, "lidar_preserve_fine", True),
                peak_percentile=getattr(self.cfg, "lidar_peak_percentile", 99.0),
                spike_trust=getattr(self.cfg, "lidar_spike_trust", 1.0),
                target_hw=target_hw,
            )
            lidar_trust = lm["lidar_trust"]

        return {
            "meta": meta,
            "baked_dir": baked_dir,
            "warp_rgb": warp_rgb,
            "coverage": coverage,
            "depth": depth,
            "depth_valid": depth_valid,
            "lidar_trust": lidar_trust,
            "static_far_rgb": sf_rgb,
            "static_far_mask": sf_mask,
            "artifact_mask": art_mask,
            "mean_t0t1_rgb": mean_t0t1_rgb,
            "rife_rgb": rife_rgb,
            "target_rgb": target_rgb,
        }

    def __getitem__(self, idx):
        s = self._load_raw(self.dirs[idx])
        warp = s["warp_rgb"].astype(np.float32) / 255.0
        cov = np.clip(s["coverage"], 0.0, 1.0)
        depth_n = _norm_depth(s["depth"])
        depth_valid = s["depth_valid"]
        lidar_trust = s["lidar_trust"]
        sf_rgb = s["static_far_rgb"].astype(np.float32) / 255.0
        sf_mask = np.clip(s["static_far_mask"], 0.0, 1.0)
        art_mask = np.clip(s["artifact_mask"], 0.0, 1.0)
        mean_t0t1 = s["mean_t0t1_rgb"].astype(np.float32) / 255.0
        rife = s["rife_rgb"].astype(np.float32) / 255.0
        target = s["target_rgb"].astype(np.float32) / 255.0

        base_init, base_mask = build_base_init(
            warp,
            cov,
            sf_rgb,
            sf_mask,
            art_mask,
            mean_t0t1,
            self.cfg.use_anchor_frames,
            warp_expand_px=getattr(self.cfg, "base_warp_expand_px", 0.0),
            post_blur_sigma=getattr(self.cfg, "base_post_blur_sigma", 0.0),
        )
        mean_in = mean_t0t1 * (1.0 - art_mask[..., None])
        eff_mask = np.ones_like(base_mask) if self.cfg.use_anchor_frames else base_mask

        to_chw_3 = lambda a: torch.from_numpy(a.transpose(2, 0, 1)).contiguous().float()
        to_chw_1 = lambda a: torch.from_numpy(a)[None].contiguous().float()
        parts = [
            to_chw_3(warp), to_chw_1(cov), to_chw_1(depth_n),
            to_chw_3(sf_rgb), to_chw_1(sf_mask), to_chw_1(art_mask),
        ]
        if self.cfg.use_anchor_frames:
            parts.append(to_chw_3(mean_in))
        inputs = torch.cat(parts, dim=0)

        return {
            "inputs": inputs,
            "effective_mask": to_chw_1(eff_mask),
            "base_init": to_chw_3(base_init),
            "target": to_chw_3(target),
            "rife_rgb": to_chw_3(rife),
            "coverage": to_chw_1(cov),
            "depth": to_chw_1(depth_n),
            "depth_valid": to_chw_1(depth_valid),
            "lidar_trust": to_chw_1(lidar_trust),
            "art": to_chw_1(art_mask),
            "meta": s["meta"],
        }


def discover_samples(cfg: ConsensusConfig) -> List[Path]:
    baked_root = Path(cfg.baked_root)
    consensus_root = Path(cfg.consensus_root)
    dataset_root = Path(cfg.dataset_root)
    out: List[Path] = []
    for baked_dir in sorted(baked_root.iterdir()):
        if not baked_dir.is_dir() or not (baked_dir / "meta.json").is_file():
            continue
        meta = json.loads((baked_dir / "meta.json").read_text(encoding="utf-8"))
        cam = meta["camera"]
        if cfg.allowed_cameras and cam not in cfg.allowed_cameras:
            continue
        sid = meta["sample_id"]
        src = Path(meta.get("source_dir", dataset_root / sid))
        warp_dir = consensus_root / sid
        ok = (
            (warp_dir / cfg.consensus_file).is_file()
            and (warp_dir / "coverage.npy").is_file()
            and (baked_dir / "d1.npy").is_file()
            and (src / "target" / f"{cam}.jpg").is_file()
        )
        if ok:
            out.append(baked_dir)
    if cfg.max_samples > 0 and len(out) > cfg.max_samples:
        rng = np.random.RandomState(cfg.seed)
        idx = rng.permutation(len(out))[: cfg.max_samples]
        out = [out[i] for i in idx]
    return out


def make_split(all_samples: List[Path], val_fraction: float, seed: int = 42):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(all_samples))
    rng.shuffle(idx)
    n_val = max(1, int(len(all_samples) * val_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return [all_samples[i] for i in train_idx], [all_samples[i] for i in val_idx]


@torch.no_grad()
def run_inference_cache(
    model: nn.Module,
    sample_dirs: List[Path],
    cfg: ConsensusConfig,
    device: torch.device,
    use_amp: bool = True,
    amp_dtype=torch.bfloat16,
) -> List[dict]:
    ds = ConsensusDataset(sample_dirs, cfg, augment=False)
    out_list = []
    for i in range(len(ds)):
        entry = _infer_batch_entry(model, ds[i], device, use_amp, amp_dtype)
        out_list.append(entry)
        if (i + 1) % 20 == 0:
            print(f"  inferred {i + 1}/{len(ds)}", flush=True)
    return out_list


VAL_CACHE_RGB_KEYS = ("model", "rife", "gt", "base")
VAL_CACHE_MAP_KEYS = ("cov", "depth", "lidar_trust", "art")
VAL_CACHE_DEFAULT_KEYS = VAL_CACHE_RGB_KEYS + VAL_CACHE_MAP_KEYS


class ValCacheStacked:
    """Ленивый доступ к npz-кэшу без list[dict] на весь val."""

    def __init__(self, z: np.lib.npyio.NpzFile, promote_rgb: bool = True):
        self._z = z
        self._promote_rgb = promote_rgb
        self._metas = [json.loads(str(x)) for x in z["meta_json"]]
        self._keys = [k for k in z.files if k != "meta_json"]

    def __len__(self) -> int:
        return int(self._z["model"].shape[0])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, i: int) -> dict:
        e = {k: self._z[k][i] for k in self._keys}
        if self._promote_rgb:
            for k in VAL_CACHE_RGB_KEYS:
                if k in e and e[k].dtype == np.float16:
                    e[k] = e[k].astype(np.float32, copy=False)
        e["meta"] = self._metas[i]
        return e


def _infer_batch_entry(model, batch, device, use_amp, amp_dtype) -> dict:
    inputs = batch["inputs"].unsqueeze(0).to(device)
    eff_mask = batch["effective_mask"].unsqueeze(0).to(device)
    base = batch["base_init"].unsqueeze(0).to(device)
    gt = batch["target"].unsqueeze(0).to(device)
    rife = batch["rife_rgb"].unsqueeze(0).to(device)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        o = model(inputs, eff_mask, base)
    pred = o["pred"].float()
    conf = o.get("confidence")
    if conf is not None:
        conf = conf.float()

    def chw_to_hwc(t):
        return t[0].detach().cpu().permute(1, 2, 0).numpy()

    entry = {
        "meta": batch["meta"],
        "model": chw_to_hwc(pred),
        "rife": chw_to_hwc(rife),
        "gt": chw_to_hwc(gt),
        "cov": batch["coverage"][0].numpy(),
        "depth": batch["depth"][0].numpy(),
        "lidar_trust": batch["lidar_trust"][0].numpy(),
        "art": batch["art"][0].numpy(),
        "base": chw_to_hwc(base),
    }
    if conf is not None:
        entry["conf"] = conf[0, 0].cpu().numpy()
    return entry


def _cast_cache_array(key: str, arr: np.ndarray, dtype) -> np.ndarray:
    if arr is None:
        return arr
    if key in VAL_CACHE_RGB_KEYS:
        return arr.astype(dtype, copy=False)
    if arr.dtype == np.float32:
        return arr.astype(dtype, copy=False)
    return arr


@torch.no_grad()
def infer_and_save_val_cache(
    model: nn.Module,
    sample_dirs: List[Path],
    cfg: ConsensusConfig,
    device: torch.device,
    path: Path | str,
    *,
    keys: Optional[Tuple[str, ...]] = None,
    compress: bool = False,
    cache_dtype=np.float16,
    use_amp: bool = True,
    amp_dtype=torch.bfloat16,
) -> ValCacheStacked:
    """
    Inference val + сохранение на диск без np.stack и без gzip (по умолчанию).

    compress=False — в разы быстрее savez_compressed на ~70×544×1024.
    cache_dtype=float16 — ~2× меньше файл, PSNR почти без изменений.
    """
    import time

    keys = tuple(keys or VAL_CACHE_DEFAULT_KEYS)
    path = Path(path).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    ds = ConsensusDataset(sample_dirs, cfg, augment=False)
    n = len(ds)
    if n == 0:
        raise ValueError("infer_and_save_val_cache: пустой sample_dirs")

    t0 = time.perf_counter()
    e0 = _infer_batch_entry(model, ds[0], device, use_amp, amp_dtype)
    stacks: dict[str, np.ndarray] = {}
    for k in keys:
        if k not in e0:
            continue
        v = _cast_cache_array(k, e0[k], cache_dtype)
        if k in VAL_CACHE_RGB_KEYS:
            stacks[k] = np.empty((n,) + v.shape, dtype=cache_dtype)
            stacks[k][0] = v
        else:
            stacks[k] = np.empty((n,) + v.shape, dtype=cache_dtype)
            stacks[k][0] = v

    has_conf = "conf" in e0
    if has_conf:
        stacks["conf"] = np.empty((n,) + e0["conf"].shape, dtype=cache_dtype)
        stacks["conf"][0] = _cast_cache_array("conf", e0["conf"], cache_dtype)

    metas = [json.dumps(e0["meta"], ensure_ascii=False)]

    for i in range(1, n):
        e = _infer_batch_entry(model, ds[i], device, use_amp, amp_dtype)
        for k in stacks:
            stacks[k][i] = _cast_cache_array(k, e[k], cache_dtype)
        metas.append(json.dumps(e["meta"], ensure_ascii=False))
        if (i + 1) % 10 == 0 or i + 1 == n:
            dt = time.perf_counter() - t0
            print(f"  inferred {i + 1}/{n}  ({dt:.1f}s)", flush=True)

    meta_arr = np.array(metas, dtype=object)
    t_save = time.perf_counter()
    if compress:
        np.savez_compressed(path, **stacks, meta_json=meta_arr)
    else:
        np.savez(path, **stacks, meta_json=meta_arr)
    mb = path.stat().st_size / (1024 * 1024)
    print(
        f"saved cache → {path}  ({mb:.1f} MB, {'zlib' if compress else 'raw npz'}, "
        f"dtype={cache_dtype})  write {time.perf_counter() - t_save:.1f}s  total {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return load_val_cache(path)


def save_val_cache(
    cache: List[dict],
    path: Path | str,
    *,
    keys: Optional[Tuple[str, ...]] = None,
    compress: bool = False,
    cache_dtype=np.float16,
) -> ValCacheStacked:
    """Сохранить list[dict] из run_inference_cache (без повторного inference)."""
    import time

    if not cache:
        raise ValueError("save_val_cache: пустой cache")
    keys = tuple(k for k in (keys or VAL_CACHE_DEFAULT_KEYS) if k in cache[0])
    path = Path(path).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(cache)
    t0 = time.perf_counter()

    stacks = {}
    for k in keys:
        v0 = _cast_cache_array(k, cache[0][k], cache_dtype)
        stacks[k] = np.empty((n,) + v0.shape, dtype=cache_dtype)
        stacks[k][0] = v0
        for i in range(1, n):
            stacks[k][i] = _cast_cache_array(k, cache[i][k], cache_dtype)

    if "conf" in cache[0]:
        v0 = _cast_cache_array("conf", cache[0]["conf"], cache_dtype)
        stacks["conf"] = np.empty((n,) + v0.shape, dtype=cache_dtype)
        for i in range(n):
            stacks["conf"][i] = _cast_cache_array("conf", cache[i]["conf"], cache_dtype)

    meta_arr = np.array([json.dumps(c["meta"], ensure_ascii=False) for c in cache], dtype=object)
    if compress:
        np.savez_compressed(path, **stacks, meta_json=meta_arr)
    else:
        np.savez(path, **stacks, meta_json=meta_arr)
    mb = path.stat().st_size / (1024 * 1024)
    print(f"saved cache → {path}  ({mb:.1f} MB) in {time.perf_counter() - t0:.1f}s", flush=True)
    return load_val_cache(path)


def load_val_cache(path: Path | str, *, promote_rgb: bool = True) -> ValCacheStacked:
    import time

    path = Path(path).with_suffix(".npz")
    t0 = time.perf_counter()
    z = np.load(path, allow_pickle=True)
    out = ValCacheStacked(z, promote_rgb=promote_rgb)
    mb = path.stat().st_size / (1024 * 1024)
    print(f"loaded cache ← {path}  ({len(out)} samples, {mb:.1f} MB) in {time.perf_counter() - t0:.1f}s", flush=True)
    return out


# --- ensemble blends ---


def blend_linear(model: np.ndarray, rife: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(alpha * model + (1.0 - alpha) * rife, 0.0, 1.0)


def blend_cov_gate(model: np.ndarray, rife: np.ndarray, cov: np.ndarray, thr: float) -> np.ndarray:
    m = (cov >= thr)[..., None]
    return np.where(m, model, rife).astype(np.float32)


def blend_cov_soft(model: np.ndarray, rife: np.ndarray, cov: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    w = np.clip(cov, 0, 1)[..., None] ** gamma
    return np.clip(w * model + (1.0 - w) * rife, 0.0, 1.0)


def blend_alpha_cov(
    model: np.ndarray, rife: np.ndarray, cov: np.ndarray, alpha: float, beta: float
) -> np.ndarray:
    """alpha — вес модели в дырках; beta — усиление модели при высоком coverage."""
    w = np.clip(alpha + beta * cov, 0.0, 1.0)[..., None]
    return np.clip(w * model + (1.0 - w) * rife, 0.0, 1.0)


def blend_conf_gate(
    model: np.ndarray, rife: np.ndarray, conf: np.ndarray, thr: float
) -> np.ndarray:
    m = (conf >= thr)[..., None]
    return np.where(m, model, rife).astype(np.float32)


def eval_cache_psnr(cache: List[dict], blend_fn) -> float:
    psnrs = [psnr_uint8_np(blend_fn(e), e["gt"]) for e in cache]
    return float(np.mean(psnrs))


# --- preprocess (anisotropic) ---


def _rgb_to_gray(img: np.ndarray) -> np.ndarray:
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def anisotropic_diffusion_rgb(
    img: np.ndarray,
    n_iter: int = 8,
    kappa: float = 0.06,
    lam: float = 0.15,
    depth_guide: Optional[np.ndarray] = None,
    art_mask: Optional[np.ndarray] = None,
    art_kappa_scale: float = 0.35,
) -> np.ndarray:
    """
    Perona-Malik с общей проводимостью по яркости (сохраняет цвет).
    В зоне ego (art) диффузия слабее (kappa_eff = kappa * art_kappa_scale).
    """
    out = img.astype(np.float32).copy()
    gray = _rgb_to_gray(out)
    for _ in range(n_iter):
        g_pad = np.pad(gray, 1, mode="edge")
        c_pad = [np.pad(out[..., ch], 1, mode="edge") for ch in range(3)]
        deltas = []
        for ch in range(3):
            north = c_pad[ch][:-2, 1:-1] - c_pad[ch][1:-1, 1:-1]
            south = c_pad[ch][2:, 1:-1] - c_pad[ch][1:-1, 1:-1]
            east = c_pad[ch][1:-1, 2:] - c_pad[ch][1:-1, 1:-1]
            west = c_pad[ch][1:-1, :-2] - c_pad[ch][1:-1, 1:-1]
            deltas.append((north, south, east, west))
        gn = g_pad[:-2, 1:-1] - g_pad[1:-1, 1:-1]
        gs = g_pad[2:, 1:-1] - g_pad[1:-1, 1:-1]
        ge = g_pad[1:-1, 2:] - g_pad[1:-1, 1:-1]
        gw = g_pad[1:-1, :-2] - g_pad[1:-1, 1:-1]
        kappa_eff = np.full_like(gray, kappa, dtype=np.float32)
        if art_mask is not None:
            kappa_eff = kappa_eff * (art_kappa_scale + (1.0 - art_kappa_scale) * (1.0 - art_mask))
        cn = np.exp(-(gn / (kappa_eff + EPS)) ** 2)
        cs = np.exp(-(gs / (kappa_eff + EPS)) ** 2)
        ce = np.exp(-(ge / (kappa_eff + EPS)) ** 2)
        cw = np.exp(-(gw / (kappa_eff + EPS)) ** 2)
        for ch in range(3):
            north, south, east, west = deltas[ch]
            out[..., ch] += lam * (cn * north + cs * south + ce * east + cw * west)
        gray = _rgb_to_gray(out)
    return np.clip(out, 0.0, 1.0)


def structure_tensor_aniso_smooth(
    img: np.ndarray,
    sigma_grad: float = 1.0,
    sigma_tensor: float = 2.0,
    n_iter: int = 6,
    step: float = 0.2,
    coherence: float = 5.0,
) -> np.ndarray:
    """
    Параметрическое анизотропное сглаживание: диффузия вдоль структур (меньше поперёк границ).
    """
    if cv2 is None:
        return anisotropic_diffusion_rgb(img, n_iter=max(n_iter, 6))

    out = img.astype(np.float32).copy()
    gray = _rgb_to_gray(out)
    g = (gray * 255).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    if sigma_grad > 0:
        gx = cv2.GaussianBlur(gx, (0, 0), sigma_grad)
        gy = cv2.GaussianBlur(gy, (0, 0), sigma_grad)
    j11 = cv2.GaussianBlur(gx * gx, (0, 0), sigma_tensor)
    j12 = cv2.GaussianBlur(gx * gy, (0, 0), sigma_tensor)
    j22 = cv2.GaussianBlur(gy * gy, (0, 0), sigma_tensor)
    tr = j11 + j22
    det = j11 * j22 - j12 * j12
    disc = np.sqrt(np.maximum(tr * tr - 4.0 * det, 0.0))
    l1 = 0.5 * (tr + disc)
    nx = 2.0 * j12
    ny = l1 - j11
    nrm = np.sqrt(nx * nx + ny * ny) + EPS
    tx, ty = -ny / nrm, nx / nrm

    def deriv_along(c, t_x, t_y):
        c_t = t_x * cv2.Sobel(c, cv2.CV_32F, 1, 0, ksize=3) + t_y * cv2.Sobel(c, cv2.CV_32F, 0, 1, ksize=3)
        return t_x * cv2.Sobel(c_t, cv2.CV_32F, 1, 0, ksize=3) + t_y * cv2.Sobel(c_t, cv2.CV_32F, 0, 1, ksize=3)

    for _ in range(n_iter):
        for ch in range(3):
            c_tt = deriv_along(out[..., ch], tx, ty)
            out[..., ch] = out[..., ch] + step * coherence * c_tt
    return np.clip(out, 0.0, 1.0)


def preprocess_model(
    model: np.ndarray,
    method: str = "pm",
    depth: Optional[np.ndarray] = None,
    art: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """Сгладить выход модели перед ансамблем. method: 'pm' | 'tensor' | 'none'."""
    if method == "none":
        return model
    if method == "tensor":
        return structure_tensor_aniso_smooth(model, **kwargs)
    return anisotropic_diffusion_rgb(
        model, depth_guide=depth, art_mask=art, **kwargs
    )


# --- depth + art blending ---


def effective_depth(
    depth: np.ndarray,
    art: Optional[np.ndarray] = None,
    art_value: float = 0.0,
) -> np.ndarray:
    """depth в [0,1]; ego-зона → art_value (почти «у камеры»)."""
    d = np.clip(depth.astype(np.float32), 0.0, 1.0)
    if art is not None:
        d = np.where(art > 0.5, art_value, d)
    return d


def depth_model_weight(
    depth: np.ndarray,
    cov: np.ndarray,
    art: Optional[np.ndarray] = None,
    depth_valid: Optional[np.ndarray] = None,
    lidar_trust: Optional[np.ndarray] = None,
    d_lo: float = 0.12,
    d_peak_lo: float = 0.28,
    d_peak_hi: float = 0.52,
    d_hi: float = 0.68,
    cov_min: float = 0.12,
    gamma_c: float = 1.0,
    art_depth: float = 0.0,
) -> np.ndarray:
    """
    Вес модели [0..1]: «трапеция» по глубине + coverage + валидный лидар.

    RIFE там, где:
    - ego (art → depth≈0),
    - нет зоны лидара (lidar_trust≈0; иначе depth_valid=0),
    - слишком близко (d < d_lo) или слишком далеко (d > d_hi) — в т.ч. небо без точек,
    - низкий coverage (дырки warp).
    """
    d = effective_depth(depth, art, art_value=art_depth)
    w = np.zeros_like(d, dtype=np.float32)
    up = np.clip((d - d_lo) / max(d_peak_lo - d_lo, EPS), 0.0, 1.0)
    w = np.where(d < d_peak_lo, _smoothstep(up), w)
    w = np.where((d >= d_peak_lo) & (d <= d_peak_hi), 1.0, w)
    down = np.clip((d_hi - d) / max(d_hi - d_peak_hi, EPS), 0.0, 1.0)
    w = np.where(d > d_peak_hi, _smoothstep(down), w)
    if lidar_trust is not None:
        w = w * np.clip(lidar_trust, 0.0, 1.0)
    elif depth_valid is not None:
        w = w * (depth_valid > 0.5).astype(np.float32)
    else:
        w = w * (depth > 1e-4).astype(np.float32)
    w = w * (np.clip(cov, 0.0, 1.0) ** gamma_c)
    if cov_min > 0:
        w = w * (cov >= cov_min).astype(np.float32)
    if art is not None:
        w = w * (1.0 - (art > 0.5).astype(np.float32))
    return np.clip(w, 0.0, 1.0)


def depth_trust_decompose(
    depth: np.ndarray,
    cov: np.ndarray,
    art: Optional[np.ndarray] = None,
    depth_valid: Optional[np.ndarray] = None,
    lidar_trust: Optional[np.ndarray] = None,
    d_lo: float = 0.12,
    d_peak_lo: float = 0.28,
    d_peak_hi: float = 0.52,
    d_hi: float = 0.68,
    cov_min: float = 0.12,
    gamma_c: float = 1.0,
    art_depth: float = 0.0,
) -> dict:
    """Пошаговые карты для отладки depth_trust (см. plot_depth_trust_debug)."""
    d = effective_depth(depth, art, art_value=art_depth)
    w_trap = np.zeros_like(d, dtype=np.float32)
    up = np.clip((d - d_lo) / max(d_peak_lo - d_lo, EPS), 0.0, 1.0)
    w_trap = np.where(d < d_peak_lo, _smoothstep(up), w_trap)
    w_trap = np.where((d >= d_peak_lo) & (d <= d_peak_hi), 1.0, w_trap)
    down = np.clip((d_hi - d) / max(d_hi - d_peak_hi, EPS), 0.0, 1.0)
    w_trap = np.where(d > d_peak_hi, _smoothstep(down), w_trap)

    if lidar_trust is not None:
        lidar = np.clip(lidar_trust, 0.0, 1.0).astype(np.float32)
    elif depth_valid is not None:
        lidar = (depth_valid > 0.5).astype(np.float32)
    else:
        lidar = (depth > 1e-4).astype(np.float32)
    w_lidar = w_trap * lidar
    cov_f = np.clip(cov, 0.0, 1.0) ** gamma_c
    w_cov = w_lidar * cov_f
    if cov_min > 0:
        w_cov = w_cov * (cov >= cov_min).astype(np.float32)
    w_final = w_cov.copy()
    if art is not None:
        w_final = w_final * (1.0 - (art > 0.5).astype(np.float32))
    ego = (art > 0.5).astype(np.float32) if art is not None else np.zeros_like(d)
    mask_close = (d < d_lo).astype(np.float32)
    mask_far_geom = (d > d_hi).astype(np.float32)
    mask_no_lidar = (1.0 - lidar)
    mask_low_cov = (cov < cov_min).astype(np.float32) if cov_min > 0 else np.zeros_like(d)
    # «дальнее без лидара»: нет точек И (геом. далеко ИЛИ норм. depth≈0)
    mask_far_void = mask_no_lidar * np.maximum(mask_far_geom, (depth <= 1e-4).astype(np.float32))
    depth_scene = np.clip(depth, 0.0, 1.0) * lidar * (1.0 - ego)
    return {
        "d_eff": d,
        "w_trap": w_trap,
        "lidar": lidar,
        "w_lidar": w_lidar,
        "cov_f": cov_f,
        "w_cov": w_cov,
        "w_final": np.clip(w_final, 0.0, 1.0),
        "mask_close": mask_close,
        "mask_far_geom": mask_far_geom,
        "mask_no_lidar": mask_no_lidar,
        "mask_far_void": mask_far_void,
        "mask_low_cov": mask_low_cov,
        "mask_ego": ego,
        "depth_scene": depth_scene,
    }


def plot_depth_trust_debug(
    entry: dict,
    *,
    d_lo: float = 0.12,
    d_peak_lo: float = 0.28,
    d_peak_hi: float = 0.52,
    d_hi: float = 0.68,
    cov_min: float = 0.12,
    gamma_c: float = 1.0,
    title: str = "",
):
    """
    Визуализация: depth, маски, «минус дальнее» (far / без лидара) → w_model.

    Требует в entry: depth, cov, art; опционально depth_valid, gt.
    """
    import matplotlib.pyplot as plt

    depth = entry["depth"]
    if depth.ndim == 3:
        depth = depth[0]
    cov = entry["cov"]
    if cov.ndim == 3:
        cov = cov[0]
    art = entry.get("art")
    if art is not None and art.ndim == 3:
        art = art[0]
    dv = entry.get("depth_valid")
    if dv is not None and dv.ndim == 3:
        dv = dv[0]
    lt = entry.get("lidar_trust")
    if lt is not None and lt.ndim == 3:
        lt = lt[0]

    dec = depth_trust_decompose(
        depth, cov, art, dv, lt,
        d_lo=d_lo, d_peak_lo=d_peak_lo, d_peak_hi=d_peak_hi, d_hi=d_hi,
        cov_min=cov_min, gamma_c=gamma_c,
    )
    kw = dict(d_lo=d_lo, d_hi=d_hi, cov_min=cov_min)

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle(title or entry.get("meta", {}).get("sample_id", "depth_trust"), fontsize=11)

    def _im(ax, arr, cmap="viridis", vmin=None, vmax=None, t=""):
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(t, fontsize=9)
        ax.axis("off")

    _im(axes[0, 0], depth, "plasma", 0, 1, "depth norm")
    _im(axes[0, 1], dec["lidar"], "viridis", 0, 1, "lidar_trust (зоны)")
    _im(axes[0, 2], dec["mask_ego"], "gray", 0, 1, "ego art")
    _im(axes[0, 3], cov, "viridis", 0, 1, "coverage")

    _im(axes[1, 0], dec["d_eff"], "plasma", 0, 1, "d_eff (art→0)")
    _im(axes[1, 1], dec["w_trap"], "viridis", 0, 1, "трапеция (только d)")
    _im(axes[1, 2], dec["w_lidar"], "viridis", 0, 1, "× lidar valid")
    _im(axes[1, 3], dec["w_final"], "viridis", 0, 1, "w_model итог")

    _im(axes[2, 0], dec["depth_scene"], "plasma", 0, 1, "depth×lidar×(1-ego)")
    _im(axes[2, 1], dec["mask_close"], "Reds", 0, 1, "RIFE: близко d<d_lo")
    _im(axes[2, 2], dec["mask_far_void"], "Blues", 0, 1, "RIFE: дальн.+нет лидара")
    _im(axes[2, 3], 1.0 - dec["w_final"], "magma", 0, 1, "1 − w_model → RIFE")

    plt.tight_layout()
    pct = {
        "близко": 100 * dec["mask_close"].mean(),
        "дальн_геом": 100 * dec["mask_far_geom"].mean(),
        "нет_лидара": 100 * dec["mask_no_lidar"].mean(),
        "дальн_void": 100 * dec["mask_far_void"].mean(),
        "ego": 100 * dec["mask_ego"].mean(),
        "w_model>0.5": 100 * (dec["w_final"] > 0.5).mean(),
    }
    print(f"пороги {kw}  |  % пикселей:", ", ".join(f"{k}={v:.1f}" for k, v in pct.items()))
    return fig, dec


def blend_depth_soft(
    model: np.ndarray,
    rife: np.ndarray,
    depth: np.ndarray,
    art: Optional[np.ndarray] = None,
    gamma: float = 1.2,
    art_depth: float = 0.0,
    invert: bool = False,
) -> np.ndarray:
    """Устаревший монотонный вариант (только для сравнения). invert не рекомендуется."""
    d = effective_depth(depth, art, art_value=art_depth)
    w = np.clip(d, 0.0, 1.0) ** gamma
    if invert:
        w = 1.0 - w
    w = w[..., None]
    return np.clip(w * model + (1.0 - w) * rife, 0.0, 1.0)


def blend_depth_band(
    model: np.ndarray,
    rife: np.ndarray,
    depth: np.ndarray,
    cov: np.ndarray,
    art: Optional[np.ndarray] = None,
    depth_valid: Optional[np.ndarray] = None,
    lidar_trust: Optional[np.ndarray] = None,
    d_lo: float = 0.12,
    d_peak_lo: float = 0.28,
    d_peak_hi: float = 0.52,
    d_hi: float = 0.68,
    cov_min: float = 0.12,
    gamma_c: float = 1.0,
    art_depth: float = 0.0,
) -> np.ndarray:
    """Рекомендуемый depth-blend: модель в средней зоне глубины с лидar-зонами и coverage."""
    w = depth_model_weight(
        depth, cov, art, depth_valid, lidar_trust,
        d_lo=d_lo, d_peak_lo=d_peak_lo, d_peak_hi=d_peak_hi, d_hi=d_hi,
        cov_min=cov_min, gamma_c=gamma_c, art_depth=art_depth,
    )
    return np.clip(w[..., None] * model + (1.0 - w[..., None]) * rife, 0.0, 1.0)


def blend_depth_cov(
    model: np.ndarray,
    rife: np.ndarray,
    depth: np.ndarray,
    cov: np.ndarray,
    art: Optional[np.ndarray] = None,
    depth_valid: Optional[np.ndarray] = None,
    gamma_d: float = 1.0,
    gamma_c: float = 1.0,
    art_depth: float = 0.0,
    **kwargs,
) -> np.ndarray:
    """Алиас на blend_depth_band (+ лишние kwargs для совместимости)."""
    kw = dict(art=art, depth_valid=depth_valid, lidar_trust=kwargs.get("lidar_trust"), gamma_c=gamma_c, art_depth=art_depth)
    kw.update({k: v for k, v in kwargs.items() if k in depth_model_weight.__code__.co_varnames})
    return blend_depth_band(model, rife, depth, cov, **kw)


def blend_pipeline(
    entry: dict,
    *,
    smooth: str = "pm",
    smooth_kwargs: Optional[dict] = None,
    blend: str = "depth_band",
    blend_kwargs: Optional[dict] = None,
    linear_alpha: Optional[float] = None,
) -> np.ndarray:
    """
    model → aniso smooth → depth/art blend → опционально linear с RIFE.
    """
    smooth_kwargs = smooth_kwargs or {}
    blend_kwargs = blend_kwargs or {}
    m = preprocess_model(
        entry["model"],
        method=smooth,
        depth=entry.get("depth"),
        art=entry.get("art"),
        **smooth_kwargs,
    )
    depth = entry.get("depth")
    art = entry.get("art")
    cov = entry.get("cov")
    rife = entry["rife"]
    depth_valid = entry.get("depth_valid")
    lidar_trust = entry.get("lidar_trust")
    if blend == "depth_soft":
        out = blend_depth_soft(m, rife, depth, art, **blend_kwargs)
    elif blend in ("depth_band", "depth_cov", "depth_trust"):
        out = blend_depth_band(m, rife, depth, cov, art, depth_valid, lidar_trust, **blend_kwargs)
    else:
        out = blend_depth_band(m, rife, depth, cov, art, depth_valid, lidar_trust, **blend_kwargs)
    if linear_alpha is not None:
        out = blend_linear(out, rife, linear_alpha)
    return out


__all__ = [
    "CONSENSUS_KIT_VERSION",
    "ConsensusConfig",
    "ConsensusDataset",
    "ConsensusUNet",
    "anisotropic_diffusion_rgb",
    "blend_alpha_cov",
    "blend_conf_gate",
    "blend_cov_gate",
    "blend_cov_soft",
    "blend_depth_band",
    "blend_depth_cov",
    "blend_depth_soft",
    "blend_depth_trust",
    "blend_linear",
    "blend_pipeline",
    "default_config",
    "depth_model_weight",
    "depth_trust_decompose",
    "discover_samples",
    "plot_depth_trust_debug",
    "effective_depth",
    "eval_cache_psnr",
    "load_consensus_model",
    "make_split",
    "preprocess_model",
    "psnr_uint8_np",
    "psnr_uint8_torch",
    "infer_and_save_val_cache",
    "load_val_cache",
    "run_inference_cache",
    "save_val_cache",
    "ValCacheStacked",
    "structure_tensor_aniso_smooth",
]

# alias
blend_depth_trust = blend_depth_band
