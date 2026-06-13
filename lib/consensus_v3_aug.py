"""Синхронные аугментации для Consensus V3 (после camera→canonical mirror)."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class V3AugCfg:
    enabled: bool = True
    # Random crop; 0 = полный кадр (только val)
    patch_h: int = 416
    patch_w: int = 768
    # Геометрия (одинаково для warp / mean / target / trust)
    hflip_p: float = 0.5
    vflip_p: float = 0.0
    # Каждый сэмпл дважды: оригинал + зеркало по H → ~2× данных
    virtual_mirror_2x: bool = True
    # Цвет в YCrCb [0..1]
    y_gain: Tuple[float, float] = (0.90, 1.10)
    chroma_offset: float = 0.03
    input_noise_std: float = 0.012
    # Cutout на warp (только вход), доля площади
    cutout_p: float = 0.25
    cutout_frac: Tuple[float, float] = (0.02, 0.12)


def _flip_h(a: np.ndarray) -> np.ndarray:
    return a[:, ::-1].copy() if a.ndim == 3 else a[:, ::-1].copy()


def _flip_v(a: np.ndarray) -> np.ndarray:
    return a[::-1].copy() if a.ndim == 3 else a[::-1].copy()


def _random_crop(
    arrays: Dict[str, np.ndarray], ph: int, pw: int
) -> Dict[str, np.ndarray]:
    ref = arrays["warp"]
    h, w = ref.shape[:2]
    if ph <= 0 or pw <= 0 or (h <= ph and w <= pw):
        return arrays
    y0 = 0 if h <= ph else random.randint(0, h - ph)
    x0 = 0 if w <= pw else random.randint(0, w - pw)
    out: Dict[str, np.ndarray] = {}
    for k, a in arrays.items():
        if a.ndim == 3:
            out[k] = a[y0 : y0 + ph, x0 : x0 + pw].copy()
        else:
            out[k] = a[y0 : y0 + ph, x0 : x0 + pw].copy()
    return out


def _cutout_warp(arrays: Dict[str, np.ndarray], cfg: V3AugCfg) -> None:
    if cfg.cutout_p <= 0 or random.random() >= cfg.cutout_p:
        return
    warp = arrays["warp"]
    h, w = warp.shape[:2]
    lo, hi = cfg.cutout_frac
    area = h * w * random.uniform(lo, hi)
    ch = int(max(8, area**0.5))
    cw = int(max(8, area / max(ch, 1)))
    y0 = random.randint(0, max(0, h - ch))
    x0 = random.randint(0, max(0, w - cw))
    fill = float(np.mean(warp[y0 : y0 + ch, x0 : x0 + cw, 0]))
    warp[y0 : y0 + ch, x0 : x0 + cw, :] = fill


def _color_jitter_ycc(arrays: Dict[str, np.ndarray], cfg: V3AugCfg) -> None:
    g_lo, g_hi = cfg.y_gain
    gain = random.uniform(g_lo, g_hi)
    for key in ("warp", "mean", "target"):
        a = arrays[key]
        a[..., 0] = np.clip(a[..., 0] * gain, 0.0, 1.0)
        if cfg.chroma_offset > 0:
            delta = random.uniform(-cfg.chroma_offset, cfg.chroma_offset)
            a[..., 1:] = np.clip(a[..., 1:] + delta, 0.0, 1.0)
    if cfg.input_noise_std > 0:
        noise = np.random.randn(*arrays["warp"].shape).astype(np.float32) * cfg.input_noise_std
        arrays["warp"] = np.clip(arrays["warp"] + noise, 0.0, 1.0)


def apply_v3_augment(
    warp_ycc: np.ndarray,
    mean_ycc: np.ndarray,
    target_ycc: np.ndarray,
    trust: np.ndarray,
    cfg: V3AugCfg,
    *,
    force_hflip: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """warp/mean/target: HWC float [0,1] YCrCb; trust: HW float."""
    if not cfg.enabled:
        return warp_ycc, mean_ycc, target_ycc, trust

    arrays: Dict[str, np.ndarray] = {
        "warp": warp_ycc,
        "mean": mean_ycc,
        "target": target_ycc,
        "trust": trust,
    }

    do_h = force_hflip or (cfg.hflip_p > 0 and random.random() < cfg.hflip_p)
    if do_h:
        arrays = {k: _flip_h(v) for k, v in arrays.items()}
    if cfg.vflip_p > 0 and random.random() < cfg.vflip_p:
        arrays = {k: _flip_v(v) for k, v in arrays.items()}

    arrays = _random_crop(arrays, cfg.patch_h, cfg.patch_w)
    _cutout_warp(arrays, cfg)
    _color_jitter_ycc(arrays, cfg)
    return arrays["warp"], arrays["mean"], arrays["target"], arrays["trust"]
