"""Синхронные аугментации для Consensus V4 (RGB [0..1])."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from lib.consensus_v3_aug import (
    V3AugCfg,
    _cutout_warp,
    _flip_h,
    _flip_v,
    _random_crop,
)


@dataclass
class V4AugCfg(V3AugCfg):
    """Те же поля, что V3; rgb_gain — alias для y_gain (яркость всех каналов)."""

    rgb_gain: Tuple[float, float] = (0.90, 1.10)
    rgb_offset: float = 0.03


def _color_jitter_rgb(arrays: Dict[str, np.ndarray], cfg: V4AugCfg) -> None:
    g_lo, g_hi = cfg.rgb_gain
    gain = random.uniform(g_lo, g_hi)
    for key in ("warp", "mean", "target"):
        a = arrays[key]
        a[:] = np.clip(a * gain, 0.0, 1.0)
        if cfg.rgb_offset > 0:
            delta = np.random.uniform(-cfg.rgb_offset, cfg.rgb_offset, size=(1, 1, 3)).astype(
                np.float32
            )
            a[:] = np.clip(a + delta, 0.0, 1.0)
    if cfg.input_noise_std > 0:
        noise = np.random.randn(*arrays["warp"].shape).astype(np.float32) * cfg.input_noise_std
        arrays["warp"] = np.clip(arrays["warp"] + noise, 0.0, 1.0)


def apply_v4_augment(
    warp_rgb: np.ndarray,
    mean_rgb: np.ndarray,
    target_rgb: np.ndarray,
    trust: np.ndarray,
    cfg: V4AugCfg,
    *,
    force_hflip: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not cfg.enabled:
        return warp_rgb, mean_rgb, target_rgb, trust

    arrays: Dict[str, np.ndarray] = {
        "warp": warp_rgb,
        "mean": mean_rgb,
        "target": target_rgb,
        "trust": trust,
    }
    do_h = force_hflip or (cfg.hflip_p > 0 and random.random() < cfg.hflip_p)
    if do_h:
        arrays = {k: _flip_h(v) for k, v in arrays.items()}
    if cfg.vflip_p > 0 and random.random() < cfg.vflip_p:
        arrays = {k: _flip_v(v) for k, v in arrays.items()}

    arrays = _random_crop(arrays, cfg.patch_h, cfg.patch_w)
    _cutout_warp(arrays, cfg)
    _color_jitter_rgb(arrays, cfg)
    return arrays["warp"], arrays["mean"], arrays["target"], arrays["trust"]
