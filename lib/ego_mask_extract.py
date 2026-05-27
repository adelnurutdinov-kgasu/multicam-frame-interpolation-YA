"""Shared helpers for manual ego-mask extraction."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def extract_red_paint(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    return (r - np.maximum(g, b) > 80) & (r > 120)


def extract_black_paint(rgb: np.ndarray, red_paint: np.ndarray | None = None) -> np.ndarray:
    if red_paint is None:
        red_paint = extract_red_paint(rgb)
    lum = rgb.astype(np.float32).mean(-1)
    return (lum < 25) & ~red_paint


def _close_mask(mask: np.ndarray) -> np.ndarray:
    if mask.sum() < 64:
        return np.zeros(mask.shape, dtype=bool)
    u8 = mask.astype(np.uint8)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    return u8.astype(bool)


def extract_red_mask(rgb: np.ndarray) -> np.ndarray:
    return _close_mask(extract_red_paint(rgb))


def extract_black_mask(rgb: np.ndarray) -> np.ndarray:
    red_paint = extract_red_paint(rgb)
    return _close_mask(extract_black_paint(rgb, red_paint))


def extract_red_plus_black_mask(rgb: np.ndarray) -> np.ndarray:
    red_paint = extract_red_paint(rgb)
    black_paint = extract_black_paint(rgb, red_paint)
    return _close_mask(red_paint | black_paint)


def extract_mask_by_mode(rgb: np.ndarray, mode: str) -> np.ndarray:
    if mode == "red":
        return extract_red_mask(rgb)
    if mode == "black":
        return extract_black_mask(rgb)
    if mode == "red_plus_black":
        return extract_red_plus_black_mask(rgb)
    raise ValueError(f"unknown mask mode: {mode}")


def detect_dominant_modes(rgb: np.ndarray) -> dict[str, int]:
    red_paint = extract_red_paint(rgb)
    black_paint = extract_black_paint(rgb, red_paint)
    return {
        "red_px": int(red_paint.sum()),
        "black_px": int(black_paint.sum()),
    }


def infer_variant_mode(rgb: np.ndarray, min_px: int = 500) -> str:
    stats = detect_dominant_modes(rgb)
    has_red = stats["red_px"] >= min_px
    has_black = stats["black_px"] >= min_px
    if has_red and has_black:
        return "red_plus_black"
    if has_red:
        return "red"
    if has_black:
        return "black"
    return "empty"


def resize_mask_to(mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    if mask.shape[:2] == hw:
        return mask
    u8 = cv2.resize(mask.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return u8.astype(bool)


def save_binary_mask(mask: np.ndarray, path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)
