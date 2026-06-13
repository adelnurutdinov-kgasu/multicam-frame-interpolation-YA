"""Политика ego-масок: группы без маскирования (пустая art_mask в пайплайне)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

# (vehicle, camera) — не маскируем ego / артефакты
ZERO_MASK_GROUPS: frozenset[Tuple[str, str]] = frozenset({
    ("natelio", "right_fwd"),
    ("orvy", "right_fwd"),
    ("crozby", "right_fwd"),
})

ZERO_MASK_GROUP_IDS: frozenset[str] = frozenset(
    f"{v}_{c}" for v, c in ZERO_MASK_GROUPS
)


def is_zero_mask_group(vehicle: str, camera: str) -> bool:
    return (vehicle, camera) in ZERO_MASK_GROUPS


def is_zero_mask_group_id(group_id: str) -> bool:
    return group_id in ZERO_MASK_GROUP_IDS


def write_zero_approved_mask(approved_dir: Path, vehicle: str, camera: str) -> Path:
    """Пустая маска в разрешении mean (или 540×1024 fallback)."""
    from import_manual_ego_masks import MEANS_DIR

    mean_path = MEANS_DIR / f"{vehicle}_{camera}.png"
    if mean_path.is_file():
        hw = np.array(Image.open(mean_path)).shape[:2]
    else:
        hw = (540, 1024)
    out = approved_dir / f"{vehicle}_{camera}.png"
    approved_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros(hw, dtype=np.uint8)).save(out)
    return out
