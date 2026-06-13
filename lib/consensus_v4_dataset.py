"""Consensus V4: те же входы что V3, всё в RGB (7ch: warp_rgb + trust + mean_rgb)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from lib.consensus_kit import ConsensusConfig
from lib.consensus_v3_dataset import (
    MIRROR_CAMERAS,
    V3TrainCfg,
    flip_h,
    load_rgb,
    load_warp_rgb,
    v3_ready_with_trust,
    v3_train_val_split,
)
from lib.consensus_v4_aug import V4AugCfg, apply_v4_augment

V4TrainCfg = V3TrainCfg


def _rgb01(rgb_u8: np.ndarray) -> np.ndarray:
    return rgb_u8.astype(np.float32) / 255.0


class V4TrainDataset(Dataset):
    def __init__(
        self,
        baked_dirs: List[Path],
        cfg: V4TrainCfg,
        ds_cfg: ConsensusConfig,
        aug_cfg: Optional[V4AugCfg] = None,
    ):
        self.dirs = [Path(p) for p in baked_dirs]
        self.cfg = cfg
        self.ds_cfg = ds_cfg
        self.aug_cfg = aug_cfg or V4AugCfg(enabled=False)
        self._virtual_2x = bool(self.aug_cfg.enabled and self.aug_cfg.virtual_mirror_2x)

    def __len__(self) -> int:
        return len(self.dirs) * (2 if self._virtual_2x else 1)

    def _resolve(self, index: int) -> Tuple[int, bool]:
        if self._virtual_2x:
            return index // 2, (index % 2) == 1
        return index, False

    def __getitem__(self, index: int):
        idx, force_flip = self._resolve(index)
        baked_dir = self.dirs[idx]
        meta = json.loads((baked_dir / "meta.json").read_text(encoding="utf-8"))
        sid = meta["sample_id"]
        cam = meta["camera"]
        src = Path(meta.get("source_dir", self.cfg.dataset_root / sid))
        hw = (self.cfg.image_h, self.cfg.image_w)

        warp_rgb = _rgb01(load_warp_rgb(self.cfg, sid, hw))
        t0 = load_rgb(src / "input" / "t0" / f"{cam}.jpg", hw)
        t1 = load_rgb(src / "input" / "t1" / f"{cam}.jpg", hw)
        target_rgb = _rgb01(load_rgb(src / "target" / f"{cam}.jpg", hw))
        mean_rgb = _rgb01(((t0.astype(np.float32) + t1.astype(np.float32)) * 0.5).astype(np.uint8))

        trust = np.load(self.cfg.trust_cache_root / sid / "lidar_trust.npy").astype(np.float32)
        if trust.shape != hw:
            trust = cv2.resize(trust, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
        trust = np.clip(trust, 0.0, 1.0)

        if cam in MIRROR_CAMERAS:
            warp_rgb, mean_rgb, target_rgb, trust = flip_h(warp_rgb, mean_rgb, target_rgb, trust)

        warp_rgb, mean_rgb, target_rgb, trust = apply_v4_augment(
            warp_rgb, mean_rgb, target_rgb, trust, self.aug_cfg, force_hflip=force_flip
        )

        x = np.concatenate([warp_rgb, trust[..., None], mean_rgb], axis=-1)
        h, w = warp_rgb.shape[:2]
        to_chw = lambda a: torch.from_numpy(a.transpose(2, 0, 1)).contiguous().float()
        to_1chw = lambda a: torch.from_numpy(a)[None].contiguous().float()

        return {
            "inputs": to_chw(x),
            "base_init": to_chw(warp_rgb),
            "effective_mask": to_1chw(np.ones((h, w), dtype=np.float32)),
            "target": to_chw(target_rgb),
            "meta": {"sample_id": sid, "camera": cam, "color_space": "rgb", "aug_flip": force_flip},
        }
