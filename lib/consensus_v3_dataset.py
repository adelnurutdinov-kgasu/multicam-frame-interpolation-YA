"""Consensus V3 train/val dataset: V2 inputs + аугментации."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from lib.consensus_kit import ConsensusConfig, _chw_npy_to_hwc_u8, discover_samples
from lib.consensus_v3_aug import V3AugCfg, apply_v3_augment

MIRROR_CAMERAS = ("left_fwd", "right_bwd")


@dataclass
class V3TrainCfg:
    image_h: int = 544
    image_w: int = 1024
    trust_zone_min: float = 0.12
    dataset_root: Path = Path(".")
    warps_root: Path = Path(".")
    warp_mix_root: Path = Path(".")
    trust_cache_root: Path = Path(".")
    ego_masks_root: Path = Path(".")
    val_fraction: float = 0.16
    seed: int = 42


def flip_h(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    return tuple(a[:, ::-1].copy() for a in arrays)


def load_rgb(path: Path, hw: Tuple[int, int]) -> np.ndarray:
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    if arr.shape[:2] != hw:
        arr = cv2.resize(arr, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    return arr


def to_ycrcb01(rgb_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255.0


def load_warp_rgb(cfg: V3TrainCfg, sid: str, hw: Tuple[int, int]) -> np.ndarray:
    mix = cfg.warp_mix_root / sid / "warp_mix.jpg"
    if mix.is_file():
        return load_rgb(mix, hw)
    raw = cfg.warps_root / sid / "consensus_raw.npy"
    if not raw.is_file():
        raise FileNotFoundError(f"no warp for {sid}")
    warp = _chw_npy_to_hwc_u8(raw)
    if warp.shape[:2] != hw:
        warp = cv2.resize(warp, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    return warp


def v3_ready_with_trust(ds_cfg: ConsensusConfig, trust_root: Path) -> List[Path]:
    ready: List[Path] = []
    for bdir in discover_samples(ds_cfg):
        meta = json.loads((bdir / "meta.json").read_text(encoding="utf-8"))
        sid = meta["sample_id"]
        if (trust_root / sid / "lidar_trust.npy").is_file():
            ready.append(bdir)
    return ready


def v3_train_val_split(
    ready: List[Path], *, seed: int, val_fraction: float, subset: str
) -> List[Path]:
    rng = np.random.RandomState(seed)
    idx = np.arange(len(ready))
    rng.shuffle(idx)
    chosen = [ready[i] for i in idx]
    n_val = max(1, int(len(chosen) * val_fraction))
    if subset == "val":
        return chosen[:n_val]
    if subset == "train":
        return chosen[n_val:]
    return chosen


class V3TrainDataset(Dataset):
    def __init__(
        self,
        baked_dirs: List[Path],
        cfg: V3TrainCfg,
        ds_cfg: ConsensusConfig,
        aug_cfg: Optional[V3AugCfg] = None,
    ):
        self.dirs = [Path(p) for p in baked_dirs]
        self.cfg = cfg
        self.ds_cfg = ds_cfg
        self.aug_cfg = aug_cfg or V3AugCfg(enabled=False)
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

        warp_rgb = load_warp_rgb(self.cfg, sid, hw)
        t0 = load_rgb(src / "input" / "t0" / f"{cam}.jpg", hw)
        t1 = load_rgb(src / "input" / "t1" / f"{cam}.jpg", hw)
        target_rgb = load_rgb(src / "target" / f"{cam}.jpg", hw)
        mean_rgb = ((t0.astype(np.float32) + t1.astype(np.float32)) * 0.5).astype(np.uint8)

        trust = np.load(self.cfg.trust_cache_root / sid / "lidar_trust.npy").astype(np.float32)
        if trust.shape != hw:
            trust = cv2.resize(trust, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
        trust = np.clip(trust, 0.0, 1.0)

        warp_ycc = to_ycrcb01(warp_rgb)
        mean_ycc = to_ycrcb01(mean_rgb)
        target_ycc = to_ycrcb01(target_rgb)

        if cam in MIRROR_CAMERAS:
            warp_ycc, mean_ycc, target_ycc, trust = flip_h(warp_ycc, mean_ycc, target_ycc, trust)

        warp_ycc, mean_ycc, target_ycc, trust = apply_v3_augment(
            warp_ycc, mean_ycc, target_ycc, trust, self.aug_cfg, force_hflip=force_flip
        )

        x = np.concatenate([warp_ycc, trust[..., None], mean_ycc], axis=-1)
        h, w = warp_ycc.shape[:2]
        to_chw = lambda a: torch.from_numpy(a.transpose(2, 0, 1)).contiguous().float()
        to_1chw = lambda a: torch.from_numpy(a)[None].contiguous().float()

        return {
            "inputs": to_chw(x),
            "base_init": to_chw(warp_ycc),
            "effective_mask": to_1chw(np.ones((h, w), dtype=np.float32)),
            "target": to_chw(target_ycc),
            "meta": {"sample_id": sid, "camera": cam, "aug_flip": force_flip},
        }
