"""Refinement V2: 27ch input, warps, hybrid confidence base (consensus + RIFE)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

MAX_WARP_SLOTS = 6
IN_CHANNELS_V1 = 12
IN_CHANNELS_V2 = 27


def load_image_uint8(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _ensure_3ch(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1)
    return arr


def _resize_uint8_rgb(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    return np.array(Image.fromarray(arr).resize((w, h), Image.BILINEAR))


def _resize_depth(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    t = torch.from_numpy(arr.astype(np.float32))[None, None]
    t = F.interpolate(t, size=(h, w), mode="nearest")
    return t[0, 0].numpy()


def _resize_hw(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    t = torch.from_numpy(arr.astype(np.float32))[None, None]
    t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def _chw_rgb_to_hwc(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[0] == 3:
        return arr.transpose(1, 2, 0)
    return arr


def _load_rgb_float(path: Path) -> np.ndarray:
    return _ensure_3ch(load_image_uint8(str(path))).astype(np.float32) / 255.0


def _load_rgb_npy_float(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    arr = _chw_rgb_to_hwc(arr)
    return np.clip(arr, 0.0, 1.0)


def _load_scalar_npy(path: Path, vis_cap: float = 4.0) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if path.name == "visibility_count.npy":
        arr = np.clip(arr / max(vis_cap, 1e-6), 0.0, 1.0)
    else:
        arr = np.clip(arr, 0.0, 1.0)
    return arr


def _top_warp_slots(warp_dir: Path, k: int = MAX_WARP_SLOTS) -> List[Tuple[str, Path, Optional[Path]]]:
    masks = {p.stem.replace("mask_", ""): p for p in warp_dir.glob("mask_*.npy")}
    scored = []
    for wp in sorted(warp_dir.glob("warp_*.npy")):
        name = wp.stem.replace("warp_", "")
        mp = masks.get(name)
        score = float(np.load(mp).mean()) if mp and mp.is_file() else 0.0
        scored.append((score, name, wp, mp))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = [(n, w, m) for _, n, w, m in scored[:k]]
    while len(out) < k:
        out.append(("", Path(), None))
    return out


def resolve_source_dir(meta: dict, dataset_root: Path) -> Path:
    sid = meta["sample_id"]
    src = Path(meta["source_dir"]) if meta.get("source_dir") else dataset_root / sid
    if not src.is_dir():
        src = dataset_root / sid
    return src


def apply_sync_geom(
    arrays: Dict[str, np.ndarray],
    patch_size: Optional[int],
    augment: bool,
    fixed_crop: Optional[str],
) -> Dict[str, np.ndarray]:
    """Random crop / flip / rot90 applied identically to all maps."""
    keys = list(arrays.keys())
    ref = arrays[keys[0]]
    H, W = ref.shape[:2]

    y0 = x0 = 0
    if patch_size and (H > patch_size or W > patch_size):
        ph = pw = patch_size
        if fixed_crop == "center":
            y0 = (H - ph) // 2
            x0 = (W - pw) // 2
        else:
            y0 = random.randint(0, H - ph)
            x0 = random.randint(0, W - pw)
        for k in keys:
            a = arrays[k]
            if a.ndim == 3:
                arrays[k] = a[y0 : y0 + ph, x0 : x0 + pw]
            else:
                arrays[k] = a[y0 : y0 + ph, x0 : x0 + pw]

    if augment:
        if random.random() < 0.5:
            for k in keys:
                a = arrays[k]
                arrays[k] = a[:, ::-1].copy() if a.ndim == 3 else a[:, ::-1].copy()
        if random.random() < 0.5:
            for k in keys:
                a = arrays[k]
                arrays[k] = a[::-1].copy() if a.ndim == 3 else a[::-1].copy()
        h, w = arrays[keys[0]].shape[:2]
        if patch_size and h == w:
            k_rot = random.randint(0, 3)
            if k_rot:
                for k in keys:
                    arrays[k] = np.rot90(arrays[k], k_rot).copy()

    return arrays


def build_x_v1(I0: np.ndarray, Ir: np.ndarray, I2: np.ndarray,
               D0: np.ndarray, D1: np.ndarray, D2: np.ndarray) -> np.ndarray:
    """[12, H, W] — I₀, RIFE, I₂, D₀, D₁, D₂."""
    def to_chw(a):
        return a.transpose(2, 0, 1) if a.ndim == 3 else a[None]
    return np.concatenate(
        [to_chw(I0), to_chw(Ir), to_chw(I2), to_chw(D0), to_chw(D1), to_chw(D2)],
        axis=0,
    ).astype(np.float32)


def build_x_v2(
    I0: np.ndarray,
    I2: np.ndarray,
    D0: np.ndarray,
    D1: np.ndarray,
    D2: np.ndarray,
    warps: List[np.ndarray],
) -> np.ndarray:
    """
    [27, H, W]:
      I₀(3) + I₂(3) + D₀,D₁,D₂(3) + top-6 warps RGB(18)
    """
    def to_chw(a):
        return a.transpose(2, 0, 1) if a.ndim == 3 else a[None]
    parts = [to_chw(I0), to_chw(I2), to_chw(D0), to_chw(D1), to_chw(D2)]
    for w in warps:
        parts.append(to_chw(w))
    x = np.concatenate(parts, axis=0).astype(np.float32)
    assert x.shape[0] == IN_CHANNELS_V2, f"expected {IN_CHANNELS_V2} ch, got {x.shape[0]}"
    return x


def hybrid_base(
    rife: torch.Tensor,
    consensus: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    """[B,3,H,W] + [B,3,H,W] + [B,1,H,W] → hybrid anchor."""
    if confidence.dim() == 3:
        confidence = confidence.unsqueeze(1)
    return confidence * consensus + (1.0 - confidence) * rife


class RefineDataset(Dataset):
    """
    V2 (channel_layout='v2', default):
      x: [27,H,W] — I₀, I₂, depth, 6 warps
      rife, consensus_tuned, confidence, gt — для hybrid base + loss

    V1 ablation (channel_layout='v1'):
      x: [12,H,W] — I₀, RIFE, I₂, depth
      rife (= anchor), consensus_tuned, confidence, gt
    """

    def __init__(
        self,
        sample_dirs: List[Path],
        dataset_root: Path,
        consensus_root: Path,
        rife_root: Path,
        consensus_file: str = "consensus_tuned.npy",
        channel_layout: str = "v2",
        patch_size: Optional[int] = 256,
        augment: bool = True,
        target_h: int = 544,
        target_w: int = 1024,
        fixed_crop: Optional[str] = None,
        cache_full: bool = False,
        visibility_cap: float = 4.0,
    ):
        self.dirs = sample_dirs
        self.dataset_root = Path(dataset_root)
        self.consensus_root = Path(consensus_root)
        self.rife_root = Path(rife_root)
        self.consensus_file = consensus_file
        self.channel_layout = channel_layout
        self.patch_size = patch_size
        self.augment = augment
        self.target_h = target_h
        self.target_w = target_w
        self.fixed_crop = fixed_crop
        self.visibility_cap = visibility_cap
        self._full_cache: Optional[list] = None
        if cache_full and self.dirs:
            import time
            print(f"  RAM cache: preloading {len(self.dirs)} samples...", flush=True)
            t0 = time.time()
            self._full_cache = [self._load_sample(d) for d in self.dirs]
            print(f"  RAM cache: done in {time.time() - t0:.1f}s", flush=True)

    def __len__(self) -> int:
        return len(self.dirs)

    def _load_baked_meta(self, baked_dir: Path) -> dict:
        return json.loads((baked_dir / "meta.json").read_text(encoding="utf-8"))

    def _load_depth_npy(self, baked_dir: Path, stem: str) -> np.ndarray:
        dmap = np.load(baked_dir / f"{stem}.npy").astype(np.float32)
        valid = np.isfinite(dmap) & (dmap > 0)
        out = np.zeros(dmap.shape, dtype=np.float32)
        if not np.any(valid):
            return out
        lo, hi = float(dmap[valid].min()), float(dmap[valid].max())
        out[valid] = (dmap[valid] - lo) / max(hi - lo, 1e-6)
        return out

    def _load_sample(self, baked_dir: Path) -> dict:
        meta = self._load_baked_meta(baked_dir)
        sid = meta["sample_id"]
        cam = meta["camera"]
        src = resolve_source_dir(meta, self.dataset_root)
        warp_dir = self.consensus_root / sid

        I0 = _load_rgb_float(src / "input" / "t0" / f"{cam}.jpg")
        I2 = _load_rgb_float(src / "input" / "t1" / f"{cam}.jpg")
        GT = _load_rgb_float(src / "target" / f"{cam}.jpg")

        rife_path = self.rife_root / f"{sid}.jpg"
        if not rife_path.is_file():
            raise FileNotFoundError(f"RIFE not found: {rife_path}")
        Ir = _load_rgb_float(rife_path)

        cons_path = warp_dir / self.consensus_file
        if not cons_path.is_file():
            raise FileNotFoundError(f"consensus not found: {cons_path}")
        Icons = _load_rgb_npy_float(cons_path)

        conf_path = warp_dir / "confidence.npy"
        if not conf_path.is_file():
            raise FileNotFoundError(f"confidence not found: {conf_path}")
        confidence = _load_scalar_npy(conf_path)

        th, tw = self.target_h, self.target_w
        I0 = _resize_rgb_float(I0, th, tw)
        I2 = _resize_rgb_float(I2, th, tw)
        GT = _resize_rgb_float(GT, th, tw)
        Ir = _resize_rgb_float(Ir, th, tw)
        Icons = _resize_rgb_float(Icons, th, tw)
        confidence = _resize_hw(confidence, th, tw)

        D0 = _resize_depth(self._load_depth_npy(baked_dir, "d0"), th, tw)
        D1 = _resize_depth(self._load_depth_npy(baked_dir, "d1"), th, tw)
        D2 = _resize_depth(self._load_depth_npy(baked_dir, "d2"), th, tw)

        warp_slots = _top_warp_slots(warp_dir, MAX_WARP_SLOTS)
        warps = []
        for _, wp, _ in warp_slots:
            if wp.is_file():
                warps.append(_resize_rgb_float(_load_rgb_npy_float(wp), th, tw))
            else:
                warps.append(np.zeros((th, tw, 3), dtype=np.float32))

        return {
            "I0": I0, "I2": I2, "GT": GT, "Ir": Ir, "Icons": Icons,
            "confidence": confidence, "D0": D0, "D1": D1, "D2": D2, "warps": warps,
        }

    def __getitem__(self, idx: int) -> dict:
        if self._full_cache is not None:
            sample = {k: (v.copy() if isinstance(v, np.ndarray) else [w.copy() for w in v])
                        for k, v in self._full_cache[idx].items()}
        else:
            sample = self._load_sample(self.dirs[idx])

        arrays = {
            "I0": sample["I0"], "I2": sample["I2"], "GT": sample["GT"],
            "Ir": sample["Ir"], "Icons": sample["Icons"],
            "confidence": sample["confidence"],
            "D0": sample["D0"], "D1": sample["D1"], "D2": sample["D2"],
        }
        for i, w in enumerate(sample["warps"]):
            arrays[f"warp{i}"] = w

        arrays = apply_sync_geom(arrays, self.patch_size, self.augment, self.fixed_crop)

        I0, I2 = arrays["I0"], arrays["I2"]
        GT, Ir, Icons = arrays["GT"], arrays["Ir"], arrays["Icons"]
        confidence = arrays["confidence"]
        D0, D1, D2 = arrays["D0"], arrays["D1"], arrays["D2"]
        warps = [arrays[f"warp{i}"] for i in range(MAX_WARP_SLOTS)]

        if self.channel_layout == "v1":
            x = build_x_v1(I0, Ir, I2, D0, D1, D2)
        else:
            x = build_x_v2(I0, I2, D0, D1, D2, warps)

        to_t = lambda a: torch.from_numpy(a).float()
        to_chw = lambda a: torch.from_numpy(a.transpose(2, 0, 1).copy()).float()

        return {
            "x": to_t(x),
            "rife": to_chw(Ir),
            "consensus_tuned": to_chw(Icons),
            "confidence": to_t(confidence),
            "gt": to_chw(GT),
        }


def _resize_rgb_float(img: np.ndarray, h: int, w: int) -> np.ndarray:
    if img.shape[0] == h and img.shape[1] == w:
        return img
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return np.array(Image.fromarray(u8).resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0


def sample_asset_paths(
    baked_dir: Path,
    dataset_root: Path,
    consensus_root: Path,
    rife_root: Path,
    consensus_file: str = "consensus_tuned.npy",
) -> Optional[dict]:
    meta_path = baked_dir / "meta.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    sid = meta["sample_id"]
    cam = meta["camera"]
    src = resolve_source_dir(meta, Path(dataset_root))
    paths = {
        "i0": src / "input" / "t0" / f"{cam}.jpg",
        "i2": src / "input" / "t1" / f"{cam}.jpg",
        "gt": src / "target" / f"{cam}.jpg",
        "rife": Path(rife_root) / f"{sid}.jpg",
        "consensus": Path(consensus_root) / sid / consensus_file,
        "confidence": Path(consensus_root) / sid / "confidence.npy",
    }
    if not all(p.is_file() for p in paths.values()):
        return None
    depth_ok = all((baked_dir / f"d{i}.npy").is_file() for i in (0, 1, 2))
    warp_ok = any((Path(consensus_root) / sid).glob("warp_*.npy"))
    return paths if depth_ok and warp_ok else None


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


class RefineUNet(nn.Module):
    """V1: residual поверх фиксированного base_img (RIFE или consensus)."""

    def __init__(self, in_ch: int = 12, out_ch: int = 3, base: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)
        self.bottleneck = ConvBlock(base * 8, base * 8)
        self.dec4 = ConvBlock(base * 8 + base * 8, base * 4)
        self.dec3 = ConvBlock(base * 4 + base * 4, base * 2)
        self.dec2 = ConvBlock(base * 2 + base * 2, base)
        self.dec1 = ConvBlock(base + base, base)
        self.out = nn.Conv2d(base, out_ch, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return self.out(d1)

    def forward(self, x: torch.Tensor, base_img: torch.Tensor) -> torch.Tensor:
        return (base_img + self._decode(x)).clamp(0, 1)


class RefineUNetV2(nn.Module):
    """27ch in, hybrid base = conf*consensus + (1-conf)*RIFE + residual."""

    def __init__(self, in_ch: int = 27, out_ch: int = 3, base: int = 32, use_hybrid_base: bool = True):
        super().__init__()
        self.use_hybrid_base = use_hybrid_base
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)
        self.bottleneck = ConvBlock(base * 8, base * 8)
        self.dec4 = ConvBlock(base * 8 + base * 8, base * 4)
        self.dec3 = ConvBlock(base * 4 + base * 4, base * 2)
        self.dec2 = ConvBlock(base * 2 + base * 2, base)
        self.dec1 = ConvBlock(base + base, base)
        self.out = nn.Conv2d(base, out_ch, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return self.out(d1)

    def forward(
        self,
        x: torch.Tensor,
        rife_img: torch.Tensor,
        consensus_tuned: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        residual = self._decode(x)
        if self.use_hybrid_base:
            base = hybrid_base(rife_img, consensus_tuned, confidence)
        else:
            base = rife_img
        return (base + residual).clamp(0, 1)


def model_predict(
    model: nn.Module,
    batch: dict,
    base_mode: str = "hybrid",
) -> torch.Tensor:
    """Unified forward for V2 / V1 / legacy (MARNet)."""
    x = batch["x"]
    if isinstance(model, RefineUNetV2):
        conf = batch["confidence"]
        if conf.dim() == 3:
            conf = conf.unsqueeze(1)
        return model(x, batch["rife"], batch["consensus_tuned"], conf)
    if isinstance(model, RefineUNet):
        if base_mode == "consensus":
            base = batch["consensus_tuned"]
        else:
            base = batch["rife"]
        return model(x, base)
    # MARNet и др. legacy
    base = batch["consensus_tuned"] if base_mode == "consensus" else batch["rife"]
    return model(x, base)


def hybrid_base_from_batch(batch: dict) -> torch.Tensor:
    conf = batch["confidence"]
    if conf.dim() == 3:
        conf = conf.unsqueeze(1)
    return hybrid_base(batch["rife"], batch["consensus_tuned"], conf)
