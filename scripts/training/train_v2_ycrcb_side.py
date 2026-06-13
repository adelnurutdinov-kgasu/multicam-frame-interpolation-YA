from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from consensus_kit import ConsensusConfig, ConsensusUNet, discover_samples
from ya_paths import BAKED_ROOT, CV_ROOT, EGO_MASKS_APPROVED, TRAIN_SPLIT, WARPS_ROOT

SIDE_CAMERAS = ("left_fwd", "right_fwd", "left_bwd", "right_bwd")
MIRROR_CAMERAS = ("left_fwd", "right_bwd")


@dataclass
class Cfg:
    image_h: int = 544
    image_w: int = 1024
    in_channels: int = 7
    base_channels: int = 32
    batch_size: int = 12
    num_workers: int = 0
    lr: float = 2e-4
    epochs: int = 3
    val_fraction: float = 0.16
    seed: int = 42
    trust_cache_root: Path = CV_ROOT / "precomputed_lidar_trust" / "train"
    dataset_root: Path = TRAIN_SPLIT
    baked_root: Path = BAKED_ROOT / "train"
    warps_root: Path = WARPS_ROOT / "train"


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_rgb(path: Path, hw: tuple[int, int]) -> np.ndarray:
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    h, w = hw
    if arr.shape[:2] != (h, w):
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
    return arr


def load_warp(path: Path, hw: tuple[int, int]) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.clip(arr.transpose(1, 2, 0), 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    h, w = hw
    if arr.shape[:2] != (h, w):
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
    return arr


def to_ycrcb_u8(rgb_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2YCrCb)


def flip_h(*arrays):
    return tuple(a[:, ::-1].copy() for a in arrays)


class DS(Dataset):
    def __init__(self, dirs: list[Path], cfg: Cfg, color_space: str = "ycrcb"):
        self.dirs = dirs
        self.cfg = cfg
        self.color_space = color_space.lower()

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        baked_dir = self.dirs[idx]
        meta = json.loads((baked_dir / "meta.json").read_text(encoding="utf-8"))
        sid = meta["sample_id"]
        cam = meta["camera"]
        src = Path(meta.get("source_dir", self.cfg.dataset_root / sid))
        hw = (self.cfg.image_h, self.cfg.image_w)

        warp_rgb = load_warp(self.cfg.warps_root / sid / "consensus_raw.npy", hw)
        t0 = load_rgb(src / "input" / "t0" / f"{cam}.jpg", hw)
        t1 = load_rgb(src / "input" / "t1" / f"{cam}.jpg", hw)
        target_rgb = load_rgb(src / "target" / f"{cam}.jpg", hw)
        mean_rgb = ((t0.astype(np.float32) + t1.astype(np.float32)) * 0.5).astype(np.uint8)

        trust = np.load(self.cfg.trust_cache_root / sid / "lidar_trust.npy").astype(np.float32)
        if trust.shape != hw:
            trust = cv2.resize(trust, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
        trust = np.clip(trust, 0.0, 1.0)

        if self.color_space == "rgb":
            warp_ycc = warp_rgb.astype(np.float32) / 255.0
            mean_ycc = mean_rgb.astype(np.float32) / 255.0
            target_ycc = target_rgb.astype(np.float32) / 255.0
        else:
            warp_ycc = to_ycrcb_u8(warp_rgb).astype(np.float32) / 255.0
            mean_ycc = to_ycrcb_u8(mean_rgb).astype(np.float32) / 255.0
            target_ycc = to_ycrcb_u8(target_rgb).astype(np.float32) / 255.0

        if cam in MIRROR_CAMERAS:
            warp_ycc, mean_ycc, target_ycc, trust = flip_h(warp_ycc, mean_ycc, target_ycc, trust)

        x = np.concatenate([warp_ycc, trust[..., None], mean_ycc], axis=-1)
        to_chw = lambda a: torch.from_numpy(a.transpose(2, 0, 1)).contiguous().float()
        to_1chw = lambda a: torch.from_numpy(a)[None].contiguous().float()

        return {
            "inputs": to_chw(x),
            "base_init": to_chw(warp_ycc),
            "effective_mask": to_1chw(np.ones(hw, dtype=np.float32)),
            "target": to_chw(target_ycc),
        }


def run_epoch(model, loader, device, criterion, opt=None):
    train = opt is not None
    model.train(train)
    losses = []
    for b in loader:
        inp = b["inputs"].to(device)
        base = b["base_init"].to(device)
        msk = b["effective_mask"].to(device)
        tgt = b["target"].to(device)
        with torch.set_grad_enabled(train):
            pred = model(inp, msk, base)["pred"].float()
            loss = criterion(pred, tgt)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--color-space", choices=["ycrcb", "rgb"], default="ycrcb")
    args = ap.parse_args()

    cfg = Cfg(batch_size=args.batch_size, epochs=args.epochs, seed=args.seed)
    seed_everything(cfg.seed)

    disc_cfg = ConsensusConfig(
        dataset_root=str(cfg.dataset_root),
        consensus_root=str(cfg.warps_root),
        baked_root=str(cfg.baked_root),
        rife_root=str(CV_ROOT / "rife_predictions_v5" / "train"),
        ego_masks_root=str(EGO_MASKS_APPROVED),
        allowed_cameras=SIDE_CAMERAS,
        image_h=cfg.image_h,
        image_w=cfg.image_w,
    )
    all_samples = discover_samples(disc_cfg)

    ready = []
    for bdir in all_samples:
        meta = json.loads((bdir / "meta.json").read_text(encoding="utf-8"))
        sid = meta["sample_id"]
        if (cfg.trust_cache_root / sid / "lidar_trust.npy").is_file():
            ready.append(bdir)
    if len(ready) < args.limit:
        print(f"Недостаточно готовых trust cache: {len(ready)} < {args.limit}")
        return 1

    rng = np.random.RandomState(cfg.seed)
    idx = np.arange(len(ready))
    rng.shuffle(idx)
    chosen = [ready[i] for i in idx[: args.limit]]

    n_val = max(1, int(args.limit * cfg.val_fraction))
    val_dirs = chosen[:n_val]
    train_dirs = chosen[n_val:]

    train_loader = DataLoader(
        DS(train_dirs, cfg, color_space=args.color_space),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        DS(val_dirs, cfg, color_space=args.color_space),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = ConsensusUNet(in_ch=cfg.in_channels, base=cfg.base_channels, predict_confidence=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = torch.nn.MSELoss()

    print(
        f"start: samples={args.limit} train={len(train_dirs)} val={len(val_dirs)} "
        f"device={device} color_space={args.color_space}"
    )
    best = 1e9
    out_dir = Path("artifacts/checkpoints/consensus_side_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"consensus_v2_side_100_{args.color_space}_best.pt"
    for ep in range(1, cfg.epochs + 1):
        tr = run_epoch(model, train_loader, device, criterion, opt=opt)
        va = run_epoch(model, val_loader, device, criterion, opt=None)
        print(f"epoch {ep:02d}: train_mse={tr:.6f} val_mse={va:.6f}")
        if va < best:
            best = va
            torch.save(
                {
                    "epoch": ep,
                    "model": model.state_dict(),
                    "cfg": vars(cfg),
                    "best_val_mse": best,
                    "samples": args.limit,
                    "color_space": args.color_space,
                    "inputs": ["warp(3)", "lidar_trust(1)", "mean_t0t1(3)"],
                    "target": "target(3)",
                    "loss": "MSE",
                },
                ckpt,
            )
            print(f"  saved best -> {ckpt}")
    print(f"done best_val_mse={best:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

