from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from consensus_kit import (  # noqa: E402
    ConsensusConfig,
    _chw_npy_to_hwc_u8,
    _load_ego_mask,
    _resize_hw,
    _resize_maps,
)
from lib.lidar_density_mask import lidar_rife_blend_maps  # noqa: E402
from scripts.training.precompute_warp_notrust_mix import (  # noqa: E402
    MIRROR_CAMERAS,
    MixCfg,
    _f01_u8,
    _final_variant_n030,
    _flip_h,
)
from ya_paths import BAKED_ROOT, CV_ROOT, EGO_MASKS_APPROVED, TEST_SPLIT, TRAIN_SPLIT, WARPS_ROOT  # noqa: E402


def _sample_dirs(baked_root: Path) -> list[Path]:
    out: list[Path] = []
    for d in sorted(baked_root.iterdir()):
        if d.is_dir() and (d / "meta.json").is_file() and (d / "d1.npy").is_file():
            out.append(d)
    return out


def _split_dataset_root(split: str) -> Path:
    if split == "test":
        return TEST_SPLIT
    return TRAIN_SPLIT


def main() -> int:
    ap = argparse.ArgumentParser(description="Precompute lidar_trust + warp_mix for split in camera-space")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-skip-existing", action="store_true")
    args = ap.parse_args()

    cfg = MixCfg()
    hw = (cfg.image_h, cfg.image_w)
    split = args.split
    dataset_root = _split_dataset_root(split)
    baked_root = BAKED_ROOT / split
    warps_root = WARPS_ROOT / split
    trust_root = CV_ROOT / "precomputed_lidar_trust" / split
    mix_root = CV_ROOT / "side_warp_mix_v1" / split

    ds_cfg = ConsensusConfig(
        dataset_root=str(dataset_root),
        consensus_root=str(warps_root),
        baked_root=str(baked_root),
        rife_root=str(CV_ROOT / "rife_predictions_v5" / split),
        ego_masks_root=str(EGO_MASKS_APPROVED),
        image_h=cfg.image_h,
        image_w=cfg.image_w,
    )

    baked_dirs = _sample_dirs(baked_root)
    if args.limit > 0:
        baked_dirs = baked_dirs[: args.limit]
    if not baked_dirs:
        print("No baked samples found")
        return 1

    ok_n = 0
    skip_n = 0
    fail_n = 0
    trust_hit = 0
    trust_miss = 0

    for i, baked_dir in enumerate(baked_dirs, 1):
        try:
            meta = json.loads((baked_dir / "meta.json").read_text(encoding="utf-8"))
            sid = meta["sample_id"]
            cam = meta["camera"]
            src = Path(meta.get("source_dir", dataset_root / sid))

            out_mix = mix_root / sid / "warp_mix.jpg"
            out_trust = trust_root / sid / "lidar_trust.npy"
            warp_file = warps_root / sid / "consensus_raw.npy"
            cov_file = warps_root / sid / "coverage.npy"

            if out_mix.is_file() and out_trust.is_file() and not args.no_skip_existing:
                skip_n += 1
                trust_hit += 1
                continue

            if not (warp_file.is_file() and cov_file.is_file()):
                fail_n += 1
                continue

            warp_u8 = _chw_npy_to_hwc_u8(warp_file)
            cov = np.load(cov_file).astype(np.float32)
            warp_u8, cov, _ = _resize_maps(warp_u8, cov, cov, hw)

            t0 = np.array(Image.open(src / "input" / "t0" / f"{cam}.jpg").convert("RGB"))
            t1 = np.array(Image.open(src / "input" / "t1" / f"{cam}.jpg").convert("RGB"))
            mean_u8 = ((_resize_hw(t0, hw, cv2.INTER_LINEAR).astype(np.float32) + _resize_hw(t1, hw, cv2.INTER_LINEAR).astype(np.float32)) * 0.5).astype(np.uint8)

            art = _load_ego_mask(ds_cfg, sid, cam, hw).astype(np.float32)

            if out_trust.is_file():
                trust = np.load(out_trust).astype(np.float32)
                if trust.shape != hw:
                    trust = cv2.resize(trust, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
                trust = np.clip(trust, 0.0, 1.0)
                trust_hit += 1
            else:
                maps = lidar_rife_blend_maps(src, cam, hw, spread_radius_fine=3.0, spread_blur_fine=1.0, zone_min=0.12)
                trust = np.clip(maps["lidar_trust"].astype(np.float32), 0.0, 1.0)
                out_trust.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_trust, trust.astype(np.float32))
                trust_miss += 1

            warp = warp_u8.astype(np.float32) / 255.0
            mean = mean_u8.astype(np.float32) / 255.0
            mirrored = cam in MIRROR_CAMERAS
            if mirrored:
                warp, cov, mean, art, trust = _flip_h(warp, cov, mean, art, trust)

            seed = abs(hash(sid)) % (2**31 - 1)
            core = _final_variant_n030(warp, cov, cfg, seed)

            no_trust = (trust <= cfg.no_trust_thr).astype(np.float32)
            if cfg.no_trust_feather_px > 0:
                no_trust = cv2.GaussianBlur(no_trust.astype(np.float32), (0, 0), sigmaX=cfg.no_trust_feather_px, sigmaY=cfg.no_trust_feather_px)
                no_trust = np.clip(no_trust, 0.0, 1.0)
            mixed = core * (1.0 - cfg.no_trust_alpha * no_trust[..., None]) + mean * (cfg.no_trust_alpha * no_trust[..., None])
            mixed = mixed * (1.0 - art[..., None]) + mean * art[..., None]

            if mirrored:
                mixed = mixed[:, ::-1].copy()

            out_mix.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(_f01_u8(mixed)).save(out_mix, quality=92)
            ok_n += 1
        except Exception:
            fail_n += 1

        if i % 25 == 0 or i == len(baked_dirs):
            print(f"[{i}/{len(baked_dirs)}] ok={ok_n} skip={skip_n} fail={fail_n} trust_hit={trust_hit} trust_miss={trust_miss}")

    print(
        f"done split={split}: ok={ok_n} skip={skip_n} fail={fail_n} "
        f"trust_hit={trust_hit} trust_miss={trust_miss} mix_root={mix_root}"
    )
    return 0 if fail_n == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
