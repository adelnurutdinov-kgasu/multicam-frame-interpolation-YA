"""Precompute train warp_mix in camera-space with caching.

Pipeline (clean):
1) Load camera-space inputs (warp/cov/mean/art/lidar_trust).
2) Mirror ONLY for processing for side canonical cameras.
3) Build fixed variant: telea_med5_outlier_soft_prefill_n030.
4) Blend with mean 50/50 in no-trust areas with feathered mask.
5) Overlay camera artifact from mean in ego-mask area.
6) Un-mirror back to camera-space and save final result only.

Saved artifacts:
- <out-root>/<split>/<sample_id>/warp_mix.jpg  (final camera-space result)
- <trust-cache>/<split>/<sample_id>/lidar_trust.npy (cached if missing)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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
    _edge_contrast_mask,
    _load_ego_mask,
    _local_median_rgb01,
    _resize_hw,
    _resize_maps,
    selective_median_outlier_fix,
)
from lib.lidar_density_mask import lidar_rife_blend_maps  # noqa: E402
from ya_paths import BAKED_ROOT, CV_ROOT, EGO_MASKS_APPROVED, TRAIN_SPLIT, WARPS_ROOT  # noqa: E402

MIRROR_CAMERAS = ("left_fwd", "right_bwd")
OUT_ROOT_DEFAULT = CV_ROOT / "side_warp_mix_v1"
TRUST_CACHE_DEFAULT = CV_ROOT / "precomputed_lidar_trust"


@dataclass
class MixCfg:
    image_h: int = 544
    image_w: int = 1024
    warp_expand_px: float = 6.0
    no_trust_alpha: float = 0.5
    no_trust_thr: float = 1e-4
    no_trust_feather_px: float = 2.0
    telea_contrast_thr: float = 0.10
    telea_outlier_thr: float = 0.045
    prefill_noise_std: float = 0.03  # n030
    prefill_black_thr: float = 0.05
    prefill_black_lift_floor: float = 0.24


def _flip_h(*arrays):
    return tuple(a[:, ::-1].copy() for a in arrays)


def _f01_u8(a: np.ndarray) -> np.ndarray:
    return (np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)


def _hole_mask_with_radius(cov: np.ndarray, radius_px: float) -> np.ndarray:
    valid = (cov > 1e-4).astype(np.uint8)
    if valid.max() == 0:
        return np.ones_like(valid, dtype=np.uint8)
    if valid.min() == 1:
        return np.zeros_like(valid, dtype=np.uint8)
    inv = (valid == 0).astype(np.uint8)
    if radius_px <= 0:
        return inv
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    return ((inv == 1) & (dist <= float(radius_px))).astype(np.uint8)


def _telea_expand(warp: np.ndarray, cov: np.ndarray, radius_px: float) -> np.ndarray:
    fill_mask = _hole_mask_with_radius(cov, radius_px)
    if fill_mask.max() == 0:
        return warp
    u8 = _f01_u8(warp)
    out = np.empty_like(u8)
    inpr = max(1.0, float(radius_px))
    for c in range(3):
        out[..., c] = cv2.inpaint(u8[..., c], fill_mask, inpr, cv2.INPAINT_TELEA)
    return out.astype(np.float32) / 255.0


def _prefill_holes_mean_noise(
    base: np.ndarray,
    cov: np.ndarray,
    noise_std: float,
    black_thr: float,
    black_lift_floor: float,
    seed: int,
) -> np.ndarray:
    holes = (np.clip(cov, 0.0, 1.0) <= 1e-4)
    gray = np.mean(np.clip(base, 0.0, 1.0), axis=-1)
    black = gray <= float(black_thr)
    fill_mask = holes | black
    if not np.any(fill_mask):
        return base

    out = base.copy()
    rng = np.random.RandomState(seed)
    mu = cv2.blur(np.clip(base, 0.0, 1.0).astype(np.float32), (5, 5))

    valid = np.clip(cov, 0.0, 1.0) > 1e-4
    ref = base[valid]
    if ref.shape[0] < 64:
        ref = base.reshape(-1, 3)
    ref = np.clip(ref.astype(np.float32), 0.0, 1.0)
    ref_centered = ref - ref.mean(axis=0, keepdims=True)
    cov_rgb = (ref_centered.T @ ref_centered) / max(1, ref_centered.shape[0] - 1)
    tr = float(np.trace(cov_rgb))
    if tr > 1e-8:
        cov_rgb = cov_rgb / tr
    cov_rgb = cov_rgb * (float(noise_std) ** 2) * 3.0 + np.eye(3, dtype=np.float32) * 1e-6

    idx = np.argwhere(fill_mask)
    if idx.size > 0:
        z = rng.multivariate_normal(np.zeros(3, np.float32), cov_rgb.astype(np.float64), size=idx.shape[0]).astype(
            np.float32
        )
        yy, xx = idx[:, 0], idx[:, 1]
        out[yy, xx] = np.clip(mu[yy, xx] + z, 0.0, 1.0)

    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    out_gray = np.mean(out, axis=-1)
    force = black & (out_gray < float(black_lift_floor))
    if np.any(force):
        scale = np.ones_like(out_gray, dtype=np.float32)
        scale[force] = float(black_lift_floor) / np.clip(out_gray[force], 1e-4, None)
        out = np.clip(out * scale[..., None], 0.0, 1.0)
    return out.astype(np.float32)


def _telea_med5_outlier_soft(base: np.ndarray, cov: np.ndarray, contrast_thr: float, outlier_thr: float) -> np.ndarray:
    edge = _edge_contrast_mask(base, contrast_thr)
    valid = (cov > 1e-4).astype(np.float32)
    region_edge = np.clip(0.85 * edge * valid, 0.0, 1.0)
    region_edge = cv2.GaussianBlur(region_edge.astype(np.float32), (0, 0), sigmaX=0.8, sigmaY=0.8)
    region_edge = np.clip(region_edge, 0.0, 1.0)
    med5 = _local_median_rgb01(base, 5)
    return selective_median_outlier_fix(
        base,
        med5,
        region_edge,
        outlier_thr=outlier_thr * 1.35,
        never_darken=True,
        cov=cov,
        min_cov=0.08,
    )


def _final_variant_n030(warp: np.ndarray, cov: np.ndarray, cfg: MixCfg, seed: int) -> np.ndarray:
    telea = _telea_expand(warp, cov, cfg.warp_expand_px)
    prefill = _prefill_holes_mean_noise(
        telea,
        cov,
        noise_std=cfg.prefill_noise_std,
        black_thr=cfg.prefill_black_thr,
        black_lift_floor=cfg.prefill_black_lift_floor,
        seed=seed,
    )
    return _telea_med5_outlier_soft(
        prefill,
        cov,
        contrast_thr=cfg.telea_contrast_thr,
        outlier_thr=cfg.telea_outlier_thr,
    )


def _sample_dirs(baked_root: Path) -> list[Path]:
    out = []
    for d in sorted(baked_root.iterdir()):
        if d.is_dir() and (d / "meta.json").is_file() and (d / "d1.npy").is_file():
            out.append(d)
    return out


def _load_or_build_lidar_trust(
    src_dir: Path,
    cam: str,
    hw: tuple[int, int],
    cache_path: Path,
) -> np.ndarray:
    if cache_path.is_file():
        arr = np.load(cache_path).astype(np.float32)
        if arr.shape == hw:
            return np.clip(arr, 0.0, 1.0)
    maps = lidar_rife_blend_maps(src_dir, cam, hw, spread_radius_fine=3.0, spread_blur_fine=1.0, zone_min=0.12)
    trust = np.clip(maps["lidar_trust"].astype(np.float32), 0.0, 1.0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, trust.astype(np.float32))
    return trust


def main() -> int:
    ap = argparse.ArgumentParser(description="Precompute fixed side warp mix for train in camera-space")
    ap.add_argument("--split", default="train", choices=["train"])
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT_DEFAULT)
    ap.add_argument("--trust-cache-root", type=Path, default=TRUST_CACHE_DEFAULT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-skip-existing", action="store_true")
    args = ap.parse_args()

    cfg = MixCfg()
    hw = (cfg.image_h, cfg.image_w)
    baked_root = BAKED_ROOT / args.split
    warps_root = WARPS_ROOT / args.split
    out_root = args.out_root / args.split
    trust_root = args.trust_cache_root / args.split

    ds_cfg = ConsensusConfig(
        dataset_root=str(TRAIN_SPLIT),
        consensus_root=str(warps_root),
        baked_root=str(baked_root),
        rife_root=str(CV_ROOT / "rife_predictions_v5" / args.split),
        ego_masks_root=str(EGO_MASKS_APPROVED),
        image_h=cfg.image_h,
        image_w=cfg.image_w,
    )

    baked_dirs = _sample_dirs(baked_root)
    if args.limit > 0:
        baked_dirs = baked_dirs[: args.limit]
    if not baked_dirs:
        print("No baked samples found.")
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
            src = Path(meta.get("source_dir", TRAIN_SPLIT / sid))
            out_dir = out_root / sid
            out_file = out_dir / "warp_mix.jpg"
            trust_file = trust_root / sid / "lidar_trust.npy"

            if out_file.is_file() and not args.no_skip_existing:
                skip_n += 1
                if trust_file.is_file():
                    trust_hit += 1
                continue

            warp_file = warps_root / sid / "consensus_raw.npy"
            cov_file = warps_root / sid / "coverage.npy"
            if not (warp_file.is_file() and cov_file.is_file()):
                fail_n += 1
                continue

            warp_u8 = _chw_npy_to_hwc_u8(warp_file)
            cov = np.load(cov_file).astype(np.float32)
            warp_u8, cov, _ = _resize_maps(warp_u8, cov, cov, hw)
            mean_u8 = (
                (
                    _resize_hw(np.array(Image.open(src / "input" / "t0" / f"{cam}.jpg").convert("RGB")), hw, cv2.INTER_LINEAR).astype(np.float32)
                    + _resize_hw(np.array(Image.open(src / "input" / "t1" / f"{cam}.jpg").convert("RGB")), hw, cv2.INTER_LINEAR).astype(np.float32)
                )
                * 0.5
            ).astype(np.uint8)
            art = _load_ego_mask(ds_cfg, sid, cam, hw).astype(np.float32)
            trust_exists = trust_file.is_file()
            trust = _load_or_build_lidar_trust(src, cam, hw, trust_file)
            trust_hit += int(trust_exists)
            trust_miss += int(not trust_exists)

            warp = warp_u8.astype(np.float32) / 255.0
            mean = mean_u8.astype(np.float32) / 255.0
            mirrored = cam in MIRROR_CAMERAS
            if mirrored:
                warp, cov, mean, art, trust = _flip_h(warp, cov, mean, art, trust)

            seed = abs(hash(sid)) % (2**31 - 1)
            core = _final_variant_n030(warp, cov, cfg, seed)

            # no-trust 50/50 blend with feather
            no_trust = (trust <= cfg.no_trust_thr).astype(np.float32)
            if cfg.no_trust_feather_px > 0:
                no_trust = cv2.GaussianBlur(
                    no_trust.astype(np.float32), (0, 0), sigmaX=cfg.no_trust_feather_px, sigmaY=cfg.no_trust_feather_px
                )
                no_trust = np.clip(no_trust, 0.0, 1.0)
            mixed = core * (1.0 - cfg.no_trust_alpha * no_trust[..., None]) + mean * (cfg.no_trust_alpha * no_trust[..., None])

            # camera artifact overlay from mean
            mixed = mixed * (1.0 - art[..., None]) + mean * art[..., None]

            if mirrored:
                mixed = mixed[:, ::-1].copy()

            out_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(_f01_u8(mixed)).save(out_file, quality=92)
            ok_n += 1
        except Exception:
            fail_n += 1

        if i % 25 == 0 or i == len(baked_dirs):
            print(
                f"[{i}/{len(baked_dirs)}] ok={ok_n} skip={skip_n} fail={fail_n} "
                f"trust_hit={trust_hit} trust_miss={trust_miss}"
            )

    print(
        f"done: ok={ok_n} skip={skip_n} fail={fail_n} "
        f"trust_hit={trust_hit} trust_miss={trust_miss} out={out_root}"
    )
    return 0 if fail_n == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

