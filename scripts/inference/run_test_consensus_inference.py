"""
Инференс consensus U-Net на test split (без GT).

- front / rear  → checkpoints_consensus/consensus_unet_best.pt
- боковые     → checkpoints_consensus_side/consensus_unet_side_best.pt
- left_fwd, right_bwd: flip по горизонтали до сети, pred обратно (как consensus_training_side)

Опционально: blend model + RIFE по LiDAR-маске.

Usage:
  python scripts/inference/run_test_consensus_inference.py --limit 5
  python scripts/inference/run_test_consensus_inference.py --blend-alpha 0.55
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ya_paths import (  # noqa: E402
    BAKED_ROOT,
    CKPT_CONSENSUS,
    CKPT_CONSENSUS_SIDE,
    CONSENSUS_TEST_OUT,
    CV_ROOT,
    DATASET_ROOT,
    EGO_MASKS_APPROVED,
    RIFE_ROOT,
    WARPS_ROOT,
)


import cv2
import numpy as np
import torch
from PIL import Image



from consensus_kit import (  # noqa: E402
    ConsensusConfig,
    build_base_init,
    load_consensus_model,
    _chw_npy_to_hwc_u8,
    _load_ego_mask,
    _norm_depth,
    _resize_hw,
    _resize_maps,
)
from lidar_density_mask import blend_model_rife_lidar, lidar_rife_blend_maps  # noqa: E402

FRONT_REAR_CKPT = CKPT_CONSENSUS
SIDE_CKPT = CKPT_CONSENSUS_SIDE
DEFAULT_OUT = CONSENSUS_TEST_OUT

FRONT_REAR_CAMERAS = ("front", "rear")
SIDE_CAMERAS = ("left_fwd", "right_fwd", "left_bwd", "right_bwd")
MIRROR_CAMERAS = ("left_fwd", "right_bwd")


def flip_hwc(a: np.ndarray) -> np.ndarray:
    return a[:, ::-1].copy()


OUTPUT_VERSION = 6  # v6: lidar trust / rife всегда в image space камеры (без flip)


def needs_blend_refresh(out_dir: Path, meta_path: Path, no_blend: bool) -> bool:
    if no_blend:
        return False
    if not (out_dir / "blend_lidar_rife.jpg").is_file():
        return True
    if not (out_dir / "lidar_density_fine.png").is_file():
        return True
    if not meta_path.is_file():
        return True
    try:
        return json.loads(meta_path.read_text(encoding="utf-8")).get("output_version", 0) < OUTPUT_VERSION
    except json.JSONDecodeError:
        return True


def save_trust_maps(out_dir: Path, maps: dict, mask_thr: float) -> None:
    """Как lidar_rife_blend.ipynb: r3+blur = density_fine, trust, бинарная mask."""
    fine = np.clip(maps["density_fine"].astype(np.float32), 0.0, 1.0)
    trust = np.clip(maps["lidar_trust"].astype(np.float32), 0.0, 1.0)
    blend = maps["blend_mask"].astype(np.float32)
    Image.fromarray((fine * 255).astype(np.uint8)).save(out_dir / "lidar_density_fine.png")
    Image.fromarray((trust * 255).astype(np.uint8)).save(out_dir / "lidar_trust.png")
    Image.fromarray((blend * 255).astype(np.uint8)).save(out_dir / "lidar_blend_mask.png")


def test_config(split: str = "test") -> ConsensusConfig:
    return ConsensusConfig(
        dataset_root=str(DATASET_ROOT / split),
        consensus_root=str(WARPS_ROOT / split),
        baked_root=str(BAKED_ROOT / split),
        rife_root=str(RIFE_ROOT / split),
        ego_masks_root=str(EGO_MASKS_APPROVED),
    )


def discover_test_baked(cfg: ConsensusConfig) -> list[Path]:
    baked_root = Path(cfg.baked_root)
    consensus_root = Path(cfg.consensus_root)
    out: list[Path] = []
    for baked_dir in sorted(baked_root.iterdir()):
        if not baked_dir.is_dir() or not (baked_dir / "meta.json").is_file():
            continue
        meta = json.loads((baked_dir / "meta.json").read_text(encoding="utf-8"))
        cam = meta["camera"]
        sid = meta["sample_id"]
        src = Path(meta.get("source_dir", Path(cfg.dataset_root) / sid))
        warp_dir = consensus_root / sid
        ok = (
            (warp_dir / cfg.consensus_file).is_file()
            and (warp_dir / "coverage.npy").is_file()
            and (baked_dir / "d1.npy").is_file()
            and (src / "input" / "lidar.npz").is_file()
            and (src / "input" / "t0" / f"{cam}.jpg").is_file()
            and (src / "input" / "t1" / f"{cam}.jpg").is_file()
        )
        if ok:
            out.append(baked_dir)
    return out


def load_rgb(path: Path, hw: tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        return np.zeros((hw[0], hw[1], 3), dtype=np.uint8)
    return _resize_hw(np.array(Image.open(path).convert("RGB")), hw, cv2.INTER_LINEAR)


def prepare_sample(baked_dir: Path, cfg: ConsensusConfig, mirror: bool) -> dict:
    meta = json.loads((baked_dir / "meta.json").read_text(encoding="utf-8"))
    sid = meta["sample_id"]
    cam = meta["camera"]
    src = Path(meta.get("source_dir", Path(cfg.dataset_root) / sid))
    hw = (cfg.image_h, cfg.image_w)
    warp_dir = Path(cfg.consensus_root) / sid

    warp_rgb = _chw_npy_to_hwc_u8(warp_dir / cfg.consensus_file)
    coverage = np.load(warp_dir / "coverage.npy").astype(np.float32)
    depth_raw = np.load(baked_dir / "d1.npy").astype(np.float32)
    warp_rgb, coverage, depth_raw = _resize_maps(warp_rgb, coverage, depth_raw, hw)

    tgt_path = src / "target" / f"{cam}.jpg"
    ref_path = tgt_path if tgt_path.is_file() else src / "input" / "t0" / f"{cam}.jpg"

    t0 = load_rgb(src / "input" / "t0" / f"{cam}.jpg", hw)
    t1 = load_rgb(src / "input" / "t1" / f"{cam}.jpg", hw)
    mean_t0t1 = ((t0.astype(np.float32) + t1.astype(np.float32)) * 0.5).astype(np.uint8)
    rife_rgb = load_rgb(Path(cfg.rife_root) / f"{sid}.jpg", hw)
    ref_rgb = load_rgb(ref_path, hw)
    art_mask = _load_ego_mask(cfg, sid, cam, hw)

    H, W = hw
    sf_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    sf_mask = np.zeros((H, W), dtype=np.float32)

    # LiDAR trust и RIFE — только для blend, в image space камеры (не flip).
    rife_cam = rife_rgb.copy()
    ref_cam = ref_rgb.copy()
    lidar_maps = lidar_rife_blend_maps(
        src,
        cam,
        hw,
        spread_radius_fine=cfg.lidar_spread_radius_fine,
        spread_blur_fine=cfg.lidar_spread_blur_fine,
        zone_min=cfg.lidar_zone_min,
    )

    if mirror:
        warp_rgb = flip_hwc(warp_rgb)
        coverage = flip_hwc(coverage)
        depth_raw = flip_hwc(depth_raw)
        sf_rgb = flip_hwc(sf_rgb)
        sf_mask = flip_hwc(sf_mask)
        art_mask = flip_hwc(art_mask)
        mean_t0t1 = flip_hwc(mean_t0t1)

    warp = warp_rgb.astype(np.float32) / 255.0
    cov = np.clip(coverage, 0.0, 1.0)
    depth_n = _norm_depth(depth_raw)
    sf_rgb_f = sf_rgb.astype(np.float32) / 255.0
    sf_mask = np.clip(sf_mask, 0.0, 1.0)
    art_mask = np.clip(art_mask, 0.0, 1.0)
    mean_t0t1_f = mean_t0t1.astype(np.float32) / 255.0
    rife = rife_cam.astype(np.float32) / 255.0

    base_init, base_mask = build_base_init(
        warp, cov, sf_rgb_f, sf_mask, art_mask, mean_t0t1_f, cfg.use_anchor_frames
    )
    mean_in = mean_t0t1_f * (1.0 - art_mask[..., None])
    eff_mask = np.ones_like(base_mask) if cfg.use_anchor_frames else base_mask

    to_chw_3 = lambda a: torch.from_numpy(a.transpose(2, 0, 1)).contiguous().float()
    to_chw_1 = lambda a: torch.from_numpy(a)[None].contiguous().float()
    parts = [
        to_chw_3(warp),
        to_chw_1(cov),
        to_chw_1(depth_n),
        to_chw_3(sf_rgb_f),
        to_chw_1(sf_mask),
        to_chw_1(art_mask),
    ]
    if cfg.use_anchor_frames:
        parts.append(to_chw_3(mean_in))
    inputs = torch.cat(parts, dim=0)

    return {
        "meta": meta,
        "camera": cam,
        "mirrored": mirror,
        "inputs": inputs,
        "effective_mask": to_chw_1(eff_mask),
        "base_init": to_chw_3(base_init),
        "rife": rife,
        "trust": lidar_maps["lidar_trust"],
        "lidar_maps": lidar_maps,
        "ref_rgb": ref_cam,
        "src": src,
    }


@torch.no_grad()
def run_model(model, batch: dict, device: torch.device) -> np.ndarray:
    inp = batch["inputs"].unsqueeze(0).to(device)
    msk = batch["effective_mask"].unsqueeze(0).to(device)
    base = batch["base_init"].unsqueeze(0).to(device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        pred = model(inp, msk, base)["pred"].float()
    out = pred[0].detach().cpu().permute(1, 2, 0).numpy()
    if batch["mirrored"]:
        out = out[:, ::-1].copy()
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def save_u8(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path, quality=92)


def save_consensus_raw(path: Path, pred: np.ndarray) -> None:
    """Сохраняем float32 [0..1], HWC, без JPEG-квантизации."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.clip(pred, 0.0, 1.0).astype(np.float32))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--blend-alpha", type=float, default=0.55)
    ap.add_argument("--rife-blur-sigma", type=float, default=0.8, help="слабый blur RIFE перед blend")
    ap.add_argument("--no-blend", action="store_true")
    ap.add_argument("--reblend-only", action="store_true", help="только blend/mask/rife; consensus_model.jpg не трогать")
    ap.add_argument("--force-infer", action="store_true", help="пересчитать U-Net даже если consensus_model.jpg есть")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = test_config(args.split)
    baked_dirs = discover_test_baked(cfg)
    if args.limit > 0:
        baked_dirs = baked_dirs[: args.limit]
    if not baked_dirs:
        print("Нет готовых test-сэмплов (bake + warps + lidar + t0/t1)")
        return 1

    if not FRONT_REAR_CKPT.is_file():
        print(f"Нет {FRONT_REAR_CKPT}")
        return 1
    if not SIDE_CKPT.is_file():
        print(f"Нет {SIDE_CKPT}")
        return 1

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"device={device}  samples={len(baked_dirs)}  out={args.out_dir}")

    model_fr, _ = load_consensus_model(FRONT_REAR_CKPT, device, cfg=cfg)
    model_side, _ = load_consensus_model(SIDE_CKPT, device, cfg=cfg)

    ok_n, skip_n, fail_n = 0, 0, 0
    t0 = time.perf_counter()

    for i, baked_dir in enumerate(baked_dirs, 1):
        meta = json.loads((baked_dir / "meta.json").read_text(encoding="utf-8"))
        sid = meta["sample_id"]
        cam = meta["camera"]
        out_dir = args.out_dir / sid
        done_flag = out_dir / "consensus_model.jpg"
        raw_flag = out_dir / "consensus_raw.npy"
        blend_flag = out_dir / "blend_lidar_rife.jpg"
        meta_path = out_dir / "meta_infer.json"
        has_consensus = done_flag.is_file()
        if args.reblend_only and not has_consensus:
            fail_n += 1
            continue
        if args.skip_existing and has_consensus and not args.force_infer:
            if not needs_blend_refresh(out_dir, meta_path, args.no_blend):
                skip_n += 1
                continue

        mirror = cam in MIRROR_CAMERAS
        model = model_side if cam in SIDE_CAMERAS else model_fr
        ckpt_name = "side" if cam in SIDE_CAMERAS else "front_rear"

        try:
            batch = prepare_sample(baked_dir, cfg, mirror=mirror)
            if has_consensus and not args.force_infer:
                pred = np.array(Image.open(done_flag).convert("RGB"), dtype=np.float32) / 255.0
            else:
                pred = run_model(model, batch, device)
                save_u8(done_flag, pred)
            save_consensus_raw(raw_flag, pred)

            # pred развёрнут в run_model; rife/trust уже в image space камеры
            rife_img = batch["rife"]
            trust_img = batch["trust"]
            maps_img = batch["lidar_maps"]
            ref_img = batch["ref_rgb"].astype(np.float32) / 255.0

            save_u8(out_dir / "rife.jpg", rife_img)
            save_u8(out_dir / "input_ref.jpg", ref_img)
            save_trust_maps(out_dir, maps_img, cfg.lidar_zone_min)

            if not args.no_blend:
                blend = blend_model_rife_lidar(
                    pred,
                    rife_img,
                    trust_img,
                    alpha=args.blend_alpha,
                    mask_thr=cfg.lidar_zone_min,
                    rife_blur_sigma=args.rife_blur_sigma,
                )
                save_u8(out_dir / "blend_lidar_rife.jpg", blend)

            summary = {
                "sample_id": sid,
                "camera": cam,
                "checkpoint": ckpt_name,
                "mirrored_input": mirror,
                "image_space_unflipped": mirror,
                "output_version": OUTPUT_VERSION,
                "consensus_raw_file": "consensus_raw.npy",
                "lidar_mask": "spread_r3_blur1_fine_only",
                "lidar_density_fine_file": "lidar_density_fine.png",
                "lidar_trust_file": "lidar_trust.png",
                "lidar_blend_mask_file": "lidar_blend_mask.png",
                "lidar_spread_radius_fine": cfg.lidar_spread_radius_fine,
                "lidar_spread_blur_fine": cfg.lidar_spread_blur_fine,
                "lidar_mask_thr": cfg.lidar_zone_min,
                "blend_alpha": None if args.no_blend else args.blend_alpha,
                "rife_blur_sigma": None if args.no_blend else args.rife_blur_sigma,
            }
            (out_dir / "meta_infer.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            ok_n += 1
        except Exception as e:
            fail_n += 1
            print(f"  FAIL {sid[:50]}...  {e}", flush=True)

        if i % 10 == 0 or i == len(baked_dirs):
            dt = time.perf_counter() - t0
            print(f"  [{i}/{len(baked_dirs)}] ok={ok_n} skip={skip_n} fail={fail_n}  {dt:.0f}s", flush=True)

    print(f"done ok={ok_n} skip={skip_n} fail={fail_n}  -> {args.out_dir}")
    return 0 if fail_n == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
