from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from consensus_kit import ConsensusConfig, ConsensusUNet, discover_samples  # noqa: E402
from scripts.training.train_v2_ycrcb_side import Cfg, DS, SIDE_CAMERAS  # noqa: E402
from ya_paths import CV_ROOT, EGO_MASKS_APPROVED  # noqa: E402


def psnr01(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def ycc01_to_rgb01(ycc: np.ndarray) -> np.ndarray:
    u8 = np.clip(ycc * 255.0, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(u8, cv2.COLOR_YCrCb2RGB).astype(np.float32) / 255.0
    return np.clip(rgb, 0.0, 1.0)


def save_rgb(path: Path, rgb01: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(rgb01 * 255.0, 0, 255).astype(np.uint8)).save(path, quality=92)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=Path("artifacts/checkpoints/consensus_side_v2/consensus_v2_side_100_best.pt"),
    )
    ap.add_argument("--color-space", choices=["ycrcb", "rgb"], default="ycrcb")
    ap.add_argument("--limit", type=int, default=100, help="same subset size as training")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-fraction", type=float, default=0.16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--save-n", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/preview/consensus_v2_side_eval"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not args.ckpt.is_file():
        print(f"Нет чекпоинта: {args.ckpt}")
        return 1

    cfg = Cfg(seed=args.seed)
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
        print(f"Недостаточно готовых sample with trust: {len(ready)} < {args.limit}")
        return 1

    rng = np.random.RandomState(args.seed)
    idx = np.arange(len(ready))
    rng.shuffle(idx)
    chosen = [ready[i] for i in idx[: args.limit]]
    n_val = max(1, int(args.limit * args.val_fraction))
    val_dirs = chosen[:n_val]

    val_ds = DS(val_dirs, cfg, color_space=args.color_space)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = ConsensusUNet(in_ch=7, base=32, predict_confidence=False).to(device)
    st = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(st["model"], strict=True)
    model.eval()

    mse_ycc_all = []
    psnr_ycc_all = []
    mse_rgb_all = []
    psnr_rgb_all = []

    save_count = 0
    with torch.no_grad():
        for batch in val_loader:
            inp = batch["inputs"].to(device)
            base = batch["base_init"].to(device)
            msk = batch["effective_mask"].to(device)
            tgt = batch["target"].to(device)

            pred = model(inp, msk, base)["pred"].float().clamp(0, 1)
            bsz = pred.shape[0]
            for i in range(bsz):
                p_ycc = pred[i].permute(1, 2, 0).cpu().numpy().astype(np.float32)
                t_ycc = tgt[i].permute(1, 2, 0).cpu().numpy().astype(np.float32)

                mse_ycc = float(np.mean((p_ycc - t_ycc) ** 2))
                psnr_ycc = psnr01(p_ycc, t_ycc)
                mse_ycc_all.append(mse_ycc)
                psnr_ycc_all.append(psnr_ycc)

                if args.color_space == "rgb":
                    p_rgb = np.clip(p_ycc, 0.0, 1.0)
                    t_rgb = np.clip(t_ycc, 0.0, 1.0)
                else:
                    p_rgb = ycc01_to_rgb01(p_ycc)
                    t_rgb = ycc01_to_rgb01(t_ycc)
                mse_rgb = float(np.mean((p_rgb - t_rgb) ** 2))
                psnr_rgb = psnr01(p_rgb, t_rgb)
                mse_rgb_all.append(mse_rgb)
                psnr_rgb_all.append(psnr_rgb)

                if save_count < args.save_n:
                    sid = val_dirs[save_count].name
                    out = args.out_dir / sid
                    save_rgb(out / "pred_rgb.jpg", p_rgb)
                    save_rgb(out / "target_rgb.jpg", t_rgb)
                    diff = np.abs(p_rgb - t_rgb)
                    save_rgb(out / "diff_rgb_x4.jpg", np.clip(diff * 4.0, 0, 1))
                    save_count += 1

    result = {
        "ckpt": str(args.ckpt),
        "num_val_samples": len(mse_ycc_all),
        "metrics": {
            "mse_ycc_mean": float(np.mean(mse_ycc_all)),
            "psnr_ycc_mean": float(np.mean(psnr_ycc_all)),
            "mse_rgb_mean": float(np.mean(mse_rgb_all)),
            "psnr_rgb_mean": float(np.mean(psnr_rgb_all)),
        },
        "color_space": args.color_space,
        "preview_dir": str(args.out_dir.resolve()),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== V2 eval ===")
    print(f"val_samples: {result['num_val_samples']}")
    print(f"MSE_YCrCb: {result['metrics']['mse_ycc_mean']:.6f}")
    print(f"PSNR_YCrCb: {result['metrics']['psnr_ycc_mean']:.3f} dB")
    print(f"MSE_RGB: {result['metrics']['mse_rgb_mean']:.6f}")
    print(f"PSNR_RGB: {result['metrics']['psnr_rgb_mean']:.3f} dB")
    print(f"Preview: {result['preview_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

