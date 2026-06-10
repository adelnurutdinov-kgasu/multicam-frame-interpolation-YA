"""Val: сравнение Stage-1 vs V3 best — чистый MSE YCrCb + PSNR RGB."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lib.consensus_kit import ConsensusConfig, ConsensusUNet  # noqa: E402
from lib.consensus_v3_aug import V3AugCfg  # noqa: E402
from lib.consensus_v3_dataset import V3TrainCfg, V3TrainDataset, v3_ready_with_trust, v3_train_val_split  # noqa: E402
from lib.consensus_v4_dataset import V4TrainDataset  # noqa: E402
from lib.consensus_v4_aug import V4AugCfg  # noqa: E402
from scripts.training.train_consensus_v3 import collate_v3  # noqa: E402
from ya_paths import CV_ROOT, EGO_MASKS_APPROVED  # noqa: E402

STAGE1 = (
    REPO / "notebooks/training/artifacts/checkpoints/consensus_v2_all/consensus_v2_all_best.pt"
)
V3_BEST = REPO / "artifacts/checkpoints/consensus_v3_all/consensus_v3_best.pt"


def ycc01_to_rgb01(ycc: np.ndarray) -> np.ndarray:
    u8 = np.clip(ycc * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_YCrCb2RGB).astype(np.float32) / 255.0


def psnr01(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 1e-12 else float(10.0 * np.log10(1.0 / mse))


@torch.no_grad()
def eval_ckpt(ckpt: Path, loader, device: torch.device, *, rgb_target: bool = False) -> dict:
    model = ConsensusUNet(in_ch=7, base=32, predict_confidence=False).to(device)
    st = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(st["model"], strict=True)
    model.eval()

    mse_ycc, psnr_ycc, psnr_rgb = [], [], []
    for batch in loader:
        inp = batch["inputs"].to(device)
        base = batch["base_init"].to(device)
        msk = batch["effective_mask"].to(device)
        tgt = batch["target"].to(device)
        pred = model(inp, msk, base)["pred"].float().clamp(0, 1)
        bsz = pred.shape[0]
        for i in range(bsz):
            p = pred[i].permute(1, 2, 0).cpu().numpy()
            t = tgt[i].permute(1, 2, 0).cpu().numpy()
            mse_ycc.append(float(np.mean((p - t) ** 2)))
            if rgb_target:
                psnr_rgb.append(psnr01(p, t))
                psnr_ycc.append(psnr_rgb[-1])
            else:
                psnr_ycc.append(psnr01(p, t))
                psnr_rgb.append(psnr01(ycc01_to_rgb01(p), ycc01_to_rgb01(t)))
    return {
        "n": len(mse_ycc),
        "mse_ycc": float(np.mean(mse_ycc)),
        "psnr_ycc": float(np.mean(psnr_ycc)),
        "psnr_rgb": float(np.mean(psnr_rgb)),
        "ckpt_epoch": st.get("epoch"),
        "ckpt_val_loss": st.get("val_mse") or st.get("best_val_mse"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", type=Path, default=STAGE1)
    ap.add_argument("--v3", type=Path, default=V3_BEST)
    ap.add_argument("--v4", type=Path, default=None, help="V4 RGB ckpt (вместо --v3)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="0=весь val")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=REPO / "artifacts" / "preview" / "v3_vs_stage1_val.json")
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    cfg = V3TrainCfg(
        dataset_root=CV_ROOT / "final_dataset_v5_participants" / "train",
        warps_root=CV_ROOT / "multiview_warps" / "train",
        warp_mix_root=CV_ROOT / "side_warp_mix_v1" / "train",
        trust_cache_root=CV_ROOT / "precomputed_lidar_trust" / "train",
        ego_masks_root=EGO_MASKS_APPROVED,
    )
    ds_cfg = ConsensusConfig(
        dataset_root=str(cfg.dataset_root),
        consensus_root=str(cfg.warps_root),
        baked_root=str(CV_ROOT / "rife_refinement_baked" / "train"),
        rife_root=str(CV_ROOT / "rife_predictions_v5" / "train"),
        ego_masks_root=str(cfg.ego_masks_root),
        allowed_cameras=(),
    )
    ready = v3_ready_with_trust(ds_cfg, cfg.trust_cache_root)
    val_dirs = v3_train_val_split(ready, seed=42, val_fraction=0.16, subset="val")
    if args.limit > 0:
        val_dirs = val_dirs[: args.limit]

    loader_ycc = DataLoader(
        V3TrainDataset(val_dirs, cfg, ds_cfg, V3AugCfg(enabled=False)),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_v3,
    )
    loader_rgb = DataLoader(
        V4TrainDataset(val_dirs, cfg, ds_cfg, V4AugCfg(enabled=False)),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_v3,
    )

    print(f"val n={len(val_dirs)} device={device}", flush=True)
    out = {"val_n": len(val_dirs)}
    if args.stage1.is_file():
        m = eval_ckpt(args.stage1, loader_ycc, device, rgb_target=False)
        out["stage1"] = {"path": str(args.stage1), **m}
        print(
            f"{'stage1':8s} ep={m.get('ckpt_epoch')}  "
            f"mse_ycc={m['mse_ycc']:.6f}  psnr_rgb={m['psnr_rgb']:.2f} dB",
            flush=True,
        )
    if args.v4 and args.v4.is_file():
        m = eval_ckpt(args.v4, loader_rgb, device, rgb_target=True)
        out["v4_rgb"] = {"path": str(args.v4), **m}
        print(
            f"{'v4_rgb':8s} ep={m.get('ckpt_epoch')}  "
            f"mse_rgb={m['mse_ycc']:.6f}  psnr_rgb={m['psnr_rgb']:.2f} dB",
            flush=True,
        )
    elif args.v3.is_file():
        m = eval_ckpt(args.v3, loader_ycc, device, rgb_target=False)
        out["v3"] = {"path": str(args.v3), **m}
        print(
            f"{'v3':8s} ep={m.get('ckpt_epoch')}  "
            f"mse_ycc={m['mse_ycc']:.6f}  psnr_rgb={m['psnr_rgb']:.2f} dB",
            flush=True,
        )

    if "stage1" in out and ("v3" in out or "v4_rgb" in out):
        other = out.get("v4_rgb") or out["v3"]
        d_rgb = other["psnr_rgb"] - out["stage1"]["psnr_rgb"]
        d_mse = other["mse_ycc"] - out["stage1"]["mse_ycc"]
        key = "v4_rgb" if "v4_rgb" in out else "v3"
        out[f"delta_{key}_minus_stage1"] = {"psnr_rgb_db": d_rgb, "mse": d_mse}
        print(f"delta vs stage1: psnr_rgb {d_rgb:+.3f} dB  mse {d_mse:+.6f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
