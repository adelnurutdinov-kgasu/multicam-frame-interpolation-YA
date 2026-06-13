"""Consensus V5: RGB + trust-weighted loss (дыры важнее), best ckpt по val PSNR."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lib.consensus_kit import ConsensusConfig, ConsensusUNet  # noqa: E402
from lib.consensus_v4_aug import V4AugCfg  # noqa: E402
from lib.consensus_v4_dataset import V4TrainCfg, V4TrainDataset, v3_ready_with_trust, v3_train_val_split  # noqa: E402
from lib.consensus_v3_train import ModelEMA, V5LossCfg, v5_weighted_rgb_loss  # noqa: E402
from ya_paths import CV_ROOT, EGO_MASKS_APPROVED  # noqa: E402

DEFAULT_V5_DIR = REPO / "artifacts" / "checkpoints" / "consensus_v5_weighted"
DEFAULT_V4_BEST = REPO / "artifacts" / "checkpoints" / "consensus_v4_scratch" / "consensus_v4_best.pt"
PROTECTED_CKPT_DIRS = (
    REPO / "notebooks" / "training" / "artifacts" / "checkpoints" / "consensus_v2_all",
    REPO / "artifacts" / "checkpoints" / "consensus_v2_all",
    REPO / "artifacts" / "checkpoints" / "consensus_side_v2",
    REPO / "artifacts" / "checkpoints" / "consensus_v3_scratch",
    REPO / "artifacts" / "checkpoints" / "consensus_v4_scratch",
)


def _assert_safe_ckpt_paths(ckpt_dir: Path) -> None:
    ckpt_dir = ckpt_dir.resolve()
    for protected in PROTECTED_CKPT_DIRS:
        if protected.resolve() == ckpt_dir:
            raise ValueError(f"ckpt-dir={ckpt_dir} совпадает с защищённой папкой")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_v5(batch):
    keys = ("inputs", "base_init", "effective_mask", "target")
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}
    out["meta"] = [b["meta"] for b in batch]
    return out


def _batch_psnr_rgb(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = float(((pred - target) ** 2).mean().detach().cpu())
    return 99.0 if mse <= 1e-12 else float(10.0 * np.log10(1.0 / mse))


def run_epoch(
    model,
    loader,
    device,
    loss_cfg: V5LossCfg,
    optimizer=None,
    scaler=None,
    ema: ModelEMA | None = None,
    *,
    use_ema_eval: bool = False,
):
    train = optimizer is not None
    model.train(train)
    eval_model = ema.shadow if (use_ema_eval and ema is not None) else model
    if not train:
        eval_model.eval()

    losses = []
    for batch in loader:
        inp = batch["inputs"].to(device, non_blocking=True)
        base = batch["base_init"].to(device, non_blocking=True)
        msk = batch["effective_mask"].to(device, non_blocking=True)
        tgt = batch["target"].to(device, non_blocking=True)
        trust = inp[:, 3:4]
        with torch.set_grad_enabled(train):
            if scaler is not None and train:
                with torch.amp.autocast("cuda"):
                    pred = model(inp, msk, base)["pred"].float()
                    loss, _ = v5_weighted_rgb_loss(pred, tgt, base, trust, loss_cfg)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(model)
            elif train:
                pred = model(inp, msk, base)["pred"].float()
                loss, _ = v5_weighted_rgb_loss(pred, tgt, base, trust, loss_cfg)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if ema is not None:
                    ema.update(model)
            else:
                with torch.no_grad():
                    pred = eval_model(inp, msk, base)["pred"].float().clamp(0, 1)
                    loss, _ = v5_weighted_rgb_loss(pred, tgt, base, trust, loss_cfg)
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def eval_val_psnr(
    model,
    loader,
    device,
    ema: ModelEMA | None = None,
    *,
    use_ema_eval: bool = True,
) -> float:
    eval_model = ema.shadow if (use_ema_eval and ema is not None) else model
    eval_model.eval()
    psnrs = []
    for batch in loader:
        inp = batch["inputs"].to(device, non_blocking=True)
        base = batch["base_init"].to(device, non_blocking=True)
        msk = batch["effective_mask"].to(device, non_blocking=True)
        tgt = batch["target"].to(device, non_blocking=True)
        pred = eval_model(inp, msk, base)["pred"].float().clamp(0, 1)
        for i in range(pred.shape[0]):
            psnrs.append(_batch_psnr_rgb(pred[i], tgt[i]))
    return float(np.mean(psnrs)) if psnrs else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Consensus V5 RGB + trust-weighted loss")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-fraction", type=float, default=0.16)
    ap.add_argument("--patch-h", type=int, default=416)
    ap.add_argument("--patch-w", type=int, default=768)
    ap.add_argument("--no-virtual-2x", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt-dir", type=Path, default=DEFAULT_V5_DIR)
    ap.add_argument("--init-ckpt", type=Path, default=DEFAULT_V4_BEST)
    ap.add_argument("--from-scratch", action="store_true", help="не грузить V4 best")
    ap.add_argument("--continue-ckpt", type=Path, default=None)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--strong-aug", action="store_true", default=True)
    ap.add_argument("--hole-weight", type=float, default=2.5)
    ap.add_argument("--valid-weight", type=float, default=0.5)
    ap.add_argument("--trust-gamma", type=float, default=1.5)
    ap.add_argument("--anchor-lambda", type=float, default=0.05)
    args = ap.parse_args()

    resume_ckpt = args.continue_ckpt
    if resume_ckpt is not None and not resume_ckpt.is_file():
        raise FileNotFoundError(f"нет continue-ckpt: {resume_ckpt}")
    _assert_safe_ckpt_paths(args.ckpt_dir)

    seed_everything(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    cfg = V4TrainCfg(
        dataset_root=CV_ROOT / "final_dataset_v5_participants" / "train",
        warps_root=CV_ROOT / "multiview_warps" / "train",
        warp_mix_root=CV_ROOT / "side_warp_mix_v1" / "train",
        trust_cache_root=CV_ROOT / "precomputed_lidar_trust" / "train",
        ego_masks_root=EGO_MASKS_APPROVED,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    ds_cfg = ConsensusConfig(
        dataset_root=str(cfg.dataset_root),
        consensus_root=str(cfg.warps_root),
        baked_root=str(CV_ROOT / "rife_refinement_baked" / "train"),
        rife_root=str(CV_ROOT / "rife_predictions_v5" / "train"),
        ego_masks_root=str(cfg.ego_masks_root),
        allowed_cameras=(),
        image_h=cfg.image_h,
        image_w=cfg.image_w,
    )

    ready = v3_ready_with_trust(ds_cfg, cfg.trust_cache_root)
    train_dirs = v3_train_val_split(ready, seed=args.seed, val_fraction=args.val_fraction, subset="train")
    val_dirs = v3_train_val_split(ready, seed=args.seed, val_fraction=args.val_fraction, subset="val")

    aug_train = V4AugCfg(
        enabled=True,
        patch_h=args.patch_h,
        patch_w=args.patch_w,
        virtual_mirror_2x=not args.no_virtual_2x,
        hflip_p=0.0 if not args.no_virtual_2x else 0.5,
        cutout_p=0.35 if args.strong_aug else 0.25,
        input_noise_std=0.02 if args.strong_aug else 0.012,
        rgb_gain=(0.85, 1.15) if args.strong_aug else (0.90, 1.10),
        rgb_offset=0.04 if args.strong_aug else 0.03,
    )
    loss_cfg = V5LossCfg(
        hole_weight=args.hole_weight,
        valid_weight=args.valid_weight,
        trust_gamma=args.trust_gamma,
        anchor_lambda=args.anchor_lambda,
    )
    aug_val = V4AugCfg(enabled=False)

    train_ds = V4TrainDataset(train_dirs, cfg, ds_cfg, aug_train)
    val_ds = V4TrainDataset(val_dirs, cfg, ds_cfg, aug_val)
    print(
        f"V5 weighted RGB  ready={len(ready)} train={len(train_dirs)} val={len(val_dirs)}",
        flush=True,
    )
    print(
        f"loss: hole={loss_cfg.hole_weight} valid={loss_cfg.valid_weight} "
        f"gamma={loss_cfg.trust_gamma} anchor={loss_cfg.anchor_lambda}",
        flush=True,
    )
    print(f"train samples/epoch={len(train_ds)} virtual_2x={aug_train.virtual_mirror_2x}", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_v5,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(4, args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_v5,
    )

    model = ConsensusUNet(in_ch=7, base=32, predict_confidence=False).to(device)
    start_ep = 0
    best_val_psnr = -1.0
    best_ep = 0

    load_ckpt = resume_ckpt
    if load_ckpt is None and not args.from_scratch and args.init_ckpt.is_file():
        load_ckpt = args.init_ckpt
    if load_ckpt is not None:
        st = torch.load(load_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(st["model"], strict=True)
        if resume_ckpt is not None:
            start_ep = int(st.get("epoch", 0))
            best_val_psnr = float(st.get("best_val_psnr", -1.0))
            best_ep = int(st.get("best_epoch", start_ep))
            print(
                f"resume ep {start_ep} best_val_psnr={best_val_psnr:.3f} dB (ep {best_ep})",
                flush=True,
            )
        else:
            print(f"init weights from {load_ckpt.name}", flush=True)
    else:
        print("init: random weights", flush=True)

    print(f"ckpt -> {args.ckpt_dir.resolve()}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs - start_ep), eta_min=args.min_lr
    )
    ema = None if args.no_ema else ModelEMA(model, decay=args.ema_decay)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    stale = 0
    history_path = args.ckpt_dir / "history.json"
    history = []
    if history_path.is_file():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            history = []

    for ep in range(start_ep + 1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, loss_cfg, opt, scaler, ema)
        va_loss = run_epoch(model, val_loader, device, loss_cfg, ema=ema, use_ema_eval=ema is not None)
        va_psnr = eval_val_psnr(model, val_loader, device, ema, use_ema_eval=ema is not None)
        sched.step()
        lr_now = sched.get_last_lr()[0]
        row = {
            "epoch": ep,
            "train_loss": tr,
            "val_loss": va_loss,
            "val_psnr_rgb": va_psnr,
            "lr": lr_now,
        }
        history.append(row)
        print(
            f"epoch {ep:03d}: train_loss={tr:.6f}  val_loss={va_loss:.6f}  "
            f"val_psnr={va_psnr:.3f} dB  lr={lr_now:.2e}",
            flush=True,
        )

        save_state = ema.state_dict() if ema is not None else model.state_dict()
        payload = {
            "epoch": ep,
            "model": save_state,
            "color_space": "rgb",
            "best_val_psnr": best_val_psnr,
            "best_epoch": best_ep,
            "val_psnr_rgb": va_psnr,
            "val_loss": va_loss,
            "train_loss": tr,
            "loss_cfg": loss_cfg.__dict__,
        }
        torch.save(payload, args.ckpt_dir / "consensus_v5_last.pt")
        if args.save_every > 0 and ep % args.save_every == 0:
            torch.save(payload, args.ckpt_dir / f"consensus_v5_epoch_{ep:03d}.pt")
            print(f"  -> snapshot consensus_v5_epoch_{ep:03d}.pt", flush=True)
        if va_psnr > best_val_psnr:
            best_val_psnr, best_ep, stale = va_psnr, ep, 0
            payload["best_val_psnr"] = best_val_psnr
            payload["best_epoch"] = best_ep
            torch.save(payload, args.ckpt_dir / "consensus_v5_best.pt")
            print(f"  -> best val_psnr={best_val_psnr:.3f} dB (ep {best_ep})", flush=True)
        else:
            stale += 1
            if args.patience > 0 and stale >= args.patience:
                print(f"early stop after {stale} epochs (best ep {best_ep})", flush=True)
                break

    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"done best_val_psnr={best_val_psnr:.3f} dB (ep {best_ep})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
