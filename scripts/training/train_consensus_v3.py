"""Обучение Consensus V3: V2 inputs (7ch YCrCb) + аугментации (mirror 2×, crop, color)."""
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
from lib.consensus_v3_aug import V3AugCfg  # noqa: E402
from lib.consensus_v3_dataset import V3TrainCfg, V3TrainDataset, v3_ready_with_trust, v3_train_val_split  # noqa: E402
from lib.consensus_v3_train import ModelEMA, V3LossCfg, v3_composite_loss  # noqa: E402
from ya_paths import CV_ROOT, EGO_MASKS_APPROVED  # noqa: E402

# Stage-1 (V2) — только чтение; V3 пишет в отдельную директорию
DEFAULT_STAGE1_CKPT = (
    REPO / "notebooks" / "training" / "artifacts" / "checkpoints" / "consensus_v2_all" / "consensus_v2_all_best.pt"
)
DEFAULT_V3_CKPT_DIR = REPO / "artifacts" / "checkpoints" / "consensus_v3_all"
DEFAULT_V3_SCRATCH_DIR = REPO / "artifacts" / "checkpoints" / "consensus_v3_scratch"
PROTECTED_CKPT_DIRS = (
    REPO / "notebooks" / "training" / "artifacts" / "checkpoints" / "consensus_v2_all",
    REPO / "artifacts" / "checkpoints" / "consensus_v2_all",
    REPO / "artifacts" / "checkpoints" / "consensus_side_v2",
)


def _assert_safe_ckpt_paths(init_ckpt: Path | None, ckpt_dir: Path) -> None:
    """V3 пишет только в ckpt_dir; V2-папки не используются как output."""
    ckpt_dir = ckpt_dir.resolve()
    for protected in PROTECTED_CKPT_DIRS:
        if protected.resolve() == ckpt_dir:
            raise ValueError(f"ckpt-dir={ckpt_dir} совпадает с защищённой папкой V2")
    if init_ckpt is not None:
        init_ckpt = init_ckpt.resolve()
        for protected in PROTECTED_CKPT_DIRS:
            if init_ckpt == (protected / "consensus_v2_all_best.pt").resolve():
                pass  # load Stage-1 — read-only, ок
            if ckpt_dir == protected.resolve():
                raise ValueError(f"ckpt-dir не может быть {protected}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_v3(batch):
    keys = ("inputs", "base_init", "effective_mask", "target")
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}
    out["meta"] = [b["meta"] for b in batch]
    return out


def run_epoch(
    model,
    loader,
    device,
    loss_cfg: V3LossCfg,
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
                    loss, _ = v3_composite_loss(pred, tgt, base, trust, loss_cfg)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(model)
            elif train:
                pred = model(inp, msk, base)["pred"].float()
                loss, _ = v3_composite_loss(pred, tgt, base, trust, loss_cfg)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if ema is not None:
                    ema.update(model)
            else:
                with torch.no_grad():
                    pred = eval_model(inp, msk, base)["pred"].float()
                    loss, _ = v3_composite_loss(pred, tgt, base, trust, loss_cfg)
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4, help="fine-tune: 1e-4; с нуля: 2e-4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-fraction", type=float, default=0.16)
    ap.add_argument("--patch-h", type=int, default=416)
    ap.add_argument("--patch-w", type=int, default=768)
    ap.add_argument("--no-virtual-2x", action="store_true", help="отключить удвоение mirror")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt-dir", type=Path, default=DEFAULT_V3_CKPT_DIR)
    ap.add_argument(
        "--init-ckpt",
        type=Path,
        default=None,
        help="веса Stage-1/V2 (только load, файл не изменяется)",
    )
    ap.add_argument("--resume", type=Path, default=None, help="alias для --init-ckpt")
    ap.add_argument("--save-every", type=int, default=5, help="промежуточный сейв каждые N эпох")
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--anchor-lambda", type=float, default=0.08, help="штраф |pred-base|²")
    ap.add_argument("--trust-floor", type=float, default=0.25)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--patience", type=int, default=8, help="early stop если val не улучшается")
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument(
        "--continue-ckpt",
        type=Path,
        default=None,
        help="продолжить V3 (consensus_v3_last.pt), не Stage-1",
    )
    ap.add_argument("--strong-aug", action="store_true", help="сильнее cutout/noise/color")
    ap.add_argument(
        "--from-scratch",
        action="store_true",
        help="случайная инициализация, без Stage-1; ckpt → consensus_v3_scratch",
    )
    args = ap.parse_args()

    if args.from_scratch:
        if args.ckpt_dir == DEFAULT_V3_CKPT_DIR:
            args.ckpt_dir = DEFAULT_V3_SCRATCH_DIR
        init_ckpt = args.continue_ckpt
        if init_ckpt is None and args.lr == 1e-4:
            args.lr = 2e-4
        if args.anchor_lambda == 0.08:
            args.anchor_lambda = 0.0
        if args.trust_floor == 0.25:
            args.trust_floor = 1.0
    else:
        init_ckpt = args.continue_ckpt or args.init_ckpt or args.resume
        if init_ckpt is None:
            init_ckpt = DEFAULT_STAGE1_CKPT

    if init_ckpt is not None and not init_ckpt.is_file():
        raise FileNotFoundError(
            f"нет init-ckpt: {init_ckpt}\n"
            f"ожидали Stage-1: {DEFAULT_STAGE1_CKPT} (exists={DEFAULT_STAGE1_CKPT.is_file()})"
        )
    _assert_safe_ckpt_paths(init_ckpt, args.ckpt_dir)

    seed_everything(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    cfg = V3TrainCfg(
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

    aug_train = V3AugCfg(
        enabled=True,
        patch_h=args.patch_h,
        patch_w=args.patch_w,
        virtual_mirror_2x=not args.no_virtual_2x,
        hflip_p=0.0 if not args.no_virtual_2x else 0.5,
        cutout_p=0.35 if args.strong_aug else 0.25,
        input_noise_std=0.02 if args.strong_aug else 0.012,
        y_gain=(0.85, 1.15) if args.strong_aug else (0.90, 1.10),
    )
    loss_cfg = V3LossCfg(
        anchor_lambda=args.anchor_lambda,
        trust_floor=args.trust_floor,
    )
    aug_val = V3AugCfg(enabled=False)

    train_ds = V3TrainDataset(train_dirs, cfg, ds_cfg, aug_train)
    val_ds = V3TrainDataset(val_dirs, cfg, ds_cfg, aug_val)
    print(f"ready={len(ready)} train={len(train_dirs)} val={len(val_dirs)}", flush=True)
    print(
        f"train_loader samples/epoch={len(train_ds)} (virtual_2x={aug_train.virtual_mirror_2x})",
        flush=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_v3,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(4, args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_v3,
    )

    model = ConsensusUNet(in_ch=7, base=32, predict_confidence=False).to(device)
    start_ep = 0
    if init_ckpt is not None:
        st = torch.load(init_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(st["model"], strict=True)
        if args.continue_ckpt is not None:
            start_ep = int(st.get("epoch", 0))
            best_val = float(st.get("best_val_mse", st.get("val_mse", 1e9)))
            best_ep = int(st.get("best_epoch", start_ep))
            print(
                f"resume from ep {start_ep}  best_val_mse={best_val:.6f} (best_ep {best_ep})",
                flush=True,
            )
        else:
            best_val, best_ep = 1e9, 0
            print(f"init weights (read-only): {init_ckpt}", flush=True)
    else:
        best_val, best_ep = 1e9, 0
        print("init: random weights (from-scratch)", flush=True)
    print(f"write checkpoints -> {args.ckpt_dir.resolve()}", flush=True)

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
    print(
        f"reg: wd={args.weight_decay} anchor={loss_cfg.anchor_lambda} "
        f"trust_floor={loss_cfg.trust_floor} ema={args.ema_decay if ema else 'off'} patience={args.patience}",
        flush=True,
    )

    if start_ep >= args.epochs:
        print(f"already at epoch {start_ep} >= {args.epochs}, nothing to do", flush=True)
        return 0

    for ep in range(start_ep + 1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, loss_cfg, opt, scaler, ema)
        va = run_epoch(model, val_loader, device, loss_cfg, ema=ema, use_ema_eval=ema is not None)
        sched.step()
        lr_now = sched.get_last_lr()[0]
        row = {"epoch": ep, "train_mse": tr, "val_mse": va, "lr": lr_now}
        history.append(row)
        print(f"epoch {ep:03d}: train_mse={tr:.6f}  val_mse={va:.6f}  lr={lr_now:.2e}", flush=True)

        save_state = ema.state_dict() if ema is not None else model.state_dict()
        payload = {
            "epoch": ep,
            "model": save_state,
            "best_val_mse": best_val,
            "best_epoch": best_ep,
            "val_mse": va,
            "train_mse": tr,
            "init_ckpt": str(init_ckpt) if init_ckpt else None,
            "from_scratch": args.from_scratch,
            "aug": aug_train.__dict__,
            "loss_cfg": loss_cfg.__dict__,
        }
        torch.save(payload, args.ckpt_dir / "consensus_v3_last.pt")
        if args.save_every > 0 and ep % args.save_every == 0:
            ep_path = args.ckpt_dir / f"consensus_v3_epoch_{ep:03d}.pt"
            torch.save(payload, ep_path)
            print(f"  -> snapshot {ep_path.name}", flush=True)
        if va < best_val:
            best_val = va
            best_ep = ep
            stale = 0
            payload["best_val_mse"] = best_val
            payload["best_epoch"] = best_ep
            torch.save(payload, args.ckpt_dir / "consensus_v3_best.pt")
            print(f"  -> best val_mse={best_val:.6f} (ep {best_ep})", flush=True)
        else:
            stale += 1
            if args.patience > 0 and stale >= args.patience:
                print(f"early stop: нет улучшения val {stale} эпох (best ep {best_ep})", flush=True)
                break

    (args.ckpt_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("done best_val_mse=", best_val, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
