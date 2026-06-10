"""RIFE с подсказкой: входы = blend(ours, t0) и blend(ours, t1) вместо сырых t0/t1.

Пример:
  python scripts/inference/run_rife_guided_inputs.py --split train --limit 50 --blend 0.5
  python scripts/inference/run_rife_guided_inputs.py --split test --pred-source warp_mix
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_BASELINES = REPO / "scripts" / "baselines"
if str(_BASELINES) not in sys.path:
    sys.path.insert(0, str(_BASELINES))

from compare_baselines import RifeWrapper, load_rgb, psnr  # noqa: E402
from lib.consensus_kit import ConsensusConfig, discover_samples  # noqa: E402
from ya_paths import CV_ROOT  # noqa: E402


def _v2_ready_with_trust(ds_cfg: ConsensusConfig, trust_root: Path) -> list[Path]:
    ready: list[Path] = []
    for bdir in discover_samples(ds_cfg):
        meta = json.loads((bdir / "meta.json").read_text(encoding="utf-8"))
        sid = meta["sample_id"]
        if (trust_root / sid / "lidar_trust.npy").is_file():
            ready.append(bdir)
    return ready


def _v2_train_val_split(
    ready: list[Path],
    *,
    seed: int,
    val_fraction: float,
    subset: str,
) -> list[Path]:
    """Тот же split, что в consensus_V2_training_side: shuffle(seed), val = первые 16%."""
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


def _baked_to_dataset_dir(bdir: Path, dataset_root: Path) -> Path:
    meta = json.loads((bdir / "meta.json").read_text(encoding="utf-8"))
    return Path(meta.get("source_dir", dataset_root / meta["sample_id"]))


def _load_pred_rgb(sid: str, cam: str, src: Path, split: str, sample_dir: Path) -> np.ndarray | None:
    if src == "pseudo":
        p = CV_ROOT / "pseudo_v2_train_rgb" / sid / "pred_rgb.npy"
        if p.is_file():
            arr = np.load(p).astype(np.float32)
            return (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    if src == "warp_mix":
        p = CV_ROOT / "side_warp_mix_v1" / split / sid / "warp_mix.jpg"
        if p.is_file():
            return load_rgb(p)
    if src == "consensus_out":
        p = CV_ROOT / "consensus_test_outputs" / sid / "consensus_model.jpg"
        if p.is_file():
            return load_rgb(p)
    return None


def _match_shape(ref: np.ndarray, img: np.ndarray) -> np.ndarray:
    if img.shape[:2] == ref.shape[:2]:
        return img
    return cv2.resize(img, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LINEAR)


def _blend_u8(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    w = float(np.clip(w, 0.0, 1.0))
    b = _match_shape(a, b)
    out = a.astype(np.float32) * w + b.astype(np.float32) * (1.0 - w)
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description="RIFE on blend(ours,t0) vs blend(ours,t1)")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument(
        "--subset",
        choices=["all", "train", "val"],
        default="all",
        help="для split=train: val/train по seed=42 и val_fraction (как в ноутбуке V2)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-fraction", type=float, default=0.16)
    ap.add_argument("--trust-root", type=Path, default=CV_ROOT / "precomputed_lidar_trust" / "train")
    ap.add_argument("--dataset-dir", type=Path, default=CV_ROOT / "final_dataset_v5_participants")
    ap.add_argument("--limit", type=int, default=0, help="0 = all with GT")
    ap.add_argument(
        "--blend",
        type=float,
        default=0.5,
        help="w для ours: in0 = blend*ours + (1-blend)*t0, in1 = blend*ours + (1-blend)*t1",
    )
    ap.add_argument(
        "--pred-source",
        choices=["pseudo", "warp_mix", "consensus_out"],
        default="pseudo",
        help="pseudo=pred_rgb.npy (train), warp_mix или consensus_model.jpg",
    )
    ap.add_argument("--out-dir", type=Path, default=CV_ROOT / "rife_guided_blend_v1")
    ap.add_argument("--save-n", type=int, default=8, help="сколько превью сохранить")
    args = ap.parse_args()

    dataset_root = args.dataset_dir / args.split
    if not dataset_root.is_dir():
        print("missing split:", dataset_root)
        return 1

    jobs: list[tuple[Path, str, str]] = []
    if args.split == "train" and args.subset in ("train", "val"):
        ds_cfg = ConsensusConfig(
            dataset_root=str(dataset_root),
            consensus_root=str(CV_ROOT / "multiview_warps" / "train"),
            baked_root=str(CV_ROOT / "rife_refinement_baked" / "train"),
            rife_root=str(CV_ROOT / "rife_predictions_v5" / "train"),
            ego_masks_root=str(REPO / "methods_gallery" / "_ego_manual_masks" / "masks_approved"),
            allowed_cameras=(),
            image_h=544,
            image_w=1024,
        )
        ready = _v2_ready_with_trust(ds_cfg, args.trust_root)
        picked = _v2_train_val_split(
            ready, seed=args.seed, val_fraction=args.val_fraction, subset=args.subset
        )
        print(
            f"V2 split subset={args.subset}: n={len(picked)} "
            f"(ready={len(ready)}, val_fraction={args.val_fraction}, seed={args.seed})"
        )
        for bdir in picked:
            meta = json.loads((bdir / "meta.json").read_text(encoding="utf-8"))
            cam = meta.get("target_camera") or meta.get("camera")
            sid = meta["sample_id"]
            sd = _baked_to_dataset_dir(bdir, dataset_root)
            gt = sd / "target" / f"{cam}.jpg"
            t0 = sd / "input" / "t0" / f"{cam}.jpg"
            t1 = sd / "input" / "t1" / f"{cam}.jpg"
            if gt.is_file() and t0.is_file() and t1.is_file():
                jobs.append((sd, sid, cam))
    else:
        for sd in sorted(dataset_root.iterdir()):
            if not sd.is_dir() or not (sd / "meta.json").is_file():
                continue
            meta = json.loads((sd / "meta.json").read_text(encoding="utf-8"))
            cam = meta.get("target_camera") or meta.get("camera")
            gt = sd / "target" / f"{cam}.jpg"
            t0 = sd / "input" / "t0" / f"{cam}.jpg"
            t1 = sd / "input" / "t1" / f"{cam}.jpg"
            if not (gt.is_file() and t0.is_file() and t1.is_file()):
                continue
            sid = meta.get("sample_id", sd.name)
            jobs.append((sd, sid, cam))

    if args.limit > 0:
        jobs = jobs[: args.limit]

    if not jobs:
        print("no samples")
        return 1

    rife = RifeWrapper()
    out_split = args.out_dir / args.split
    out_split.mkdir(parents=True, exist_ok=True)

    metrics = {
        "rife_raw": [],
        "rife_guided": [],
        "rife_precomputed": [],
        "ours_only": [],
        "blend_07_ours_03_guided": [],
        "blend_07_ours_03_rife_raw": [],
        "blend_07_ours_03_rife_precomp": [],
    }
    rife_disk = CV_ROOT / "rife_predictions_v5" / args.split
    missing_pred = 0
    t0_run = time.time()

    for i, (sd, sid, cam) in enumerate(jobs, 1):
        pred = _load_pred_rgb(sid, cam, args.pred_source, args.split, sd)
        if pred is None:
            missing_pred += 1
            continue

        img0 = load_rgb(sd / "input" / "t0" / f"{cam}.jpg")
        img1 = load_rgb(sd / "input" / "t1" / f"{cam}.jpg")
        gt = load_rgb(sd / "target" / f"{cam}.jpg")
        pred = _match_shape(img0, pred)

        w = args.blend
        in0 = _blend_u8(pred, img0, w)
        in1 = _blend_u8(pred, img1, w)

        rife_raw = rife.infer(img0, img1)
        rife_guided = rife.infer(in0, in1)

        metrics["rife_raw"].append(psnr(rife_raw, gt))
        metrics["rife_guided"].append(psnr(rife_guided, gt))
        metrics["ours_only"].append(psnr(pred, gt))
        metrics["blend_07_ours_03_guided"].append(psnr(_blend_u8(pred, rife_guided, 0.7), gt))
        metrics["blend_07_ours_03_rife_raw"].append(psnr(_blend_u8(pred, rife_raw, 0.7), gt))

        rife_p = rife_disk / f"{sid}.jpg"
        if rife_p.is_file():
            rife_pc = _match_shape(img0, load_rgb(rife_p))
            metrics["rife_precomputed"].append(psnr(rife_pc, gt))
            metrics["blend_07_ours_03_rife_precomp"].append(psnr(_blend_u8(pred, rife_pc, 0.7), gt))

        if i <= args.save_n:
            od = out_split / sid
            od.mkdir(parents=True, exist_ok=True)
            Image.fromarray(in0).save(od / "guided_in0.jpg", quality=92)
            Image.fromarray(in1).save(od / "guided_in1.jpg", quality=92)
            Image.fromarray(rife_guided).save(od / "rife_guided_out.jpg", quality=92)
            Image.fromarray(rife_raw).save(od / "rife_raw_out.jpg", quality=92)
            Image.fromarray(pred).save(od / "ours_pred.jpg", quality=92)
            Image.fromarray(gt).save(od / "target.jpg", quality=92)

        if i % 10 == 0 or i == len(jobs):
            print(f"[{i}/{len(jobs)}] missing_pred={missing_pred}", flush=True)

    dt = time.time() - t0_run
    print(f"\nblend(ours,t*) weight={args.blend}  pred_source={args.pred_source}  split={args.split}")
    print(f"samples={len(metrics['rife_raw'])}  missing_pred={missing_pred}  time={dt:.0f}s")
    for name, vals in metrics.items():
        if vals:
            mean_psnr = float(np.mean(vals))
            print(f"  {name:18s}  PSNR={mean_psnr:.3f} dB  (n={len(vals)})")

    summary = {
        "split": args.split,
        "subset": args.subset,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "blend": args.blend,
        "pred_source": args.pred_source,
        "n": len(metrics["rife_raw"]),
        "mean_psnr": {k: float(np.mean(v)) if v else None for k, v in metrics.items()},
    }
    metrics_name = "metrics.json" if args.subset == "all" else f"metrics_{args.subset}.json"
    (out_split / metrics_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("saved", out_split / metrics_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
