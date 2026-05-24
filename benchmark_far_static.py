"""Benchmark mean(t0,t1) on far-static mask across full dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from layered_parallax import get_lidar_depth_raw
from static_mask import MaskParams, build_static_mask, compute_flow_maps, load_best_params, z_ref_from_lidar


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def psnr(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        if not mask.any():
            return float("nan")
        pred = pred[mask]
        gt = gt[mask]
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    return 20.0 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float("inf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=Path,
                   default=Path(r"C:\Users\adel\Downloads\cv_dataset\final_dataset_v5_participants"))
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--far-ratio", type=float, default=None,
                   help="override far_min_ratio; default from sweep json or config")
    p.add_argument("--sweep-json", type=Path, default=Path("far_boundary_sweep.json"))
    p.add_argument("--config", type=Path, default=Path("static_mask_tune_best.json"))
    p.add_argument("--out", type=Path, default=Path("far_static_benchmark.json"))
    args = p.parse_args()

    params = load_best_params(args.config)
    if args.far_ratio is not None:
        ratio = args.far_ratio
    elif args.sweep_json.is_file():
        ratio = json.loads(args.sweep_json.read_text(encoding="utf-8"))["recommended_far_min_ratio"]
    else:
        ratio = params.far_min_ratio
    params = replace(params, far_min_ratio=ratio)

    split_dir = args.dataset_dir / args.split
    samples = sorted(d for d in split_dir.iterdir() if d.is_dir())
    has_gt = args.split == "train"

    rows = []
    for i, sd in enumerate(samples):
        meta = json.load(open(sd / "meta.json"))
        cam = meta["target_camera"]
        img_t0 = load_rgb(sd / "input" / "t0" / f"{cam}.jpg")
        img_t1 = load_rgb(sd / "input" / "t1" / f"{cam}.jpg")
        mean_pred = ((img_t0.astype(np.float32) + img_t1.astype(np.float32)) * 0.5).astype(np.uint8)
        flow_mag, comp_err, sigma, _ = compute_flow_maps(img_t0, img_t1)
        depth_raw = get_lidar_depth_raw(sd, cam, "target")
        z_ref = z_ref_from_lidar(depth_raw)
        static = build_static_mask(flow_mag, comp_err, None, sigma, params, z_ref, depth_raw)

        row = {
            "sample_id": meta["sample_id"],
            "static_pct": 100 * static.mean(),
            "z_ref": z_ref,
        }
        if has_gt:
            gt = load_rgb(sd / "target" / f"{cam}.jpg")
            row["psnr_full"] = psnr(mean_pred, gt)
            row["psnr_static_far"] = psnr(mean_pred, gt, static)
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(samples)}", flush=True)

    summary = {
        "split": args.split,
        "n_samples": len(rows),
        "far_min_ratio": ratio,
        "flow_t": params.flow_t,
        "comp_t": params.comp_t,
        "avg_static_pct": float(np.mean([r["static_pct"] for r in rows])),
    }
    if has_gt:
        summary["avg_psnr_full"] = float(np.nanmean([r["psnr_full"] for r in rows]))
        summary["avg_psnr_static_far"] = float(np.nanmean([r["psnr_static_far"] for r in rows]))

    out = {"summary": summary, "samples": rows}
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n=== far static benchmark ({args.split}, ratio={ratio:.1f}x z_ref) ===")
    print(f"samples: {len(rows)}  avg static: {summary['avg_static_pct']:.1f}%")
    if has_gt:
        print(f"PSNR full: {summary['avg_psnr_full']:.2f} dB")
        print(f"PSNR static_far: {summary['avg_psnr_static_far']:.2f} dB")
        print(f"delta: +{summary['avg_psnr_static_far'] - summary['avg_psnr_full']:.2f} dB")
    print(f"Saved {args.out.resolve()}")


if __name__ == "__main__":
    main()
