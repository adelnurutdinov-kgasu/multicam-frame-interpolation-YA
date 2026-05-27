"""Sweep far depth boundary: coverage, overlap with well-predicted, PSNR on masked regions."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
import sys
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from layered_parallax import get_lidar_depth
from static_mask import (
    MaskParams,
    build_static_mask,
    compute_flow_maps,
    far_or_no_depth_gate,
    load_best_params,
    mask_metrics,
)


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


def mae(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        if not mask.any():
            return float("nan")
        pred = pred[mask]
        gt = gt[mask]
    return float(np.mean(np.abs(pred.astype(np.float32) - gt.astype(np.float32))))


@dataclass
class RatioStats:
    ratio: float
    n_samples: int
    far_gate_pct: float
    static_pct: float
    good_far_pct: float
    overlap_pct: float
    iou: float
    f1: float
    precision: float
    recall: float
    psnr_full: float
    psnr_far_gate: float
    psnr_static_far: float
    psnr_static_good: float
    mae_static_far: float
    score_static_far: float  # contest score from psnr_static_far

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def score_from_psnr(psnr_db: float) -> float:
    return max(0.0, min(100.0, (max(10.0, min(30.0, psnr_db)) - 10.0) / 20.0 * 100.0))


def load_sample(sample_dir: Path, err_pct: float):
    meta = json.load(open(sample_dir / "meta.json"))
    cam = meta["target_camera"]
    img_t0 = load_rgb(sample_dir / "input" / "t0" / f"{cam}.jpg")
    img_t1 = load_rgb(sample_dir / "input" / "t1" / f"{cam}.jpg")
    gt = load_rgb(sample_dir / "target" / f"{cam}.jpg")
    mean_pred = ((img_t0.astype(np.float32) + img_t1.astype(np.float32)) * 0.5).astype(np.uint8)
    err = np.mean(np.abs(mean_pred.astype(np.float32) - gt.astype(np.float32)), axis=2)
    good = err <= np.percentile(err, err_pct)
    flow_mag, comp_err, sigma, _ = compute_flow_maps(img_t0, img_t1)
    depth = get_lidar_depth(sample_dir, cam, "target")
    z_ref = float(np.median(depth[np.isfinite(depth) & (depth > 0)]))
    return dict(
        meta=meta, mean_pred=mean_pred, gt=gt, good=good,
        flow_mag=flow_mag, comp_err=comp_err, sigma=sigma,
        depth=depth, z_ref=z_ref,
    )


def eval_ratio(caches: list[dict], base_params: MaskParams, ratio: float) -> RatioStats:
    p = replace(base_params, far_min_ratio=ratio, name=f"far_r{ratio}")

    acc = {k: [] for k in [
        "far_gate", "static", "good_far", "overlap",
        "iou", "f1", "precision", "recall",
        "psnr_full", "psnr_far_gate", "psnr_static_far", "psnr_static_good", "mae_static_far",
    ]}

    for c in caches:
        gate = far_or_no_depth_gate(c["depth"], c["z_ref"], ratio, 0.0, p.include_no_depth)
        static = build_static_mask(c["flow_mag"], c["comp_err"], c["depth"], c["sigma"], p, c["z_ref"])
        good_far = c["good"] & gate
        static_good = static & good_far
        met = mask_metrics(static, good_far)

        acc["far_gate"].append(gate.mean())
        acc["static"].append(static.mean())
        acc["good_far"].append(good_far.mean())
        acc["overlap"].append(static_good.mean())
        acc["iou"].append(met["iou"])
        acc["f1"].append(met["f1"])
        acc["precision"].append(met["precision"])
        acc["recall"].append(met["recall"])
        acc["psnr_full"].append(psnr(c["mean_pred"], c["gt"]))
        acc["psnr_far_gate"].append(psnr(c["mean_pred"], c["gt"], gate))
        acc["psnr_static_far"].append(psnr(c["mean_pred"], c["gt"], static))
        acc["psnr_static_good"].append(psnr(c["mean_pred"], c["gt"], static_good))
        acc["mae_static_far"].append(mae(c["mean_pred"], c["gt"], static))

    psnr_sf = float(np.nanmean(acc["psnr_static_far"]))
    return RatioStats(
        ratio=ratio,
        n_samples=len(caches),
        far_gate_pct=100 * float(np.mean(acc["far_gate"])),
        static_pct=100 * float(np.mean(acc["static"])),
        good_far_pct=100 * float(np.mean(acc["good_far"])),
        overlap_pct=100 * float(np.mean(acc["overlap"])),
        iou=float(np.mean(acc["iou"])),
        f1=float(np.mean(acc["f1"])),
        precision=float(np.mean(acc["precision"])),
        recall=float(np.mean(acc["recall"])),
        psnr_full=float(np.mean(acc["psnr_full"])),
        psnr_far_gate=float(np.nanmean(acc["psnr_far_gate"])),
        psnr_static_far=psnr_sf,
        psnr_static_good=float(np.nanmean(acc["psnr_static_good"])),
        mae_static_far=float(np.nanmean(acc["mae_static_far"])),
        score_static_far=score_from_psnr(psnr_sf),
    )


def pick_recommendations(stats: list[RatioStats]) -> dict[str, float]:
    """Several strategies for different goals."""
    far_only = [s for s in stats if s.ratio >= 1.5]
    out = {}

    # max F1 on far|no-depth well-pred (any ratio)
    out["max_f1"] = max(stats, key=lambda s: s.f1).ratio

    # far-only: best F1 among ratio >= 1.5
    if far_only:
        out["far_best_f1"] = max(far_only, key=lambda s: s.f1).ratio
        viable = [s for s in far_only if s.static_pct >= 5.0]
        if viable:
            out["far_best_psnr"] = max(viable, key=lambda s: s.psnr_static_far).ratio
            out["far_balanced"] = max(
                viable,
                key=lambda s: s.f1 * 0.5 + (s.psnr_static_far - 20.0) * 0.05 + s.iou * 0.3,
            ).ratio
        else:
            out["far_best_psnr"] = far_only[-1].ratio
            out["far_balanced"] = 2.0
    else:
        out["far_best_f1"] = 2.0
        out["far_best_psnr"] = 2.5
        out["far_balanced"] = 2.0

    return out


def print_table(stats: list[RatioStats]):
    hdr = (
        f"{'ratio':>5} {'far%':>5} {'static%':>7} {'goodF%':>6} {'ovlp%':>5} "
        f"{'IoU':>5} {'F1':>5} {'P':>5} {'R':>5} "
        f"{'PSNR all':>8} {'PSNR far':>8} {'PSNR st':>8} {'MAE st':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for s in stats:
        print(
            f"{s.ratio:5.1f} {s.far_gate_pct:5.1f} {s.static_pct:7.1f} {s.good_far_pct:6.1f} "
            f"{s.overlap_pct:5.1f} {s.iou:5.3f} {s.f1:5.3f} {s.precision:5.2f} {s.recall:5.2f} "
            f"{s.psnr_full:8.2f} {s.psnr_far_gate:8.2f} {s.psnr_static_far:8.2f} {s.mae_static_far:6.1f}"
        )


def save_plot(stats: list[RatioStats], out: Path, recommended: float):
    ratios = [s.ratio for s in stats]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Far boundary sweep: mean(t0,t1) prediction quality", fontsize=11)

    ax = axes[0, 0]
    ax.plot(ratios, [s.far_gate_pct for s in stats], "o-", label="far gate")
    ax.plot(ratios, [s.static_pct for s in stats], "s-", label="static far")
    ax.plot(ratios, [s.good_far_pct for s in stats], "^-", label="well pred far")
    ax.plot(ratios, [s.overlap_pct for s in stats], "d-", label="static & good")
    ax.axvline(recommended, color="red", ls="--", alpha=0.7, label=f"rec={recommended:.1f}")
    ax.set_xlabel("far_min_ratio × z_ref")
    ax.set_ylabel("coverage %")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(ratios, [s.iou for s in stats], "o-", label="IoU")
    ax.plot(ratios, [s.f1 for s in stats], "s-", label="F1")
    ax.plot(ratios, [s.precision for s in stats], "^-", label="precision")
    ax.plot(ratios, [s.recall for s in stats], "d-", label="recall")
    ax.axvline(recommended, color="red", ls="--", alpha=0.7)
    ax.set_xlabel("far_min_ratio × z_ref")
    ax.set_ylabel("vs well pred (far)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(ratios, [s.psnr_full for s in stats], "o-", label="full frame")
    ax.plot(ratios, [s.psnr_far_gate for s in stats], "s-", label="far gate")
    ax.plot(ratios, [s.psnr_static_far for s in stats], "^-", label="static far", lw=2)
    ax.plot(ratios, [s.psnr_static_good for s in stats], "d-", label="static & good")
    ax.axvline(recommended, color="red", ls="--", alpha=0.7)
    ax.set_xlabel("far_min_ratio × z_ref")
    ax.set_ylabel("PSNR (dB)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(ratios, [s.mae_static_far for s in stats], "o-", color="C4")
    ax.axvline(recommended, color="red", ls="--", alpha=0.7)
    ax.set_xlabel("far_min_ratio × z_ref")
    ax.set_ylabel("MAE on static far")
    ax.grid(True, alpha=0.3)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=Path,
                   default=Path(r"C:\Users\adel\Downloads\cv_dataset\final_dataset_v5_participants"))
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--num-samples", type=int, default=0,
                   help="0 = all samples in split")
    p.add_argument("--ratios", type=str, default="0.5,0.8,1.0,1.2,1.5,2.0,2.5,3.0,3.5")
    p.add_argument("--err-percentile", type=float, default=30.0)
    p.add_argument("--config", type=Path, default=Path("static_mask_tune_best.json"))
    p.add_argument("--out-json", type=Path, default=Path("far_boundary_sweep.json"))
    p.add_argument("--out-csv", type=Path, default=Path("far_boundary_sweep.csv"))
    p.add_argument("--out-plot", type=Path, default=Path("methods_gallery/_static_masks/far_boundary_sweep.png"))
    args = p.parse_args()

    ratios = [float(x) for x in args.ratios.split(",")]
    base_params = load_best_params(args.config)
    # keep flow/comp from tuned config; ratio swept separately
    base_params = replace(base_params, far_min_ratio=1.0)

    split_dir = args.dataset_dir / args.split
    all_s = sorted(d for d in split_dir.iterdir() if d.is_dir())
    if args.num_samples > 0:
        step = max(1, len(all_s) // args.num_samples)
        sample_dirs = [all_s[i * step] for i in range(min(args.num_samples, len(all_s)))]
    else:
        sample_dirs = all_s

    print(f"Loading {len(sample_dirs)} {args.split} samples...", flush=True)
    caches = []
    for i, sd in enumerate(sample_dirs):
        caches.append(load_sample(sd, args.err_percentile))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(sample_dirs)}", flush=True)

    print(f"\nSweep ratios {ratios}  (flow<={base_params.flow_t}, comp<={base_params.comp_t})\n")
    stats = [eval_ratio(caches, base_params, r) for r in ratios]
    print_table(stats)

    recommendations = pick_recommendations(stats)
    rec_ratio = recommendations["far_balanced"]
    rec = next(s for s in stats if s.ratio == rec_ratio)
    print(f"\nRecommendations (far_min_ratio x z_ref):")
    for name, ratio in recommendations.items():
        s = next(x for x in stats if x.ratio == ratio)
        print(f"  {name:14s} r={ratio:.1f}  static={s.static_pct:.1f}%  "
              f"F1={s.f1:.3f}  PSNR_st={s.psnr_static_far:.2f} dB")
    print(f"\nDefault for benchmark: far_balanced = {rec_ratio:.1f}x z_ref")
    print(f"  PSNR static_far = {rec.psnr_static_far:.2f} dB  (full frame {rec.psnr_full:.2f} dB, +{rec.psnr_static_far - rec.psnr_full:.2f} dB)")

    out_data = {
        "split": args.split,
        "n_samples": len(caches),
        "err_percentile": args.err_percentile,
        "flow_t": base_params.flow_t,
        "comp_t": base_params.comp_t,
        "include_no_depth": base_params.include_no_depth,
        "recommendations": recommendations,
        "recommended_far_min_ratio": rec_ratio,
        "recommended_stats": rec.to_dict(),
        "sweep": [s.to_dict() for s in stats],
    }
    args.out_json.write_text(json.dumps(out_data, indent=2), encoding="utf-8")

    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0].to_dict().keys()))
        w.writeheader()
        w.writerows(s.to_dict() for s in stats)

    save_plot(stats, args.out_plot, rec_ratio)
    print(f"\nSaved {args.out_json.resolve()}")
    print(f"Saved {args.out_csv.resolve()}")
    print(f"Saved {args.out_plot.resolve()}")


if __name__ == "__main__":
    main()
