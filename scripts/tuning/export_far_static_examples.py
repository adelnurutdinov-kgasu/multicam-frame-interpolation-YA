"""Export 10 example images: what far-static mask actually selects."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
import sys
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from layered_parallax import get_lidar_depth_raw
from static_mask import (
    MaskParams, build_static_mask, compute_flow_maps, far_or_no_depth_gate,
    load_best_params, z_ref_from_lidar,
)


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def overlay(img: np.ndarray, mask: np.ndarray, rgb: tuple[int, int, int], alpha: float = 0.55) -> np.ndarray:
    out = img.copy()
    tint = np.zeros_like(out)
    tint[..., 0], tint[..., 1], tint[..., 2] = rgb
    m = mask & np.any(img > 0, axis=2)  # ignore black borders if any
    out[m] = (alpha * tint[m] + (1 - alpha) * out[m]).astype(np.uint8)
    return out


def classify_static(static: np.ndarray, depth_raw: np.ndarray, min_z: float) -> dict[str, np.ndarray]:
    raw_hit = np.isfinite(depth_raw) & (depth_raw > 0)
    return {
        "lidar_far": static & raw_hit & (depth_raw >= min_z),
        "no_lidar_static": static & ~raw_hit,
        "near_rejected": (~static) & raw_hit & (depth_raw < min_z),
    }


def pct(mask: np.ndarray) -> float:
    return 100.0 * mask.mean()


def save_example(
    sample_dir: Path,
    params: MaskParams,
    z_ref: float,
    out_path: Path,
):
    meta = json.load(open(sample_dir / "meta.json"))
    cam = meta["target_camera"]
    sid = meta["sample_id"]
    img_t0 = load_rgb(sample_dir / "input" / "t0" / f"{cam}.jpg")
    img_t1 = load_rgb(sample_dir / "input" / "t1" / f"{cam}.jpg")
    gt = load_rgb(sample_dir / "target" / f"{cam}.jpg")

    flow_mag, comp_err, sigma, _ = compute_flow_maps(img_t0, img_t1)
    depth_raw = get_lidar_depth_raw(sample_dir, cam, "target")
    min_z = params.far_min_ratio * z_ref

    gate = far_or_no_depth_gate(depth_raw, z_ref, params.far_min_ratio, params.far_min_abs, params.include_no_depth)
    static = build_static_mask(flow_mag, comp_err, None, sigma, params, z_ref, depth_raw)
    parts = classify_static(static, depth_raw, min_z)

    vis = img_t0.copy()
    vis = overlay(vis, parts["lidar_far"], (80, 255, 80), 0.52)
    vis = overlay(vis, parts["no_lidar_static"], (120, 200, 255), 0.48)

    mean_pred = ((img_t0.astype(np.float32) + img_t1.astype(np.float32)) * 0.5).astype(np.uint8)
    err = np.mean(np.abs(mean_pred.astype(np.float32) - gt.astype(np.float32)), axis=2)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        f"{sid}\ncam={cam}  z_ref={z_ref:.1f}m  far>={params.far_min_ratio:.1f}x z_ref  "
        f"static={pct(static):.1f}%",
        fontsize=9,
    )

    axes[0, 0].imshow(img_t0)
    axes[0, 0].set_title("t0")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(vis)
    axes[0, 1].set_title("static overlay\ngreen=LiDAR far  blue=sky/no-LiDAR")
    axes[0, 1].axis("off")

    im = axes[0, 2].imshow(np.where(raw_hit := (np.isfinite(depth_raw) & (depth_raw > 0)), depth_raw, np.nan), cmap="viridis")
    axes[0, 2].set_title(f"raw LiDAR depth (cutoff {min_z:.1f}m)")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
    axes[0, 2].axis("off")

    axes[1, 0].imshow(static.astype(float), cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title(f"static far mask ({pct(static):.1f}%)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(gate.astype(float), cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title(f"far gate ({pct(gate):.1f}%)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(err, cmap="hot", vmin=0, vmax=np.percentile(err, 95))
    static_err = err[static]
    mae_st = float(static_err.mean()) if static_err.size else 0
    axes[1, 2].set_title(f"|mean-GT| err  MAE on static={mae_st:.1f}")
    axes[1, 2].axis("off")

    legend = [
        mpatches.Patch(color=(0.3, 1, 0.3), label=f"LiDAR far static: {pct(parts['lidar_far']):.1f}%"),
        mpatches.Patch(color=(0.47, 0.78, 1), label=f"no-LiDAR static (sky): {pct(parts['no_lidar_static']):.1f}%"),
        mpatches.Patch(color=(0.5, 0.5, 0.5), label=f"near rejected: {pct(parts['near_rejected']):.1f}%"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=8, frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "sample_id": sid,
        "camera": cam,
        "z_ref": z_ref,
        "static_pct": pct(static),
        "static_lidar_far_pct": pct(parts["lidar_far"]),
        "static_no_lidar_pct": pct(parts["no_lidar_static"]),
        "mae_on_static": mae_st,
        "png": str(out_path),
    }


def save_grid(examples: list[dict], img_paths: list[Path], out: Path, title: str):
    n = len(img_paths)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for ax, path, info in zip(axes.flat, img_paths, examples):
        img = np.array(Image.open(path))
        ax.imshow(img)
        ax.set_title(
            f"{info['static_pct']:.0f}% static\n"
            f"sky {info['static_no_lidar_pct']:.0f}% + lidar {info['static_lidar_far_pct']:.0f}%",
            fontsize=8,
        )
    fig.suptitle(title, fontsize=11)
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=Path,
                   default=Path(r"C:\Users\adel\Downloads\cv_dataset\final_dataset_v5_participants"))
    p.add_argument("--split", default="train")
    p.add_argument("--num", type=int, default=10)
    p.add_argument("--samples", type=str, default="",
                   help="comma-separated sample_id substrings (overrides uniform pick)")
    p.add_argument("--far-ratio", type=float, default=2.0)
    p.add_argument("--config", type=Path, default=Path("static_mask_tune_best.json"))
    p.add_argument("--out-dir", type=Path, default=Path("methods_gallery/_static_masks/examples"))
    args = p.parse_args()

    base = load_best_params(args.config)
    params = replace(base, far_min_ratio=args.far_ratio)

    split_dir = args.dataset_dir / args.split
    all_s = sorted(d for d in split_dir.iterdir() if d.is_dir())
    if args.samples.strip():
        keys = [k.strip() for k in args.samples.split(",") if k.strip()]
        sample_dirs = []
        for k in keys:
            match = next((d for d in all_s if k in d.name), None)
            if match is None:
                print(f"WARN: no sample matching '{k}'", flush=True)
            else:
                sample_dirs.append(match)
    else:
        step = max(1, len(all_s) // args.num)
        sample_dirs = [all_s[i * step] for i in range(min(args.num, len(all_s)))]

    print(f"Exporting {len(sample_dirs)} examples (far_ratio={args.far_ratio})...", flush=True)
    examples = []
    paths = []
    for i, sd in enumerate(sample_dirs):
        meta = json.load(open(sd / "meta.json"))
        cam = meta["target_camera"]
        depth_raw = get_lidar_depth_raw(sd, cam, "target")
        z_ref = z_ref_from_lidar(depth_raw)
        out = args.out_dir / f"{meta['sample_id']}_far_static_example.png"
        info = save_example(sd, params, z_ref, out)
        examples.append(info)
        paths.append(out)
        print(f"  [{i+1}/{len(sample_dirs)}] {info['sample_id'][:40]}...  "
              f"static={info['static_pct']:.1f}%  lidar={info['static_lidar_far_pct']:.1f}%  "
              f"sky={info['static_no_lidar_pct']:.1f}%", flush=True)

    grid_out = args.out_dir / "index_10_examples.png"
    save_grid(
        examples, paths, grid_out,
        f"Far static mask examples (ratio={args.far_ratio}x z_ref, flow<={params.flow_t}, comp<={params.comp_t})\n"
        "green=LiDAR far  blue=sky/no-LiDAR",
    )

    manifest = {
        "far_min_ratio": args.far_ratio,
        "flow_t": params.flow_t,
        "comp_t": params.comp_t,
        "examples": examples,
        "grid": str(grid_out),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nGrid: {grid_out.resolve()}")
    print(f"Manifest: {(args.out_dir / 'manifest.json').resolve()}")


if __name__ == "__main__":
    main()
