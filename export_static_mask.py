"""Visualize far-only static zones vs well-predicted regions."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from layered_parallax import get_lidar_depth_raw
from static_mask import (
    FAR_DEFAULT,
    MaskParams,
    build_static_mask,
    compute_flow_maps,
    describe_params,
    far_or_no_depth_gate,
    load_best_params,
    mask_metrics,
    z_ref_from_lidar,
)


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def abs_err(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(pred.astype(np.float32) - gt.astype(np.float32)), axis=2)


def save_figure(
    img_t0, img_t1, gt, mean_pred,
    flow_mag, comp_err, depth, sigma_map,
    far_gate, masks: dict[str, np.ndarray],
    good_mask, good_far, out_path: Path, title: str,
):
    err_mean = abs_err(mean_pred, gt)
    primary = masks.get("far", next(iter(masks.values())))

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle(title, fontsize=10)

    imgs = [
        (img_t0, "t0", None),
        (img_t1, "t1", None),
        (gt, "TARGET (GT)", None),
        (mean_pred, "mean(t0,t1)", None),
        (flow_mag, "flow_mag", "plasma"),
        (comp_err, "comp_err", "hot"),
        (depth, "raw LiDAR depth", "viridis"),
        (far_gate.astype(float), f"far | no-depth ({100*far_gate.mean():.0f}%)", "gray"),
        (err_mean, "|mean - GT|", "hot"),
        (good_far.astype(float), f"WELL pred far ({100*good_far.mean():.0f}%)", "gray"),
    ]
    for ax, (data, t, cmap) in zip(axes.flat[:10], imgs):
        if cmap == "gray":
            ax.imshow(data, cmap="gray", vmin=0, vmax=1 if data.max() <= 1 else None)
        elif cmap:
            im = ax.imshow(data, cmap=cmap)
            plt.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.imshow(data)
        ax.set_title(t, fontsize=9)
        ax.axis("off")

    for ax, (name, m) in zip(axes.flat[10:12], list(masks.items())[:2]):
        met = mask_metrics(m, good_far)
        ax.imshow(m.astype(float), cmap="gray", vmin=0, vmax=1)
        ax.set_title(
            f"{name}\n{100*m.mean():.0f}% px  IoU={met['iou']:.2f} F1={met['f1']:.2f}",
            fontsize=8,
        )
        ax.axis("off")

    ax = axes[2, 2]
    near_excluded = (~far_gate) & (good_mask)
    ax.imshow(near_excluded.astype(float), cmap="hot", vmin=0, vmax=1)
    ax.set_title(f"near WELL pred\n(excluded, {100*near_excluded.mean():.0f}%)")
    ax.axis("off")

    ax = axes[2, 3]
    over = img_t0.copy()
    green = np.zeros_like(over)
    green[..., 1] = 255
    orange = np.zeros_like(over)
    orange[..., 0] = 255
    orange[..., 1] = 120
    far_m = masks.get("far", primary)
    over[good_far] = (0.45 * over[good_far] + 0.55 * green[good_far]).astype(np.uint8)
    only_far_static = far_m & ~good_far
    over[only_far_static] = (0.55 * over[only_far_static] + 0.45 * orange[only_far_static]).astype(np.uint8)
    ax.imshow(over)
    ax.set_title("green=well pred far\norange=far static only")
    ax.axis("off")

    ax = axes[3, 0]
    ax.imshow((~primary).astype(float), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"not far-static ({100*(~primary).mean():.0f}%)")
    ax.axis("off")

    ax = axes[3, 1]
    ax.axis("off")
    lines = []
    for name, m in masks.items():
        met = mask_metrics(m, good_far)
        lines.append(
            f"{name}: {100*m.mean():.0f}% px, IoU={met['iou']:.2f}, "
            f"overlap_far_good={100*met['overlap_with_good']:.0f}%"
        )
    fig.text(
        0.02, 0.02, "\n".join(lines),
        fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    for ax in axes.flat[12:]:
        if ax.has_data() or ax.images:
            continue
        ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    stem = out_path.stem
    far_m = masks.get("far", primary)
    static_good = far_m & good_far
    Image.fromarray((far_m.astype(np.uint8) * 255)).save(out_path.parent / f"{stem}_static_mask.png")
    Image.fromarray((good_far.astype(np.uint8) * 255)).save(out_path.parent / f"{stem}_good_pred_mask.png")
    Image.fromarray((static_good.astype(np.uint8) * 255)).save(out_path.parent / f"{stem}_static_and_good.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("methods_gallery/_static_masks"))
    p.add_argument("--err-percentile", type=float, default=30.0)
    p.add_argument("--config", type=Path, default=Path("static_mask_tune_best.json"))
    args = p.parse_args()

    meta = json.load(open(args.sample_dir / "meta.json"))
    cam = meta["target_camera"]
    img_t0 = load_rgb(args.sample_dir / "input" / "t0" / f"{cam}.jpg")
    img_t1 = load_rgb(args.sample_dir / "input" / "t1" / f"{cam}.jpg")
    gt = load_rgb(args.sample_dir / "target" / f"{cam}.jpg")

    flow_mag, comp_err, sigma_map, _ = compute_flow_maps(img_t0, img_t1)
    depth_raw = get_lidar_depth_raw(args.sample_dir, cam, "target")
    z_ref = z_ref_from_lidar(depth_raw)

    mean_pred = ((img_t0.astype(np.float32) + img_t1.astype(np.float32)) * 0.5).astype(np.uint8)
    err_mean = abs_err(mean_pred, gt)
    good_mask = err_mean <= np.percentile(err_mean, args.err_percentile)

    best_p = load_best_params(args.config)
    default_far_p = MaskParams.from_dict(FAR_DEFAULT)

    far_gate = far_or_no_depth_gate(
        depth_raw, z_ref, best_p.far_min_ratio, best_p.far_min_abs, best_p.include_no_depth,
    )
    good_far = good_mask & far_gate

    masks = {
        "default_far": build_static_mask(flow_mag, comp_err, None, sigma_map, default_far_p, z_ref, depth_raw),
        "far": build_static_mask(flow_mag, comp_err, None, sigma_map, best_p, z_ref, depth_raw),
    }

    sid = meta["sample_id"]
    title = (
        f"{sid}  cam={cam}  z_ref={z_ref:.1f}m\n"
        f"default: {describe_params(default_far_p)}\n"
        f"best: {describe_params(best_p)}"
    )
    out = args.out_dir / f"{sid}_static_analysis.png"
    save_figure(
        img_t0, img_t1, gt, mean_pred,
        flow_mag, comp_err, depth_raw, sigma_map,
        far_gate, masks, good_mask, good_far, out, title,
    )

    fm = mask_metrics(masks["far"], good_far)
    print(f"Saved {out}")
    print(f"  far gate: {100*far_gate.mean():.1f}% px  well_pred far: {100*good_far.mean():.1f}%")
    print(f"  far static ({best_p.name}): {100*masks['far'].mean():.1f}% px, IoU={fm['iou']:.3f}")


if __name__ == "__main__":
    main()
