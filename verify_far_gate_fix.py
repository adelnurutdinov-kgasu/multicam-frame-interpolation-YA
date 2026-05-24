"""Verify far gate fix: no-LiDAR zones must not be wrongly excluded (jurita test)."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from layered_parallax import get_lidar_depth, get_lidar_depth_raw
from static_mask import (
    MaskParams,
    build_static_mask,
    compute_flow_maps,
    far_or_no_depth_gate,
    lidar_hit_mask,
    load_best_params,
    z_ref_from_lidar,
)


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def far_gate_buggy_filled(
    depth_filled: np.ndarray,
    z_ref: float,
    far_min_ratio: float,
    include_no_depth: bool,
) -> np.ndarray:
    """Old behaviour: inpainted depth treats sky as valid (often near) -> excluded."""
    valid = np.isfinite(depth_filled) & (depth_filled > 0)
    min_z = far_min_ratio * z_ref
    far = valid & (depth_filled >= min_z)
    no_d = (~valid) & include_no_depth
    return far | no_d


def overlay(img: np.ndarray, mask: np.ndarray, rgb: tuple[int, int, int], alpha: float = 0.5) -> np.ndarray:
    out = img.copy()
    tint = np.zeros_like(out)
    tint[..., 0], tint[..., 1], tint[..., 2] = rgb
    out[mask] = (alpha * tint[mask] + (1 - alpha) * out[mask]).astype(np.uint8)
    return out


def core_static(flow_mag, comp_err, sigma, params):
    p = replace(params, far_min_ratio=0.0, far_min_abs=0.0)
    return build_static_mask(flow_mag, comp_err, None, sigma, p, 10.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample-dir", type=Path,
                   default=Path(r"C:\Users\adel\Downloads\cv_dataset\final_dataset_v5_participants\train"
                              r"\2025-10-26_18_34_10_18_54_23_jurita_1761494142800055000__000"))
    p.add_argument("--far-ratio", type=float, default=2.0)
    p.add_argument("--out", type=Path,
                   default=Path("methods_gallery/_static_masks/examples/jurita_far_gate_fix.png"))
    args = p.parse_args()

    meta = json.load(open(args.sample_dir / "meta.json"))
    cam = meta["target_camera"]
    img_t0 = load_rgb(args.sample_dir / "input" / "t0" / f"{cam}.jpg")

    flow_mag, comp_err, sigma, _ = compute_flow_maps(
        img_t0, load_rgb(args.sample_dir / "input" / "t1" / f"{cam}.jpg"),
    )
    depth_filled = get_lidar_depth(args.sample_dir, cam, "target")
    depth_raw = get_lidar_depth_raw(args.sample_dir, cam, "target")
    z_ref = z_ref_from_lidar(depth_raw)

    base = load_best_params()
    params = replace(base, far_min_ratio=args.far_ratio)

    hit = lidar_hit_mask(depth_raw)
    no_hit = ~hit
    min_z = args.far_ratio * z_ref

    gate_old = far_gate_buggy_filled(depth_filled, z_ref, args.far_ratio, True)
    gate_new = far_or_no_depth_gate(depth_raw, z_ref, args.far_ratio, 0.0, True)

    core = core_static(flow_mag, comp_err, sigma, params)
    static_old = core & gate_old
    static_new = build_static_mask(flow_mag, comp_err, depth_filled, sigma, params, z_ref, depth_raw)

    # wrongly excluded: no LiDAR, passes flow/comp, was cut by old far gate
    wrongly_old = no_hit & core & ~gate_old
    wrongly_new = no_hit & core & ~gate_new
    rescued = wrongly_old & static_new

    print(f"Sample: {meta['sample_id']}  cam={cam}  z_ref={z_ref:.1f}m  cutoff={min_z:.1f}m")
    print(f"  no LiDAR pixels: {100*no_hit.mean():.1f}%")
    print(f"  far gate OLD (filled depth): {100*gate_old.mean():.1f}%  no-hit in gate: {100*(gate_old & no_hit).mean():.1f}%")
    print(f"  far gate NEW (raw depth):    {100*gate_new.mean():.1f}%  no-hit in gate: {100*(gate_new & no_hit).mean():.1f}%")
    print(f"  static OLD: {100*static_old.mean():.1f}%  NEW: {100*static_new.mean():.1f}%")
    print(f"  no-hit + flow/comp ok but OLD excluded: {100*wrongly_old.mean():.1f}%")
    print(f"  rescued by fix: {100*rescued.mean():.1f}%")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        f"jurita far gate fix  ratio={args.far_ratio}x z_ref  "
        f"(red = no-LiDAR wrongly cut by OLD gate)",
        fontsize=10,
    )

    axes[0, 0].imshow(img_t0)
    axes[0, 0].set_title("t0")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(np.where(hit, depth_raw, np.nan), cmap="viridis")
    axes[0, 1].set_title(f"raw LiDAR  no-hit={100*no_hit.mean():.0f}%")
    axes[0, 1].axis("off")

    vis_old = overlay(img_t0, static_old, (80, 255, 80))
    vis_old = overlay(vis_old, wrongly_old, (255, 60, 60), 0.55)
    axes[0, 2].imshow(vis_old)
    axes[0, 2].set_title(f"OLD static {100*static_old.mean():.0f}%\nred=wrongly cut no-LiDAR")
    axes[0, 2].axis("off")

    vis_new = overlay(img_t0, static_new, (80, 255, 80))
    axes[0, 3].imshow(vis_new)
    axes[0, 3].set_title(f"NEW static {100*static_new.mean():.0f}%\nno-hit always in far gate")
    axes[0, 3].axis("off")

    axes[1, 0].imshow(gate_old.astype(float), cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title(f"OLD far gate {100*gate_old.mean():.0f}%")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(gate_new.astype(float), cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title(f"NEW far gate {100*gate_new.mean():.0f}%")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(wrongly_old.astype(float), cmap="hot", vmin=0, vmax=1)
    axes[1, 2].set_title(f"wrongly excluded OLD\n{100*wrongly_old.mean():.1f}% px")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(rescued.astype(float), cmap="hot", vmin=0, vmax=1)
    axes[1, 3].set_title(f"rescued by fix\n{100*rescued.mean():.1f}% px")
    axes[1, 3].axis("off")

    legend = [
        mpatches.Patch(color=(0.3, 1, 0.3), label="static"),
        mpatches.Patch(color=(1, 0.24, 0.24), label="no-LiDAR wrongly cut (OLD)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=9)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved {args.out.resolve()}")


if __name__ == "__main__":
    main()
