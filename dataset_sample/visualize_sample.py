#!/usr/bin/env python3
"""
Visualize one sample of the multi-camera + LiDAR dataset, including its precomputes.

Outputs (next to this script):
  <id>_overview.png   - input cameras at t0 and t1 + the target image
  <id>_pipeline.png   - precompute artifacts (only the ones that exist for the sample)

Usage:
    python visualize_sample.py cv_dataset/final_dataset_v5_participants/train/<id>
    python visualize_sample.py --all          # every sample under the base train dir

Dependencies:
    pip install numpy matplotlib pillow
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

CAMERAS = ["front", "left_fwd", "left_bwd", "right_fwd", "right_bwd", "rear"]
BASE_TREE = "final_dataset_v5_participants"


# ----------------------------------------------------------------------------- helpers
def load_image(path: Path):
    if not path.exists():
        return None
    try:
        return np.asarray(Image.open(path).convert("RGB"))
    except Exception as exc:  # noqa: BLE001
        print(f"  ! could not read {path.name}: {exc}", file=sys.stderr)
        return None


def load_npy_as_image(path: Path):
    """Load a .npy array and squeeze it into something imshow-able."""
    if not path.exists():
        return None
    try:
        arr = np.load(path)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! could not read {path.name}: {exc}", file=sys.stderr)
        return None
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        if hi > lo:
            a = (a - lo) / (hi - lo)
        arr = (a * 255).clip(0, 255).astype(np.uint8)
    return arr


def show(ax, img, title, cmap=None):
    if img is None:
        ax.set_facecolor("#222222")
        ax.text(0.5, 0.5, "— missing —", color="#888888",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
    else:
        ax.imshow(img, cmap=cmap)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def find_dataset_root(sample_dir: Path) -> Path | None:
    """Walk up until we find the folder that contains BASE_TREE (the cv_dataset root)."""
    for parent in sample_dir.resolve().parents:
        if (parent / BASE_TREE).is_dir():
            return parent
    return None


def lidar_point_count(sample_dir: Path, meta: dict) -> int | None:
    info = meta.get("lidar_info") or {}
    if "n_points" in info:
        return int(info["n_points"])
    return None


# ----------------------------------------------------------------------------- panels
def overview(sample_dir: Path, meta: dict, out_path: Path) -> None:
    target_cam = meta.get("target_camera", "?")
    n_lidar = lidar_point_count(sample_dir, meta)
    n_cols = len(CAMERAS)
    fig, axes = plt.subplots(3, n_cols, figsize=(3 * n_cols, 8.2))

    for j, cam in enumerate(CAMERAS):
        show(axes[0, j], load_image(sample_dir / "input" / "t0" / f"{cam}.jpg"), f"t0 · {cam}")
        show(axes[1, j], load_image(sample_dir / "input" / "t1" / f"{cam}.jpg"), f"t1 · {cam}")
        if cam == target_cam:
            show(axes[2, j], load_image(sample_dir / "target" / f"{cam}.jpg"), f"TARGET · {cam}")
        else:
            axes[2, j].axis("off")

    lidar_txt = f"{n_lidar:,} pts" if n_lidar is not None else "n/a"
    fig.suptitle(
        f"{meta.get('sample_id', sample_dir.name)}\n"
        f"target = {target_cam}   |   delta_s = {meta.get('delta_s')}   |   lidar = {lidar_txt}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  saved {out_path}")


def pipeline(root: Path, sample_id: str, out_path: Path) -> None:
    """Collect whatever precompute previews exist and lay them out."""
    candidates = [
        ("RIFE interp (irife)", root / "rife_refinement_baked" / "train" / sample_id / "irife.jpg", "img"),
        ("RIFE raw out",        root / "rife_guided_blend_v1" / "train" / sample_id / "rife_raw_out.jpg", "img"),
        ("RIFE-guided out",     root / "rife_guided_blend_v1" / "train" / sample_id / "rife_guided_out.jpg", "img"),
        ("ours_pred",           root / "rife_guided_blend_v1" / "train" / sample_id / "ours_pred.jpg", "img"),
        ("side warp mix",       root / "side_warp_mix_v1" / "train" / sample_id / "warp_mix.jpg", "img"),
        ("multiview consensus", root / "multiview_warps" / "train" / sample_id / "consensus_tuned.npy", "npy"),
        ("coverage",            root / "multiview_warps" / "train" / sample_id / "coverage.npy", "npy"),
        ("lidar trust",         root / "precomputed_lidar_trust" / "train" / sample_id / "lidar_trust.npy", "npy"),
        ("pseudo pred_rgb",     root / "pseudo_v2_train_rgb" / sample_id / "pred_rgb.npy", "npy"),
        ("target (blend)",      root / "rife_guided_blend_v1" / "train" / sample_id / "target.jpg", "img"),
    ]
    items = []
    for title, path, kind in candidates:
        img = load_image(path) if kind == "img" else load_npy_as_image(path)
        if img is not None:
            items.append((title, img))

    if not items:
        return

    n = len(items)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows + 0.6), squeeze=False)
    for k in range(rows * cols):
        ax = axes[k // cols][k % cols]
        if k < n:
            show(ax, items[k][1], items[k][0])
        else:
            ax.axis("off")
    fig.suptitle(f"{sample_id} — precompute pipeline", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  saved {out_path}")


# ----------------------------------------------------------------------------- driver
def visualize(sample_dir: Path) -> None:
    meta_path = sample_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {sample_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    sample_id = meta.get("sample_id", sample_dir.name)
    script_dir = Path(__file__).resolve().parent

    overview(sample_dir, meta, script_dir / f"{sample_id}_overview.png")

    root = find_dataset_root(sample_dir)
    if root is not None:
        pipeline(root, sample_id, script_dir / f"{sample_id}_pipeline.png")
    else:
        print("  (dataset root not found — skipping pipeline panel)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sample", nargs="?", help="path to a base sample folder")
    ap.add_argument("--all", action="store_true",
                    help="process every sample under cv_dataset/<BASE_TREE>/train")
    args = ap.parse_args()

    if args.all:
        base = (Path(__file__).resolve().parent
                / "cv_dataset" / BASE_TREE / "train")
        dirs = sorted(p for p in base.iterdir() if (p / "meta.json").exists()) if base.is_dir() else []
        if not dirs:
            sys.exit(f"no samples found in {base}")
        for d in dirs:
            print(d.name)
            visualize(d)
        return

    if not args.sample:
        ap.error("give a base sample folder, or use --all")
    visualize(Path(args.sample))


if __name__ == "__main__":
    main()
