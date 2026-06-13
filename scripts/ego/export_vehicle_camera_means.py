"""
Mean target RGB per (vehicle, camera) for manual ego-mask drawing.

For each group: up to N diverse trips -> align -> mean -> PNG.

Usage:
  python export_vehicle_camera_means.py
  python export_vehicle_camera_means.py --max-scenes 10 --out methods_gallery/_ego_manual_masks
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
import sys
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from test_ego_consistent_edges import DEFAULT_DATASET, SAMPLE_RE, diversify_paths

DEFAULT_OUT = Path(__file__).resolve().parent / "methods_gallery/_ego_manual_masks"


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def align_stack(imgs: list[np.ndarray]) -> np.ndarray:
    h = max(i.shape[0] for i in imgs)
    w = max(i.shape[1] for i in imgs)
    out = []
    for img in imgs:
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        out.append(img)
    return np.stack(out, axis=0)


def collect_groups(dataset: Path) -> dict[tuple[str, str], list[Path]]:
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for d in sorted(dataset.iterdir()):
        if not d.is_dir():
            continue
        m = SAMPLE_RE.match(d.name)
        if not m:
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        groups[(m.group(1), meta["target_camera"])].append(d)
    return dict(groups)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float32) / 255.0
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def soften_exposure(
    rgb: np.ndarray,
    target_p95: float = 0.86,
    trigger_p95: float = 0.82,
    target_mean: float = 0.58,
    trigger_mean: float = 0.66,
    min_gain: float = 0.78,
) -> tuple[np.ndarray, float]:
    """Slightly darken overexposed means; leave normal ones unchanged."""
    x = rgb.astype(np.float32) / 255.0
    L = _luminance(rgb)
    p95 = float(np.percentile(L, 95))
    mean_l = float(L.mean())

    gain = 1.0
    if p95 > trigger_p95:
        gain = min(gain, target_p95 / p95)
    if mean_l > trigger_mean:
        gain = min(gain, target_mean / mean_l)
    gain = float(np.clip(gain, min_gain, 1.0))
    if gain >= 0.999:
        return rgb, 1.0

    out = np.clip(x * gain, 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8), gain


def mean_for_group(paths: list[Path], camera: str, max_scenes: int) -> tuple[np.ndarray, list[str]]:
    paths = diversify_paths(paths)[:max_scenes]
    imgs = [load_rgb(p / "target" / f"{camera}.jpg") for p in paths]
    stack = align_stack(imgs)
    mean_rgb = stack.mean(axis=0).astype(np.uint8)
    return mean_rgb, [p.name for p in paths]


def save_index_grid(rows: list[dict], out_path: Path, cols: int = 6, thumb_h: int = 170):
    if not rows:
        return
    n = len(rows)
    ncols = min(cols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.2 * nrows))
    axes = np.atleast_2d(axes)
    for i, row in enumerate(rows):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        img = np.array(Image.open(row["mean_png"]))
        ax.imshow(img)
        ax.set_title(f"{row['vehicle']}/{row['camera']}\nn={row['n_used']}", fontsize=8)
        ax.axis("off")
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")
    fig.suptitle(f"Mean RGB per vehicle/camera ({n} groups)", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-scenes", type=int, default=10)
    ap.add_argument("--min-scenes", type=int, default=1)
    ap.add_argument("--grid-cols", type=int, default=6)
    ap.add_argument("--no-exposure-fix", action="store_true")
    ap.add_argument("--target-p95", type=float, default=0.86)
    ap.add_argument("--trigger-p95", type=float, default=0.82)
    args = ap.parse_args()

    means_dir = args.out / "means"
    means_dir.mkdir(parents=True, exist_ok=True)

    groups = collect_groups(args.dataset)
    rows: list[dict] = []
    skipped = 0

    for (vehicle, camera), paths in sorted(groups.items()):
        if len(paths) < args.min_scenes:
            skipped += 1
            continue
        mean_rgb, used_ids = mean_for_group(paths, camera, args.max_scenes)
        gain = 1.0
        if not args.no_exposure_fix:
            mean_rgb, gain = soften_exposure(
                mean_rgb,
                target_p95=args.target_p95,
                trigger_p95=args.trigger_p95,
            )
        out_png = means_dir / f"{vehicle}_{camera}.png"
        Image.fromarray(mean_rgb).save(out_png)
        L = _luminance(mean_rgb)
        rows.append({
            "vehicle": vehicle,
            "camera": camera,
            "n_available": len(paths),
            "n_used": len(used_ids),
            "sample_ids": used_ids,
            "mean_png": str(out_png.resolve()),
            "size_hw": list(mean_rgb.shape[:2]),
            "exposure_gain": gain,
            "l_mean": round(float(L.mean()), 4),
            "l_p95": round(float(np.percentile(L, 95)), 4),
        })

    summary = {
        "dataset": str(args.dataset.resolve()),
        "out_dir": str(args.out.resolve()),
        "max_scenes": args.max_scenes,
        "vehicles": len({r["vehicle"] for r in rows}),
        "groups": len(rows),
        "skipped_below_min_scenes": skipped,
        "items": rows,
    }
    (args.out / "index.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    save_index_grid(rows, args.out / "index_grid.png", cols=args.grid_cols)

    lt10 = sum(1 for r in rows if r["n_used"] < args.max_scenes)
    darkened = sum(1 for r in rows if r["exposure_gain"] < 0.999)
    print(f"Saved {len(rows)} means -> {means_dir}")
    print(f"  vehicles: {summary['vehicles']}  groups with <{args.max_scenes} scenes: {lt10}")
    print(f"  exposure corrected: {darkened}/{len(rows)}")
    print(f"  index: {args.out / 'index.json'}")
    print(f"  grid:  {args.out / 'index_grid.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
