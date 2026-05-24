"""
Ego mask v3: sharp edges -> edge vote -> boundary from repeatable ridges.

Instead of color thresholds, we:
  1. bilateral + morph gradient -> sharp edge map per scene
  2. edge_freq = how often an edge appears at each pixel (fixed camera)
  3. per column: strongest edge_freq ridge in bottom ROI = ego/scene boundary
  4. fill below boundary (+ optional SLIC stable-region trim)

Test on ONE vehicle (default: hilma).

Usage:
  python test_ego_seg_v3.py --vehicle hilma --camera right_fwd
  python test_ego_seg_v3.py --vehicle hilma --all-cameras
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DEFAULT_DATASET = Path(r"C:/Users/adel/Downloads/cv_dataset/final_dataset_v5_participants/train")
DEFAULT_VIZ = Path(__file__).resolve().parent / "methods_gallery/_ego_artifacts_v2_test"

SAMPLE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_([a-zA-Z]+)_\d+__\d{3}$"
)


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def keep_bottom_connected(mask: np.ndarray) -> np.ndarray:
    n, labels = cv2.connectedComponents(mask.astype(np.uint8))
    keep = np.zeros(mask.shape, dtype=bool)
    for lab in np.unique(labels[-1]):
        if lab:
            keep |= labels == lab
    return keep


def align_stack(imgs: list[np.ndarray]) -> list[np.ndarray]:
    h = max(i.shape[0] for i in imgs)
    w = max(i.shape[1] for i in imgs)
    return [
        cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA) if img.shape[:2] != (h, w) else img
        for img in imgs
    ]


def load_vehicle_stack(dataset: Path, vehicle: str, camera: str, max_scenes: int) -> np.ndarray:
    recs: list[tuple[str, Path]] = []
    for d in sorted(dataset.iterdir()):
        if not d.is_dir():
            continue
        m = SAMPLE_RE.match(d.name)
        if not m or m.group(1) != vehicle:
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if meta["target_camera"] != camera:
            continue
        recs.append((d.name.rsplit("__", 1)[0], d))

    by_trip = {trip: path for trip, path in recs}
    paths = [by_trip[t] for t in sorted(by_trip.keys())[:max_scenes]]
    return np.stack(align_stack([load_rgb(p / "target" / f"{camera}.jpg") for p in paths]))


def sharp_edge_map(img: np.ndarray, y0: int, grad_pct: float = 88.0) -> np.ndarray:
    roi = img[y0:]
    smooth = cv2.bilateralFilter(roi, 9, 80, 80)
    gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mg = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, k)
    mg = cv2.GaussianBlur(mg, (3, 3), 0)
    thr = max(10.0, float(np.percentile(mg, grad_pct)))
    out = np.zeros(img.shape[:2], dtype=bool)
    out[y0:] = mg >= thr
    return out


def compute_edge_freq(stack: np.ndarray, bottom_frac: float, grad_pct: float) -> tuple[np.ndarray, int]:
    y0 = int(stack.shape[1] * (1.0 - bottom_frac))
    edges = np.stack([sharp_edge_map(img, y0, grad_pct) for img in stack], axis=0)
    freq = edges.mean(axis=0)
    freq_roi = freq[y0:].astype(np.float32)
    freq_roi = cv2.GaussianBlur(freq_roi, (5, 5), 0)
    freq = freq.copy()
    freq[y0:] = freq_roi
    return freq, y0


def smooth1d(v: np.ndarray, k: int = 31) -> np.ndarray:
    k = max(3, k | 1)
    ker = np.ones(k, dtype=np.float32) / k
    return np.convolve(v.astype(np.float32), ker, mode="same")


def boundary_from_edge_ridge(
    edge_freq: np.ndarray,
    y0: int,
    min_peak: float = 0.12,
    smooth_k: int = 41,
) -> np.ndarray:
    """Per column: top of ego = strongest repeatable edge ridge in bottom ROI."""
    h, w = edge_freq.shape
    roi = edge_freq[y0:h]
    boundary = np.full(w, h - 1, dtype=np.int32)
    for x in range(w):
        col = roi[:, x]
        peak_i = int(np.argmax(col))
        if col[peak_i] >= min_peak:
            boundary[x] = y0 + peak_i
    boundary = np.clip(smooth1d(boundary, smooth_k), y0, h - 1).astype(np.int32)
    return boundary


def mask_below_boundary(boundary: np.ndarray, y0: int, h: int, w: int) -> np.ndarray:
    yy = np.arange(h, dtype=np.int32)[:, None]
    mask = yy >= boundary[None, :]
    mask[:y0] = False
    return keep_bottom_connected(mask.astype(np.uint8))


def trim_by_color_stability(
    stack: np.ndarray,
    mask: np.ndarray,
    std_pct: float = 35.0,
) -> np.ndarray:
    """Drop unstable pixels inside mask (road patches etc.)."""
    if not mask.any():
        return mask
    labs = np.stack([cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32) for img in stack])
    std_map = labs.std(axis=0).mean(axis=-1)
    inside = std_map[mask]
    thr = float(np.percentile(inside, std_pct))
    trimmed = mask & (std_map <= thr)
    return keep_bottom_connected(trimmed.astype(np.uint8))


def build_mask_v3(
    stack: np.ndarray,
    bottom_frac: float = 0.55,
    grad_pct: float = 88.0,
    min_peak: float = 0.12,
    trim_std_pct: float | None = 35.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_freq, y0 = compute_edge_freq(stack, bottom_frac, grad_pct)
    h, w = edge_freq.shape
    boundary = boundary_from_edge_ridge(edge_freq, y0, min_peak=min_peak)
    mask = mask_below_boundary(boundary, y0, h, w)
    if trim_std_pct is not None:
        mask = trim_by_color_stability(stack, mask, std_pct=trim_std_pct)
    return mask.astype(bool), edge_freq, boundary


def overlay_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    red = np.zeros_like(out)
    red[..., 0] = 255
    out[mask] = (0.5 * red[mask] + 0.5 * out[mask]).astype(np.uint8)
    return out


def draw_boundary(rgb: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    vis = rgb.copy()
    h, w = rgb.shape[:2]
    for x in range(0, w, 3):
        y = int(boundary[x])
        vis[y : min(h, y + 2), x] = [0, 255, 0]
    return vis


def v2_mask_for_compare(stack: np.ndarray, bottom_frac: float) -> np.ndarray | None:
    try:
        from test_ego_seg_v2 import aggregate_vote, ego_is_bright, segment_local

        locals_ = [
            segment_local(img, bottom_frac=bottom_frac, bright=ego_is_bright(stack))
            for img in stack
        ]
        mask, _ = aggregate_vote(locals_)
        return mask
    except ImportError:
        return None


def save_figure(
    vehicle: str,
    camera: str,
    ref: np.ndarray,
    edge_freq: np.ndarray,
    boundary: np.ndarray,
    mask: np.ndarray,
    v2_mask: np.ndarray | None,
    out_path: Path,
    n_scenes: int,
):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    axes[0, 0].imshow(ref)
    axes[0, 0].set_title("reference")
    axes[0, 0].axis("off")

    im = axes[0, 1].imshow(edge_freq, cmap="magma", vmin=0, vmax=1)
    axes[0, 1].set_title(f"edge vote freq ({n_scenes} scenes)")
    axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    axes[0, 2].imshow(draw_boundary(ref, boundary))
    axes[0, 2].set_title("boundary ridge (green)")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(overlay_mask(ref, mask))
    axes[1, 0].set_title(f"v3 edge ridge {100 * mask.mean():.1f}%")
    axes[1, 0].axis("off")

    if v2_mask is not None:
        axes[1, 1].imshow(overlay_mask(ref, v2_mask))
        axes[1, 1].set_title(f"v2 color vote {100 * v2_mask.mean():.1f}%")
    axes[1, 1].axis("off")

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02,
        0.98,
        "\n".join(
            [
                f"{vehicle} / {camera}",
                f"scenes: {n_scenes}",
                f"coverage: {100 * mask.mean():.2f}%",
                "",
                "1) bilateral + morph gradient",
                "2) edge vote freq per pixel",
                "3) per-x ridge = ego top boundary",
                "4) fill below + trim unstable color",
            ]
        ),
        va="top",
        family="monospace",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    fig.suptitle(f"Ego mask v3 — {vehicle}/{camera}", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def list_cameras(dataset: Path, vehicle: str) -> list[str]:
    cams: set[str] = set()
    for d in dataset.iterdir():
        if not d.is_dir():
            continue
        m = SAMPLE_RE.match(d.name)
        if not m or m.group(1) != vehicle:
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        cams.add(meta["target_camera"])
    return sorted(cams)


def process_camera(
    dataset: Path,
    viz_dir: Path,
    vehicle: str,
    camera: str,
    max_scenes: int,
    bottom_frac: float,
    min_peak: float,
    trim_std_pct: float | None,
) -> dict:
    stack = load_vehicle_stack(dataset, vehicle, camera, max_scenes)
    if len(stack) < 2:
        return {"vehicle": vehicle, "camera": camera, "error": "not enough scenes"}

    mask, edge_freq, boundary = build_mask_v3(
        stack,
        bottom_frac=bottom_frac,
        min_peak=min_peak,
        trim_std_pct=trim_std_pct,
    )
    v2_mask = v2_mask_for_compare(stack, bottom_frac)

    out_path = viz_dir / f"{vehicle}_{camera}_v3.png"
    save_figure(vehicle, camera, stack[0], edge_freq, boundary, mask, v2_mask, out_path, len(stack))

    return {
        "vehicle": vehicle,
        "camera": camera,
        "n_scenes": len(stack),
        "coverage": float(mask.mean()),
        "v2_coverage": float(v2_mask.mean()) if v2_mask is not None else None,
        "viz": str(out_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--viz-dir", type=Path, default=DEFAULT_VIZ)
    ap.add_argument("--vehicle", default="hilma")
    ap.add_argument("--camera", default="right_fwd")
    ap.add_argument("--all-cameras", action="store_true")
    ap.add_argument("--max-scenes", type=int, default=12)
    ap.add_argument("--bottom-frac", type=float, default=0.55)
    ap.add_argument("--min-peak", type=float, default=0.12)
    ap.add_argument("--no-trim", action="store_true")
    ap.add_argument("--trim-std-pct", type=float, default=35.0)
    args = ap.parse_args()

    trim = None if args.no_trim else args.trim_std_pct
    cameras = list_cameras(args.dataset, args.vehicle) if args.all_cameras else [args.camera]
    results = []
    for cam in cameras:
        r = process_camera(
            args.dataset, args.viz_dir, args.vehicle, cam,
            args.max_scenes, args.bottom_frac, args.min_peak, trim,
        )
        results.append(r)
        if "error" in r:
            print(f"  {cam}: {r['error']}")
        else:
            v2 = f"  v2={100 * r['v2_coverage']:.1f}%" if r["v2_coverage"] is not None else ""
            print(f"  {cam:10s}  v3={100 * r['coverage']:.1f}%{v2}")

    summary = args.viz_dir / f"{args.vehicle}_v3_summary.json"
    summary.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nViz: {args.viz_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
