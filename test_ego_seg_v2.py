"""
Ego artifact mask v2: local color/texture seg -> cross-scene pixel vote.

Test on ONE vehicle first (default: hilma).

Pipeline per (vehicle, camera):
  1. Per scene: bottom ROI, dark + low-gradient, keep CC touching bottom edge
  2. Stack at fixed pixel coords (rigid camera mount)
  3. freq = fraction of scenes where pixel is ego-candidate
  4. Morph close -> cohesive region

Usage:
  python test_ego_seg_v2.py
  python test_ego_seg_v2.py --vehicle hilma --camera right_fwd
  python test_ego_seg_v2.py --vehicle hilma --all-cameras
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent
DEFAULT_DATASET = Path(r"C:/Users/adel/Downloads/cv_dataset/final_dataset_v5_participants/train")
DEFAULT_VIZ = REPO / "methods_gallery/_ego_artifacts_v2_test"
OLD_MASK_ROOT = Path(r"C:/Users/adel/Downloads/cv_dataset/ego_artifact_masks")

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


def segment_local_roi(
    roi: np.ndarray,
    l_pct: float,
    grad_pct: float,
    bright: bool,
) -> np.ndarray:
    lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
    l_ch = lab[..., 0].astype(np.float32)
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    smooth = grad <= np.percentile(grad, grad_pct)
    if bright:
        uniform = l_ch >= np.percentile(l_ch, 100.0 - l_pct)
    else:
        uniform = l_ch <= np.percentile(l_ch, l_pct)
    return keep_bottom_connected((uniform & smooth).astype(np.uint8))


def ego_is_bright(stack: np.ndarray, bottom_frac: float = 0.52) -> bool:
    h = stack.shape[1]
    y0 = int(h * (1.0 - bottom_frac))
    med = np.median(stack[:, y0:, :].astype(np.float32), axis=0)
    l_med = cv2.cvtColor(med.astype(np.uint8), cv2.COLOR_RGB2LAB)[..., 0]
    return float(np.median(l_med[-25:, :])) > 118.0


def segment_local(
    img: np.ndarray,
    bottom_frac: float = 0.52,
    l_pct: float = 48.0,
    grad_pct: float = 52.0,
    bright: bool = False,
) -> np.ndarray:
    h, w = img.shape[:2]
    y0 = int(h * (1.0 - bottom_frac))
    m = segment_local_roi(img[y0:], l_pct, grad_pct, bright=bright)
    full = np.zeros((h, w), dtype=bool)
    full[y0:] = m
    return full


def align_stack(imgs: list[np.ndarray]) -> tuple[list[np.ndarray], int, int]:
    h = max(i.shape[0] for i in imgs)
    w = max(i.shape[1] for i in imgs)
    out = []
    for img in imgs:
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        out.append(img)
    return out, h, w


def aggregate_vote(
    local_masks: list[np.ndarray],
    vote_thr: float = 0.35,
    close_k: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    freq = np.mean(np.stack(local_masks, axis=0), axis=0)
    m = freq >= vote_thr
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k, iterations=2)
    m = keep_bottom_connected(m)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k2, iterations=1).astype(bool)
    return m, freq


def load_vehicle_camera(
    dataset: Path,
    vehicle: str,
    camera: str,
    max_scenes: int,
) -> tuple[list[Path], list[np.ndarray]]:
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
        trip = d.name.rsplit("__", 1)[0]
        recs.append((trip, d))

    by_trip: dict[str, Path] = {}
    for trip, path in recs:
        by_trip.setdefault(trip, path)

    paths = [by_trip[t] for t in sorted(by_trip.keys())[:max_scenes]]
    imgs = [load_rgb(p / "target" / f"{camera}.jpg") for p in paths]
    imgs, _, _ = align_stack(imgs)
    return paths, imgs


def load_old_mask(vehicle: str, camera: str, shape: tuple[int, int]) -> np.ndarray | None:
    p = OLD_MASK_ROOT / vehicle / f"{camera}_mask.npy"
    if not p.is_file():
        return None
    m = np.load(p).astype(bool)
    h, w = shape
    if m.shape != (h, w):
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return m


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    out = rgb.copy()
    red = np.zeros_like(out)
    red[..., 0] = 255
    out[mask] = (alpha * red[mask] + (1.0 - alpha) * out[mask]).astype(np.uint8)
    return out


def save_camera_figure(
    vehicle: str,
    camera: str,
    ref: np.ndarray,
    local_ex: np.ndarray,
    freq: np.ndarray,
    mask: np.ndarray,
    old: np.ndarray | None,
    n_scenes: int,
    out_path: Path,
    vote_thr: float,
):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    axes[0, 0].imshow(ref)
    axes[0, 0].set_title("reference")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(overlay_mask(ref, local_ex))
    axes[0, 1].set_title(f"local seg (1 scene) {100 * local_ex.mean():.1f}%")
    axes[0, 1].axis("off")

    im = axes[0, 2].imshow(freq, cmap="viridis", vmin=0, vmax=1)
    axes[0, 2].set_title(f"vote freq ({n_scenes} scenes)")
    axes[0, 2].axis("off")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    axes[1, 0].imshow(overlay_mask(ref, mask))
    axes[1, 0].set_title(f"v2 vote>={vote_thr:.2f} {100 * mask.mean():.1f}%")
    axes[1, 0].axis("off")

    if old is not None:
        axes[1, 1].imshow(overlay_mask(ref, old))
        axes[1, 1].set_title(f"old flow agg {100 * old.mean():.1f}%")
    else:
        axes[1, 1].axis("off")
    axes[1, 1].axis("off")

    axes[1, 2].axis("off")
    bottom = 100.0 * mask[int(ref.shape[0] * 0.5) :].sum() / max(mask.sum(), 1)
    axes[1, 2].text(
        0.02,
        0.98,
        "\n".join(
            [
                f"{vehicle} / {camera}",
                f"scenes: {n_scenes}",
                f"coverage: {100 * mask.mean():.2f}%",
                f"bottom50: {bottom:.0f}%",
                "",
                "local: dark/bright L + low grad, bottom CC",
                f"agg: pixel vote >= {vote_thr:.2f} + morph close",
            ]
        ),
        va="top",
        family="monospace",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    fig.suptitle(f"Ego mask v2 — {vehicle}/{camera}", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def process_camera(
    dataset: Path,
    viz_dir: Path,
    vehicle: str,
    camera: str,
    max_scenes: int,
    vote_thr: float,
    bottom_frac: float,
    close_k: int,
) -> dict:
    paths, imgs = load_vehicle_camera(dataset, vehicle, camera, max_scenes)
    if len(imgs) < 2:
        return {"vehicle": vehicle, "camera": camera, "error": "not enough scenes"}

    locals_ = [segment_local(img, bottom_frac=bottom_frac, bright=ego_is_bright(np.stack(imgs))) for img in imgs]
    mask, freq = aggregate_vote(locals_, vote_thr=vote_thr, close_k=close_k)
    old = load_old_mask(vehicle, camera, mask.shape)

    out_path = viz_dir / f"{vehicle}_{camera}_v2.png"
    save_camera_figure(
        vehicle, camera, imgs[0], locals_[0], freq, mask, old,
        len(imgs), out_path, vote_thr,
    )

    return {
        "vehicle": vehicle,
        "camera": camera,
        "n_scenes": len(imgs),
        "coverage": float(mask.mean()),
        "bottom_frac": float(mask[int(mask.shape[0] * 0.5) :].sum() / max(mask.sum(), 1)),
        "old_coverage": float(old.mean()) if old is not None else None,
        "viz": str(out_path),
    }


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--viz-dir", type=Path, default=DEFAULT_VIZ)
    ap.add_argument("--vehicle", default="hilma")
    ap.add_argument("--camera", default="right_fwd")
    ap.add_argument("--all-cameras", action="store_true")
    ap.add_argument("--max-scenes", type=int, default=12)
    ap.add_argument("--vote-thr", type=float, default=0.35)
    ap.add_argument("--bottom-frac", type=float, default=0.52)
    ap.add_argument("--close-k", type=int, default=21)
    args = ap.parse_args()

    cameras = list_cameras(args.dataset, args.vehicle) if args.all_cameras else [args.camera]
    results = []
    for cam in cameras:
        r = process_camera(
            args.dataset, args.viz_dir, args.vehicle, cam,
            args.max_scenes, args.vote_thr, args.bottom_frac, args.close_k,
        )
        results.append(r)
        if "error" in r:
            print(f"  {cam}: {r['error']}")
        else:
            old = f"  old={100 * r['old_coverage']:.1f}%" if r["old_coverage"] is not None else ""
            print(
                f"  {cam:10s}  cov={100 * r['coverage']:.1f}%  "
                f"bottom={100 * r['bottom_frac']:.0f}%  n={r['n_scenes']}{old}"
            )

    summary_path = args.viz_dir / f"{args.vehicle}_v2_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nViz: {args.viz_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
