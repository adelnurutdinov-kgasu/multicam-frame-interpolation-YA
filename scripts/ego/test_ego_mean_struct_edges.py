"""
Mean RGB -> blur dense ghost edges -> structural edge detection.

Idea: motion-blurred background on mean has many weak/dense edges;
      ego artifacts stay sharp. Smooth mean, then find real boundaries.

Usage:
  python test_ego_mean_struct_edges.py --vehicle hilma --camera right_fwd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


import cv2
import matplotlib.pyplot as plt
import numpy as np



from test_ego_consistent_edges import DEFAULT_DATASET, load_stack, sharp_edge_magnitude

OUT = REPO / "methods_gallery/_ego_artifacts_v2_test/consistent_edges"


def process_mean_edges(
    mean_rgb: np.ndarray,
    y0: int,
    bilateral_d: int = 15,
    bilateral_sigma: float = 100.0,
    open_k: int = 7,
    grad_pct: float = 85.0,
) -> dict:
    roi = mean_rgb[y0:].astype(np.uint8)
    smooth = cv2.bilateralFilter(roi, bilateral_d, bilateral_sigma, bilateral_sigma)
    gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, k)
    soft = cv2.GaussianBlur(opened, (5, 5), 0)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mg = cv2.morphologyEx(soft, cv2.MORPH_GRADIENT, k3)
    mg = cv2.GaussianBlur(mg, (3, 3), 0)
    thr = max(8.0, float(np.percentile(mg, grad_pct)))
    edges = mg >= thr
    return {
        "smooth": smooth,
        "opened": opened,
        "soft": soft,
        "mg": mg,
        "edges": edges,
        "thr": thr,
    }


def overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    red = np.zeros_like(out)
    red[..., 0] = 255
    out[mask] = (0.55 * red[mask] + 0.45 * out[mask]).astype(np.uint8)
    return out


def run(
    vehicle: str,
    camera: str,
    n: int,
    bottom_frac: float,
    consensus_min: int,
    bilateral_d: int,
    bilateral_sigma: float,
    open_k: int,
    grad_pct: float,
) -> Path:
    _, stack = load_stack(DEFAULT_DATASET, vehicle, camera, n)
    h, w = stack.shape[1:3]
    y0 = int(h * (1.0 - bottom_frac))
    mean_rgb = stack.mean(axis=0).astype(np.uint8)

    proc = process_mean_edges(
        mean_rgb, y0, bilateral_d, bilateral_sigma, open_k, grad_pct,
    )
    edges_full = np.zeros((h, w), dtype=bool)
    edges_full[y0:] = proc["edges"]

    edge_sum = np.stack(
        [sharp_edge_magnitude(img, y0, 88.0)[0] for img in stack], axis=0,
    ).sum(axis=0)
    consensus = edge_sum >= consensus_min
    combined = edges_full & consensus

    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3))
    closed = cv2.morphologyEx(combined.astype(np.uint8), cv2.MORPH_CLOSE, kh, 1).astype(bool)

    fig, ax = plt.subplots(2, 5, figsize=(18, 7))
    ax[0, 0].imshow(mean_rgb)
    ax[0, 0].set_title("mean RGB")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(proc["smooth"])
    ax[0, 1].set_title("bilateral smooth")
    ax[0, 1].axis("off")

    ax[0, 2].imshow(proc["opened"], cmap="gray")
    ax[0, 2].set_title(f"morph open k={open_k}")
    ax[0, 2].axis("off")

    ax[0, 3].imshow(proc["mg"], cmap="magma")
    ax[0, 3].set_title(f"grad thr={proc['thr']:.0f}")
    ax[0, 3].axis("off")

    ax[0, 4].imshow(overlay(mean_rgb, edges_full))
    ax[0, 4].set_title("edges on clean mean")
    ax[0, 4].axis("off")

    ax[1, 0].imshow(edge_sum, cmap="hot", vmin=0, vmax=n)
    ax[1, 0].set_title(f"edge sum 0-{n}")
    ax[1, 0].axis("off")

    ax[1, 1].imshow(overlay(mean_rgb, consensus))
    ax[1, 1].set_title(f"consensus >={consensus_min}")
    ax[1, 1].axis("off")

    ax[1, 2].imshow(overlay(mean_rgb, combined))
    ax[1, 2].set_title("mean-edge AND consensus")
    ax[1, 2].axis("off")

    ax[1, 3].imshow(overlay(mean_rgb, closed))
    ax[1, 3].set_title("+ horiz close")
    ax[1, 3].axis("off")

    ax[1, 4].imshow(overlay(stack[0], closed))
    ax[1, 4].set_title("on ref scene 1")
    ax[1, 4].axis("off")

    fig.suptitle(
        f"{vehicle}/{camera} — mean blur + structural edges ({n} frames)",
        fontsize=12,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{vehicle}_{camera}_mean_struct_edges.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(
        f"{vehicle}/{camera}: mean-only={edges_full.sum()} px  "
        f"combined={combined.sum()}  closed={closed.sum()}"
    )
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicle", default="hilma")
    ap.add_argument("--camera", default="right_fwd")
    ap.add_argument("--cameras", nargs="*")
    ap.add_argument("--max-scenes", type=int, default=10)
    ap.add_argument("--bottom-frac", type=float, default=0.45)
    ap.add_argument("--consensus-min", type=int, default=9)
    ap.add_argument("--bilateral-d", type=int, default=15)
    ap.add_argument("--bilateral-sigma", type=float, default=100.0)
    ap.add_argument("--open-k", type=int, default=7)
    ap.add_argument("--grad-pct", type=float, default=85.0)
    args = ap.parse_args()

    cameras = args.cameras or [args.camera]
    for cam in cameras:
        p = run(
            args.vehicle, cam, args.max_scenes, args.bottom_frac,
            args.consensus_min, args.bilateral_d, args.bilateral_sigma,
            args.open_k, args.grad_pct,
        )
        print(f"  saved {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
