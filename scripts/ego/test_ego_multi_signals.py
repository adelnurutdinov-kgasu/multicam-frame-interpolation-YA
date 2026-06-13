"""
Ego artifact signals #2 #3 #4 on diverse cross-scene stack (one vehicle/camera).

  #2 cross-scene RGB std (+ Otsu on bottom ROI)
  #3 optical flow vs geometry disagreement (Farneback vs depth+poses), averaged
  #4 Canny edge persistence

Usage:
  python test_ego_multi_signals.py --vehicle hilma --camera right_fwd
  python test_ego_multi_signals.py --max-scenes 35 --flow-max 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



from layered_parallax import get_lidar_depth
from lidar_depth_map import intrinsics_to_K
from static_mask import compute_flow_maps
from test_ego_consistent_edges import DEFAULT_DATASET, DEFAULT_VIZ, SAMPLE_RE, diversify_paths

OUT = DEFAULT_VIZ / "multi_signals"


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    red = np.zeros_like(out)
    red[..., 0] = 255
    out[mask] = (0.55 * red[mask] + 0.45 * out[mask]).astype(np.uint8)
    return out


def align_rgb(imgs: list[np.ndarray]) -> np.ndarray:
    h = max(i.shape[0] for i in imgs)
    w = max(i.shape[1] for i in imgs)
    out = []
    for img in imgs:
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        out.append(img)
    return np.stack(out, axis=0)


def collect_paths(dataset: Path, vehicle: str, camera: str, max_scenes: int) -> list[Path]:
    paths: list[Path] = []
    for d in sorted(dataset.iterdir()):
        if not d.is_dir():
            continue
        m = SAMPLE_RE.match(d.name)
        if not m or m.group(1) != vehicle:
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if meta["target_camera"] != camera:
            continue
        paths.append(d)
    return diversify_paths(paths)[:max_scenes]


def otsu_low_mask(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float]:
    roi = values[valid].astype(np.float32)
    if roi.size < 64:
        thr = float(np.median(roi))
        return values < thr, thr
    norm = roi - roi.min()
    if norm.max() <= 1e-6:
        return values < roi.max(), float(roi.max())
    u8 = (norm / norm.max() * 255).astype(np.uint8)
    thr_u8, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = roi.min() + (thr_u8 / 255.0) * (roi.max() - roi.min())
    return values < thr, float(thr)


def geo_flow_from_depth(
    depth: np.ndarray,
    c2w_t0: np.ndarray,
    c2w_t1: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    h, w = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u0, v0 = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    valid = np.isfinite(depth) & (depth > 0.5)
    z = np.where(valid, depth, 1.0).astype(np.float64)
    x = (u0 - cx) * z / fx
    y = (v0 - cy) * z / fy
    ones = np.ones_like(z)
    pts = np.stack([x, y, z, ones], axis=-1)
    w2c_t1 = np.linalg.inv(c2w_t1)
    pts_w = np.einsum("ij,...j->...i", c2w_t0, pts)
    pts1 = np.einsum("ij,...j->...i", w2c_t1, pts_w)
    z1 = np.maximum(pts1[..., 2], 1e-6)
    u1 = fx * pts1[..., 0] / z1 + cx
    v1 = fy * pts1[..., 1] / z1 + cy
    flow = np.stack([u1 - u0, v1 - v0], axis=-1).astype(np.float32)
    flow[~valid] = np.nan
    return flow


def keep_bottom_cc(mask: np.ndarray) -> np.ndarray:
    n, labels = cv2.connectedComponents(mask.astype(np.uint8))
    keep = np.zeros(mask.shape, dtype=bool)
    for lab in np.unique(labels[-1]):
        if lab:
            keep |= labels == lab
    return keep


def compute_signals(
    dataset: Path,
    vehicle: str,
    camera: str,
    max_scenes: int,
    flow_max: int,
    bottom_frac: float,
    canny_lo: int,
    canny_hi: int,
    edge_persist_thr: float,
) -> dict:
    paths = collect_paths(dataset, vehicle, camera, max_scenes)
    if len(paths) < 3:
        raise ValueError("need >= 3 diverse scenes")

    targets = align_rgb([load_rgb(p / "target" / f"{camera}.jpg") for p in paths])
    n, h, w, _ = targets.shape
    y0 = int(h * (1.0 - bottom_frac))
    roi_mask = np.zeros((h, w), dtype=bool)
    roi_mask[y0:] = True

    std_map = targets.astype(np.float32).std(axis=0).mean(axis=-1)
    artifact_std, otsu_thr = otsu_low_mask(std_map, roi_mask)
    artifact_std = keep_bottom_cc(artifact_std.astype(np.uint8))

    canny_stack = []
    for img in targets:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, canny_lo, canny_hi).astype(np.float32) / 255.0
        canny_stack.append(edges)
    edge_persist = np.mean(canny_stack, axis=0)
    edge_artifact = (edge_persist >= edge_persist_thr) & roi_mask
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3))
    edge_artifact = cv2.morphologyEx(
        edge_artifact.astype(np.uint8), cv2.MORPH_CLOSE, kh, iterations=1,
    ).astype(bool)
    edge_artifact = keep_bottom_cc(edge_artifact.astype(np.uint8))

    flow_paths = paths[: min(flow_max, len(paths))]
    disagree_stack = []
    actual_mag_stack = []
    for sd in flow_paths:
        meta = json.loads((sd / "meta.json").read_text(encoding="utf-8"))
        K = intrinsics_to_K(meta["intrinsics"][camera])
        img_t0 = load_rgb(sd / "input" / "t0" / f"{camera}.jpg")
        img_t1 = load_rgb(sd / "input" / "t1" / f"{camera}.jpg")
        if img_t0.shape[:2] != (h, w):
            img_t0 = cv2.resize(img_t0, (w, h), interpolation=cv2.INTER_AREA)
            img_t1 = cv2.resize(img_t1, (w, h), interpolation=cv2.INTER_AREA)

        _, _, _, flow = compute_flow_maps(img_t0, img_t1)
        actual_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).astype(np.float32)

        c2w_t0 = np.array(meta["poses_c2w"]["t0"][camera], dtype=np.float64)
        c2w_t1 = np.array(meta["poses_c2w"]["t1"][camera], dtype=np.float64)
        depth = get_lidar_depth(sd, camera, "t0")
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        exp_flow = geo_flow_from_depth(depth, c2w_t0, c2w_t1, K)
        exp_mag = np.linalg.norm(exp_flow, axis=-1)
        valid = np.isfinite(exp_mag) & (exp_mag > 0.05)

        disagree = np.zeros((h, w), np.float32)
        disagree[valid] = np.abs(actual_mag[valid] - exp_mag[valid]) / (exp_mag[valid] + 1e-3)
        nod = ~valid & roi_mask
        disagree[nod] = np.where(actual_mag[nod] < 1.0, 1.0, 0.0)

        disagree_stack.append(disagree)
        actual_mag_stack.append(actual_mag)

    disagree_mean = np.nanmean(np.stack(disagree_stack, axis=0), axis=0)
    actual_mag_mean = np.mean(np.stack(actual_mag_stack, axis=0), axis=0)
    actual_mag_std = np.std(np.stack(actual_mag_stack, axis=0), axis=0)

    d_roi = disagree_mean[roi_mask]
    d_hi = float(np.percentile(d_roi[np.isfinite(d_roi)], 85)) if np.isfinite(d_roi).any() else 1.0
    # hood: low flow magnitude AND low std of |flow| across diverse scenes
    mag_mean_roi = actual_mag_mean[roi_mask]
    mag_std_roi = actual_mag_std[roi_mask]
    m_hi = float(np.percentile(mag_mean_roi, 35))
    s_hi = float(np.percentile(mag_std_roi, 40))
    flow_static = (
        (actual_mag_mean <= m_hi)
        & (actual_mag_std <= s_hi)
        & roi_mask
    )
    flow_disagree = (disagree_mean >= d_hi) & roi_mask
    artifact_flow = (flow_static | flow_disagree) & (std_map <= otsu_thr * 1.15)
    artifact_flow = keep_bottom_cc(artifact_flow.astype(np.uint8))

    votes = (
        artifact_std.astype(np.uint8)
        + edge_artifact.astype(np.uint8)
        + artifact_flow.astype(np.uint8)
    )
    combined = votes >= 2

    return {
        "paths": paths,
        "ref": targets[0],
        "n": n,
        "n_flow": len(flow_paths),
        "y0": y0,
        "std_map": std_map,
        "otsu_thr": otsu_thr,
        "artifact_std": artifact_std,
        "edge_persist": edge_persist,
        "artifact_edge": edge_artifact,
        "disagree_mean": disagree_mean,
        "actual_mag_mean": actual_mag_mean,
        "actual_mag_std": actual_mag_std,
        "artifact_flow": artifact_flow,
        "combined": combined,
    }


def save_figure(vehicle: str, camera: str, sig: dict, out_path: Path):
    ref = sig["ref"]
    y0 = sig["y0"]
    fig, ax = plt.subplots(3, 4, figsize=(18, 12))

    ax[0, 0].imshow(ref)
    ax[0, 0].set_title(f"ref (N={sig['n']} scenes)")
    ax[0, 0].axhline(y0, color="cyan", lw=0.8)
    ax[0, 0].axis("off")

    im = ax[0, 1].imshow(sig["std_map"], cmap="magma")
    ax[0, 1].set_title(f"#2 RGB std (Otsu<{sig['otsu_thr']:.2f})")
    ax[0, 1].axis("off")
    plt.colorbar(im, ax=ax[0, 1], fraction=0.046)

    ax[0, 2].imshow(overlay(ref, sig["artifact_std"]))
    ax[0, 2].set_title(f"std artifact {100 * sig['artifact_std'].mean():.2f}%")
    ax[0, 2].axis("off")

    im2 = ax[0, 3].imshow(sig["edge_persist"], cmap="hot", vmin=0, vmax=1)
    ax[0, 3].set_title("#4 edge persistence")
    ax[0, 3].axis("off")
    plt.colorbar(im2, ax=ax[0, 3], fraction=0.046)

    ax[1, 0].imshow(overlay(ref, sig["artifact_edge"]))
    ax[1, 0].set_title(f"edge persist {100 * sig['artifact_edge'].mean():.2f}%")
    ax[1, 0].axis("off")

    im3 = ax[1, 1].imshow(sig["disagree_mean"], cmap="viridis")
    ax[1, 1].set_title(f"#3 flow disagree (n={sig['n_flow']})")
    ax[1, 1].axis("off")
    plt.colorbar(im3, ax=ax[1, 1], fraction=0.046)

    ax[1, 2].imshow(overlay(ref, sig["artifact_flow"]))
    ax[1, 2].set_title(f"flow artifact {100 * sig['artifact_flow'].mean():.2f}%")
    ax[1, 2].axis("off")

    im4 = ax[1, 3].imshow(sig["actual_mag_std"], cmap="plasma")
    ax[1, 3].set_title("flow |mag| std across scenes")
    ax[1, 3].axis("off")
    plt.colorbar(im4, ax=ax[1, 3], fraction=0.046)

    ax[2, 0].imshow(overlay(ref, sig["combined"]))
    ax[2, 0].set_title(f"vote 2/3 {100 * sig['combined'].mean():.2f}%")
    ax[2, 0].axis("off")

    ax[2, 1].imshow(sig["actual_mag_mean"], cmap="hot")
    ax[2, 1].set_title("mean flow |mag|")
    ax[2, 1].axis("off")

    ax[2, 2].text(
        0.02, 0.98,
        "\n".join([
            f"{vehicle}/{camera}",
            f"scenes: {sig['n']}  flow: {sig['n_flow']}",
            f"std Otsu: {sig['otsu_thr']:.3f}",
            "",
            "#2 low RGB std",
            "#3 flow vs lidar geom",
            "#4 Canny persistence",
            "combined: 2 of 3",
        ]),
        va="top", family="monospace", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    ax[2, 2].axis("off")
    ax[2, 3].axis("off")

    fig.suptitle(f"Ego multi-signals — {vehicle}/{camera}", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--vehicle", default="hilma")
    ap.add_argument("--camera", default="right_fwd")
    ap.add_argument("--cameras", nargs="*")
    ap.add_argument("--max-scenes", type=int, default=35)
    ap.add_argument("--flow-max", type=int, default=20)
    ap.add_argument("--bottom-frac", type=float, default=0.55)
    ap.add_argument("--canny-lo", type=int, default=50)
    ap.add_argument("--canny-hi", type=int, default=150)
    ap.add_argument("--edge-persist-thr", type=float, default=0.25)
    args = ap.parse_args()

    cameras = args.cameras or [args.camera]
    results = []
    for cam in cameras:
        sig = compute_signals(
            args.dataset, args.vehicle, cam,
            args.max_scenes, args.flow_max, args.bottom_frac,
            args.canny_lo, args.canny_hi, args.edge_persist_thr,
        )
        out = OUT / f"{args.vehicle}_{cam}_signals.png"
        save_figure(args.vehicle, cam, sig, out)
        row = {
            "vehicle": args.vehicle,
            "camera": cam,
            "n_scenes": sig["n"],
            "n_flow": sig["n_flow"],
            "otsu_thr": sig["otsu_thr"],
            "std_cov": float(sig["artifact_std"].mean()),
            "edge_cov": float(sig["artifact_edge"].mean()),
            "flow_cov": float(sig["artifact_flow"].mean()),
            "combined_cov": float(sig["combined"].mean()),
            "viz": str(out),
        }
        results.append(row)
        print(
            f"{cam}: N={sig['n']}  std={100 * row['std_cov']:.1f}%  "
            f"edge={100 * row['edge_cov']:.1f}%  flow={100 * row['flow_cov']:.1f}%  "
            f"2/3={100 * row['combined_cov']:.1f}%"
        )

    (OUT / f"{args.vehicle}_signals_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Viz: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
