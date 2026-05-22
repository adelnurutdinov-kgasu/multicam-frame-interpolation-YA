"""Layered depth-based parallax warp: t0 + t1 -> target.

Splits the scene into depth layers (from LiDAR), warps each layer with
geometry from camera poses (parallax), then composites.

Compares against global Farneback half-flow from testdeltadif.py.
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dataclasses import dataclass


@dataclass
class BlurTuneConfig:
    sigma_near: float = 1.5
    sigma_far: float = 2.0
    flow_k: float = 0.05
    comp_k: float = 0.0


def layer_sigma(i: int, n_layers: int, flow_mag: np.ndarray, mask: np.ndarray, cfg: BlurTuneConfig) -> float:
    t = i / max(n_layers - 1, 1)
    sigma = cfg.sigma_near + (cfg.sigma_far - cfg.sigma_near) * t
    if cfg.flow_k > 0 and np.any(mask):
        sigma += cfg.flow_k * float(np.mean(flow_mag[mask]))
    if cfg.comp_k > 0 and np.any(mask):
        pass  # comp_err optional
    return max(0.0, sigma)


def apply_layer_blur(layer: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 1e-4:
        return layer.astype(np.float32)
    return cv2.GaussianBlur(layer.astype(np.uint8), (0, 0), sigmaX=sigma, sigmaY=sigma).astype(np.float32)


@dataclass
class AlphaTuneConfig:
    beta: float = 0.05
    gamma: float = -0.10
    mean_w: float = 0.55
    pose_beta: float = 0.0
    use_pose_interp: bool = False


def layer_alpha(
    z: float,
    layer_idx: int,
    n_layers: int,
    alpha_time: float,
    z_ref: float,
    cfg: AlphaTuneConfig,
) -> tuple[float, float]:
    depth_term = (z_ref / max(z, 0.5)) - 1.0
    layer_term = layer_idx / max(n_layers - 1, 1)
    a_blend = alpha_time + cfg.beta * depth_term + cfg.gamma * layer_term
    a_pose = alpha_time + cfg.pose_beta * depth_term
    return float(np.clip(a_blend, 0.0, 1.0)), float(np.clip(a_pose, 0.0, 1.0))


def interp_c2w(c2w0: np.ndarray, c2w1: np.ndarray, a: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation, Slerp
    R0, t0 = c2w0[:3, :3], c2w0[:3, 3]
    R1, t1 = c2w1[:3, :3], c2w1[:3, 3]
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([R0, R1])))
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = slerp(float(np.clip(a, 0.0, 1.0))).as_matrix()
    out[:3, 3] = (1 - a) * t0 + a * t1
    return out


from lidar_depth_map import (
    build_depth_map,
    fill_depth_holes,
    intrinsics_to_K,
    project_to_camera,
    rasterize_depth,
)


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    return 20.0 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float("inf")


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def backwarp_map(
    c2w_target: np.ndarray,
    c2w_src: np.ndarray,
    K: np.ndarray,
    Z: np.ndarray | float,
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each target pixel, where to sample in src image (constant or per-pixel Z)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    z = np.broadcast_to(np.asarray(Z, dtype=np.float64), (h, w)).copy()

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts_tgt = np.stack([x, y, z, np.ones_like(z)], axis=-1)  # H W 4

    pts_w = np.einsum("ij,...j->...i", c2w_target, pts_tgt)
    w2c_src = np.linalg.inv(c2w_src)
    pts_src = np.einsum("ij,...j->...i", w2c_src, pts_w)

    zs = pts_src[..., 2]
    valid = zs > 0.5
    u_src = fx * pts_src[..., 0] / np.maximum(zs, 1e-6) + cx
    v_src = fy * pts_src[..., 1] / np.maximum(zs, 1e-6) + cy
    return u_src.astype(np.float32), v_src.astype(np.float32), valid


def remap_rgb(img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def farneback_half(img_t0: np.ndarray, img_t1: np.ndarray) -> np.ndarray:
    """Global half-flow warp of t1 (testdeltadif.py style)."""
    g0 = cv2.cvtColor(img_t0, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(img_t1, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g1, g0, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = img_t0.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(
        img_t1,
        xs - flow[..., 0] / 2.0,
        ys - flow[..., 1] / 2.0,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def farneback_alpha(img_t0: np.ndarray, img_t1: np.ndarray, alpha: float) -> np.ndarray:
    g0 = cv2.cvtColor(img_t0, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(img_t1, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = img_t0.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    w0 = cv2.remap(img_t0, xs - alpha * flow[..., 0], ys - alpha * flow[..., 1],
                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    w1 = cv2.remap(img_t1, xs + (1 - alpha) * flow[..., 0], ys + (1 - alpha) * flow[..., 1],
                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return ((1 - alpha) * w0.astype(np.float32) + alpha * w1.astype(np.float32)).astype(np.uint8)


def geo_blend(
    img_t0: np.ndarray,
    img_t1: np.ndarray,
    c2w_t0: np.ndarray,
    c2w_t1: np.ndarray,
    c2w_tgt: np.ndarray,
    K: np.ndarray,
    depth: np.ndarray,
    alpha: float,
    layer_mask: np.ndarray | None = None,
    layer_depth: float | None = None,
) -> np.ndarray:
    h, w = depth.shape
    Z = layer_depth if layer_depth is not None else depth
    if layer_mask is not None and layer_depth is not None:
        Z = np.where(layer_mask, layer_depth, depth)

    mx0, my0, _ = backwarp_map(c2w_tgt, c2w_t0, K, Z, h, w)
    mx1, my1, _ = backwarp_map(c2w_tgt, c2w_t1, K, Z, h, w)
    w0 = remap_rgb(img_t0, mx0, my0)
    w1 = remap_rgb(img_t1, mx1, my1)
    return ((1 - alpha) * w0.astype(np.float32) + alpha * w1.astype(np.float32)).astype(np.uint8)


def make_depth_layers(depth: np.ndarray, n_layers: int) -> list[tuple[np.ndarray, float]]:
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return []
    vals = depth[valid]
    edges = np.quantile(vals, np.linspace(0, 1, n_layers + 1))
    edges[0] -= 1e-3
    edges[-1] += 1e-3

    layers = []
    for i in range(n_layers):
        lo, hi = edges[i], edges[i + 1]
        mask = valid & (depth >= lo) & (depth < hi)
        if not np.any(mask):
            continue
        z_med = float(np.median(depth[mask]))
        layers.append((mask, z_med))
    return layers


def layered_parallax_predict(
    img_t0: np.ndarray,
    img_t1: np.ndarray,
    c2w_t0: np.ndarray,
    c2w_t1: np.ndarray,
    c2w_tgt: np.ndarray,
    K: np.ndarray,
    depth: np.ndarray,
    alpha: float,
    n_layers: int,
    use_mean_blend: bool = True,
    alpha_cfg: AlphaTuneConfig | None = None,
    blur_cfg: BlurTuneConfig | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = depth.shape
    layers = make_depth_layers(depth, n_layers)
    mean = ((img_t0.astype(np.float32) + img_t1.astype(np.float32)) * 0.5).astype(np.uint8)
    cfg = alpha_cfg or AlphaTuneConfig()
    bcfg = blur_cfg or BlurTuneConfig()
    mw = cfg.mean_w if use_mean_blend else 0.0
    z_ref = float(np.median(depth[np.isfinite(depth) & (depth > 0)]))

    g0 = cv2.cvtColor(img_t0, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(img_t1, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    out = np.zeros((h, w, 3), dtype=np.float32)
    filled = np.zeros((h, w), dtype=bool)
    layer_preds = []

    for i, (mask, z_layer) in enumerate(layers):
        a_blend, a_pose = layer_alpha(z_layer, i, n_layers, alpha, z_ref, cfg)

        if cfg.use_pose_interp and abs(cfg.pose_beta) > 1e-6:
            c2w_ref = interp_c2w(c2w_t0, c2w_t1, a_pose)
            mx0, my0, _ = backwarp_map(c2w_ref, c2w_t0, K, z_layer, h, w)
            mx1, my1, _ = backwarp_map(c2w_ref, c2w_t1, K, z_layer, h, w)
            w0 = remap_rgb(img_t0, mx0, my0).astype(np.float32)
            w1 = remap_rgb(img_t1, mx1, my1).astype(np.float32)
        else:
            mx0, my0, _ = backwarp_map(c2w_tgt, c2w_t0, K, z_layer, h, w)
            mx1, my1, _ = backwarp_map(c2w_tgt, c2w_t1, K, z_layer, h, w)
            w0 = remap_rgb(img_t0, mx0, my0).astype(np.float32)
            w1 = remap_rgb(img_t1, mx1, my1).astype(np.float32)

        layer_pred = (1 - a_blend) * w0 + a_blend * w1
        if mw > 0:
            layer_pred = (1 - mw) * layer_pred + mw * mean.astype(np.float32)

        sig = layer_sigma(i, n_layers, flow_mag, mask, bcfg)
        layer_pred = apply_layer_blur(layer_pred.astype(np.uint8), sig)

        out[mask] = layer_pred[mask]
        filled |= mask
        layer_preds.append(layer_pred.clip(0, 255).astype(np.uint8))

    # holes: global farneback alpha blend
    if not np.all(filled):
        fallback = farneback_alpha(img_t0, img_t1, alpha).astype(np.float32)
        hole = ~filled
        out[hole] = fallback[hole]

    return out.clip(0, 255).astype(np.uint8), layer_preds


def get_lidar_depth(sample_dir: Path, camera: str, timestep: str = "target") -> np.ndarray:
    meta = json.load(open(sample_dir / "meta.json"))
    intr = meta["intrinsics"][camera]
    K = intrinsics_to_K(intr)
    W, H = int(intr["width"]), int(intr["height"])
    c2w = np.array(meta["poses_c2w"][timestep][camera], dtype=np.float64)

    npz = np.load(sample_dir / "input" / "lidar.npz")
    xyz = npz["xyz"].astype(np.float64)
    u, v, z, _ = project_to_camera(xyz, c2w, K, W, H)
    depth_map, _ = rasterize_depth(u, v, z, H, W, splat_radius=1)
    valid = np.isfinite(depth_map) & (depth_map > 0)
    raw = np.where(valid, depth_map, np.nan).astype(np.float32)
    return fill_depth_holes(raw)


def visualize(
    rgb_gt: np.ndarray,
    depth: np.ndarray,
    preds: dict[str, np.ndarray],
    scores: dict[str, float],
    layers: list,
    layer_preds: list,
    out_path: Path,
    n_layers: int,
):
    n_show = min(4, len(layer_preds))
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, max(6, n_show + 2), hspace=0.3, wspace=0.15)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rgb_gt)
    ax.set_title("TARGET (GT)")
    ax.axis("off")

    keys = ["farneback_half", "farneback_alpha", "geo_perpixel", "layered"]
    for i, k in enumerate(keys):
        if k not in preds:
            continue
        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(preds[k])
        ax.set_title(f"{k}\nPSNR {scores[k]:.2f} dB")
        ax.axis("off")

    ax = fig.add_subplot(gs[1, 0])
    d = depth.copy()
    d[~np.isfinite(d)] = np.nan
    ax.imshow(d, cmap="turbo_r")
    ax.set_title("LiDAR depth @ target")
    ax.axis("off")

    for i, (mask, z) in enumerate(layers[:n_show]):
        ax = fig.add_subplot(gs[1, i + 1])
        show = np.zeros((*mask.shape, 3), dtype=np.uint8)
        show[mask] = [255, 80, 80] if i % 2 == 0 else [80, 180, 255]
        ax.imshow(show)
        ax.set_title(f"layer {i+1}\nZ~{z:.1f}m")
        ax.axis("off")

    for i in range(n_show):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(layer_preds[i])
        ax.set_title(f"layer {i+1} pred")
        ax.axis("off")

    ax = fig.add_subplot(gs[2, n_show])
    ax.imshow(preds["layered"])
    ax.set_title(f"composite ({n_layers} layers)")
    ax.axis("off")

    err = np.mean(np.abs(preds["layered"].astype(np.float32) - rgb_gt.astype(np.float32)), axis=2)
    ax = fig.add_subplot(gs[2, n_show + 1])
    im = ax.imshow(err, cmap="hot", vmin=0, vmax=np.percentile(err, 98))
    ax.set_title("|err| layered")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def process_sample(sample_dir: Path, n_layers: int, out_dir: Path) -> dict[str, float]:
    meta = json.load(open(sample_dir / "meta.json"))
    cam = meta["target_camera"]
    ts = meta["timestamps_ns"]
    alpha = float((ts["target"] - ts["t0"]) / (ts["t1"] - ts["t0"]))

    img_t0 = load_rgb(sample_dir / "input" / "t0" / f"{cam}.jpg")
    img_t1 = load_rgb(sample_dir / "input" / "t1" / f"{cam}.jpg")
    gt = load_rgb(sample_dir / "target" / f"{cam}.jpg")

    intr = meta["intrinsics"][cam]
    K = intrinsics_to_K(intr)
    c2w_t0 = np.array(meta["poses_c2w"]["t0"][cam], dtype=np.float64)
    c2w_t1 = np.array(meta["poses_c2w"]["t1"][cam], dtype=np.float64)
    c2w_tgt = np.array(meta["poses_c2w"]["target"][cam], dtype=np.float64)

    depth = get_lidar_depth(sample_dir, cam, "target")
    layers = make_depth_layers(depth, n_layers)

    pred_half = farneback_half(img_t0, img_t1)
    pred_alpha = farneback_alpha(img_t0, img_t1, alpha)
    pred_geo = geo_blend(img_t0, img_t1, c2w_t0, c2w_t1, c2w_tgt, K, depth, alpha)
    mean = ((img_t0.astype(np.float32) + img_t1.astype(np.float32)) * 0.5).astype(np.uint8)
    pred_geo = (0.55 * pred_geo.astype(np.float32) + 0.45 * mean).astype(np.uint8)

    pred_layered_old, _ = layered_parallax_predict(
        img_t0, img_t1, c2w_t0, c2w_t1, c2w_tgt, K, depth, alpha, n_layers,
        alpha_cfg=AlphaTuneConfig(beta=0, gamma=0, mean_w=0.45),
    )
    pred_layered, layer_preds = layered_parallax_predict(
        img_t0, img_t1, c2w_t0, c2w_t1, c2w_tgt, K, depth, alpha, n_layers,
    )

    preds = {
        "farneback_half": pred_half,
        "farneback_alpha": pred_alpha,
        "geo_perpixel": pred_geo,
        "layered_old": pred_layered_old,
        "layered": pred_layered,
        "mean": mean,
    }
    scores = {k: psnr(v, gt) for k, v in preds.items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    sid = sample_dir.name[:40]
    visualize(gt, depth, preds, scores, layers, layer_preds,
              out_dir / f"layers_{sid}.png", n_layers)

    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample-dir", type=Path, default=None)
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(r"C:\Users\adel\Downloads\cv_dataset\final_dataset_v5_participants"),
    )
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--out-dir", type=Path, default=Path("layered_parallax_out"))
    args = p.parse_args()

    if args.sample_dir:
        samples = [args.sample_dir]
    else:
        train = args.dataset_dir / "train"
        all_s = sorted(p for p in train.iterdir() if p.is_dir())
        step = max(1, len(all_s) // args.num_samples)
        samples = [all_s[i * step] for i in range(args.num_samples)]

    sums: dict[str, float] = {}
    for i, sd in enumerate(samples):
        sc = process_sample(sd, args.n_layers, args.out_dir)
        for k, v in sc.items():
            sums[k] = sums.get(k, 0) + v
        print(f"[{i+1}/{len(samples)}] {sd.name[:50]}")
        for k in ["farneback_half", "farneback_alpha", "geo_perpixel", "layered", "mean"]:
            print(f"  {k:18s} {sc[k]:.2f} dB")

    n = len(samples)
    print("\n=== Average PSNR ===")
    for k in ["mean", "farneback_half", "farneback_alpha", "geo_perpixel", "layered"]:
        print(f"  {k:18s} {sums[k]/n:.2f} dB")
    print(f"\nVisualizations -> {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
