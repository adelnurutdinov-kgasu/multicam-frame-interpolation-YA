"""Build a sparse/dense depth map by projecting LiDAR into a camera view.

Depth = distance along camera Z axis (OpenCV: z forward), in meters.
Uses z-buffer: closest point wins per pixel.
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def intrinsics_to_K(intr: dict) -> np.ndarray:
    return np.array(
        [
            [intr["fx"], 0.0, intr["cx"]],
            [0.0, intr["fy"], intr["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def project_to_camera(
    xyz_world: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    min_depth: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns u, v, depth_z, valid indices in original cloud."""
    w2c = np.linalg.inv(c2w)
    R, t = w2c[:3, :3], w2c[:3, 3]
    pts_cam = (R @ xyz_world.T).T + t
    z = pts_cam[:, 2]
    valid = z > min_depth

    x = pts_cam[valid, 0] / z[valid]
    y = pts_cam[valid, 1] / z[valid]
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    in_img = (u >= 0) & (u < width) & (v >= 0) & (v < height)

    src_idx = np.where(valid)[0][in_img]
    return u[in_img], v[in_img], z[valid][in_img], src_idx


def rasterize_depth(
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    height: int,
    width: int,
    splat_radius: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Z-buffer rasterization. Returns (depth_map, hit_count)."""
    depth_map = np.full((height, width), np.inf, dtype=np.float32)
    hits = np.zeros((height, width), dtype=np.int32)

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    if splat_radius <= 0:
        for uu, vv, d in zip(ui, vi, depth):
            if d < depth_map[vv, uu]:
                depth_map[vv, uu] = d
            hits[vv, uu] += 1
        return depth_map, hits

    offsets = [
        (du, dv)
        for dv in range(-splat_radius, splat_radius + 1)
        for du in range(-splat_radius, splat_radius + 1)
        if du * du + dv * dv <= splat_radius * splat_radius
    ]
    for uu, vv, d in zip(ui, vi, depth):
        for du, dv in offsets:
            x, y = uu + du, vv + dv
            if 0 <= x < width and 0 <= y < height:
                if d < depth_map[y, x]:
                    depth_map[y, x] = d
                hits[y, x] += 1
    return depth_map, hits


def fill_depth_holes(depth_map: np.ndarray, max_gap: int = 25) -> np.ndarray:
    """Fill small holes via Navier-Stokes inpainting on normalized depth."""
    valid = np.isfinite(depth_map) & (depth_map > 0)
    if not np.any(valid):
        return depth_map.copy()

    d_min, d_max = depth_map[valid].min(), depth_map[valid].max()
    span = max(d_max - d_min, 1e-6)
    norm = np.zeros_like(depth_map, dtype=np.float32)
    norm[valid] = (depth_map[valid] - d_min) / span

    mask = (~valid).astype(np.uint8)
    filled = cv2.inpaint((norm * 255).astype(np.uint8), mask, max_gap, cv2.INPAINT_NS)
    out = filled.astype(np.float32) / 255.0 * span + d_min
    out[valid] = depth_map[valid]  # keep measured depths
    return out


def fill_depth_nearest(depth_map: np.ndarray, max_iters: int = 4) -> np.ndarray:
    """Dilate known depths into neighbors (cheap hole fill)."""
    out = depth_map.copy()
    valid = np.isfinite(out) & (out > 0)
    for _ in range(max_iters):
        if np.all(valid):
            break
        dilated = cv2.dilate(
            np.where(valid, out, 0).astype(np.float32),
            np.ones((3, 3), np.float32),
        )
        dil_valid = cv2.dilate(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        new = ~valid & dil_valid
        out[new] = dilated[new]
        valid = valid | new
    return out


def colorize_depth(depth: np.ndarray, valid_mask: np.ndarray, vmax_pct: float = 98.0) -> np.ndarray:
    d = depth.copy()
    d[~valid_mask] = np.nan
    if not np.any(valid_mask):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    vmax = np.nanpercentile(d, vmax_pct)
    vmin = np.nanmin(d)
    norm = np.clip((d - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    norm_u8 = (np.nan_to_num(norm, nan=0) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm_u8, cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored[~valid_mask] = 0
    return colored


def make_figure(
    rgb: np.ndarray,
    depth_raw: np.ndarray,
    depth_filled: np.ndarray | None,
    hits: np.ndarray,
    meta: dict,
    camera: str,
    timestep: str,
) -> np.ndarray:
    valid = np.isfinite(depth_raw) & (depth_raw > 0)
    coverage = 100.0 * valid.mean()

    panels = [
        ("RGB", rgb, None),
        ("Depth (sparse)", colorize_depth(depth_raw, valid), f"cover {coverage:.1f}%"),
        ("Hit count", None, None),
    ]
    if depth_filled is not None:
        vf = np.isfinite(depth_filled) & (depth_filled > 0)
        panels.append(("Depth (filled)", colorize_depth(depth_filled, vf), ""))

    # overlay
    over = rgb.copy().astype(np.float32)
    dc = colorize_depth(depth_raw, valid).astype(np.float32)
    m = valid[..., np.newaxis]
    blend = (0.55 * over + 0.45 * dc * m).astype(np.uint8)
    panels.insert(2, ("RGB + depth", blend, ""))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2))
    if n == 1:
        axes = [axes]
    sid = meta["sample_id"][-45:]
    fig.suptitle(
        f"{sid}\ncam={camera}  timestep={timestep}  delta={meta['delta_s']}s",
        fontsize=10,
    )

    for ax, (title, img, sub) in zip(axes, panels):
        if title == "Hit count":
            im = ax.imshow(np.log1p(hits), cmap="magma")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.imshow(img)
        t = title if not sub else f"{title}\n{sub}"
        ax.set_title(t, fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    fig.canvas.draw()
    out = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return out


def build_depth_map(
    sample_dir: Path,
    camera: str | None = None,
    timestep: str = "target",
    splat_radius: int = 1,
    fill: str = "none",
    max_points: int | None = None,
) -> dict:
    meta = json.load(open(sample_dir / "meta.json"))
    cam = camera or meta["target_camera"]
    intr = meta["intrinsics"][cam]
    K = intrinsics_to_K(intr)
    W, H = int(intr["width"]), int(intr["height"])
    c2w = np.array(meta["poses_c2w"][timestep][cam], dtype=np.float64)

    img_path = sample_dir / "input" / timestep / f"{cam}.jpg"
    if not img_path.exists():
        img_path = sample_dir / "target" / f"{cam}.jpg"
    rgb = np.array(Image.open(img_path).convert("RGB"))

    npz = np.load(sample_dir / "input" / "lidar.npz")
    xyz = npz["xyz"].astype(np.float64)
    if max_points and xyz.shape[0] > max_points:
        idx = np.random.default_rng(42).choice(xyz.shape[0], max_points, replace=False)
        xyz = xyz[idx]

    u, v, depth, _ = project_to_camera(xyz, c2w, K, W, H)
    depth_map, hits = rasterize_depth(u, v, depth, H, W, splat_radius=splat_radius)

    valid = np.isfinite(depth_map) & (depth_map > 0)
    depth_out = np.where(valid, depth_map, np.nan).astype(np.float32)

    depth_filled = None
    if fill == "inpaint":
        depth_filled = fill_depth_holes(np.where(valid, depth_map, np.nan))
    elif fill == "dilate":
        dm = depth_map.copy()
        dm[~valid] = 0
        depth_filled = fill_depth_nearest(dm)
        depth_filled[~valid & (depth_filled <= 0)] = np.nan

    return {
        "rgb": rgb,
        "depth": depth_out,
        "depth_filled": depth_filled,
        "hits": hits,
        "meta": meta,
        "camera": cam,
        "timestep": timestep,
        "n_projected": len(u),
    }


def main():
    p = argparse.ArgumentParser(description="LiDAR -> camera depth map")
    p.add_argument("--sample-dir", type=Path, required=True)
    p.add_argument("--camera", default=None, help="default: target_camera from meta")
    p.add_argument("--timestep", choices=["t0", "t1", "target"], default="target")
    p.add_argument("--splat-radius", type=int, default=1, help="disk radius in pixels (0=single pixel)")
    p.add_argument("--fill", choices=["none", "inpaint", "dilate"], default="inpaint")
    p.add_argument("--max-points", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("depth_out"))
    p.add_argument("--all-timesteps", action="store_true")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timesteps = ("t0", "t1", "target") if args.all_timesteps else (args.timestep,)

    for ts in timesteps:
        result = build_depth_map(
            args.sample_dir,
            args.camera,
            ts,
            args.splat_radius,
            args.fill,
            args.max_points,
        )
        cam = result["camera"]
        stem = f"{cam}_{ts}"
        np.save(args.out_dir / f"depth_{stem}.npy", result["depth"])
        if result["depth_filled"] is not None:
            np.save(args.out_dir / f"depth_{stem}_filled.npy", result["depth_filled"])

        fig = make_figure(
            result["rgb"],
            result["depth"],
            result["depth_filled"],
            result["hits"],
            result["meta"],
            cam,
            ts,
        )
        png = args.out_dir / f"depth_{stem}.png"
        Image.fromarray(fig).save(png)

        valid = np.isfinite(result["depth"])
        d = result["depth"][valid]
        print(f"[{ts}] camera={cam}  projected={result['n_projected']:,}  "
              f"coverage={100*valid.mean():.1f}%  "
              f"depth range={d.min():.1f}..{d.max():.1f} m  -> {png}")


if __name__ == "__main__":
    main()
