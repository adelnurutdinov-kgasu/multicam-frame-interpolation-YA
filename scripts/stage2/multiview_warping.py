"""
multiview_warping.py — геометрический warping для refinement RIFE.

Адаптировано под датасет YA-:
  source: final_dataset_v5_participants/train/<sample_id>/
    meta.json, input/t0|t1/<camera>.jpg, input/lidar.npz, target/<camera>.jpg
  baked:  rife_refinement_baked/train/<sample_id>/
    d1.npy (depth @ target), meta.json (camera, source_dir)

Примеры:
  # один сэмпл + превью
  python multiview_warping.py --sample-id 2025-10-01_11_45_46_12_19_35_robb_1759310438799892000__000 --preview

  # все train (или --limit N)
  python multiview_warping.py --all --limit 10

Выход на сэмпл (out-root/<sample_id>/):
  warp_*.npy, mask_*.npy, visibility_count.npy, coverage.npy, consensus_raw.npy
  consensus_tuned.npy, confidence.npy, disagreement.npy, soft_coverage.npy  (если есть RIFE)
  warp_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from ya_paths import CV_ROOT, DATASET_ROOT, REPO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lidar_depth_map import intrinsics_to_K, project_to_camera, rasterize_depth  # noqa: E402

# Камеры с FOV-overlap относительно target_camera
SOURCE_CAMERAS_BY_TARGET: Dict[str, List[str]] = {
    "front": ["front", "left_fwd", "right_fwd"],
    "left_fwd": ["left_fwd", "front", "left_bwd"],
    "right_fwd": ["right_fwd", "front", "right_bwd"],
    "left_bwd": ["left_bwd", "left_fwd", "rear"],
    "right_bwd": ["right_bwd", "right_fwd", "rear"],
    "rear": ["rear", "left_bwd", "right_bwd"],
}

OCCLUSION_TOLERANCE_M: float = 0.5

DEFAULT_DATASET = Path(r"C:/Users/adel/Downloads/cv_dataset/final_dataset_v5_participants")
DEFAULT_BAKED = Path(r"C:/Users/adel/Downloads/cv_dataset/rife_refinement_baked")
DEFAULT_RIFE = Path(r"C:/Users/adel/Downloads/cv_dataset/rife_predictions_v5")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_meta(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_source_dir(
    sample_id: str,
    dataset_dir: Path,
    baked_dir: Path | None,
) -> Tuple[Path, Path | None]:
    """Возвращает (source_sample_dir, baked_sample_dir|None)."""
    src = dataset_dir / sample_id
    baked = baked_dir / sample_id if baked_dir else None
    if baked and baked.is_dir() and (baked / "meta.json").is_file():
        bmeta = load_meta(baked / "meta.json")
        src_from_meta = Path(bmeta.get("source_dir", ""))
        if src_from_meta.is_dir():
            src = src_from_meta
        elif not src.is_dir():
            src = src_from_meta
    if not src.is_dir() or not (src / "meta.json").is_file():
        raise FileNotFoundError(f"source sample not found for {sample_id}: {src}")
    return src, baked if baked and baked.is_dir() else None


def source_cameras_for_target(target_camera: str) -> List[str]:
    cams = SOURCE_CAMERAS_BY_TARGET.get(target_camera, [target_camera])
    if target_camera not in cams:
        cams = [target_camera] + cams
    return cams


def load_rgb(path: Path) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))  # [3,H,W]


def K_from_intr(intr: dict) -> torch.Tensor:
    K = torch.zeros(3, 3, dtype=torch.float64)
    K[0, 0] = intr["fx"]
    K[1, 1] = intr["fy"]
    K[0, 2] = intr["cx"]
    K[1, 2] = intr["cy"]
    K[2, 2] = 1.0
    return K


def pose_from_list(pose: list) -> torch.Tensor:
    return torch.tensor(pose, dtype=torch.float64)


def pose_inverse(T: torch.Tensor) -> torch.Tensor:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = torch.eye(4, dtype=T.dtype, device=T.device)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def project_lidar_depth(
    sample_dir: Path,
    camera: str,
    timestep: str,
) -> np.ndarray:
    meta = load_meta(sample_dir / "meta.json")
    intr = meta["intrinsics"][camera]
    h, w = int(intr["height"]), int(intr["width"])
    K = intrinsics_to_K(intr)
    c2w = np.array(meta["poses_c2w"][timestep][camera], dtype=np.float64)
    xyz = np.load(sample_dir / "input" / "lidar.npz")["xyz"].astype(np.float64)

    u, v, z, _ = project_to_camera(xyz, c2w, K, w, h)
    depth_map, _ = rasterize_depth(u, v, z, h, w, splat_radius=1)
    valid = np.isfinite(depth_map) & (depth_map > 0)
    out = np.zeros((h, w), dtype=np.float32)
    out[valid] = depth_map[valid].astype(np.float32)
    return out


def load_target_depth(
    sample_dir: Path,
    baked_dir: Path | None,
    target_camera: str,
) -> torch.Tensor:
    if baked_dir and (baked_dir / "d1.npy").is_file():
        d = np.load(baked_dir / "d1.npy").astype(np.float32)
    else:
        d = project_lidar_depth(sample_dir, target_camera, "target")

    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(d)


def lidar_to_depth_zbuffer(
    lidar_xyz_world: torch.Tensor,
    K: torch.Tensor,
    pose_c2w: torch.Tensor,
    H: int,
    W: int,
) -> torch.Tensor:
    device = K.device
    dtype = K.dtype
    pose_w2c = pose_inverse(pose_c2w.to(dtype))
    pts = lidar_xyz_world.to(dtype).to(device)
    ones = torch.ones(pts.shape[0], 1, dtype=dtype, device=device)
    pts_h = torch.cat([pts, ones], dim=1)
    pts_cam = (pose_w2c @ pts_h.T).T[:, :3]

    Z = pts_cam[:, 2]
    in_front = Z > 0.1
    pts_cam = pts_cam[in_front]
    Z = Z[in_front]

    u = K[0, 0] * pts_cam[:, 0] / Z + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / Z + K[1, 2]
    u_i = u.long()
    v_i = v.long()
    in_frame = (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)
    u_i, v_i, Z = u_i[in_frame], v_i[in_frame], Z[in_frame]

    depth = torch.full((H, W), float("inf"), dtype=dtype, device=device)
    order = torch.argsort(Z, descending=True)
    u_i, v_i, Z = u_i[order], v_i[order], Z[order]
    depth[v_i, u_i] = Z
    depth[torch.isinf(depth)] = 0.0
    return depth


# ---------------------------------------------------------------------------
# warping core (из исходного скрипта)
# ---------------------------------------------------------------------------

def unproject_target_pixels(
    target_depth: torch.Tensor,
    K_target: torch.Tensor,
    pose_target_c2w: torch.Tensor,
) -> torch.Tensor:
    H, W = target_depth.shape
    device = target_depth.device
    dtype = target_depth.dtype

    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    fx, fy = K_target[0, 0].to(dtype), K_target[1, 1].to(dtype)
    cx, cy = K_target[0, 2].to(dtype), K_target[1, 2].to(dtype)
    Z = target_depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    P_cam = torch.stack([X, Y, Z, torch.ones_like(Z)], dim=-1)
    pose = pose_target_c2w.to(dtype).to(device)
    return torch.einsum("ij,hwj->hwi", pose, P_cam)[..., :3]


def project_to_source(
    P_world: torch.Tensor,
    K_source: torch.Tensor,
    pose_source_c2w: torch.Tensor,
    H_s: int,
    W_s: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = P_world.dtype
    device = P_world.device
    pose_w2c = pose_inverse(pose_source_c2w.to(dtype).to(device))

    Hh, Ww, _ = P_world.shape
    ones = torch.ones(Hh, Ww, 1, dtype=dtype, device=device)
    P_world_h = torch.cat([P_world, ones], dim=-1)
    P_src = torch.einsum("ij,hwj->hwi", pose_w2c, P_world_h)[..., :3]

    X_s, Y_s, Z_s = P_src[..., 0], P_src[..., 1], P_src[..., 2]
    K = K_source.to(dtype).to(device)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    Z_safe = torch.where(Z_s > 1e-3, Z_s, torch.full_like(Z_s, 1e6))
    u_s = fx * X_s / Z_safe + cx
    v_s = fy * Y_s / Z_safe + cy
    in_frame = (
        (Z_s > 0.1)
        & (u_s >= 0) & (u_s < W_s)
        & (v_s >= 0) & (v_s < H_s)
    )
    return torch.stack([u_s, v_s], dim=-1), Z_s, in_frame


def sample_source_image(
    source_img: torch.Tensor,
    uv_source: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    _, H_s, W_s = source_img.shape
    u_norm = 2.0 * uv_source[..., 0] / max(W_s - 1, 1) - 1.0
    v_norm = 2.0 * uv_source[..., 1] / max(H_s - 1, 1) - 1.0
    grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)
    sampled = F.grid_sample(
        source_img.unsqueeze(0).float(),
        grid.float(),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)
    return sampled * valid_mask.unsqueeze(0).float()


def check_occlusion(
    Z_expected: torch.Tensor,
    source_depth: torch.Tensor,
    uv_source: torch.Tensor,
    in_frame: torch.Tensor,
    tolerance_m: float = OCCLUSION_TOLERANCE_M,
) -> torch.Tensor:
    H_s, W_s = source_depth.shape
    u_norm = 2.0 * uv_source[..., 0] / max(W_s - 1, 1) - 1.0
    v_norm = 2.0 * uv_source[..., 1] / max(H_s - 1, 1) - 1.0
    grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)
    actual = F.grid_sample(
        source_depth.unsqueeze(0).unsqueeze(0).float(),
        grid.float(),
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze()
    no_data = actual < 1e-3
    diff_ok = torch.abs(Z_expected - actual) < tolerance_m
    return in_frame & (no_data | diff_ok)


def warp_source_to_target(
    source_img: torch.Tensor,
    K_source: torch.Tensor,
    pose_source_c2w: torch.Tensor,
    target_depth: torch.Tensor,
    K_target: torch.Tensor,
    pose_target_c2w: torch.Tensor,
    source_depth: Optional[torch.Tensor] = None,
    occlusion_tolerance_m: float = OCCLUSION_TOLERANCE_M,
) -> Tuple[torch.Tensor, torch.Tensor]:
    P_world = unproject_target_pixels(target_depth, K_target, pose_target_c2w)
    _, H_s, W_s = source_img.shape
    uv_s, Z_s, in_frame = project_to_source(P_world, K_source, pose_source_c2w, H_s, W_s)
    if source_depth is not None:
        valid = check_occlusion(Z_s, source_depth, uv_s, in_frame, occlusion_tolerance_m)
    else:
        valid = in_frame
    return sample_source_image(source_img, uv_s, valid), valid.float()


def build_multiview_features(
    sources: List[Dict],
    target_depth: torch.Tensor,
    K_target: torch.Tensor,
    pose_target_c2w: torch.Tensor,
    occlusion_tolerance_m: float = OCCLUSION_TOLERANCE_M,
) -> Dict:
    warps: Dict[str, torch.Tensor] = {}
    masks: Dict[str, torch.Tensor] = {}
    for src in sources:
        warped, mask = warp_source_to_target(
            source_img=src["img"],
            K_source=src["K"],
            pose_source_c2w=src["pose_c2w"],
            target_depth=target_depth,
            K_target=K_target,
            pose_target_c2w=pose_target_c2w,
            source_depth=src.get("depth"),
            occlusion_tolerance_m=occlusion_tolerance_m,
        )
        warps[src["name"]] = warped
        masks[src["name"]] = mask

    stacked_masks = torch.stack(list(masks.values()), dim=0)
    vis_count = stacked_masks.sum(dim=0)
    stacked_warps = torch.stack(list(warps.values()), dim=0)
    weights = stacked_masks.unsqueeze(1)
    weight_sum = weights.sum(dim=0).clamp(min=1e-6)
    consensus = (stacked_warps * weights).sum(dim=0) / weight_sum
    coverage = (vis_count > 0).float()
    consensus = consensus * coverage.unsqueeze(0)

    return {
        "warps": warps,
        "masks": masks,
        "visibility_count": vis_count,
        "coverage": coverage,
        "consensus": consensus,
    }


# ---------------------------------------------------------------------------
# tuned consensus (color corr, soft masks, disagreement, RIFE fill)
# ---------------------------------------------------------------------------

def color_correct_warp(
    warp: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor,
    min_valid_pixels: int = 100,
) -> torch.Tensor:
    valid = mask > 0.5
    if int(valid.sum().item()) < min_valid_pixels:
        return warp

    out = warp.clone()
    for c in range(3):
        w_vals = warp[c][valid]
        r_vals = reference[c][valid]
        w_mean, w_std = w_vals.mean(), w_vals.std().clamp(min=1e-4)
        r_mean, r_std = r_vals.mean(), r_vals.std().clamp(min=1e-4)
        out[c] = (warp[c] - w_mean) * (r_std / w_std) + r_mean
    return out.clamp(0.0, 1.0) * mask.unsqueeze(0)


def soften_mask(
    mask: torch.Tensor,
    erode_px: int = 2,
    blur_sigma: float = 1.5,
) -> torch.Tensor:
    device = mask.device
    dtype = mask.dtype
    m = mask.float()

    if erode_px > 0:
        k = 2 * erode_px + 1
        m_inv = 1.0 - m
        m_inv = F.max_pool2d(m_inv.unsqueeze(0).unsqueeze(0), k, stride=1, padding=erode_px)
        m = (1.0 - m_inv).squeeze(0).squeeze(0)

    if blur_sigma > 0:
        k = int(2 * round(3 * blur_sigma) + 1)
        if k % 2 == 0:
            k += 1
        coords = torch.arange(k, dtype=torch.float32, device=device) - (k - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * blur_sigma ** 2))
        g = g / g.sum()
        gx = g.view(1, 1, 1, k)
        gy = g.view(1, 1, k, 1)
        mm = m.unsqueeze(0).unsqueeze(0)
        mm = F.conv2d(mm, gx, padding=(0, k // 2))
        mm = F.conv2d(mm, gy, padding=(k // 2, 0))
        m = mm.squeeze(0).squeeze(0)

    return m.clamp(0.0, 1.0).to(dtype)


def compute_disagreement(
    warps: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
) -> torch.Tensor:
    stacked_w = torch.stack(list(warps.values()), dim=0)
    stacked_m = torch.stack(list(masks.values()), dim=0).unsqueeze(1)
    count = stacked_m.sum(dim=0).clamp(min=1)
    mean = (stacked_w * stacked_m).sum(dim=0) / count
    var = ((stacked_w - mean.unsqueeze(0)) ** 2 * stacked_m).sum(dim=0) / count
    std = var.sqrt().mean(dim=0)
    multi_visible = (count.squeeze(0) >= 2).float()
    return std * multi_visible


def build_tuned_consensus(
    warps: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    visibility_count: torch.Tensor,
    rife_pred: torch.Tensor,
    color_reference: Optional[torch.Tensor] = None,
    confidence_visibility_cap: float = 4.0,
    confidence_disagreement_scale: float = 0.1,
) -> Dict[str, torch.Tensor]:
    color_ref = color_reference if color_reference is not None else rife_pred

    corrected_warps = {
        name: color_correct_warp(w, color_ref, masks[name])
        for name, w in warps.items()
    }
    soft_masks = {name: soften_mask(m) for name, m in masks.items()}

    stacked_w = torch.stack(list(corrected_warps.values()), dim=0)
    stacked_sm = torch.stack(list(soft_masks.values()), dim=0).unsqueeze(1)
    weight_sum = stacked_sm.sum(dim=0).clamp(min=1e-6)
    consensus_wm = (stacked_w * stacked_sm).sum(dim=0) / weight_sum
    soft_coverage = weight_sum.squeeze(0).clamp(0, 1)

    disagreement = compute_disagreement(corrected_warps, masks)

    alpha = soft_coverage.unsqueeze(0)
    consensus_tuned = (consensus_wm * alpha + rife_pred * (1.0 - alpha)).clamp(0.0, 1.0)

    vis_norm = (visibility_count.float() / confidence_visibility_cap).clamp(0, 1)
    agree = (1.0 - (disagreement / confidence_disagreement_scale).clamp(0, 1))
    confidence = (vis_norm * agree * soft_coverage).clamp(0, 1)

    return {
        "consensus_tuned": consensus_tuned,
        "consensus_raw_wm": consensus_wm,
        "confidence": confidence,
        "disagreement": disagreement,
        "soft_coverage": soft_coverage,
    }


def load_rife_pred(
    sample_id: str,
    rife_root: Path | None,
    split: str,
    target_shape: Tuple[int, int],
) -> torch.Tensor | None:
    if rife_root is None:
        return None
    rife_path = rife_root / split / f"{sample_id}.jpg"
    if not rife_path.is_file():
        return None
    img = load_rgb(rife_path)
    H, W = target_shape
    if img.shape[-2:] != (H, W):
        img = F.interpolate(img.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
    return img


# ---------------------------------------------------------------------------
# dataset-specific loading
# ---------------------------------------------------------------------------

def load_sample_data(
    sample_dir: Path,
    baked_dir: Path | None = None,
    source_cameras: List[str] | None = None,
) -> dict:
    meta = load_meta(sample_dir / "meta.json")
    target_camera = meta["target_camera"]
    if source_cameras is None:
        source_cameras = source_cameras_for_target(target_camera)

    target_depth = load_target_depth(sample_dir, baked_dir, target_camera)
    target_K = K_from_intr(meta["intrinsics"][target_camera])
    target_pose_c2w = pose_from_list(meta["poses_c2w"]["target"][target_camera])

    xyz = np.load(sample_dir / "input" / "lidar.npz")["xyz"].astype(np.float64)
    lidar_world = torch.from_numpy(xyz).float()

    sources = []
    for time_key in ("t0", "t1"):
        for cam_name in source_cameras:
            if cam_name not in meta["intrinsics"]:
                continue
            if time_key not in meta["poses_c2w"] or cam_name not in meta["poses_c2w"][time_key]:
                continue
            img_path = sample_dir / "input" / time_key / f"{cam_name}.jpg"
            if not img_path.is_file():
                print(f"WARN: missing {img_path}")
                continue

            intr = meta["intrinsics"][cam_name]
            H, W = int(intr["height"]), int(intr["width"])
            img = load_rgb(img_path)
            if img.shape[-2:] != (H, W):
                img = F.interpolate(
                    img.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False,
                ).squeeze(0)

            sources.append({
                "name": f"{cam_name}_{time_key}",
                "img": img,
                "K": K_from_intr(intr),
                "pose_c2w": pose_from_list(meta["poses_c2w"][time_key][cam_name]),
                "depth": None,
                "_H": H,
                "_W": W,
            })

    gt_path = sample_dir / "target" / f"{target_camera}.jpg"
    gt = load_rgb(gt_path) if gt_path.is_file() else None
    H_t, W_t = int(target_depth.shape[0]), int(target_depth.shape[1])

    return {
        "meta": meta,
        "sample_id": meta.get("sample_id", sample_dir.name),
        "target_camera": target_camera,
        "target_depth": target_depth,
        "target_K": target_K,
        "target_pose_c2w": target_pose_c2w,
        "sources": sources,
        "lidar_world": lidar_world,
        "gt": gt,
        "target_shape": (H_t, W_t),
    }


def process_sample(
    sample_id: str,
    output_dir: Path,
    dataset_dir: Path,
    baked_dir: Path | None,
    source_cameras: List[str] | None = None,
    occlusion_tolerance_m: float = OCCLUSION_TOLERANCE_M,
    compute_source_depth_from_lidar: bool = True,
    device: str = "cuda",
    preview_path: Path | None = None,
    rife_root: Path | None = None,
) -> dict:
    sample_dir, baked_sample = resolve_source_dir(sample_id, dataset_dir, baked_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_sample_data(sample_dir, baked_sample, source_cameras)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    target_depth = data["target_depth"].to(dev)
    valid_depth = target_depth > 0.1
    print(
        f"[{data['sample_id']}] camera={data['target_camera']}  "
        f"depth {tuple(target_depth.shape)}  valid={valid_depth.float().mean():.1%}  "
        f"sources={len(data['sources'])}  device={dev}",
        flush=True,
    )

    target_K = data["target_K"].to(dev)
    target_pose_c2w = data["target_pose_c2w"].to(dev)

    sources = []
    for src in data["sources"]:
        img = src["img"].to(dev)
        K = src["K"].to(dev)
        pose_c2w = src["pose_c2w"].to(dev)
        depth = None
        if compute_source_depth_from_lidar and data["lidar_world"] is not None:
            depth = lidar_to_depth_zbuffer(
                data["lidar_world"].to(dev), K, pose_c2w, src["_H"], src["_W"],
            )
        sources.append({
            "name": src["name"],
            "img": img,
            "K": K,
            "pose_c2w": pose_c2w,
            "depth": depth,
        })

    result = build_multiview_features(
        sources=sources,
        target_depth=target_depth,
        K_target=target_K,
        pose_target_c2w=target_pose_c2w,
        occlusion_tolerance_m=occlusion_tolerance_m,
    )

    split = dataset_dir.name
    rife_pred = load_rife_pred(data["sample_id"], rife_root, split, data["target_shape"])
    tuned = None
    if rife_pred is not None:
        tuned = build_tuned_consensus(
            warps=result["warps"],
            masks=result["masks"],
            visibility_count=result["visibility_count"],
            rife_pred=rife_pred.to(dev),
            color_reference=rife_pred.to(dev),
        )
    else:
        print("WARN: RIFE prediction not found → tuned consensus skipped", flush=True)

    for name, w in result["warps"].items():
        np.save(output_dir / f"warp_{name}.npy", w.cpu().numpy().astype(np.float32))
    for name, m in result["masks"].items():
        np.save(output_dir / f"mask_{name}.npy", m.cpu().numpy().astype(np.float32))
    np.save(output_dir / "visibility_count.npy", result["visibility_count"].cpu().numpy().astype(np.float32))
    np.save(output_dir / "coverage.npy", result["coverage"].cpu().numpy().astype(np.float32))
    np.save(output_dir / "consensus_raw.npy", result["consensus"].cpu().numpy().astype(np.float32))

    if tuned is not None:
        np.save(output_dir / "consensus_tuned.npy", tuned["consensus_tuned"].cpu().numpy().astype(np.float32))
        np.save(output_dir / "confidence.npy", tuned["confidence"].cpu().numpy().astype(np.float32))
        np.save(output_dir / "disagreement.npy", tuned["disagreement"].cpu().numpy().astype(np.float32))
        np.save(output_dir / "soft_coverage.npy", tuned["soft_coverage"].cpu().numpy().astype(np.float32))

    summary = {
        "sample_id": data["sample_id"],
        "target_camera": data["target_camera"],
        "n_sources": len(sources),
        "coverage_total": float(result["coverage"].mean().item()),
        "visibility_max": int(result["visibility_count"].max().item()),
        "visibility_mean_in_covered": float(
            (result["visibility_count"] * result["coverage"]).sum().item()
            / max(result["coverage"].sum().item(), 1)
        ),
        "per_source_coverage": {name: float(m.mean().item()) for name, m in result["masks"].items()},
        "tuned_built": tuned is not None,
    }
    if tuned is not None:
        summary["confidence_mean"] = float(tuned["confidence"].mean().item())
        summary["soft_coverage_mean"] = float(tuned["soft_coverage"].mean().item())
        disagree = tuned["disagreement"]
        summary["disagreement_mean_in_overlap"] = float(
            disagree[disagree > 0].mean().item() if (disagree > 0).any() else 0.0
        )
    (output_dir / "warp_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    if preview_path is not None:
        save_preview(
            data=data,
            result=result,
            tuned=tuned,
            rife_pred=rife_pred,
            preview_path=preview_path,
        )

    return summary


def _to_uint8_img(t: torch.Tensor) -> np.ndarray:
    if t.ndim == 3:
        arr = t.detach().cpu().numpy().transpose(1, 2, 0)
    else:
        arr = t.detach().cpu().numpy()
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def save_preview(
    data: dict,
    result: dict,
    tuned: dict | None,
    rife_pred: torch.Tensor | None,
    preview_path: Path,
) -> None:
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cam = data["target_camera"]
    sid = data["sample_id"]

    gt = _to_uint8_img(data["gt"]) if data["gt"] is not None else None
    consensus_raw = _to_uint8_img(result["consensus"])
    coverage = result["coverage"].cpu().numpy()
    rife = _to_uint8_img(rife_pred) if rife_pred is not None else None

    n_cols = 4
    n_rows = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows))
    axes = np.atleast_2d(axes)

    def show(ax, img, title):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    def show_map(ax, arr, title, cmap="viridis", vmin=None, vmax=None):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        return im

    # row 0: GT, RIFE, raw vs tuned consensus
    show(axes[0, 0], gt if gt is not None else consensus_raw, "GT target" if gt is not None else "no GT")
    if rife is not None:
        show(axes[0, 1], rife, "RIFE pred")
    show(axes[0, 2], consensus_raw, f"consensus_raw\ncov={coverage.mean():.1%}")
    if tuned is not None:
        conf_mean = float(tuned["confidence"].mean().item())
        show(axes[0, 3], _to_uint8_img(tuned["consensus_tuned"]),
             f"consensus_tuned\nconf={conf_mean:.2f}")
    else:
        axes[0, 3].axis("off")

    # row 1: maps
    show_map(axes[1, 0], coverage, f"coverage\n{coverage.mean():.1%}")
    if tuned is not None:
        show_map(axes[1, 1], tuned["soft_coverage"].cpu().numpy(),
                 f"soft_coverage\n{tuned['soft_coverage'].mean():.1%}")
        show_map(axes[1, 2], tuned["confidence"].cpu().numpy(),
                 f"confidence\n{conf_mean:.2f}", cmap="magma", vmin=0, vmax=1)
        disagree = tuned["disagreement"].cpu().numpy()
        dmax = float(np.percentile(disagree[disagree > 0], 95)) if (disagree > 0).any() else 0.1
        show_map(axes[1, 3], disagree, "disagreement", cmap="hot", vmin=0, vmax=max(dmax, 0.05))
    else:
        for j in range(1, 4):
            axes[1, j].axis("off")

    # row 2: key warps
    priority = [f"{cam}_t0", f"{cam}_t1", f"left_fwd_t0", f"right_fwd_t0"]
    warp_names = [n for n in priority if n in result["warps"]]
    warp_names += [n for n in sorted(result["warps"]) if n not in warp_names]
    for i in range(n_cols):
        if i < len(warp_names):
            name = warp_names[i]
            show(axes[2, i], _to_uint8_img(result["warps"][name]),
                 f"warp {name}\nmask={result['masks'][name].mean():.1%}")
        else:
            axes[2, i].axis("off")

    fig.suptitle(f"{sid}\ncamera={cam}", fontsize=11)
    fig.tight_layout()
    fig.savefig(preview_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  preview -> {preview_path}", flush=True)


def iter_sample_ids(dataset_dir: Path, baked_dir: Path | None, limit: int = 0) -> List[str]:
    root = baked_dir if baked_dir and baked_dir.is_dir() else dataset_dir
    ids = sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "meta.json").is_file())
    if limit > 0:
        ids = ids[:limit]
    return ids


def main() -> int:
    p = argparse.ArgumentParser(description="Multi-view warping for YA- dataset")
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET / "train")
    p.add_argument("--baked-dir", type=Path, default=DEFAULT_BAKED / "train")
    p.add_argument("--out-root", type=Path, default=DEFAULT_BAKED.parent / "multiview_warps" / "train")
    p.add_argument("--rife-root", type=Path, default=DEFAULT_RIFE)
    p.add_argument("--sample-id", type=str, default="", help="Один sample_id")
    p.add_argument("--all", action="store_true", help="Все сэмплы из baked/train (или dataset/train)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--source-cameras", nargs="+", default=None,
                   help="Переопределить список source cameras")
    p.add_argument("--tolerance", type=float, default=OCCLUSION_TOLERANCE_M)
    p.add_argument("--no-lidar-depth", action="store_true")
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--preview-dir", type=Path,
                   default=ROOT / "methods_gallery" / "_preview" / "warps")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    baked_dir = args.baked_dir if args.baked_dir.is_dir() else None

    if args.sample_id:
        sample_ids = [args.sample_id]
    elif args.all or args.limit > 0:
        sample_ids = iter_sample_ids(args.dataset_dir, baked_dir, args.limit)
    else:
        sample_ids = iter_sample_ids(args.dataset_dir, baked_dir, limit=1)
        print(f"No --sample-id: using first sample {sample_ids[0]}")

    if not sample_ids:
        print("No samples found.")
        return 1

    for i, sid in enumerate(sample_ids):
        out_dir = args.out_root / sid
        preview = None
        if not args.no_preview and (len(sample_ids) == 1 or i == 0):
            preview = args.preview_dir / f"{sid[:48]}_warp_preview.jpg"
        try:
            summary = process_sample(
                sample_id=sid,
                output_dir=out_dir,
                dataset_dir=args.dataset_dir,
                baked_dir=baked_dir,
                source_cameras=args.source_cameras,
                occlusion_tolerance_m=args.tolerance,
                compute_source_depth_from_lidar=not args.no_lidar_depth,
                device=args.device,
                preview_path=preview,
                rife_root=args.rife_root if args.rife_root.is_dir() else None,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"ERROR {sid}: {e}", flush=True)
            if len(sample_ids) == 1:
                raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
