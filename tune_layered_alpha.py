"""Tune per-layer temporal alpha for layered parallax blending.

alpha_eff(Z) = clamp(alpha_time + beta * (Z_ref/Z - 1) + gamma * layer_idx/(n-1), 0, 1)

Optionally unproject at interpolated pose c2w_interp(alpha_pose(Z)) instead of target.
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

from lidar_depth_map import intrinsics_to_K
from layered_parallax import (
    backwarp_map,
    get_lidar_depth,
    load_rgb,
    make_depth_layers,
    psnr,
    remap_rgb,
)


@dataclass
class AlphaTuneConfig:
    beta: float = 0.0          # depth correction: +beta*(Z_ref/Z - 1)
    gamma: float = 0.0         # layer index bias
    mean_w: float = 0.45       # weight of mean(t0,t1) in output
    pose_beta: float = 0.0     # shift unproject pose: alpha_pose = alpha_time + pose_beta*(Z_ref/Z-1)
    use_pose_interp: bool = False


def score_from_psnr(p: float) -> float:
    return max(0.0, min(100.0, (max(10.0, min(30.0, p)) - 10.0) / 20.0 * 100.0))


def interp_c2w(c2w0: np.ndarray, c2w1: np.ndarray, a: float) -> np.ndarray:
    R0, t0 = c2w0[:3, :3], c2w0[:3, 3]
    R1, t1 = c2w1[:3, :3], c2w1[:3, 3]
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([R0, R1])))
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = slerp(float(np.clip(a, 0.0, 1.0))).as_matrix()
    out[:3, 3] = (1 - a) * t0 + a * t1
    return out


def camera_baseline_m(c2w_t0: np.ndarray, c2w_t1: np.ndarray) -> float:
    return float(np.linalg.norm(c2w_t1[:3, 3] - c2w_t0[:3, 3]))


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


@dataclass
class SampleCache:
    img_t0: np.ndarray
    img_t1: np.ndarray
    gt: np.ndarray
    mean: np.ndarray
    depth: np.ndarray
    layers: list
    alpha_time: float
    z_ref: float
    baseline_m: float
    c2w_t0: np.ndarray
    c2w_t1: np.ndarray
    c2w_tgt: np.ndarray
    K: np.ndarray
    # per layer: w0, w1 from target-pose unproject (default)
    layer_w0: list
    layer_w1: list
    masks: list


def prepare_sample(sample_dir: Path, n_layers: int) -> SampleCache:
    meta = json.load(open(sample_dir / "meta.json"))
    cam = meta["target_camera"]
    ts = meta["timestamps_ns"]
    alpha_time = float((ts["target"] - ts["t0"]) / (ts["t1"] - ts["t0"]))

    img_t0 = load_rgb(sample_dir / "input" / "t0" / f"{cam}.jpg")
    img_t1 = load_rgb(sample_dir / "input" / "t1" / f"{cam}.jpg")
    gt = load_rgb(sample_dir / "target" / f"{cam}.jpg")
    mean = ((img_t0.astype(np.float32) + img_t1.astype(np.float32)) * 0.5).astype(np.uint8)

    K = intrinsics_to_K(meta["intrinsics"][cam])
    c2w_t0 = np.array(meta["poses_c2w"]["t0"][cam], dtype=np.float64)
    c2w_t1 = np.array(meta["poses_c2w"]["t1"][cam], dtype=np.float64)
    c2w_tgt = np.array(meta["poses_c2w"]["target"][cam], dtype=np.float64)

    depth = get_lidar_depth(sample_dir, cam, "target")
    layers = make_depth_layers(depth, n_layers)
    h, w = depth.shape
    z_ref = float(np.median(depth[np.isfinite(depth) & (depth > 0)]))

    layer_w0, layer_w1, masks = [], [], []
    for mask, z_layer in layers:
        mx0, my0, _ = backwarp_map(c2w_tgt, c2w_t0, K, z_layer, h, w)
        mx1, my1, _ = backwarp_map(c2w_tgt, c2w_t1, K, z_layer, h, w)
        layer_w0.append(remap_rgb(img_t0, mx0, my0))
        layer_w1.append(remap_rgb(img_t1, mx1, my1))
        masks.append(mask)

    return SampleCache(
        img_t0=img_t0, img_t1=img_t1, gt=gt, mean=mean, depth=depth,
        layers=layers, alpha_time=alpha_time, z_ref=z_ref,
        baseline_m=camera_baseline_m(c2w_t0, c2w_t1),
        c2w_t0=c2w_t0, c2w_t1=c2w_t1, c2w_tgt=c2w_tgt, K=K,
        layer_w0=layer_w0, layer_w1=layer_w1, masks=masks,
    )


def predict_tuned(cache: SampleCache, cfg: AlphaTuneConfig, n_layers: int) -> np.ndarray:
    h, w = cache.depth.shape
    out = np.zeros((h, w, 3), dtype=np.float32)
    filled = np.zeros((h, w), dtype=bool)
    mw = cfg.mean_w

    for i, (mask, z_layer) in enumerate(cache.layers):
        a_blend, a_pose = layer_alpha(
            z_layer, i, n_layers, cache.alpha_time, cache.z_ref, cfg,
        )

        if cfg.use_pose_interp and abs(cfg.pose_beta) > 1e-6:
            c2w_ref = interp_c2w(cache.c2w_t0, cache.c2w_t1, a_pose)
            mx0, my0, _ = backwarp_map(c2w_ref, cache.c2w_t0, cache.K, z_layer, h, w)
            mx1, my1, _ = backwarp_map(c2w_ref, cache.c2w_t1, cache.K, z_layer, h, w)
            w0 = remap_rgb(cache.img_t0, mx0, my0).astype(np.float32)
            w1 = remap_rgb(cache.img_t1, mx1, my1).astype(np.float32)
        else:
            w0 = cache.layer_w0[i].astype(np.float32)
            w1 = cache.layer_w1[i].astype(np.float32)

        layer = (1 - a_blend) * w0 + a_blend * w1
        if mw > 0:
            layer = (1 - mw) * layer + mw * cache.mean.astype(np.float32)

        out[mask] = layer[mask]
        filled |= mask

    if not np.all(filled):
        # fallback: global alpha_time
        a = cache.alpha_time
        g0 = cv2.cvtColor(cache.img_t0, cv2.COLOR_RGB2GRAY)
        g1 = cv2.cvtColor(cache.img_t1, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        w0 = cv2.remap(cache.img_t0, xs - a * flow[..., 0], ys - a * flow[..., 1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        w1 = cv2.remap(cache.img_t1, xs + (1 - a) * flow[..., 0], ys + (1 - a) * flow[..., 1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        fb = ((1 - a) * w0.astype(np.float32) + a * w1.astype(np.float32))
        out[~filled] = fb[~filled]

    return out.clip(0, 255).astype(np.uint8)


def eval_config(caches: list[SampleCache], cfg: AlphaTuneConfig, n_layers: int) -> float:
    psnrs = [psnr(predict_tuned(c, cfg, n_layers), c.gt) for c in caches]
    return float(np.mean(psnrs))


def grid_search(caches: list[SampleCache], n_layers: int) -> tuple[AlphaTuneConfig, dict]:
    baseline = AlphaTuneConfig()
    best_cfg = baseline
    best_psnr = eval_config(caches, baseline, n_layers)
    history = {"baseline": best_psnr}

    print(f"Baseline (alpha=time, beta=0): {best_psnr:.3f} dB  score={score_from_psnr(best_psnr):.1f}")

    # --- stage 1: beta (depth correction) x mean_w ---
    for beta in np.arange(-0.25, 0.26, 0.05):
        for mean_w in [0.35, 0.40, 0.45, 0.50, 0.55]:
            cfg = AlphaTuneConfig(beta=float(beta), mean_w=mean_w)
            p = eval_config(caches, cfg, n_layers)
            key = f"beta={beta:.2f},mw={mean_w:.2f}"
            history[key] = p
            if p > best_psnr:
                best_psnr, best_cfg = p, cfg

    print(f"After beta x mean_w: {best_psnr:.3f} dB  cfg={asdict(best_cfg)}")

    # --- stage 2: gamma (layer index) around best beta ---
    for gamma in np.arange(-0.15, 0.16, 0.05):
        cfg = AlphaTuneConfig(**{**asdict(best_cfg), "gamma": float(gamma)})
        p = eval_config(caches, cfg, n_layers)
        history[f"gamma={gamma:.2f}"] = p
        if p > best_psnr:
            best_psnr, best_cfg = p, cfg

    print(f"After gamma: {best_psnr:.3f} dB  cfg={asdict(best_cfg)}")

    # --- stage 3: pose interpolation ref ---
    for pose_beta in np.arange(-0.20, 0.21, 0.05):
        cfg = AlphaTuneConfig(**{**asdict(best_cfg), "pose_beta": float(pose_beta), "use_pose_interp": True})
        p = eval_config(caches, cfg, n_layers)
        history[f"pose_beta={pose_beta:.2f}"] = p
        if p > best_psnr:
            best_psnr, best_cfg = p, cfg

    # also try pose_interp=False with best blend params only
    cfg_no_pose = AlphaTuneConfig(
        beta=best_cfg.beta, gamma=best_cfg.gamma, mean_w=best_cfg.mean_w,
        pose_beta=0.0, use_pose_interp=False,
    )
    p = eval_config(caches, cfg_no_pose, n_layers)
    if p >= best_psnr:
        best_psnr, best_cfg = p, cfg_no_pose

    print(f"Final best: {best_psnr:.3f} dB  score={score_from_psnr(best_psnr):.1f}")
    print(f"  {asdict(best_cfg)}")
    return best_cfg, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(r"C:\Users\adel\Downloads\cv_dataset\final_dataset_v5_participants"),
    )
    p.add_argument("--num-samples", type=int, default=40)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--out", type=Path, default=Path("alpha_tune_best.json"))
    args = p.parse_args()

    train = args.dataset_dir / "train"
    all_s = sorted(d for d in train.iterdir() if d.is_dir())
    step = max(1, len(all_s) // args.num_samples)
    samples = [all_s[i * step] for i in range(args.num_samples)]

    print(f"Caching {len(samples)} samples...", flush=True)
    caches = []
    for i, sd in enumerate(samples):
        caches.append(prepare_sample(sd, args.n_layers))
        print(f"  [{i+1}/{len(samples)}] {sd.name[:45]}", flush=True)

    baselines = [c.baseline_m for c in caches]
    alphas = [c.alpha_time for c in caches]
    print(f"Camera baseline: {np.mean(baselines):.2f}m (range {min(baselines):.2f}-{max(baselines):.2f})")
    print(f"alpha_time:      {np.mean(alphas):.3f} (range {min(alphas):.3f}-{max(alphas):.3f})")

    best_cfg, history = grid_search(caches, args.n_layers)

    out = {
        "best_config": asdict(best_cfg),
        "best_psnr_db": max(history.values()),
        "best_score": score_from_psnr(max(history.values())),
        "n_samples": len(caches),
        "n_layers": args.n_layers,
        "formula": "alpha_blend = clamp(alpha_time + beta*(Z_ref/Z - 1) + gamma*layer_idx/(n-1), 0, 1)",
        "top_trials": dict(sorted(history.items(), key=lambda x: -x[1])[:15]),
    }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {args.out.resolve()}")


if __name__ == "__main__":
    main()
