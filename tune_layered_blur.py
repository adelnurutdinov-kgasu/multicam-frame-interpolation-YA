"""Tune per-layer Gaussian blur after parallax warp+blend.

Uses best alpha config from alpha_tune_best.json.
sigma(layer i) = sigma_near + (sigma_far - sigma_near) * (i / (n-1))
              + flow_k * mean_flow_mag_on_layer
              + comp_k * mean_comp_err_on_layer  (optional)

Also tests per-pixel adaptive blur inside each layer (testdeltadifsoft-style).
"""

import argparse
import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from layered_parallax import psnr
from tune_layered_alpha import (
    AlphaTuneConfig,
    SampleCache,
    layer_alpha,
    prepare_sample,
    score_from_psnr,
)


@dataclass
class BlurTuneConfig:
    sigma_near: float = 0.0
    sigma_far: float = 0.0
    flow_k: float = 0.0
    comp_k: float = 0.0
    # adaptive per-pixel inside layer (testdeltadifsoft-like)
    adaptive: bool = False
    adapt_base: float = 0.3
    adapt_alpha: float = 0.4
    adapt_beta: float = 0.15
    adapt_max: float = 4.0


@dataclass
class EnrichedCache(SampleCache):
    layer_base: list          # float32 HxWx3 blended layer (no blur)
    flow_mag: np.ndarray      # HxW
    comp_err: np.ndarray      # HxW
    n_layers: int


def enrich_cache(cache: SampleCache, alpha_cfg: AlphaTuneConfig, n_layers: int) -> EnrichedCache:
    h, w = cache.depth.shape
    g0 = cv2.cvtColor(cache.img_t0, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(cache.img_t1, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    warped = cv2.remap(
        cache.img_t1, xs - flow[..., 0], ys - flow[..., 1],
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    comp_err = np.mean(np.abs(cache.img_t0.astype(np.float32) - warped.astype(np.float32)), axis=2)

    mw = alpha_cfg.mean_w
    layer_base = []
    for i, (mask, z_layer) in enumerate(cache.layers):
        a_blend, _ = layer_alpha(z_layer, i, n_layers, cache.alpha_time, cache.z_ref, alpha_cfg)
        w0 = cache.layer_w0[i].astype(np.float32)
        w1 = cache.layer_w1[i].astype(np.float32)
        layer = (1 - a_blend) * w0 + a_blend * w1
        if mw > 0:
            layer = (1 - mw) * layer + mw * cache.mean.astype(np.float32)
        layer_base.append(layer)

    return EnrichedCache(
        **{f.name: getattr(cache, f.name) for f in SampleCache.__dataclass_fields__.values()},
        layer_base=layer_base,
        flow_mag=flow_mag,
        comp_err=comp_err,
        n_layers=n_layers,
    )


def sigma_for_layer(i: int, n: int, cache: EnrichedCache, cfg: BlurTuneConfig) -> float:
    t = i / max(n - 1, 1)
    sigma = cfg.sigma_near + (cfg.sigma_far - cfg.sigma_near) * t
    mask = cache.layers[i][0]
    if cfg.flow_k > 0 and np.any(mask):
        sigma += cfg.flow_k * float(np.mean(cache.flow_mag[mask]))
    if cfg.comp_k > 0 and np.any(mask):
        sigma += cfg.comp_k * float(np.mean(cache.comp_err[mask]))
    return max(0.0, sigma)


def blur_layer(layer: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 1e-4:
        return layer
    return cv2.GaussianBlur(layer.astype(np.uint8), (0, 0), sigmaX=sigma, sigmaY=sigma).astype(np.float32)


def adaptive_blur_layer(
    layer: np.ndarray,
    mask: np.ndarray,
    flow_mag: np.ndarray,
    comp_err: np.ndarray,
    cfg: BlurTuneConfig,
    extra_sigma: float,
) -> np.ndarray:
    sigma_map = cfg.adapt_base + cfg.adapt_alpha * flow_mag + cfg.adapt_beta * comp_err + extra_sigma
    sigma_map = np.clip(sigma_map, 0.0, cfg.adapt_max)
    levels = [0.3, 0.5, 1.0, 2.0, 4.0]
    levels = [s for s in levels if s <= cfg.adapt_max]
    if len(levels) < 2:
        return blur_layer(layer, extra_sigma)

    blurred = [cv2.GaussianBlur(layer.astype(np.uint8), (0, 0), sigmaX=s, sigmaY=s).astype(np.float32)
               for s in levels]
    out = layer.copy().astype(np.float32)
    for i, s_val in enumerate(levels[:-1]):
        s_next = levels[i + 1]
        m = mask & (sigma_map >= s_val) & (sigma_map < s_next)
        if not np.any(m):
            continue
        alpha_mix = ((sigma_map[m] - s_val) / (s_next - s_val))[..., np.newaxis]
        out[m] = blurred[i][m] * (1 - alpha_mix) + blurred[i + 1][m] * alpha_mix
    m = mask & (sigma_map >= levels[-1])
    out[m] = blurred[-1][m]
    return out


def predict_blur(cache: EnrichedCache, cfg: BlurTuneConfig) -> np.ndarray:
    h, w = cache.depth.shape
    n = cache.n_layers
    out = np.zeros((h, w, 3), dtype=np.float32)
    filled = np.zeros((h, w), dtype=bool)

    for i, (mask, _z) in enumerate(cache.layers):
        layer = cache.layer_base[i]
        sig = sigma_for_layer(i, n, cache, cfg)
        if cfg.adaptive:
            layer = adaptive_blur_layer(layer, mask, cache.flow_mag, cache.comp_err, cfg, sig)
        else:
            layer = blur_layer(layer, sig).astype(np.float32)
        out[mask] = layer[mask]
        filled |= mask

    if not np.all(filled):
        a = cache.alpha_time
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        g0 = cv2.cvtColor(cache.img_t0, cv2.COLOR_RGB2GRAY)
        g1 = cv2.cvtColor(cache.img_t1, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        w0 = cv2.remap(cache.img_t0, xs - a * flow[..., 0], ys - a * flow[..., 1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        w1 = cv2.remap(cache.img_t1, xs + (1 - a) * flow[..., 0], ys + (1 - a) * flow[..., 1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        fb = (1 - a) * w0.astype(np.float32) + a * w1.astype(np.float32)
        hole = ~filled
        sig_h = cfg.sigma_far if cfg.sigma_far > 0 else 1.0
        fb = blur_layer(fb.astype(np.uint8), sig_h).astype(np.float32)
        out[hole] = fb[hole]

    return out.clip(0, 255).astype(np.uint8)


def eval_blur(caches: list[EnrichedCache], cfg: BlurTuneConfig) -> float:
    return float(np.mean([psnr(predict_blur(c, cfg), c.gt) for c in caches]))


def layer_sigma_table(cfg: BlurTuneConfig, n_layers: int, cache_example: EnrichedCache) -> list[dict]:
    rows = []
    for i in range(n_layers):
        if i < len(cache_example.layers):
            z = cache_example.layers[i][1]
            sig = sigma_for_layer(i, n_layers, cache_example, cfg)
            rows.append({"layer": i + 1, "z_median_m": round(z, 2), "sigma": round(sig, 3)})
    return rows


def grid_search(caches: list[EnrichedCache], n_layers: int) -> tuple[BlurTuneConfig, dict]:
    baseline = BlurTuneConfig()
    best_cfg = baseline
    best_psnr = eval_blur(caches, baseline)
    history = {"no_blur": best_psnr}
    print(f"Baseline (no blur): {best_psnr:.3f} dB  score={score_from_psnr(best_psnr):.1f}", flush=True)

    for sn in [0.0, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
        for sf in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
            if sf < sn - 1e-6:
                continue
            cfg = BlurTuneConfig(sigma_near=sn, sigma_far=sf)
            p = eval_blur(caches, cfg)
            history[f"sn={sn:.1f},sf={sf:.1f}"] = p
            if p > best_psnr:
                best_psnr, best_cfg = p, cfg

    print(f"After sigma_near/far: {best_psnr:.3f} dB  {asdict(best_cfg)}", flush=True)

    base = BlurTuneConfig(sigma_near=best_cfg.sigma_near, sigma_far=best_cfg.sigma_far)
    for fk in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]:
        for ck in [0.0, 0.005, 0.01, 0.02, 0.03]:
            cfg = BlurTuneConfig(
                sigma_near=base.sigma_near, sigma_far=base.sigma_far,
                flow_k=fk, comp_k=ck,
            )
            p = eval_blur(caches, cfg)
            history[f"fk={fk:.3f},ck={ck:.3f}"] = p
            if p > best_psnr:
                best_psnr, best_cfg = p, cfg

    print(f"After flow/comp: {best_psnr:.3f} dB  {asdict(best_cfg)}", flush=True)

    for adapt in [False, True]:
        cfg = BlurTuneConfig(
            sigma_near=best_cfg.sigma_near,
            sigma_far=best_cfg.sigma_far,
            flow_k=best_cfg.flow_k,
            comp_k=best_cfg.comp_k,
            adaptive=adapt,
        )
        p = eval_blur(caches, cfg)
        history[f"adaptive={adapt}"] = p
        if p > best_psnr:
            best_psnr, best_cfg = p, cfg

    print(f"Final: {best_psnr:.3f} dB  score={score_from_psnr(best_psnr):.1f}", flush=True)
    print(f"  {asdict(best_cfg)}", flush=True)
    return best_cfg, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(r"C:\Users\adel\Downloads\cv_dataset\final_dataset_v5_participants"),
    )
    p.add_argument("--num-samples", type=int, default=30)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--alpha-json", type=Path, default=Path("alpha_tune_best.json"))
    p.add_argument("--cache", type=Path, default=Path("blur_tune_cache.pkl"))
    p.add_argument("--out", type=Path, default=Path("blur_tune_best.json"))
    p.add_argument("--rebuild-cache", action="store_true")
    args = p.parse_args()

    alpha_cfg = AlphaTuneConfig(**json.loads(args.alpha_json.read_text())["best_config"])

    if args.cache.exists() and not args.rebuild_cache:
        print(f"Loading cache {args.cache}", flush=True)
        caches = pickle.loads(args.cache.read_bytes())
    else:
        train = args.dataset_dir / "train"
        all_s = sorted(d for d in train.iterdir() if d.is_dir())
        step = max(1, len(all_s) // args.num_samples)
        samples = [all_s[i * step] for i in range(args.num_samples)]

        print(f"Building cache for {len(samples)} samples...", flush=True)
        caches = []
        for i, sd in enumerate(samples):
            base = prepare_sample(sd, args.n_layers)
            caches.append(enrich_cache(base, alpha_cfg, args.n_layers))
            print(f"  [{i+1}/{len(samples)}] {sd.name[:45]}", flush=True)
        args.cache.write_bytes(pickle.dumps(caches))

    n_layers = caches[0].n_layers
    print(f"Samples: {len(caches)}  layers: {n_layers}", flush=True)

    best_cfg, history = grid_search(caches, n_layers)
    sigma_table = layer_sigma_table(best_cfg, n_layers, caches[0])

    out = {
        "best_blur_config": asdict(best_cfg),
        "best_psnr_db": max(history.values()),
        "best_score": score_from_psnr(max(history.values())),
        "baseline_no_blur_db": history.get("no_blur"),
        "gain_db": max(history.values()) - history.get("no_blur", 0),
        "n_samples": len(caches),
        "n_layers": n_layers,
        "alpha_config_used": asdict(alpha_cfg),
        "formula": "sigma(i) = sigma_near + (sigma_far-sigma_near)*i/(n-1) + flow_k*mean_flow_layer + comp_k*mean_comp_layer",
        "example_sigma_per_layer": sigma_table,
        "top_trials": dict(sorted(history.items(), key=lambda x: -x[1])[:20]),
    }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {args.out.resolve()}", flush=True)
    print("Per-layer sigma (example sample):", flush=True)
    for row in sigma_table:
        print(f"  layer {row['layer']}: Z~{row['z_median_m']}m  sigma={row['sigma']}", flush=True)


if __name__ == "__main__":
    main()
