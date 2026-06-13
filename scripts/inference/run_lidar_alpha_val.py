"""Val alpha sweep: val_cache_front_rear.npz (71 val) + r3+blur trust, без discover_samples."""
REPO = Path(__file__).resolve().parents[2]
import sys
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import json
import sys
import time
from pathlib import Path

import numpy as np



from consensus_kit import ValCacheStacked, default_config, psnr_uint8_np
from lidar_density_mask import blend_model_rife_lidar, lidar_trust_for_sample

CACHE_PATH = REPO / "checkpoints_consensus" / "val_cache_front_rear.npz"
OUT_JSON = REPO / "checkpoints_consensus" / "lidar_alpha_sweep_val.json"


def main():
    cfg = default_config()
    cfg.allowed_cameras = ("front", "rear")
    cfg.lidar_spread_radius_fine = 3.0
    cfg.lidar_spread_blur_fine = 1.0
    cfg.lidar_zone_min = 0.12
    mask_thr = cfg.lidar_zone_min
    hw = (cfg.image_h, cfg.image_w)

    if not CACHE_PATH.is_file():
        raise FileNotFoundError(CACHE_PATH)

    t0 = time.perf_counter()
    z = np.load(CACHE_PATH, allow_pickle=True)
    cache = ValCacheStacked(z, promote_rgb=True)
    n = len(cache)
    print(f"cache {n} samples  ({time.perf_counter() - t0:.1f}s)", flush=True)

    rows = []
    for i in range(n):
        e = cache[i]
        src = Path(e["meta"].get("source_dir", Path(cfg.dataset_root) / e["meta"]["sample_id"]))
        trust = lidar_trust_for_sample(
            src,
            e["meta"]["camera"],
            hw,
            spread_radius_fine=cfg.lidar_spread_radius_fine,
            spread_blur_fine=cfg.lidar_spread_blur_fine,
            zone_min=cfg.lidar_zone_min,
        )
        rows.append({"model": e["model"], "rife": e["rife"], "gt": e["gt"], "trust": trust})
        if (i + 1) % 10 == 0 or i + 1 == n:
            print(f"  {i+1}/{n}  {e['meta']['sample_id'][:36]}...", flush=True)

    b_model = float(np.mean([psnr_uint8_np(r["model"], r["gt"]) for r in rows]))
    b_rife = float(np.mean([psnr_uint8_np(r["rife"], r["gt"]) for r in rows]))
    print(f"baseline  model={b_model:.3f}  RIFE={b_rife:.3f}", flush=True)

    alphas = np.round(np.linspace(0, 1, 21), 2)
    mean_psnr = []
    for a in alphas:
        ps = [
            psnr_uint8_np(
                blend_model_rife_lidar(
                    r["model"], r["rife"], r["trust"], alpha=float(a), mask_thr=mask_thr
                ),
                r["gt"],
            )
            for r in rows
        ]
        m = float(np.mean(ps))
        mean_psnr.append(m)
        print(f"  alpha={a:.2f}  mean PSNR={m:.3f}  dRIFE={m - b_rife:+.3f}", flush=True)

    best_i = int(np.argmax(mean_psnr))
    best_alpha = float(alphas[best_i])
    print(
        f"\nBEST alpha={best_alpha:.2f}  mean PSNR={mean_psnr[best_i]:.3f}  "
        f"(gain vs RIFE {mean_psnr[best_i] - b_rife:+.3f})",
        flush=True,
    )

    OUT_JSON.write_text(
        json.dumps(
            {
                "n_val": n,
                "cache": str(CACHE_PATH),
                "mask_thr": mask_thr,
                "lidar_r3": cfg.lidar_spread_radius_fine,
                "lidar_blur": cfg.lidar_spread_blur_fine,
                "b_model": b_model,
                "b_rife": b_rife,
                "alphas": alphas.tolist(),
                "mean_psnr": mean_psnr,
                "best_alpha": best_alpha,
                "best_mean_psnr": mean_psnr[best_i],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
