"""
Доп. blend: consensus blur σ=1, RIFE blur σ=3 → LiDAR-маска.
  erode 3px → RIFE; полоса границы 10px → min(cons,RIFE); ядро → 80/20.
Поверх: mean(t0,t1) 100% в ego-зонах.

Читает готовые файлы из consensus_test_outputs, без U-Net / LiDAR rebuild.

Usage:
  python run_test_blend_blur.py
  python run_test_blend_blur.py --limit 10 --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ya_paths import (  # noqa: E402
    BAKED_ROOT,
    CONSENSUS_TEST_OUT,
    EGO_MASKS_APPROVED,
    RIFE_ROOT,
    TEST_SPLIT,
    WARPS_ROOT,
)


import cv2
import numpy as np
from PIL import Image



from consensus_kit import ConsensusConfig, _load_ego_mask, _resize_hw  # noqa: E402
from lidar_density_mask import blend_ego_mean, blend_model_rife_lidar  # noqa: E402

OUT_ROOT = CONSENSUS_TEST_OUT
BAKED_TEST = BAKED_ROOT / "test"
TEST_ROOT = TEST_SPLIT
OUT_NAME = "blend_blur50.jpg"
MASK_THR = 0.12
ALPHA = 0.8
MODEL_BLUR = 1.0
RIFE_BLUR = 3.0
MASK_ERODE = 3.0
BOUNDARY_DARK = 10.0
MASK_FEATHER = 0.0
EGO_FEATHER = 3.0
EGO_ALPHA = 1.0


def load_rgb01(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_trust(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path, quality=92)


def ego_cfg() -> ConsensusConfig:
    return ConsensusConfig(
        dataset_root=str(TEST_SPLIT),
        consensus_root=str(WARPS_ROOT / "test"),
        baked_root=str(BAKED_TEST),
        rife_root=str(RIFE_ROOT / "test"),
        ego_masks_root=str(EGO_MASKS_APPROVED),
    )


def load_mean_and_ego(sid: str, cam: str, hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    cfg = ego_cfg()
    baked_meta = json.loads((BAKED_TEST / sid / "meta.json").read_text(encoding="utf-8"))
    src = Path(baked_meta.get("source_dir", Path(cfg.dataset_root) / sid))
    t0 = _resize_hw(np.array(Image.open(src / "input" / "t0" / f"{cam}.jpg").convert("RGB")), hw, cv2.INTER_LINEAR)
    t1 = _resize_hw(np.array(Image.open(src / "input" / "t1" / f"{cam}.jpg").convert("RGB")), hw, cv2.INTER_LINEAR)
    mean = ((t0.astype(np.float32) + t1.astype(np.float32)) * 0.5 / 255.0).astype(np.float32)
    art = _load_ego_mask(cfg, sid, cam, hw)
    return mean, art


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--force", action="store_true", help="перезаписать blend_blur50.jpg")
    ap.add_argument("--ego-feather", type=float, default=EGO_FEATHER)
    ap.add_argument("--ego-alpha", type=float, default=EGO_ALPHA)
    args = ap.parse_args()

    dirs = sorted(d for d in args.out_dir.iterdir() if d.is_dir() and (d / "meta_infer.json").is_file())
    if args.limit > 0:
        dirs = dirs[: args.limit]

    ok, skip, fail, ego_applied = 0, 0, 0, 0
    t0 = time.perf_counter()

    for i, d in enumerate(dirs, 1):
        out_path = d / OUT_NAME
        if args.skip_existing and not args.force and out_path.is_file():
            skip += 1
            continue
        try:
            model = load_rgb01(d / "consensus_model.jpg")
            rife = load_rgb01(d / "rife.jpg")
            trust = load_trust(d / "lidar_trust.png")
            blend = blend_model_rife_lidar(
                model,
                rife,
                trust,
                alpha=ALPHA,
                mask_thr=MASK_THR,
                model_blur_sigma=MODEL_BLUR,
                rife_blur_sigma=RIFE_BLUR,
                mask_feather_px=MASK_FEATHER,
                mask_erode_px=MASK_ERODE,
                boundary_dark_px=BOUNDARY_DARK,
            )

            meta_infer = json.loads((d / "meta_infer.json").read_text(encoding="utf-8"))
            sid = meta_infer.get("sample_id", d.name)
            cam = meta_infer["camera"]
            hw = blend.shape[:2]
            mean_t0t1, art_mask = load_mean_and_ego(sid, cam, hw)
            if art_mask.max() > 0:
                blend = blend_ego_mean(
                    blend,
                    mean_t0t1,
                    art_mask,
                    feather_px=args.ego_feather,
                    alpha=args.ego_alpha,
                )
                ego_applied += 1

            save_rgb(out_path, blend)
            meta_infer["blend_blur50"] = {
                "file": OUT_NAME,
                "alpha": ALPHA,
                "mask_thr": MASK_THR,
                "model_blur_sigma": MODEL_BLUR,
                "rife_blur_sigma": RIFE_BLUR,
                "mask_erode_px": MASK_ERODE,
                "boundary_dark_px": BOUNDARY_DARK,
                "mask_feather_px": MASK_FEATHER,
                "ego_mean_alpha": args.ego_alpha,
                "ego_feather_px": args.ego_feather,
            }
            (d / "meta_infer.json").write_text(
                json.dumps(meta_infer, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            ok += 1
        except Exception as e:
            fail += 1
            print(f"FAIL {d.name[:50]}  {e}", flush=True)

        if i % 50 == 0 or i == len(dirs):
            print(
                f"  [{i}/{len(dirs)}] ok={ok} skip={skip} fail={fail} ego={ego_applied}  "
                f"{time.perf_counter()-t0:.0f}s",
                flush=True,
            )

    print(f"done ok={ok} skip={skip} fail={fail} ego_applied={ego_applied}  -> {OUT_NAME}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
