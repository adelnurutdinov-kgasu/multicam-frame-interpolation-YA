"""Quick PSNR eval: consensus_tuned vs GT on one sample."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def psnr_uint8(pred: np.ndarray, gt: np.ndarray) -> float:
    p = (np.clip(pred, 0, 1) * 255).round().astype(np.uint8)
    g = (np.clip(gt, 0, 1) * 255).round().astype(np.uint8)
    mse = np.mean((p.astype(np.float64) - g.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 99.0
    return float(10 * np.log10(255**2 / mse))


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_chw_npy(path: Path) -> np.ndarray:
    x = np.load(path).astype(np.float32)
    if x.ndim == 3 and x.shape[0] == 3:
        x = x.transpose(1, 2, 0)
    return x


def main() -> None:
    sid = "2025-10-01_11_45_46_12_19_35_robb_1759310438799892000__000"
    warp_dir = Path(r"C:/Users/adel/Downloads/cv_dataset/multiview_warps/train") / sid
    src = Path(r"C:/Users/adel/Downloads/cv_dataset/final_dataset_v5_participants/train") / sid
    rife_path = Path(r"C:/Users/adel/Downloads/cv_dataset/rife_predictions_v5/train") / f"{sid}.jpg"

    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    cam = meta["target_camera"]

    gt = load_rgb(src / "target" / f"{cam}.jpg")
    rife = load_rgb(rife_path)
    raw = load_chw_npy(warp_dir / "consensus_raw.npy")
    tuned = load_chw_npy(warp_dir / "consensus_tuned.npy")
    conf = np.load(warp_dir / "confidence.npy")
    soft = np.load(warp_dir / "soft_coverage.npy")
    summary = json.loads((warp_dir / "warp_summary.json").read_text(encoding="utf-8"))

    methods = {"RIFE": rife, "consensus_raw": raw, "consensus_tuned": tuned}
    rife_psnr = psnr_uint8(rife, gt)

    print(f"sample: {sid}")
    print(f"camera: {cam}  shape: {gt.shape[0]}x{gt.shape[1]}")
    print(f"coverage: {summary['coverage_total']:.1%}  confidence_mean: {summary.get('confidence_mean', 0):.3f}")
    print()
    print(f"{'method':<18} {'PSNR':>8} {'MAE':>10} {'dPSNR vs RIFE':>14}")
    print("-" * 54)
    for name, img in methods.items():
        p = psnr_uint8(img, gt)
        m = float(np.mean(np.abs(img - gt)))
        print(f"{name:<18} {p:8.3f} {m:10.5f} {p - rife_psnr:+14.3f}")

    print("\n--- по маскам ---")
    for label, mask in [("confidence", conf), ("soft_coverage", soft)]:
        for thr in (0.3, 0.5, 0.7):
            m = mask >= thr
            n = int(m.sum())
            if n < 100:
                continue
            pt = psnr_uint8(tuned[m].reshape(-1, 3), gt[m].reshape(-1, 3))
            pr = psnr_uint8(rife[m].reshape(-1, 3), gt[m].reshape(-1, 3))
            print(f"{label}>={thr}: tuned={pt:.2f}  RIFE={pr:.2f}  delta={pt - pr:+.2f}  px={n} ({100*n/m.size:.1f}%)")

    print("\n--- raw hi-conf поверх RIFE (остальное RIFE) ---")
    print(f"{'thr':>6} {'px':>10} {'%':>7} {'full PSNR':>10} {'dRIFE':>8} {'raw@mask':>10}")
    print("-" * 56)
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        m = conf >= thr
        n = int(m.sum())
        if n == 0:
            continue
        comp = np.where(m[..., None], raw, rife)
        p_full = psnr_uint8(comp, gt)
        p_raw = psnr_uint8(raw[m].reshape(-1, 3), gt[m].reshape(-1, 3))
        print(f"{thr:6.1f} {n:10d} {100*n/conf.size:6.1f}% {p_full:10.3f} {p_full-rife_psnr:+8.3f} {p_raw:10.2f}")


if __name__ == "__main__":
    main()
