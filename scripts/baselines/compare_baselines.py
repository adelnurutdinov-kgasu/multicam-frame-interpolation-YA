"""Compare official baselines vs testdeltadif.py methods on N train samples."""

REPO = Path(__file__).resolve().parents[2]
import sys
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent
BASELINE_DIR = ROOT / "baseline_files" / "baseline_ensemble"
sys.path.insert(0, str(BASELINE_DIR / "ECCV2022-RIFE"))
from train_log.RIFE_HDv3 import Model  # noqa: E402

# --- params from testdeltadif.py ---
FLOW_THRESH = 5.0
COMP_ERROR_THRESH = 20
BLUR_SIGMA_FALLBACK = 4.0
BLOCK_SIZE = 32

SOFT_PARAMS = {
    "base_sigma": 0.3,
    "alpha": 0.4,
    "beta": 0.15,
    "max_sigma": 4.0,
    "sigma_levels": [0.3, 0.5, 1.0, 2.0, 4.0],
}
AGGR_PARAMS = {
    "base_sigma": 0.5,
    "alpha": 0.8,
    "beta": 0.3,
    "max_sigma": 8.0,
    "sigma_levels": [0.5, 1.0, 2.0, 4.0, 8.0],
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    return 20.0 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float("inf")


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def warp_flow(img: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(
        img,
        (xs + dx).astype(np.float32),
        (ys + dy).astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def warp_half(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    return warp_flow(img, -flow[..., 0] / 2.0, -flow[..., 1] / 2.0)


def compensation_error(img_t0: np.ndarray, img_t1: np.ndarray, flow: np.ndarray) -> np.ndarray:
    warped = warp_flow(img_t1, -flow[..., 0], -flow[..., 1])
    return np.mean(np.abs(img_t0.astype(np.float32) - warped.astype(np.float32)), axis=2)


def adaptive_blur(img_t1: np.ndarray, flow: np.ndarray, comp_err: np.ndarray, params: dict) -> np.ndarray:
    flow_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    sigma_map = params["base_sigma"] + params["alpha"] * flow_mag + params["beta"] * comp_err
    levels = params["sigma_levels"]
    sigma_map = np.clip(sigma_map, levels[0], params["max_sigma"])

    pred_flow = warp_half(img_t1, flow)
    blurred_levels = [cv2.GaussianBlur(pred_flow, (0, 0), sigmaX=s, sigmaY=s) for s in levels]
    pred = np.zeros_like(pred_flow, dtype=np.float32)
    for i, s_val in enumerate(levels[:-1]):
        s_next = levels[i + 1]
        mask = (sigma_map >= s_val) & (sigma_map < s_next)
        if not np.any(mask):
            continue
        alpha_mix = ((sigma_map[mask] - s_val) / (s_next - s_val))[..., np.newaxis]
        pred[mask] = blurred_levels[i][mask] * (1 - alpha_mix) + blurred_levels[i + 1][mask] * alpha_mix
    pred[sigma_map >= levels[-1]] = blurred_levels[-1][sigma_map >= levels[-1]]
    return pred.astype(np.uint8)


def phase_correlation_blocks(img_t0: np.ndarray, img_t1: np.ndarray) -> np.ndarray:
    h, w = img_t0.shape[:2]
    gray0 = cv2.cvtColor(img_t0, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(img_t1, cv2.COLOR_RGB2GRAY)
    pred = np.zeros((h, w, 3), dtype=np.float64)
    count = np.zeros((h, w), dtype=np.float64)

    for y0 in range(0, h - BLOCK_SIZE + 1, BLOCK_SIZE):
        for x0 in range(0, w - BLOCK_SIZE + 1, BLOCK_SIZE):
            b0 = gray0[y0 : y0 + BLOCK_SIZE, x0 : x0 + BLOCK_SIZE].astype(np.float32)
            b1 = gray1[y0 : y0 + BLOCK_SIZE, x0 : x0 + BLOCK_SIZE].astype(np.float32)
            try:
                shift, _ = cv2.phaseCorrelate(b0, b1)
            except cv2.error:
                shift = (0.0, 0.0)
            dx, dy = -shift[0] / 2.0, -shift[1] / 2.0
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            patch = cv2.warpAffine(
                img_t1[y0 : y0 + BLOCK_SIZE, x0 : x0 + BLOCK_SIZE],
                M,
                (BLOCK_SIZE, BLOCK_SIZE),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            pred[y0 : y0 + BLOCK_SIZE, x0 : x0 + BLOCK_SIZE] += patch
            count[y0 : y0 + BLOCK_SIZE, x0 : x0 + BLOCK_SIZE] += 1.0

    count3 = np.repeat(count[:, :, np.newaxis], 3, axis=2)
    return (pred / count3).astype(np.uint8)


class RifeWrapper:
    def __init__(self):
        self.model = Model()
        self.model.load_model(str(BASELINE_DIR / "train_log"), -1)
        self.model.eval()

    def infer(self, img0: np.ndarray, img1: np.ndarray) -> np.ndarray:
        h, w = img0.shape[:2]
        ph = ((h - 1) // 64 + 1) * 64
        pw = ((w - 1) // 64 + 1) * 64

        def to_t(x):
            return torch.from_numpy(x).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(DEVICE)

        t0 = F.pad(to_t(img0), (0, pw - w, 0, ph - h))
        t1 = F.pad(to_t(img1), (0, pw - w, 0, ph - h))
        with torch.no_grad():
            out = self.model.inference(t0, t1)
        return (out[0, :, :h, :w].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def official_dis(img0, img1, alpha, dis):
    mean = (img0.astype(np.float32) + img1.astype(np.float32)) * 0.5
    gray0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    flow = dis.calc(gray0, gray1, None)
    dx, dy = flow[..., 0], flow[..., 1]
    w0 = warp_flow(img0, -alpha * dx, -alpha * dy)
    w1 = warp_flow(img1, (1 - alpha) * dx, (1 - alpha) * dy)
    dis_warped = (1 - alpha) * w0.astype(np.float32) + alpha * w1.astype(np.float32)
    return (0.55 * dis_warped + 0.45 * mean).clip(0, 255).astype(np.uint8)


def predict_sample(sample_dir: Path, dis, rife: RifeWrapper, skip_phase: bool = True) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    meta = json.loads((sample_dir / "meta.json").read_text())
    cam = meta["target_camera"]
    ts = meta["timestamps_ns"]
    alpha = float((ts["target"] - ts["t0"]) / (ts["t1"] - ts["t0"]))

    img0 = load_rgb(sample_dir / "input" / "t0" / f"{cam}.jpg")
    img1 = load_rgb(sample_dir / "input" / "t1" / f"{cam}.jpg")
    gt = load_rgb(sample_dir / "target" / f"{cam}.jpg")

    gray0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    mean = ((img0.astype(np.float32) + img1.astype(np.float32)) * 0.5).astype(np.uint8)

    flow_farn = cv2.calcOpticalFlowFarneback(
        gray1, gray0, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    comp_err = compensation_error(img0, img1, flow_farn)
    flow_mag = np.sqrt(flow_farn[..., 0] ** 2 + flow_farn[..., 1] ** 2)

    pred_farn = warp_half(img1, flow_farn)
    bad_mask = (flow_mag > FLOW_THRESH) | (comp_err > COMP_ERROR_THRESH)
    pred_farn_agg = pred_farn.copy()
    blurred_t1 = cv2.GaussianBlur(img1, (0, 0), sigmaX=BLUR_SIGMA_FALLBACK, sigmaY=BLUR_SIGMA_FALLBACK)
    pred_farn_agg[bad_mask] = blurred_t1[bad_mask]

    flow_dis_u = dis.calc(gray1, gray0, None)
    pred_dis_half = warp_half(img1, flow_dis_u)

    preds = {
        "t0": img0,
        "t1": img1,
        "GT": gt,
        "mean": mean,
        "global_blur": cv2.GaussianBlur(img1, (0, 0), sigmaX=2.0, sigmaY=2.0),
        "farneback_half": pred_farn,
        "farneback_aggr": pred_farn_agg,
        "dis_half": pred_dis_half,
        "adaptive_soft": adaptive_blur(img1, flow_farn, comp_err, SOFT_PARAMS),
        "adaptive_aggr": adaptive_blur(img1, flow_farn, comp_err, AGGR_PARAMS),
        "dis_official": official_dis(img0, img1, alpha, dis),
        "rife": rife.infer(img0, img1),
    }
    preds["ensemble"] = (
        0.60 * preds["dis_official"].astype(np.float32) + 0.40 * preds["rife"].astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    if not skip_phase:
        preds["phase_blocks"] = phase_correlation_blocks(img0, img1)

    return preds, gt, meta


def run_sample(sample_dir: Path, dis, rife: RifeWrapper, skip_phase: bool) -> dict[str, float]:
    preds, gt, _ = predict_sample(sample_dir, dis, rife, skip_phase)
    return {name: psnr(p, gt) for name, p in preds.items() if name != "GT"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(r"C:\Users\adel\Downloads\cv_dataset\final_dataset_v5_participants"),
    )
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--skip-phase", action="store_true", help="Skip slow phase correlation")
    args = p.parse_args()

    train_dir = args.dataset_dir / "train"
    samples = sorted(p for p in train_dir.iterdir() if p.is_dir())[: args.num_samples]
    if not samples:
        raise SystemExit(f"No samples in {train_dir}")

    print(f"Device: {DEVICE}")
    print(f"Samples: {len(samples)}")
    t0 = time.time()

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    rife = RifeWrapper()

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    fails = 0

    for i, sample_dir in enumerate(samples):
        try:
            scores = run_sample(sample_dir, dis, rife, args.skip_phase)
        except Exception as e:
            fails += 1
            print(f"[{i + 1}] FAIL {sample_dir.name}: {e}")
            continue
        for name, val in scores.items():
            sums[name] = sums.get(name, 0.0) + val
            counts[name] = counts.get(name, 0) + 1
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[{i + 1}/{len(samples)}] {sample_dir.name}  ensemble={scores.get('ensemble', 0):.2f}  adaptive_soft={scores.get('adaptive_soft', 0):.2f}")

    elapsed = time.time() - t0
    avgs = {k: sums[k] / counts[k] for k in sorted(sums, key=lambda x: -sums[x] / counts[x])}

    official = {"t0", "t1", "mean", "dis_official", "rife", "ensemble"}
    yours = {
        "global_blur", "farneback_half", "farneback_aggr", "dis_half",
        "adaptive_soft", "adaptive_aggr", "phase_blocks",
    }

    print("\n" + "=" * 62)
    print(f"Average PSNR on {counts[next(iter(counts))]} samples  ({elapsed:.0f}s, fails={fails})")
    print("=" * 62)
    print("\n--- Official baselines ---")
    for name in avgs:
        if name in official:
            print(f"  {name:18s} {avgs[name]:.2f} dB")
    print("\n--- Your baselines (testdeltadif / soft) ---")
    for name in avgs:
        if name in yours:
            print(f"  {name:18s} {avgs[name]:.2f} dB")

    best = max(avgs, key=avgs.get)
    print(f"\nBest overall: {best} = {avgs[best]:.2f} dB")

    out = ROOT / "compare_baselines_results.json"
    out.write_text(json.dumps({"averages_db": avgs, "n_samples": counts[best], "fails": fails}, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
