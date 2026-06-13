"""Submission: mean(V3 YCrCb→RGB, V4 RGB) на test."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lib.consensus_kit import ConsensusUNet  # noqa: E402
from ya_paths import CV_ROOT, TEST_SPLIT  # noqa: E402

MIRROR_CAMERAS = ("left_fwd", "right_bwd")
V3_CKPT = REPO / "artifacts" / "checkpoints" / "consensus_v3_scratch" / "consensus_v3_best.pt"
V4_CKPT = REPO / "artifacts" / "checkpoints" / "consensus_v4_scratch" / "consensus_v4_best.pt"


def _load_rgb(path: Path, hw: tuple[int, int]) -> np.ndarray:
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    if arr.shape[:2] != hw:
        arr = cv2.resize(arr, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    return arr


def _rgb01(rgb_u8: np.ndarray) -> np.ndarray:
    return rgb_u8.astype(np.float32) / 255.0


def _to_ycc01(rgb_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255.0


def _ycc01_to_rgb01(ycc01: np.ndarray) -> np.ndarray:
    u8 = np.clip(ycc01 * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_YCrCb2RGB).astype(np.float32) / 255.0


def _flip_h(*arrays):
    return tuple(a[:, ::-1].copy() for a in arrays)


def _native_hw(sample_meta_path: Path) -> tuple[int, int]:
    m = json.loads(sample_meta_path.read_text(encoding="utf-8"))
    cam = m.get("target_camera") or m.get("camera")
    intr = m["intrinsics"][cam]
    return int(intr["height"]), int(intr["width"])


def _discover_test_samples(test_root: Path) -> list[Path]:
    return [d for d in sorted(test_root.iterdir()) if d.is_dir() and (d / "meta.json").is_file()]


def _load_model(ckpt: Path, device: torch.device) -> ConsensusUNet:
    st = torch.load(ckpt, map_location=device, weights_only=False)
    base_ch = int(st.get("cfg", {}).get("base_channels", 32))
    model = ConsensusUNet(in_ch=7, base=base_ch, predict_confidence=False).to(device)
    model.load_state_dict(st["model"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description="Export submission: blend V3 (YCrCb) + V4 (RGB)")
    ap.add_argument("--v3-ckpt", type=Path, default=V3_CKPT)
    ap.add_argument("--v4-ckpt", type=Path, default=V4_CKPT)
    ap.add_argument("--alpha-v4", type=float, default=0.5, help="out = alpha*v4 + (1-alpha)*v3_rgb")
    ap.add_argument("--test-root", type=Path, default=TEST_SPLIT)
    ap.add_argument("--warp-mix-root", type=Path, default=CV_ROOT / "side_warp_mix_v1" / "test")
    ap.add_argument("--trust-root", type=Path, default=CV_ROOT / "precomputed_lidar_trust" / "test")
    ap.add_argument("--out-root", type=Path, default=CV_ROOT / "submission_v3_v4_mean")
    ap.add_argument("--image-h", type=int, default=544)
    ap.add_argument("--image-w", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    for label, ckpt in (("v3", args.v3_ckpt), ("v4", args.v4_ckpt)):
        if not ckpt.is_file():
            print(f"{label} checkpoint not found: {ckpt}")
            return 1

    test_samples = _discover_test_samples(args.test_root)
    if not test_samples:
        print(f"no test samples in {args.test_root}")
        return 1

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    alpha = float(np.clip(args.alpha_v4, 0.0, 1.0))
    print(
        f"device={device} samples={len(test_samples)} "
        f"blend={alpha:.2f}*v4 + {1-alpha:.2f}*v3_rgb",
        flush=True,
    )

    model_v3 = _load_model(args.v3_ckpt, device)
    model_v4 = _load_model(args.v4_ckpt, device)
    hw = (args.image_h, args.image_w)
    ok_n, fail_n = 0, 0

    for i in range(0, len(test_samples), args.batch_size):
        batch_dirs = test_samples[i : i + args.batch_size]
        xs_v3, bases_v3, xs_v4, bases_v4, masks, metas = [], [], [], [], [], []
        for sd in batch_dirs:
            try:
                meta = json.loads((sd / "meta.json").read_text(encoding="utf-8"))
                sid = sd.name
                cam = meta["target_camera"]

                warp_path = args.warp_mix_root / sid / "warp_mix.jpg"
                trust_path = args.trust_root / sid / "lidar_trust.npy"
                t0_path = sd / "input" / "t0" / f"{cam}.jpg"
                t1_path = sd / "input" / "t1" / f"{cam}.jpg"

                warp_rgb_u8 = _load_rgb(warp_path, hw)
                t0 = _load_rgb(t0_path, hw)
                t1 = _load_rgb(t1_path, hw)
                mean_rgb_u8 = ((t0.astype(np.float32) + t1.astype(np.float32)) * 0.5).astype(np.uint8)

                trust = np.load(trust_path).astype(np.float32)
                if trust.shape != hw:
                    trust = cv2.resize(trust, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
                trust = np.clip(trust, 0.0, 1.0)

                warp_ycc = _to_ycc01(warp_rgb_u8)
                mean_ycc = _to_ycc01(mean_rgb_u8)
                warp_rgb = _rgb01(warp_rgb_u8)
                mean_rgb = _rgb01(mean_rgb_u8)

                mirrored = cam in MIRROR_CAMERAS
                if mirrored:
                    warp_ycc, mean_ycc, trust_v3 = _flip_h(warp_ycc, mean_ycc, trust)
                    warp_rgb, mean_rgb, trust_v4 = _flip_h(warp_rgb, mean_rgb, trust)
                else:
                    trust_v3 = trust_v4 = trust

                x_v3 = np.concatenate([warp_ycc, trust_v3[..., None], mean_ycc], axis=-1)
                x_v4 = np.concatenate([warp_rgb, trust_v4[..., None], mean_rgb], axis=-1)
                to_chw = lambda a: torch.from_numpy(a.transpose(2, 0, 1)).contiguous().float()
                xs_v3.append(to_chw(x_v3))
                bases_v3.append(to_chw(warp_ycc))
                xs_v4.append(to_chw(x_v4))
                bases_v4.append(to_chw(warp_rgb))
                masks.append(torch.ones(1, hw[0], hw[1], dtype=torch.float32))
                metas.append((sd, sid, mirrored))
            except Exception as e:
                fail_n += 1
                print(f"FAIL prepare {sd.name}: {e}")

        if not xs_v3:
            continue

        inp_v3 = torch.stack(xs_v3, dim=0).to(device)
        base_v3 = torch.stack(bases_v3, dim=0).to(device)
        inp_v4 = torch.stack(xs_v4, dim=0).to(device)
        base_v4 = torch.stack(bases_v4, dim=0).to(device)
        msk = torch.stack(masks, dim=0).to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred_v3 = model_v3(inp_v3, msk, base_v3)["pred"].float().clamp(0, 1)
            pred_v4 = model_v4(inp_v4, msk, base_v4)["pred"].float().clamp(0, 1)

        for bi, (sd, sid, mirrored) in enumerate(metas):
            try:
                ycc = pred_v3[bi].permute(1, 2, 0).detach().cpu().numpy()
                rgb_v4 = pred_v4[bi].permute(1, 2, 0).detach().cpu().numpy()
                if mirrored:
                    ycc = ycc[:, ::-1].copy()
                    rgb_v4 = rgb_v4[:, ::-1].copy()
                rgb_v3 = _ycc01_to_rgb01(ycc)
                rgb01 = np.clip(alpha * rgb_v4 + (1.0 - alpha) * rgb_v3, 0.0, 1.0)

                native_h, native_w = _native_hw(sd / "meta.json")
                rgb_u8 = (rgb01 * 255).astype(np.uint8)
                if rgb_u8.shape[:2] != (native_h, native_w):
                    rgb_u8 = cv2.resize(rgb_u8, (native_w, native_h), interpolation=cv2.INTER_LINEAR)

                out_dir = args.out_root / sid
                out_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb_u8).save(out_dir / "pred.jpg", quality=95)
                ok_n += 1
            except Exception as e:
                fail_n += 1
                print(f"FAIL write {sid}: {e}")

        if (i + len(batch_dirs)) % 20 == 0 or (i + len(batch_dirs)) == len(test_samples):
            print(f"[{i + len(batch_dirs)}/{len(test_samples)}] ok={ok_n} fail={fail_n}", flush=True)

    print(f"done -> {args.out_root}  ok={ok_n} fail={fail_n}", flush=True)
    return 0 if fail_n == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
