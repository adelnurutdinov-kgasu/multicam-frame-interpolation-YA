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

from consensus_kit import ConsensusUNet  # noqa: E402
from ya_paths import CV_ROOT, TEST_SPLIT  # noqa: E402

MIRROR_CAMERAS = ("left_fwd", "right_bwd")
DEFAULT_CKPT = REPO / "notebooks" / "training" / "artifacts" / "checkpoints" / "consensus_v2_all" / "consensus_v2_all_best.pt"


def _load_rgb(path: Path, hw: tuple[int, int]) -> np.ndarray:
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    if arr.shape[:2] != hw:
        arr = cv2.resize(arr, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    return arr


def _to_ycc01(rgb_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255.0


def _to_rgb01_from_ycc(ycc01: np.ndarray) -> np.ndarray:
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


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description="Export submission with consensus_v2_all model")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--test-root", type=Path, default=TEST_SPLIT)
    ap.add_argument("--warp-mix-root", type=Path, default=CV_ROOT / "side_warp_mix_v1" / "test")
    ap.add_argument("--trust-root", type=Path, default=CV_ROOT / "precomputed_lidar_trust" / "test")
    ap.add_argument("--rife-root", type=Path, default=CV_ROOT / "rife_predictions_v5" / "test")
    ap.add_argument("--out-root", type=Path, default=CV_ROOT / "submission_v2_all")
    ap.add_argument("--image-h", type=int, default=544)
    ap.add_argument("--image-w", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--blend-alpha",
        type=float,
        default=-1.0,
        help=">=0: blend with RIFE: out = a*pred + (1-a)*rife",
    )
    args = ap.parse_args()

    if not args.ckpt.is_file():
        print(f"checkpoint not found: {args.ckpt}")
        return 1

    test_samples = _discover_test_samples(args.test_root)
    if not test_samples:
        print(f"no test samples in {args.test_root}")
        return 1

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"device={device} samples={len(test_samples)} ckpt={args.ckpt}")

    st = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg_saved = st.get("cfg", {})
    base_ch = int(cfg_saved.get("base_channels", 32))
    model = ConsensusUNet(in_ch=7, base=base_ch, predict_confidence=False).to(device)
    model.load_state_dict(st["model"], strict=True)
    model.eval()

    hw = (args.image_h, args.image_w)
    ok_n, fail_n = 0, 0

    for i in range(0, len(test_samples), args.batch_size):
        batch_dirs = test_samples[i : i + args.batch_size]
        xs, bases, masks, metas = [], [], [], []
        for sd in batch_dirs:
            try:
                meta = json.loads((sd / "meta.json").read_text(encoding="utf-8"))
                sid = sd.name
                cam = meta["target_camera"]

                warp_path = args.warp_mix_root / sid / "warp_mix.jpg"
                trust_path = args.trust_root / sid / "lidar_trust.npy"
                t0_path = sd / "input" / "t0" / f"{cam}.jpg"
                t1_path = sd / "input" / "t1" / f"{cam}.jpg"

                warp_rgb = _load_rgb(warp_path, hw)
                t0 = _load_rgb(t0_path, hw)
                t1 = _load_rgb(t1_path, hw)
                mean_rgb = ((t0.astype(np.float32) + t1.astype(np.float32)) * 0.5).astype(np.uint8)

                trust = np.load(trust_path).astype(np.float32)
                if trust.shape != hw:
                    trust = cv2.resize(trust, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
                trust = np.clip(trust, 0.0, 1.0)

                warp_ycc = _to_ycc01(warp_rgb)
                mean_ycc = _to_ycc01(mean_rgb)

                mirrored = cam in MIRROR_CAMERAS
                if mirrored:
                    warp_ycc, mean_ycc, trust = _flip_h(warp_ycc, mean_ycc, trust)

                x = np.concatenate([warp_ycc, trust[..., None], mean_ycc], axis=-1)
                xs.append(torch.from_numpy(x.transpose(2, 0, 1)).contiguous().float())
                bases.append(torch.from_numpy(warp_ycc.transpose(2, 0, 1)).contiguous().float())
                masks.append(torch.ones(1, hw[0], hw[1], dtype=torch.float32))
                metas.append((sd, sid, cam, mirrored))
            except Exception as e:
                fail_n += 1
                print(f"FAIL prepare {sd.name}: {e}")

        if not xs:
            continue

        inp = torch.stack(xs, dim=0).to(device)
        base = torch.stack(bases, dim=0).to(device)
        msk = torch.stack(masks, dim=0).to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = model(inp, msk, base)["pred"].float().clamp(0, 1)

        for bi, (sd, sid, cam, mirrored) in enumerate(metas):
            try:
                ycc = pred[bi].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
                if mirrored:
                    ycc = ycc[:, ::-1].copy()
                rgb01 = _to_rgb01_from_ycc(ycc)

                if args.blend_alpha >= 0.0:
                    rife_path = args.rife_root / f"{sid}.jpg"
                    rife = _load_rgb(rife_path, hw).astype(np.float32) / 255.0
                    rgb01 = np.clip(float(args.blend_alpha) * rgb01 + (1.0 - float(args.blend_alpha)) * rife, 0.0, 1.0)

                native_h, native_w = _native_hw(sd / "meta.json")
                rgb_u8 = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
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
            print(f"[{i + len(batch_dirs)}/{len(test_samples)}] ok={ok_n} fail={fail_n}")

    print(f"done -> {args.out_root}  ok={ok_n} fail={fail_n}")
    return 0 if fail_n == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
