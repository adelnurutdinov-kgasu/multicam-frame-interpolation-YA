"""Оценка реконструкционной ошибки цветовых преобразований (PSNR/MAE).

Тестирует схему:
    x -> T(x) -> T^-1(x)
на подвыборке train/test и сравнивает, какие представления теряют меньше всего.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

import sys as _sys
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))

from ya_paths import DATASET_ROOT


def psnr_rgb(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    a = a_u8.astype(np.float32) / 255.0
    b = b_u8.astype(np.float32) / 255.0
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def mae_rgb(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    return float(np.mean(np.abs(a_u8.astype(np.float32) - b_u8.astype(np.float32))))


@dataclass
class TransformSpec:
    name: str
    fwd: Callable[[np.ndarray], np.ndarray]
    inv: Callable[[np.ndarray], np.ndarray]


def _clip_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def tf_identity() -> TransformSpec:
    return TransformSpec(
        name="rgb_identity",
        fwd=lambda x: x.copy(),
        inv=lambda z: z.copy(),
    )


def tf_ycrcb_cv2() -> TransformSpec:
    return TransformSpec(
        name="ycrcb_cv2",
        fwd=lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2YCrCb),
        inv=lambda z: cv2.cvtColor(z, cv2.COLOR_YCrCb2RGB),
    )


def tf_hsv_cv2() -> TransformSpec:
    return TransformSpec(
        name="hsv_cv2",
        fwd=lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2HSV),
        inv=lambda z: cv2.cvtColor(z, cv2.COLOR_HSV2RGB),
    )


def tf_lab_cv2() -> TransformSpec:
    return TransformSpec(
        name="lab_cv2",
        fwd=lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2LAB),
        inv=lambda z: cv2.cvtColor(z, cv2.COLOR_LAB2RGB),
    )


def tf_gray3() -> TransformSpec:
    def _fwd(x: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        return np.repeat(g[..., None], 3, axis=2)

    return TransformSpec(name="gray_3ch", fwd=_fwd, inv=lambda z: z.copy())


def tf_pca_rgb() -> TransformSpec:
    """Линейное decorrelation-преобразование RGB (почти без потерь при float)."""

    def _fwd(x: np.ndarray) -> np.ndarray:
        xf = x.astype(np.float32) / 255.0
        sh = xf.shape
        v = xf.reshape(-1, 3)
        mean = v.mean(axis=0, keepdims=True)
        vc = v - mean
        cov = (vc.T @ vc) / max(1, vc.shape[0] - 1)
        w, u = np.linalg.eigh(cov)
        # Сохраняем mean/u в первых пикселях служебно (для честного T^-1 в одном тензоре).
        z = vc @ u
        out = z.reshape(sh)
        # масштаб в диапазон примерно 0..1 для симметрии сравнения
        out = np.clip(out * 0.5 + 0.5, 0.0, 1.0)
        meta = np.concatenate([mean.flatten(), u.flatten()]).astype(np.float32)
        meta_pad = np.zeros((2, 3), dtype=np.float32)
        meta_pad.reshape(-1)[: meta.size] = meta
        out[:2, :1, :] = meta_pad.reshape(2, 1, 3)
        return _clip_u8(out * 255.0)

    def _inv(z_u8: np.ndarray) -> np.ndarray:
        zf = z_u8.astype(np.float32) / 255.0
        sh = zf.shape
        meta = zf[:2, :1, :].reshape(-1)
        mean = meta[:3][None, :]
        u = meta[3:12].reshape(3, 3)
        z = (zf.reshape(-1, 3) - 0.5) / 0.5
        v = z @ u.T + mean
        out = np.clip(v.reshape(sh), 0.0, 1.0)
        return _clip_u8(out * 255.0)

    return TransformSpec(name="pca_rgb_lossy_u8", fwd=_fwd, inv=_inv)


def all_transforms() -> list[TransformSpec]:
    return [
        tf_identity(),
        tf_ycrcb_cv2(),
        tf_hsv_cv2(),
        tf_lab_cv2(),
        tf_gray3(),
    ]


def discover_images(split_dir: Path) -> list[Path]:
    out: list[Path] = []
    for sample_dir in sorted(split_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        meta_path = sample_dir / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cam = meta.get("target_camera")
            if not cam:
                continue
            img = sample_dir / "target" / f"{cam}.jpg"
            if img.is_file():
                out.append(img)
        except Exception:
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate color-space reconstruction fidelity via PSNR.")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--limit", type=int, default=200, help="0 = all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", type=Path, default=Path("artifacts/color_recon_eval.json"))
    args = ap.parse_args()

    split_dir = DATASET_ROOT / args.split
    paths = discover_images(split_dir)
    if not paths:
        print(f"Нет изображений в {split_dir}")
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(paths)
    if args.limit > 0:
        paths = paths[: args.limit]

    tfs = all_transforms()
    scores: dict[str, list[float]] = {t.name: [] for t in tfs}
    maes: dict[str, list[float]] = {t.name: [] for t in tfs}

    for i, p in enumerate(paths, 1):
        x = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
        for t in tfs:
            z = t.fwd(x)
            xr = t.inv(z)
            scores[t.name].append(psnr_rgb(x, xr))
            maes[t.name].append(mae_rgb(x, xr))
        if i % 50 == 0 or i == len(paths):
            print(f"[{i}/{len(paths)}] processed")

    summary = []
    for t in tfs:
        s = np.array(scores[t.name], dtype=np.float32)
        m = np.array(maes[t.name], dtype=np.float32)
        summary.append(
            {
                "transform": t.name,
                "psnr_mean": float(s.mean()),
                "psnr_median": float(np.median(s)),
                "psnr_p10": float(np.percentile(s, 10)),
                "mae_mean": float(m.mean()),
            }
        )
    summary.sort(key=lambda r: r["psnr_mean"], reverse=True)

    print("\n=== Reconstruction ranking (higher PSNR is better) ===")
    for r in summary:
        print(
            f"{r['transform']:22s}  PSNR mean={r['psnr_mean']:.3f}  "
            f"median={r['psnr_median']:.3f}  p10={r['psnr_p10']:.3f}  MAE={r['mae_mean']:.3f}"
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "split": args.split,
                "num_images": len(paths),
                "seed": args.seed,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved: {args.out_json.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

