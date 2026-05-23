"""Проверка train_log/flownet.pkl и smoke-infer RIFE."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent
TRAIN_LOG = ROOT / "train_log"
sys.path.insert(0, str(ROOT / "ECCV2022-RIFE"))

from train_log.RIFE_HDv3 import Model  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def list_train_log() -> None:
    print(f"train_log: {TRAIN_LOG}")
    for p in sorted(TRAIN_LOG.iterdir()):
        if p.is_file():
            print(f"  {p.name:20s}  {p.stat().st_size / 1e6:.2f} MB")
        else:
            print(f"  {p.name}/")


def infer(model, img0: np.ndarray, img1: np.ndarray) -> np.ndarray:
    h, w = img0.shape[:2]
    ph = ((h - 1) // 64 + 1) * 64
    pw = ((w - 1) // 64 + 1) * 64

    def to_t(x):
        return torch.from_numpy(x).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model.inference(
            F.pad(to_t(img0), (0, pw - w, 0, ph - h)),
            F.pad(to_t(img1), (0, pw - w, 0, ph - h)),
        )
    return (out[0, :, :h, :w].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def main() -> int:
    list_train_log()
    pkl = TRAIN_LOG / "flownet.pkl"
    if not pkl.is_file():
        print(f"ERROR: missing {pkl}")
        return 1

    model = Model()
    model.load_model(str(TRAIN_LOG), -1)
    model.eval()
    print(f"model loaded OK  device={DEVICE}")

    sample = Path(
        r"C:\Users\adel\Downloads\cv_dataset\final_dataset_v5_participants\train"
        r"\2026-03-16_15_59_49_18_43_54_lerita_1773678938500161000__000"
    )
    meta = json.loads((sample / "meta.json").read_text(encoding="utf-8"))
    cam = meta["target_camera"]
    img0 = np.array(Image.open(sample / "input" / "t0" / f"{cam}.jpg").convert("RGB"))
    img1 = np.array(Image.open(sample / "input" / "t1" / f"{cam}.jpg").convert("RGB"))
    pred = infer(model, img0, img1)
    out = ROOT / "_verify_rife_smoke.jpg"
    Image.fromarray(pred).save(out, quality=92)
    print(f"smoke infer OK  shape={pred.shape}  saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
