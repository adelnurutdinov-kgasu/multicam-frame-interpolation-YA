"""Прогон RIFE (baseline) по t0+t1 → target для всех сэмплов с GT."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_baselines import RifeWrapper, load_rgb, psnr  # noqa: E402


def iter_samples(split_dir: Path, *, require_target: bool = True):
    for sd in sorted(split_dir.iterdir()):
        if not sd.is_dir():
            continue
        meta_path = sd / "meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cam = meta["target_camera"]
        t0 = sd / "input" / "t0" / f"{cam}.jpg"
        t1 = sd / "input" / "t1" / f"{cam}.jpg"
        gt = sd / "target" / f"{cam}.jpg"
        if not (t0.is_file() and t1.is_file()):
            continue
        if require_target and not gt.is_file():
            continue
        yield sd, meta, cam


def main() -> int:
    p = argparse.ArgumentParser(description="Batch RIFE predictions for samples with GT target")
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(r"C:\Users\adel\Downloads\cv_dataset\final_dataset_v5_participants"),
    )
    p.add_argument("--split", choices=["train", "test", "both"], default="train")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(r"C:\Users\adel\Downloads\cv_dataset\rife_predictions_v5"),
    )
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    p.add_argument(
        "--allow-no-target",
        action="store_true",
        help="Сэмплы без target/<cam>.jpg (inference-only, без PSNR в scores)",
    )
    args = p.parse_args()

    splits = ["train", "test"] if args.split == "both" else [args.split]
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[Path, dict, str, Path]] = []
    for split in splits:
        split_dir = args.dataset_dir / split
        if not split_dir.is_dir():
            print(f"skip missing split: {split_dir}")
            continue
        split_out = out_dir / split
        split_out.mkdir(parents=True, exist_ok=True)
        for sd, meta, cam in iter_samples(split_dir, require_target=not args.allow_no_target):
            sid = meta.get("sample_id", sd.name)
            out_path = split_out / f"{sid}.jpg"
            if args.skip_existing and out_path.is_file():
                continue
            jobs.append((sd, meta, cam, out_path))

    if args.limit > 0:
        jobs = jobs[: args.limit]

    print(f"RIFE jobs: {len(jobs)}  out: {out_dir}")
    if not jobs:
        print("nothing to do (all exist or no samples with target)")
        return 0

    rife = RifeWrapper()
    scores_path = out_dir / "scores.jsonl"
    t0_run = time.time()
    ok, fail = 0, 0

    with scores_path.open("a", encoding="utf-8") as sf:
        for i, (sd, meta, cam, out_path) in enumerate(jobs, start=1):
            try:
                img0 = load_rgb(sd / "input" / "t0" / f"{cam}.jpg")
                img1 = load_rgb(sd / "input" / "t1" / f"{cam}.jpg")
                pred = rife.infer(img0, img1)
                Image.fromarray(pred).save(out_path, quality=92)
                gt_path = sd / "target" / f"{cam}.jpg"
                rec = {
                    "sample_id": meta.get("sample_id", sd.name),
                    "split": out_path.parent.name,
                    "camera": cam,
                    "path": str(out_path),
                }
                if gt_path.is_file():
                    score = psnr(pred, load_rgb(gt_path))
                    rec["psnr"] = round(score, 4)
                if i % 20 == 0 or i == len(jobs):
                    elapsed = time.time() - t0_run
                    extra = f"  last_psnr={rec['psnr']:.2f}" if "psnr" in rec else ""
                    print(f"  [{i}/{len(jobs)}] ok={ok} fail={fail}{extra}  {elapsed:.0f}s")
                sf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sf.flush()
                ok += 1
            except Exception as e:
                fail += 1
                print(f"  FAIL {sd.name}: {e}")

    print(f"done: ok={ok} fail={fail}  scores -> {scores_path}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
