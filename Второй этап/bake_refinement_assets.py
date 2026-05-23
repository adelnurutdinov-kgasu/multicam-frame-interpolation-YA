"""
Запекаем только LiDAR depth (.npy) для rife_depth_refinement.

  <out_root>/train/<sample_id>/
    d0.npy  — depth @ t0, target_camera
    d1.npy  — depth @ target
    d2.npy  — depth @ t1, target_camera
    meta.json

RGB / RIFE читаются при обучении из dataset_root + rife_root (без копирования).
Preview PNG: python render_baked_depth.py --sample-id ...
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lidar_depth_map import intrinsics_to_K, project_to_camera, rasterize_depth  # noqa: E402


def project_depth(xyz: np.ndarray, c2w: np.ndarray, K: np.ndarray, h: int, w: int) -> np.ndarray:
    u, v, z, _ = project_to_camera(xyz, c2w, K, w, h)
    depth_map, _ = rasterize_depth(u, v, z, h, w, splat_radius=1)
    valid = np.isfinite(depth_map) & (depth_map > 0) & (depth_map < np.inf)
    out = np.full((h, w), np.nan, dtype=np.float32)
    out[valid] = depth_map[valid].astype(np.float32)
    return out


def bake_sample_depth(sample_dir: Path, out_dir: Path, skip_existing: bool = True) -> dict:
    need = ["d0.npy", "d1.npy", "d2.npy", "meta.json"]
    if skip_existing and out_dir.is_dir() and all((out_dir / n).is_file() for n in need):
        return {"sample_id": sample_dir.name, "skipped": True}

    meta_path = sample_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cam = meta["target_camera"]
    sid = meta.get("sample_id", sample_dir.name)
    intr = meta["intrinsics"][cam]
    h, w = int(intr["height"]), int(intr["width"])
    K = intrinsics_to_K(intr)

    xyz = np.load(sample_dir / "input" / "lidar.npz")["xyz"].astype(np.float64)

    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, ts in (("d0", "t0"), ("d1", "target"), ("d2", "t1")):
        c2w = np.array(meta["poses_c2w"][ts][cam], dtype=np.float64)
        np.save(out_dir / f"{stem}.npy", project_depth(xyz, c2w, K, h, w))

    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "sample_id": sid,
                "camera": cam,
                "source_dir": str(sample_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"sample_id": sid, "skipped": False}


def _worker(args: tuple[Path, Path, bool]) -> tuple[str, bool, str | None]:
    sample_dir, out_dir, skip_existing = args
    try:
        rec = bake_sample_depth(sample_dir, out_dir, skip_existing)
        return sample_dir.name, rec.get("skipped", False), None
    except Exception as e:
        return sample_dir.name, False, str(e)


def main() -> int:
    p = argparse.ArgumentParser(description="Bake LiDAR depth npy only (fast)")
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(r"C:\Users\adel\Downloads\cv_dataset\final_dataset_v5_participants"),
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path(r"C:\Users\adel\Downloads\cv_dataset\rife_refinement_baked"),
    )
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-skip-existing", action="store_true")
    args = p.parse_args()

    split_dir = args.dataset_dir / args.split
    out_split = args.out_root / args.split
    out_split.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[Path, Path, bool]] = []
    for sd in sorted(split_dir.iterdir()):
        if not sd.is_dir():
            continue
        if not (sd / "meta.json").is_file() or not (sd / "input" / "lidar.npz").is_file():
            continue
        meta = json.loads((sd / "meta.json").read_text(encoding="utf-8"))
        sid = meta.get("sample_id", sd.name)
        jobs.append((sd, out_split / sid, not args.no_skip_existing))

    if args.limit > 0:
        jobs = jobs[: args.limit]

    skip_flag = not args.no_skip_existing
    print(f"depth bake: {len(jobs)} samples -> {out_split}  workers={args.workers}")

    ok, skip, fail = 0, 0, 0
    if args.workers <= 1:
        for i, (sd, out_dir, sk) in enumerate(jobs, 1):
            name, was_skip, err = _worker((sd, out_dir, sk))
            if err:
                fail += 1
                print(f"  FAIL {name}: {err}")
            elif was_skip:
                skip += 1
            else:
                ok += 1
            if i % 100 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] ok={ok} skip={skip} fail={fail}")
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_worker, job) for job in jobs]
            for fut in as_completed(futs):
                done += 1
                name, was_skip, err = fut.result()
                if err:
                    fail += 1
                    if fail <= 5:
                        print(f"  FAIL {name}: {err}")
                elif was_skip:
                    skip += 1
                else:
                    ok += 1
                if done % 100 == 0 or done == len(jobs):
                    print(f"  [{done}/{len(jobs)}] ok={ok} skip={skip} fail={fail}")

    n_dirs = sum(1 for d in out_split.iterdir() if d.is_dir() and (d / "d0.npy").is_file())
    manifest = {
        "split": args.split,
        "n_samples": n_dirs,
        "format": "depth_npy_only",
        "dataset_dir": str(args.dataset_dir),
    }
    (out_split.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    print(f"done ok={ok} skip={skip} fail={fail}  with_depth={n_dirs}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
