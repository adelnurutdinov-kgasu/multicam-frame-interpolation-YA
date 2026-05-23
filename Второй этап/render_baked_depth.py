"""PNG-превью из запечённых d0/d1/d2.npy (по желанию, после bake)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def depth_to_png(depth: np.ndarray, out: Path) -> None:
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        Image.fromarray(np.zeros(depth.shape, dtype=np.uint8)).save(out)
        return
    lo, hi = np.percentile(depth[valid], [2, 98])
    norm = np.clip((depth.astype(np.float32) - lo) / max(hi - lo, 1e-3), 0, 1)
    norm[~valid] = 0
    Image.fromarray((norm * 255).astype(np.uint8)).save(out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--depth-root",
        type=Path,
        default=Path(r"C:\Users\adel\Downloads\cv_dataset\rife_refinement_baked\train"),
    )
    p.add_argument("--sample-id", default="")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    dirs = sorted(d for d in args.depth_root.iterdir() if d.is_dir())
    if args.sample_id:
        dirs = [args.depth_root / args.sample_id]
    if args.limit > 0:
        dirs = dirs[: args.limit]

    for d in dirs:
        for stem in ("d0", "d1", "d2"):
            npy = d / f"{stem}.npy"
            if npy.is_file():
                depth_to_png(np.load(npy), d / f"{stem}.png")
        print(d.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
