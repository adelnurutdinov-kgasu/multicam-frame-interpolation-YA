"""Горизонтальный flip LiDAR-масок для left_fwd / right_bwd (быстро, без reblend)."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
import sys
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


import numpy as np
from PIL import Image

OUT = Path(r"C:\Users\adel\Downloads\cv_dataset\consensus_test_outputs")
MIRROR = {"left_fwd", "right_bwd"}
FILES = ("lidar_trust.png", "lidar_blend_mask.png", "lidar_density_fine.png")


def flip_png(path: Path) -> None:
    img = Image.open(path)
    Image.fromarray(np.array(img)[:, ::-1]).save(path)


def main() -> None:
    n = 0
    for d in sorted(OUT.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "meta_infer.json"
        if not meta_path.is_file():
            continue
        cam = json.loads(meta_path.read_text(encoding="utf-8")).get("camera")
        if cam not in MIRROR:
            continue
        ver = json.loads(meta_path.read_text(encoding="utf-8")).get("output_version", 0)
        if ver >= 6:
            continue
        for fn in FILES:
            p = d / fn
            if p.is_file():
                flip_png(p)
        n += 1
    print(f"flipped {n} mirror samples  ({', '.join(FILES)})")


if __name__ == "__main__":
    main()
