"""Собрать submission из blend_blur50.jpg → submission/<sample_id>/pred.jpg

Размер pred.jpg должен совпадать с native resolution камеры из test meta.json
(intrinsics[target_camera].width/height), а не с фиксированным 1024×544 сети.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
import sys
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


import cv2

DEFAULT_SRC = Path(r"C:\Users\adel\Downloads\cv_dataset\consensus_test_outputs")
DEFAULT_DST = Path(r"C:\Users\adel\Downloads\cv_dataset\submission")
DEFAULT_TEST = Path(r"C:\Users\adel\Downloads\cv_dataset\final_dataset_v5_participants\test")
BLEND_NAME = "blend_blur50.jpg"


def native_hw(meta_path: Path) -> tuple[int, int]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cam = meta.get("target_camera") or meta.get("camera")
    if not cam:
        raise KeyError(f"no target_camera in {meta_path}")
    intr = meta["intrinsics"][cam]
    return int(intr["height"]), int(intr["width"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--dst-root", type=Path, default=DEFAULT_DST)
    ap.add_argument("--test-root", type=Path, default=DEFAULT_TEST)
    ap.add_argument("--blend-file", default=BLEND_NAME)
    args = ap.parse_args()

    ok, fail, resized = 0, 0, 0
    for d in sorted(args.src_root.iterdir()):
        if not d.is_dir():
            continue
        src = d / args.blend_file
        if not src.is_file():
            fail += 1
            continue
        meta_path = args.test_root / d.name / "meta.json"
        if not meta_path.is_file():
            print(f"WARN no meta: {meta_path}")
            fail += 1
            continue

        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            print(f"WARN unreadable: {src}")
            fail += 1
            continue

        native_h, native_w = native_hw(meta_path)
        if img.shape[:2] != (native_h, native_w):
            img = cv2.resize(img, (native_w, native_h), interpolation=cv2.INTER_LINEAR)
            resized += 1

        out_dir = args.dst_root / d.name
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "pred.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        ok += 1

    print(f"-> {args.dst_root}")
    print(f"ok={ok} fail={fail} resized={resized}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
