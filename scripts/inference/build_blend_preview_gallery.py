"""Превью-галерея test outputs: consensus / rife / blend / 3 LiDAR-маски."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


import numpy as np
from PIL import Image



from run_test_consensus_inference import (  # noqa: E402
    CV_ROOT,
    MIRROR_CAMERAS,
    discover_test_baked,
    flip_hwc,
    save_trust_maps,
    test_config,
)
from lidar_density_mask import build_lidar_density, lidar_rife_blend_maps  # noqa: E402

OUT_ROOT = CV_ROOT / "consensus_test_outputs"
GALLERY_HTML = OUT_ROOT / "_preview_gallery.html"
CAMERAS = ("front", "rear", "left_fwd", "right_fwd", "left_bwd", "right_bwd")


def pick_samples_per_camera() -> list[Path]:
    cfg = test_config("test")
    by_cam: dict[str, Path] = {}
    for baked in discover_test_baked(cfg):
        meta = json.loads((baked / "meta.json").read_text(encoding="utf-8"))
        cam = meta["camera"]
        sid = meta["sample_id"]
        out_dir = OUT_ROOT / sid
        if cam in CAMERAS and cam not in by_cam and (out_dir / "consensus_model.jpg").is_file():
            by_cam[cam] = out_dir
    return [by_cam[c] for c in CAMERAS if c in by_cam]


def ensure_lidar_maps(out_dir: Path, baked_dir: Path, cfg) -> None:
    meta = json.loads((baked_dir / "meta.json").read_text(encoding="utf-8"))
    cam = meta["camera"]
    src = Path(meta.get("source_dir", Path(cfg.dataset_root) / meta["sample_id"]))
    hw = (cfg.image_h, cfg.image_w)
    maps = lidar_rife_blend_maps(
        src, cam, hw,
        spread_radius_fine=cfg.lidar_spread_radius_fine,
        spread_blur_fine=cfg.lidar_spread_blur_fine,
        zone_min=cfg.lidar_zone_min,
    )
    save_trust_maps(out_dir, maps, cfg.lidar_zone_min)


def write_html(rows: list[dict]) -> None:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Consensus test preview</title>",
        "<style>",
        "body{font-family:Segoe UI,Arial,sans-serif;margin:20px;background:#111;color:#eee;}",
        "h1{font-size:20px;} .note{color:#aaa;font-size:13px;margin-bottom:20px;}",
        "section{border:1px solid #333;border-radius:8px;padding:12px;margin:16px 0;background:#1a1a1a;}",
        "h2{font-size:15px;margin:0 0 10px;} .grid{display:grid;grid-template-columns:repeat(7,minmax(120px,1fr));gap:8px;}",
        ".cell{text-align:center;} .cell img{width:100%;border-radius:4px;border:1px solid #444;}",
        ".lbl{font-size:11px;color:#bbb;margin:4px 0 6px;}",
        "a{color:#7db3ff;}",
        "</style></head><body>",
        "<h1>Test inference preview</h1>",
        "<p class='note'>Сравнение как в lidar_rife_blend.ipynb: "
        "<b>density_fine</b> (r3+blur), <b>lidar_trust</b>, <b>blend_mask</b> (trust≥0.12), "
        "consensus / RIFE / blend. Папка: <code>consensus_test_outputs</code></p>",
    ]
    for r in rows:
        sid = r["sid"]
        parts.append(f"<section><h2>{r['camera']} — {sid}</h2><div class='grid'>")
        for label, fn in r["images"]:
            rel = f"{sid}/{fn}"
            parts.append(
                f"<div class='cell'><div class='lbl'>{label}</div>"
                f"<a href='{rel}' target='_blank'><img src='{rel}' loading='lazy'></a></div>"
            )
        parts.append("</div></section>")
    parts.append("</body></html>")
    GALLERY_HTML.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    cfg = test_config("test")
    baked_by_sid = {json.loads((b / "meta.json").read_text(encoding="utf-8"))["sample_id"]: b
                    for b in discover_test_baked(cfg)}
    rows = []
    for out_dir in pick_samples_per_camera():
        sid = out_dir.name
        baked = baked_by_sid.get(sid)
        if baked is None:
            continue
        ensure_lidar_maps(out_dir, baked, cfg)
        meta = json.loads((out_dir / "meta_infer.json").read_text(encoding="utf-8"))
        rows.append({
            "sid": sid,
            "camera": meta["camera"],
            "images": [
                ("consensus", "consensus_model.jpg"),
                ("RIFE", "rife.jpg"),
                ("blend", "blend_lidar_rife.jpg"),
                ("r3+blur (fine)", "lidar_density_fine.png"),
                ("lidar_trust", "lidar_trust.png"),
                ("blend mask", "lidar_blend_mask.png"),
                ("ref", "input_ref.jpg"),
            ],
        })
    write_html(rows)
    print(f"Gallery: {GALLERY_HTML}")
    print(f"Samples: {len(rows)}")


if __name__ == "__main__":
    main()
