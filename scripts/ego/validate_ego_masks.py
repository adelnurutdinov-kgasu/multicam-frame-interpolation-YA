"""
Validate approved ego masks on real target frames.

Checks:
  1. overlay on mean + N diverse samples (visual)
  2. boundary edge stability across scenes (edge_freq_mean)
  3. coverage sanity (empty / too large)

Outputs:
  validation_v2/{vehicle}_{camera}_align.png
  validation_v2/report.json
  validation_v2/index.html  — open in browser to review all groups

Usage:
  python validate_ego_masks.py
  python validate_ego_masks.py --max-scenes 6 --edge-freq-thr 0.35
"""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
import sys
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from import_manual_ego_masks import (
    APPROVED_DIR,
    MEANS_DIR,
    ROOT,
    alignment_score,
    collect_sample_paths,
    load_index,
)
from test_ego_consistent_edges import DEFAULT_DATASET

VIZ_DIR = ROOT / "validation_v2"
REPORT_PATH = VIZ_DIR / "report.json"
INDEX_PATH = VIZ_DIR / "index.html"


def load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 127


def save_overlay_grid(
    vehicle: str,
    camera: str,
    mask: np.ndarray,
    mean_rgb: np.ndarray | None,
    sample_paths: list[Path],
    out_path: Path,
    title_suffix: str = "",
) -> None:
    n_samples = min(4, len(sample_paths))
    n_cols = n_samples + (1 if mean_rgb is not None else 0)
    fig, axes = plt.subplots(1, max(1, n_cols), figsize=(3.2 * max(1, n_cols), 3.2))
    if n_cols == 1:
        axes = [axes]

    red = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    red[..., 0] = 255
    col = 0

    if mean_rgb is not None:
        if mean_rgb.shape[:2] != mask.shape[:2]:
            mean_rgb = cv2.resize(mean_rgb, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
        ov = mean_rgb.copy()
        ov[mask] = (0.5 * red[mask] + 0.5 * ov[mask]).astype(np.uint8)
        axes[col].imshow(ov)
        axes[col].set_title("mean+mask", fontsize=9)
        axes[col].axis("off")
        col += 1

    for sd in sample_paths[:n_samples]:
        img = np.array(Image.open(sd / "target" / f"{camera}.jpg").convert("RGB"))
        if img.shape[:2] != mask.shape[:2]:
            img = cv2.resize(img, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
        ov = img.copy()
        ov[mask] = (0.5 * red[mask] + 0.5 * ov[mask]).astype(np.uint8)
        axes[col].imshow(ov)
        axes[col].set_title(sd.name.split("_")[-1][:10], fontsize=8)
        axes[col].axis("off")
        col += 1

    for j in range(col, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{vehicle}/{camera}{title_suffix}", fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_html(report: dict, out_path: Path) -> None:
    rows = []
    for item in report["approved"]:
        warn = item.get("edge_align_warn", False)
        cls = "warn" if warn else "ok"
        img = item.get("viz_png", "")
        rel = Path(img).name if img else ""
        rows.append(
            f'<tr class="{cls}">'
            f'<td>{escape(item["vehicle"])}</td>'
            f'<td>{escape(item["camera"])}</td>'
            f'<td>{item["mask_pct"]:.1%}</td>'
            f'<td>{item.get("edge_freq_mean", 0):.3f}</td>'
            f'<td>{item.get("n_scenes", 0)}</td>'
            f'<td>{"⚠" if warn else "✓"}</td>'
            f'<td><a href="{rel}"><img src="{rel}" width="320"></a></td>'
            f"</tr>"
        )

    zero_rows = "".join(
        f"<li>{escape(x['vehicle'])}/{escape(x['camera'])} — {x['status']}</li>"
        for x in report.get("zero_mask", [])
    )
    reject_rows = "".join(
        f"<li>{escape(x['vehicle'])}/{escape(x['camera'])}</li>"
        for x in report.get("reject", [])
    )
    warn_rows = "".join(
        f"<li>{escape(x['vehicle'])}/{escape(x['camera'])} "
        f"edge={x.get('edge_freq_mean', 0):.3f}</li>"
        for x in report.get("warnings", [])
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Ego mask validation</title>
<style>
body {{ font-family: sans-serif; margin: 16px; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border: 1px solid #ccc; padding: 6px; vertical-align: top; }}
tr.ok {{ background: #f8fff8; }}
tr.warn {{ background: #fff8f0; }}
.summary {{ margin-bottom: 16px; }}
</style></head><body>
<h1>Проверка ego-масок</h1>
<div class="summary">
<p>Approved: {report['n_approved']} | warnings: {report['n_warnings']} |
zero-mask: {report['n_zero_mask']} | reject: {report['n_reject']}</p>
<p>edge_freq_mean &lt; {report['edge_freq_thr']} → граница маски нестабильна между сценами</p>
</div>
<h2>⚠ Подозрительные ({report['n_warnings']})</h2>
<ul>{warn_rows or '<li>нет</li>'}</ul>
<h2>Нулевые маски ({report['n_zero_mask']})</h2>
<ul>{zero_rows or '<li>нет</li>'}</ul>
<h2>Reject ({report['n_reject']})</h2>
<ul>{reject_rows or '<li>нет</li>'}</ul>
<h2>Все approved</h2>
<table>
<tr><th>vehicle</th><th>camera</th><th>mask%</th><th>edge_freq</th><th>n_scenes</th><th>ok</th><th>preview</th></tr>
{''.join(rows)}
</table>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--max-scenes", type=int, default=6)
    ap.add_argument("--edge-freq-thr", type=float, default=0.35)
    ap.add_argument("--only-warn", action="store_true", help="viz only flagged groups")
    args = ap.parse_args()

    manifest = json.loads((ROOT / "manual_masks_manifest.json").read_text(encoding="utf-8"))
    by_key = load_index()
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    approved_rows: list[dict] = []
    warnings: list[dict] = []
    zero_mask: list[dict] = []
    reject: list[dict] = []

    for item in manifest["items"]:
        vehicle, camera = item["vehicle"], item["camera"]
        status = item.get("status", "")

        if status == "reject_black":
            reject.append({"vehicle": vehicle, "camera": camera})
            continue

        if status in ("needs_annotation", "bad_mean_few_scenes"):
            zero_mask.append({"vehicle": vehicle, "camera": camera, "status": status})
            continue

        if status != "approved":
            continue

        mask_path = APPROVED_DIR / f"{vehicle}_{camera}.png"
        if not mask_path.is_file():
            continue

        mask = load_mask(mask_path)
        mask_pct = float(mask.mean())
        if mask_pct < 0.001:
            zero_mask.append({"vehicle": vehicle, "camera": camera, "status": "approved_empty"})
            continue

        paths = collect_sample_paths(args.dataset, vehicle, camera, max_scenes=args.max_scenes)
        scores = alignment_score(mask, paths, camera)
        edge_freq = scores["edge_freq_mean"]
        edge_warn = edge_freq < args.edge_freq_thr and scores["n_scenes"] > 0

        mean_path = MEANS_DIR / f"{vehicle}_{camera}.png"
        mean_rgb = np.array(Image.open(mean_path).convert("RGB")) if mean_path.is_file() else None
        if mean_rgb is not None and mean_rgb.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (mean_rgb.shape[1], mean_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        suffix = ""
        if item.get("default_variant"):
            suffix = f"  v{item['default_variant']}"

        viz_path = VIZ_DIR / f"{vehicle}_{camera}_align.png"
        if not args.only_warn or edge_warn:
            save_overlay_grid(vehicle, camera, mask, mean_rgb, paths, viz_path, suffix)

        row = {
            "vehicle": vehicle,
            "camera": camera,
            "mask_pct": mask_pct,
            "edge_freq_mean": edge_freq,
            "edge_freq_min": scores["edge_freq_min"],
            "n_scenes": scores["n_scenes"],
            "edge_align_warn": edge_warn,
            "viz_png": str(viz_path.resolve()),
            "default_variant": item.get("default_variant"),
        }
        approved_rows.append(row)
        if edge_warn:
            warnings.append(row)

    approved_rows.sort(key=lambda x: (x["edge_freq_mean"], x["vehicle"], x["camera"]))
    warnings.sort(key=lambda x: x["edge_freq_mean"])

    report = {
        "dataset": str(args.dataset.resolve()),
        "edge_freq_thr": args.edge_freq_thr,
        "viz_dir": str(VIZ_DIR.resolve()),
        "n_approved": len(approved_rows),
        "n_warnings": len(warnings),
        "n_zero_mask": len(zero_mask),
        "n_reject": len(reject),
        "approved": approved_rows,
        "warnings": warnings,
        "zero_mask": zero_mask,
        "reject": reject,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(report, INDEX_PATH)

    print(f"Approved checked: {report['n_approved']}")
    print(f"  edge warnings (<{args.edge_freq_thr}): {report['n_warnings']}")
    print(f"  zero / no-artifact groups: {report['n_zero_mask']}")
    print(f"  reject: {report['n_reject']}")
    print(f"Report: {REPORT_PATH}")
    print(f"Gallery: {INDEX_PATH}")

    if warnings:
        print("\nTop warnings (low edge stability on mask boundary):")
        for w in warnings[:15]:
            print(f"  {w['vehicle']}/{w['camera']:12} edge={w['edge_freq_mean']:.3f}  mask={w['mask_pct']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
