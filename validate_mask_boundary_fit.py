"""
Detect ego-mask boundary misalignment on real target frames.

Scores each sample:
  - boundary_near_edge: share of mask boundary pixels within ~3px of a strong image edge
  - boundary_mean_dist: mean distance from boundary to nearest strong edge (px, lower=better)
  - t0t1_leak / t0t1_contrast: temporal stability inside vs outside mask

Exports worst frames as overlays + HTML gallery for manual review.

Usage:
  python validate_mask_boundary_fit.py
  python validate_mask_boundary_fit.py --max-samples 12 --top 80
"""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from import_manual_ego_masks import APPROVED_DIR, ROOT
from select_masks_t0t1_stability import collect_all_samples, resize_mask, score_sample, t0t1_diff
from test_ego_consistent_edges import DEFAULT_DATASET, load_rgb

OUT_DIR = ROOT / "validation_boundary_fit"
REPORT_PATH = OUT_DIR / "report.json"
INDEX_PATH = OUT_DIR / "index.html"
FRAMES_DIR = OUT_DIR / "bad_frames"


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    u8 = mask.astype(np.uint8)
    return cv2.morphologyEx(u8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0


def boundary_fit_scores(gray: np.ndarray, mask: np.ndarray) -> dict:
    if mask.sum() < 64:
        return {
            "boundary_near_edge": 1.0,
            "boundary_mean_dist": 0.0,
            "boundary_edge_strength": 1.0,
        }

    boundary = mask_boundary(mask)
    if boundary.sum() < 32:
        return {
            "boundary_near_edge": 1.0,
            "boundary_mean_dist": 0.0,
            "boundary_edge_strength": 1.0,
        }

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    peak = float(np.percentile(mag, 99))
    mag_n = mag / max(peak, 1e-6)

    band = cv2.dilate(boundary.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    thr = float(np.percentile(mag_n[band], 70))
    strong = mag_n >= thr

    dist = cv2.distanceTransform((~strong).astype(np.uint8), cv2.DIST_L2, 3)
    bd = dist[boundary]
    near = float((bd <= 3.0).mean())
    mean_dist = float(bd.mean())
    edge_strength = float(mag_n[boundary].mean())

    return {
        "boundary_near_edge": near,
        "boundary_mean_dist": mean_dist,
        "boundary_edge_strength": edge_strength,
    }


def badness(scores: dict) -> float:
    leak = scores.get("leak", 0.0)
    contrast = scores.get("contrast", 0.0)
    ratio = scores.get("ratio", 1.0)
    near = scores.get("boundary_near_edge", 1.0)
    mean_dist = scores.get("boundary_mean_dist", 0.0)

    return float(
        2.2 * leak
        + 1.4 * max(0.0, 0.55 - contrast)
        + 1.0 * max(0.0, ratio - 0.45)
        + 1.8 * (1.0 - near)
        + 0.12 * mean_dist
    )


def overlay_frame(rgb: np.ndarray, mask: np.ndarray, show_boundary: bool = True) -> np.ndarray:
    out = rgb.copy()
    if not mask.any():
        return out
    red = np.zeros_like(out)
    red[..., 0] = 255
    out[mask] = (0.52 * red[mask] + 0.48 * out[mask]).astype(np.uint8)
    if show_boundary:
        b = mask_boundary(mask)
        out[b] = (0.35 * np.array([255, 255, 0], dtype=np.uint8) + 0.65 * out[b]).astype(np.uint8)
    return out


def score_frame(
    sample_dir: Path,
    camera: str,
    mask: np.ndarray,
) -> dict | None:
    tgt = sample_dir / "target" / f"{camera}.jpg"
    t0p = sample_dir / "input" / "t0" / f"{camera}.jpg"
    t1p = sample_dir / "input" / "t1" / f"{camera}.jpg"
    if not tgt.is_file():
        return None

    rgb = load_rgb(tgt)
    hw = rgb.shape[:2]
    m = resize_mask(mask, hw)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    fit = boundary_fit_scores(gray, m)

    t0t1 = {}
    if t0p.is_file() and t1p.is_file():
        t0 = load_rgb(t0p)
        t1 = load_rgb(t1p)
        s = score_sample(m, t0, t1)
        if s:
            t0t1 = s

    row = {
        "sample_id": sample_dir.name,
        **fit,
        **{k: t0t1.get(k, 0.0) for k in ("leak", "contrast", "ratio", "inside_diff", "outside_diff")},
    }
    row["badness"] = badness(row)
    return row


def write_html(report: dict, out_path: Path) -> None:
    cards = []
    for item in report["worst_frames"]:
        rel = item["overlay_rel"]
        cards.append(
            f'<div class="card">'
            f'<img src="{escape(rel)}" loading="lazy">'
            f'<div class="meta">'
            f'<b>{escape(item["vehicle"])}/{escape(item["camera"])}</b><br>'
            f'sample: …{escape(item["sample_id"][-28:])}<br>'
            f'badness={item["badness"]:.2f} | near_edge={item["boundary_near_edge"]:.0%} | '
            f'dist={item["boundary_mean_dist"]:.1f}px | leak={item.get("leak", 0):.0%} | '
            f'contrast={item.get("contrast", 0):.2f}'
            f'</div></div>'
        )

    group_rows = []
    for g in report["worst_groups"]:
        group_rows.append(
            f'<tr>'
            f'<td>{escape(g["vehicle"])}</td>'
            f'<td>{escape(g["camera"])}</td>'
            f'<td>{g["mask_pct"]:.1%}</td>'
            f'<td>{g["median_near_edge"]:.0%}</td>'
            f'<td>{g["median_badness"]:.2f}</td>'
            f'<td>{g["worst_badness"]:.2f}</td>'
            f'<td>{g["n_samples"]}</td>'
            f'</tr>'
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Mask boundary misalignment</title>
<style>
body {{ font-family: sans-serif; margin: 16px; background: #111; color: #eee; }}
h1,h2 {{ color: #fff; }}
.summary {{ background: #1c1c1c; padding: 12px; border-radius: 8px; margin-bottom: 16px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 12px; }}
.card {{ background: #1c1c1c; border-radius: 8px; overflow: hidden; }}
.card img {{ width: 100%; display: block; }}
.meta {{ padding: 8px 10px; font-size: 13px; line-height: 1.45; }}
table {{ border-collapse: collapse; width: 100%; background: #1c1c1c; }}
td, th {{ border: 1px solid #333; padding: 6px; }}
</style></head><body>
<h1>Несовпадение границы маски с фото</h1>
<div class="summary">
<p>Проверено approved-групп: {report["n_groups"]} | кадров: {report["n_frames_scored"]}</p>
<p>Критерии: граница маски далеко от сильных краёв на фото (жёлтый контур), высокий leak t0↔t1 внутри маски, низкий contrast inside/outside.</p>
<p>Показаны топ-{report["n_worst_shown"]} худших кадров.</p>
</div>
<h2>Худшие кадры</h2>
<div class="grid">{''.join(cards)}</div>
<h2>Худшие группы (медиана)</h2>
<table>
<tr><th>vehicle</th><th>camera</th><th>mask%</th><th>median near edge</th><th>median badness</th><th>worst badness</th><th>n</th></tr>
{''.join(group_rows)}
</table>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--max-samples", type=int, default=16, help="max samples scored per group")
    ap.add_argument("--top", type=int, default=60, help="worst frames to export")
    ap.add_argument("--min-badness", type=float, default=0.55, help="min badness to include frame")
    ap.add_argument("--group-min-badness", type=float, default=0.45, help="min median badness for group table")
    args = ap.parse_args()

    manifest = json.loads((ROOT / "manual_masks_manifest.json").read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    all_frames: list[dict] = []
    group_stats: list[dict] = []

    for item in manifest["items"]:
        if item.get("status") != "approved":
            continue
        vehicle, camera = item["vehicle"], item["camera"]
        mask_path = APPROVED_DIR / f"{vehicle}_{camera}.png"
        if not mask_path.is_file():
            continue
        mask = np.array(Image.open(mask_path).convert("L")) > 127
        mask_pct = float(mask.mean())
        if mask_pct < 0.001:
            continue

        samples = collect_all_samples(args.dataset, vehicle, camera)[: args.max_samples]
        per_sample: list[dict] = []
        for sd in samples:
            row = score_frame(sd, camera, mask)
            if row is None:
                continue
            row["vehicle"] = vehicle
            row["camera"] = camera
            row["mask_pct"] = mask_pct
            per_sample.append(row)
            all_frames.append(row)

        if not per_sample:
            continue

        near_vals = [x["boundary_near_edge"] for x in per_sample]
        bad_vals = [x["badness"] for x in per_sample]
        group_stats.append({
            "vehicle": vehicle,
            "camera": camera,
            "mask_pct": mask_pct,
            "median_near_edge": float(np.median(near_vals)),
            "median_badness": float(np.median(bad_vals)),
            "worst_badness": float(max(bad_vals)),
            "n_samples": len(per_sample),
        })

    all_frames.sort(key=lambda x: x["badness"], reverse=True)
    group_stats.sort(key=lambda x: x["median_badness"], reverse=True)

    worst_frames: list[dict] = []
    for row in all_frames:
        if row["badness"] < args.min_badness:
            break
        if len(worst_frames) >= args.top:
            break

        vehicle, camera = row["vehicle"], row["camera"]
        sample_id = row["sample_id"]
        key = f"{vehicle}_{camera}"
        safe = sample_id.replace(":", "-")
        rel = f"bad_frames/{key}/{safe}.jpg"
        out_path = OUT_DIR / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        tgt = args.dataset / sample_id / "target" / f"{camera}.jpg"
        rgb = load_rgb(tgt)
        m = resize_mask(
            np.array(Image.open(APPROVED_DIR / f"{vehicle}_{camera}.png").convert("L")) > 127,
            rgb.shape[:2],
        )
        Image.fromarray(overlay_frame(rgb, m)).save(out_path, quality=88)

        worst_frames.append({**row, "overlay_rel": rel})

    worst_groups = [g for g in group_stats if g["median_badness"] >= args.group_min_badness][:40]

    report = {
        "dataset": str(args.dataset.resolve()),
        "n_groups": len(group_stats),
        "n_frames_scored": len(all_frames),
        "n_worst_shown": len(worst_frames),
        "min_badness": args.min_badness,
        "worst_frames": worst_frames,
        "worst_groups": worst_groups,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(report, INDEX_PATH)

    print(f"Scored {report['n_frames_scored']} frames across {report['n_groups']} groups")
    print(f"Worst frames exported: {report['n_worst_shown']} (badness >= {args.min_badness})")
    print(f"Gallery: {INDEX_PATH}")

    if worst_groups:
        print("\nTop misaligned groups (median badness):")
        for g in worst_groups[:15]:
            print(
                f"  {g['vehicle']}/{g['camera']:12} "
                f"near={g['median_near_edge']:.0%} bad={g['median_badness']:.2f} "
                f"mask={g['mask_pct']:.1%}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
