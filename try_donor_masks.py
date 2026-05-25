"""
Try borrowing masks from other (vehicle, camera) groups with the same camera.

For each poorly fitting group, scores all approved same-camera masks on its samples,
picks the best donor, and exports before/after comparison on worst frames.

Usage:
  python try_donor_masks.py
  python try_donor_masks.py --vehicles crozby kynde
"""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from import_manual_ego_masks import APPROVED_DIR, MEANS_DIR, ROOT, VARIANTS_DIR
from select_masks_t0t1_stability import collect_all_samples, resize_mask
from test_ego_consistent_edges import DEFAULT_DATASET, load_rgb
from validate_mask_boundary_fit import badness, overlay_frame, score_frame

FIT_REPORT = ROOT / "validation_boundary_fit" / "report.json"
OUT_DIR = ROOT / "validation_donor_try"
REPORT_PATH = OUT_DIR / "donor_report.json"
INDEX_PATH = OUT_DIR / "index.html"
SUGGESTED_DIR = ROOT / "masks_donor_suggested"


def load_mask(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return np.array(Image.open(path).convert("L")) > 127


def canonical_hw(vehicle: str, camera: str) -> tuple[int, int] | None:
    p = MEANS_DIR / f"{vehicle}_{camera}.png"
    if p.is_file():
        return np.array(Image.open(p)).shape[:2]
    return None


def list_candidates(
    target_vehicle: str,
    camera: str,
    manifest_items: list[dict],
) -> dict[str, tuple[str, Path, np.ndarray]]:
    """tag -> (donor_label, path, mask at donor canonical size)."""
    out: dict[str, tuple[str, Path, np.ndarray]] = {}

    for item in manifest_items:
        if item.get("status") != "approved":
            continue
        v, c = item["vehicle"], item["camera"]
        if c != camera:
            continue
        key = f"{v}_{c}"
        p = APPROVED_DIR / f"{key}.png"
        m = load_mask(p)
        if m is None or m.mean() < 0.001:
            continue
        tag = "own" if v == target_vehicle else f"donor:{v}"
        out[tag] = (f"{v}/{c}", p, m)

        if v == target_vehicle:
            for vt in ("v2", "v3"):
                vp = VARIANTS_DIR / f"{key}_{vt}.png"
                vm = load_mask(vp)
                if vm is not None and vm.mean() >= 0.001:
                    out[vt] = (f"{v}/{c} {vt}", vp, vm)

    return out


def score_candidate_on_samples(
    mask: np.ndarray,
    samples: list[Path],
    camera: str,
) -> tuple[float, float, list[dict]]:
    rows: list[dict] = []
    for sd in samples:
        row = score_frame(sd, camera, mask)
        if row:
            rows.append(row)
    if not rows:
        return 999.0, 999.0, rows
    bads = [r["badness"] for r in rows]
    return float(np.median(bads)), float(max(bads)), rows


def label_image(img: np.ndarray, text: str) -> np.ndarray:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    draw.rectangle((0, 0, pil.width, 28), fill=(0, 0, 0))
    draw.text((8, 6), text, fill=(255, 255, 255))
    return np.array(pil)


def side_by_side(left: np.ndarray, right: np.ndarray, left_lbl: str, right_lbl: str) -> np.ndarray:
    a = label_image(left, left_lbl)
    b = label_image(right, right_lbl)
    if a.shape[0] != b.shape[0]:
        b = cv2.resize(b, (b.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    return np.hstack([a, b])


def write_html(report: dict, out_path: Path) -> None:
    cards = []
    for r in report["results"]:
        if not r.get("comparisons"):
            continue
        comps = "".join(
            f'<img src="{escape(c["rel"])}" loading="lazy" style="width:100%;margin-bottom:8px;">'
            for c in r["comparisons"]
        )
        improved = r["median_badness_own"] - r["median_badness_best"]
        cards.append(
            f'<div class="card">'
            f'<div class="meta">'
            f'<b>{escape(r["vehicle"])}/{escape(r["camera"])}</b><br>'
            f'своя: bad={r["median_badness_own"]:.2f} near={r["near_edge_own"]:.0%}<br>'
            f'лучший: <b>{escape(r["best_donor_label"])}</b> bad={r["median_badness_best"]:.2f} '
            f'near={r["near_edge_best"]:.0%} '
            f'({"+" if improved >= 0 else ""}{improved:.2f})'
            f'</div>{comps}</div>'
        )

    rows = []
    for r in report["results"]:
        delta = r["median_badness_own"] - r["median_badness_best"]
        rows.append(
            f'<tr class="{"ok" if delta >= 0.2 else "meh"}">'
            f'<td>{escape(r["vehicle"])}</td>'
            f'<td>{escape(r["camera"])}</td>'
            f'<td>{r["median_badness_own"]:.2f}</td>'
            f'<td>{escape(r["best_donor_label"])}</td>'
            f'<td>{r["median_badness_best"]:.2f}</td>'
            f'<td>{delta:+.2f}</td>'
            f'<td>{r["near_edge_own"]:.0%} → {r["near_edge_best"]:.0%}</td>'
            f'</tr>'
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Donor mask try</title>
<style>
body {{ font-family: sans-serif; margin: 16px; background: #111; color: #eee; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(520px, 1fr)); gap: 12px; }}
.card {{ background: #1c1c1c; border-radius: 8px; overflow: hidden; padding-bottom: 8px; }}
.meta {{ padding: 10px; font-size: 13px; line-height: 1.5; }}
table {{ border-collapse: collapse; width: 100%; background: #1c1c1c; margin-top: 20px; }}
td, th {{ border: 1px solid #333; padding: 6px; font-size: 13px; }}
tr.ok {{ background: #142014; }}
tr.meh {{ background: #1c1c1c; }}
</style></head><body>
<h1>Подбор маски-донора (та же camera)</h1>
<p>Групп проверено: {report["n_groups"]}. Слева — текущая маска, справа — лучший донор.</p>
<div class="grid">{''.join(cards) if cards else '<p>нет улучшений</p>'}</div>
<h2>Сводка</h2>
<table>
<tr><th>vehicle</th><th>camera</th><th>own bad</th><th>best donor</th><th>donor bad</th><th>Δ</th><th>near edge</th></tr>
{''.join(rows)}
</table>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--fit-report", type=Path, default=FIT_REPORT)
    ap.add_argument("--max-samples", type=int, default=20)
    ap.add_argument("--min-group-badness", type=float, default=2.0)
    ap.add_argument("--min-improve", type=float, default=0.15)
    ap.add_argument("--compare-frames", type=int, default=3)
    ap.add_argument("--vehicles", nargs="*", help="only these target vehicles")
    args = ap.parse_args()

    fit = json.loads(args.fit_report.read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "manual_masks_manifest.json").read_text(encoding="utf-8"))
    items = manifest["items"]

    targets = [
        g for g in fit.get("worst_groups", [])
        if g.get("median_badness", 0) >= args.min_group_badness
    ]
    if args.vehicles:
        allow = set(args.vehicles)
        targets = [g for g in targets if g["vehicle"] in allow]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUGGESTED_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for tg in targets:
        vehicle, camera = tg["vehicle"], tg["camera"]
        own_path = APPROVED_DIR / f"{vehicle}_{camera}.png"
        own_mask = load_mask(own_path)
        if own_mask is None:
            continue

        samples = collect_all_samples(args.dataset, vehicle, camera)[: args.max_samples]
        if not samples:
            continue

        candidates = list_candidates(vehicle, camera, items)
        if len(candidates) < 2:
            continue

        scored: list[dict] = []
        for tag, (label, path, mask) in candidates.items():
            med, worst, rows = score_candidate_on_samples(mask, samples, camera)
            near = float(np.median([r["boundary_near_edge"] for r in rows])) if rows else 0.0
            scored.append({
                "tag": tag,
                "label": label,
                "path": str(path),
                "median_badness": med,
                "worst_badness": worst,
                "median_near_edge": near,
                "rows": rows,
                "mask": mask,
            })

        scored.sort(key=lambda x: x["median_badness"])
        own_row = next(x for x in scored if x["tag"] == "own")
        best = scored[0]

        result = {
            "vehicle": vehicle,
            "camera": camera,
            "median_badness_own": own_row["median_badness"],
            "near_edge_own": own_row["median_near_edge"],
            "best_donor_tag": best["tag"],
            "best_donor_label": best["label"],
            "median_badness_best": best["median_badness"],
            "near_edge_best": best["median_near_edge"],
            "improved": own_row["median_badness"] - best["median_badness"],
            "ranked_top5": [
                {
                    "tag": x["tag"],
                    "label": x["label"],
                    "median_badness": x["median_badness"],
                    "median_near_edge": x["median_near_edge"],
                }
                for x in scored[:5]
            ],
            "comparisons": [],
        }

        improved_enough = result["improved"] >= args.min_improve and best["tag"] != "own"
        if improved_enough:
            hw = canonical_hw(vehicle, camera)
            if hw is not None and best["mask"].shape[:2] != hw:
                m_save = cv2.resize(
                    best["mask"].astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                m_save = best["mask"]
            save_path = SUGGESTED_DIR / f"{vehicle}_{camera}.png"
            Image.fromarray((m_save.astype(np.uint8) * 255)).save(save_path)
            result["suggested_mask"] = str(save_path.resolve())

        own_rows = sorted(own_row["rows"], key=lambda x: x["badness"], reverse=True)
        best_mask = best["mask"]
        key = f"{vehicle}_{camera}"
        comp_dir = OUT_DIR / "compare" / key
        comp_dir.mkdir(parents=True, exist_ok=True)

        for i, row in enumerate(own_rows[: args.compare_frames]):
            sid = row["sample_id"]
            tgt = args.dataset / sid / "target" / f"{camera}.jpg"
            rgb = load_rgb(tgt)
            hw = rgb.shape[:2]
            m_own = resize_mask(own_mask, hw)
            m_best = resize_mask(best_mask, hw)
            combo = side_by_side(
                overlay_frame(rgb, m_own),
                overlay_frame(rgb, m_best),
                f"own bad={row['badness']:.2f}",
                f"{best['label']} bad={score_frame(Path(args.dataset / sid), camera, best_mask)['badness']:.2f}",
            )
            safe = sid.replace(":", "-")
            rel = f"compare/{key}/{safe}.jpg"
            Image.fromarray(combo).save(OUT_DIR / rel, quality=88)
            result["comparisons"].append({"sample_id": sid, "rel": rel, "own_badness": row["badness"]})

        results.append(result)
        print(
            f"{vehicle}/{camera:12} own={own_row['median_badness']:.2f} "
            f"best={best['label']:20} {best['median_badness']:.2f} "
            f"d={result['improved']:+.2f}"
            + (" OK" if improved_enough else "")
        )

    results.sort(key=lambda x: x["improved"], reverse=True)
    report = {
        "dataset": str(args.dataset.resolve()),
        "n_groups": len(results),
        "min_improve": args.min_improve,
        "results": results,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(report, INDEX_PATH)
    print(f"\nGallery: {INDEX_PATH}")
    print(f"Suggested masks: {SUGGESTED_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
