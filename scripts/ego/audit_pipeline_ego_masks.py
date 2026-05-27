"""
Проверка привязки ego-масок к сэмплам пайплайна (та же логика, что consensus_kit._load_ego_mask).

Для каждого sample_id + target_camera:
  masks_approved/{vehicle}_{camera}.png

Без target (test): оверлей на input/t0/<camera>.jpg.

Выход:
  methods_gallery/_ego_manual_masks/pipeline_mask_audit/report.json
  methods_gallery/_ego_manual_masks/pipeline_mask_audit/index.html
  pipeline_mask_audit/overlays/{split}/{sample_id}.jpg  — только проблемные
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from html import escape
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
import sys
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


import cv2
import numpy as np
from PIL import Image



sys.path.insert(0, str(REPO))
from consensus_kit import ConsensusConfig, _load_ego_mask, _parse_vehicle, default_config  # noqa: E402

VEHICLE_RE = re.compile(r"_([a-zA-Z]+)_\d+__\d{3}$")
OUT_ROOT = REPO / "methods_gallery/_ego_manual_masks/pipeline_mask_audit"
DEFAULT_DATASET = Path(r"C:\Users\adel\Downloads\cv_dataset\final_dataset_v5_participants")


def iter_split_samples(split_dir: Path):
    for sd in sorted(split_dir.iterdir()):
        if not sd.is_dir() or not (sd / "meta.json").is_file():
            continue
        meta = json.loads((sd / "meta.json").read_text(encoding="utf-8"))
        sid = meta.get("sample_id", sd.name)
        cam = meta["target_camera"]
        yield sd, sid, cam, meta


def audit_split(cfg: ConsensusConfig, split: str, dataset_root: Path, *, make_overlays: int) -> dict:
    split_dir = dataset_root / split
    target_hw = (cfg.image_h, cfg.image_w)
    rows = []
    by_status = defaultdict(int)
    by_vehicle_missing = defaultdict(set)

    for sd, sid, cam, meta in iter_split_samples(split_dir):
        vehicle = _parse_vehicle(sid)
        mask_path = Path(cfg.ego_masks_root) / f"{vehicle}_{cam}.png" if vehicle else None
        has_file = mask_path.is_file() if mask_path else False
        art = _load_ego_mask(cfg, sid, cam, target_hw)
        cov = float(art.mean())

        if not vehicle:
            status = "parse_failed"
        elif not has_file:
            status = "missing_mask_file"
            by_vehicle_missing[f"{vehicle}/{cam}"].add(sid)
        elif cov < 0.001:
            status = "empty_mask"
        else:
            status = "ok"

        by_status[status] += 1
        row = {
            "sample_id": sid,
            "split": split,
            "camera": cam,
            "vehicle": vehicle,
            "mask_path": str(mask_path) if mask_path else "",
            "has_mask_file": has_file,
            "mask_coverage": round(cov, 4),
            "status": status,
        }
        rows.append(row)

        if make_overlays > 0 and status != "ok":
            if sum(1 for r in rows if r["status"] != "ok" and r.get("_viz")) >= make_overlays:
                continue
            ref = sd / "target" / f"{cam}.jpg"
            if not ref.is_file():
                ref = sd / "input" / "t0" / f"{cam}.jpg"
            if ref.is_file() and has_file:
                row["_viz"] = True
                save_overlay(ref, mask_path, target_hw, OUT_ROOT / "overlays" / split / f"{sid[:60]}.jpg", sid, status)

    return {
        "split": split,
        "n_samples": len(rows),
        "by_status": dict(by_status),
        "missing_groups": {k: len(v) for k, v in sorted(by_vehicle_missing.items())},
        "samples": rows,
    }


def save_overlay(
    img_path: Path,
    mask_path: Path,
    target_hw: tuple[int, int],
    out_path: Path,
    sid: str,
    status: str,
) -> None:
    import matplotlib.pyplot as plt

    img = np.array(Image.open(img_path).convert("RGB"))
    m = np.array(Image.open(mask_path).convert("L"))
    H, W = target_hw
    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    if m.shape[:2] != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    mask = m > 127
    red = np.zeros_like(img)
    red[..., 0] = 255
    ov = img.copy()
    ov[mask] = (0.55 * red[mask] + 0.45 * ov[mask]).astype(np.uint8)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].imshow(img)
    ax[0].set_title("frame", fontsize=9)
    ax[1].imshow(ov)
    ax[1].set_title(f"mask {mask.mean()/255:.1%}", fontsize=9)
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{status}\n{sid[-50:]}", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_html(reports: list[dict], out_path: Path) -> None:
    parts = ['<!DOCTYPE html><html><head><meta charset="utf-8"><title>Pipeline ego mask audit</title>',
             "<style>body{font-family:sans-serif;margin:16px} table{border-collapse:collapse;width:100%}",
             "td,th{border:1px solid #ccc;padding:6px} tr.missing{background:#fff0f0}",
             "tr.ok{background:#f8fff8}</style></head><body>",
             "<h1>Привязка ego-масок к пайплайну</h1>",
             "<p>Ключ: <code>{vehicle}_{camera}.png</code> из <code>sample_id</code> (VEHICLE_RE) + <code>meta.target_camera</code></p>"]

    for rep in reports:
        parts.append(f"<h2>Split: {escape(rep['split'])} — {rep['n_samples']} samples</h2>")
        parts.append("<ul>")
        for st, n in sorted(rep["by_status"].items(), key=lambda x: -x[1]):
            parts.append(f"<li><b>{escape(st)}</b>: {n}</li>")
        parts.append("</ul>")
        if rep.get("missing_groups"):
            parts.append("<h3>Группы без файла маски (число сэмплов)</h3><ul>")
            for g, n in list(rep["missing_groups"].items())[:40]:
                parts.append(f"<li>{escape(g)} — {n}</li>")
            parts.append("</ul>")

        bad = [s for s in rep["samples"] if s["status"] != "ok"][:80]
        parts.append(f"<h3>Примеры проблем ({len(bad)} показано)</h3><table><tr><th>status</th><th>vehicle</th><th>cam</th><th>sample</th><th>overlay</th></tr>")
        for s in bad:
            cls = "missing" if s["status"] != "ok" else "ok"
            ov = OUT_ROOT / "overlays" / rep["split"] / f"{s['sample_id'][:60]}.jpg"
            img = f'<img src="../../overlays/{rep["split"]}/{escape(ov.name)}" width="240">' if ov.is_file() else "—"
            parts.append(
                f'<tr class="{cls}"><td>{escape(s["status"])}</td><td>{escape(str(s["vehicle"]))}</td>'
                f'<td>{escape(s["camera"])}</td><td>{escape(s["sample_id"][-45:])}</td><td>{img}</td></tr>'
            )
        parts.append("</table>")

    parts.append("</body></html>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--splits", default="test,train")
    ap.add_argument("--allowed-cameras", default="", help="напр. front,rear — пусто = все")
    ap.add_argument("--overlay-limit", type=int, default=30, help="макс. PNG оверлеев на split")
    ap.add_argument("--out-name", default="report", help="имя отчёта без расширения")
    args = ap.parse_args()

    cfg = default_config()
    cam_filter: tuple[str, ...] | None = None
    if args.allowed_cameras:
        cam_filter = tuple(c.strip() for c in args.allowed_cameras.split(",") if c.strip())

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    reports = []
    for split in args.splits.split(","):
        split = split.strip()
        rep = audit_split(cfg, split, args.dataset, make_overlays=args.overlay_limit)
        if cam_filter:
            rep["samples"] = [s for s in rep["samples"] if s["camera"] in cam_filter]
            rep["n_samples"] = len(rep["samples"])
            by_st = defaultdict(int)
            for s in rep["samples"]:
                by_st[s["status"]] += 1
            rep["by_status"] = dict(by_st)
        reports.append(rep)
        print(f"{split}: n={rep['n_samples']}  {rep['by_status']}")

    report_path = OUT_ROOT / f"{args.out_name}.json"
    html_path = OUT_ROOT / f"{args.out_name}.html"
    report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(reports, html_path)
    print(f"report -> {report_path}")
    print(f"html   -> {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
