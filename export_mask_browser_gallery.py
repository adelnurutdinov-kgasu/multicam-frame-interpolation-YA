"""
Browser gallery: mask overlays on individual target frames (not mean PNGs).

Shows every sample_id that went into the mean + a few extra diverse trips,
so you can review masks on frames you never opened during annotation.

Outputs:
  methods_gallery/_ego_manual_masks/browser_gallery/index.html
  methods_gallery/_ego_manual_masks/browser_gallery/overlays/{vehicle}_{camera}/...

Usage:
  python export_mask_browser_gallery.py
  python export_mask_browser_gallery.py --max-extra 4
"""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from import_manual_ego_masks import APPROVED_DIR, MEANS_DIR, REJECT_DIR, ROOT, collect_sample_paths, load_index
from test_ego_consistent_edges import DEFAULT_DATASET, load_rgb

OUT_DIR = ROOT / "browser_gallery"
OVERLAY_DIR = OUT_DIR / "overlays"
INDEX_HTML = OUT_DIR / "gallery.json"

ZERO_MASK_GROUPS = {
    ("natelio", "right_fwd"),
    ("orvy", "right_fwd"),
}


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.52) -> np.ndarray:
    out = rgb.copy()
    if not mask.any():
        return out
    red = np.zeros_like(out)
    red[..., 0] = 255
    out[mask] = (alpha * red[mask] + (1.0 - alpha) * out[mask]).astype(np.uint8)
    return out


def resize_mask(mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    if mask.shape[:2] == hw:
        return mask
    u8 = cv2.resize(mask.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return u8.astype(bool)


def load_mask_for_group(vehicle: str, camera: str, status: str) -> np.ndarray | None:
    key = f"{vehicle}_{camera}"
    if (vehicle, camera) in ZERO_MASK_GROUPS or status == "needs_annotation":
        return None
    if status == "reject_black":
        p = REJECT_DIR / f"{key}.png"
    else:
        p = APPROVED_DIR / f"{key}.png"
    if not p.is_file():
        p = ROOT / "masks" / f"{key}.png"
    if not p.is_file():
        return None
    return np.array(Image.open(p).convert("L")) > 127


def sample_label(sample_id: str) -> str:
    parts = sample_id.split("_")
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]} …{sample_id.split('__')[-1]}"
    return sample_id[-24:]


def write_html(gallery: dict, out_path: Path) -> None:
    sections = []
    for g in gallery["groups"]:
        cards = []
        for fr in g["frames"]:
            rel = fr["overlay_rel"]
            cards.append(
                f'<div class="card">'
                f'<img src="{escape(rel)}" loading="lazy" alt="{escape(fr["sample_id"])}">'
                f'<div class="meta">'
                f'<div class="sid">{escape(fr["short_id"])}</div>'
                f'<div class="tag">{escape(fr["kind"])}</div>'
                f'</div></div>'
            )
        status_cls = g["status"].replace("_", "-")
        sections.append(
            f'<section class="group" id="{escape(g["id"])}" data-status="{escape(g["status"])}">'
            f'<h2>{escape(g["vehicle"])} / {escape(g["camera"])}'
            f' <span class="badge {status_cls}">{escape(g["status"])}</span></h2>'
            f'<p class="hint">{escape(g.get("note", ""))} · {len(g["frames"])} кадров</p>'
            f'<div class="grid">{"".join(cards)}</div></section>'
        )

    vehicle_opts = "".join(
        f'<option value="{escape(v)}">{escape(v)}</option>' for v in gallery["vehicles"]
    )
    status_opts = "".join(
        f'<option value="{escape(s)}">{escape(s)}</option>' for s in gallery["statuses"]
    )

    html = f"""<!DOCTYPE html>
<html lang="ru"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ego masks — оверлеи на реальных кадрах</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: system-ui, sans-serif; margin: 0; background: #111; color: #eee; }}
header {{ position: sticky; top: 0; z-index: 10; background: #1a1a1a; border-bottom: 1px solid #333; padding: 12px 16px; }}
h1 {{ margin: 0 0 8px; font-size: 1.2rem; }}
.toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
.toolbar input, .toolbar select {{ padding: 6px 10px; border-radius: 6px; border: 1px solid #444; background: #222; color: #eee; }}
.toolbar label {{ font-size: 0.85rem; color: #aaa; }}
main {{ padding: 16px; max-width: 1600px; margin: 0 auto; }}
.group {{ margin-bottom: 32px; scroll-margin-top: 80px; }}
.group h2 {{ font-size: 1.1rem; margin: 0 0 4px; }}
.hint {{ color: #888; font-size: 0.85rem; margin: 0 0 12px; }}
.badge {{ font-size: 0.75rem; padding: 2px 8px; border-radius: 999px; background: #333; }}
.badge.approved {{ background: #1e4620; }}
.badge.reject-black {{ background: #4a1e1e; }}
.badge.needs-annotation, .badge.bad-mean-few-scenes {{ background: #3a3a18; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }}
.card {{ background: #1c1c1c; border-radius: 8px; overflow: hidden; border: 1px solid #2a2a2a; }}
.card img {{ width: 100%; display: block; aspect-ratio: 16/9; object-fit: contain; background: #000; }}
.meta {{ padding: 8px 10px; font-size: 0.78rem; }}
.sid {{ color: #ccc; word-break: break-all; }}
.tag {{ color: #888; margin-top: 2px; }}
nav.toc {{ margin: 12px 0; max-height: 120px; overflow-y: auto; font-size: 0.8rem; }}
nav.toc a {{ color: #7cb; margin-right: 10px; white-space: nowrap; }}
.hidden {{ display: none !important; }}
</style></head><body>
<header>
<h1>Маски на реальных target-кадрах (не mean)</h1>
<p style="margin:4px 0 8px;color:#888;font-size:0.85rem">
Красный = маска. «mean» — усреднённый кадр; остальные — отдельные поездки из датасета.
<a href="composer.html" style="color:#7cb;margin-left:12px">→ Выбор маски (picker)</a>
</p>
<div class="toolbar">
<label>Поиск <input type="search" id="q" placeholder="vehicle / camera"></label>
<label>Машина <select id="veh"><option value="">все</option>{vehicle_opts}</select></label>
<label>Статус <select id="st"><option value="">все</option>{status_opts}</select></label>
<span id="cnt" style="color:#888;font-size:0.85rem"></span>
</div>
<nav class="toc" id="toc"></nav>
</header>
<main>
{"".join(sections)}
</main>
<script>
const q = document.getElementById('q');
const veh = document.getElementById('veh');
const st = document.getElementById('st');
const groups = document.querySelectorAll('.group');
const toc = document.getElementById('toc');
groups.forEach(g => {{
  const a = document.createElement('a');
  a.href = '#' + g.id;
  a.textContent = g.id.replace('_', '/');
  toc.appendChild(a);
}});
function apply() {{
  const query = q.value.toLowerCase();
  let vis = 0;
  groups.forEach(g => {{
    const id = g.id.toLowerCase();
    const okQ = !query || id.includes(query);
    const okV = !veh.value || id.startsWith(veh.value + '_');
    const okS = !st.value || g.dataset.status === st.value;
    const show = okQ && okV && okS;
    g.classList.toggle('hidden', !show);
    if (show) vis++;
  }});
  document.getElementById('cnt').textContent = vis + ' групп';
}}
[q, veh, st].forEach(el => el.addEventListener('input', apply));
apply();
</script>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--max-extra", type=int, default=4,
                    help="extra diverse samples beyond mean sample_ids")
    ap.add_argument("--also-mean", action="store_true", default=True)
    args = ap.parse_args()

    manifest = json.loads((ROOT / "manual_masks_manifest.json").read_text(encoding="utf-8"))
    index = load_index()
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    groups_out: list[dict] = []
    vehicles: set[str] = set()
    statuses: set[str] = set()
    n_frames = 0

    items = sorted(manifest["items"], key=lambda x: (x["vehicle"], x["camera"]))

    for item in items:
        vehicle, camera = item["vehicle"], item["camera"]
        status = item.get("status", "")
        if status not in ("approved", "reject_black", "needs_annotation", "bad_mean_few_scenes"):
            continue

        vehicles.add(vehicle)
        statuses.add(status)
        key = f"{vehicle}_{camera}"
        group_dir = OVERLAY_DIR / key
        group_dir.mkdir(parents=True, exist_ok=True)

        mask_bool = load_mask_for_group(vehicle, camera, status)
        meta = index.get((vehicle, camera), {})
        sample_ids = list(meta.get("sample_ids", item.get("sample_ids_mean", [])))

        extra_paths = collect_sample_paths(args.dataset, vehicle, camera, max_scenes=args.max_extra + 8)
        extra_ids = [p.name for p in extra_paths if p.name not in sample_ids][: args.max_extra]
        all_ids = sample_ids + extra_ids

        frames: list[dict] = []

        if args.also_mean:
            mean_path = MEANS_DIR / f"{key}.png"
            if mean_path.is_file():
                mean_rgb = load_rgb(mean_path)
                hw = mean_rgb.shape[:2]
                m = resize_mask(mask_bool, hw) if mask_bool is not None else np.zeros(hw, dtype=bool)
                ov = overlay_mask(mean_rgb, m)
                rel = f"overlays/{key}/__mean.jpg"
                Image.fromarray(ov).save(OUT_DIR / rel, quality=88)
                frames.append({
                    "sample_id": "__mean__",
                    "short_id": "MEAN (усреднение)",
                    "kind": "mean",
                    "overlay_rel": rel,
                })
                n_frames += 1

        for sid in all_ids:
            tgt = args.dataset / sid / "target" / f"{camera}.jpg"
            if not tgt.is_file():
                continue
            rgb = load_rgb(tgt)
            hw = rgb.shape[:2]
            m = resize_mask(mask_bool, hw) if mask_bool is not None else np.zeros(hw, dtype=bool)
            ov = overlay_mask(rgb, m)
            safe = sid.replace(":", "-")
            rel = f"overlays/{key}/{safe}.jpg"
            Image.fromarray(ov).save(OUT_DIR / rel, quality=88)
            kind = "в mean" if sid in sample_ids else "доп. поездка"
            frames.append({
                "sample_id": sid,
                "short_id": sample_label(sid),
                "kind": kind,
                "overlay_rel": rel,
            })
            n_frames += 1

        note = ""
        if (vehicle, camera) in ZERO_MASK_GROUPS or status == "needs_annotation":
            note = "без маски (артефактов нет)"
        elif status == "reject_black":
            note = "reject — маска не совпала на mean"
        elif mask_bool is None:
            note = "маска отсутствует"

        groups_out.append({
            "id": key,
            "vehicle": vehicle,
            "camera": camera,
            "status": status,
            "note": note,
            "frames": frames,
        })

    gallery = {
        "dataset": str(args.dataset.resolve()),
        "n_groups": len(groups_out),
        "n_frames": n_frames,
        "vehicles": sorted(vehicles),
        "statuses": sorted(statuses),
        "groups": groups_out,
    }
    INDEX_HTML.write_text(json.dumps(gallery, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(gallery, OUT_DIR / "index.html")

    print(f"Gallery: {OUT_DIR / 'index.html'}")
    print(f"  groups={len(groups_out)}  overlay frames={n_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
