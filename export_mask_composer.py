"""
Mask picker: compare all same-camera masks on real frames and save your choice.

Exports:
  browser_gallery/composer.html      — pick one mask, preview on frames
  browser_gallery/composer_data.json
  browser_gallery/masks_bank/*.png
  browser_gallery/frames/{group}/*.jpg  — raw target frames (no overlay)

Usage:
  python export_mask_composer.py
  # save downloaded PNG to masks_composed/{vehicle}_{camera}.png
  python import_composed_masks.py
"""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from export_mask_browser_gallery import OUT_DIR, ZERO_MASK_GROUPS, overlay_mask, sample_label
from import_manual_ego_masks import APPROVED_DIR, MEANS_DIR, REJECT_DIR, ROOT, VARIANTS_DIR, collect_sample_paths, load_index
from rebuild_composer_html import build_html
from test_ego_consistent_edges import DEFAULT_DATASET, load_rgb

MASK_BANK = OUT_DIR / "masks_bank"
FRAMES_DIR = OUT_DIR / "frames"
COMPOSED_DIR = ROOT / "masks_composed"
DATA_JSON = OUT_DIR / "composer_data.json"
HTML_PATH = OUT_DIR / "composer.html"
CAMERAS = ("front", "rear", "left_fwd", "right_fwd", "left_bwd", "right_bwd")


def canonical_hw(vehicle: str, camera: str) -> tuple[int, int] | None:
    mean_path = MEANS_DIR / f"{vehicle}_{camera}.png"
    if mean_path.is_file():
        return np.array(Image.open(mean_path)).shape[:2]
    return None


def load_mask_path(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return np.array(Image.open(path).convert("L")) > 127


def export_mask_bank(key: str, tag: str, src: Path, hw: tuple[int, int]) -> str | None:
    m = load_mask_path(src)
    if m is None or m.mean() < 0.001:
        return None
    if m.shape[:2] != hw:
        m = cv2.resize(m.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    name = f"{key}.png" if tag == "own" else f"{key}_{tag}.png"
    rel = f"masks_bank/{name}"
    MASK_BANK.mkdir(parents=True, exist_ok=True)
    Image.fromarray((m.astype(np.uint8) * 255)).save(OUT_DIR / rel)
    return rel


def mask_pct(path: Path) -> float:
    m = load_mask_path(path)
    return float(m.mean()) if m is not None else 0.0


def collect_candidates(
    vehicle: str,
    camera: str,
    status: str,
    hw: tuple[int, int],
    all_by_camera: dict[str, list[str]],
) -> list[dict]:
    key = f"{vehicle}_{camera}"
    out: list[dict] = []

    def add(tag: str, label: str, path: Path, source: str) -> None:
        bank_key = key if source == "own" else f"{source}_{camera}"
        rel = export_mask_bank(bank_key, tag, path, hw)
        if rel is None:
            return
        out.append({
            "id": f"{source}:{tag}",
            "label": label,
            "rel": rel,
            "source": source,
            "tag": tag,
            "mask_pct": round(mask_pct(path), 4),
        })

    if status == "reject_black":
        p = REJECT_DIR / f"{key}.png"
        if p.is_file():
            add("own", "своя (reject)", p, "own")
    elif (vehicle, camera) not in ZERO_MASK_GROUPS and status != "needs_annotation":
        for tag, label, suffix in (
            ("own", "своя approved", ""),
            ("v2", "своя v2", "_v2"),
            ("v3", "своя v3", "_v3"),
        ):
            if tag == "own":
                p = APPROVED_DIR / f"{key}.png"
            else:
                p = VARIANTS_DIR / f"{key}{suffix}.png"
            if p.is_file():
                add(tag, label, p, "own")

    for dv in sorted(all_by_camera.get(camera, [])):
        if dv == vehicle:
            continue
        dkey = f"{dv}_{camera}"
        for tag, label, suffix in (
            ("own", f"{dv} approved", ""),
            ("v2", f"{dv} v2", "_v2"),
            ("v3", f"{dv} v3", "_v3"),
        ):
            if tag == "own":
                p = APPROVED_DIR / f"{dkey}.png"
            else:
                p = VARIANTS_DIR / f"{dkey}{suffix}.png"
            if p.is_file():
                add(tag, label, p, dv)

    suggested = ROOT / "masks_donor_suggested" / f"{key}.png"
    if suggested.is_file():
        rel = export_mask_bank(key, "suggested", suggested, hw)
        if rel:
            out.append({
                "id": "suggested:auto",
                "label": "auto-suggested (донор)",
                "rel": f"masks_bank/{key}_suggested.png",
                "source": "suggested",
                "tag": "suggested",
                "mask_pct": round(mask_pct(suggested), 4),
            })

    return out


def export_raw_frame(dataset: Path, sid: str, camera: str, out_path: Path) -> bool:
    if sid == "__mean__":
        return False
    tgt = dataset / sid / "target" / f"{camera}.jpg"
    if not tgt.is_file():
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = load_rgb(tgt)
    Image.fromarray(rgb).save(out_path, quality=90)
    return True


def build_group_entry(
    item: dict,
    index: dict,
    dataset: Path,
    max_extra: int,
    all_by_camera: dict[str, list[str]],
) -> dict | None:
    vehicle, camera = item["vehicle"], item["camera"]
    status = item.get("status", "")
    if status not in ("approved", "reject_black", "needs_annotation", "bad_mean_few_scenes"):
        return None

    hw = canonical_hw(vehicle, camera)
    if hw is None:
        return None

    key = f"{vehicle}_{camera}"
    candidates = collect_candidates(vehicle, camera, status, hw, all_by_camera)
    if not candidates and status not in ("needs_annotation", "bad_mean_few_scenes"):
        return None

    meta = index.get((vehicle, camera), {})
    sample_ids = list(meta.get("sample_ids", item.get("sample_ids_mean", [])))
    extra_paths = collect_sample_paths(dataset, vehicle, camera, max_scenes=max_extra + 8)
    extra_ids = [p.name for p in extra_paths if p.name not in sample_ids][:max_extra]
    all_ids = sample_ids + extra_ids

    frames: list[dict] = []
    frame_dir = FRAMES_DIR / key
    mean_path = MEANS_DIR / f"{key}.png"
    if mean_path.is_file():
        frame_dir.mkdir(parents=True, exist_ok=True)
        mean_out = frame_dir / "__mean.jpg"
        if not mean_out.is_file():
            rgb = np.array(Image.open(mean_path).convert("RGB"))
            Image.fromarray(rgb).save(mean_out, quality=90)
        frames.append({
            "id": "__mean__",
            "label": "MEAN",
            "rel": f"frames/{key}/__mean.jpg",
            "in_mean": True,
        })

    for sid in all_ids:
        safe = sid.replace(":", "-")
        rel = f"frames/{key}/{safe}.jpg"
        out_path = OUT_DIR / rel
        if not out_path.is_file():
            export_raw_frame(dataset, sid, camera, out_path)
        if out_path.is_file():
            frames.append({
                "id": sid,
                "label": sample_label(sid),
                "rel": rel,
                "in_mean": sid in sample_ids,
            })

    note = ""
    if (vehicle, camera) in ZERO_MASK_GROUPS or status == "needs_annotation":
        note = "нет своей маски — выберите донора"
    elif status == "reject_black":
        note = "reject — выберите другую маску"

    return {
        "id": key,
        "vehicle": vehicle,
        "camera": camera,
        "status": status,
        "note": note,
        "h": hw[0],
        "w": hw[1],
        "candidates": candidates,
        "frames": frames,
    }


def write_composer_html(out_path: Path, payload: dict) -> None:
    out_path.write_text(build_html(payload), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--max-extra", type=int, default=4)
    ap.add_argument("--html-only", action="store_true", help="rebuild composer.html from existing composer_data.json")
    args = ap.parse_args()

    if args.html_only:
        if not DATA_JSON.is_file():
            print(f"Missing {DATA_JSON}")
            return 1
        payload = json.loads(DATA_JSON.read_text(encoding="utf-8"))
        write_composer_html(HTML_PATH, payload)
        print(f"Rebuilt: {HTML_PATH}  groups={payload.get('n_groups', '?')}")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    COMPOSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "manual_masks_manifest.json").read_text(encoding="utf-8"))
    index = load_index()

    all_by_camera: dict[str, list[str]] = {c: [] for c in CAMERAS}
    for item in manifest["items"]:
        v, c = item["vehicle"], item["camera"]
        if c in all_by_camera and v not in all_by_camera[c]:
            all_by_camera[c].append(v)

    groups: list[dict] = []
    for item in sorted(manifest["items"], key=lambda x: (x["vehicle"], x["camera"])):
        entry = build_group_entry(item, index, args.dataset, args.max_extra, all_by_camera)
        if entry and entry["frames"]:
            groups.append(entry)

    payload = {
        "masks_composed_dir": str(COMPOSED_DIR.resolve()),
        "n_groups": len(groups),
        "groups": groups,
    }
    DATA_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_composer_html(HTML_PATH, payload)

    n_cand = sum(len(g["candidates"]) for g in groups)
    print(f"Mask picker: {HTML_PATH}")
    print(f"  groups={len(groups)}  candidates={n_cand}")
    print(f"Save PNG to: {COMPOSED_DIR}")
    print("Then: python import_composed_masks.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
