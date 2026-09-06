"""
Mask picker: compare all same-camera masks on real frames and save your choice.

Exports:
  browser_gallery/composer.html      — pick one mask, preview on frames
  browser_gallery/composer_data.json
  browser_gallery/masks_bank/*.png
  browser_gallery/frames/{group}/*.jpg  — raw target frames (no overlay)

Usage:
  python scripts/ego/export_mask_composer.py
  # save downloaded PNG to masks_composed/{vehicle}_{camera}.png
  python scripts/ego/import_composed_masks.py
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
import numpy as np
from PIL import Image

from ego_mask_policy import ZERO_MASK_GROUPS, is_zero_mask_group
from export_mask_browser_gallery import OUT_DIR, overlay_mask, sample_label
from import_manual_ego_masks import APPROVED_DIR, MEANS_DIR, REJECT_DIR, ROOT, VARIANTS_DIR, collect_sample_paths, load_index
from rebuild_composer_html import build_html
from consensus_kit import _parse_vehicle
from test_ego_consistent_edges import DEFAULT_DATASET, load_rgb

DATA_JSON_TEST = OUT_DIR / "composer_data_test.json"
HTML_PATH_TEST = OUT_DIR / "composer_test.html"

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


def canonical_hw_from_test_frame(
    datasets: list[Path],
    sample_ids: list[str],
    camera: str,
) -> tuple[int, int] | None:
    """Размер кадра test, если нет mean (группы только из test split)."""
    for sid in sample_ids:
        sd = resolve_sample_dir(datasets, sid)
        if sd is None:
            continue
        for parts in (
            ("target", f"{camera}.jpg"),
            ("input", "t0", f"{camera}.jpg"),
            ("input", "t1", f"{camera}.jpg"),
        ):
            src = sd.joinpath(*parts)
            if src.is_file():
                return np.array(Image.open(src).convert("RGB")).shape[:2]
    return None


def load_mask_path(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return np.array(Image.open(path).convert("L")) > 127


def export_empty_mask_bank(key: str, hw: tuple[int, int]) -> str:
    """Пустая маска для ZERO_MASK_GROUPS (выбор в picker)."""
    rel = f"masks_bank/{key}_zero.png"
    MASK_BANK.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros(hw, dtype=np.uint8)).save(OUT_DIR / rel)
    return rel


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
    *,
    test_picker: bool = False,
) -> list[dict]:
    key = f"{vehicle}_{camera}"
    out: list[dict] = []

    if is_zero_mask_group(vehicle, camera):
        rel = export_empty_mask_bank(key, hw)
        out.append({
            "id": "zero:none",
            "label": "БЕЗ маски (пусто)",
            "rel": rel,
            "source": "zero",
            "tag": "none",
            "mask_pct": 0.0,
        })
        if not test_picker:
            return out

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
    elif status != "needs_annotation":
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


def resolve_sample_dir(datasets: list[Path], sid: str) -> Path | None:
    for ds in datasets:
        p = ds / sid
        if p.is_dir() and (p / "meta.json").is_file():
            return p
    return None


def export_raw_frame(datasets: list[Path], sid: str, camera: str, out_path: Path) -> bool:
    if sid == "__mean__":
        return False
    sd = resolve_sample_dir(datasets, sid)
    if sd is None:
        return False
    for rel in (f"target/{camera}.jpg", f"input/t0/{camera}.jpg", f"input/t1/{camera}.jpg"):
        src = sd.joinpath(*rel.split("/"))
        if not src.is_file():
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rgb = load_rgb(src)
        Image.fromarray(rgb).save(out_path, quality=90)
        return True
    return False


def collect_test_split_groups(test_dir: Path) -> dict[tuple[str, str], list[str]]:
    """Все sample_id из test/, сгруппированные по (vehicle, target_camera)."""
    out: dict[tuple[str, str], list[str]] = {}
    for sd in sorted(test_dir.iterdir()):
        if not sd.is_dir() or not (sd / "meta.json").is_file():
            continue
        meta = json.loads((sd / "meta.json").read_text(encoding="utf-8"))
        sid = meta.get("sample_id", sd.name)
        cam = meta["target_camera"]
        vehicle = _parse_vehicle(sid)
        if not vehicle:
            continue
        out.setdefault((vehicle, cam), []).append(sid)
    return out


def manifest_item_for(vehicle: str, camera: str, manifest_items: list[dict]) -> dict:
    for it in manifest_items:
        if it["vehicle"] == vehicle and it["camera"] == camera:
            return it
    key = f"{vehicle}_{camera}"
    approved = APPROVED_DIR / f"{key}.png"
    if approved.is_file():
        status = "approved"
    elif (REJECT_DIR / f"{key}.png").is_file():
        status = "reject_black"
    else:
        status = "needs_annotation"
    return {"vehicle": vehicle, "camera": camera, "status": status}


def build_group_entry(
    item: dict,
    index: dict,
    datasets: list[Path],
    max_extra: int,
    all_by_camera: dict[str, list[str]],
    *,
    test_frame_ids: list[str] | None = None,
    include_mean: bool = True,
    test_picker: bool = False,
) -> dict | None:
    vehicle, camera = item["vehicle"], item["camera"]
    status = item.get("status", "")
    if status not in ("approved", "reject_black", "needs_annotation", "bad_mean_few_scenes"):
        return None

    hw = canonical_hw(vehicle, camera)
    if hw is None and test_frame_ids:
        hw = canonical_hw_from_test_frame(datasets, test_frame_ids, camera)
    if hw is None:
        return None

    key = f"{vehicle}_{camera}"
    candidates = collect_candidates(
        vehicle, camera, status, hw, all_by_camera, test_picker=test_picker
    )
    if not candidates and status not in ("needs_annotation", "bad_mean_few_scenes"):
        return None

    approved_path = APPROVED_DIR / f"{key}.png"
    needs_ego_mask = test_picker and (
        not approved_path.is_file()
        or mask_pct(approved_path) < 0.001
    ) and not is_zero_mask_group(vehicle, camera)
    needs_ego_mask_zero = test_picker and is_zero_mask_group(vehicle, camera)

    if test_frame_ids is not None:
        all_ids = list(test_frame_ids)
        sample_ids = list(test_frame_ids)
    else:
        meta = index.get((vehicle, camera), {})
        sample_ids = list(meta.get("sample_ids", item.get("sample_ids_mean", [])))
        extra_paths: list[Path] = []
        seen: set[str] = set()
        for ds in datasets:
            for p in collect_sample_paths(ds, vehicle, camera, max_scenes=max_extra + 12):
                if p.name not in seen:
                    seen.add(p.name)
                    extra_paths.append(p)
        extra_ids = [p.name for p in extra_paths if p.name not in sample_ids][:max_extra]
        all_ids = sample_ids + extra_ids

    frames: list[dict] = []
    frame_dir = FRAMES_DIR / key
    mean_path = MEANS_DIR / f"{key}.png"
    if include_mean and mean_path.is_file():
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
            export_raw_frame(datasets, sid, camera, out_path)
        if out_path.is_file():
            frames.append({
                "id": sid,
                "label": sample_label(sid),
                "rel": rel,
                "in_mean": sid in sample_ids,
            })

    note = ""
    if test_frame_ids is not None:
        note = f"test: {len(test_frame_ids)} кадр(ов)"
    if needs_ego_mask:
        note = (note + " — " if note else "") + "⚠ НУЖНА маска (нет approved) — выберите донора"
    elif needs_ego_mask_zero:
        note = (note + " — " if note else "") + "zero по политике; для test можно выбрать маску из кандидатов"
    elif is_zero_mask_group(vehicle, camera):
        note = (note + " — " if note else "") + "БЕЗ маски (zero)"
    elif status == "needs_annotation":
        note = (note + " — " if note else "") + "нет своей маски — выберите донора"
    elif status == "reject_black":
        note = (note + " — " if note else "") + "reject — выберите другую маску"

    return {
        "id": key,
        "vehicle": vehicle,
        "camera": camera,
        "status": status,
        "note": note,
        "needs_ego_mask": bool(needs_ego_mask or needs_ego_mask_zero),
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
    ap.add_argument(
        "--also-test",
        action="store_true",
        help="Добавить кадры из sibling test/ (t0 если нет target)",
    )
    ap.add_argument(
        "--test-only",
        action="store_true",
        help="Только 199 test: composer_test.html + кадры из test/",
    )
    ap.add_argument("--max-extra", type=int, default=4)
    ap.add_argument("--html-only", action="store_true", help="rebuild composer.html from existing composer_data.json")
    args = ap.parse_args()

    if args.html_only:
        src = DATA_JSON_TEST if args.test_only else DATA_JSON
        dst = HTML_PATH_TEST if args.test_only else HTML_PATH
        if not src.is_file():
            print(f"Missing {src}")
            return 1
        payload = json.loads(src.read_text(encoding="utf-8"))
        write_composer_html(dst, payload)
        print(f"Rebuilt: {dst}  groups={payload.get('n_groups', '?')}")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    COMPOSED_DIR.mkdir(parents=True, exist_ok=True)

    if args.test_only:
        test_dir = args.dataset.parent / "test"
        if not test_dir.is_dir():
            print(f"Нет test split: {test_dir}")
            return 1
        test_groups = collect_test_split_groups(test_dir)
        n_frames = sum(len(v) for v in test_groups.values())
        manifest = json.loads((ROOT / "manual_masks_manifest.json").read_text(encoding="utf-8"))
        index = load_index()
        all_by_camera: dict[str, list[str]] = {c: [] for c in CAMERAS}
        for item in manifest["items"]:
            v, c = item["vehicle"], item["camera"]
            if c in all_by_camera and v not in all_by_camera[c]:
                all_by_camera[c].append(v)
        groups: list[dict] = []
        for (vehicle, camera), sids in sorted(test_groups.items()):
            item = manifest_item_for(vehicle, camera, manifest["items"])
            entry = build_group_entry(
                item,
                index,
                [test_dir],
                max_extra=0,
                all_by_camera=all_by_camera,
                test_frame_ids=sids,
                include_mean=False,
                test_picker=True,
            )
            if entry and entry["frames"]:
                groups.append(entry)
        n_todo = sum(1 for g in groups if g.get("needs_ego_mask"))
        payload = {
            "masks_composed_dir": str(COMPOSED_DIR.resolve()),
            "n_groups": len(groups),
            "n_test_samples": n_frames,
            "n_needs_ego_mask": n_todo,
            "test_only": True,
            "groups": groups,
        }
        print(f"  needs_ego_mask={n_todo}  (фильтр в composer_test)")
        DATA_JSON_TEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        write_composer_html(HTML_PATH_TEST, payload)
        print(f"Test mask picker: {HTML_PATH_TEST}")
        print(f"  groups={len(groups)}  test_frames={n_frames}")
        print(f"Open: http://127.0.0.1:8765/composer_test.html")
        return 0

    datasets = [args.dataset]
    test_dir = args.dataset.parent / "test"
    if args.also_test and test_dir.is_dir():
        datasets.append(test_dir)
        print(f"  + test frames: {test_dir}")

    manifest = json.loads((ROOT / "manual_masks_manifest.json").read_text(encoding="utf-8"))
    index = load_index()

    all_by_camera: dict[str, list[str]] = {c: [] for c in CAMERAS}
    for item in manifest["items"]:
        v, c = item["vehicle"], item["camera"]
        if c in all_by_camera and v not in all_by_camera[c]:
            all_by_camera[c].append(v)

    groups: list[dict] = []
    for item in sorted(manifest["items"], key=lambda x: (x["vehicle"], x["camera"])):
        entry = build_group_entry(item, index, datasets, args.max_extra, all_by_camera)
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
    print("Then: python scripts/ego/import_composed_masks.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
