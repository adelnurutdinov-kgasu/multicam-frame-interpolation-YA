"""
Build v2/v3 mask variants and refresh approved masks per user review.

  доп/{vehicle}_{camera} 2.png|3.png  — user-provided two variants
  auto front groups                      — v2=red, v3=red+black from `1.png`

Usage:
  python process_manual_mask_variants.py
  python scripts/ego/import_manual_ego_masks.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
import sys
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


import numpy as np
from PIL import Image

from ego_mask_extract import (
    detect_dominant_modes,
    extract_black_paint,
    extract_mask_by_mode,
    extract_red_paint,
    infer_variant_mode,
    resize_mask_to,
    save_binary_mask,
)

CAMERAS = ("right_fwd", "left_fwd", "right_bwd", "left_bwd", "front", "rear")


def parse_group_stem(stem_name: str) -> tuple[str, str] | None:
    for camera in CAMERAS:
        suffix = f"_{camera}"
        if stem_name.endswith(suffix):
            return stem_name[: -len(suffix)], camera
    return None


ROOT = REPO / "methods_gallery/_ego_manual_masks"
ART_DIR = ROOT / "mean mask art"
DOP_DIR = ART_DIR / "доп"
MEANS_DIR = ROOT / "means"
VARIANTS_DIR = ROOT / "masks_variants"
VARIANTS_JSON = ROOT / "mask_variants_manifest.json"

# User: these reject_black groups are actually fine — use red-only from `1.png`
FORCE_APPROVED_RED = {
    ("bett", "left_bwd"),
    ("bett", "right_fwd"),
    ("floi", "front"),
    ("hanna", "right_fwd"),
    ("jurita", "left_fwd"),
    ("jurita", "rear"),
    ("natelio", "right_bwd"),
    ("rickie", "right_bwd"),
    ("targi", "left_bwd"),
    ("weel", "front"),
}

# User: assistant builds 2 variants from original `1.png`
AUTO_TWO_VARIANT = {
    ("badat", "front"),
    ("crozby", "front"),
    ("leid", "front"),
    ("robb", "front"),
}

# User attached in `доп/` — which variant is default approved mask
DOP_DEFAULT_VARIANT = {
    ("baland", "right_fwd"): 3,
    ("crozby", "rear"): 3,
    ("greton", "left_bwd"): 3,
    ("soan", "left_fwd"): 2,
}

TWO_VARIANT_DEFAULT = 2  # red-only


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def canonical_hw(vehicle: str, camera: str) -> tuple[int, int]:
    mean_path = MEANS_DIR / f"{vehicle}_{camera}.png"
    if mean_path.is_file():
        return np.array(Image.open(mean_path)).shape[:2]
    art_path = ART_DIR / f"{vehicle}_{camera} 1.png"
    if art_path.is_file():
        return load_rgb(art_path).shape[:2]
    raise FileNotFoundError(f"no mean/art for {vehicle}_{camera}")


def build_front_variant_pngs(vehicle: str, camera: str) -> None:
    art1 = ART_DIR / f"{vehicle}_{camera} 1.png"
    if not art1.is_file():
        raise FileNotFoundError(art1)
    rgb1 = load_rgb(art1)
    mean_path = MEANS_DIR / f"{vehicle}_{camera}.png"
    base = load_rgb(mean_path) if mean_path.is_file() else rgb1.copy()

    red_paint = extract_red_paint(rgb1)
    black_paint = extract_black_paint(rgb1, red_paint)

    v2 = base.copy()
    v2[red_paint] = (255, 0, 0)

    v3 = base.copy()
    v3[red_paint] = (255, 0, 0)
    v3[black_paint] = (0, 0, 0)

    DOP_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(v2).save(DOP_DIR / f"{vehicle}_{camera} 2.png")
    Image.fromarray(v3).save(DOP_DIR / f"{vehicle}_{camera} 3.png")


def process_variant_art(
    vehicle: str,
    camera: str,
    variant_idx: int,
    art_path: Path,
    hw: tuple[int, int],
) -> dict:
    rgb = load_rgb(art_path)
    mode = infer_variant_mode(rgb)
    if mode == "empty":
        mode = "red" if variant_idx == TWO_VARIANT_DEFAULT else "red_plus_black"
    mask = extract_mask_by_mode(rgb, mode)
    mask = resize_mask_to(mask, hw)
    out = VARIANTS_DIR / f"{vehicle}_{camera}_v{variant_idx}.png"
    save_binary_mask(mask, out)
    stats = detect_dominant_modes(rgb)
    return {
        "variant": variant_idx,
        "art_png": str(art_path.resolve()),
        "mask_png": str(out.resolve()),
        "mode": mode,
        "mask_pct": float(mask.mean()),
        "red_px": stats["red_px"],
        "black_px": stats["black_px"],
    }


def main() -> int:
    VARIANTS_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []

    for vehicle, camera in sorted(AUTO_TWO_VARIANT):
        build_front_variant_pngs(vehicle, camera)
        entries.append({
            "vehicle": vehicle,
            "camera": camera,
            "source": "auto_front_two_variant",
            "default_variant": TWO_VARIANT_DEFAULT,
            "v2_mode": "red",
            "v3_mode": "red_plus_black",
        })

    dop_keys: set[tuple[str, str]] = set()
    for art_path in sorted(DOP_DIR.glob("*.png")):
        stem = art_path.stem
        if " " not in stem:
            continue
        name, idx_s = stem.rsplit(" ", 1)
        if idx_s not in {"2", "3"}:
            continue
        parsed = parse_group_stem(name)
        if not parsed:
            continue
        vehicle, camera = parsed
        dop_keys.add((vehicle, camera))

    all_variant_keys = set(AUTO_TWO_VARIANT) | dop_keys
    variant_details: dict[tuple[str, str], list[dict]] = {}

    for vehicle, camera in sorted(all_variant_keys):
        hw = canonical_hw(vehicle, camera)
        details = []
        for variant_idx in (2, 3):
            art_path = DOP_DIR / f"{vehicle}_{camera} {variant_idx}.png"
            if not art_path.is_file():
                continue
            details.append(process_variant_art(vehicle, camera, variant_idx, art_path, hw))
        if details:
            variant_details[(vehicle, camera)] = details

    overrides = {
        f"{v}_{c}": "force_approved_red"
        for v, c in sorted(FORCE_APPROVED_RED)
    }

    manifest = {
        "variants_dir": str(VARIANTS_DIR.resolve()),
        "dop_dir": str(DOP_DIR.resolve()),
        "force_approved_red": [f"{v}_{c}" for v, c in sorted(FORCE_APPROVED_RED)],
        "auto_two_variant": [f"{v}_{c}" for v, c in sorted(AUTO_TWO_VARIANT)],
        "dop_default_variant": {
            f"{v}_{c}": idx for (v, c), idx in sorted(DOP_DEFAULT_VARIANT.items())
        },
        "two_variant_default": TWO_VARIANT_DEFAULT,
        "overrides": overrides,
        "groups": [],
    }

    for key in sorted(set(list(all_variant_keys) + list(FORCE_APPROVED_RED))):
        vehicle, camera = key
        default_v = DOP_DEFAULT_VARIANT.get(key, TWO_VARIANT_DEFAULT if key in AUTO_TWO_VARIANT else None)
        manifest["groups"].append({
            "vehicle": vehicle,
            "camera": camera,
            "default_variant": default_v,
            "variants": variant_details.get(key, []),
            "override": overrides.get(f"{vehicle}_{camera}"),
        })

    VARIANTS_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Built variants -> {VARIANTS_DIR}")
    print(f"  auto front: {len(AUTO_TWO_VARIANT)} groups")
    print(f"  dop groups: {len(dop_keys)}")
    print(f"  force approved red: {len(FORCE_APPROVED_RED)}")
    print(f"Manifest: {VARIANTS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
