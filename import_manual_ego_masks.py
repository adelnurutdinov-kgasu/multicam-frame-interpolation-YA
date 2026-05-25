"""
Import manual ego artifact masks from `mean mask art/` annotations.

  red  — artifact region (hood / mirror / body in frame)
  black — user flag: mask does NOT align across scenes (skip or review)

Exports:
  masks/{vehicle}_{camera}.png          — binary mask (from red)
  masks/{vehicle}_{camera}_reject.png   — only if black marks present
  manual_masks_manifest.json            — status per group + alignment score

Usage:
  python import_manual_ego_masks.py
  python import_manual_ego_masks.py --validate --dataset .../train
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ego_mask_extract import extract_black_paint, extract_red_mask, save_binary_mask
from test_ego_consistent_edges import DEFAULT_DATASET, SAMPLE_RE, diversify_paths

REPO = Path(__file__).resolve().parent
ROOT = REPO / "methods_gallery/_ego_manual_masks"
ART_DIR = ROOT / "mean mask art"
MEANS_DIR = ROOT / "means"
MASKS_DIR = ROOT / "masks"
APPROVED_DIR = ROOT / "masks_approved"
REJECT_DIR = ROOT / "masks_reject"
VARIANTS_DIR = ROOT / "masks_variants"
VARIANTS_JSON = ROOT / "mask_variants_manifest.json"
VIZ_DIR = ROOT / "validation"
INDEX_PATH = ROOT / "index.json"

NAME_RE = re.compile(
    r"^(.+)_(front|rear|left_fwd|right_fwd|left_bwd|right_bwd)\s+1\.png$",
    re.I,
)


def parse_art_name(path: Path) -> tuple[str, str] | None:
    m = NAME_RE.match(path.name)
    if not m:
        return None
    return m.group(1), m.group(2)


def extract_black_flag(rgb: np.ndarray, red: np.ndarray) -> np.ndarray:
    return extract_black_paint(rgb, red)


def load_variant_manifest() -> dict:
    if not VARIANTS_JSON.is_file():
        return {}
    return json.loads(VARIANTS_JSON.read_text(encoding="utf-8"))


def load_index() -> dict[tuple[str, str], dict]:
    data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    return {(x["vehicle"], x["camera"]): x for x in data["items"]}


def collect_sample_paths(dataset: Path, vehicle: str, camera: str, max_scenes: int = 8) -> list[Path]:
    paths: list[Path] = []
    for d in sorted(dataset.iterdir()):
        if not d.is_dir():
            continue
        m = SAMPLE_RE.match(d.name)
        if not m or m.group(1) != vehicle:
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if meta["target_camera"] != camera:
            continue
        paths.append(d)
    return diversify_paths(paths)[:max_scenes]


def alignment_score(mask: np.ndarray, sample_paths: list[Path], camera: str) -> dict:
    """Boundary edge frequency across scenes — low = mask boundary not stable."""
    if not sample_paths or mask.sum() < 64:
        return {"edge_freq_mean": 0.0, "edge_freq_min": 0.0, "n_scenes": 0}

    edges = []
    h, w = mask.shape
    for sd in sample_paths:
        img = np.array(Image.open(sd / "target" / f"{camera}.jpg").convert("RGB"))
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        e = cv2.Canny(gray, 50, 150) > 0
        edges.append(e)

    stack = np.stack(edges, axis=0)
    edge_freq = stack.mean(axis=0)

    boundary = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    if boundary.sum() < 32:
        return {"edge_freq_mean": 0.0, "edge_freq_min": 0.0, "n_scenes": len(sample_paths)}

    vals = edge_freq[boundary]
    return {
        "edge_freq_mean": float(vals.mean()),
        "edge_freq_min": float(vals.min()),
        "n_scenes": len(sample_paths),
    }


def save_validation_grid(
    vehicle: str,
    camera: str,
    mean_rgb: np.ndarray,
    mask: np.ndarray,
    sample_paths: list[Path],
    out_path: Path,
    max_cols: int = 4,
):
    cols = min(max_cols, max(1, len(sample_paths)))
    fig, axes = plt.subplots(1, cols + 1, figsize=(3.2 * (cols + 1), 3.2))
    if cols == 0:
        axes = [axes]

    overlay = mean_rgb.copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    overlay[mask] = (0.5 * red[mask] + 0.5 * overlay[mask]).astype(np.uint8)
    axes[0].imshow(overlay)
    axes[0].set_title(f"mean+mask\n{vehicle}/{camera}")
    axes[0].axis("off")

    for i, sd in enumerate(sample_paths[:cols]):
        img = np.array(Image.open(sd / "target" / f"{camera}.jpg").convert("RGB"))
        if img.shape[:2] != mask.shape[:2]:
            img = cv2.resize(img, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
        ov = img.copy()
        ov[mask] = (0.5 * red[mask] + 0.5 * ov[mask]).astype(np.uint8)
        axes[i + 1].imshow(ov)
        axes[i + 1].set_title(sd.name.split("_")[-1][:12], fontsize=8)
        axes[i + 1].axis("off")

    for j in range(len(sample_paths[:cols]) + 1, len(axes)):
        axes[j].axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def apply_variant_default(
    vehicle: str,
    camera: str,
    variant_manifest: dict,
) -> tuple[np.ndarray | None, int | None, str | None]:
    key = f"{vehicle}_{camera}"
    default_v = variant_manifest.get("dop_default_variant", {}).get(key)
    if default_v is None and key in variant_manifest.get("auto_two_variant", []):
        default_v = variant_manifest.get("two_variant_default", 2)
    if default_v is None:
        return None, None, None

    variant_path = VARIANTS_DIR / f"{vehicle}_{camera}_v{default_v}.png"
    if not variant_path.is_file():
        return None, default_v, None
    mask = np.array(Image.open(variant_path).convert("L")) > 127
    return mask, default_v, str(variant_path.resolve())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--edge-freq-thr", type=float, default=0.35,
                    help="below this on boundary → likely misaligned")
    ap.add_argument("--viz-bad-only", action="store_true", default=True)
    ap.add_argument("--skip-variants", action="store_true",
                    help="do not run process_manual_mask_variants.py first")
    args = ap.parse_args()

    if not args.skip_variants:
        import process_manual_mask_variants as pmv
        pmv.main()

    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    by_key = load_index()
    variant_manifest = load_variant_manifest()
    force_approved = set(variant_manifest.get("force_approved_red", []))
    manifest: list[dict] = []

    for art_path in sorted(ART_DIR.glob("*.png")):
        parsed = parse_art_name(art_path)
        if not parsed:
            continue
        vehicle, camera = parsed
        rgb = np.array(Image.open(art_path).convert("RGB"))
        mask = extract_red_mask(rgb)
        black = extract_black_flag(rgb, mask)
        black_px = int(black.sum())

        meta = by_key.get((vehicle, camera), {})
        mean_path = MEANS_DIR / f"{vehicle}_{camera}.png"
        if mean_path.is_file() and rgb.shape[:2] != np.array(Image.open(mean_path)).shape[:2]:
            # resize mask to canonical mean size
            mean_hw = np.array(Image.open(mean_path)).shape[:2]
            mask_u8 = cv2.resize(mask.astype(np.uint8), (mean_hw[1], mean_hw[0]), interpolation=cv2.INTER_NEAREST)
            mask = mask_u8.astype(bool)

        out_mask = MASKS_DIR / f"{vehicle}_{camera}.png"
        Image.fromarray((mask.astype(np.uint8) * 255)).save(out_mask)

        group_key = f"{vehicle}_{camera}"
        force_art = vehicle in variant_manifest.get("force_art_red_vehicles", [])
        if force_art:
            variant_mask, default_v, variant_path = None, None, None
        else:
            variant_mask, default_v, variant_path = apply_variant_default(
                vehicle, camera, variant_manifest,
            )
        if variant_mask is not None:
            mask = variant_mask
            save_binary_mask(mask, out_mask)

        if group_key in force_approved and mask.mean() >= 0.001:
            status = "approved"
        elif force_art and mask.mean() >= 0.001:
            status = "approved"
        elif variant_mask is not None and mask.mean() >= 0.001:
            status = "approved"
        elif black_px > 500:
            status = "reject_black"
        elif mask.mean() < 0.001:
            status = "empty_red"
        else:
            status = "approved"

        row = {
            "vehicle": vehicle,
            "camera": camera,
            "art_png": str(art_path.resolve()),
            "mask_png": str(out_mask.resolve()),
            "mask_pct": float(mask.mean()),
            "black_flag_px": black_px,
            "status": status,
            "n_used_mean": meta.get("n_used", 0),
            "sample_ids_mean": meta.get("sample_ids", []),
        }
        if default_v is not None:
            row["default_variant"] = default_v
            row["variant_mask_png"] = variant_path
            row["override"] = variant_manifest.get("overrides", {}).get(group_key)

        if args.validate and status == "approved":
            paths = collect_sample_paths(args.dataset, vehicle, camera, max_scenes=6)
            scores = alignment_score(mask, paths, camera)
            row.update(scores)
            row["edge_align_warn"] = scores["edge_freq_mean"] < args.edge_freq_thr
        elif args.validate:
            row["edge_align_warn"] = None

        if status == "approved":
            APPROVED_DIR.mkdir(parents=True, exist_ok=True)
            Image.fromarray((mask.astype(np.uint8) * 255)).save(
                APPROVED_DIR / f"{vehicle}_{camera}.png",
            )
        elif status == "reject_black":
            REJECT_DIR.mkdir(parents=True, exist_ok=True)
            Image.fromarray((mask.astype(np.uint8) * 255)).save(
                REJECT_DIR / f"{vehicle}_{camera}.png",
            )
            if args.validate and args.viz_bad_only:
                paths = collect_sample_paths(args.dataset, vehicle, camera, max_scenes=4)
                mean_rgb = np.array(Image.open(mean_path).convert("RGB")) if mean_path.is_file() else rgb
                save_validation_grid(
                    vehicle, camera, mean_rgb, mask, paths,
                    VIZ_DIR / f"{vehicle}_{camera}_reject.png",
                )

        manifest.append(row)

    # groups with variants in `доп/` but no `{vehicle}_{camera} 1.png`
    annotated = {(r["vehicle"], r["camera"]) for r in manifest}
    for group in variant_manifest.get("groups", []):
        vehicle = group["vehicle"]
        camera = group["camera"]
        if (vehicle, camera) in annotated:
            continue
        if not group.get("variants"):
            continue
        variant_mask, default_v, variant_path = apply_variant_default(
            vehicle, camera, variant_manifest,
        )
        if variant_mask is None:
            continue
        meta = by_key.get((vehicle, camera), {})
        out_mask = MASKS_DIR / f"{vehicle}_{camera}.png"
        save_binary_mask(variant_mask, out_mask)
        row = {
            "vehicle": vehicle,
            "camera": camera,
            "art_png": None,
            "mask_png": str(out_mask.resolve()),
            "mask_pct": float(variant_mask.mean()),
            "black_flag_px": 0,
            "status": "approved" if variant_mask.mean() >= 0.001 else "empty_red",
            "n_used_mean": meta.get("n_used", 0),
            "sample_ids_mean": meta.get("sample_ids", []),
            "default_variant": default_v,
            "variant_mask_png": variant_path,
            "source": "dop_only",
        }
        if row["status"] == "approved":
            APPROVED_DIR.mkdir(parents=True, exist_ok=True)
            save_binary_mask(variant_mask, APPROVED_DIR / f"{vehicle}_{camera}.png")
        manifest.append(row)

    # groups without annotation
    annotated = {(r["vehicle"], r["camera"]) for r in manifest}
    for (vehicle, camera), meta in sorted(by_key.items()):
        if (vehicle, camera) in annotated:
            continue
        st = "needs_annotation"
        if meta.get("n_used", 99) <= 3:
            st = "bad_mean_few_scenes"
        manifest.append({
            "vehicle": vehicle,
            "camera": camera,
            "status": st,
            "n_used_mean": meta.get("n_used", 0),
            "mean_png": meta.get("mean_png"),
        })

    summary = {
        "masks_dir": str(MASKS_DIR.resolve()),
        "approved_dir": str(APPROVED_DIR.resolve()),
        "reject_dir": str(REJECT_DIR.resolve()),
        "variants_dir": str(VARIANTS_DIR.resolve()),
        "variants_manifest": str(VARIANTS_JSON.resolve()) if VARIANTS_JSON.is_file() else None,
        "n_annotated": sum(1 for r in manifest if r.get("art_png")),
        "n_approved": sum(1 for r in manifest if r.get("status") == "approved"),
        "n_reject_black": sum(1 for r in manifest if r.get("status") == "reject_black"),
        "n_with_variants": sum(1 for r in manifest if r.get("default_variant")),
        "n_edge_warn": sum(1 for r in manifest if r.get("edge_align_warn")),
        "n_needs_annotation": sum(1 for r in manifest if r.get("status") == "needs_annotation"),
        "n_bad_mean_few_scenes": sum(1 for r in manifest if r.get("status") == "bad_mean_few_scenes"),
        "items": manifest,
    }
    (ROOT / "manual_masks_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    print(f"Imported {summary['n_annotated']} masks -> {MASKS_DIR}")
    print(f"  approved={summary['n_approved']} -> {APPROVED_DIR}")
    print(f"  reject_black={summary['n_reject_black']}  with_variants={summary['n_with_variants']}")
    print(f"  edge_warn={summary['n_edge_warn']}")
    print(f"  needs_annotation={summary['n_needs_annotation']}  bad_mean(n<=3)={summary['n_bad_mean_few_scenes']}")

    print("\nReject (чёрная разметка):")
    for r in manifest:
        if r.get("status") == "reject_black":
            print(f"  {r['vehicle']}/{r['camera']}  black_px={r['black_flag_px']}  n_mean={r.get('n_used_mean')}")

    print("\nПлохой mean (n<=3, не размечено):")
    for r in manifest:
        if r.get("status") == "bad_mean_few_scenes":
            print(f"  {r['vehicle']}/{r['camera']}  n_mean={r.get('n_used_mean')}")

    baland = [r for r in manifest if r.get("vehicle") == "baland"]
    if baland:
        print("\nbaland:")
        for r in baland:
            print(f"  {r['camera']:12} {r.get('status','?'):18} n_mean={r.get('n_used_mean', '?')}")

    if args.validate:
        print(f"\nBad alignment viz: {VIZ_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
