"""
Conservative auto-selection among manual mask variants (current / v2 / v3).

Uses cross-scene RGB std + boundary persistence + IoU with auto std blob.
Falls back to auto_std only when ALL manual candidates score poorly.

Usage:
  python auto_select_ego_masks.py
  python auto_select_ego_masks.py --apply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ego_mask_extract import save_binary_mask
from import_manual_ego_masks import APPROVED_DIR, ROOT, VARIANTS_DIR, collect_sample_paths
from test_ego_consistent_edges import DEFAULT_DATASET, load_rgb
from test_ego_multi_signals import align_rgb, keep_bottom_cc, otsu_low_mask

SELECTED_DIR = ROOT / "masks_selected"
REPORT_PATH = ROOT / "auto_mask_selection.json"
VIZ_DIR = ROOT / "validation_auto_select"

ZERO_MASK_GROUPS = {
    ("natelio", "right_fwd"),
    ("orvy", "right_fwd"),
}

MANUAL_BONUS = 0.18
MANUAL_OK_THR = 0.32
MANUAL_BAD_THR = 0.12
AUTO_FALLBACK_MARGIN = 0.15


def anchor_mask(camera: str, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    if camera in ("front", "rear"):
        m[int(h * 0.42) :, :] = True
    elif camera.startswith("left"):
        m[:, : int(w * 0.38)] = True
    elif camera.startswith("right"):
        m[:, int(w * 0.62) :] = True
    else:
        m[int(h * 0.42) :, :] = True
    return m


def camera_roi(camera: str, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    if camera in ("front", "rear"):
        m[int(h * (1.0 - 0.55)) :, :] = True
    elif camera.startswith("left"):
        m[:, : int(w * 0.42)] = True
    elif camera.startswith("right"):
        m[:, int(w * 0.58) :] = True
    else:
        m[int(h * 0.45) :, :] = True
    return m


def load_stack(paths: list[Path], camera: str) -> np.ndarray:
    return align_rgb([load_rgb(p / "target" / f"{camera}.jpg") for p in paths])


def resize_mask(mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    if mask.shape[:2] == hw:
        return mask.astype(bool)
    u8 = cv2.resize(mask.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return u8.astype(bool)


def auto_std_mask(stack: np.ndarray, camera: str) -> np.ndarray:
    h, w = stack.shape[1:3]
    roi = camera_roi(camera, h, w)
    std_map = stack.astype(np.float32).std(axis=0).mean(axis=-1)
    low, _ = otsu_low_mask(std_map, roi)
    low = low.astype(bool)
    if camera in ("front", "rear"):
        return keep_bottom_cc(low.astype(np.uint8)).astype(bool) & roi

    anchor = anchor_mask(camera, h, w)
    edge = anchor & ~cv2.erode(anchor.astype(np.uint8), np.ones((5, 5), np.uint8), 2).astype(bool)
    n, labels, _, _ = cv2.connectedComponentsWithStats(low.astype(np.uint8), 8)
    keep = np.zeros_like(low)
    for lab in range(1, n):
        comp = labels == lab
        if (comp & edge).any():
            keep |= comp
    return keep & roi


def boundary_persistence(mask: np.ndarray, stack: np.ndarray) -> float:
    bnd = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    if bnd.sum() < 16:
        return 0.0
    vals = []
    for img in stack:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        thr = max(8.0, float(np.percentile(mag[bnd], 35)))
        vals.append(float((mag[bnd] >= thr).mean()))
    return float(np.mean(vals))


def score_mask(
    mask: np.ndarray,
    stack: np.ndarray,
    camera: str,
    auto_ref: np.ndarray | None = None,
) -> dict:
    h, w = mask.shape
    std_map = stack.astype(np.float32).std(axis=0).mean(axis=-1)
    anchor = anchor_mask(camera, h, w)
    p50 = float(np.percentile(std_map, 50))

    if mask.sum() < 32:
        low_blob = (std_map < np.percentile(std_map[anchor], 22)) & anchor
        blob_ratio = float(low_blob.sum()) / max(anchor.sum(), 1)
        if blob_ratio > 0.07:
            return {"total": -0.2, "kind": "zero", "blob_ratio": blob_ratio}
        return {"total": 0.40, "kind": "zero", "blob_ratio": blob_ratio}

    inside_std = float(std_map[mask].mean())
    static = max(0.0, 1.0 - inside_std / max(p50, 1e-6))

    outside = ~mask
    if outside.sum() < 64:
        leak = 0.5
    else:
        scene_thr = float(np.percentile(std_map[outside], 58))
        leak = float((std_map[mask] > scene_thr).mean())

    bp = boundary_persistence(mask, stack)
    coverage = float(mask.mean())
    anchor_overlap = float((mask & anchor).sum()) / max(mask.sum(), 1)

    iou_auto = 0.0
    if auto_ref is not None and auto_ref.sum() > 64 and mask.sum() > 64:
        inter = float((mask & auto_ref).sum())
        union = float((mask | auto_ref).sum())
        iou_auto = inter / max(union, 1e-6)

    coverage_pen = max(0.0, coverage - 0.30) * 2.0
    if anchor_overlap < 0.20:
        coverage_pen += 0.25

    total = (
        2.4 * static
        + 0.9 * bp
        + 0.7 * min(0.85, iou_auto)
        + 0.4 * anchor_overlap
        - 2.2 * leak
        - coverage_pen
    )
    return {
        "total": float(total),
        "static": static,
        "leak": leak,
        "boundary_persist": bp,
        "iou_auto": iou_auto,
        "anchor_overlap": anchor_overlap,
        "coverage": coverage,
        "inside_std": inside_std,
    }


def mask_paths_for_group(vehicle: str, camera: str) -> dict[str, Path]:
    key = f"{vehicle}_{camera}"
    out: dict[str, Path] = {}
    for name, base in (
        ("current", APPROVED_DIR / f"{key}.png"),
        ("v2", VARIANTS_DIR / f"{key}_v2.png"),
        ("v3", VARIANTS_DIR / f"{key}_v3.png"),
    ):
        if base.is_file():
            out[name] = base
    return out


def load_mask_from_path(path: Path, hw: tuple[int, int]) -> np.ndarray:
    m = np.array(Image.open(path).convert("L")) > 127
    return resize_mask(m, hw)


def save_compare_viz(
    vehicle: str,
    camera: str,
    stack: np.ndarray,
    chosen: str,
    ranked: list[tuple[str, float]],
    masks: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    ref = stack[0]
    names = [n for n, _ in ranked[:4]]
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.4))
    if len(names) == 1:
        axes = [axes]
    red = np.zeros_like(ref)
    red[..., 0] = 255
    score_map = dict(ranked)
    for ax, name in zip(axes, names):
        ov = ref.copy()
        m = masks[name]
        ov[m] = (0.55 * red[m] + 0.45 * ov[m]).astype(np.uint8)
        star = " *" if name == chosen else ""
        ax.imshow(ov)
        ax.set_title(f"{name}{star}\n{score_map[name]:.2f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{vehicle}/{camera}", fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--max-scenes", type=int, default=8)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--min-variant-improve", type=float, default=0.06)
    args = ap.parse_args()

    manifest = json.loads((ROOT / "manual_masks_manifest.json").read_text(encoding="utf-8"))
    by_key = {(x["vehicle"], x["camera"]): x for x in manifest["items"]}

    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    changed = 0

    groups = sorted({(x["vehicle"], x["camera"]) for x in manifest["items"]})

    for vehicle, camera in groups:
        meta = by_key.get((vehicle, camera), {})
        status = meta.get("status", "")

        paths = collect_sample_paths(args.dataset, vehicle, camera, max_scenes=args.max_scenes)
        if len(paths) < 3:
            continue

        stack = load_stack(paths, camera)
        hw = stack.shape[1:3]
        auto_ref = auto_std_mask(stack, camera)

        if (vehicle, camera) in ZERO_MASK_GROUPS or status == "needs_annotation":
            mask = np.zeros(hw, dtype=bool)
            save_binary_mask(mask, SELECTED_DIR / f"{vehicle}_{camera}.png")
            results.append({
                "vehicle": vehicle, "camera": camera, "chosen": "zero",
                "changed": False, "reason": "no_artifact_group",
            })
            continue

        paths_map = mask_paths_for_group(vehicle, camera)
        manual_names = [n for n in ("current", "v2", "v3") if n in paths_map]
        if not manual_names:
            continue

        variant_manifest = {}
        vpath = ROOT / "mask_variants_manifest.json"
        if vpath.is_file():
            variant_manifest = json.loads(vpath.read_text(encoding="utf-8"))
        group_key = f"{vehicle}_{camera}"
        preferred = variant_manifest.get("dop_default_variant", {}).get(group_key)
        if preferred is None and group_key in variant_manifest.get("auto_two_variant", []):
            preferred = variant_manifest.get("two_variant_default", 2)
        preferred_name = f"v{preferred}" if preferred in (2, 3) else None

        masks: dict[str, np.ndarray] = {}
        raw_scores: dict[str, dict] = {}
        for name in manual_names + ["auto_std", "zero"]:
            if name == "auto_std":
                masks[name] = auto_ref
            elif name == "zero":
                masks[name] = np.zeros(hw, dtype=bool)
            else:
                masks[name] = load_mask_from_path(paths_map[name], hw)
            raw_scores[name] = score_mask(masks[name], stack, camera, auto_ref)

        ranked_manual: list[tuple[str, float]] = []
        for name in manual_names:
            s = raw_scores[name]["total"] + MANUAL_BONUS
            ranked_manual.append((name, s))
        ranked_manual.sort(key=lambda x: x[1], reverse=True)

        best_manual_name, best_manual_score = ranked_manual[0]
        if preferred_name and preferred_name in manual_names:
            pref_score = next(s for n, s in ranked_manual if n == preferred_name)
            if pref_score >= best_manual_score - 0.10:
                best_manual_name = preferred_name
                best_manual_score = pref_score

        current_name = "current" if "current" in manual_names else manual_names[0]
        current_score = next(s for n, s in ranked_manual if n == current_name)

        chosen = best_manual_name
        reason = "best_manual"

        if best_manual_name != current_name:
            if best_manual_score >= current_score + args.min_variant_improve:
                chosen = best_manual_name
                reason = "better_variant"
            else:
                chosen = current_name
                reason = "keep_current"

        switched = chosen != current_name
        if switched:
            changed += 1
            save_compare_viz(
                vehicle, camera, stack, chosen,
                [(n, raw_scores[n]["total"] + (MANUAL_BONUS if n in manual_names else 0)) for n in masks
                 if n in manual_names or n == chosen],
                masks,
                VIZ_DIR / f"{vehicle}_{camera}_select.png",
            )

        save_binary_mask(masks[chosen], SELECTED_DIR / f"{vehicle}_{camera}.png")
        if args.apply:
            dst = APPROVED_DIR / f"{vehicle}_{camera}.png"
            if masks[chosen].sum() >= 32:
                save_binary_mask(masks[chosen], dst)
            elif dst.is_file():
                dst.unlink()

        results.append({
            "vehicle": vehicle,
            "camera": camera,
            "status": status,
            "chosen": chosen,
            "previous": current_name,
            "changed": switched,
            "reason": reason,
            "ranked_manual": ranked_manual,
            "raw_scores": {k: v for k, v in raw_scores.items() if k in manual_names + [chosen, "auto_std"]},
        })

    report = {
        "n_groups": len(results),
        "n_changed": changed,
        "applied": args.apply,
        "items": results,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Evaluated {len(results)} groups, changed {changed}")
    if changed:
        print("\nSwitched (manual variants only):")
        for r in results:
            if r["changed"]:
                rm = r["ranked_manual"]
                print(
                    f"  {r['vehicle']}/{r['camera']:12} "
                    f"{r['previous']} -> {r['chosen']}  ({r['reason']})  "
                    f"scores: {', '.join(f'{n}={s:.2f}' for n, s in rm)}"
                )
    print(f"\nReport: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
