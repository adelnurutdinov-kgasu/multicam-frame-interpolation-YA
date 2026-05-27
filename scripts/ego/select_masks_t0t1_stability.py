"""
Select best ego mask per (vehicle, camera) using t0↔t1 temporal stability.

Ego artifacts (hood/mirror) change little between t0 and t1; scene moves more.
Score = low |ΔL| inside mask vs outside. Used to pick v2 vs v3 (and own when alone).

Evaluates ALL samples in dataset with target+t0+t1 for each group.

Usage:
  python select_masks_t0t1_stability.py
  python select_masks_t0t1_stability.py --apply
  python select_masks_t0t1_stability.py --apply --update-variant-defaults
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
import sys
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


import cv2
import numpy as np
from PIL import Image

from ego_mask_extract import save_binary_mask
from import_manual_ego_masks import APPROVED_DIR, MEANS_DIR, ROOT, VARIANTS_DIR, load_index
from process_manual_mask_variants import canonical_hw
from test_ego_consistent_edges import DEFAULT_DATASET, SAMPLE_RE, load_rgb

REPORT_PATH = ROOT / "t0t1_mask_selection.json"
VARIANTS_JSON = ROOT / "mask_variants_manifest.json"

ZERO_MASK_GROUPS = {
    ("natelio", "right_fwd"),
    ("orvy", "right_fwd"),
}


def collect_all_samples(dataset: Path, vehicle: str, camera: str) -> list[Path]:
    out: list[Path] = []
    for d in sorted(dataset.iterdir()):
        if not d.is_dir():
            continue
        m = SAMPLE_RE.match(d.name)
        if not m or m.group(1) != vehicle:
            continue
        meta_path = d / "meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("target_camera") != camera:
            continue
        t0 = d / "input" / "t0" / f"{camera}.jpg"
        t1 = d / "input" / "t1" / f"{camera}.jpg"
        tgt = d / "target" / f"{camera}.jpg"
        if t0.is_file() and t1.is_file() and tgt.is_file():
            out.append(d)
    return out


def resize_mask(mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    if mask.shape[:2] == hw:
        return mask.astype(bool)
    u8 = cv2.resize(mask.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return u8.astype(bool)


def load_mask_file(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 127


def t0t1_diff(t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    a = t0.astype(np.float32)
    b = t1.astype(np.float32)
    return np.abs(a - b).mean(axis=-1) / 255.0


def score_sample(mask: np.ndarray, t0: np.ndarray, t1: np.ndarray) -> dict | None:
    if mask.sum() < 64:
        return None
    if t0.shape[:2] != t1.shape[:2]:
        t1 = cv2.resize(t1, (t0.shape[1], t0.shape[0]), interpolation=cv2.INTER_AREA)
    if mask.shape[:2] != t0.shape[:2]:
        mask = resize_mask(mask, t0.shape[:2])

    diff = t0t1_diff(t0, t1)
    inside = diff[mask]
    outside = diff[~mask]
    if outside.size < 64:
        return None

    inside_m = float(inside.mean())
    outside_m = float(outside.mean())
    contrast = outside_m - inside_m
    ratio = inside_m / max(outside_m, 1e-6)
    leak_thr = float(np.percentile(outside, 58))
    leak = float((inside > leak_thr).mean())
    p95_in = float(np.percentile(inside, 95))

    total = (
        3.0 * contrast
        + 1.2 * max(0.0, 1.0 - ratio)
        - 2.0 * leak
        - 0.5 * p95_in
    )
    return {
        "total": total,
        "inside_diff": inside_m,
        "outside_diff": outside_m,
        "contrast": contrast,
        "ratio": ratio,
        "leak": leak,
        "p95_inside": p95_in,
        "coverage": float(mask.mean()),
    }


def aggregate_scores(per_sample: list[dict]) -> dict:
    if not per_sample:
        return {"total": -999.0, "n_samples": 0}
    keys = ["total", "inside_diff", "outside_diff", "contrast", "ratio", "leak", "p95_inside"]
    out = {"n_samples": len(per_sample)}
    for k in keys:
        out[k] = float(np.median([x[k] for x in per_sample]))
    return out


def list_candidates(vehicle: str, camera: str) -> dict[str, Path]:
    key = f"{vehicle}_{camera}"
    cands: dict[str, Path] = {}
    for tag, suffix in (("current", ".png"), ("v2", "_v2.png"), ("v3", "_v3.png")):
        if tag == "current":
            p = APPROVED_DIR / f"{key}.png"
            if not p.is_file():
                p = ROOT / "masks" / f"{key}.png"
        else:
            p = VARIANTS_DIR / f"{key}{suffix}"
        if p.is_file():
            cands[tag] = p
    # dedupe identical paths
    seen: set[str] = set()
    uniq: dict[str, Path] = {}
    for tag, p in cands.items():
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        uniq[tag] = p
    return uniq


def variant_index(tag: str) -> int | None:
    if tag == "v2":
        return 2
    if tag == "v3":
        return 3
    return None


def is_border_warp_mask(mask: np.ndarray) -> bool:
    """v2=black letterbox — stable t0/t1 but not ego artifact."""
    h, w = mask.shape
    border = np.zeros((h, w), dtype=bool)
    border[: max(1, int(h * 0.10)), :] = True
    border[-max(1, int(h * 0.10)) :, :] = True
    border[:, : max(1, int(w * 0.10))] = True
    border[:, -max(1, int(w * 0.10)) :] = True
    if mask.sum() < 64:
        return False
    return float((mask & border).sum()) / float(mask.sum()) > 0.45


def pick_among_candidates(
    cand_scores: dict[str, dict],
    candidates: dict[str, Path],
    hw: tuple[int, int],
    min_v23_margin: float,
) -> tuple[str, str]:
    ranked = sorted(cand_scores.items(), key=lambda x: x[1]["total"], reverse=True)
    current_tag = "current" if "current" in cand_scores else ranked[0][0]

    pool = dict(cand_scores)
    v2_path = candidates.get("v2")
    if v2_path and "v2" in pool:
        m2 = resize_mask(load_mask_file(v2_path), hw)
        if is_border_warp_mask(m2):
            pool.pop("v2", None)

    if not pool:
        pool = cand_scores

    ranked_pool = sorted(pool.items(), key=lambda x: x[1]["total"], reverse=True)
    chosen = ranked_pool[0][0]
    reason = "best_t0t1_stability"

    v2s = pool.get("v2")
    v3s = pool.get("v3")
    if v2s and v3s:
        if abs(v2s["total"] - v3s["total"]) < min_v23_margin:
            v2p, v3p = candidates.get("v2"), candidates.get("v3")
            if v2p and v3p:
                m2 = resize_mask(load_mask_file(v2p), hw)
                m3 = resize_mask(load_mask_file(v3p), hw)
                chosen = "v2" if m2.mean() <= m3.mean() else "v3"
                reason = "v23_tie_tighter"
        elif v3s["total"] > v2s["total"] + min_v23_margin:
            chosen = "v3"
            reason = "v3_more_stable"
        elif v2s["total"] > v3s["total"] + min_v23_margin:
            chosen = "v2"
            reason = "v2_more_stable"

    if chosen == current_tag or candidates.get(chosen) == candidates.get(current_tag):
        return current_tag, "keep_current"
    return chosen, reason


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--update-variant-defaults", action="store_true")
    ap.add_argument("--min-v23-margin", type=float, default=0.03,
                    help="min total-score gain to switch v2<->v3")
    args = ap.parse_args()

    manifest = json.loads((ROOT / "manual_masks_manifest.json").read_text(encoding="utf-8"))
    by_key = {(x["vehicle"], x["camera"]): x for x in manifest["items"]}

    results: list[dict] = []
    variant_updates: dict[str, int] = {}
    changed = 0

    groups = sorted({(x["vehicle"], x["camera"]) for x in manifest["items"]})

    for vehicle, camera in groups:
        meta = by_key.get((vehicle, camera), {})
        status = meta.get("status", "")

        if (vehicle, camera) in ZERO_MASK_GROUPS or status == "needs_annotation":
            results.append({
                "vehicle": vehicle,
                "camera": camera,
                "chosen": "zero",
                "reason": "no_artifact",
                "n_samples": 0,
            })
            continue

        samples = collect_all_samples(args.dataset, vehicle, camera)
        if not samples:
            continue

        try:
            hw = canonical_hw(vehicle, camera)
        except FileNotFoundError:
            continue

        candidates = list_candidates(vehicle, camera)
        if not candidates:
            continue

        cand_scores: dict[str, dict] = {}
        for tag, path in candidates.items():
            base_mask = resize_mask(load_mask_file(path), hw)
            per_sample: list[dict] = []
            for sd in samples:
                t0 = load_rgb(sd / "input" / "t0" / f"{camera}.jpg")
                t1 = load_rgb(sd / "input" / "t1" / f"{camera}.jpg")
                s = score_sample(base_mask, t0, t1)
                if s:
                    per_sample.append(s)
            cand_scores[tag] = aggregate_scores(per_sample)

        chosen, reason = pick_among_candidates(
            cand_scores, candidates, hw, args.min_v23_margin,
        )
        ranked = sorted(cand_scores.items(), key=lambda x: x[1]["total"], reverse=True)
        current_tag = "current" if "current" in cand_scores else ranked[0][0]
        switched = chosen != current_tag and candidates.get(chosen) != candidates.get(current_tag)
        if switched:
            changed += 1

        vi = variant_index(chosen)
        if vi is not None:
            variant_updates[f"{vehicle}_{camera}"] = vi

        row = {
            "vehicle": vehicle,
            "camera": camera,
            "status": status,
            "n_samples": cand_scores[ranked[0][0]]["n_samples"],
            "chosen": chosen,
            "previous": current_tag,
            "changed": switched,
            "reason": reason,
            "ranked": [{"tag": t, **s} for t, s in ranked],
        }
        results.append(row)

        if args.apply and chosen != "zero":
            src = candidates[chosen]
            mask = resize_mask(load_mask_file(src), hw)
            save_binary_mask(mask, APPROVED_DIR / f"{vehicle}_{camera}.png")

    report = {
        "dataset": str(args.dataset.resolve()),
        "criterion": "low L1(t0,t1) inside mask vs outside; pick v2/v3 by median over all samples",
        "n_groups": len(results),
        "n_changed": changed,
        "applied": args.apply,
        "items": results,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.update_variant_defaults and variant_updates and VARIANTS_JSON.is_file():
        vm = json.loads(VARIANTS_JSON.read_text(encoding="utf-8"))
        dop = vm.setdefault("dop_default_variant", {})
        auto = set(vm.get("auto_two_variant", []))
        for key, idx in variant_updates.items():
            dop[key] = idx
            if key in auto or key.split("_", 1)[0] in {k.split("_")[0] for k in auto}:
                pass
        vm["t0t1_selected_defaults"] = dop
        VARIANTS_JSON.write_text(json.dumps(vm, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Groups: {len(results)}  changed: {changed}")
    print(f"Report: {REPORT_PATH}")
    if changed:
        print("\nSwitched (t0/t1 stability):")
        for r in results:
            if r.get("changed"):
                top = r["ranked"][0]
                print(
                    f"  {r['vehicle']}/{r['camera']:12} "
                    f"{r['previous']} -> {r['chosen']}  ({r['reason']})  "
                    f"n={r['n_samples']}  best_total={top['total']:.3f}"
                )
                if len(r["ranked"]) > 1:
                    parts = ", ".join(f"{x['tag']}={x['total']:.3f}" for x in r["ranked"])
                    print(f"    scores: {parts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
