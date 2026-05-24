"""
Consistent ego edges: find edges that repeat across scenes (no fill yet).

Per (vehicle, camera), 4-5 diverse target frames:
  1. bilateral + morph gradient -> sharp edge map per scene
  2. edge_freq = fraction of scenes with edge at pixel
  3. threshold -> keep only high-consensus edges
  4. morph close + filter CC -> cohesive edge curves

Usage:
  python test_ego_consistent_edges.py
  python test_ego_consistent_edges.py --vehicle hilma --camera right_fwd --max-scenes 5
  python test_ego_consistent_edges.py --vehicle hilma --cameras right_fwd front --freq-thr 0.75
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent
DEFAULT_DATASET = Path(r"C:/Users/adel/Downloads/cv_dataset/final_dataset_v5_participants/train")
DEFAULT_VIZ = REPO / "methods_gallery/_ego_artifacts_v2_test/consistent_edges"

SAMPLE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_([a-zA-Z]+)_\d+__\d{3}$"
)


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def diversify_paths(paths: list[Path]) -> list[Path]:
    by_trip: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        by_trip[p.name.rsplit("__", 1)[0]].append(p)
    for t in by_trip:
        by_trip[t].sort(key=lambda x: x.name)
    trips = sorted(by_trip.keys())
    out: list[Path] = []
    pos = 0
    while len(out) < sum(len(v) for v in by_trip.values()):
        added = False
        for t in trips:
            if pos < len(by_trip[t]):
                out.append(by_trip[t][pos])
                added = True
        if not added:
            break
        pos += 1
    return out


def load_stack(
    dataset: Path,
    vehicle: str,
    camera: str,
    max_scenes: int,
) -> tuple[list[Path], np.ndarray]:
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

    paths = diversify_paths(paths)[:max_scenes]
    imgs = [load_rgb(p / "target" / f"{camera}.jpg") for p in paths]
    h = max(i.shape[0] for i in imgs)
    w = max(i.shape[1] for i in imgs)
    aligned = []
    for img in imgs:
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        aligned.append(img)
    return paths, np.stack(aligned)


def sharp_edge_magnitude(img: np.ndarray, y0: int, grad_pct: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (binary edge, gradient magnitude) for bottom ROI."""
    roi = img[y0:]
    smooth = cv2.bilateralFilter(roi, 9, 80, 80)
    gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mg = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, k)
    mg = cv2.GaussianBlur(mg, (3, 3), 0)
    thr = max(10.0, float(np.percentile(mg, grad_pct)))
    binary = np.zeros(img.shape[:2], dtype=bool)
    mag = np.zeros(img.shape[:2], dtype=np.float32)
    binary[y0:] = mg >= thr
    mag[y0:] = mg
    return binary, mag


def compute_consistent_edges(
    stack: np.ndarray,
    bottom_frac: float = 0.55,
    grad_pct: float = 88.0,
    freq_thr: float = 0.75,
    min_votes: int | None = None,
    close_k: int = 5,
    close_iter: int = 2,
    close_k_h: int = 21,
    close_k_v: int = 3,
    min_cc_pixels: int = 40,
    bottom_touch_rows: int = 8,
    keep_bottom_only: bool = True,
    min_comp_freq: float = 0.65,
) -> dict:
    n = len(stack)
    h, w = stack.shape[1:3]
    y0 = int(h * (1.0 - bottom_frac))

    binaries: list[np.ndarray] = []
    mags: list[np.ndarray] = []
    for img in stack:
        b, m = sharp_edge_magnitude(img, y0, grad_pct)
        binaries.append(b)
        mags.append(m)

    edge_stack = np.stack(binaries, axis=0)
    vote_count = edge_stack.sum(axis=0).astype(np.int32)
    edge_freq = vote_count.astype(np.float32) / n

    if min_votes is None:
        min_votes = max(1, int(np.ceil(freq_thr * n)))
    raw = vote_count >= min_votes

    # soften freq map for viz
    freq_vis = edge_freq.copy()
    freq_vis[y0:] = cv2.GaussianBlur(freq_vis[y0:].astype(np.float32), (5, 5), 0)

    # threshold on float freq too (slightly softer than hard count)
    thr_mask = edge_freq >= freq_thr

    # isotropic close, then horizontal close (connect hood ridge across x)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k_h, close_k_v))
    closed = cv2.morphologyEx(thr_mask.astype(np.uint8), cv2.MORPH_CLOSE, k, iterations=close_iter)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kh, iterations=1).astype(bool)

    # drop speckles, optionally keep only CC touching bottom
    n_cc, labels = cv2.connectedComponents(closed.astype(np.uint8))
    cohesive = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for lab in range(1, n_cc):
        comp = labels == lab
        area = int(comp.sum())
        if area < min_cc_pixels:
            continue
        touches_bottom = bool(comp[h - bottom_touch_rows :].any())
        if keep_bottom_only and not touches_bottom:
            continue
        mean_freq = float(edge_freq[comp].mean())
        if mean_freq < min_comp_freq:
            continue
        max_freq = float(edge_freq[comp].max())
        components.append({
            "label": lab,
            "area": area,
            "mean_freq": mean_freq,
            "max_freq": max_freq,
            "touches_bottom": touches_bottom,
        })
        cohesive |= comp

    # weighted edge: freq * mean gradient (stronger real edges)
    mean_mag = np.mean(np.stack(mags, axis=0), axis=0)
    weighted = edge_freq * (mean_mag / max(mean_mag.max(), 1e-6))
    w_thr = np.percentile(weighted[weighted > 0], 75) if weighted.any() else 0.0
    weighted_edges = (weighted >= w_thr) & (edge_freq >= freq_thr)
    weighted_closed = cv2.morphologyEx(
        weighted_edges.astype(np.uint8), cv2.MORPH_CLOSE, kh, iterations=1,
    ).astype(bool)

    # sweep thresholds for tuning viz
    sweep: dict[float, np.ndarray] = {}
    for t in (0.5, 0.6, 0.75, 0.8):
        mv = max(1, int(np.ceil(t * n)))
        sm = vote_count >= mv
        sc = cv2.morphologyEx(sm.astype(np.uint8), cv2.MORPH_CLOSE, kh, iterations=1).astype(bool)
        sweep[t] = sc

    return {
        "y0": y0,
        "binaries": binaries,
        "edge_freq": freq_vis,
        "vote_count": vote_count,
        "raw_consensus": raw,
        "thr_mask": thr_mask,
        "closed": closed,
        "cohesive": cohesive,
        "weighted_closed": weighted_closed,
        "sweep": sweep,
        "components": components,
        "n_scenes": n,
        "min_votes": min_votes,
    }


def overlay_edges(rgb: np.ndarray, edges: np.ndarray, color=(255, 80, 80), alpha: float = 0.85) -> np.ndarray:
    out = rgb.copy()
    if not edges.any():
        return out
    tint = np.zeros_like(out)
    tint[..., 0] = color[0]
    tint[..., 1] = color[1]
    tint[..., 2] = color[2]
    out[edges] = (alpha * tint[edges] + (1.0 - alpha) * out[edges]).astype(np.uint8)
    return out


def save_figure(
    vehicle: str,
    camera: str,
    paths: list[Path],
    stack: np.ndarray,
    result: dict,
    out_path: Path,
    freq_thr: float,
):
    n = result["n_scenes"]
    ref = stack[0]
    y0 = result["y0"]

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, max(n, 4), hspace=0.25, wspace=0.08)

    # row 0: per-scene edges
    for i in range(n):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(overlay_edges(stack[i], result["binaries"][i]))
        trip = paths[i].name.rsplit("__", 1)[0][-20:]
        ax.set_title(f"scene {i+1}\n…{trip}", fontsize=8)
        ax.axis("off")

    # row 1: freq + raw thr + closed + cohesive
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(result["edge_freq"], cmap="magma", vmin=0, vmax=1)
    ax.set_title(f"edge vote freq\nneed >={result['min_votes']}/{n}")
    ax.axhline(y0, color="cyan", lw=0.8, alpha=0.7)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    panels = [
        ("raw consensus", result["raw_consensus"]),
        (f"freq>={freq_thr:.2f}", result["thr_mask"]),
        ("morph close", result["closed"]),
        ("cohesive CC", result["cohesive"]),
    ]
    for j, (title, edges) in enumerate(panels):
        ax = fig.add_subplot(gs[1, j + 1])
        ax.imshow(overlay_edges(ref, edges, color=(255, 100, 100)))
        ax.set_title(f"{title}\n{100 * edges.mean():.2f}% px")
        ax.axis("off")

    # row 2: cohesive + threshold sweep + stats
    ax = fig.add_subplot(gs[2, 0:2])
    ax.imshow(overlay_edges(ref, result["cohesive"], color=(255, 60, 60)))
    ax.set_title("cohesive edges on ref (no fill)")
    ax.axis("off")

    ax = fig.add_subplot(gs[2, 2:4])
    ax.imshow(ref)
    colors = [(255, 80, 80), (255, 160, 80), (255, 80, 160), (200, 80, 255)]
    for (t, edges), col in zip(sorted(result["sweep"].items()), colors):
        layer = np.zeros((*edges.shape, 4), dtype=np.float32)
        layer[edges, 0] = col[0] / 255
        layer[edges, 1] = col[1] / 255
        layer[edges, 2] = col[2] / 255
        layer[edges, 3] = 0.45
        ax.imshow(layer)
    ax.set_title("thr sweep (horiz close)")
    ax.axis("off")

    ax = fig.add_subplot(gs[2, 4] if gs.ncols > 4 else gs[2, -1])
    ax.axis("off")
    comp_lines = [
        f"  #{c['label']} area={c['area']} mean_f={c['mean_freq']:.2f} bottom={c['touches_bottom']}"
        for c in sorted(result["components"], key=lambda x: -x["area"])[:8]
    ]
    lines = [
        f"{vehicle} / {camera}",
        f"scenes: {n}  freq_thr: {freq_thr:.2f}  min_votes: {result['min_votes']}",
        f"cohesive CC: {len(result['components'])}  px={100 * result['cohesive'].mean():.2f}%",
        "",
        "components (largest):",
        *comp_lines,
        "",
        "thr sweep 0.5/0.6/0.75/0.8",
    ]
    ax.text(
        0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92),
    )

    fig.suptitle(
        f"Consistent ego edges — {vehicle}/{camera}  (bottom ROI y>={y0})",
        fontsize=13,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def process_camera(
    dataset: Path,
    viz_dir: Path,
    vehicle: str,
    camera: str,
    max_scenes: int,
    freq_thr: float,
    bottom_frac: float,
    grad_pct: float,
    close_k: int,
    min_cc_pixels: int,
) -> dict:
    paths, stack = load_stack(dataset, vehicle, camera, max_scenes)
    if len(stack) < 2:
        return {"vehicle": vehicle, "camera": camera, "error": "not enough scenes"}

    result = compute_consistent_edges(
        stack,
        bottom_frac=bottom_frac,
        grad_pct=grad_pct,
        freq_thr=freq_thr,
        close_k=close_k,
        min_cc_pixels=min_cc_pixels,
    )

    out_path = viz_dir / f"{vehicle}_{camera}_edges.png"
    save_figure(vehicle, camera, paths, stack, result, out_path, freq_thr)

    return {
        "vehicle": vehicle,
        "camera": camera,
        "n_scenes": len(stack),
        "paths": [str(p) for p in paths],
        "freq_thr": freq_thr,
        "min_votes": result["min_votes"],
        "n_components": len(result["components"]),
        "cohesive_coverage": float(result["cohesive"].mean()),
        "viz": str(out_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--viz-dir", type=Path, default=DEFAULT_VIZ)
    ap.add_argument("--vehicle", default="hilma")
    ap.add_argument("--camera", default="right_fwd")
    ap.add_argument("--cameras", nargs="*", help="multiple cameras, e.g. right_fwd front")
    ap.add_argument("--max-scenes", type=int, default=5)
    ap.add_argument("--freq-thr", type=float, default=0.75, help="min fraction of scenes with edge")
    ap.add_argument("--bottom-frac", type=float, default=0.55)
    ap.add_argument("--grad-pct", type=float, default=88.0)
    ap.add_argument("--close-k", type=int, default=5)
    ap.add_argument("--min-cc", type=int, default=40)
    args = ap.parse_args()

    cameras = args.cameras if args.cameras else [args.camera]
    results = []
    for cam in cameras:
        r = process_camera(
            args.dataset, args.viz_dir, args.vehicle, cam,
            args.max_scenes, args.freq_thr, args.bottom_frac,
            args.grad_pct, args.close_k, args.min_cc,
        )
        results.append(r)
        if "error" in r:
            print(f"  {cam}: {r['error']}")
        else:
            print(
                f"  {cam:10s}  n={r['n_scenes']}  votes>={r['min_votes']}  "
                f"CC={r['n_components']}  cov={100 * r['cohesive_coverage']:.2f}%"
            )

    summary = args.viz_dir / f"{args.vehicle}_edges_summary.json"
    summary.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nViz: {args.viz_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
