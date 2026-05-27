"""
Incremental ego/camera artifact masks: few samples -> refine until stable.

Per (vehicle, camera):
  1. Start with 2 diverse scenes
  2. Add 2 more, re-aggregate
  3. Stop when mask IoU vs previous step >= converge_iou (default 0.92)
     or max_samples reached

Usage:
  python build_ego_artifact_masks.py
  python build_ego_artifact_masks.py --init 2 --batch 2 --max-samples 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from layered_parallax import get_lidar_depth
from static_mask import MaskParams, build_static_mask, compute_flow_maps, mask_metrics

DEFAULT_DATASET = Path(r"C:/Users/adel/Downloads/cv_dataset/final_dataset_v5_participants/train")
DEFAULT_OUT = Path(r"C:/Users/adel/Downloads/cv_dataset/ego_artifact_masks")
DEFAULT_VIZ = REPO / "methods_gallery/_ego_artifacts"
CACHE_DIR = DEFAULT_OUT / "_cache_instant"

SAMPLE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_([a-zA-Z]+)_\d+__\d{3}$"
)

INSTANT_PARAMS = MaskParams(
    name="instant_near_ok",
    flow_t=8.0,
    comp_t=20.0,
    far_min_ratio=0.0,
    include_no_depth=True,
)


@dataclass
class SampleRec:
    path: Path
    vehicle: str
    camera: str
    trip: str  # sample_id without __NNN


@dataclass
class GroupResult:
    vehicle: str
    camera: str
    n_used: int
    n_available: int
    converged: bool
    steps: list[dict] = field(default_factory=list)
    freq: np.ndarray | None = None
    artifact: np.ndarray | None = None


def parse_sample(sample_dir: Path) -> SampleRec | None:
    m = SAMPLE_RE.match(sample_dir.name)
    if not m:
        return None
    meta_path = sample_dir / "meta.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return SampleRec(
        path=sample_dir,
        vehicle=m.group(1),
        camera=meta["target_camera"],
        trip=sample_dir.name.rsplit("__", 1)[0],
    )


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def cache_path(rec: SampleRec) -> Path:
    return CACHE_DIR / rec.vehicle / rec.camera / f"{rec.path.name}.npy"


def _needs_lidar_depth(params: MaskParams) -> bool:
    return params.uses_far_gate or params.use_depth_scale or params.use_fdn or params.require_depth


def compute_instant_static(rec: SampleRec, scale: float = 0.5, use_cache: bool = True) -> np.ndarray:
    cp = cache_path(rec)
    if use_cache and cp.is_file():
        return np.load(cp).astype(bool)

    cam = rec.camera
    sd = rec.path
    img_t0 = load_rgb(sd / "input" / "t0" / f"{cam}.jpg")
    img_t1 = load_rgb(sd / "input" / "t1" / f"{cam}.jpg")
    h0, w0 = img_t0.shape[:2]

    if scale != 1.0:
        nw, nh = int(w0 * scale), int(h0 * scale)
        img_t0 = cv2.resize(img_t0, (nw, nh), interpolation=cv2.INTER_AREA)
        img_t1 = cv2.resize(img_t1, (nw, nh), interpolation=cv2.INTER_AREA)

    flow_mag, comp_err, sigma_map, _ = compute_flow_maps(img_t0, img_t1)
    depth = None
    z_ref = 10.0
    if _needs_lidar_depth(INSTANT_PARAMS):
        depth = get_lidar_depth(sd, cam, "target")
        if scale != 1.0:
            depth = cv2.resize(depth, (img_t0.shape[1], img_t0.shape[0]), interpolation=cv2.INTER_NEAREST)
        valid = np.isfinite(depth) & (depth > 0)
        z_ref = float(np.median(depth[valid])) if np.any(valid) else 10.0
    mask = build_static_mask(flow_mag, comp_err, depth, sigma_map, INSTANT_PARAMS, z_ref)

    if scale != 1.0:
        mask = cv2.resize(mask.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)

    if use_cache:
        cp.parent.mkdir(parents=True, exist_ok=True)
        np.save(cp, mask.astype(np.uint8))
    return mask


def diversify_order(recs: list[SampleRec]) -> list[SampleRec]:
    """Round-robin across trips so first picks are maximally different scenes."""
    by_trip: dict[str, list[SampleRec]] = defaultdict(list)
    for r in recs:
        by_trip[r.trip].append(r)
    for t in by_trip:
        by_trip[t].sort(key=lambda x: x.path.name)
    trips = sorted(by_trip.keys())
    out: list[SampleRec] = []
    pos = 0
    while len(out) < len(recs):
        added = False
        for t in trips:
            if pos < len(by_trip[t]):
                out.append(by_trip[t][pos])
                added = True
        if not added:
            break
        pos += 1
    return out


def aggregate(masks: list[np.ndarray], thresh: float) -> tuple[np.ndarray, np.ndarray]:
    if not masks:
        raise ValueError("aggregate: empty mask list")
    h = max(m.shape[0] for m in masks)
    w = max(m.shape[1] for m in masks)
    aligned: list[np.ndarray] = []
    for m in masks:
        if m.shape[:2] != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        aligned.append(m)
    freq = np.mean(np.stack(aligned, axis=0), axis=0)
    return freq, freq >= thresh


def bottom_fraction(mask: np.ndarray, frac: float = 0.5) -> float:
    h = mask.shape[0]
    y0 = int(h * (1.0 - frac))
    if not np.any(mask):
        return 0.0
    return float(mask[y0:].sum() / mask.sum())


def refine_group(
    recs: list[SampleRec],
    thresh: float,
    init: int,
    batch: int,
    max_samples: int,
    converge_iou: float,
    scale: float,
) -> GroupResult:
    ordered = diversify_order(recs)
    n_avail = len(ordered)
    if n_avail < 2:
        return GroupResult(ordered[0].vehicle, ordered[0].camera, 0, n_avail, False)

    masks: list[np.ndarray] = []
    steps: list[dict] = []
    prev_artifact: np.ndarray | None = None
    converged = False
    idx = 0

    def ingest(up_to: int):
        nonlocal idx, prev_artifact, converged
        while idx < up_to and idx < n_avail and idx < max_samples:
            masks.append(compute_instant_static(ordered[idx], scale=scale))
            idx += 1
        freq, artifact = aggregate(masks, thresh)
        step_iou = 1.0 if prev_artifact is None else mask_metrics(artifact, prev_artifact)["iou"]
        steps.append({
            "n": len(masks),
            "coverage": float(artifact.mean()),
            "step_iou": float(step_iou),
            "mean_freq_in": float(freq[artifact].mean()) if artifact.any() else 0.0,
        })
        if prev_artifact is not None and step_iou >= converge_iou and len(masks) >= init:
            converged = True
        prev_artifact = artifact.copy()
        return freq, artifact

    ingest(init)
    while not converged and idx < min(n_avail, max_samples):
        ingest(idx + batch)

    freq, artifact = aggregate(masks, thresh)
    return GroupResult(
        vehicle=ordered[0].vehicle,
        camera=ordered[0].camera,
        n_used=len(masks),
        n_available=n_avail,
        converged=converged,
        steps=steps,
        freq=freq,
        artifact=artifact,
    )


def save_group(out_dir: Path, gr: GroupResult, thresh: float):
    d = out_dir / gr.vehicle
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{gr.camera}_freq.npy", gr.freq.astype(np.float32))
    np.save(d / f"{gr.camera}_mask.npy", gr.artifact.astype(np.uint8))
    meta = {
        "vehicle": gr.vehicle,
        "camera": gr.camera,
        "n_used": gr.n_used,
        "n_available": gr.n_available,
        "converged": gr.converged,
        "thresh": thresh,
        "coverage": float(gr.artifact.mean()),
        "bottom_frac": bottom_fraction(gr.artifact),
        "steps": gr.steps,
    }
    (d / f"{gr.camera}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def make_summary_figure(recs: list[SampleRec], gr: GroupResult, global_stats: dict, out_path: Path):
    cam = gr.camera
    ordered = diversify_order(recs)[: gr.n_used]
    picks = [ordered[i] for i in np.linspace(0, len(ordered) - 1, min(4, len(ordered)), dtype=int)]

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 4, hspace=0.28, wspace=0.08)

    for col, rec in enumerate(picks):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(load_rgb(rec.path / "target" / f"{cam}.jpg"))
        ax.set_title(f"scene {col+1}", fontsize=9)
        ax.axis("off")

    gt0 = load_rgb(picks[0].path / "target" / f"{cam}.jpg")
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(gt0)
    ax.set_title("GT example", fontsize=9)
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(gr.freq, cmap="magma", vmin=0, vmax=1)
    ax.set_title(f"freq ({gr.n_used}/{gr.n_available} scenes)", fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(gr.artifact.astype(float), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"artifact {100*gr.artifact.mean():.1f}%", fontsize=9)
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 3])
    over = gt0.copy()
    red = np.zeros_like(over)
    red[..., 0] = 255
    m = gr.artifact
    over[m] = (0.55 * over[m] + 0.45 * red[m]).astype(np.uint8)
    ax.imshow(over)
    ax.set_title("overlay", fontsize=9)
    ax.axis("off")

    inst = compute_instant_static(picks[0], scale=0.5)
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(inst.astype(float), cmap="gray")
    ax.set_title(f"single static ({100*inst.mean():.1f}%)", fontsize=9)
    ax.axis("off")

    ax = fig.add_subplot(gs[2, 1])
    both = np.zeros((*gr.artifact.shape, 3), dtype=np.uint8)
    both[inst] = [80, 180, 255]
    both[gr.artifact] = [255, 80, 80]
    both[inst & gr.artifact] = [255, 255, 80]
    ax.imshow(both)
    ax.set_title("cyan=single red=agg", fontsize=8)
    ax.axis("off")

    ax = fig.add_subplot(gs[2, 2:])
    ax.axis("off")
    ex_iou = mask_metrics(gr.artifact, inst)["iou"]
    lines = [
        f"{gr.vehicle} / {cam}  converged={gr.converged}  used {gr.n_used}/{gr.n_available}",
        f"thresh={global_stats['thresh']:.2f}  coverage={100*gr.artifact.mean():.2f}%  "
        f"bottom50={100*bottom_fraction(gr.artifact):.0f}%",
        f"example IoU agg vs 1st scene: {ex_iou:.3f}",
        f"refinement steps: " + "  ".join(
            f"n={s['n']} cov={100*s['coverage']:.1f}% ΔIoU={s['step_iou']:.2f}" for s in gr.steps
        ),
        "",
        f"ALL GROUPS: {global_stats['n_groups']}  "
        f"total flow runs: {global_stats['total_computed']} (not {global_stats['n_samples']}!)",
        f"converged: {global_stats['n_converged']}/{global_stats['n_groups']}  "
        f"median used: {global_stats['median_used']:.0f}  median coverage: {100*global_stats['median_cov']:.1f}%",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=10, family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    fig.suptitle(
        f"Ego artifact mask (incremental) — {gr.vehicle}/{cam}\n"
        f"flow<{INSTANT_PARAMS.flow_t} comp<{INSTANT_PARAMS.comp_t}  stop @ step IoU≥{global_stats['converge_iou']:.2f}",
        fontsize=12,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--viz-dir", type=Path, default=DEFAULT_VIZ)
    ap.add_argument("--thresh", type=float, default=0.85)
    ap.add_argument("--init", type=int, default=2, help="initial scenes per group")
    ap.add_argument("--batch", type=int, default=2, help="add this many per refine step")
    ap.add_argument("--max-samples", type=int, default=12, help="hard cap per group")
    ap.add_argument("--converge-iou", type=float, default=0.92, help="stop when mask IoU between steps >= this")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--viz-group", default="hilma:right_fwd")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--skip-done", action="store_true", help="skip groups that already have _meta.json")
    args = ap.parse_args()

    if args.no_cache and CACHE_DIR.exists():
        pass  # keep cache unless user clears manually

    samples: list[SampleRec] = []
    for d in sorted(args.dataset.iterdir()):
        if d.is_dir():
            r = parse_sample(d)
            if r:
                samples.append(r)

    groups: dict[tuple[str, str], list[SampleRec]] = defaultdict(list)
    for r in samples:
        groups[(r.vehicle, r.camera)].append(r)

    print(
        f"Samples {len(samples)}  groups {len(groups)}  "
        f"strategy: init={args.init} batch={args.batch} max={args.max_samples} stop_iou={args.converge_iou}",
        flush=True,
    )

    results: list[GroupResult] = []
    metas: list[dict] = []
    total_computed = 0

    for key in sorted(groups.keys()):
        recs = groups[key]
        if len(recs) < 2:
            continue
        meta_path = args.out_dir / key[0] / f"{key[1]}_meta.json"
        if args.skip_done and meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            metas.append(meta)
            total_computed += meta["n_used"]
            results.append(
                GroupResult(
                    vehicle=key[0],
                    camera=key[1],
                    n_used=meta["n_used"],
                    n_available=meta["n_available"],
                    converged=meta["converged"],
                    steps=meta.get("steps", []),
                    freq=np.load(args.out_dir / key[0] / f"{key[1]}_freq.npy"),
                    artifact=np.load(args.out_dir / key[0] / f"{key[1]}_mask.npy").astype(bool),
                )
            )
            print(f"  {key[0]:8s} {key[1]:10s}  skip (done)", flush=True)
            continue
        gr = refine_group(
            recs, args.thresh, args.init, args.batch,
            args.max_samples, args.converge_iou, args.scale,
        )
        total_computed += gr.n_used
        if gr.freq is None:
            continue
        meta = save_group(args.out_dir, gr, args.thresh)
        results.append(gr)
        metas.append(meta)
        status = "OK" if gr.converged else "cap"
        print(
            f"  {gr.vehicle:8s} {gr.camera:10s}  {gr.n_used:2d}/{gr.n_available:2d}  {status}  "
            f"cov={100*gr.artifact.mean():.1f}%",
            flush=True,
        )

    if not results:
        print("No groups processed")
        return 1

    covs = [m["coverage"] for m in metas]
    used = [m["n_used"] for m in metas]
    global_stats = {
        "n_samples": len(samples),
        "n_groups": len(results),
        "total_computed": total_computed,
        "n_converged": sum(1 for r in results if r.converged),
        "median_used": float(np.median(used)),
        "mean_used": float(np.mean(used)),
        "median_cov": float(np.median(covs)),
        "thresh": args.thresh,
        "converge_iou": args.converge_iou,
        "init": args.init,
        "batch": args.batch,
        "max_samples": args.max_samples,
        "groups": metas,
    }

    args.viz_dir.mkdir(parents=True, exist_ok=True)
    (args.viz_dir / "stats.json").write_text(
        json.dumps(global_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # bar: samples used vs available
    top = sorted(metas, key=lambda x: -x["n_available"])[:20]
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [f"{g['vehicle'][:6]}/{g['camera']}" for g in top]
    x = np.arange(len(top))
    ax.barh(x, [g["n_available"] for g in top], color="lightgray", label="available")
    ax.barh(x, [g["n_used"] for g in top], color="steelblue", label="used (incremental)")
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.legend()
    ax.set_title("Scenes used vs available (incremental early stop)")
    fig.tight_layout()
    fig.savefig(args.viz_dir / "stats_bars.png", dpi=130)
    plt.close(fig)

    vname, cname = args.viz_group.split(":", 1)
    viz_key = (vname, cname)
    if viz_key not in groups:
        viz_key = max(groups.keys(), key=lambda k: len(groups[k]))
    gr = next(r for r in results if r.vehicle == viz_key[0] and r.camera == viz_key[1])
    make_summary_figure(groups[viz_key], gr, global_stats, args.viz_dir / "summary.png")

    print(f"\nDone: {total_computed} flow runs (saved vs {len(samples)} full scan)")
    print(f"Converged {global_stats['n_converged']}/{global_stats['n_groups']}  "
          f"median used {global_stats['median_used']:.0f} scenes/group")
    print(f"Output: {args.out_dir}  viz: {args.viz_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
