"""Precompute far-static masks for all dataset samples."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
import sys
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


import numpy as np
from PIL import Image

from layered_parallax import get_lidar_depth_raw
from static_mask import (
    FAR_DEFAULT,
    MaskParams,
    build_static_mask,
    compute_flow_maps,
    far_or_no_depth_gate,
    lidar_hit_mask,
    load_best_params,
    z_ref_from_lidar,
)


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def process_sample(sample_dir: Path, params: MaskParams, out_dir: Path, save_png: bool) -> dict:
    meta = json.load(open(sample_dir / "meta.json"))
    cam = meta["target_camera"]
    sid = meta["sample_id"]

    img_t0 = load_rgb(sample_dir / "input" / "t0" / f"{cam}.jpg")
    img_t1 = load_rgb(sample_dir / "input" / "t1" / f"{cam}.jpg")
    flow_mag, comp_err, sigma, _ = compute_flow_maps(img_t0, img_t1)
    depth_raw = get_lidar_depth_raw(sample_dir, cam, "target")
    z_ref = z_ref_from_lidar(depth_raw)

    far_gate = far_or_no_depth_gate(
        depth_raw, z_ref, params.far_min_ratio, params.far_min_abs, params.include_no_depth,
    )
    static = build_static_mask(flow_mag, comp_err, None, sigma, params, z_ref, depth_raw)

    hit = lidar_hit_mask(depth_raw)
    static_lidar = static & hit
    static_no_lidar = static & ~hit
    mean_pred = ((img_t0.astype(np.float32) + img_t1.astype(np.float32)) * 0.5).astype(np.uint8)

    sample_out = out_dir / sid
    sample_out.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        sample_out / "static_far.npz",
        static=static.astype(np.uint8),
        far_gate=far_gate.astype(np.uint8),
        mean_pred=mean_pred,
        z_ref=np.float32(z_ref),
    )

    if save_png:
        Image.fromarray((static.astype(np.uint8) * 255)).save(sample_out / "static_far.png")

    row = {
        "sample_id": sid,
        "camera": cam,
        "z_ref": z_ref,
        "static_pct": float(static.mean()),
        "far_gate_pct": float(far_gate.mean()),
        "static_lidar_pct": float(static_lidar.mean()),
        "static_no_lidar_pct": float(static_no_lidar.mean()),
        "path": str(sample_out / "static_far.npz"),
    }
    (sample_out / "meta.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=Path,
                   default=Path(r"C:\Users\adel\Downloads\cv_dataset\final_dataset_v5_participants"))
    p.add_argument("--out-dir", type=Path,
                   default=Path(r"C:\Users\adel\Downloads\cv_dataset\precomputed_static_far"))
    p.add_argument("--splits", type=str, default="train,test")
    p.add_argument("--far-ratio", type=float, default=2.0)
    p.add_argument("--config", type=Path, default=Path("static_mask_tune_best.json"))
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--save-png", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    base = load_best_params(args.config)
    params = replace(base, far_min_ratio=args.far_ratio, name=f"far_r{args.far_ratio}_precomputed")
    if params.far_min_ratio <= 0:
        params = MaskParams.from_dict(FAR_DEFAULT)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "params": asdict(params),
        "dataset_dir": str(args.dataset_dir),
        "splits": {},
    }

    t0 = time.time()
    total_ok = 0
    total_skip = 0

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        split_dir = args.dataset_dir / split
        if not split_dir.is_dir():
            print(f"WARN: missing split {split_dir}", flush=True)
            continue

        samples = sorted(d for d in split_dir.iterdir() if d.is_dir())
        if args.limit > 0:
            samples = samples[: args.limit]

        rows = []
        for i, sd in enumerate(samples):
            meta_path = sd / "meta.json"
            if not meta_path.is_file():
                continue
            sid = json.load(open(meta_path))["sample_id"]
            out_npz = args.out_dir / sid / "static_far.npz"
            if args.resume and out_npz.is_file():
                total_skip += 1
                if (i + 1) % 100 == 0:
                    print(f"  [{split}] {i+1}/{len(samples)} skip={total_skip}", flush=True)
                continue

            try:
                rows.append(process_sample(sd, params, args.out_dir, args.save_png))
                total_ok += 1
            except Exception as e:
                print(f"FAIL {sid}: {e}", flush=True)
                continue

            if total_ok % 20 == 0:
                elapsed = time.time() - t0
                print(f"  [{split}] done {total_ok}  skip {total_skip}  {elapsed:.0f}s", flush=True)

        # merge with existing manifest rows on resume
        split_manifest = args.out_dir / f"manifest_{split}.json"
        existing = []
        if split_manifest.is_file():
            existing = json.loads(split_manifest.read_text(encoding="utf-8")).get("samples", [])
        by_id = {r["sample_id"]: r for r in existing}
        by_id.update({r["sample_id"]: r for r in rows})
        all_rows = [by_id[k] for k in sorted(by_id)]
        split_data = {
            "n_samples": len(all_rows),
            "avg_static_pct": float(np.mean([r["static_pct"] for r in all_rows])) if all_rows else 0.0,
            "avg_no_lidar_pct": float(np.mean([r["static_no_lidar_pct"] for r in all_rows])) if all_rows else 0.0,
            "samples": all_rows,
        }
        split_manifest.write_text(json.dumps(split_data, indent=2), encoding="utf-8")
        manifest["splits"][split] = {
            "manifest": str(split_manifest),
            "n_samples": split_data["n_samples"],
            "avg_static_pct": split_data["avg_static_pct"],
        }
        print(f"[{split}] +{len(rows)} new  total={len(all_rows)}  avg static={100*split_data['avg_static_pct']:.1f}%", flush=True)

    manifest["total_computed_this_run"] = total_ok
    manifest["total_skipped"] = total_skip
    manifest["elapsed_sec"] = time.time() - t0
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nDone: computed {total_ok}  skipped {total_skip}  elapsed {manifest['elapsed_sec']:.0f}s")
    print(f"Output: {args.out_dir.resolve()}")


def load_precomputed(sample_id: str, out_dir: Path | None = None) -> dict:
    """Load precomputed static_far.npz for a sample."""
    root = out_dir or Path(r"C:\Users\adel\Downloads\cv_dataset\precomputed_static_far")
    path = root / sample_id / "static_far.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path)
    return {
        "static": data["static"].astype(bool),
        "far_gate": data["far_gate"].astype(bool),
        "mean_pred": data["mean_pred"],
        "z_ref": float(data["z_ref"]),
    }


if __name__ == "__main__":
    main()
