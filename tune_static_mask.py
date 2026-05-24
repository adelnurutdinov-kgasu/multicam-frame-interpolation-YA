"""Tune far-only static masks (distant + no-depth) vs well-predicted regions."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from layered_parallax import get_lidar_depth
from static_mask import (
    FAR_DEFAULT,
    MaskParams,
    build_static_mask,
    compute_flow_maps,
    far_or_no_depth_gate,
    mask_metrics,
)


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_sample(sample_dir: Path, err_pct: float):
    meta = json.load(open(sample_dir / "meta.json"))
    cam = meta["target_camera"]
    img_t0 = load_rgb(sample_dir / "input" / "t0" / f"{cam}.jpg")
    img_t1 = load_rgb(sample_dir / "input" / "t1" / f"{cam}.jpg")
    gt = load_rgb(sample_dir / "target" / f"{cam}.jpg")
    mean_pred = ((img_t0.astype(np.float32) + img_t1.astype(np.float32)) * 0.5).astype(np.uint8)
    err = np.mean(np.abs(mean_pred.astype(np.float32) - gt.astype(np.float32)), axis=2)
    good = err <= np.percentile(err, err_pct)
    flow_mag, comp_err, sigma, _ = compute_flow_maps(img_t0, img_t1)
    depth = get_lidar_depth(sample_dir, cam, "target")
    z_ref = float(np.median(depth[np.isfinite(depth) & (depth > 0)]))
    return dict(meta=meta, img_t0=img_t0, gt=gt, good=good, flow_mag=flow_mag,
                comp_err=comp_err, sigma=sigma, depth=depth, z_ref=z_ref, mean_pred=mean_pred)


def good_far(good: np.ndarray, depth: np.ndarray, z_ref: float, p: MaskParams) -> np.ndarray:
    gate = far_or_no_depth_gate(depth, z_ref, p.far_min_ratio, p.far_min_abs, p.include_no_depth)
    return good & gate


def grid_masks() -> list[MaskParams]:
    configs = []
    for ratio in [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]:
        for no_d in [True, False]:
            for ct in [12.0, 15.0, 18.0, 20.0, 25.0]:
                for ft in [4.0, 5.0, 6.0, 8.0]:
                    nd = "nod" if no_d else "lid"
                    configs.append(MaskParams(
                        f"far_r{ratio}_{nd}_f{ft}_c{ct}",
                        flow_t=ft, comp_t=ct,
                        far_min_ratio=ratio, include_no_depth=no_d,
                    ))
                configs.append(MaskParams(
                    f"far_r{ratio}_{nd}_comp{ct}",
                    comp_t=ct, use_flow=False,
                    far_min_ratio=ratio, include_no_depth=no_d,
                ))

    for ratio in [0.8, 1.0, 1.2, 1.5]:
        for ct in [15.0, 18.0, 20.0]:
            for fdn in [60.0, 80.0, 100.0, 120.0]:
                configs.append(MaskParams(
                    f"far_r{ratio}_fdn{fdn}_c{ct}",
                    fdn_t=fdn, comp_t=ct, use_fdn=True, use_flow=False,
                    far_min_ratio=ratio, include_no_depth=True,
                ))

    for abs_z in [8.0, 10.0, 12.0, 15.0, 20.0]:
        for ct in [15.0, 18.0, 20.0]:
            configs.append(MaskParams(
                f"far_abs{abs_z}_c{ct}",
                comp_t=ct, use_flow=False,
                far_min_abs=abs_z, include_no_depth=True,
            ))

    return configs


def tune(caches: list[dict], configs: list[MaskParams]) -> tuple[MaskParams, dict]:
    best_f1 = -1.0
    best_p = configs[0]
    scores = {}
    for p in configs:
        ms = {"iou": [], "f1": [], "precision": [], "recall": [], "coverage": []}
        for c in caches:
            m = build_static_mask(c["flow_mag"], c["comp_err"], c["depth"], c["sigma"], p, c["z_ref"])
            gf = good_far(c["good"], c["depth"], c["z_ref"], p)
            met = mask_metrics(m, gf)
            for k in ms:
                ms[k].append(met[k])
        avg = {k: float(np.mean(v)) for k, v in ms.items()}
        scores[p.name] = avg
        if avg["f1"] > best_f1:
            best_f1 = avg["f1"]
            best_p = p
    return best_p, scores


def save_comparison(sample: dict, masks: dict[str, np.ndarray], good: np.ndarray,
                    depth: np.ndarray, z_ref: float, p: MaskParams, out: Path, title: str):
    gate = far_or_no_depth_gate(depth, z_ref, p.far_min_ratio, p.far_min_abs, p.include_no_depth)
    good_f = good & gate
    n = len(masks) + 3
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 8))
    axes = axes.flat
    axes[0].imshow(sample["img_t0"])
    axes[0].set_title("t0")
    axes[0].axis("off")
    axes[1].imshow(gate.astype(float), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"far|no-depth\n({100*gate.mean():.0f}%)")
    axes[1].axis("off")
    axes[2].imshow(good_f.astype(float), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"WELL pred far\n({100*good_f.mean():.0f}%)")
    axes[2].axis("off")
    for i, (name, m) in enumerate(masks.items(), start=3):
        met = mask_metrics(m, good_f)
        axes[i].imshow(m.astype(float), cmap="gray", vmin=0, vmax=1)
        axes[i].set_title(f"{name}\nIoU={met['iou']:.2f} F1={met['f1']:.2f}\n{100*m.mean():.0f}% px")
        axes[i].axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=Path,
                   default=Path(r"C:\Users\adel\Downloads\cv_dataset\final_dataset_v5_participants"))
    p.add_argument("--num-samples", type=int, default=15)
    p.add_argument("--err-percentile", type=float, default=30.0)
    p.add_argument("--out", type=Path, default=Path("static_mask_tune_best.json"))
    p.add_argument("--viz-dir", type=Path, default=Path("methods_gallery/_static_masks"))
    p.add_argument("--viz-samples", type=int, default=3)
    args = p.parse_args()

    train = args.dataset_dir / "train"
    all_s = sorted(d for d in train.iterdir() if d.is_dir())
    step = max(1, len(all_s) // args.num_samples)
    sample_dirs = [all_s[i * step] for i in range(args.num_samples)]

    print(f"Loading {len(sample_dirs)} samples (far-only tuning)...", flush=True)
    caches = []
    for i, sd in enumerate(sample_dirs):
        caches.append(load_sample(sd, args.err_percentile))
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(sample_dirs)}", flush=True)

    configs = grid_masks()
    print(f"Tuning {len(configs)} far-only configs vs WELL pred on far|no-depth...", flush=True)
    best, all_scores = tune(caches, configs)

    ranked = sorted(all_scores.items(), key=lambda x: -x[1]["f1"])[:15]
    print("\n=== Top 15 far-only masks (F1 vs WELL pred on far|no-depth) ===")
    for name, s in ranked:
        print(f"  {name:32s}  F1={s['f1']:.3f}  IoU={s['iou']:.3f}  "
              f"P={s['precision']:.2f} R={s['recall']:.2f}  cov={100*s['coverage']:.0f}%")

    far_def = all_scores.get(FAR_DEFAULT["name"], {})
    print(f"\nDefault far preset: F1={far_def.get('f1', 0):.3f}")
    print(f"BEST: {best.name}  F1={all_scores[best.name]['f1']:.3f}")

    out = {
        "mode": "far_only",
        "note": "static mask gated to depth>=ratio*z_ref or no LiDAR; tuned vs good_pred on same domain",
        "best_mask": best.__dict__,
        "best_metrics_avg": all_scores[best.name],
        "far_default": far_def,
        "n_samples": len(caches),
        "err_percentile": args.err_percentile,
        "top15": {k: v for k, v in ranked},
    }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {args.out.resolve()}")

    args.viz_dir.mkdir(parents=True, exist_ok=True)
    near_p = MaskParams("near_excluded", flow_t=5.0, comp_t=20.0, far_min_ratio=0.0)
    for sd, cache in zip(sample_dirs[: args.viz_samples], caches[: args.viz_samples]):
        sid = cache["meta"]["sample_id"]
        masks = {
            "all_depth": build_static_mask(cache["flow_mag"], cache["comp_err"], cache["depth"],
                                           cache["sigma"], near_p, cache["z_ref"]),
            "BEST_far": build_static_mask(cache["flow_mag"], cache["comp_err"], cache["depth"],
                                         cache["sigma"], best, cache["z_ref"]),
        }
        save_comparison(
            cache, masks, cache["good"], cache["depth"], cache["z_ref"], best,
            args.viz_dir / f"{sid}_far_mask_compare.png",
            f"{sid}\nfar-only static vs WELL pred (far|no-depth, p{args.err_percentile:.0f})",
        )
        print(f"Viz {args.viz_dir / (sid + '_far_mask_compare.png')}")


if __name__ == "__main__":
    main()
