"""Export all interpolation methods into methods_gallery/<method>/<sample>.jpg"""

REPO = Path(__file__).resolve().parents[2]
import sys
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import argparse
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from compare_baselines import (
    RifeWrapper,
    load_rgb,
    predict_sample,
    psnr,
    phase_correlation_blocks,
)
from layered_parallax import (
    farneback_alpha,
    geo_blend,
    get_lidar_depth,
    intrinsics_to_K,
    layered_parallax_predict,
)
from mega_parallax import mega_parallax_from_sample, MegaParallaxConfig

SEMANTIC_JSONL = Path(r"C:/Users/adel/Downloads/cv_dataset/annotations/semantic/semantic.jsonl")

ROOT = Path(__file__).resolve().parent

METHODS = [
    ("01_t0", "t0", "Копия кадра t0"),
    ("02_t1", "t1", "Копия кадра t1"),
    ("03_mean", "mean", "Среднее (t0+t1)/2"),
    ("04_global_blur", "global_blur", "Blur t1, sigma=2"),
    ("05_farneback_half", "farneback_half", "Warp t1 на flow/2"),
    ("06_farneback_alpha", "farneback_alpha", "Symmetric: t0 на alpha*flow, t1 на (1-alpha)*flow"),
    ("07_farneback_aggr", "farneback_aggr", "Farneback + blur в bad zones"),
    ("08_dis_half", "dis_half", "DIS warp t1 на полпотока"),
    ("09_adaptive_soft", "adaptive_soft", "Adaptive blur soft"),
    ("10_adaptive_aggr", "adaptive_aggr", "Adaptive blur aggressive"),
    ("11_dis_official", "dis_official", "DIS official (оба кадра)"),
    ("12_rife", "rife", "RIFE neural"),
    ("13_ensemble", "ensemble", "0.6 DIS + 0.4 RIFE"),
    ("14_geo_perpixel", "geo_perpixel", "LiDAR geo warp per-pixel"),
    ("15_layered_parallax", "layered", "8 layers + tuned alpha + blur"),
    ("16_phase_blocks", "phase_blocks", "Phase correlation blocks"),
    ("17_mega_parallax", "mega", "depth × semantic mega parallax"),
]


def score_from_psnr(p: float) -> float:
    return max(0.0, min(100.0, (max(10.0, min(30.0, p)) - 10.0) / 20.0 * 100.0))


def collect_preds(sample_dir: Path, dis, rife, n_layers: int, skip_phase: bool, split: str = "train") -> tuple[dict, np.ndarray, dict]:
    preds, gt, meta = predict_sample(sample_dir, dis, rife, skip_phase=True)

    cam = meta["target_camera"]
    ts = meta["timestamps_ns"]
    alpha = float((ts["target"] - ts["t0"]) / (ts["t1"] - ts["t0"]))

    img0 = preds["t0"]
    img1 = preds["t1"]

    preds["farneback_alpha"] = farneback_alpha(img0, img1, alpha)

    K = intrinsics_to_K(meta["intrinsics"][cam])
    c2w_t0 = np.array(meta["poses_c2w"]["t0"][cam], dtype=np.float64)
    c2w_t1 = np.array(meta["poses_c2w"]["t1"][cam], dtype=np.float64)
    c2w_tgt = np.array(meta["poses_c2w"]["target"][cam], dtype=np.float64)
    depth = get_lidar_depth(sample_dir, cam, "target")

    geo = geo_blend(img0, img1, c2w_t0, c2w_t1, c2w_tgt, K, depth, alpha)
    mean = preds["mean"]
    preds["geo_perpixel"] = (0.55 * geo.astype(np.float32) + 0.45 * mean.astype(np.float32)).astype(np.uint8)

    layered, _ = layered_parallax_predict(
        img0, img1, c2w_t0, c2w_t1, c2w_tgt, K, depth, alpha, n_layers,
    )
    preds["layered"] = layered

    if SEMANTIC_JSONL.is_file():
        try:
            mega, _ = mega_parallax_from_sample(
                sample_dir,
                split,
                SEMANTIC_JSONL,
                mega_cfg=MegaParallaxConfig(n_depth_layers=n_layers),
            )
            preds["mega"] = mega
        except Exception:
            pass

    if not skip_phase:
        preds["phase_blocks"] = phase_correlation_blocks(img0, img1)

    preds["GT"] = gt
    return preds, gt, meta


def write_method_readmes(out_dir: Path):
    for folder, key, desc in METHODS:
        d = out_dir / folder
        d.mkdir(parents=True, exist_ok=True)
        readme = d / "README.txt"
        if not readme.exists():
            readme.write_text(f"Метод: {key}\n{desc}\n", encoding="utf-8")


def build_html(out_dir: Path, samples_info: list, avg_psnr: dict):
    method_rows = ""
    for folder, key, desc in METHODS:
        p = avg_psnr.get(key, 0)
        method_rows += f"<tr><td>{folder}</td><td>{html.escape(desc)}</td><td>{p:.2f}</td><td>{score_from_psnr(p):.1f}</td></tr>"

    sections = ""
    for info in samples_info:
        sid = info["sample_id"]
        cam = info["camera"]
        imgs = ""
        for folder, key, desc in METHODS:
            if key not in info["scores"]:
                continue
            p = info["scores"][key]
            path = f"{folder}/{sid}.jpg"
            imgs += f"""
            <div class="card">
              <a href="{path}"><img src="{path}" loading="lazy"></a>
              <div class="lbl">{folder}<br>PSNR {p:.2f}</div>
            </div>"""
        gt = f"_meta/{sid}_gt.jpg"
        t0 = f"_meta/{sid}_t0.jpg"
        t1 = f"_meta/{sid}_t1.jpg"
        sections += f"""
        <section>
          <h2>{html.escape(sid)}</h2>
          <p>camera: <b>{cam}</b></p>
          <div class="row refs">
            <div class="card"><img src="{t0}"><div class="lbl">t0</div></div>
            <div class="card"><img src="{t1}"><div class="lbl">t1</div></div>
            <div class="card"><img src="{gt}"><div class="lbl">GT target</div></div>
          </div>
          <div class="row">{imgs}</div>
        </section>"""

    page = f"""<!DOCTYPE html>
<html lang="ru"><head>
<meta charset="utf-8">
<title>Methods gallery</title>
<style>
  body {{ font-family: system-ui,sans-serif; background:#111; color:#eee; margin:0; padding:16px; }}
  h1 {{ text-align:center; }}
  table {{ border-collapse:collapse; margin:20px auto; }}
  td,th {{ border:1px solid #444; padding:6px 12px; }}
  th {{ background:#333; }}
  section {{ max-width:1600px; margin:40px auto; background:#1a1a1a; padding:16px; border-radius:8px; }}
  .row {{ display:flex; flex-wrap:wrap; gap:8px; }}
  .refs .card {{ max-width:280px; }}
  .card {{ flex:0 0 auto; width:200px; }}
  .card img {{ width:100%; border-radius:4px; }}
  .lbl {{ font-size:11px; color:#aaa; padding:4px 0; text-align:center; }}
</style></head><body>
<h1>Галерея методов интерполяции</h1>
<table><tr><th>#</th><th>Описание</th><th>avg PSNR</th><th>avg score</th></tr>
{method_rows}
</table>
{sections}
</body></html>"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(r"C:\Users\adel\Downloads\cv_dataset\final_dataset_v5_participants"),
    )
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--out-dir", type=Path, default=Path("methods_gallery"))
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--skip-phase", action="store_true")
    p.add_argument("--indices", type=str, default="")
    args = p.parse_args()

    split_dir = args.dataset_dir / args.split
    all_s = sorted(d for d in split_dir.iterdir() if d.is_dir())
    if args.indices:
        idxs = [int(x) for x in args.indices.split(",")]
        samples = [all_s[i] for i in idxs if i < len(all_s)]
    else:
        step = max(1, len(all_s) // args.num_samples)
        samples = [all_s[i * step] for i in range(args.num_samples)]

    out_dir = args.out_dir.resolve()
    meta_dir = out_dir / "_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    write_method_readmes(out_dir)

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    rife = RifeWrapper()

    psnr_sums: dict[str, float] = {}
    psnr_cnt: dict[str, int] = {}
    samples_info = []

    print(f"Export {len(samples)} samples -> {out_dir}", flush=True)

    for i, sd in enumerate(samples):
        try:
            preds, gt, meta = collect_preds(sd, dis, rife, args.n_layers, args.skip_phase, args.split)
        except Exception as e:
            print(f"  FAIL {sd.name}: {e}", flush=True)
            continue

        sid = meta["sample_id"]
        cam = meta["target_camera"]
        scores = {}

        Image.fromarray(preds["t0"]).save(meta_dir / f"{sid}_t0.jpg")
        Image.fromarray(preds["t1"]).save(meta_dir / f"{sid}_t1.jpg")
        Image.fromarray(gt).save(meta_dir / f"{sid}_gt.jpg")

        for folder, key, _ in METHODS:
            if key not in preds:
                continue
            img = preds[key]
            method_dir = out_dir / folder
            method_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(method_dir / f"{sid}.jpg")
            s = psnr(img, gt)
            scores[key] = s
            psnr_sums[key] = psnr_sums.get(key, 0) + s
            psnr_cnt[key] = psnr_cnt.get(key, 0) + 1

        samples_info.append({"sample_id": sid, "camera": cam, "scores": scores})
        print(f"  [{i+1}/{len(samples)}] {sid[:50]}  best={max(scores,key=scores.get)} {max(scores.values()):.2f}dB", flush=True)

    avg_psnr = {k: psnr_sums[k] / psnr_cnt[k] for k in psnr_sums}
    build_html(out_dir, samples_info, avg_psnr)

    summary = {
        "n_samples": len(samples_info),
        "avg_psnr_db": avg_psnr,
        "avg_score": {k: score_from_psnr(v) for k, v in avg_psnr.items()},
        "methods": [{"folder": f, "key": k, "desc": d} for f, k, d in METHODS],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone. Open {out_dir / 'index.html'}", flush=True)
    for folder, key, _ in METHODS:
        if key in avg_psnr:
            print(f"  {folder:22s} {avg_psnr[key]:.2f} dB  score={score_from_psnr(avg_psnr[key]):.1f}", flush=True)


if __name__ == "__main__":
    main()
