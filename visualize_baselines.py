"""Visual comparison: all baselines vs GT. Saves PNG per sample + HTML gallery."""

import argparse
import html
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from compare_baselines import RifeWrapper, predict_sample, psnr

# порядок панелей: входы → GT → официальные → ваши
PANEL_ORDER = [
    "t0", "t1", "GT",
    "mean", "dis_official", "rife", "ensemble",
    "farneback_half", "dis_half", "adaptive_soft", "adaptive_aggr",
]
ERROR_METHODS = ["mean", "dis_official", "ensemble", "adaptive_soft"]
LABELS = {
    "t0": "t0 (вход)",
    "t1": "t1 (вход)",
    "GT": "TARGET (GT)",
    "mean": "mean",
    "dis_official": "DIS official",
    "rife": "RIFE",
    "ensemble": "ensemble",
    "farneback_half": "Farneback ½",
    "dis_half": "DIS ½",
    "adaptive_soft": "adaptive soft",
    "adaptive_aggr": "adaptive aggr",
}


def abs_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(pred.astype(np.float32) - gt.astype(np.float32)), axis=2)


def save_sample_figure(
    sample_dir: Path,
    preds: dict,
    gt: np.ndarray,
    meta: dict,
    out_path: Path,
    thumb_h: int = 220,
) -> dict[str, float]:
    scores = {k: psnr(v, gt) for k, v in preds.items() if k != "GT"}

    panels = [(k, preds[k]) for k in PANEL_ORDER if k in preds]
    n_cols = len(panels)
    n_err = len(ERROR_METHODS)

    fig = plt.figure(figsize=(2.4 * n_cols, 7.5))
    gs = GridSpec(3, n_cols, figure=fig, height_ratios=[1, 1, 0.9], hspace=0.35, wspace=0.08)

    short_id = meta["sample_id"][-40:]
    fig.suptitle(
        f"{short_id}  |  cam={meta['target_camera']}  delta={meta['delta_s']}s",
        fontsize=11,
        y=0.98,
    )

    for col, (name, img) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img)
        ax.axis("off")
        if name == "GT":
            title = LABELS[name]
        else:
            title = f"{LABELS.get(name, name)}\nPSNR {scores.get(name, 0):.2f} dB"
        ax.set_title(title, fontsize=8)

    # zoom crop — центр-низ (дорога + объекты впереди)
    h, w = gt.shape[:2]
    y0, y1 = int(h * 0.35), h
    x0, x1 = int(w * 0.25), int(w * 0.75)
    for col, (name, img) in enumerate(panels):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(img[y0:y1, x0:x1])
        ax.axis("off")
        if col == 0:
            ax.set_ylabel("zoom", fontsize=8)

    err_vmax = 0
    err_maps = {}
    for name in ERROR_METHODS:
        if name in preds:
            err_maps[name] = abs_error(preds[name], gt)
            err_vmax = max(err_vmax, np.percentile(err_maps[name], 98))

    err_cols = len(ERROR_METHODS)
    for i, name in enumerate(ERROR_METHODS):
        if name not in err_maps:
            continue
        ax = fig.add_subplot(gs[2, i * n_cols // err_cols : (i + 1) * n_cols // err_cols])
        im = ax.imshow(err_maps[name], cmap="hot", vmin=0, vmax=err_vmax)
        ax.set_title(f"|err| {LABELS.get(name, name)}", fontsize=8)
        ax.axis("off")
    fig.colorbar(im, ax=fig.axes[-err_cols:], fraction=0.02, pad=0.02, label="|RGB err|")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return scores


def write_html(out_dir: Path, entries: list[dict]) -> None:
    rows = []
    for e in entries:
        img = e["png"].name
        sid = html.escape(e["sample_id"])
        cam = html.escape(e["camera"])
        best = html.escape(e["best"])
        rows.append(f"""
        <section class="sample">
          <h2>{sid}</h2>
          <p>camera: <b>{cam}</b> &nbsp;|&nbsp; best: <b>{best}</b> ({e['best_psnr']:.2f} dB)</p>
          <a href="{img}"><img src="{img}" loading="lazy" alt="{sid}"></a>
          <details>
            <summary>PSNR всех методов</summary>
            <table>
              <tr><th>метод</th><th>PSNR</th></tr>
              {''.join(f'<tr><td>{html.escape(k)}</td><td>{v:.2f}</td></tr>' for k,v in sorted(e['scores'].items(), key=lambda x:-x[1]))}
            </table>
          </details>
        </section>""")

    page = f"""<!DOCTYPE html>
<html lang="ru"><head>
<meta charset="utf-8">
<title>Baseline visual comparison</title>
<style>
  body {{ font-family: system-ui, sans-serif; background:#111; color:#eee; margin:0; padding:20px; }}
  h1 {{ text-align:center; }}
  .sample {{ max-width:1400px; margin:40px auto; background:#1a1a1a; padding:16px; border-radius:8px; }}
  .sample img {{ width:100%; border-radius:4px; cursor:zoom-in; }}
  table {{ border-collapse:collapse; margin-top:8px; }}
  td, th {{ border:1px solid #444; padding:4px 10px; }}
  th {{ background:#333; }}
  summary {{ cursor:pointer; color:#8cf; margin-top:8px; }}
</style></head><body>
<h1>Сравнение бейзлайнов с TARGET (GT)</h1>
<p style="text-align:center;color:#aaa">
  Строка 1: предсказания &nbsp;|&nbsp; Строка 2: zoom на дорогу &nbsp;|&nbsp; Строка 3: карты ошибки |pred−GT|
</p>
{''.join(rows)}
</body></html>"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(r"C:\Users\adel\Downloads\cv_dataset\final_dataset_v5_participants"),
    )
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--out-dir", type=Path, default=Path("baseline_viz"))
    p.add_argument("--indices", type=str, default="", help="comma-separated sample indices, e.g. 0,5,42")
    args = p.parse_args()

    train_dir = args.dataset_dir / "train"
    all_samples = sorted(p for p in train_dir.iterdir() if p.is_dir())
    if args.indices:
        idxs = [int(x.strip()) for x in args.indices.split(",")]
        samples = [all_samples[i] for i in idxs if i < len(all_samples)]
    else:
        # равномерно по датасету + первый
        step = max(1, len(all_samples) // args.num_samples)
        samples = [all_samples[i * step] for i in range(args.num_samples)]

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    rife = RifeWrapper()
    entries = []

    print(f"Generating {len(samples)} visualizations -> {out_dir}")
    for i, sample_dir in enumerate(samples):
        preds, gt, meta = predict_sample(sample_dir, dis, rife, skip_phase=True)
        png = out_dir / f"{i:02d}_{sample_dir.name[:50]}.png"
        scores = save_sample_figure(sample_dir, preds, gt, meta, png)
        best = max(scores, key=scores.get)
        entries.append({
            "sample_id": meta["sample_id"],
            "camera": meta["target_camera"],
            "png": png,
            "scores": scores,
            "best": best,
            "best_psnr": scores[best],
        })
        print(f"  [{i+1}/{len(samples)}] {meta['target_camera']:10s}  best={best} {scores[best]:.2f} dB  -> {png.name}")

    write_html(out_dir, entries)
    print(f"\nOpen in browser:\n  {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
