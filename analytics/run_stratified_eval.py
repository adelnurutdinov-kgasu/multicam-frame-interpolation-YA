"""
Stratified evaluation: MAE/PSNR по глубине LiDAR и семантическим слоям.

Методы: 06_farneback_alpha, 13_ensemble, 15_layered_parallax
GPU: RIFE (ensemble) + torch для карт ошибок.

Пример:
  python analytics/run_stratified_eval.py
  python analytics/run_stratified_eval.py --num-samples 200 --split test
  python analytics/build_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.stratified_metrics import (
    METHOD_KEYS,
    EvalAccumulator,
    depth_bin_map,
    load_semantic_index,
    pixel_mae,
    pixel_mse,
    rasterize_semantic,
)
from layered_parallax import (
    AlphaTuneConfig,
    BlurTuneConfig,
    farneback_alpha,
    get_lidar_depth,
    intrinsics_to_K,
    layered_parallax_predict,
    load_rgb,
)
from mega_parallax import MegaParallaxConfig, mega_parallax_predict, rasterize_semantic_map
from yolo.cityscapes_classes import CITYSCAPES_NAMES

GPU_METHODS = frozenset({"13_ensemble"})


def init_device(requested: str) -> str:
    """CUDA только если torch грузится; иначе cpu + подсказка."""
    if not requested.startswith("cuda"):
        return "cpu"
    try:
        import torch  # noqa: F401 — до pandas в других модулях

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"GPU: {name} ({torch.__version__})")
            return requested
    except PermissionError:
        print(
            "CUDA DLL занята другим процессом (WinError 32).\n"
            "Закройте другие python/jupyter/notebook и повторите.\n"
            "Проверка:  Get-Process python*  |  taskkill /PID <id> /F\n"
            "Пока работаем на CPU."
        )
        return "cpu"
    except OSError as e:
        print(f"torch/CUDA не загрузился ({e}), fallback CPU")
        return "cpu"
    print("CUDA недоступна, fallback CPU")
    return "cpu"


def load_tune_configs() -> tuple[AlphaTuneConfig, BlurTuneConfig]:
    alpha_cfg = AlphaTuneConfig()
    blur_cfg = BlurTuneConfig()
    alpha_path = ROOT / "alpha_tune_best.json"
    blur_path = ROOT / "blur_tune_best.json"
    if alpha_path.is_file():
        bc = json.loads(alpha_path.read_text()).get("best_config", {})
        alpha_cfg = AlphaTuneConfig(**{k: bc[k] for k in alpha_cfg.__dataclass_fields__ if k in bc})
    if blur_path.is_file():
        bc = json.loads(blur_path.read_text()).get("best_blur_config", {})
        blur_cfg = BlurTuneConfig(
            sigma_near=bc.get("sigma_near", 1.5),
            sigma_far=bc.get("sigma_far", 2.0),
            flow_k=bc.get("flow_k", 0.05),
            comp_k=bc.get("comp_k", 0.0),
        )
    return alpha_cfg, blur_cfg


def compute_error_maps(pred: np.ndarray, gt: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    if device.startswith("cuda"):
        try:
            import torch

            pt = torch.from_numpy(pred).to(device=device, dtype=torch.float32)
            gt_t = torch.from_numpy(gt).to(device=device, dtype=torch.float32)
            diff = pt - gt_t
            return diff.abs().mean(dim=2).cpu().numpy(), (diff ** 2).mean(dim=2).cpu().numpy()
        except (PermissionError, OSError):
            pass
    return pixel_mae(pred, gt), pixel_mse(pred, gt)


def load_sample_bundle(sample_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
    cam = meta["target_camera"]
    img0 = load_rgb(sample_dir / "input" / "t0" / f"{cam}.jpg")
    img1 = load_rgb(sample_dir / "input" / "t1" / f"{cam}.jpg")
    gt = load_rgb(sample_dir / "target" / f"{cam}.jpg")
    return img0, img1, gt, meta


def predict_methods(
    sample_dir: Path,
    split: str,
    dis,
    rife,
    n_layers: int,
    alpha_cfg: AlphaTuneConfig,
    blur_cfg: BlurTuneConfig,
    sem_index: dict[str, dict],
    methods_filter: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    need = set(methods_filter) if methods_filter else None
    use_rife = need is None or bool(need & GPU_METHODS)

    if use_rife:
        from compare_baselines import predict_sample

        preds_base, gt, meta = predict_sample(sample_dir, dis, rife, skip_phase=True)
        img0 = preds_base["t0"]
        img1 = preds_base["t1"]
    else:
        img0, img1, gt, meta = load_sample_bundle(sample_dir)
        preds_base = {}

    cam = meta["target_camera"]
    ts = meta["timestamps_ns"]
    alpha = float((ts["target"] - ts["t0"]) / (ts["t1"] - ts["t0"]))
    preds: dict[str, np.ndarray] = {}

    if need is None or "06_farneback_alpha" in need:
        preds["06_farneback_alpha"] = farneback_alpha(img0, img1, alpha)
    if need is None or "13_ensemble" in need:
        preds["13_ensemble"] = preds_base["ensemble"]

    K = intrinsics_to_K(meta["intrinsics"][cam])
    c2w_t0 = np.array(meta["poses_c2w"]["t0"][cam], dtype=np.float64)
    c2w_t1 = np.array(meta["poses_c2w"]["t1"][cam], dtype=np.float64)
    c2w_tgt = np.array(meta["poses_c2w"]["target"][cam], dtype=np.float64)
    depth = get_lidar_depth(sample_dir, cam, "target")

    if need is None or "15_layered_parallax" in need:
        layered, _ = layered_parallax_predict(
            img0, img1, c2w_t0, c2w_t1, c2w_tgt, K, depth, alpha, n_layers,
            alpha_cfg=alpha_cfg, blur_cfg=blur_cfg,
        )
        preds["15_layered_parallax"] = layered

    if need is None or "17_mega_parallax" in need:
        rel = f"{split}/{meta['sample_id']}/target/{cam}.jpg"
        sem_rec = sem_index.get(rel.replace("\\", "/"))
        sem_map = rasterize_semantic_map(sem_rec.get("regions") if sem_rec else None, gt.shape[0], gt.shape[1])
        mega_cfg = MegaParallaxConfig(
            n_depth_layers=n_layers,
            alpha_cfg=alpha_cfg,
            blur_cfg=blur_cfg,
        )
        mega, _, _ = mega_parallax_predict(
            img0, img1, c2w_t0, c2w_t1, c2w_tgt, K, depth, alpha, sem_map, mega_cfg,
        )
        preds["17_mega_parallax"] = mega

    return preds, gt, meta


def list_samples(dataset_dir: Path, split: str, num_samples: int) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    splits = ["train", "test"] if split == "both" else [split]
    for sp in splits:
        d = dataset_dir / sp
        if not d.is_dir():
            continue
        for p in sorted(x for x in d.iterdir() if x.is_dir()):
            if not _sample_ready(p):
                continue
            out.append((sp, p))
    if num_samples > 0:
        out = out[:num_samples]
    return out


def _sample_ready(sample_dir: Path) -> bool:
    meta_path = sample_dir / "meta.json"
    if not meta_path.is_file():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cam = meta.get("target_camera")
    if not cam:
        return False
    need = [
        sample_dir / "input" / "t0" / f"{cam}.jpg",
        sample_dir / "input" / "t1" / f"{cam}.jpg",
        sample_dir / "target" / f"{cam}.jpg",
        sample_dir / "input" / "lidar.npz",
    ]
    return all(p.is_file() for p in need)


def main() -> int:
    p = argparse.ArgumentParser(description="Stratified error analytics")
    p.add_argument("--config", default="analytics/config.yaml")
    p.add_argument("--dataset-dir", default=None)
    p.add_argument("--semantic-jsonl", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--split", default=None, choices=["train", "test", "both"])
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--backfill-new",
        action="store_true",
        help="Догнать методы из config, которых нет в checkpoint (напр. 17_mega_parallax)",
    )
    args = p.parse_args()

    cfg_path = ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    dataset_dir = Path(args.dataset_dir or cfg["dataset_dir"])
    semantic_jsonl = Path(args.semantic_jsonl or cfg["semantic_jsonl"])
    output_dir = Path(args.output_dir or cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    split = args.split or cfg.get("split", "train")
    num_samples = args.num_samples if args.num_samples is not None else int(cfg.get("num_samples", 0))
    device = init_device(args.device or cfg.get("device", "cuda"))

    methods = cfg.get("methods") or list(METHOD_KEYS)
    depth_edges = [float(x) for x in cfg.get("depth_edges_m", [0, 5, 15, 30, 60, 9999])]
    sem_filter = set(cfg.get("semantic_classes") or []) or None
    n_layers = int(cfg.get("n_layers", 8))
    save_every = int(cfg.get("save_every", 50))

    results_path = output_dir / "stratified_results.json"
    ckpt_path = output_dir / "stratified_checkpoint.json"

    acc = EvalAccumulator()
    done_ids: set[str] = set()
    if (args.resume or args.backfill_new) and ckpt_path.is_file():
        ck = json.loads(ckpt_path.read_text(encoding="utf-8"))
        acc = EvalAccumulator.from_dict(ck["accumulator"])
        done_ids = set(ck.get("done_sample_ids", []))
        print(f"Checkpoint: {len(done_ids)} samples, methods in acc: {list(acc.overall.keys())}")

    missing_methods = [
        m for m in methods
        if m not in acc.overall or acc.overall[m].pixels == 0
    ]
    backfill = args.backfill_new or bool(missing_methods and done_ids)
    if backfill and missing_methods:
        print(f"Backfill новых методов: {missing_methods}")
        for m in missing_methods:
            acc.overall.pop(m, None)
            acc.by_depth.pop(m, None)
            acc.by_semantic.pop(m, None)
        methods_run = missing_methods
        all_samples = list_samples(dataset_dir, split, num_samples)
        samples = all_samples
    else:
        methods_run = methods
        samples = list_samples(dataset_dir, split, num_samples)
        samples = [(sp, sd) for sp, sd in samples if sd.name not in done_ids]

    print("Loading semantic index (target frames only)...")
    sem_index = load_semantic_index(semantic_jsonl, target_only=True)
    print(f"  semantic records: {len(sem_index)}")

    if not samples:
        if missing_methods:
            print("Нет сэмплов для backfill. Проверьте split/dataset.")
        else:
            print("Nothing to process.")
        _write_results(results_path, ckpt_path, acc, done_ids, dataset_dir, split, methods, depth_edges, device, 0)
        print(f"Report: python analytics/build_report.py")
        return 0

    print(f"Device: {device}")
    print(f"Samples: {len(samples)}  methods: {methods_run}")

    alpha_cfg, blur_cfg = load_tune_configs()

    needs_gpu = bool(set(methods_run) & GPU_METHODS)
    dis = None
    rife = None
    if needs_gpu:
        from compare_baselines import RifeWrapper

        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        rife = RifeWrapper()
    elif set(methods_run) <= {"17_mega_parallax"}:
        print("Backfill mega: RIFE/GPU не нужны, только CPU parallax")

    t0 = time.time()
    for i, (sp, sample_dir) in enumerate(tqdm(samples, desc="eval")):
        try:
            preds, gt, meta = predict_methods(
                sample_dir, sp, dis, rife, n_layers, alpha_cfg, blur_cfg, sem_index, methods_run,
            )
            cam = meta["target_camera"]
            rel = f"{sp}/{meta['sample_id']}/target/{cam}.jpg".replace("\\", "/")
            h, w = gt.shape[:2]

            depth = get_lidar_depth(sample_dir, cam, "target")
            d_bins, d_labels = depth_bin_map(depth, depth_edges)
            sem_map = rasterize_semantic(sem_index.get(rel), h, w, sem_filter)

            for method in methods_run:
                if method not in preds:
                    continue
                mae, mse = compute_error_maps(preds[method], gt, device)
                acc.update_sample(method, mae, mse, d_bins, d_labels, sem_map, CITYSCAPES_NAMES)

            if not backfill:
                acc.samples_ok += 1
                done_ids.add(sample_dir.name)
            else:
                acc.samples_ok = max(acc.samples_ok, len(done_ids))
        except Exception as e:
            acc.samples_fail += 1
            tqdm.write(f"FAIL {sample_dir.name}: {e}")

        if (i + 1) % save_every == 0:
            ckpt_path.write_text(
                json.dumps({"accumulator": acc.to_dict(), "done_sample_ids": sorted(done_ids)}, ensure_ascii=False),
                encoding="utf-8",
            )

    elapsed = time.time() - t0
    _write_results(results_path, ckpt_path, acc, done_ids, dataset_dir, split, methods, depth_edges, device, elapsed)
    print(f"\nDone in {elapsed:.0f}s  ok={acc.samples_ok} fail={acc.samples_fail}")
    print(f"Results: {results_path}")
    print(f"Report:  python analytics/build_report.py")
    return 0


def _write_results(
    results_path: Path,
    ckpt_path: Path,
    acc: EvalAccumulator,
    done_ids: set[str],
    dataset_dir: Path,
    split: str,
    methods: list[str],
    depth_edges: list[float],
    device: str,
    elapsed: float,
) -> None:
    out = {
        "config": {
            "dataset_dir": str(dataset_dir),
            "split": split,
            "methods": methods,
            "depth_edges_m": depth_edges,
            "device": device,
            "elapsed_sec": round(elapsed, 1),
        },
        **acc.to_dict(),
    }
    results_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    ckpt_path.write_text(
        json.dumps({"accumulator": acc.to_dict(), "done_sample_ids": sorted(done_ids)}, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
