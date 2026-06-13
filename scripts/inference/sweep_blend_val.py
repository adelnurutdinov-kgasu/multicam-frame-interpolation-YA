"""Val: сетка + Bayesian (Optuna) для blend pred / RIFE(raw) / RIFE(guided mean).

final = w_m * pred + w_r * rife_raw + w_g * rife_guided(w_in)
с w_m + w_r + w_g = 1 (Dirichlet в Optuna).

Пример:
  python scripts/inference/sweep_blend_val.py --grid --bayes-trials 60
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_BASELINES = REPO / "scripts" / "baselines"
if str(_BASELINES) not in sys.path:
    sys.path.insert(0, str(_BASELINES))

from compare_baselines import RifeWrapper, load_rgb  # noqa: E402
from lib.consensus_kit import ConsensusConfig  # noqa: E402
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "run_rife_guided_inputs",
    REPO / "scripts" / "inference" / "run_rife_guided_inputs.py",
)
_rgi = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_rgi)
_baked_to_dataset_dir = _rgi._baked_to_dataset_dir
_blend_u8 = _rgi._blend_u8
_load_pred_rgb = _rgi._load_pred_rgb
_match_shape = _rgi._match_shape
_v2_ready_with_trust = _rgi._v2_ready_with_trust
_v2_train_val_split = _rgi._v2_train_val_split
from ya_paths import CV_ROOT  # noqa: E402


def _mean_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def _fuse(pred: np.ndarray, rife_raw: np.ndarray, rife_guided: np.ndarray, w_m: float, w_r: float, w_g: float) -> np.ndarray:
    s = w_m + w_r + w_g
    if s <= 1e-8:
        s = 1.0
    w_m, w_r, w_g = w_m / s, w_r / s, w_g / s
    out = pred.astype(np.float32) * w_m + rife_raw.astype(np.float32) * w_r + rife_guided.astype(np.float32) * w_g
    return np.clip(out, 0, 255).astype(np.uint8)


def _val_jobs(dataset_root: Path, trust_root: Path, seed: int, val_fraction: float) -> list[tuple[Path, str, str]]:
    ds_cfg = ConsensusConfig(
        dataset_root=str(dataset_root),
        consensus_root=str(CV_ROOT / "multiview_warps" / "train"),
        baked_root=str(CV_ROOT / "rife_refinement_baked" / "train"),
        rife_root=str(CV_ROOT / "rife_predictions_v5" / "train"),
        ego_masks_root=str(REPO / "methods_gallery" / "_ego_manual_masks" / "masks_approved"),
        allowed_cameras=(),
        image_h=544,
        image_w=1024,
    )
    ready = _v2_ready_with_trust(ds_cfg, trust_root)
    picked = _v2_train_val_split(ready, seed=seed, val_fraction=val_fraction, subset="val")
    jobs: list[tuple[Path, str, str]] = []
    for bdir in picked:
        meta = json.loads((bdir / "meta.json").read_text(encoding="utf-8"))
        cam = meta.get("target_camera") or meta.get("camera")
        sid = meta["sample_id"]
        sd = _baked_to_dataset_dir(bdir, dataset_root)
        if (sd / "target" / f"{cam}.jpg").is_file():
            jobs.append((sd, sid, cam))
    return jobs


def _load_rife_precomputed(sid: str, ref: np.ndarray) -> np.ndarray | None:
    p = CV_ROOT / "rife_predictions_v5" / "train" / f"{sid}.jpg"
    if not p.is_file():
        return None
    return _match_shape(ref, load_rgb(p))


class ValCache:
    """Кэш pred / rife_raw / gt и guided по w_in (округление)."""

    def __init__(self, jobs: list, rife: RifeWrapper, pred_source: str):
        self.jobs = jobs
        self.rife = rife
        self.pred_source = pred_source
        self.pred: list[np.ndarray] = []
        self.rife_raw: list[np.ndarray] = []
        self.gt: list[np.ndarray] = []
        self.t0: list[np.ndarray] = []
        self.t1: list[np.ndarray] = []
        self.guided: dict[float, list[np.ndarray]] = {}
        self._fill_base()

    def _fill_base(self) -> None:
        rife_disk = CV_ROOT / "rife_predictions_v5" / "train"
        for sd, sid, cam in self.jobs:
            pred = _load_pred_rgb(sid, cam, self.pred_source, "train", sd)
            if pred is None:
                raise RuntimeError(f"no pred for {sid}")
            img0 = load_rgb(sd / "input" / "t0" / f"{cam}.jpg")
            img1 = load_rgb(sd / "input" / "t1" / f"{cam}.jpg")
            gt = load_rgb(sd / "target" / f"{cam}.jpg")
            pred = _match_shape(img0, pred)
            rpc = _load_rife_precomputed(sid, img0)
            if rpc is None:
                rpc = self.rife.infer(img0, img1)
            self.pred.append(pred)
            self.rife_raw.append(rpc)
            self.gt.append(gt)
            self.t0.append(img0)
            self.t1.append(img1)
        print(f"cache base: n={len(self.pred)}  rife_disk={rife_disk.is_dir()}", flush=True)

    def guided_at(self, w_in: float) -> list[np.ndarray]:
        key = round(float(w_in), 2)
        if key in self.guided:
            return self.guided[key]
        out: list[np.ndarray] = []
        for pred, t0, t1 in zip(self.pred, self.t0, self.t1):
            in0 = _blend_u8(pred, t0, key)
            in1 = _blend_u8(pred, t1, key)
            out.append(self.rife.infer(in0, in1))
        self.guided[key] = out
        print(f"  guided w_in={key} done ({len(out)} samples)", flush=True)
        return out

    def score(self, w_in: float, w_m: float, w_r: float, w_g: float) -> float:
        g = self.guided_at(w_in)
        psnrs = [_mean_psnr(_fuse(p, r, gi, w_m, w_r, w_g), gt) for p, r, gi, gt in zip(self.pred, self.rife_raw, g, self.gt)]
        return float(np.mean(psnrs))


def run_grid(cache: ValCache, w_in_grid: np.ndarray, w_out_grid: np.ndarray, w_g_grid: np.ndarray) -> list[dict]:
    """Сетка: w_out = вес pred, (1-w_out) делится между raw/guided через w_g."""
    rows: list[dict] = []
    best_psnr = -1.0
    best_row: dict | None = None
    n_total = len(w_in_grid) * len(w_out_grid) * len(w_g_grid)
    k = 0
    for w_in in w_in_grid:
        cache.guided_at(float(w_in))
        for w_out in w_out_grid:
            w_m = float(w_out)
            rem = 1.0 - w_m
            for w_g in w_g_grid:
                k += 1
                w_r = rem * (1.0 - float(w_g))
                w_gi = rem * float(w_g)
                psnr = cache.score(float(w_in), w_m, w_r, w_gi)
                row = {
                    "w_in": float(w_in),
                    "w_model": w_m,
                    "w_rife_raw": w_r,
                    "w_rife_guided": w_gi,
                    "mean_psnr": psnr,
                }
                rows.append(row)
                if psnr > best_psnr:
                    best_psnr = psnr
                    best_row = row
                if k % 50 == 0 or k == n_total:
                    print(f"grid [{k}/{n_total}] best={best_psnr:.3f} dB", flush=True)
    rows.sort(key=lambda r: r["mean_psnr"], reverse=True)
    return rows


def run_bayes(cache: ValCache, n_trials: int, seed: int, w_in_bounds: tuple[float, float]) -> dict:
    try:
        import optuna
    except ImportError:
        return _run_bayes_random(cache, n_trials, seed, w_in_bounds)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        w_in = trial.suggest_float("w_in", w_in_bounds[0], w_in_bounds[1])
        w_m = trial.suggest_float("w_model", 0.45, 0.98)
        w_r = trial.suggest_float("w_rife_raw", 0.0, 0.55)
        w_g = trial.suggest_float("w_rife_guided", 0.0, 0.55)
        return cache.score(w_in, w_m, w_r, w_g)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(15, n_trials // 3)),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_trial
    return {
        "mean_psnr": float(best.value),
        "w_in": float(best.params["w_in"]),
        "w_model": float(best.params["w_model"]),
        "w_rife_raw": float(best.params["w_rife_raw"]),
        "w_rife_guided": float(best.params["w_rife_guided"]),
        "n_trials": n_trials,
        "backend": "optuna_tpe",
    }


def _run_bayes_random(cache: ValCache, n_trials: int, seed: int, w_in_bounds: tuple[float, float]) -> dict:
    rng = np.random.RandomState(seed)
    best_val = -1.0
    best_p: dict = {}
    for t in range(n_trials):
        w_in = float(rng.uniform(w_in_bounds[0], w_in_bounds[1]))
        w_m = float(rng.uniform(0.45, 0.98))
        w_r = float(rng.uniform(0.0, 0.55))
        w_g = float(rng.uniform(0.0, 0.55))
        v = cache.score(w_in, w_m, w_r, w_g)
        if v > best_val:
            best_val = v
            best_p = {"w_in": w_in, "w_model": w_m, "w_rife_raw": w_r, "w_rife_guided": w_g}
        if (t + 1) % 10 == 0:
            print(f"random bayes [{t+1}/{n_trials}] best={best_val:.3f} dB", flush=True)
    best_p["mean_psnr"] = best_val
    best_p["n_trials"] = n_trials
    best_p["backend"] = "random_search"
    return best_p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-fraction", type=float, default=0.16)
    ap.add_argument("--pred-source", default="pseudo")
    ap.add_argument("--out", type=Path, default=CV_ROOT / "rife_guided_blend_v1" / "blend_sweep_val.json")
    ap.add_argument("--grid", action="store_true", help="полный перебор сетки")
    ap.add_argument("--bayes-trials", type=int, default=50)
    ap.add_argument("--w-in-step", type=float, default=0.1, help="шаг w_in на сетке")
    ap.add_argument("--w-out-step", type=float, default=0.05, help="шаг веса pred на сетке")
    args = ap.parse_args()

    dataset_root = CV_ROOT / "final_dataset_v5_participants" / "train"
    trust_root = CV_ROOT / "precomputed_lidar_trust" / "train"
    jobs = _val_jobs(dataset_root, trust_root, args.seed, args.val_fraction)
    print(f"val jobs: {len(jobs)}", flush=True)

    t0 = time.time()
    rife = RifeWrapper()
    cache = ValCache(jobs, rife, args.pred_source)

    baselines = {
        "pred_only": cache.score(0.5, 1.0, 0.0, 0.0),
        "rife_raw_only": cache.score(0.5, 0.0, 1.0, 0.0),
        "rife_guided_w05": cache.score(0.5, 0.0, 0.0, 1.0),
        "blend_07_pred_03_raw": cache.score(0.5, 0.7, 0.3, 0.0),
        "blend_07_pred_03_guided": cache.score(0.5, 0.7, 0.0, 0.3),
    }
    print("baselines:", {k: round(v, 3) for k, v in baselines.items()}, flush=True)

    result: dict = {"n_val": len(jobs), "baselines": baselines, "seed": args.seed}

    w_in_grid = np.arange(0.25, 0.76, args.w_in_step)
    w_out_grid = np.arange(0.55, 0.991, args.w_out_step)
    w_g_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    if args.grid:
        rows = run_grid(cache, w_in_grid, w_out_grid, w_g_grid)
        result["grid"] = {
            "top10": rows[:10],
            "best": rows[0] if rows else None,
            "n_combos": len(rows),
        }
        print("GRID best:", rows[0] if rows else None, flush=True)

    if args.bayes_trials > 0:
        bayes = run_bayes(cache, args.bayes_trials, args.seed, (0.2, 0.8))
        result["bayesian"] = bayes
        print("BAYES best:", bayes, flush=True)

    result["elapsed_s"] = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
