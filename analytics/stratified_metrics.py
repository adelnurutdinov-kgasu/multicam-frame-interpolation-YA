"""Метрики ошибок по глубине и семантическим классам."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from yolo.cityscapes_classes import NAME_TO_ID

METHOD_KEYS = {
    "06_farneback_alpha": "farneback_alpha",
    "13_ensemble": "ensemble",
    "15_layered_parallax": "layered",
    "17_mega_parallax": "mega",
}


def load_semantic_index(jsonl_path: Path, target_only: bool = True) -> dict[str, dict]:
    idx: dict[str, dict] = {}
    if not jsonl_path.is_file():
        return idx
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            img = rec["image"].replace("\\", "/")
            if target_only and "/target/" not in img:
                continue
            idx[img] = rec
    return idx


def rasterize_semantic(rec: dict | None, h: int, w: int, allowed: set[str] | None) -> np.ndarray:
    """Карта class_id на пиксель, -1 = нет класса."""
    out = np.full((h, w), -1, dtype=np.int16)
    if not rec:
        return out
    for reg in rec.get("regions", []):
        name = reg.get("class_name", "")
        if allowed and name not in allowed:
            continue
        cls_id = NAME_TO_ID.get(name)
        if cls_id is None:
            continue
        poly = reg.get("polygon_xy") or []
        if len(poly) < 3:
            continue
        pts = np.array(poly, dtype=np.int32)
        cv2.fillPoly(out, [pts], cls_id)
    return out


def depth_bin_map(depth: np.ndarray, edges_m: list[float]) -> tuple[np.ndarray, list[str]]:
    """bin id 0..K-1, -1 = нет глубины."""
    labels = []
    for i in range(len(edges_m) - 1):
        lo, hi = edges_m[i], edges_m[i + 1]
        if hi >= 9000:
            labels.append(f"{lo:.0f}m+")
        else:
            labels.append(f"{lo:.0f}-{hi:.0f}m")

    valid = np.isfinite(depth) & (depth > 0)
    out = np.full(depth.shape, -1, dtype=np.int16)
    for i in range(len(edges_m) - 1):
        lo, hi = edges_m[i], edges_m[i + 1]
        mask = valid & (depth >= lo) & (depth < hi)
        out[mask] = i
    return out, labels


def pixel_mae(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(pred.astype(np.float32) - gt.astype(np.float32)), axis=2)


def pixel_mse(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.mean((pred.astype(np.float32) - gt.astype(np.float32)) ** 2, axis=2)


def mae_to_psnr(mae: float) -> float:
    mse = (mae / (255.0 / np.sqrt(3))) ** 2 if mae > 0 else 0.0
    return 20.0 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float("inf")


@dataclass
class StratumAccumulator:
    sum_mae: float = 0.0
    sum_mse: float = 0.0
    pixels: int = 0

    def add(self, mae_map: np.ndarray, mse_map: np.ndarray, mask: np.ndarray) -> None:
        if not np.any(mask):
            return
        self.sum_mae += float(mae_map[mask].sum())
        self.sum_mse += float(mse_map[mask].sum())
        self.pixels += int(mask.sum())

    def stats(self) -> dict:
        if self.pixels == 0:
            return {"mae": None, "mse": None, "psnr_db": None, "pixels": 0}
        mae = self.sum_mae / self.pixels
        mse = self.sum_mse / self.pixels
        psnr = 20.0 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float("inf")
        return {"mae": round(mae, 4), "mse": round(mse, 4), "psnr_db": round(psnr, 3), "pixels": self.pixels}


@dataclass
class EvalAccumulator:
    by_depth: dict[str, dict[str, StratumAccumulator]] = field(default_factory=dict)
    by_semantic: dict[str, dict[str, StratumAccumulator]] = field(default_factory=dict)
    overall: dict[str, StratumAccumulator] = field(default_factory=dict)
    samples_ok: int = 0
    samples_fail: int = 0

    def _depth_acc(self, method: str, label: str) -> StratumAccumulator:
        self.by_depth.setdefault(method, {})
        if label not in self.by_depth[method]:
            self.by_depth[method][label] = StratumAccumulator()
        return self.by_depth[method][label]

    def _sem_acc(self, method: str, label: str) -> StratumAccumulator:
        self.by_semantic.setdefault(method, {})
        if label not in self.by_semantic[method]:
            self.by_semantic[method][label] = StratumAccumulator()
        return self.by_semantic[method][label]

    def _overall_acc(self, method: str) -> StratumAccumulator:
        if method not in self.overall:
            self.overall[method] = StratumAccumulator()
        return self.overall[method]

    def update_sample(
        self,
        method: str,
        mae_map: np.ndarray,
        mse_map: np.ndarray,
        depth_bins: np.ndarray,
        depth_labels: list[str],
        sem_map: np.ndarray,
        id_to_name: dict[int, str],
    ) -> None:
        valid = np.ones(mae_map.shape, dtype=bool)
        self._overall_acc(method).add(mae_map, mse_map, valid)

        for i, label in enumerate(depth_labels):
            mask = depth_bins == i
            self._depth_acc(method, label).add(mae_map, mse_map, mask)

        for cls_id in np.unique(sem_map):
            cls_id = int(cls_id)
            if cls_id < 0:
                continue
            name = id_to_name.get(cls_id, str(cls_id))
            mask = sem_map == cls_id
            self._sem_acc(method, name).add(mae_map, mse_map, mask)

    def to_dict(self) -> dict:
        def nest(acc_map: dict[str, dict[str, StratumAccumulator]]) -> dict:
            return {m: {k: v.stats() for k, v in sorted(rows.items())} for m, rows in acc_map.items()}

        return {
            "overall": {m: a.stats() for m, a in self.overall.items()},
            "by_depth": nest(self.by_depth),
            "by_semantic": nest(self.by_semantic),
            "samples_ok": self.samples_ok,
            "samples_fail": self.samples_fail,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvalAccumulator":
        acc = cls(samples_ok=d.get("samples_ok", 0), samples_fail=d.get("samples_fail", 0))

        def load_nest(target: dict, src: dict) -> None:
            for method, rows in src.items():
                target[method] = {}
                for label, st in rows.items():
                    a = StratumAccumulator()
                    if st.get("pixels", 0):
                        a.pixels = st["pixels"]
                        a.sum_mae = st["mae"] * st["pixels"]
                        a.sum_mse = st["mse"] * st["pixels"]
                    target[method][label] = a

        load_nest(acc.by_depth, d.get("by_depth", {}))
        load_nest(acc.by_semantic, d.get("by_semantic", {}))
        for method, st in d.get("overall", {}).items():
            a = StratumAccumulator()
            if st.get("pixels", 0):
                a.pixels = st["pixels"]
                a.sum_mae = st["mae"] * st["pixels"]
                a.sum_mse = st["mse"] * st["pixels"]
            acc.overall[method] = a
        return acc

    def merge(self, other: "EvalAccumulator") -> None:
        for method in set(list(self.overall) + list(other.overall)):
            for label, a in other.by_depth.get(method, {}).items():
                t = self._depth_acc(method, label)
                t.sum_mae += a.sum_mae
                t.sum_mse += a.sum_mse
                t.pixels += a.pixels
            for label, a in other.by_semantic.get(method, {}).items():
                t = self._sem_acc(method, label)
                t.sum_mae += a.sum_mae
                t.sum_mse += a.sum_mse
                t.pixels += a.pixels
            oa = other.overall.get(method)
            if oa:
                t = self._overall_acc(method)
                t.sum_mae += oa.sum_mae
                t.sum_mse += oa.sum_mse
                t.pixels += oa.pixels
        self.samples_ok += other.samples_ok
        self.samples_fail += other.samples_fail


def compare_methods_table(by_sem: dict, method_a: str, method_b: str) -> list[dict]:
    """Где B лучше A (меньше MAE) — delta_mae отрицательный."""
    rows = []
    a_map = by_sem.get(method_a, {})
    b_map = by_sem.get(method_b, {})
    for cls in sorted(set(a_map) | set(b_map)):
        ma = a_map.get(cls, {}).get("mae")
        mb = b_map.get(cls, {}).get("mae")
        if ma is None or mb is None:
            continue
        rows.append(
            {
                "class": cls,
                f"{method_a}_mae": ma,
                f"{method_b}_mae": mb,
                "delta_mae": round(mb - ma, 4),
                "better": method_b if mb < ma else method_a,
            }
        )
    return rows
