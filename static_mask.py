"""Static region masks from t0/t1 flow, comp_err, optional LiDAR depth."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_CONFIG_PATH = Path("static_mask_tune_best.json")

BASELINE = dict(
    name="baseline_flow5_comp20",
    flow_t=5.0, comp_t=20.0, fdn_t=80.0, depth_exp=0.5, sigma_t=1.2,
    far_min_ratio=0.0, far_min_abs=0.0, include_no_depth=True,
    use_comp=True, use_flow=True, use_depth_scale=False, use_fdn=False,
    use_sigma=False, require_depth=False,
)
# Only far LiDAR hits and sky/no-depth pixels (near = excluded as coincidence).
FAR_DEFAULT = dict(
    name="far_r2.0_nod_f8_c20",
    flow_t=8.0, comp_t=20.0, fdn_t=80.0, depth_exp=0.5, sigma_t=1.2,
    far_min_ratio=2.0, far_min_abs=0.0, include_no_depth=True,
    use_comp=True, use_flow=True, use_depth_scale=False, use_fdn=False,
    use_sigma=False, require_depth=False,
)


@dataclass
class MaskParams:
    name: str = "custom"
    flow_t: float = 5.0
    comp_t: float = 20.0
    fdn_t: float = 80.0
    depth_exp: float = 0.5
    sigma_t: float = 1.2
    far_min_ratio: float = 0.0   # depth >= ratio * z_ref; 0 = off
    far_min_abs: float = 0.0     # depth >= abs meters; 0 = off
    include_no_depth: bool = True
    use_comp: bool = True
    use_flow: bool = True
    use_depth_scale: bool = False
    use_fdn: bool = False
    use_sigma: bool = False
    require_depth: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> MaskParams:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def uses_far_gate(self) -> bool:
        return self.far_min_ratio > 0 or self.far_min_abs > 0


def compute_flow_maps(img_t0: np.ndarray, img_t1: np.ndarray):
    g0 = cv2.cvtColor(img_t0, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(img_t1, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g1, g0, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    h, w = img_t0.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    warped = cv2.remap(
        img_t1, xs - flow[..., 0], ys - flow[..., 1],
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    comp_err = np.mean(np.abs(img_t0.astype(np.float32) - warped.astype(np.float32)), axis=2)
    sigma_map = np.clip(0.3 + 0.4 * flow_mag + 0.15 * comp_err, 0.3, 4.0)
    return flow_mag, comp_err, sigma_map, flow


def lidar_hit_mask(depth_raw: np.ndarray) -> np.ndarray:
    return np.isfinite(depth_raw) & (depth_raw > 0)


def z_ref_from_lidar(depth_raw: np.ndarray, fallback: float = 10.0) -> float:
    hit = lidar_hit_mask(depth_raw)
    return float(np.median(depth_raw[hit])) if hit.any() else fallback


def far_or_no_depth_gate(
    depth_raw: np.ndarray | None,
    z_ref: float,
    far_min_ratio: float,
    far_min_abs: float,
    include_no_depth: bool,
) -> np.ndarray:
    """True where raw LiDAR is far enough OR pixel has no LiDAR hit (sky etc.).

    Must use *raw* sparse depth — filled/inpainted maps mark sky as valid near depth
    and wrongly exclude those pixels from the far gate.
    """
    if depth_raw is None:
        raise ValueError("far_or_no_depth_gate requires raw LiDAR depth (use get_lidar_depth_raw)")

    hit = lidar_hit_mask(depth_raw)
    h, w = depth_raw.shape
    if far_min_ratio <= 0 and far_min_abs <= 0:
        return np.ones((h, w), dtype=bool)

    min_z = max(far_min_abs, far_min_ratio * z_ref) if far_min_ratio > 0 else far_min_abs
    far = hit & (depth_raw >= min_z)
    no_hit = (~hit) & include_no_depth
    return far | no_hit


def build_static_mask(
    flow_mag: np.ndarray,
    comp_err: np.ndarray,
    depth: np.ndarray | None,
    sigma_map: np.ndarray,
    params: MaskParams,
    z_ref: float = 10.0,
    depth_raw: np.ndarray | None = None,
) -> np.ndarray:
    h, w = flow_mag.shape
    if depth is None:
        valid_d = np.zeros((h, w), dtype=bool)
        depth_arr = np.full((h, w), z_ref, dtype=np.float32)
    else:
        valid_d = np.isfinite(depth) & (depth > 0)
        depth_arr = depth

    if params.use_fdn:
        fdn = np.full((h, w), np.inf, dtype=np.float32)
        fdn[valid_d] = flow_mag[valid_d] * depth_arr[valid_d]
        flow_ok = fdn <= params.fdn_t
        flow_ok[~valid_d] = flow_mag[~valid_d] <= params.flow_t
    elif params.use_depth_scale:
        thr = params.flow_t * np.power(np.maximum(depth_arr, 0.5) / max(z_ref, 0.5), params.depth_exp)
        thr[~valid_d] = params.flow_t
        flow_ok = flow_mag <= thr
    elif params.use_flow:
        flow_ok = flow_mag <= params.flow_t
    else:
        flow_ok = np.ones((h, w), dtype=bool)

    comp_ok = comp_err <= params.comp_t if params.use_comp else np.ones((h, w), dtype=bool)
    sigma_ok = sigma_map <= params.sigma_t if params.use_sigma else np.ones((h, w), dtype=bool)

    mask = flow_ok & comp_ok & sigma_ok
    if params.require_depth:
        mask &= valid_d
    if params.uses_far_gate:
        if depth_raw is None:
            raise ValueError("build_static_mask with far gate requires depth_raw (get_lidar_depth_raw)")
        mask &= far_or_no_depth_gate(
            depth_raw, z_ref, params.far_min_ratio, params.far_min_abs, params.include_no_depth,
        )
    return mask


def mask_metrics(static: np.ndarray, good: np.ndarray) -> dict[str, float]:
    inter = static & good
    union = static | good
    prec = inter.sum() / max(static.sum(), 1)
    rec = inter.sum() / max(good.sum(), 1)
    return {
        "iou": float(inter.sum() / max(union.sum(), 1)),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(2 * prec * rec / max(prec + rec, 1e-9)),
        "coverage": float(static.mean()),
        "overlap_with_good": float(inter.mean()),
    }


def load_best_params(path: Path = DEFAULT_CONFIG_PATH) -> MaskParams:
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        return MaskParams.from_dict(data["best_mask"])
    return MaskParams.from_dict(FAR_DEFAULT)


def describe_params(p: MaskParams) -> str:
    parts = []
    if p.uses_far_gate:
        far = []
        if p.far_min_ratio > 0:
            far.append(f"z>={p.far_min_ratio:.1f}*zref")
        if p.far_min_abs > 0:
            far.append(f"z>={p.far_min_abs:.0f}m")
        if p.include_no_depth:
            far.append("no-depth")
        parts.append("(" + " | ".join(far) + ")")
    if p.use_fdn:
        parts.append(f"flow*depth<{p.fdn_t:.0f}")
    elif p.use_depth_scale:
        parts.append(f"flow<{p.flow_t:.0f}*(z/zref)^{p.depth_exp:.1f}")
    elif p.use_flow:
        parts.append(f"flow<{p.flow_t:.0f}")
    if p.use_comp:
        parts.append(f"comp<{p.comp_t:.0f}")
    if p.use_sigma:
        parts.append(f"sigma<{p.sigma_t:.1f}")
    if p.require_depth:
        parts.append("depth req")
    return " & ".join(parts) if parts else p.name
