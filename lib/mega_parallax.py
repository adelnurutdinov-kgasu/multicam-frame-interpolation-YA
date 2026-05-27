"""
Mega parallax: depth layers × semantic classes → отдельный warp/blend на каждый кусок.

Комбинирует LiDAR-слои (как 15_layered_parallax) и семантические маски
(road, sky, vegetation, car…). Статичные классы (sky) сводятся через mean-blend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from lib.layered_parallax import (
    AlphaTuneConfig,
    BlurTuneConfig,
    apply_layer_blur,
    backwarp_map,
    farneback_alpha,
    layer_alpha,
    layer_sigma,
    make_depth_layers,
    remap_rgb,
)
from yolo.cityscapes_classes import CITYSCAPES_NAMES, NAME_TO_ID

# Классы без сильного parallax — сводим через mean(t0,t1)
STATIC_SEMANTIC = frozenset({"sky", "building", "wall"})

# Соседние классы, которые логично warp'ить одной глубиной (столб + знак и т.п.)
AFFINITY_CLASS_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"pole", "traffic sign"}),
    frozenset({"fence", "wall"}),
)

# Крупные «поля» — делим только по depth, не по connected components
BULK_SEMANTIC = frozenset({"road", "terrain", "vegetation", "sidewalk", "sky"})


@dataclass
class MegaRegion:
    mask: np.ndarray
    z_median: float
    depth_layer_idx: int
    class_id: int
    class_name: str


@dataclass
class MegaParallaxConfig:
    n_depth_layers: int = 8
    min_region_pixels: int = 80
    # target — legacy: зоны на GT; source_match — зоны на t0/t1, match, потом mid на target pose
    zoning_frame: str = "source_match"
    region_mode: str = "object"
    depth_split_m: float = 4.0
    max_depth_subparts: int = 2
    merge_affinity_classes: bool = True
    affinity_depth_m: float = 3.0
    match_min_iou: float = 0.04
    topology_align: bool = True
    topology_fallback: bool = True
    # depth_rank | hungarian_iou | hungarian_centroid | hungarian_geo | hungarian_target_topo | ...
    topology_match_mode: str = "hungarian_target_topo"
    use_target_depth_match: bool = True
    alpha_cfg: AlphaTuneConfig = field(default_factory=AlphaTuneConfig)
    blur_cfg: BlurTuneConfig = field(default_factory=BlurTuneConfig)
    use_mean_blend: bool = True
    static_use_mean_only: bool = True
    static_mean_weight: float = 0.75
    allowed_classes: frozenset[str] | None = None


@dataclass
class MatchedPair:
    reg_t0: MegaRegion
    reg_t1: MegaRegion
    iou: float
    flow_du: float
    flow_dv: float
    from_topology: bool = False


TOPOLOGY_MATCH_MODES = (
    "depth_rank",
    "hungarian_iou",
    "hungarian_centroid",
    "hungarian_geo",
    "hungarian_target_topo",
    "greedy_iou",
    "depth_nearest",
    "none",
)


@dataclass
class GeoMatchContext:
    c2w_t0: np.ndarray
    c2w_t1: np.ndarray
    c2w_tgt: np.ndarray
    K: np.ndarray
    depth_tgt: np.ndarray | None
    alpha: float
    h: int
    w: int
    target_topo: np.ndarray | None = None
    target_layer_z: list[float] = field(default_factory=list)
    n_depth_layers: int = 8


@dataclass
class TopologyAlignResult:
    topo_t0: np.ndarray
    topo_t1: np.ndarray
    flow_refined: np.ndarray
    coarse_pairs: list[tuple[MegaRegion, MegaRegion]]
    topo_iou: float
    class_iou: float = 0.0
    target_topo_iou: float = 0.0
    match_mode: str = "depth_rank"
    target_topo: np.ndarray | None = None


def rasterize_semantic_map(
    regions: list[dict] | None,
    h: int,
    w: int,
    allowed: frozenset[str] | None = None,
) -> np.ndarray:
    """class_id на пиксель, -1 = не размечено."""
    out = np.full((h, w), -1, dtype=np.int16)
    if not regions:
        return out
    for reg in regions:
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


def load_semantic_regions(
    semantic_jsonl: Path,
    rel_image: str,
) -> list[dict] | None:
    if not semantic_jsonl.is_file():
        return None
    rel = rel_image.replace("\\", "/")
    with semantic_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("image", "").replace("\\", "/") == rel:
                return rec.get("regions", [])
    return None


def _depth_layer_index(z: float, depth: np.ndarray, n_layers: int) -> int:
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        return 0
    edges = np.quantile(valid, np.linspace(0, 1, n_layers + 1))
    idx = int(np.searchsorted(edges[1:], z, side="right"))
    return min(max(idx, 0), n_layers - 1)


def _split_mask_by_depth(
    mask: np.ndarray,
    depth: np.ndarray,
    depth_split_m: float,
    max_subparts: int,
) -> list[tuple[np.ndarray, float]]:
    z = depth[mask]
    z = z[np.isfinite(z) & (z > 0)]
    if z.size == 0:
        return []
    z_lo, z_hi = float(np.percentile(z, 5)), float(np.percentile(z, 95))
    if z_hi - z_lo <= depth_split_m or max_subparts <= 1:
        return [(mask, float(np.median(z)))]

    n_parts = min(max_subparts, max(2, int(np.ceil((z_hi - z_lo) / depth_split_m))))
    edges = np.quantile(z, np.linspace(0, 1, n_parts + 1))
    parts: list[tuple[np.ndarray, float]] = []
    for i in range(n_parts):
        lo, hi = edges[i], edges[i + 1]
        sub = mask & (depth >= lo) & (depth < hi if i < n_parts - 1 else depth <= hi)
        if int(sub.sum()) < 1:
            continue
        parts.append((sub, float(np.median(depth[sub]))))
    return parts or [(mask, float(np.median(z)))]


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return [labels == i for i in range(1, n)]


def _class_affinity_key(cls_id: int) -> int:
    name = CITYSCAPES_NAMES.get(cls_id, "")
    for group in AFFINITY_CLASS_GROUPS:
        if name in group:
            ids = sorted(NAME_TO_ID[n] for n in group if n in NAME_TO_ID)
            return ids[0] if ids else cls_id
    return cls_id


def _merge_affinity_regions(regions: list[MegaRegion], depth_tol_m: float) -> list[MegaRegion]:
    if len(regions) < 2:
        return regions

    merged: list[MegaRegion] = []
    used = [False] * len(regions)

    for i, reg in enumerate(regions):
        if used[i]:
            continue
        group_mask = reg.mask.copy()
        group_z = [reg.z_median]
        group_cls = reg.class_id
        group_name = reg.class_name
        group_layer = reg.depth_layer_idx
        used[i] = True

        key_i = _class_affinity_key(reg.class_id)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated = cv2.dilate(group_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        changed = True
        while changed:
            changed = False
            for j, other in enumerate(regions):
                if used[j]:
                    continue
                if _class_affinity_key(other.class_id) != key_i:
                    continue
                if abs(float(np.median(group_z)) - other.z_median) > depth_tol_m:
                    continue
                if not np.any(dilated & other.mask):
                    continue
                group_mask |= other.mask
                group_z.append(other.z_median)
                used[j] = True
                changed = True
                dilated = cv2.dilate(group_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        z_med = float(np.median(group_z))
        merged.append(MegaRegion(group_mask, z_med, group_layer, group_cls, group_name))

    return merged


def make_mega_regions_grid(
    depth: np.ndarray,
    sem_map: np.ndarray,
    n_depth_layers: int,
    min_pixels: int,
) -> list[MegaRegion]:
    depth_layers = make_depth_layers(depth, n_depth_layers)
    if not depth_layers:
        return []

    present_classes = sorted(int(c) for c in np.unique(sem_map) if int(c) >= 0)
    regions: list[MegaRegion] = []

    for layer_idx, (depth_mask, _z_layer) in enumerate(depth_layers):
        for cls_id in present_classes:
            mask = depth_mask & (sem_map == cls_id)
            n = int(mask.sum())
            if n < min_pixels:
                continue
            z_med = float(np.median(depth[mask]))
            name = CITYSCAPES_NAMES.get(cls_id, str(cls_id))
            regions.append(MegaRegion(mask, z_med, layer_idx, cls_id, name))

    return regions


def make_mega_regions_object(
    depth: np.ndarray,
    sem_map: np.ndarray,
    n_depth_layers: int,
    min_pixels: int,
    depth_split_m: float = 4.0,
    max_depth_subparts: int = 2,
    merge_affinity: bool = True,
    affinity_depth_m: float = 3.0,
) -> list[MegaRegion]:
    """Семантика → цельные объекты: bulk-классы по depth-слоям, мелкие — по CC."""
    valid = np.isfinite(depth) & (depth > 0)
    present_classes = sorted(int(c) for c in np.unique(sem_map) if int(c) >= 0)
    regions: list[MegaRegion] = []

    for cls_id in present_classes:
        name = CITYSCAPES_NAMES.get(cls_id, str(cls_id))
        cls_mask = (sem_map == cls_id) & valid
        if not np.any(cls_mask):
            continue

        if name in BULK_SEMANTIC:
            parts = _split_mask_by_depth(
                cls_mask, depth, depth_split_m, max(n_depth_layers // 2, max_depth_subparts),
            )
            for sub_mask, z_med in parts:
                if int(sub_mask.sum()) < min_pixels:
                    continue
                layer_idx = _depth_layer_index(z_med, depth, n_depth_layers)
                regions.append(MegaRegion(sub_mask, z_med, layer_idx, cls_id, name))
            continue

        if name in STATIC_SEMANTIC:
            for comp in _connected_components(cls_mask):
                if int(comp.sum()) < min_pixels:
                    continue
                z_med = float(np.median(depth[comp]))
                layer_idx = _depth_layer_index(z_med, depth, n_depth_layers)
                regions.append(MegaRegion(comp, z_med, layer_idx, cls_id, name))
            continue

        for comp in _connected_components(cls_mask):
            if int(comp.sum()) < min_pixels:
                continue
            for sub_mask, z_med in _split_mask_by_depth(comp, depth, depth_split_m, max_depth_subparts):
                if int(sub_mask.sum()) < min_pixels:
                    continue
                layer_idx = _depth_layer_index(z_med, depth, n_depth_layers)
                regions.append(MegaRegion(sub_mask, z_med, layer_idx, cls_id, name))

    if merge_affinity:
        regions = _merge_affinity_regions(regions, affinity_depth_m)

    regions.sort(key=lambda r: r.z_median)
    return regions


def make_mega_regions(
    depth: np.ndarray,
    sem_map: np.ndarray,
    n_depth_layers: int,
    min_pixels: int,
    cfg: MegaParallaxConfig | None = None,
) -> list[MegaRegion]:
    cfg = cfg or MegaParallaxConfig()
    if cfg.region_mode == "grid":
        return make_mega_regions_grid(depth, sem_map, n_depth_layers, min_pixels)
    return make_mega_regions_object(
        depth,
        sem_map,
        n_depth_layers,
        min_pixels,
        cfg.depth_split_m,
        cfg.max_depth_subparts,
        cfg.merge_affinity_classes,
        cfg.affinity_depth_m,
    )


def _classes_match(a: MegaRegion, b: MegaRegion) -> bool:
    if a.class_id == b.class_id:
        return True
    return _class_affinity_key(a.class_id) == _class_affinity_key(b.class_id)


def _flow_warp_mask(mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    warped = np.zeros((h, w), dtype=bool)
    ys, xs = np.where(mask)
    if ys.size == 0:
        return warped
    fx = xs.astype(np.float32) + flow[ys, xs, 0]
    fy = ys.astype(np.float32) + flow[ys, xs, 1]
    xi = np.clip(np.round(fx).astype(np.int32), 0, w - 1)
    yi = np.clip(np.round(fy).astype(np.int32), 0, h - 1)
    warped[yi, xi] = True
    return warped


def _median_flow_in_mask(flow: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0.0, 0.0
    vals = flow[ys, xs]
    return float(np.median(vals[:, 0])), float(np.median(vals[:, 1]))


def build_source_topology_map(regions: list[MegaRegion], h: int, w: int) -> np.ndarray:
    topo = np.full((h, w), -1, dtype=np.int16)
    for i, reg in enumerate(sorted(regions, key=lambda r: (r.class_name, r.z_median))):
        topo[reg.mask] = i + 1
    return topo


def topology_map_to_rgb(topo: np.ndarray, regions: list[MegaRegion]) -> np.ndarray:
    from yolo.cityscapes_classes import CITYSCAPES_COLORS

    out = np.zeros((*topo.shape, 3), dtype=np.uint8)
    ordered = sorted(regions, key=lambda r: (r.class_name, r.z_median))
    for i, reg in enumerate(ordered):
        c = CITYSCAPES_COLORS.get(reg.class_name, (128, 128, 128))
        out[topo == i + 1] = c
    return out


def _region_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0.0, 0.0
    return float(xs.mean()), float(ys.mean())


def _pixel_to_world(u: float, v: float, z: float, c2w: np.ndarray, K: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return c2w @ np.array([x, y, z, 1.0], dtype=np.float64)


def _world_to_pixel(Pw: np.ndarray, c2w: np.ndarray, K: np.ndarray) -> tuple[float, float, float, bool]:
    P = np.linalg.inv(c2w) @ Pw
    if P[2] <= 0.5:
        return 0.0, 0.0, 0.0, False
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return float(fx * P[0] / P[2] + cx), float(fy * P[1] / P[2] + cy), float(P[2]), True


def geo_flow_t0_to_t1(
    c2w_t0: np.ndarray,
    c2w_t1: np.ndarray,
    K: np.ndarray,
    z: float,
    h: int,
    w: int,
) -> np.ndarray:
    """Геометрический flow t0→t1 при постоянной глубине z (метры)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u0, v0 = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    zmap = np.full((h, w), float(z), dtype=np.float64)
    x = (u0 - cx) * zmap / fx
    y = (v0 - cy) * zmap / fy
    ones = np.ones_like(zmap)
    pts = np.stack([x, y, zmap, ones], axis=-1)
    pts_w = np.einsum("ij,...j->...i", c2w_t0, pts)
    pts1 = np.einsum("ij,...j->...i", np.linalg.inv(c2w_t1), pts_w)
    u1 = fx * pts1[..., 0] / np.maximum(pts1[..., 2], 1e-6) + cx
    v1 = fy * pts1[..., 1] / np.maximum(pts1[..., 2], 1e-6) + cy
    return np.stack([u1 - u0, v1 - v0], axis=-1).astype(np.float32)


def _region_world_point(reg: MegaRegion, c2w: np.ndarray, K: np.ndarray) -> np.ndarray | None:
    if not np.any(reg.mask):
        return None
    cx, cy = _region_centroid(reg.mask)
    return _pixel_to_world(cx, cy, reg.z_median, c2w, K)


def _pair_geo_iou(r0: MegaRegion, r1: MegaRegion, geo: GeoMatchContext, z: float | None = None) -> float:
    z_use = float(z if z is not None else 0.5 * (r0.z_median + r1.z_median))
    flow = geo_flow_t0_to_t1(geo.c2w_t0, geo.c2w_t1, geo.K, z_use, geo.h, geo.w)
    return _pair_iou(r0, r1, flow)


def _pair_target_penalty(
    r0: MegaRegion,
    r1: MegaRegion,
    geo: GeoMatchContext,
    use_target_depth: bool = True,
) -> float:
    """0=хорошо: проекции r0/r1 на target согласованы с LiDAR и друг с другом."""
    if geo.depth_tgt is None:
        return 1.0
    t_iou = _pair_target_topo_iou(r0, r1, geo)
    if not use_target_depth:
        return 1.0 - t_iou
    hist = _pair_target_layer_sim(r0, r1, geo)
    return (1.0 - t_iou) + 0.35 * (1.0 - hist)


def build_target_depth_topology(
    depth_tgt: np.ndarray,
    n_layers: int,
) -> tuple[np.ndarray, list[float]]:
    """Топология target pose: id слоя LiDAR (без sem)."""
    h, w = depth_tgt.shape
    topo = np.full((h, w), -1, dtype=np.int16)
    layer_z: list[float] = []
    for i, (mask, z_med) in enumerate(make_depth_layers(depth_tgt, n_layers)):
        topo[mask] = i + 1
        layer_z.append(z_med)
    return topo, layer_z


def build_geo_match_context(
    c2w_t0: np.ndarray,
    c2w_t1: np.ndarray,
    c2w_tgt: np.ndarray,
    K: np.ndarray,
    depth_tgt: np.ndarray | None,
    alpha: float,
    h: int,
    w: int,
    n_depth_layers: int = 8,
) -> GeoMatchContext:
    target_topo, layer_z = None, []
    if depth_tgt is not None:
        target_topo, layer_z = build_target_depth_topology(depth_tgt, n_depth_layers)
    return GeoMatchContext(
        c2w_t0, c2w_t1, c2w_tgt, K, depth_tgt, alpha, h, w,
        target_topo, layer_z, n_depth_layers,
    )


def target_topology_to_rgb(topo: np.ndarray, layer_z: list[float]) -> np.ndarray:
    out = np.zeros((*topo.shape, 3), dtype=np.uint8)
    if not layer_z:
        return out
    zmin, zmax = min(layer_z), max(layer_z)
    span = max(zmax - zmin, 1.0)
    for i, z in enumerate(layer_z):
        t = (z - zmin) / span
        out[topo == i + 1] = (
            int(40 + 180 * t),
            int(80 + 120 * (1 - t)),
            int(160 - 80 * t),
        )
    return out


def project_region_to_target(
    reg: MegaRegion,
    c2w_src: np.ndarray,
    c2w_tgt: np.ndarray,
    K: np.ndarray,
    h: int,
    w: int,
) -> np.ndarray:
    mx, my, valid = backwarp_map(c2w_tgt, c2w_src, K, reg.z_median, h, w)
    return _source_mask_on_target(reg.mask, mx, my, valid)


def _region_target_layer_hist(
    reg: MegaRegion,
    c2w_src: np.ndarray,
    geo: GeoMatchContext,
) -> np.ndarray | None:
    if geo.target_topo is None:
        return None
    m = project_region_to_target(reg, c2w_src, geo.c2w_tgt, geo.K, geo.h, geo.w)
    layers = geo.target_topo[m]
    layers = layers[layers >= 0]
    if layers.size == 0:
        return None
    n = max(len(geo.target_layer_z), int(layers.max()) + 1, 1)
    hist = np.bincount(layers.astype(np.int32), minlength=n + 1).astype(np.float64)
    hist[0] = 0.0
    s = hist.sum()
    return hist / s if s > 0 else None


def _pair_target_topo_iou(r0: MegaRegion, r1: MegaRegion, geo: GeoMatchContext) -> float:
    m0 = project_region_to_target(r0, geo.c2w_t0, geo.c2w_tgt, geo.K, geo.h, geo.w)
    m1 = project_region_to_target(r1, geo.c2w_t1, geo.c2w_tgt, geo.K, geo.h, geo.w)
    union = m0 | m1
    if not np.any(union):
        return 0.0
    return float((m0 & m1).sum() / union.sum())


def _pair_target_layer_sim(r0: MegaRegion, r1: MegaRegion, geo: GeoMatchContext) -> float:
    h0 = _region_target_layer_hist(r0, geo.c2w_t0, geo)
    h1 = _region_target_layer_hist(r1, geo.c2w_t1, geo)
    if h0 is None or h1 is None:
        return 0.0
    n = max(len(h0), len(h1))
    h0 = np.pad(h0, (0, n - len(h0)))
    h1 = np.pad(h1, (0, n - len(h1)))
    return float(np.minimum(h0, h1).sum())


def _pair_target_topo_cost(
    r0: MegaRegion,
    r1: MegaRegion,
    geo: GeoMatchContext,
    geo_weight: float = 0.2,
) -> float:
    """Меньше = лучше. Target pose — общий эталон для t0↔t1."""
    t_iou = _pair_target_topo_iou(r0, r1, geo)
    hist = _pair_target_layer_sim(r0, r1, geo)
    cost = (1.0 - t_iou) + 0.4 * (1.0 - hist)
    if geo_weight > 0:
        g_iou = _pair_geo_iou(r0, r1, geo)
        cost += geo_weight * (1.0 - g_iou)
    return cost


def _flow_for_region(
    r0: MegaRegion,
    flow_fb: np.ndarray,
    geo: GeoMatchContext | None,
    r1: MegaRegion | None = None,
) -> np.ndarray:
    if geo is None:
        return flow_fb
    z = r0.z_median if r1 is None else 0.5 * (r0.z_median + r1.z_median)
    return geo_flow_t0_to_t1(geo.c2w_t0, geo.c2w_t1, geo.K, z, geo.h, geo.w)


def _median_flow_on_region(
    r0: MegaRegion,
    flow_fb: np.ndarray,
    geo: GeoMatchContext | None,
    r1: MegaRegion | None = None,
) -> tuple[float, float]:
    flow = _flow_for_region(r0, flow_fb, geo, r1)
    return _median_flow_in_mask(flow, r0.mask)


def _match_key(reg: MegaRegion) -> int:
    return _class_affinity_key(reg.class_id)


def build_class_topology_map(regions: list[MegaRegion], h: int, w: int) -> np.ndarray:
    topo = np.full((h, w), -1, dtype=np.int16)
    for reg in regions:
        topo[reg.mask] = _class_affinity_key(reg.class_id)
    return topo


def _group_regions_by_affinity(regions: list[MegaRegion]) -> dict[int, list[MegaRegion]]:
    groups: dict[int, list[MegaRegion]] = {}
    for r in regions:
        groups.setdefault(_match_key(r), []).append(r)
    return groups


def _hungarian_pairs(
    r0s: list[MegaRegion],
    r1s: list[MegaRegion],
    cost_fn,
    max_cost: float = 1e5,
) -> list[tuple[MegaRegion, MegaRegion]]:
    if not r0s or not r1s:
        return []
    from scipy.optimize import linear_sum_assignment

    n, m = len(r0s), len(r1s)
    size = max(n, m)
    cost = np.full((size, size), max_cost, dtype=np.float64)
    for i, r0 in enumerate(r0s):
        for j, r1 in enumerate(r1s):
            cost[i, j] = cost_fn(r0, r1)
    row, col = linear_sum_assignment(cost)
    pairs: list[tuple[MegaRegion, MegaRegion]] = []
    for i, j in zip(row, col):
        if i < n and j < m and cost[i, j] < max_cost * 0.99:
            pairs.append((r0s[i], r1s[j]))
    return pairs


def _greedy_iou_pairs(
    r0s: list[MegaRegion],
    r1s: list[MegaRegion],
    flow: np.ndarray,
    min_iou: float = 0.0,
) -> list[tuple[MegaRegion, MegaRegion]]:
    candidates: list[tuple[float, int, int]] = []
    for i, r0 in enumerate(r0s):
        for j, r1 in enumerate(r1s):
            iou = _pair_iou(r0, r1, flow)
            if iou >= min_iou:
                candidates.append((iou, i, j))
    candidates.sort(reverse=True)
    used0: set[int] = set()
    used1: set[int] = set()
    pairs: list[tuple[MegaRegion, MegaRegion]] = []
    for iou, i, j in candidates:
        if i in used0 or j in used1:
            continue
        used0.add(i)
        used1.add(j)
        pairs.append((r0s[i], r1s[j]))
    return pairs


def _depth_nearest_pairs(
    r0s: list[MegaRegion],
    r1s: list[MegaRegion],
    flow: np.ndarray,
    min_iou: float = 0.0,
) -> list[tuple[MegaRegion, MegaRegion]]:
    used1: set[int] = set()
    pairs: list[tuple[MegaRegion, MegaRegion]] = []
    for r0 in sorted(r0s, key=lambda r: -int(r.mask.sum())):
        best_j, best_cost = -1, float("inf")
        for j, r1 in enumerate(r1s):
            if j in used1:
                continue
            iou = _pair_iou(r0, r1, flow)
            if iou < min_iou:
                continue
            cost = abs(r0.z_median - r1.z_median) - 10.0 * iou
            if cost < best_cost:
                best_cost, best_j = cost, j
        if best_j >= 0:
            used1.add(best_j)
            pairs.append((r0, r1s[best_j]))
    return pairs


def coarse_topology_pairs_depth_rank(
    regions_t0: list[MegaRegion],
    regions_t1: list[MegaRegion],
) -> list[tuple[MegaRegion, MegaRegion]]:
    groups_t0 = _group_regions_by_affinity(regions_t0)
    groups_t1 = _group_regions_by_affinity(regions_t1)
    pairs: list[tuple[MegaRegion, MegaRegion]] = []
    for key in sorted(groups_t0.keys()):
        if key not in groups_t1:
            continue
        r0s = sorted(groups_t0[key], key=lambda r: r.z_median)
        r1s = sorted(groups_t1[key], key=lambda r: r.z_median)
        pairs.extend(zip(r0s, r1s))
    return pairs


def coarse_topology_pairs(
    regions_t0: list[MegaRegion],
    regions_t1: list[MegaRegion],
    flow: np.ndarray,
    mode: str = "depth_rank",
    min_iou: float = 0.0,
    geo: GeoMatchContext | None = None,
    use_target_depth: bool = True,
) -> list[tuple[MegaRegion, MegaRegion]]:
    if mode == "none":
        return []
    if mode == "depth_rank":
        return coarse_topology_pairs_depth_rank(regions_t0, regions_t1)

    groups_t0 = _group_regions_by_affinity(regions_t0)
    groups_t1 = _group_regions_by_affinity(regions_t1)
    pairs: list[tuple[MegaRegion, MegaRegion]] = []

    for key in sorted(groups_t0.keys()):
        if key not in groups_t1:
            continue
        r0s, r1s = groups_t0[key], groups_t1[key]
        if mode == "hungarian_iou":
            group = _hungarian_pairs(
                r0s, r1s,
                lambda a, b: 1.0 - _pair_iou(a, b, flow),
                max_cost=1.0,
            )
        elif mode == "hungarian_centroid":
            def _centroid_cost(a: MegaRegion, b: MegaRegion) -> float:
                if geo is not None and geo.target_topo is not None:
                    return _pair_target_topo_cost(a, b, geo, geo_weight=0.15)
                cx0, cy0 = _region_centroid(a.mask)
                du, dv = _median_flow_in_mask(flow, a.mask)
                cx1, cy1 = _region_centroid(b.mask)
                dist = float(np.hypot((cx0 + du) - cx1, (cy0 + dv) - cy1))
                return dist + 0.05 * abs(a.z_median - b.z_median)

            group = _hungarian_pairs(r0s, r1s, _centroid_cost, max_cost=1e4 if geo is None else 5.0)
        elif mode == "hungarian_geo":
            if geo is None:
                raise ValueError("hungarian_geo requires GeoMatchContext")
            def _geo_cost(a: MegaRegion, b: MegaRegion) -> float:
                iou = _pair_geo_iou(a, b, geo)
                if iou < 0.03:
                    return 5.0
                penalty = _pair_target_penalty(a, b, geo, use_target_depth)
                return (1.0 - iou) + 0.45 * penalty

            group = _hungarian_pairs(r0s, r1s, _geo_cost, max_cost=5.0)
        elif mode == "hungarian_target_topo":
            if geo is None or geo.target_topo is None:
                raise ValueError("hungarian_target_topo requires target depth topology")
            group = _hungarian_pairs(
                r0s, r1s,
                lambda a, b: _pair_target_topo_cost(a, b, geo, geo_weight=0.25),
                max_cost=5.0,
            )
        elif mode == "greedy_iou":
            if geo is not None:
                group = []
                candidates: list[tuple[float, int, int]] = []
                for i, r0 in enumerate(r0s):
                    for j, r1 in enumerate(r1s):
                        iou = _pair_geo_iou(r0, r1, geo)
                        if iou >= min_iou:
                            candidates.append((iou, i, j))
                candidates.sort(reverse=True)
                used0: set[int] = set()
                used1: set[int] = set()
                for iou, i, j in candidates:
                    if i in used0 or j in used1:
                        continue
                    used0.add(i)
                    used1.add(j)
                    group.append((r0s[i], r1s[j]))
            else:
                group = _greedy_iou_pairs(r0s, r1s, flow, min_iou)
        elif mode == "depth_nearest":
            group = _depth_nearest_pairs(r0s, r1s, flow, min_iou)
        else:
            raise ValueError(f"unknown topology_match_mode: {mode}")
        pairs.extend(group)
    return pairs


def _topology_iou(topo_warped: np.ndarray, topo_t1: np.ndarray) -> float:
    valid = (topo_warped > 0) & (topo_t1 > 0)
    if not np.any(valid):
        return 0.0
    return float((topo_warped[valid] == topo_t1[valid]).sum() / valid.sum())


def _class_topology_iou(topo_warped: np.ndarray, class_t1: np.ndarray) -> float:
    valid = (topo_warped >= 0) & (class_t1 >= 0)
    if not np.any(valid):
        return 0.0
    return float((topo_warped[valid] == class_t1[valid]).sum() / valid.sum())


def warp_topology_by_flow(topo: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = topo.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    warped = cv2.remap(
        topo.astype(np.float32),
        xs + flow[..., 0],
        ys + flow[..., 1],
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=-1,
    )
    return warped.astype(np.int16)


def _coarse_target_topo_iou(
    coarse: list[tuple[MegaRegion, MegaRegion]],
    geo: GeoMatchContext,
) -> float:
    if geo.target_topo is None or not coarse:
        return 0.0
    topo0 = np.full((geo.h, geo.w), -1, dtype=np.int16)
    topo1 = np.full((geo.h, geo.w), -1, dtype=np.int16)
    for i, (r0, r1) in enumerate(coarse):
        lid = i + 1
        m0 = project_region_to_target(r0, geo.c2w_t0, geo.c2w_tgt, geo.K, geo.h, geo.w)
        m1 = project_region_to_target(r1, geo.c2w_t1, geo.c2w_tgt, geo.K, geo.h, geo.w)
        topo0[m0] = lid
        topo1[m1] = lid
    valid = (topo0 > 0) & (topo1 > 0)
    if not np.any(valid):
        return 0.0
    return float((topo0[valid] == topo1[valid]).sum() / valid.sum())


def align_topology_flow(
    regions_t0: list[MegaRegion],
    regions_t1: list[MegaRegion],
    flow: np.ndarray,
    h: int,
    w: int,
    match_mode: str = "depth_rank",
    min_iou: float = 0.0,
    geo: GeoMatchContext | None = None,
    use_target_depth: bool = True,
) -> TopologyAlignResult:
    topo_t0 = build_source_topology_map(regions_t0, h, w)
    topo_t1 = build_source_topology_map(regions_t1, h, w)
    class_t1 = build_class_topology_map(regions_t1, h, w)
    coarse = coarse_topology_pairs(
        regions_t0, regions_t1, flow, match_mode, min_iou, geo, use_target_depth,
    )

    flow_refined = flow.astype(np.float32).copy()
    use_geo_flow = geo is not None and match_mode in ("hungarian_geo", "hungarian_target_topo", "greedy_iou")
    for r0, r1 in coarse:
        if not np.any(r0.mask):
            continue
        if use_geo_flow:
            geo_flow = _flow_for_region(r0, flow, geo, r1)
            flow_refined[r0.mask] = geo_flow[r0.mask]
        else:
            du, dv = _median_flow_in_mask(flow_refined, r0.mask)
            cx0, cy0 = _region_centroid(r0.mask)
            cx1, cy1 = _region_centroid(r1.mask)
            flow_refined[r0.mask, 0] += cx1 - (cx0 + du)
            flow_refined[r0.mask, 1] += cy1 - (cy0 + dv)

    topo_warp = warp_topology_by_flow(topo_t0, flow_refined)
    class_t0 = build_class_topology_map(regions_t0, h, w)
    class_warp = warp_topology_by_flow(class_t0, flow_refined)
    return TopologyAlignResult(
        topo_t0,
        topo_t1,
        flow_refined,
        coarse,
        _topology_iou(topo_warp, topo_t1),
        _class_topology_iou(class_warp, class_t1),
        _coarse_target_topo_iou(coarse, geo) if geo is not None else 0.0,
        match_mode,
        geo.target_topo if geo is not None else None,
    )


def _pair_iou(r0: MegaRegion, r1: MegaRegion, flow: np.ndarray) -> float:
    warped = _flow_warp_mask(r0.mask, flow)
    union = warped | r1.mask
    if not np.any(union):
        return 0.0
    return float((warped & r1.mask).sum() / union.sum())


def match_source_regions(
    regions_t0: list[MegaRegion],
    regions_t1: list[MegaRegion],
    flow: np.ndarray,
    min_iou: float = 0.04,
    topology_align: bool = True,
    topology_fallback: bool = True,
    topology_match_mode: str = "hungarian_geo",
    geo: GeoMatchContext | None = None,
    use_target_depth: bool = True,
) -> tuple[list[MatchedPair], list[MegaRegion], list[MegaRegion], TopologyAlignResult | None]:
    h, w = flow.shape[:2]
    topo_result: TopologyAlignResult | None = None
    flow_use = flow

    match_mode = topology_match_mode
    if match_mode in ("hungarian_geo", "hungarian_target_topo") and geo is None:
        match_mode = "hungarian_centroid"
    if match_mode == "hungarian_target_topo" and (geo is None or geo.target_topo is None):
        match_mode = "hungarian_geo" if geo is not None else "hungarian_centroid"

    if topology_align and regions_t0 and regions_t1 and match_mode != "none":
        topo_result = align_topology_flow(
            regions_t0, regions_t1, flow, h, w,
            match_mode, min_iou, geo, use_target_depth,
        )
        flow_use = topo_result.flow_refined

    used_t1: set[int] = set()
    pairs: list[MatchedPair] = []
    coarse_ids = {
        (id(r0), id(r1)) for r0, r1 in (topo_result.coarse_pairs if topo_result else [])
    }

    def _region_iou(r0: MegaRegion, r1: MegaRegion) -> float:
        if geo is not None and geo.target_topo is not None:
            t_iou = _pair_target_topo_iou(r0, r1, geo)
            if topology_match_mode in ("hungarian_geo", "greedy_iou"):
                return 0.65 * t_iou + 0.35 * _pair_geo_iou(r0, r1, geo)
            return t_iou
        if geo is not None and topology_match_mode in ("hungarian_geo", "greedy_iou"):
            return _pair_geo_iou(r0, r1, geo)
        return _pair_iou(r0, r1, flow_use)

    for r0 in sorted(regions_t0, key=lambda r: -int(r.mask.sum())):
        best_score, best_j, best_r1 = -1.0, -1, None
        for j, r1 in enumerate(regions_t1):
            if j in used_t1 or not _classes_match(r0, r1):
                continue
            iou = _region_iou(r0, r1)
            score = iou + (0.08 if (id(r0), id(r1)) in coarse_ids else 0.0)
            if geo is not None and use_target_depth and geo.target_topo is not None:
                score += 0.12 * _pair_target_layer_sim(r0, r1, geo)
                score -= 0.06 * _pair_target_topo_cost(r0, r1, geo, geo_weight=0.0)
            elif geo is not None and use_target_depth:
                score -= 0.04 * _pair_target_penalty(r0, r1, geo, use_target_depth)
            if score > best_score:
                best_score, best_j, best_r1 = score, j, r1
        if best_r1 is not None and best_score >= min_iou:
            used_t1.add(best_j)
            du, dv = _median_flow_on_region(r0, flow_use, geo, best_r1)
            pairs.append(MatchedPair(
                r0, best_r1, _region_iou(r0, best_r1), du, dv,
                from_topology=(id(r0), id(best_r1)) in coarse_ids,
            ))

    if topology_fallback and topo_result:
        matched_t0 = {id(p.reg_t0) for p in pairs}
        matched_t1 = {id(p.reg_t1) for p in pairs}
        for r0, r1 in topo_result.coarse_pairs:
            if id(r0) in matched_t0 or id(r1) in matched_t1:
                continue
            du, dv = _median_flow_on_region(r0, flow_use, geo, r1)
            pairs.append(MatchedPair(
                r0, r1, _region_iou(r0, r1), du, dv, from_topology=True,
            ))

    matched_t0 = {id(p.reg_t0) for p in pairs}
    matched_t1 = {id(p.reg_t1) for p in pairs}
    unmatched_t0 = [r for r in regions_t0 if id(r) not in matched_t0]
    unmatched_t1 = [r for r in regions_t1 if id(r) not in matched_t1]
    return pairs, unmatched_t0, unmatched_t1, topo_result


def _source_mask_on_target(
    src_mask: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    h, w = src_mask.shape
    finite = np.isfinite(map_x) & np.isfinite(map_y)
    sx = np.clip(np.round(np.where(finite, map_x, 0.0)).astype(np.int32), 0, w - 1)
    sy = np.clip(np.round(np.where(finite, map_y, 0.0)).astype(np.int32), 0, h - 1)
    return valid & finite & src_mask[sy, sx]


def _blend_pair_on_target(
    img_t0: np.ndarray,
    img_t1: np.ndarray,
    c2w_t0: np.ndarray,
    c2w_t1: np.ndarray,
    c2w_tgt: np.ndarray,
    K: np.ndarray,
    pair: MatchedPair,
    alpha: float,
    static_mean_only: bool,
    static_mean_weight: float,
    mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r0, r1 = pair.reg_t0, pair.reg_t1
    h, w = img_t0.shape[:2]

    if static_mean_only and r0.class_name in STATIC_SEMANTIC:
        fb = farneback_alpha(img_t0, img_t1, alpha).astype(np.float32)
        layer = static_mean_weight * mean.astype(np.float32) + (1.0 - static_mean_weight) * fb
        mx0, my0, v0 = backwarp_map(c2w_tgt, c2w_t0, K, r0.z_median, h, w)
        mask = _source_mask_on_target(r0.mask, mx0, my0, v0)
        return layer, mask

    mx0, my0, v0 = backwarp_map(c2w_tgt, c2w_t0, K, r0.z_median, h, w)
    mx1, my1, v1 = backwarp_map(c2w_tgt, c2w_t1, K, r1.z_median, h, w)
    w0 = remap_rgb(img_t0, mx0, my0).astype(np.float32)
    w1 = remap_rgb(img_t1, mx1, my1).astype(np.float32)
    layer = (1.0 - alpha) * w0 + alpha * w1
    m0 = _source_mask_on_target(r0.mask, mx0, my0, v0)
    m1 = _source_mask_on_target(r1.mask, mx1, my1, v1)
    mask = m0 & m1 if pair.from_topology and pair.iou >= 0.06 else (m0 | m1)
    return layer, mask


def _blend_single_source_on_target(
    img: np.ndarray,
    c2w_src: np.ndarray,
    c2w_tgt: np.ndarray,
    K: np.ndarray,
    reg: MegaRegion,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    mx, my, valid = backwarp_map(c2w_tgt, c2w_src, K, reg.z_median, h, w)
    warped = remap_rgb(img, mx, my).astype(np.float32)
    mask = _source_mask_on_target(reg.mask, mx, my, valid)
    return warped, mask


@dataclass
class SourceMatchLayer:
    label: str
    layer: np.ndarray
    mask: np.ndarray
    z_sort: float
    seg_id: int


def _collect_source_match_layers(
    img_t0: np.ndarray,
    img_t1: np.ndarray,
    c2w_t0: np.ndarray,
    c2w_t1: np.ndarray,
    c2w_tgt: np.ndarray,
    K: np.ndarray,
    pairs: list[MatchedPair],
    unmatched_t0: list[MegaRegion],
    unmatched_t1: list[MegaRegion],
    alpha: float,
    cfg: MegaParallaxConfig,
    mean: np.ndarray,
) -> list[SourceMatchLayer]:
    layers: list[SourceMatchLayer] = []
    for i, pair in enumerate(pairs):
        layer, mask = _blend_pair_on_target(
            img_t0, img_t1, c2w_t0, c2w_t1, c2w_tgt, K, pair, alpha,
            cfg.static_use_mean_only, cfg.static_mean_weight, mean,
        )
        r0, r1 = pair.reg_t0, pair.reg_t1
        label = f"P{i} {r0.class_name}|{r1.class_name} z={0.5*(r0.z_median+r1.z_median):.0f}m"
        layers.append(SourceMatchLayer(label, layer, mask, 0.5 * (r0.z_median + r1.z_median), i + 1))

    base = len(pairs) + 1
    for j, reg in enumerate(unmatched_t0):
        layer, mask = _blend_single_source_on_target(img_t0, c2w_t0, c2w_tgt, K, reg)
        layers.append(SourceMatchLayer(
            f"U0{j} {reg.class_name} z={reg.z_median:.0f}m", layer, mask, reg.z_median, base + j,
        ))
    base += len(unmatched_t0)
    for j, reg in enumerate(unmatched_t1):
        layer, mask = _blend_single_source_on_target(img_t1, c2w_t1, c2w_tgt, K, reg)
        layers.append(SourceMatchLayer(
            f"U1{j} {reg.class_name} z={reg.z_median:.0f}m", layer, mask, reg.z_median, base + j,
        ))
    return layers


def _composite_source_layers(
    layers: list[SourceMatchLayer],
    img_t0: np.ndarray,
    img_t1: np.ndarray,
    alpha: float,
    track_segments: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    h, w = img_t0.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.float32)
    filled = np.zeros((h, w), dtype=bool)
    seg_ids = np.full((h, w), -1, dtype=np.int16) if track_segments else None
    labels = ["background"]

    for lay in sorted(layers, key=lambda x: x.z_sort):
        write = lay.mask & (~filled)
        if not np.any(write):
            continue
        out[write] = lay.layer[write]
        filled |= write
        if seg_ids is not None:
            seg_ids[write] = lay.seg_id
        labels.append(lay.label)

    if not np.all(filled):
        fallback = farneback_alpha(img_t0, img_t1, alpha).astype(np.float32)
        hole = ~filled
        out[hole] = fallback[hole]
        if seg_ids is not None:
            seg_ids[hole] = 9999
            labels.append("fallback farneback")

    return out.clip(0, 255).astype(np.uint8), seg_ids, labels


def mega_parallax_predict_source_match(
    img_t0: np.ndarray,
    img_t1: np.ndarray,
    c2w_t0: np.ndarray,
    c2w_t1: np.ndarray,
    c2w_tgt: np.ndarray,
    K: np.ndarray,
    depth_t0: np.ndarray,
    depth_t1: np.ndarray,
    sem_t0: np.ndarray,
    sem_t1: np.ndarray,
    alpha: float,
    cfg: MegaParallaxConfig | None = None,
    track_segments: bool = False,
    depth_tgt: np.ndarray | None = None,
) -> tuple[np.ndarray, list[MegaRegion], list[MegaRegion], list[MatchedPair], np.ndarray | None, list[str], TopologyAlignResult | None]:
    """Зоны на t0/t1 → topology align (pose + target LiDAR) → match → mid на target pose."""
    cfg = cfg or MegaParallaxConfig()
    mean = ((img_t0.astype(np.float32) + img_t1.astype(np.float32)) * 0.5).astype(np.uint8)
    h, w = img_t0.shape[:2]

    g0 = cv2.cvtColor(img_t0, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(img_t1, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    geo = build_geo_match_context(
        c2w_t0, c2w_t1, c2w_tgt, K, depth_tgt, alpha, h, w, cfg.n_depth_layers,
    )
    regions_t0 = make_mega_regions(depth_t0, sem_t0, cfg.n_depth_layers, cfg.min_region_pixels, cfg)
    regions_t1 = make_mega_regions(depth_t1, sem_t1, cfg.n_depth_layers, cfg.min_region_pixels, cfg)
    pairs, unmatched_t0, unmatched_t1, topo_result = match_source_regions(
        regions_t0, regions_t1, flow, cfg.match_min_iou,
        cfg.topology_align, cfg.topology_fallback, cfg.topology_match_mode,
        geo, cfg.use_target_depth_match,
    )

    layers = _collect_source_match_layers(
        img_t0, img_t1, c2w_t0, c2w_t1, c2w_tgt, K,
        pairs, unmatched_t0, unmatched_t1, alpha, cfg, mean,
    )
    pred, seg_ids, labels = _composite_source_layers(layers, img_t0, img_t1, alpha, track_segments)
    return pred, regions_t0, regions_t1, pairs, seg_ids, labels, topo_result


def build_t0_topology_on_target(
    regions_t0: list[MegaRegion],
    c2w_t0: np.ndarray,
    c2w_tgt: np.ndarray,
    K: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, list[str]]:
    """Ожидаемая топология: зоны t0, спроецированные на target pose (эталон для сравнения)."""
    h, w = shape
    topo = np.full((h, w), -1, dtype=np.int16)
    labels = ["bg"]
    order = sorted(regions_t0, key=lambda r: r.z_median)
    for i, reg in enumerate(order):
        mx, my, valid = backwarp_map(c2w_tgt, c2w_t0, K, reg.z_median, h, w)
        m = _source_mask_on_target(reg.mask, mx, my, valid)
        topo[m] = i + 1
        labels.append(f"T0#{i} {reg.class_name} z={reg.z_median:.0f}m")
    return topo, labels


def _warp_blend_region(
    img_t0: np.ndarray,
    img_t1: np.ndarray,
    c2w_t0: np.ndarray,
    c2w_t1: np.ndarray,
    c2w_tgt: np.ndarray,
    K: np.ndarray,
    z_layer: float,
    alpha_time: float,
    layer_idx: int,
    n_layers: int,
    z_ref: float,
    mean: np.ndarray,
    cfg: AlphaTuneConfig,
    bcfg: BlurTuneConfig,
    flow_mag: np.ndarray,
    mask: np.ndarray,
    static_mean_only: bool,
    static_mean_weight: float,
    class_name: str,
) -> np.ndarray:
    a_blend, _ = layer_alpha(z_layer, layer_idx, n_layers, alpha_time, z_ref, cfg)
    mw = cfg.mean_w if cfg.mean_w > 0 else 0.0

    if static_mean_only and class_name in STATIC_SEMANTIC:
        fb = farneback_alpha(img_t0, img_t1, alpha_time).astype(np.float32)
        w = static_mean_weight
        layer_pred = w * mean.astype(np.float32) + (1.0 - w) * fb
        sig = layer_sigma(layer_idx, n_layers, flow_mag, mask, bcfg)
        if sig <= 1e-4:
            return layer_pred
        blurred = cv2.GaussianBlur(layer_pred.astype(np.uint8), (0, 0), sigmaX=sig, sigmaY=sig).astype(np.float32)
        out = layer_pred.copy()
        out[mask] = blurred[mask]
        return out

    mx0, my0, _ = backwarp_map(c2w_tgt, c2w_t0, K, z_layer, img_t0.shape[0], img_t0.shape[1])
    mx1, my1, _ = backwarp_map(c2w_tgt, c2w_t1, K, z_layer, img_t0.shape[0], img_t0.shape[1])
    w0 = remap_rgb(img_t0, mx0, my0).astype(np.float32)
    w1 = remap_rgb(img_t1, mx1, my1).astype(np.float32)
    layer_pred = (1 - a_blend) * w0 + a_blend * w1
    if mw > 0:
        layer_pred = (1 - mw) * layer_pred + mw * mean.astype(np.float32)

    sig = layer_sigma(layer_idx, n_layers, flow_mag, mask, bcfg)
    if sig <= 1e-4:
        return layer_pred
    blurred = cv2.GaussianBlur(layer_pred.astype(np.uint8), (0, 0), sigmaX=sig, sigmaY=sig).astype(np.float32)
    out = layer_pred.copy()
    out[mask] = blurred[mask]
    return out


def mega_parallax_predict(
    img_t0: np.ndarray,
    img_t1: np.ndarray,
    c2w_t0: np.ndarray,
    c2w_t1: np.ndarray,
    c2w_tgt: np.ndarray,
    K: np.ndarray,
    depth: np.ndarray,
    alpha: float,
    sem_map: np.ndarray | None = None,
    cfg: MegaParallaxConfig | None = None,
) -> tuple[np.ndarray, list[MegaRegion], list[np.ndarray]]:
    """
    Returns: composite RGB, list of regions, per-region predictions.
    """
    cfg = cfg or MegaParallaxConfig()
    h, w = depth.shape
    mean = ((img_t0.astype(np.float32) + img_t1.astype(np.float32)) * 0.5).astype(np.uint8)

    g0 = cv2.cvtColor(img_t0, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(img_t1, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    valid_depth = np.isfinite(depth) & (depth > 0)
    z_ref = float(np.median(depth[valid_depth])) if np.any(valid_depth) else 10.0

    sem = sem_map if sem_map is not None else np.full((h, w), -1, dtype=np.int16)
    if cfg.allowed_classes:
        allowed_ids = {NAME_TO_ID[c] for c in cfg.allowed_classes if c in NAME_TO_ID}
        sem = np.where(np.isin(sem, list(allowed_ids)), sem, -1)

    regions = make_mega_regions(depth, sem, cfg.n_depth_layers, cfg.min_region_pixels, cfg)
    n_layers = max(cfg.n_depth_layers, 1)

    out = np.zeros((h, w, 3), dtype=np.float32)
    filled = np.zeros((h, w), dtype=bool)
    region_preds: list[np.ndarray] = []

    for reg in regions:
        layer_pred = _warp_blend_region(
            img_t0,
            img_t1,
            c2w_t0,
            c2w_t1,
            c2w_tgt,
            K,
            reg.z_median,
            alpha,
            reg.depth_layer_idx,
            n_layers,
            z_ref,
            mean,
            cfg.alpha_cfg,
            cfg.blur_cfg,
            flow_mag,
            reg.mask,
            cfg.static_use_mean_only,
            cfg.static_mean_weight,
            reg.class_name,
        )
        out[reg.mask] = layer_pred[reg.mask]
        filled |= reg.mask
        region_preds.append(layer_pred)

    # Пиксели с глубиной, но без семантики — depth-only слои
    unlabeled = valid_depth & (sem < 0) & (~filled)
    if np.any(unlabeled):
        depth_only_layers = make_depth_layers(depth, cfg.n_depth_layers)
        for i, (dmask, z_layer) in enumerate(depth_only_layers):
            mask = dmask & unlabeled
            if not np.any(mask):
                continue
            layer_pred = _warp_blend_region(
                img_t0,
                img_t1,
                c2w_t0,
                c2w_t1,
                c2w_tgt,
                K,
                z_layer,
                alpha,
                i,
                n_layers,
                z_ref,
                mean,
                cfg.alpha_cfg,
                cfg.blur_cfg,
                flow_mag,
                mask,
                False,
                cfg.static_mean_weight,
                "unknown",
            )
            out[mask] = layer_pred[mask]
            filled |= mask

    if not np.all(filled):
        fallback = farneback_alpha(img_t0, img_t1, alpha).astype(np.float32)
        hole = ~filled
        out[hole] = fallback[hole]

    return out.clip(0, 255).astype(np.uint8), regions, region_preds


def mega_parallax_from_sample(
    sample_dir: Path,
    split: str,
    semantic_jsonl: Path,
    alpha_cfg: AlphaTuneConfig | None = None,
    blur_cfg: BlurTuneConfig | None = None,
    mega_cfg: MegaParallaxConfig | None = None,
) -> tuple[np.ndarray, list[MegaRegion]]:
    from lib.layered_parallax import get_lidar_depth, intrinsics_to_K, load_rgb

    meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
    cam = meta["target_camera"]
    ts = meta["timestamps_ns"]
    alpha = float((ts["target"] - ts["t0"]) / (ts["t1"] - ts["t0"]))

    img_t0 = load_rgb(sample_dir / "input" / "t0" / f"{cam}.jpg")
    img_t1 = load_rgb(sample_dir / "input" / "t1" / f"{cam}.jpg")
    h, w = img_t0.shape[:2]

    K = intrinsics_to_K(meta["intrinsics"][cam])
    c2w_t0 = np.array(meta["poses_c2w"]["t0"][cam], dtype=np.float64)
    c2w_t1 = np.array(meta["poses_c2w"]["t1"][cam], dtype=np.float64)
    c2w_tgt = np.array(meta["poses_c2w"]["target"][cam], dtype=np.float64)

    base = mega_cfg or MegaParallaxConfig()
    if alpha_cfg:
        base.alpha_cfg = alpha_cfg
    if blur_cfg:
        base.blur_cfg = blur_cfg

    if base.zoning_frame == "source_match":
        depth_t0 = get_lidar_depth(sample_dir, cam, "t0")
        depth_t1 = get_lidar_depth(sample_dir, cam, "t1")
        depth_tgt = get_lidar_depth(sample_dir, cam, "target")
        sid = meta["sample_id"]
        rel_t0 = f"{split}/{sid}/input/t0/{cam}.jpg"
        rel_t1 = f"{split}/{sid}/input/t1/{cam}.jpg"
        allowed = base.allowed_classes
        sem_t0 = rasterize_semantic_map(load_semantic_regions(semantic_jsonl, rel_t0), h, w, allowed)
        sem_t1 = rasterize_semantic_map(load_semantic_regions(semantic_jsonl, rel_t1), h, w, allowed)
        pred, regions_t0, regions_t1, _pairs, _seg, _labels, _topo = mega_parallax_predict_source_match(
            img_t0, img_t1, c2w_t0, c2w_t1, c2w_tgt, K,
            depth_t0, depth_t1, sem_t0, sem_t1, alpha, base, depth_tgt=depth_tgt,
        )
        return pred, regions_t0 + regions_t1

    depth = get_lidar_depth(sample_dir, cam, "target")
    rel = f"{split}/{meta['sample_id']}/target/{cam}.jpg"
    regions_json = load_semantic_regions(semantic_jsonl, rel)
    sem_map = rasterize_semantic_map(regions_json, h, w, base.allowed_classes)

    pred, mega_regions, _ = mega_parallax_predict(
        img_t0, img_t1, c2w_t0, c2w_t1, c2w_tgt, K, depth, alpha, sem_map, base,
    )
    return pred, mega_regions
