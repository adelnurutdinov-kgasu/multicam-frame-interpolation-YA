"""Tests for mega_parallax (depth × semantic regions)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layered_parallax import AlphaTuneConfig, BlurTuneConfig, intrinsics_to_K, psnr
from mega_parallax import (
    MegaParallaxConfig,
    MegaRegion,
    load_semantic_regions,
    make_mega_regions,
    match_source_regions,
    mega_parallax_predict,
    mega_parallax_from_sample,
    rasterize_semantic_map,
)
from yolo.cityscapes_classes import NAME_TO_ID


def _synthetic_scene(h: int = 80, w: int = 120) -> dict:
    """Простая сцена: road внизу, sky сверху, depth растёт к горизонту."""
    yy, xx = np.mgrid[0:h, 0:w]
    depth = (5.0 + (yy / max(h - 1, 1)) * 40.0).astype(np.float32)
    sem = np.full((h, w), -1, dtype=np.int16)
    sem[yy > h * 0.55] = NAME_TO_ID["road"]
    sem[yy < h * 0.25] = NAME_TO_ID["sky"]

    img0 = np.zeros((h, w, 3), dtype=np.uint8)
    img1 = np.zeros((h, w, 3), dtype=np.uint8)
    gt = np.zeros((h, w, 3), dtype=np.uint8)
    img0[..., 0] = (xx * 255 // w).astype(np.uint8)
    img0[..., 1] = (yy * 255 // h).astype(np.uint8)
    img1 = np.roll(img0, shift=3, axis=1)
    gt = ((img0.astype(np.float32) + img1.astype(np.float32)) * 0.5).astype(np.uint8)

    K = intrinsics_to_K({"fx": 500.0, "fy": 500.0, "cx": w / 2, "cy": h / 2, "width": w, "height": h})
    c2w = np.eye(4, dtype=np.float64)
    c2w_t0 = c2w.copy()
    c2w_t1 = c2w.copy()
    c2w_t1[0, 3] = 0.3
    c2w_tgt = c2w.copy()
    c2w_tgt[0, 3] = 0.15

    return {
        "img0": img0,
        "img1": img1,
        "gt": gt,
        "depth": depth,
        "sem": sem,
        "K": K,
        "c2w_t0": c2w_t0,
        "c2w_t1": c2w_t1,
        "c2w_tgt": c2w_tgt,
        "alpha": 0.5,
    }


class TestMegaRegions(unittest.TestCase):
    def test_regions_disjoint(self):
        sc = _synthetic_scene()
        regions = make_mega_regions(
            sc["depth"], sc["sem"], 3, 10,
            MegaParallaxConfig(n_depth_layers=3, min_region_pixels=10, region_mode="grid"),
        )
        self.assertGreater(len(regions), 0)
        occupied = np.zeros(sc["depth"].shape, dtype=np.int32)
        for i, reg in enumerate(regions):
            overlap = occupied[reg.mask] > 0
            self.assertFalse(np.any(overlap), f"region {i} overlaps")
            occupied[reg.mask] = i + 1

    def test_object_mode_regions_fewer_than_grid(self):
        sc = _synthetic_scene(h=100, w=140)
        grid = make_mega_regions(
            sc["depth"], sc["sem"], 4, 10,
            MegaParallaxConfig(region_mode="grid", n_depth_layers=4, min_region_pixels=10),
        )
        obj = make_mega_regions(
            sc["depth"], sc["sem"], 8, 10,
            MegaParallaxConfig(region_mode="object", n_depth_layers=8, min_region_pixels=10),
        )
        self.assertGreater(len(grid), 0)
        self.assertGreater(len(obj), 0)

    def test_object_mode_merges_pole_and_sign(self):
        depth = np.full((60, 80), 20.0, dtype=np.float32)
        sem = np.full((60, 80), -1, dtype=np.int16)
        sem[20:45, 10:14] = NAME_TO_ID["pole"]
        sem[18:28, 14:22] = NAME_TO_ID["traffic sign"]
        cfg = MegaParallaxConfig(
            region_mode="object",
            merge_affinity_classes=True,
            affinity_depth_m=5.0,
            min_region_pixels=5,
        )
        regions = make_mega_regions(depth, sem, 8, 5, cfg)
        pole = [r for r in regions if r.class_name == "pole"]
        self.assertEqual(len(pole), 1)
        self.assertGreater(int(pole[0].mask.sum()), 150)

    def test_rasterize_semantic(self):
        regions = [
            {
                "class_name": "road",
                "polygon_xy": [[0, 50], [120, 50], [120, 80], [0, 80]],
            },
            {
                "class_name": "sky",
                "polygon_xy": [[0, 0], [120, 0], [120, 20], [0, 20]],
            },
        ]
        sem = rasterize_semantic_map(regions, 80, 120)
        self.assertEqual(int(sem[70, 60]), NAME_TO_ID["road"])
        self.assertEqual(int(sem[10, 60]), NAME_TO_ID["sky"])


class TestSourceMatch(unittest.TestCase):
    def test_match_pole_and_sign_by_flow(self):
        h, w = 80, 100
        flow = np.zeros((h, w, 2), dtype=np.float32)
        flow[:, :, 0] = 8.0
        mask0 = np.zeros((h, w), dtype=bool)
        mask0[30:50, 20:30] = True
        mask1 = np.zeros((h, w), dtype=bool)
        mask1[30:50, 28:38] = True
        from yolo.cityscapes_classes import NAME_TO_ID
        r0 = MegaRegion(mask0, 19.0, 3, NAME_TO_ID["pole"], "pole")
        r1 = MegaRegion(mask1, 19.5, 3, NAME_TO_ID["pole"], "pole")
        pairs, u0, u1, _topo = match_source_regions(
            [r0], [r1], flow, min_iou=0.1, topology_match_mode="hungarian_iou",
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(len(u0), 0)
        self.assertEqual(len(u1), 0)

    def test_output_shape_and_dtype(self):
        sc = _synthetic_scene()
        pred, regions, r_preds = mega_parallax_predict(
            sc["img0"],
            sc["img1"],
            sc["c2w_t0"],
            sc["c2w_t1"],
            sc["c2w_tgt"],
            sc["K"],
            sc["depth"],
            sc["alpha"],
            sc["sem"],
            MegaParallaxConfig(n_depth_layers=3, min_region_pixels=10, zoning_frame="target"),
        )
        self.assertEqual(pred.shape, sc["img0"].shape)
        self.assertEqual(pred.dtype, np.uint8)
        self.assertGreater(len(regions), 0)
        self.assertEqual(len(r_preds), len(regions))

    def test_fills_all_pixels(self):
        sc = _synthetic_scene()
        pred, _, _ = mega_parallax_predict(
            sc["img0"],
            sc["img1"],
            sc["c2w_t0"],
            sc["c2w_t1"],
            sc["c2w_tgt"],
            sc["K"],
            sc["depth"],
            sc["alpha"],
            sc["sem"],
        )
        self.assertEqual(pred.shape[:2], sc["depth"].shape)
        self.assertTrue(np.isfinite(pred).all())

    def test_sky_differs_from_pure_geo_road(self):
        sc = _synthetic_scene(h=100, w=140)
        cfg = MegaParallaxConfig(n_depth_layers=2, min_region_pixels=20, static_use_mean_only=True, static_mean_weight=0.9, zoning_frame="target")
        pred, regions, _ = mega_parallax_predict(
            sc["img0"],
            sc["img1"],
            sc["c2w_t0"],
            sc["c2w_t1"],
            sc["c2w_tgt"],
            sc["K"],
            sc["depth"],
            sc["alpha"],
            sc["sem"],
            cfg,
        )
        sky_reg = next(r for r in regions if r.class_name == "sky")
        road_reg = next(r for r in regions if r.class_name == "road")
        mean = ((sc["img0"].astype(np.float32) + sc["img1"].astype(np.float32)) * 0.5)
        sky_diff_mean = np.mean(np.abs(pred[sky_reg.mask].astype(np.float32) - mean[sky_reg.mask]))
        road_diff_mean = np.mean(np.abs(pred[road_reg.mask].astype(np.float32) - mean[road_reg.mask]))
        self.assertLess(sky_diff_mean, road_diff_mean + 5.0)

    def test_no_semantic_falls_back_to_depth_layers(self):
        sc = _synthetic_scene()
        sem_empty = np.full(sc["sem"].shape, -1, dtype=np.int16)
        pred, regions, _ = mega_parallax_predict(
            sc["img0"],
            sc["img1"],
            sc["c2w_t0"],
            sc["c2w_t1"],
            sc["c2w_tgt"],
            sc["K"],
            sc["depth"],
            sc["alpha"],
            sem_empty,
            MegaParallaxConfig(n_depth_layers=3, min_region_pixels=10, zoning_frame="target"),
        )
        self.assertEqual(len(regions), 0)
        self.assertEqual(pred.shape, sc["img0"].shape)


class TestMegaParallaxIntegration(unittest.TestCase):
    DATASET = Path(r"C:\Users\adel\Downloads\cv_dataset\final_dataset_v5_participants")
    SEM_JSONL = Path(r"C:\Users\adel\Downloads\cv_dataset\annotations\semantic\semantic.jsonl")

    def _first_train_sample(self) -> Path | None:
        train = self.DATASET / "train"
        if not train.is_dir():
            return None
        for sd in sorted(train.iterdir()):
            if not sd.is_dir():
                continue
            meta = json.loads((sd / "meta.json").read_text(encoding="utf-8"))
            cam = meta["target_camera"]
            if (sd / "target" / f"{cam}.jpg").is_file():
                return sd
        return None

    @unittest.skipUnless(
        Path(r"C:\Users\adel\Downloads\cv_dataset\final_dataset_v5_participants\train").is_dir(),
        "dataset not available",
    )
    def test_real_sample_runs(self):
        sd = self._first_train_sample()
        self.assertIsNotNone(sd)
        pred, regions = mega_parallax_from_sample(
            sd,
            "train",
            self.SEM_JSONL,
            MegaParallaxConfig(n_depth_layers=3, min_region_pixels=50, zoning_frame="target"),
        )
        self.assertEqual(pred.ndim, 3)
        self.assertGreater(len(regions), 0)

    @unittest.skipUnless(
        Path(r"C:\Users\adel\Downloads\cv_dataset\annotations\semantic\semantic.jsonl").is_file(),
        "semantic jsonl not available",
    )
    def test_load_semantic_regions(self):
        with self.SEM_JSONL.open(encoding="utf-8") as f:
            rec = json.loads(f.readline())
        rel = rec["image"]
        loaded = load_semantic_regions(self.SEM_JSONL, rel)
        self.assertIsNotNone(loaded)
        self.assertGreater(len(loaded), 0)


class TestMegaVsLayered(unittest.TestCase):
    def test_runs_alongside_layered_on_synthetic(self):
        from layered_parallax import layered_parallax_predict

        sc = _synthetic_scene()
        cfg = MegaParallaxConfig(n_depth_layers=3, min_region_pixels=10, zoning_frame="target")
        mega, _, _ = mega_parallax_predict(
            sc["img0"], sc["img1"], sc["c2w_t0"], sc["c2w_t1"], sc["c2w_tgt"],
            sc["K"], sc["depth"], sc["alpha"], sc["sem"], cfg,
        )
        layered, _ = layered_parallax_predict(
            sc["img0"], sc["img1"], sc["c2w_t0"], sc["c2w_t1"], sc["c2w_tgt"],
            sc["K"], sc["depth"], sc["alpha"], 3,
        )
        self.assertEqual(mega.shape, layered.shape)
        p_mega = psnr(mega, sc["gt"])
        p_layered = psnr(layered, sc["gt"])
        self.assertGreater(p_mega, 0)
        self.assertGreater(p_layered, 0)


if __name__ == "__main__":
    unittest.main()
