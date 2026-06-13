# Dataset sample — `final_dataset_v5` (+ all precomputes)

A small, self-contained **6-sample slice** of the full multi-camera + LiDAR dataset
**and every precompute that goes with it**, kept in the repo for documentation,
visualization and quick sanity checks. The full dataset (~320 GB) is **not** stored
here — these six samples are enough to show the exact on-disk format, the prediction
task, and the full precompute pipeline.

## Task

Novel-view / future-frame synthesis. Given the surround-camera views and an
aggregated LiDAR cloud at two timestamps `t0` and `t1`, predict the image of one
**target camera** at an intermediate timestamp `t_target` (`delta_s = 1.0 s`).

## Layout (mirrors the original dataset)

After running the copy script, `cv_dataset/` mirrors the original folder tree. For a
given sample id `<id>` the data is spread across several top-level trees:

| Folder | Per-sample contents | What it is |
|---|---|---|
| `final_dataset_v5_participants/train/<id>/` | `input/t0/*.jpg`, `input/t1/*.jpg` (6 cameras each), `input/lidar.npz`, `target/<cam>.jpg`, `meta.json` | **Base data**: surround views at t0/t1, aggregated LiDAR, target image, calibration. |
| `annotations/detect/labels/{train,test}/<id>/` | `…/<cam>.txt` | YOLO **detection** labels (per image, where available). |
| `annotations/segment/labels_seg/{train,test}/<id>/` | `…/<cam>.txt` | **Segmentation** labels (per image, where available). |
| `rife_refinement_baked/train/<id>/` | `i0.jpg`, `i2.jpg`, `gt.jpg`, `irife.jpg`, `d0..d2.npy` + `.png`, `meta.json` | RIFE frame-interpolation baked outputs + flow/depth maps. |
| `multiview_warps/train/<id>/` | `warp_<cam>_t{0,1}.npy`, `mask_<cam>_t{0,1}.npy`, `consensus*.npy`, `coverage.npy`, `soft_coverage.npy`, `visibility_count.npy`, `confidence.npy`, `disagreement.npy`, `warp_summary.json` | Multi-view geometric warps into the target view + consensus / confidence maps. |
| `precomputed_static_far/<id>/` | `static_far.npz`, `meta.json` | Precomputed static / far-field component. |
| `precomputed_lidar_trust/train/<id>/` | `lidar_trust.npy` | Per-pixel LiDAR trust / reliability map. |
| `side_warp_mix_v1/train/<id>/` | `warp_mix.jpg` | Side-camera warp mixture preview. |
| `pseudo_v2_train_rgb/<id>/` | `pred_rgb.npy`, `target_rgb.npy`, `meta.json` | Pseudo-label RGB (prediction + target arrays). |
| `rife_guided_blend_v1/train/<id>/` | `guided_in0.jpg`, `guided_in1.jpg`, `rife_guided_out.jpg`, `rife_raw_out.jpg`, `ours_pred.jpg`, `target.jpg` | RIFE-guided blend comparison (inputs, raw vs guided, ours vs target). |

There is also a third annotation type, **`annotations/semantic/`**, which is stored
only as aggregate files (no per-sample folders) — see "Global files" below.

Notes:
- Two trees (`precomputed_static_far`, `pseudo_v2_train_rgb`) store the sample
  **directly** under the folder (no `train/` level); the rest use `…/{train,test}/<id>/`.
- Coverage varies per sample — not every precompute exists for every sample, and
  annotations are sparse (only some cameras are labelled). Missing pieces are skipped.
- The dataset has both `train` and `test` splits. The six samples kept here are all
  from `train`; the copy script also keeps any global/aggregate files for both splits.
- Images are pinhole, ~1024×540 px. Intrinsics are per-camera (see `meta.json`).

## Global files (kept in full)

These are **not** tied to a single sample, so the copy script keeps them entirely —
they describe the task, the labelling configs, dataset-wide summaries and build logs:

- `final_dataset_v5_participants/TASK.md` — task description.
- `annotations/detect/` — `config_used.yaml`, `detections.csv`, `summary_by_class.csv`, `run_meta.json`.
- `annotations/segment/` — `config_used.yaml`, `segments.csv`, `summary_by_class.csv`, `run_meta.json`.
- `annotations/semantic/` — `config_used.yaml`, `semantic_regions.csv`, `summary_by_class.csv`, `run_meta.json`.
- `rife_refinement_baked/` — `manifest.json`, `manifest_test.json`, `bake.log`.
- `multiview_warps/` — `batch.log`.
- `precomputed_static_far/` — `manifest.json`, `manifest_train.json`, `manifest_test.json`.
- `rife_guided_blend_v1/` — `blend_sweep_val.json`.

The script copies **every** file that is not inside a per-sample folder, so any global
file not listed here is preserved too.

## `meta.json` fields (base data)

| Field | Meaning |
|---|---|
| `sample_id` | Unique sample id (`<scene>__<NNN>`). |
| `scene` | Source drive / recording id. |
| `delta_s` | Seconds between `t0` and `t1` (1.0). |
| `target_camera` | Which camera the `target/` image belongs to. |
| `frame_convention` | Coordinate conventions (OpenCV camera axes; SDG `world_3d` world frame; LiDAR in world frame). |
| `lidar_info` | `n_points`, `n_sweeps`, `t0_ns`, `t1_ns` for the aggregated cloud. |
| `timestamps_ns` | Nanosecond timestamps for `t0`, `t1`, `target`. |
| `sync_ids` | Per-camera frame sync ids at `t0`, `t1`, and the target. |
| `intrinsics` | Per-camera `fx, fy, cx, cy, width, height, distortion_model, distortion_coeffs`. |
| `poses_c2w` | Per-camera **camera-to-world** 4×4 matrices at `t0`, `t1`, `target`. |

## Samples in this slice

One sample per target camera, each from a different scene/participant:

| Sample id | Target camera |
|---|---|
| `2025-10-01_11_45_46_12_19_35_robb_1759310438799892000__000`    | `front` |
| `2025-10-02_12_28_59_12_33_15_luka_1759397924100053000__000`    | `rear` |
| `2025-10-01_18_29_52_19_31_28_kynde_1759339701900197000__000`   | `right_bwd` |
| `2025-10-10_06_36_32_09_00_01_natelio_1760076111099953000__000` | `right_fwd` |
| `2025-10-11_16_12_04_18_35_14_crozby_1760197023299977000__000`  | `left_fwd` |
| `2025-10-12_11_30_56_12_10_38_anabel_1760263528700021000__000`  | `left_bwd` |

## How to build the slice

`copy_samples.ps1` walks the full dataset (`%USERPROFILE%\Downloads\cv_dataset`) once,
top-down, and:

- copies each of the six selected sample folders **whole**, wherever they appear;
- copies **every global file** (anything not inside a per-sample folder) preserving its path;
- **skips** all other samples — that's the ~320 GB being dropped.

Everything lands in `cv_dataset/` here, mirroring the original layout. Run
`copy_samples.bat` (double-click) or the `.ps1` directly. It prints a per-sample
summary plus the number of global files copied. Verify, then delete the full dataset.

## Visualize

```bash
pip install numpy matplotlib pillow
# point at a base sample folder:
python visualize_sample.py cv_dataset/final_dataset_v5_participants/train/2025-10-01_11_45_46_12_19_35_robb_1759310438799892000__000
# or render all six at once:
python visualize_sample.py --all
```

The script renders the 6 input views at `t0` and `t1`, the target image, and — when
the matching precompute trees are present — a second panel with pipeline artifacts
(RIFE output, multi-view warp mix, RIFE-guided blend, pseudo-label prediction).
Contact sheets are written next to the script.
