"""
Однократная реорганизация корня репозитория.
Запуск из корня: python tools/apply_repo_layout.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# (src relative to REPO, dst relative to REPO) — только если src существует
MOVES: list[tuple[str, str]] = [
    # docs
    ("README_CV_DATASET.md", "docs/README_CV_DATASET.md"),
    ("README_YOLO.md", "docs/README_YOLO.md"),
    ("TEST_INFERENCE_PIPELINE.md", "docs/TEST_INFERENCE_PIPELINE.md"),
    ("_dump_dataset.txt", "docs/archive/_dump_dataset.txt"),
    ("_dump_git_dataset.txt", "docs/archive/_dump_git_dataset.txt"),
    ("_recovered_06.txt", "docs/archive/_recovered_06.txt"),
    # configs tuning
    ("alpha_tune_best.json", "configs/tuning/alpha_tune_best.json"),
    ("blur_tune_best.json", "configs/tuning/blur_tune_best.json"),
    ("static_mask_tune_best.json", "configs/tuning/static_mask_tune_best.json"),
    ("far_boundary_sweep.json", "configs/tuning/far_boundary_sweep.json"),
    ("far_boundary_sweep.csv", "configs/tuning/far_boundary_sweep.csv"),
    ("compare_baselines_results.json", "configs/tuning/compare_baselines_results.json"),
    # lib
    ("consensus_kit.py", "lib/consensus_kit.py"),
    ("lidar_depth_map.py", "lib/lidar_depth_map.py"),
    ("lidar_density_mask.py", "lib/lidar_density_mask.py"),
    ("layered_parallax.py", "lib/layered_parallax.py"),
    ("mega_parallax.py", "lib/mega_parallax.py"),
    ("static_mask.py", "lib/static_mask.py"),
    ("ego_mask_extract.py", "lib/ego_mask_extract.py"),
    ("ego_mask_policy.py", "lib/ego_mask_policy.py"),
    # notebooks
    ("consensus_training.ipynb", "notebooks/training/consensus_training.ipynb"),
    ("consensus_training_side.ipynb", "notebooks/training/consensus_training_side.ipynb"),
    ("consensus_training_corrupt_backup.ipynb", "notebooks/training/consensus_training_corrupt_backup.ipynb"),
    ("consensus_ensemble_front_rear.ipynb", "notebooks/training/consensus_ensemble_front_rear.ipynb"),
    ("lidar_density_zones.ipynb", "notebooks/training/lidar_density_zones.ipynb"),
    ("lidar_rife_blend.ipynb", "notebooks/training/lidar_rife_blend.ipynb"),
    # stage2
    ("Второй этап/bake_refinement_assets.py", "scripts/stage2/bake_refinement_assets.py"),
    ("Второй этап/multiview_warping.py", "scripts/stage2/multiview_warping.py"),
    ("Второй этап/refine_v2.py", "scripts/stage2/refine_v2.py"),
    ("Второй этап/render_baked_depth.py", "scripts/stage2/render_baked_depth.py"),
    ("Второй этап/eval_consensus_tuned.py", "scripts/stage2/eval_consensus_tuned.py"),
    ("Второй этап/rife_depth_refinement_starter.ipynb", "notebooks/stage2/rife_depth_refinement_starter.ipynb"),
    # inference
    ("precompute_cv_split.py", "scripts/inference/precompute_cv_split.py"),
    ("run_test_consensus_inference.py", "scripts/inference/run_test_consensus_inference.py"),
    ("run_test_blend_blur.py", "scripts/inference/run_test_blend_blur.py"),
    ("export_submission_blend50.py", "scripts/inference/export_submission_blend50.py"),
    ("flip_mirror_lidar_masks.py", "scripts/inference/flip_mirror_lidar_masks.py"),
    ("run_lidar_alpha_val.py", "scripts/inference/run_lidar_alpha_val.py"),
    ("build_test_outputs_gallery.py", "scripts/inference/build_test_outputs_gallery.py"),
    ("build_blend_preview_gallery.py", "scripts/inference/build_blend_preview_gallery.py"),
    # baselines
    ("compare_baselines.py", "scripts/baselines/compare_baselines.py"),
    ("visualize_baselines.py", "scripts/baselines/visualize_baselines.py"),
    ("export_all_methods.py", "scripts/baselines/export_all_methods.py"),
    ("export_rife_batch.py", "scripts/baselines/export_rife_batch.py"),
    ("mega_parallax_debug.py", "scripts/baselines/mega_parallax_debug.py"),
    ("mega_parallax_topology_compare.py", "scripts/baselines/mega_parallax_topology_compare.py"),
    # tuning
    ("tune_static_mask.py", "scripts/tuning/tune_static_mask.py"),
    ("tune_layered_alpha.py", "scripts/tuning/tune_layered_alpha.py"),
    ("tune_layered_blur.py", "scripts/tuning/tune_layered_blur.py"),
    ("precompute_static_masks.py", "scripts/tuning/precompute_static_masks.py"),
    ("sweep_far_boundary.py", "scripts/tuning/sweep_far_boundary.py"),
    ("benchmark_far_static.py", "scripts/tuning/benchmark_far_static.py"),
    ("verify_far_gate_fix.py", "scripts/tuning/verify_far_gate_fix.py"),
    ("export_static_mask.py", "scripts/tuning/export_static_mask.py"),
    ("export_far_static_examples.py", "scripts/tuning/export_far_static_examples.py"),
    # ego
    ("import_manual_ego_masks.py", "scripts/ego/import_manual_ego_masks.py"),
    ("process_manual_mask_variants.py", "scripts/ego/process_manual_mask_variants.py"),
    ("export_mask_browser_gallery.py", "scripts/ego/export_mask_browser_gallery.py"),
    ("export_mask_composer.py", "scripts/ego/export_mask_composer.py"),
    ("import_composed_masks.py", "scripts/ego/import_composed_masks.py"),
    ("import_mask_picker_selections.py", "scripts/ego/import_mask_picker_selections.py"),
    ("serve_mask_picker.py", "scripts/ego/serve_mask_picker.py"),
    ("rebuild_composer_html.py", "scripts/ego/rebuild_composer_html.py"),
    ("check_mask_picker_selections.py", "scripts/ego/check_mask_picker_selections.py"),
    ("audit_pipeline_ego_masks.py", "scripts/ego/audit_pipeline_ego_masks.py"),
    ("auto_select_ego_masks.py", "scripts/ego/auto_select_ego_masks.py"),
    ("select_masks_t0t1_stability.py", "scripts/ego/select_masks_t0t1_stability.py"),
    ("validate_ego_masks.py", "scripts/ego/validate_ego_masks.py"),
    ("validate_mask_boundary_fit.py", "scripts/ego/validate_mask_boundary_fit.py"),
    ("try_donor_masks.py", "scripts/ego/try_donor_masks.py"),
    ("export_vehicle_camera_means.py", "scripts/ego/export_vehicle_camera_means.py"),
    ("export_mean_source_samples.py", "scripts/ego/export_mean_source_samples.py"),
    ("build_ego_artifact_masks.py", "scripts/ego/build_ego_artifact_masks.py"),
    ("test_ego_consistent_edges.py", "scripts/ego/test_ego_consistent_edges.py"),
    ("test_ego_mean_struct_edges.py", "scripts/ego/test_ego_mean_struct_edges.py"),
    ("test_ego_multi_signals.py", "scripts/ego/test_ego_multi_signals.py"),
    ("test_ego_seg_v2.py", "scripts/ego/test_ego_seg_v2.py"),
    ("test_ego_seg_v3.py", "scripts/ego/test_ego_seg_v3.py"),
    ("run_ego_v3_examples.py", "scripts/ego/run_ego_v3_examples.py"),
    # tools
    ("build_ensemble_notebook.py", "tools/build_ensemble_notebook.py"),
    ("build_lidar_density_notebook.py", "tools/build_lidar_density_notebook.py"),
    ("build_lidar_rife_blend_notebook.py", "tools/build_lidar_rife_blend_notebook.py"),
    ("build_consensus_training_side.py", "tools/build_consensus_training_side.py"),
    ("patch_training_lidar_cells.py", "tools/patch_training_lidar_cells.py"),
    ("patch_training_user_config.py", "tools/patch_training_user_config.py"),
    ("fix_training_lidar_final.py", "tools/fix_training_lidar_final.py"),
    ("restore_dataset_cell.py", "tools/restore_dataset_cell.py"),
    ("repair_notebook_clear_outputs.py", "tools/repair_notebook_clear_outputs.py"),
    # experiments
    ("testblur.py", "experiments/testblur.py"),
    ("testdelta.py", "experiments/testdelta.py"),
    ("testdeltaall.py", "experiments/testdeltaall.py"),
    ("testdeltadif.py", "experiments/testdeltadif.py"),
    ("testdeltadifsoft.py", "experiments/testdeltadifsoft.py"),
    ("testconv.py", "experiments/testconv.py"),
    ("test_images.py", "experiments/test_images.py"),
    ("test_t1-t0.py", "experiments/test_t1-t0.py"),
]

LIB_SHIMS = [
    "consensus_kit",
    "lidar_depth_map",
    "lidar_density_mask",
    "layered_parallax",
    "mega_parallax",
    "static_mask",
    "ego_mask_extract",
    "ego_mask_policy",
]

SCRIPT_SHIMS = [
    "precompute_cv_split",
    "run_test_consensus_inference",
    "run_test_blend_blur",
    "export_submission_blend50",
    "flip_mirror_lidar_masks",
    "run_lidar_alpha_val",
    "build_test_outputs_gallery",
    "build_blend_preview_gallery",
    "compare_baselines",
    "visualize_baselines",
    "export_all_methods",
    "export_rife_batch",
    "serve_mask_picker",
    "export_mask_composer",
    "import_mask_picker_selections",
    "import_manual_ego_masks",
    "import_composed_masks",
    "export_mask_browser_gallery",
    "rebuild_composer_html",
    "audit_pipeline_ego_masks",
    "tune_static_mask",
    "tune_layered_alpha",
    "tune_layered_blur",
]


def move_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file():
        return
    shutil.move(str(src), str(dst))
    print(f"  move {src.name} -> {dst.relative_to(REPO)}")


def write_lib_shim(name: str) -> None:
    p = REPO / f"{name}.py"
    p.write_text(
        f'"""Shim: use `from lib.{name}` or keep `import {name}` from repo root."""\n'
        f"from lib.{name} import *  # noqa: F401,F403\n",
        encoding="utf-8",
    )


def write_script_shim(name: str, subpath: str) -> None:
    p = REPO / f"{name}.py"
    p.write_text(
        f'"""Launcher -> scripts/{subpath}/{name}.py"""\n'
        "import runpy\n"
        "from pathlib import Path\n"
        f'runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "{subpath}" / "{name}.py"), run_name="__main__")\n',
        encoding="utf-8",
    )


def move_checkpoints() -> None:
    for src_name, dst_rel in (
        ("checkpoints_consensus", "artifacts/checkpoints/consensus"),
        ("checkpoints_consensus_side", "artifacts/checkpoints/consensus_side"),
    ):
        src = REPO / src_name
        dst = REPO / dst_rel
        if not src.is_dir():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file():
                t = dst / f.name
                if not t.is_file():
                    shutil.move(str(f), str(t))
                    print(f"  ckpt {f.name} -> {dst_rel}/")
        if src.is_dir() and not any(src.iterdir()):
            src.rmdir()


def main() -> None:
    print("=== apply_repo_layout ===\n")
    for s, d in MOVES:
        move_file(REPO / s, REPO / d)

    move_checkpoints()

    (REPO / "lib" / "__init__.py").write_text('"""Core libraries (consensus, lidar, parallax, ego)."""\n', encoding="utf-8")
    (REPO / "scripts" / "__init__.py").write_text("", encoding="utf-8")

    print("\n=== shims ===")
    for name in LIB_SHIMS:
        write_lib_shim(name)
        print(f"  shim lib.{name}")

    submap = {
        "precompute_cv_split": "inference",
        "run_test_consensus_inference": "inference",
        "run_test_blend_blur": "inference",
        "export_submission_blend50": "inference",
        "flip_mirror_lidar_masks": "inference",
        "run_lidar_alpha_val": "inference",
        "build_test_outputs_gallery": "inference",
        "build_blend_preview_gallery": "inference",
        "compare_baselines": "baselines",
        "visualize_baselines": "baselines",
        "export_all_methods": "baselines",
        "export_rife_batch": "baselines",
        "serve_mask_picker": "ego",
        "export_mask_composer": "ego",
        "import_mask_picker_selections": "ego",
        "import_manual_ego_masks": "ego",
        "import_composed_masks": "ego",
        "export_mask_browser_gallery": "ego",
        "rebuild_composer_html": "ego",
        "audit_pipeline_ego_masks": "ego",
        "tune_static_mask": "tuning",
        "tune_layered_alpha": "tuning",
        "tune_layered_blur": "tuning",
    }
    for name, sub in submap.items():
        if (REPO / "scripts" / sub / f"{name}.py").is_file():
            write_script_shim(name, sub)
            print(f"  shim scripts/{sub}/{name}.py")

    print("\nDone. See docs/REPO_LAYOUT.md")


if __name__ == "__main__":
    main()
