"""Единые пути репозитория и внешнего cv_dataset (переменная YA_CV_DATASET)."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent

CV_ROOT = Path(os.environ.get("YA_CV_DATASET", r"C:\Users\adel\Downloads\cv_dataset"))
DATASET_ROOT = CV_ROOT / "final_dataset_v5_participants"
TRAIN_SPLIT = DATASET_ROOT / "train"
TEST_SPLIT = DATASET_ROOT / "test"

BAKED_ROOT = CV_ROOT / "rife_refinement_baked"
RIFE_ROOT = CV_ROOT / "rife_predictions_v5"
WARPS_ROOT = CV_ROOT / "multiview_warps"
CONSENSUS_TEST_OUT = CV_ROOT / "consensus_test_outputs"
SUBMISSION_ROOT = CV_ROOT / "submission"
STATIC_FAR_ROOT = CV_ROOT / "precomputed_static_far"
EGO_ARTIFACT_MASKS = CV_ROOT / "ego_artifact_masks"

METHODS_GALLERY = REPO / "methods_gallery"
EGO_MANUAL_ROOT = METHODS_GALLERY / "_ego_manual_masks"
EGO_MASKS_APPROVED = EGO_MANUAL_ROOT / "masks_approved"
EGO_BROWSER_GALLERY = EGO_MANUAL_ROOT / "browser_gallery"
EGO_PICKER_SELECTIONS = EGO_MANUAL_ROOT / "mask_picker_selections.json"

CKPT_CONSENSUS = REPO / "artifacts" / "checkpoints" / "consensus" / "consensus_unet_best.pt"
CKPT_CONSENSUS_SIDE = REPO / "artifacts" / "checkpoints" / "consensus_side" / "consensus_unet_side_best.pt"
# fallback если ещё не перенесены
if not CKPT_CONSENSUS.is_file():
    _legacy = REPO / "checkpoints_consensus" / "consensus_unet_best.pt"
    if _legacy.is_file():
        CKPT_CONSENSUS = _legacy
if not CKPT_CONSENSUS_SIDE.is_file():
    _legacy = REPO / "checkpoints_consensus_side" / "consensus_unet_side_best.pt"
    if _legacy.is_file():
        CKPT_CONSENSUS_SIDE = _legacy

BASELINE_ENSEMBLE = REPO / "baseline_files" / "baseline_ensemble"
STAGE2_SCRIPTS = REPO / "scripts" / "stage2"

CONFIGS_DIR = REPO / "configs"
TUNING_DIR = CONFIGS_DIR / "tuning"
