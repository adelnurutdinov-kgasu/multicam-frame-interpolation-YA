"""Патч путей REPO / ya_paths / lib imports после apply_repo_layout."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# lib internal imports
LIB_FILES = list((REPO / "lib").glob("*.py"))
LIB_REPLACEMENTS = [
    (r"from lidar_depth_map import", "from lib.lidar_depth_map import"),
    (r"from lidar_density_mask import", "from lib.lidar_density_mask import"),
    (r"from layered_parallax import", "from lib.layered_parallax import"),
    (r"from static_mask import", "from lib.static_mask import"),
    (r"from mega_parallax import", "from lib.mega_parallax import"),
    (r"from ego_mask_extract import", "from lib.ego_mask_extract import"),
    (r"from ego_mask_policy import", "from lib.ego_mask_policy import"),
    (r"from consensus_kit import", "from lib.consensus_kit import"),
]

# scripts depth -> REPO
def repo_line_for(path: Path) -> str:
    rel = path.relative_to(REPO)
    depth = len(rel.parts) - 1  # scripts/inference/foo.py -> 2
    return f"REPO = Path(__file__).resolve().parents[{depth}]"


def patch_file(path: Path, replacements: list[tuple[str, str]]) -> bool:
    text = path.read_text(encoding="utf-8")
    orig = text
    for old, new in replacements:
        text = text.replace(old, new)
    if text != orig:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_scripts_dir(sub: str) -> int:
    n = 0
    d = REPO / "scripts" / sub
    for py in d.glob("*.py"):
        text = py.read_text(encoding="utf-8")
        orig = text
        # normalize REPO
        text = re.sub(
            r"REPO\s*=\s*Path\(__file__\)\.resolve\(\)\.parent(?:\s*\n\s*sys\.path\.insert\(0,\s*str\(REPO\)\))?",
            "",
            text,
            count=1,
        )
        text = re.sub(
            r"ROOT\s*=\s*Path\(__file__\)\.resolve\(\)\.parents\[1\]",
            "",
            text,
        )
        depth = len(py.relative_to(REPO).parts) - 1
        header = (
            f"REPO = Path(__file__).resolve().parents[{depth}]\n"
            "import sys\n"
            "if str(REPO) not in sys.path:\n"
            "    sys.path.insert(0, str(REPO))\n\n"
        )
        if "REPO = Path(__file__)" not in text:
            # after docstring
            m = re.match(r'(\s*"""[\s\S]*?"""\s*\n+)', text)
            if m:
                text = m.group(1) + header + text[m.end() :]
            else:
                text = header + text
        # ya_paths for cv_dataset
        text = text.replace(
            'Path(r"C:\\Users\\adel\\Downloads\\cv_dataset")',
            "CV_ROOT  # ya_paths",
        )
        text = text.replace(
            'Path(r"C:/Users/adel/Downloads/cv_dataset")',
            "CV_ROOT  # ya_paths",
        )
        if "from ya_paths import" not in text and "CV_ROOT  # ya_paths" in text:
            text = text.replace(
                "import sys\n",
                "import sys\nfrom ya_paths import CV_ROOT, REPO as _REPO\n",
                1,
            )
        if text != orig:
            py.write_text(text, encoding="utf-8")
            n += 1
            print(f"  patched {py.relative_to(REPO)}")
    return n


def patch_lib() -> None:
    for py in LIB_FILES:
        if py.name == "__init__.py":
            continue
        if patch_file(py, LIB_REPLACEMENTS):
            print(f"  lib {py.name}")


def patch_consensus_kit_defaults() -> None:
    p = REPO / "lib" / "consensus_kit.py"
    text = p.read_text(encoding="utf-8")
    old = '''def default_config() -> ConsensusConfig:
    root = Path(r"C:/Users/adel/Downloads/cv_dataset")
    repo = Path(r"C:/Users/adel/Documents/GitHub/YA-")
    return ConsensusConfig(
        dataset_root=str(root / "final_dataset_v5_participants/train"),
        consensus_root=str(root / "multiview_warps/train"),
        baked_root=str(root / "rife_refinement_baked/train"),
        rife_root=str(root / "rife_predictions_v5/train"),
        ego_masks_root=str(repo / "methods_gallery/_ego_manual_masks/masks_approved"),
        static_far_root=str(root / "precomputed_static_far"),
    )'''
    new = '''def default_config() -> ConsensusConfig:
    from ya_paths import (
        BAKED_ROOT,
        DATASET_ROOT,
        EGO_MASKS_APPROVED,
        RIFE_ROOT,
        STATIC_FAR_ROOT,
        TRAIN_SPLIT,
        WARPS_ROOT,
    )

    return ConsensusConfig(
        dataset_root=str(TRAIN_SPLIT),
        consensus_root=str(WARPS_ROOT / "train"),
        baked_root=str(BAKED_ROOT / "train"),
        rife_root=str(RIFE_ROOT / "train"),
        ego_masks_root=str(EGO_MASKS_APPROVED),
        static_far_root=str(STATIC_FAR_ROOT),
    )'''
    if old in text:
        p.write_text(text.replace(old, new), encoding="utf-8")
        print("  lib/consensus_kit default_config")


def patch_precompute() -> None:
    p = REPO / "scripts" / "inference" / "precompute_cv_split.py"
    text = p.read_text(encoding="utf-8")
    text = text.replace(
        'str(REPO / "Второй этап" / "bake_refinement_assets.py")',
        'str(REPO / "scripts" / "stage2" / "bake_refinement_assets.py")',
    )
    text = text.replace(
        'str(REPO / "Второй этап" / "multiview_warping.py")',
        'str(REPO / "scripts" / "stage2" / "multiview_warping.py")',
    )
    text = text.replace(
        'str(REPO / "export_rife_batch.py")',
        'str(REPO / "scripts" / "baselines" / "export_rife_batch.py")',
    )
    if "from ya_paths import" not in text:
        text = text.replace(
            "from pathlib import Path\n",
            "from pathlib import Path\n\nfrom ya_paths import CV_ROOT, DATASET_ROOT, REPO\n",
            1,
        )
    text = re.sub(
        r"DEFAULT_DATASET = Path\(r.*?\)",
        "DEFAULT_DATASET = DATASET_ROOT",
        text,
    )
    text = re.sub(
        r"DEFAULT_CV_ROOT = Path\(r.*?\)",
        "DEFAULT_CV_ROOT = CV_ROOT",
        text,
    )
    p.write_text(text, encoding="utf-8")
    print("  scripts/inference/precompute_cv_split.py")


def patch_static_mask_tune() -> None:
    p = REPO / "lib" / "static_mask.py"
    if not p.is_file():
        return
    text = p.read_text(encoding="utf-8")
    for name in ("static_mask_tune_best.json", "alpha_tune_best.json", "blur_tune_best.json"):
        text = text.replace(
            f'Path("{name}")',
            f'Path(__file__).resolve().parents[1] / "configs" / "tuning" / "{name}"',
        )
        text = text.replace(
            f"Path('{name}')",
            f'Path(__file__).resolve().parents[1] / "configs" / "tuning" / "{name}"',
        )
    p.write_text(text, encoding="utf-8")
    print("  lib/static_mask tune paths")


def patch_stage2() -> None:
    for py in (REPO / "scripts" / "stage2").glob("*.py"):
        text = py.read_text(encoding="utf-8")
        text = text.replace("parents[1]", "parents[2]")
        text = text.replace(
            'ROOT = Path(__file__).resolve().parents[2]',
            "REPO = Path(__file__).resolve().parents[2]",
        )
        if "sys.path.insert" in text and "ya_paths" not in text:
            text = text.replace(
                "import sys",
                "import sys\nfrom ya_paths import CV_ROOT, DATASET_ROOT, REPO",
                1,
            )
        py.write_text(text, encoding="utf-8")
        print(f"  {py.name}")


def patch_inference_key() -> None:
    """Явные правки ключевых inference-скриптов."""
    from pathlib import Path as P

    fixes = {
        "run_test_consensus_inference.py": [
            ('CKPT = REPO / "checkpoints_consensus"', "from ya_paths import CKPT_CONSENSUS as FRONT_REAR_CKPT\nSIDE_CKPT = __import__('ya_paths').CKPT_CONSENSUS_SIDE"),
            ("FRONT_REAR_CKPT = REPO / \"checkpoints_consensus\" / \"consensus_unet_best.pt\"", ""),
            ("SIDE_CKPT = REPO / \"checkpoints_consensus_side\" / \"consensus_unet_side_best.pt\"", ""),
        ],
    }
    for name, reps in fixes.items():
        p = REPO / "scripts" / "inference" / name
        if not p.is_file():
            continue
        t = p.read_text(encoding="utf-8")
        for a, b in reps:
            t = t.replace(a, b)
        p.write_text(t, encoding="utf-8")


def main() -> None:
    print("patch imports...")
    patch_lib()
    patch_consensus_kit_defaults()
    patch_static_mask_tune()
    patch_stage2()
    for sub in ("inference", "ego", "baselines", "tuning"):
        patch_scripts_dir(sub)
    patch_precompute()
    # run_test - manual fix
    rt = REPO / "scripts" / "inference" / "run_test_consensus_inference.py"
    if rt.is_file():
        t = rt.read_text(encoding="utf-8")
        t = t.replace(
            "CV_ROOT = Path(r\"C:\\Users\\adel\\Downloads\\cv_dataset\")",
            "from ya_paths import (\n    CV_ROOT,\n    CKPT_CONSENSUS,\n    CKPT_CONSENSUS_SIDE,\n    CONSENSUS_TEST_OUT,\n    EGO_MASKS_APPROVED,\n    REPO,\n)\n",
        )
        t = t.replace(
            "FRONT_REAR_CKPT = REPO / \"checkpoints_consensus\" / \"consensus_unet_best.pt\"",
            "FRONT_REAR_CKPT = CKPT_CONSENSUS",
        )
        t = t.replace(
            "SIDE_CKPT = REPO / \"checkpoints_consensus_side\" / \"consensus_unet_side_best.pt\"",
            "SIDE_CKPT = CKPT_CONSENSUS_SIDE",
        )
        t = t.replace("DEFAULT_OUT = CV_ROOT / \"consensus_test_outputs\"", "DEFAULT_OUT = CONSENSUS_TEST_OUT")
        t = t.replace(
            "ego_masks_root=str(REPO / \"methods_gallery/_ego_manual_masks/masks_approved\")",
            "ego_masks_root=str(EGO_MASKS_APPROVED)",
        )
        rt.write_text(t, encoding="utf-8")
        print("  run_test_consensus_inference")

    rb = REPO / "scripts" / "inference" / "run_test_blend_blur.py"
    if rb.is_file():
        t = rb.read_text(encoding="utf-8")
        if "from ya_paths import" not in t:
            t = t.replace(
                "from consensus_kit import",
                "from ya_paths import BAKED_ROOT, CONSENSUS_TEST_OUT, CV_ROOT, DATASET_ROOT, EGO_MASKS_APPROVED, REPO, TEST_SPLIT\nfrom consensus_kit import",
            )
        t = t.replace('OUT_ROOT = CV_ROOT / "consensus_test_outputs"', "OUT_ROOT = CONSENSUS_TEST_OUT")
        t = t.replace('BAKED_ROOT = CV_ROOT / "rife_refinement_baked" / "test"', "BAKED_TEST = BAKED_ROOT / \"test\"")
        t = t.replace("BAKED_ROOT / sid", "BAKED_TEST / sid")
        t = t.replace(
            "ego_masks_root=str(REPO / \"methods_gallery/_ego_manual_masks/masks_approved\")",
            "ego_masks_root=str(EGO_MASKS_APPROVED)",
        )
        rb.write_text(t, encoding="utf-8")
        print("  run_test_blend_blur")

    print("done")


if __name__ == "__main__":
    main()
