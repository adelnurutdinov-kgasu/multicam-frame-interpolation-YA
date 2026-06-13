"""
Прекомпьют для split без GT (test): depth bake → RIFE → multiview warps.

Пример:
  python precompute_cv_split.py --split test
  python precompute_cv_split.py --split test --steps bake,rife
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ya_paths import CV_ROOT, DATASET_ROOT  # noqa: E402

DEFAULT_DATASET = DATASET_ROOT
DEFAULT_CV_ROOT = CV_ROOT


def _run(cmd: list[str], label: str) -> int:
    print(f"\n=== {label} ===", flush=True)
    print(" ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(REPO))
    if r.returncode != 0:
        print(f"FAIL {label} exit={r.returncode}", flush=True)
    return r.returncode


def main() -> int:
    p = argparse.ArgumentParser(description="Precompute bake + RIFE + warps for a dataset split")
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--cv-root", type=Path, default=DEFAULT_CV_ROOT, help="Корень артефактов (baked, rife, warps)")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument(
        "--steps",
        default="bake,rife,warps",
        help="Через запятую: bake, rife, warps",
    )
    p.add_argument("--workers", type=int, default=8, help="Параллельность depth bake")
    p.add_argument("--limit", type=int, default=0, help="0 = все сэмплы split")
    p.add_argument("--device", default="cuda", help="cuda/cpu для multiview_warping")
    p.add_argument("--no-skip-existing", action="store_true")
    args = p.parse_args()

    split = args.split
    dataset_split = args.dataset_dir / split
    if not dataset_split.is_dir():
        print(f"Нет split: {dataset_split}")
        return 1

    baked = args.cv_root / "rife_refinement_baked" / split
    rife = args.cv_root / "rife_predictions_v5" / split
    warps = args.cv_root / "multiview_warps" / split

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    py = sys.executable
    limit_args = ["--limit", str(args.limit)] if args.limit > 0 else []
    rc = 0

    if "bake" in steps:
        cmd = [
            py,
            str(REPO / "scripts" / "stage2" / "bake_refinement_assets.py"),
            "--dataset-dir",
            str(args.dataset_dir),
            "--out-root",
            str(args.cv_root / "rife_refinement_baked"),
            "--split",
            split,
            "--workers",
            str(args.workers),
        ]
        if args.no_skip_existing:
            cmd.append("--no-skip-existing")
        cmd.extend(limit_args)
        rc = _run(cmd, f"bake depth -> {baked}") or rc

    if "rife" in steps:
        cmd = [
            py,
            str(REPO / "scripts" / "baselines" / "export_rife_batch.py"),
            "--dataset-dir",
            str(args.dataset_dir),
            "--split",
            split,
            "--out-dir",
            str(args.cv_root / "rife_predictions_v5"),
            "--allow-no-target",
        ]
        if args.no_skip_existing:
            cmd.append("--no-skip-existing")
        cmd.extend(limit_args)
        rc = _run(cmd, f"RIFE -> {rife}") or rc

    if "warps" in steps:
        cmd = [
            py,
            str(REPO / "scripts" / "stage2" / "multiview_warping.py"),
            "--dataset-dir",
            str(dataset_split),
            "--baked-dir",
            str(baked),
            "--out-root",
            str(warps),
            "--rife-root",
            str(args.cv_root / "rife_predictions_v5"),
            "--all",
            "--device",
            args.device,
            "--no-preview",
        ]
        cmd.extend(limit_args)
        rc = _run(cmd, f"multiview warps -> {warps}") or rc

    print(f"\nГотово steps={steps} split={split} rc={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
