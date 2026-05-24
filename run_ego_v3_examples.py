"""Batch v3 ego mask examples for manual review."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from test_ego_seg_v2 import ego_is_bright
from test_ego_seg_v3 import (
    DEFAULT_DATASET,
    build_mask_v3,
    load_vehicle_stack,
    overlay_mask,
    v2_mask_for_compare,
)

OUT = REPO / "methods_gallery/_ego_artifacts_v2_test/examples"

EXAMPLES = [
    ("hilma", "right_fwd"),
    ("hilma", "front"),
    ("robb", "front"),
    ("robb", "right_fwd"),
    ("jurita", "front"),
    ("ravine", "left_fwd"),
    ("mika", "front"),
    ("shelly", "rear"),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []

    for vehicle, cam in EXAMPLES:
        try:
            stack = load_vehicle_stack(DEFAULT_DATASET, vehicle, cam, 12)
        except Exception as exc:
            print(f"SKIP {vehicle}/{cam}: {exc}")
            continue

        trim = 65.0 if ego_is_bright(stack, 0.55) else 35.0
        mask, edge_freq, _ = build_mask_v3(stack, trim_std_pct=trim)
        v2 = v2_mask_for_compare(stack, 0.55)
        ref = stack[0]

        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        ax[0].imshow(ref)
        ax[0].set_title(f"{vehicle}/{cam}")
        ax[0].axis("off")

        ax[1].imshow(edge_freq, cmap="magma", vmin=0, vmax=1)
        ax[1].set_title("edge vote")
        ax[1].axis("off")

        ax[2].imshow(overlay_mask(ref, mask))
        ax[2].set_title(f"v3 {100 * mask.mean():.1f}%")
        ax[2].axis("off")

        if v2 is not None:
            ax[3].imshow(overlay_mask(ref, v2))
            ax[3].set_title(f"v2 {100 * v2.mean():.1f}%")
        ax[3].axis("off")

        out_path = OUT / f"{vehicle}_{cam}.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        row = {
            "vehicle": vehicle,
            "camera": cam,
            "v3": float(mask.mean()),
            "v2": float(v2.mean()) if v2 is not None else None,
            "trim": trim,
            "viz": str(out_path),
        }
        results.append(row)
        v2_txt = f"  v2={100 * row['v2']:.1f}%" if row["v2"] is not None else ""
        trim_txt = "none" if trim is None else str(int(trim))
        print(f"{vehicle:8s}/{cam:10s}  v3={100 * row['v3']:.1f}%{v2_txt}  trim={trim_txt}")

    # index grid
    n = len(results)
    if n:
        cols = 2
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        axes = np.atleast_2d(axes)
        for i, row in enumerate(results):
            r, c = divmod(i, cols)
            img = plt.imread(row["viz"])
            axes[r, c].imshow(img)
            axes[r, c].set_title(
                f"{row['vehicle']}/{row['camera']}  v3={100 * row['v3']:.1f}%",
                fontsize=10,
            )
            axes[r, c].axis("off")
        for j in range(n, rows * cols):
            r, c = divmod(j, cols)
            axes[r, c].axis("off")
        fig.suptitle("Ego mask v3 — examples batch", fontsize=13)
        index_path = OUT / "index_grid.png"
        fig.savefig(index_path, dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"\nIndex: {index_path}")

    (OUT / "batch_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(results)} examples -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
