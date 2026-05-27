"""Re-apply user CONFIG/dataset tweaks on top of git-restored notebook."""
import json
from pathlib import Path

NB = Path("consensus_training.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

for c in nb["cells"]:
    if c.get("id") != "04_code":
        continue
    src = "".join(c["source"])
    repls = [
        ('    use_artifact_mask: bool = False', '    use_artifact_mask: bool = True'),
        ('    max_samples: int = 100', '    max_samples: int = 0'),
    ]
    for a, b in repls:
        src = src.replace(a, b)
    if "rife_root" not in src:
        src = src.replace(
            '    baked_root: str = str(CV_ROOT / "rife_refinement_baked/train")\n',
            '    baked_root: str = str(CV_ROOT / "rife_refinement_baked/train")\n'
            '    rife_root: str = str(CV_ROOT / "rife_predictions_v5/train")\n',
        )
    if "ego_masks_root" not in src:
        src = src.replace(
            '    static_far_root: str = str(CV_ROOT / "precomputed_static_far")',
            '    static_far_root: str = str(CV_ROOT / "precomputed_static_far")\n'
            '    ego_masks_root: str = str(\n'
            '        Path(r"C:/Users/adel/Documents/GitHub/YA-/methods_gallery/_ego_manual_masks/masks_approved")\n'
            '    )',
        )
    if "allowed_cameras" not in src:
        src = src.replace(
            '    use_anchor_frames: bool = True',
            '    use_anchor_frames: bool = True\n'
            '    allowed_cameras: tuple = ("front", "rear")',
        )
    if "import re" not in src and c.get("id") == "02_code":
        pass
    c["source"] = [ln + "\n" for ln in src.splitlines()]
    break

# imports cell
for c in nb["cells"]:
    if c.get("id") == "02_code":
        src = "".join(c["source"])
        if "import re" not in src:
            src = src.replace("import random\n", "import random\nimport re\n")
        c["source"] = [ln + "\n" for ln in src.splitlines()]

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("user config patched")
