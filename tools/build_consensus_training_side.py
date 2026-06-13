"""Build consensus_training_side.ipynb from consensus_training.ipynb."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "consensus_training.ipynb"
DST = ROOT / "consensus_training_side.ipynb"

REPLACEMENTS: list[tuple[str, str]] = [
    (
        '# Consensus Refinement — обучение модели для свёртки геометрического консенсуса',
        '# Consensus Refinement — **боковые камеры** (side canonical)\n\n'
        'Копия `consensus_training.ipynb`. `left_fwd` и `right_bwd` отражаются по горизонтали.\n'
        'Веса: `checkpoints_consensus_side/consensus_unet_side_best.pt` — не трогают front/rear.\n\n'
        '---\n\n'
        '# Consensus Refinement — обучение модели для свёртки геометрического консенсуса',
    ),
    (
        '    allowed_cameras: tuple = ("front", "rear")  # пока только перед/зад\n'
        '    max_samples: int = 0                        # 0 = все подходящие front/rear',
        '    allowed_cameras: tuple = ("left_fwd", "right_fwd", "left_bwd", "right_bwd")\n'
        '    mirror_cameras: tuple = ("left_fwd", "right_bwd")  # flip H → канон. перспектива\n'
        '    augment_random_hflip: bool = False          # не ломать канонизацию side\n'
        '    max_samples: int = 0',
    ),
    (
        '    save_dir: str = "./checkpoints_consensus"',
        '    save_dir: str = "./checkpoints_consensus_side"\n'
        '    checkpoint_tag: str = "consensus_unet_side"  # имя *_best.pt',
    ),
    (
        'print(f"  cameras:   {CFG.allowed_cameras}")\n'
        'print(f"  max_samples: {CFG.max_samples or \'all\'}")',
        'print(f"  cameras:   {CFG.allowed_cameras}")\n'
        'print(f"  mirror:    {CFG.mirror_cameras}")\n'
        'print(f"  ckpt tag:  {CFG.checkpoint_tag}  →  {CFG.save_dir}/{CFG.checkpoint_tag}_best.pt")\n'
        'print(f"  max_samples: {CFG.max_samples or \'all\'}")',
    ),
    (
        'Depth нормализуется per-sample в `[0..1]` по `depth / (depth.max() + eps)`.',
        'Depth нормализуется per-sample в `[0..1]` по `depth / (depth.max() + eps)`.\n\n'
        '**Side canonical:** `left_fwd`, `right_bwd` → horizontal flip всех карт.',
    ),
    (
        '    return warp_rgb, coverage, depth\n\n\nclass ConsensusDataset(Dataset):',
        '    return warp_rgb, coverage, depth\n\n\n'
        'def _flip_maps_h(warp, cov, depth_n, sf_rgb, sf_mask, art_mask, mean_t0t1, rife, target):\n'
        '    flip = lambda a: a[:, ::-1].copy()\n'
        '    return (\n'
        '        flip(warp), flip(cov), flip(depth_n), flip(sf_rgb), flip(sf_mask),\n'
        '        flip(art_mask), flip(mean_t0t1), flip(rife), flip(target),\n'
        '    )\n\n\nclass ConsensusDataset(Dataset):',
    ),
    (
        '        return {\n            "warp_rgb": warp_rgb,',
        '        return {\n            "camera": cam,\n            "warp_rgb": warp_rgb,',
    ),
    (
        '        target = s["target_rgb"].astype(np.float32) / 255.0\n\n        H, W = warp.shape[:2]',
        '        target = s["target_rgb"].astype(np.float32) / 255.0\n'
        '        cam = s["camera"]\n'
        '        mirrored = cam in getattr(self.cfg, "mirror_cameras", ())\n'
        '        if mirrored:\n'
        '            warp, cov, depth_n, sf_rgb, sf_mask, art_mask, mean_t0t1, rife, target = _flip_maps_h(\n'
        '                warp, cov, depth_n, sf_rgb, sf_mask, art_mask, mean_t0t1, rife, target\n'
        '            )\n\n        H, W = warp.shape[:2]',
    ),
    (
        '        if self.augment:\n            if random.random() < 0.5:',
        '        if self.augment:\n            if getattr(self.cfg, "augment_random_hflip", True) and random.random() < 0.5:',
    ),
    (
        '            "rife_rgb": to_chw_3(rife),\n        }',
        '            "rife_rgb": to_chw_3(rife),\n'
        '            "camera": cam,\n'
        '            "mirrored": mirrored,\n'
        '        }',
    ),
    (
        '    raw = ConsensusDataset([baked_dir], cfg, patch_size=cfg.patch_size, augment=False)._load(baked_dir)\n\n'
        '    warp = raw["warp_rgb"].astype(np.float32) / 255.0',
        '    raw = ConsensusDataset([baked_dir], cfg, patch_size=cfg.patch_size, augment=False)._load(baked_dir)\n'
        '    cam = raw["camera"]\n'
        '    mirrored = cam in getattr(cfg, "mirror_cameras", ())\n\n'
        '    warp = raw["warp_rgb"].astype(np.float32) / 255.0',
    ),
    (
        '    target = raw["target_rgb"].astype(np.float32) / 255.0\n\n    mean_in = mean',
        '    target = raw["target_rgb"].astype(np.float32) / 255.0\n'
        '    if mirrored:\n'
        '        flip = lambda a: a[:, ::-1].copy()\n'
        '        warp, cov, sf_rgb, sf_mask, art, mean, target = (\n'
        '            flip(warp), flip(cov), flip(sf_rgb), flip(sf_mask), flip(art), flip(mean), flip(target)\n'
        '        )\n\n    mean_in = mean',
    ),
    (
        '        f"{meta[\'sample_id\']} | cam={meta[\'camera\']} | vehicle={vehicle} | art={int(art.sum())} px",',
        '        f"{meta[\'sample_id\']} | cam={meta[\'camera\']}{\' (mirrored)\' if mirrored else \'\'} | "\n'
        '        f"vehicle={vehicle} | art={int(art.sum())} px",',
    ),
    (
        '        tag="consensus_unet",\n    )\n\n\nbaseline_result = run_baseline_unet(num_epochs=200)',
        '        tag=CFG.checkpoint_tag,\n    )\n\n\n# baseline_result = run_baseline_unet(num_epochs=200)',
    ),
    (
        '        tag="consensus_marnet",',
        '        tag="consensus_marnet_side",',
    ),
    (
        'marnet_result = run_baseline_marnet(num_epochs=200)',
        '# marnet_result = run_baseline_marnet(num_epochs=200)',
    ),
    (
        'def demo_compare_on_val(ckpt_path=None, n_samples=3, tag="consensus_unet"):',
        'def demo_compare_on_val(ckpt_path=None, n_samples=3, tag=None):\n'
        '    tag = tag or CFG.checkpoint_tag',
    ),
]


def patch_source(src: str) -> str:
    out = src
    for old, new in REPLACEMENTS:
        if old in out:
            out = out.replace(old, new, 1)
    return out


def main() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    out = deepcopy(nb)
    for cell in out["cells"]:
        cell["outputs"] = []
        cell["execution_count"] = None
        if isinstance(cell.get("source"), list):
            src = "".join(cell["source"])
            cell["source"] = patch_source(src).splitlines(keepends=True)

    DST.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {DST} ({len(out['cells'])} cells, outputs cleared)")


if __name__ == "__main__":
    main()
