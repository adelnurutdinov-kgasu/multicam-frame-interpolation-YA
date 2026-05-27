# Устаревшая папка stage 2

Скрипты и ноутбук перенесены:

| Было здесь | Стало |
|------------|--------|
| `bake_refinement_assets.py` | `scripts/stage2/bake_refinement_assets.py` |
| `multiview_warping.py` | `scripts/stage2/multiview_warping.py` |
| `refine_v2.py` | `scripts/stage2/refine_v2.py` |
| `render_baked_depth.py` | `scripts/stage2/render_baked_depth.py` |
| `eval_consensus_tuned.py` | `scripts/stage2/eval_consensus_tuned.py` |
| `rife_depth_refinement_starter.ipynb` | `notebooks/stage2/rife_depth_refinement_starter.ipynb` |

`precompute_cv_split.py` вызывает `scripts/stage2/`, не эту папку.

Локальные `methods_gallery/_preview/warps/` и `__pycache__` можно не трогать.
