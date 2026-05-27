# Структура репозитория YA-

Документ описывает реорганизацию корня проекта: **как было**, **куда перенесено**, **где что искать сейчас**.

Дата реорганизации: май 2026.

---

## Как было (до уборки)

В корне лежало **~73 Python-скрипта**, 6 ноутбуков, JSON-тюнинг, чекпоинты и документация в одной плоскости:

```
YA-/
  consensus_kit.py, lidar_*.py, run_test_*.py, export_*.py, test_ego_*.py, …
  consensus_training.ipynb
  Второй этап/*.py          # кириллица в путях subprocess
  checkpoints_consensus/
  methods_gallery/          # уже была отдельной папкой
  README_*.md, TEST_INFERENCE_PIPELINE.md
```

Проблемы: сложно найти entry point, дублирование путей `C:\Users\adel\Downloads\cv_dataset`, импорты `sys.path.insert(0, REPO)` из каждого файла.

---

## Как стало (сейчас)

```
YA-/
├── README.md                 # индекс + быстрый старт
├── ya_paths.py               # единые пути REPO + cv_dataset
│
├── docs/                     # вся документация
│   ├── REPO_LAYOUT.md        # этот файл
│   ├── TEST_INFERENCE_PIPELINE.md
│   ├── README_CV_DATASET.md
│   ├── README_YOLO.md
│   └── archive/              # _dump_*.txt
│
├── configs/
│   ├── dataset.yaml          # схема путей к внешнему датасету
│   └── tuning/               # alpha/blur/static_mask/far_boundary JSON
│
├── lib/                      # переиспользуемые модули
│   ├── consensus_kit.py
│   ├── lidar_depth_map.py
│   ├── lidar_density_mask.py
│   ├── layered_parallax.py
│   ├── mega_parallax.py
│   ├── static_mask.py
│   ├── ego_mask_extract.py
│   └── ego_mask_policy.py
│
├── scripts/
│   ├── inference/            # test pipeline, submission, галереи
│   ├── stage2/               # bake, multiview warps, refine (бывший «Второй этап»)
│   ├── ego/                  # маски, picker, import
│   ├── baselines/            # compare, RIFE batch, export methods
│   └── tuning/               # tune_*, static far, sweep
│
├── tools/                    # генераторы ноутбуков, patch ноутбуков
├── experiments/              # testblur, testdelta*, прототипы
├── notebooks/
│   ├── training/             # consensus_*, lidar_*
│   └── stage2/               # rife_depth_refinement_starter.ipynb
│
├── artifacts/
│   └── checkpoints/
│       ├── consensus/          # consensus_unet_best.pt, val_cache, …
│       └── consensus_side/     # side cameras
│
├── methods_gallery/            # БЕЗ ПЕРЕНОСА — экспорт методов, ego, preview
├── baseline_files/
├── baseline_viz/
├── yolo/
├── viewer/
├── analytics/
├── tests/
└── data/
```

### Обратная совместимость (корень)

В корне остались **тонкие launcher/shim** — старые команды работают:

| Команда в корне | Реальный файл |
|-----------------|---------------|
| `python run_test_consensus_inference.py` | `scripts/inference/run_test_consensus_inference.py` |
| `python precompute_cv_split.py` | `scripts/inference/precompute_cv_split.py` |
| `import consensus_kit` | `lib/consensus_kit.py` (через shim `consensus_kit.py`) |
| `python serve_mask_picker.py` | `scripts/ego/serve_mask_picker.py` |

---

## Внешние данные (не в git)

Задаются в `ya_paths.py` или переменной **`YA_CV_DATASET`**:

| Переменная в коде | Путь по умолчанию |
|-------------------|-------------------|
| `CV_ROOT` | `C:/Users/adel/Downloads/cv_dataset` |
| `DATASET_ROOT` | `{CV_ROOT}/final_dataset_v5_participants` |
| `CONSENSUS_TEST_OUT` | `{CV_ROOT}/consensus_test_outputs` |
| `SUBMISSION_ROOT` | `{CV_ROOT}/submission` |
| `BAKED_ROOT` | `{CV_ROOT}/rife_refinement_baked` |
| `RIFE_ROOT` | `{CV_ROOT}/rife_predictions_v5` |
| `WARPS_ROOT` | `{CV_ROOT}/multiview_warps` |

---

## Где что внутри репозитория

| Что | Путь |
|-----|------|
| Approved ego-маски | `methods_gallery/_ego_manual_masks/masks_approved/` |
| Mask picker (test) | `http://127.0.0.1:8765/composer_test.html` → `methods_gallery/_ego_manual_masks/browser_gallery/` |
| Picker selections | `methods_gallery/_ego_manual_masks/mask_picker_selections.json` |
| Чекпоинт front/rear | `artifacts/checkpoints/consensus/consensus_unet_best.pt` |
| Чекпоинт side | `artifacts/checkpoints/consensus_side/consensus_unet_side_best.pt` |
| Галерея test outputs | `{CV_ROOT}/consensus_test_outputs/index.html` |
| Тюнинг static mask | `configs/tuning/static_mask_tune_best.json` |
| RIFE baseline code | `baseline_files/baseline_ensemble/` |

---

## Карта переноса файлов

### Документация → `docs/`
- `README_CV_DATASET.md`, `README_YOLO.md`, `TEST_INFERENCE_PIPELINE.md`
- `_dump_*.txt` → `docs/archive/`

### Тюнинг JSON → `configs/tuning/`
- `alpha_tune_best.json`, `blur_tune_best.json`, `static_mask_tune_best.json`
- `far_boundary_sweep.*`, `compare_baselines_results.json`

### Библиотеки → `lib/` (+ shim в корне)
- `consensus_kit.py`, `lidar_*.py`, `layered_parallax.py`, `mega_parallax.py`, `static_mask.py`, `ego_mask_*.py`

### Ноутбуки → `notebooks/`
- `consensus_training*.ipynb`, `lidar_*.ipynb`, `consensus_ensemble_front_rear.ipynb` → `notebooks/training/`
- `Второй этап/rife_depth_refinement_starter.ipynb` → `notebooks/stage2/`

### Stage 2 → `scripts/stage2/`
- `bake_refinement_assets.py`, `multiview_warping.py`, `refine_v2.py`, `render_baked_depth.py`, `eval_consensus_tuned.py`

### Inference / submission → `scripts/inference/`
- `precompute_cv_split.py`, `run_test_consensus_inference.py`, `run_test_blend_blur.py`
- `export_submission_blend50.py`, `flip_mirror_lidar_masks.py`, `run_lidar_alpha_val.py`
- `build_test_outputs_gallery.py`, `build_blend_preview_gallery.py`

### Ego → `scripts/ego/`
- `import_manual_ego_masks.py`, `export_mask_composer.py`, `serve_mask_picker.py`, `import_mask_picker_selections.py`, …

### Baselines → `scripts/baselines/`
- `compare_baselines.py`, `export_rife_batch.py`, `export_all_methods.py`, `visualize_baselines.py`, …

### Tuning → `scripts/tuning/`
- `tune_static_mask.py`, `tune_layered_*.py`, `precompute_static_masks.py`, `sweep_far_boundary.py`, …

### Tools → `tools/`
- `build_*_notebook.py`, `patch_training_*.py`, `restore_dataset_cell.py`, `apply_repo_layout.py`

### Experiments → `experiments/`
- `testblur.py`, `testdelta*.py`, `testdeltadif*.py`, `testconv.py`, `test_images.py`

### Checkpoints → `artifacts/checkpoints/`
- `checkpoints_consensus/*` → `artifacts/checkpoints/consensus/`
- `checkpoints_consensus_side/*` → `artifacts/checkpoints/consensus_side/`

---

## Что не переносилось (намеренно)

| Папка | Причина |
|-------|---------|
| `methods_gallery/` | Тысячи HTML/JSON/кадров со старыми относительными путями |
| `yolo/`, `viewer/`, `analytics/` | Уже изолированы |
| `baseline_files/` | RIFE submodule + train_log |
| `depth_out/`, `layered_parallax_out/` | Локальный scratch |
| Папка `Второй этап/` | Может остаться с кэшем/preview; **канонические скрипты** — в `scripts/stage2/` |

---

## Типовые команды (после уборки)

```bash
# Test E2E (из корня — shims)
python precompute_cv_split.py --split test
python run_test_consensus_inference.py
python run_test_blend_blur.py --force
python export_submission_blend50.py

# Или явные пути
python scripts/inference/precompute_cv_split.py --split test

# Ego picker
python serve_mask_picker.py
python import_mask_picker_selections.py

# Пересборка composer test
python scripts/ego/export_mask_composer.py --test-only
```

---

## Повторная уборка / миграция

Скрипты (идемпотентны, если файлы уже перенесены):

```bash
python tools/apply_repo_layout.py
python tools/patch_imports_after_layout.py
python tools/fix_script_headers.py
python tools/make_lib_shim.py
```

---

## Связанные документы

- [TEST_INFERENCE_PIPELINE.md](TEST_INFERENCE_PIPELINE.md) — полный test inference
- [README_CV_DATASET.md](README_CV_DATASET.md) — формат сэмплов
- [methods/README.md](../methods/README.md) — нумерация методов 01–17 в `methods_gallery/`
