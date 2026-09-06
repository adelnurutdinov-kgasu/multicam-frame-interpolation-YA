# Test inference pipeline — полное описание

Документ описывает end-to-end пайплайн прогона **test split (199 сэмплов, без GT)**:
прекомпьюты → ego-маски → consensus U-Net → LiDAR-blend с RIFE → HTML-галерея.

Корень репозитория — папка с этим README.  
Корень датасета и артефактов: `YA_CV_DATASET` (по умолчанию — см. `ya_paths.py`)

---

## 1. Общая схема

```
final_dataset_v5_participants/test/<sample_id>/
  input/lidar.npz, input/t0/*.jpg, input/t1/*.jpg, meta.json
        │
        ├─► [1] depth bake      → rife_refinement_baked/test/<id>/
        ├─► [2] RIFE batch      → rife_predictions_v5/test/<id>.jpg
        ├─► [3] multiview warps → multiview_warps/test/<id>/
        │
        ├─► ego-маски (approved) → methods_gallery/_ego_manual_masks/masks_approved/
        │
        └─► [4] U-Net inference → consensus_test_outputs/<id>/
                  ├─ consensus_model.jpg
                  ├─ blend_lidar_rife.jpg
                  ├─ blend_blur50.jpg
                  └─ lidar_trust.png, …
```

---

## 2. Пути и каталоги

| Назначение | Путь |
|------------|------|
| Test split (исходники) | `cv_dataset/final_dataset_v5_participants/test/` |
| Depth bake | `cv_dataset/rife_refinement_baked/test/<sample_id>/` |
| RIFE predictions | `cv_dataset/rife_predictions_v5/test/<sample_id>.jpg` |
| Multiview warps | `cv_dataset/multiview_warps/test/<sample_id>/` |
| **Финальные выходы** | `cv_dataset/consensus_test_outputs/<sample_id>/` |
| HTML-галерея | `cv_dataset/consensus_test_outputs/index.html` |
| Ego-маски (approved) | `methods_gallery/_ego_manual_masks/masks_approved/{vehicle}_{camera}.png` |
| Mask picker | `methods_gallery/_ego_manual_masks/browser_gallery/composer_test.html` |
| Checkpoint front/rear | `artifacts/checkpoints/consensus/consensus_unet_best.pt` |
| Checkpoint side | `artifacts/checkpoints/consensus_side/consensus_unet_side_best.pt` |

Размер кадра для сети: **544×1024** (`ConsensusConfig.image_h/w` в `consensus_kit.py`).

---

## 3. Прекомпьюты (шаги 1–3)

Оркестратор: **`scripts/inference/precompute_cv_split.py`**

```bash
python scripts/inference/precompute_cv_split.py --split test
python scripts/inference/precompute_cv_split.py --split test --steps bake,rife,warps
```

| Шаг | Скрипт | Выход |
|-----|--------|-------|
| **bake** | `scripts/stage2/bake_refinement_assets.py` | `d1.npy`, `meta.json` в baked-папке |
| **rife** | `scripts/baselines/export_rife_batch.py --allow-no-target` | `rife_predictions_v5/test/<id>.jpg` |
| **warps** | `scripts/stage2/multiview_warping.py` | `consensus_raw.npy`, `coverage.npy` |

**Важно для warps:** `--rife-root` должен указывать на `rife_predictions_v5` (не на подпапку `test`), иначе `tuned_built: false`.

Для test без GT в RIFE используется флаг `--allow-no-target`.

---

## 4. Ego-маски

| Скрипт | Назначение |
|--------|------------|
| `scripts/ego/export_mask_composer.py --test-only` | Экспорт picker для test |
| `scripts/ego/serve_mask_picker.py` | HTTP-сервер (порт 8765), не открывать через `file://` |
| `scripts/ego/import_mask_picker_selections.py` | Применение выборов → `masks_approved/` |
| `ego_mask_policy.py` | Политика пустых масок |
| `scripts/ego/audit_pipeline_ego_masks.py` | Аудит масок в пайплайне |

**Пустые маски (ZERO):** `crozby_right_fwd`, `natelio_right_fwd`, `orvy_right_fwd` — через `ego_mask_policy.py`.

Маска подставляется в U-Net как канал `art_mask` (не участвует в LiDAR-blend).

---

## 5. Consensus U-Net — инференс (шаг 4)

Скрипт: **`scripts/inference/run_test_consensus_inference.py`**

```bash
# Полный прогон
python scripts/inference/run_test_consensus_inference.py

# Только пересобрать blend/маски (consensus не трогать)
python scripts/inference/run_test_consensus_inference.py --reblend-only

# Принудительно пересчитать U-Net
python scripts/inference/run_test_consensus_inference.py --force-infer
```

### Маршрутизация моделей

| Камеры | Checkpoint |
|--------|------------|
| `front`, `rear` | `artifacts/checkpoints/consensus/consensus_unet_best.pt` |
| `left_fwd`, `right_fwd`, `left_bwd`, `right_bwd` | `artifacts/checkpoints/consensus_side/consensus_unet_side_best.pt` |

### Side canonical (зеркалирование)

Камеры **`left_fwd`** и **`right_bwd`**: перед U-Net все **входные карты** (warp, coverage, depth, ego, mean t0/t1) flip по горизонтали; **pred разворачивается обратно** после сети.

**LiDAR-trust, RIFE и ref** для blend **не flip** — всегда в image space камеры (исправлено в v6).

### Вход U-Net (13 каналов)

Из `consensus_kit.ConsensusDataset` / `prepare_sample()`:

1. warp RGB  
2. coverage  
3. depth (normalized `d1.npy`)  
4. static RGB (нули)  
5. static mask (нули)  
6. ego art_mask  
7. mean(t0, t1)  

Плюс `base_init` и `effective_mask` для partial conv.

---

## 6. LiDAR-маска (r3 + blur)

Логика: **`lidar_density_mask.py`**, как в ноутбуке **`lidar_rife_blend.ipynb`**.

### Построение `lidar_trust`

1. Проекция `input/lidar.npz` в камеру @ pose **`target`** (`meta.json`).
2. Rasterize hits (`splat_radius=2`).
3. **Spread r=3** от hits (`distanceTransform`, linear falloff).
4. **Gaussian blur σ=1** на fine-слое → `density_fine` (**«r3+blur»**).
5. Spike на hits (`trust = max(fine, hits)`).
6. `zone_min=0.12`: значения ниже порога обнуляются.

### Параметры blend-маски

| Параметр | Значение |
|----------|----------|
| `spread_radius_fine` | 3.0 |
| `spread_blur_fine` | 1.0 |
| `spread_radius` | 0 (fine_only) |
| `zone_min` / `mask_thr` | 0.12 |
| `spike_trust` | 1.0 |

Функции:
- `build_lidar_density()` — полный pack
- `lidar_rife_blend_maps()` — trust + blend_mask для сохранения
- `blend_model_rife_lidar()` — смешивание model + RIFE

---

## 7. Варианты blend

### 7.1 Основной — `blend_lidar_rife.jpg`

Скрипт: `scripts/inference/run_test_consensus_inference.py`

```
где lidar_trust ≥ 0.12:
  out = α·consensus + (1−α)·RIFE_blur
иначе:
  out = RIFE
```

| Параметр | По умолчанию |
|----------|--------------|
| `α` (`--blend-alpha`) | 0.55 |
| RIFE blur σ (`--rife-blur-sigma`) | 0.8 |
| mask_thr | 0.12 |

### 7.2 Дополнительный — `blend_blur50.jpg`

Скрипт: **`scripts/inference/run_test_blend_blur.py`**

```bash
python scripts/inference/run_test_blend_blur.py
python scripts/inference/run_test_blend_blur.py --force   # перезаписать все
```

| Параметр | Значение |
|----------|----------|
| consensus blur σ | **1.0** |
| RIFE blur σ | **3.0** |
| α | **0.5** |
| mask_thr | 0.12 |
| **mask_feather_px** | **3.0** (линейный градиент веса от края маски внутрь) |

Читает готовые `consensus_model.jpg`, `rife.jpg`, `lidar_trust.png` — **без GPU и без пересчёта LiDAR**.

---

## 8. Файлы в каждой папке выхода

`consensus_test_outputs/<sample_id>/`:

| Файл | Описание |
|------|----------|
| `consensus_model.jpg` | Выход U-Net (image space камеры) |
| `consensus_raw.npy` | То же, float32 HWC [0..1] |
| `rife.jpg` | RIFE prediction |
| `input_ref.jpg` | target или t0 (референс) |
| `blend_lidar_rife.jpg` | Основной LiDAR-blend |
| `blend_blur50.jpg` | Blur + feather blend |
| `lidar_trust.png` | Grayscale trust [0..1] |
| `lidar_blend_mask.png` | Бинарная маска (trust ≥ 0.12) |
| `lidar_density_fine.png` | r3+blur (если генерировался) |
| `meta_infer.json` | camera, checkpoint, параметры blend |

---

## 9. HTML-галерея

| Скрипт | Выход |
|--------|-------|
| **`scripts/inference/build_test_outputs_gallery.py`** | `consensus_test_outputs/index.html` (все 199) |
| `scripts/inference/build_blend_preview_gallery.py` | `_preview_gallery.html` (по 1 сэмплу на камеру) |

```bash
python scripts/inference/build_test_outputs_gallery.py
```

Колонки: ref · consensus · RIFE · blend · **blend blur50** · lidar_trust · blend mask  

Фильтры в UI: камера, поиск по sample_id, только mirror-камеры.

Открывать `index.html` из папки `consensus_test_outputs` (или через локальный HTTP-сервер).

---

## 10. Вспомогательные скрипты

| Скрипт | Назначение |
|--------|------------|
| `scripts/inference/flip_mirror_lidar_masks.py` | Быстрый горизонтальный flip LiDAR PNG для `left_fwd`/`right_bwd` (legacy fix) |
| `scripts/inference/run_lidar_alpha_val.py` | Перебор α на val (с кэшем) |
| `scripts/ego/audit_pipeline_ego_masks.py` | Отчёт по ego-маскам |

---

## 11. Ноутбуки

| Ноутбук | Содержание |
|---------|------------|
| `consensus_training.ipynb` | Обучение U-Net front/rear |
| `consensus_training_side.ipynb` | Обучение U-Net side + mirror canonical |
| `lidar_rife_blend.ipynb` | LiDAR r3+blur маска + blend model/RIFE, подбор α |
| `lidar_density_zones.ipynb` | Исследование зон LiDAR (spread vs gaussian) |
| `consensus_ensemble_front_rear.ipynb` | Ensemble / val cache |

Генераторы ноутбуков: `build_consensus_training_side.py`, `build_lidar_rife_blend_notebook.py`, `build_lidar_density_notebook.py`.

---

## 12. Ключевые модули Python

| Модуль | Роль |
|--------|------|
| **`consensus_kit.py`** | U-Net, `ConsensusConfig`, dataset, load model, ego mask |
| **`lidar_density_mask.py`** | LiDAR → trust, spread r3+blur, `blend_model_rife_lidar()` |
| **`lidar_depth_map.py`** | Проекция LiDAR в image plane |
| **`ego_mask_policy.py`** | ZERO-маски, политика групп |
| **`scripts/baselines/export_rife_batch.py`** | Batch RIFE inference |

---

## 13. Типичные команды (шпаргалка)

```bash
# 1. Прекомпьюты test
python scripts/inference/precompute_cv_split.py --split test

# 2. Ego-маски (если обновляли picker)
python scripts/ego/import_mask_picker_selections.py

# 3. U-Net + основной blend (или только reblend)
python scripts/inference/run_test_consensus_inference.py
python scripts/inference/run_test_consensus_inference.py --reblend-only

# 4. Доп. blend blur50 + feather
python scripts/inference/run_test_blend_blur.py --force

# 5. Галерея
python scripts/inference/build_test_outputs_gallery.py
```

---

## 14. Известные нюансы и фиксы

1. **Warps + RIFE root:** для `multiview_warping.py` передавать `rife_predictions_v5`, не `.../test`.
2. **Test без GT:** RIFE с `--allow-no-target`; ref = `t0` если нет `target.jpg`.
3. **Mirror LiDAR:** trust/RIFE не flip; только входы U-Net flip для `left_fwd`/`right_bwd`.
4. **Spread distanceTransform:** маска считается от расстояния до hit (инверсия бинарной маски перед `distanceTransform`).
5. **Feather blend:** `mask_feather_px=3` — мягкая склейка на границе LiDAR-зоны (только `blend_blur50`).
6. **Диск C:** при нехватке места warps могли падать — проверять `meta.json` / `coverage.npy`.

---

## 15. Статус test прогона

- Прекомпьюты: **199/199**
- U-Net inference: **199/199** (`consensus_test_outputs/`)
- `blend_lidar_rife.jpg`: **199/199**
- `blend_blur50.jpg`: **199/199** (blur σ1/3, α=0.5, feather 3px)
- Галерея: `consensus_test_outputs/index.html`
