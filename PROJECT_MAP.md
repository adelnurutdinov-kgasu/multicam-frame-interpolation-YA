# PROJECT_MAP — что это и в каком порядке

Обзорная карта репозитория: назначение каждой папки/файла и **очередность пайплайна**.
Документ только описывает проект — ничего не переносит и не меняет.
Составлено автоматическим обзором; за деталями — `docs/`.

---

## 0. Что это за проект

Решение задачи **интерполяции/синтеза кадра** на многокамерном автономном датасете
(`final_dataset_v5_participants`): по surround-камерам в моменты `t0` и `t1` + агрегированному
LiDAR нужно предсказать кадр целевой камеры в промежуточный момент.

Сквозной пайплайн (из `README.md`):

> baselines → LiDAR parallax → consensus U-Net → ego-маски → test inference → submission

Стек: Python, PyTorch (U-Net «consensus»), OpenCV (Farneback/DIS optical flow),
RIFE (нейросетевая интерполяция), Ultralytics YOLO26 (детекция/сегментация/семантика),
LiDAR-проекция и глубинные карты.

Код разложен по `lib/` (переиспользуемые модули) и `scripts/` (CLI по этапам пайплайна);
структура описана в `docs/REPO_LAYOUT.md`.

---

## 1. Очередность пайплайна (главное)

Этапы по порядку. Для каждого — канонический код в `scripts/`/`lib/` и команда запуска.

### Этап 0 — Данные и пути
- Формат сэмпла: `docs/README_CV_DATASET.md`; пример выборки: `dataset_sample/`.
- Все пути к внешнему датасету: **`ya_paths.py`** и `configs/dataset.yaml`
  (переопределяются переменной окружения `YA_CV_DATASET`).
- Сам датасет в git не хранится (внешний, ~320 ГБ).

### Этап A — Baseline-методы и их сравнение
1. Прототипы оптического потока/блюра: `experiments/testdelta*.py`, `testblur.py`, `testconv.py`.
2. Экспорт всех методов в галерею: `scripts/baselines/export_all_methods.py` → `methods_gallery/` (локальная папка, в снимок не входит).
   - Каталог методов 01–17 (от простого к сложному): см. **`methods/README.md`**.
   - Результат: `methods_gallery/<метод>/<sample>.jpg` + `methods_gallery/index.html`.
3. RIFE батчем: `scripts/baselines/export_rife_batch.py`; сравнение: `scripts/baselines/compare_baselines.py`, `scripts/baselines/visualize_baselines.py`.
4. Стратифицированная аналитика ошибок по глубине/семантике: `analytics/` (`run_stratified_eval.py` → `build_report.py` → `analytics/out/report.html`).

### Этап B — YOLO-аннотации (вспомогательные)
- `yolo/detect_dataset.py` (боксы), `yolo/segment_dataset.py` (маски), `yolo/sem_dataset.py` (семантика).
- Конфиги: `yolo/config*.yaml`; документация: **`docs/README_YOLO.md`**.
- Используются для ego-артефактов (см. этап D) и аналитики (этап A.4).

### Этап C — Прекомпьюты (оркестратор `scripts/inference/precompute_cv_split.py`)
Команда: `python scripts/inference/precompute_cv_split.py --split {train|test}` →
`scripts/inference/precompute_cv_split.py`. Шаги:

| Шаг | Скрипт | Выход (во внешнем cv_dataset) |
|-----|--------|------|
| bake (глубина) | `scripts/stage2/bake_refinement_assets.py` | `rife_refinement_baked/` (`d*.npy`, meta) |
| rife | `scripts/baselines/export_rife_batch.py --allow-no-target` | `rife_predictions_v5/` |
| warps | `scripts/stage2/multiview_warping.py` | `multiview_warps/` (`consensus_raw.npy`, `coverage.npy`, …) |
| static/far | `scripts/tuning/precompute_static_masks.py`, `scripts/inference/precompute_lidar_trust_and_warp_mix.py` | `precomputed_static_far/`, lidar_trust |

LiDAR-логика: `lib/lidar_depth_map.py` (проекция), `lib/lidar_density_mask.py` (trust r3+blur, blend).

### Этап D — Ego-маски (артефакты кузова)
1. `scripts/ego/export_mask_composer.py --test-only` — собрать picker.
2. `scripts/ego/serve_mask_picker.py` — локальный сервер (порт 8765), ручной выбор масок.
3. `scripts/ego/import_mask_picker_selections.py` — применить выбор → `methods_gallery/_ego_manual_masks/masks_approved/`.
4. Политика пустых масок: `lib/ego_mask_policy.py`; аудит: `scripts/ego/audit_pipeline_ego_masks.py`.
5. Обучение YOLO-сегментации ego-артефакта: `scripts/ego/*`, `yolo/train_ego_artifact_seg.py`.

### Этап E — Обучение Consensus U-Net
- Ноутбуки: `notebooks/training/consensus_training*.ipynb`, `lidar_rife_blend.ipynb`, `lidar_density_zones.ipynb`.
- Скрипты (актуальные версии): `scripts/training/train_consensus_v3.py` / `v4` / `v5`,
  `train_v2_ycrcb_side.py`; библиотека: `lib/consensus_kit.py`, `lib/consensus_v3_*.py`, `lib/consensus_v4_*.py`.
- Описание V3 (аугментации/регуляризация): **`docs/CONSENSUS_V3.md`**.
- Чекпоинты: `artifacts/checkpoints/consensus*/...` (в git не коммитятся, `*.pt` в `.gitignore`).

### Этап F — Test inference и submission
Полное описание: **`docs/TEST_INFERENCE_PIPELINE.md`**. Порядок:

```
python scripts/inference/precompute_cv_split.py --split test        # этап C для test
python scripts/inference/run_test_consensus_inference.py            # U-Net + основной LiDAR-blend
python scripts/inference/run_test_blend_blur.py --force             # доп. blend (blur+feather)
python scripts/inference/export_submission_blend50.py               # финальный submission
python scripts/inference/build_test_outputs_gallery.py              # HTML-галерея 199 сэмплов
```
- Маршрутизация моделей: front/rear → `consensus`, side → `consensus_side` (+ mirror canonical для `left_fwd`/`right_bwd`).
- Доп. экспорт: `scripts/inference/export_submission_v4.py`, `scripts/inference/export_submission_v3_v4_blend.py`.

### Этап G — Тюнинг (по необходимости, влияет на C/F)
- `scripts/tuning/`: `scripts/tuning/tune_layered_alpha.py`, `scripts/tuning/tune_layered_blur.py`, `scripts/tuning/tune_static_mask.py`,
  `sweep_far_boundary.py`, `benchmark_far_static.py`.
- Результаты тюнинга: `configs/tuning/*.json` (alpha/blur/static_mask/far_boundary).

### Просмотр результатов
- `viewer/app.py` + `viewer/render.py` — локальный просмотрщик.
- Галереи: `methods_gallery/index.html`, `consensus_test_outputs/index.html` (внешний).

---

## 2. Карта папок и ключевых файлов

| Путь | Что это |
|------|---------|
| `README.md` | Индекс + быстрый старт |
| `PROJECT_MAP.md` | Этот файл |
| `ya_paths.py` | Единые пути (REPO + внешний cv_dataset) |
| `docs/` | Документация: `REPO_LAYOUT.md`, `TEST_INFERENCE_PIPELINE.md`, `README_CV_DATASET.md`, `README_YOLO.md`, `CONSENSUS_V3.md` |
| `configs/` | `dataset.yaml` (пути), `tuning/` (JSON параметры) |
| `lib/` | Переиспользуемые модули: `consensus_kit`, `consensus_v3/v4_*`, `lidar_depth_map`, `lidar_density_mask`, `layered_parallax`, `mega_parallax`, `static_mask`, `ego_mask_*` |
| `scripts/inference/` | Test-пайплайн, submission, галереи, precompute |
| `scripts/stage2/` | bake / multiview warps / refine / render depth (бывший «Второй этап») |
| `scripts/ego/` | Ego-маски: picker, импорт, валидация, YOLO-артефакт |
| `scripts/baselines/` | compare, RIFE batch, export methods, mega_parallax debug |
| `scripts/training/` | Обучение consensus v2–v5, eval |
| `scripts/tuning/` | tune_*, static/far, sweep |
| `notebooks/training/` | Обучающие ноутбуки consensus/lidar |
| `notebooks/stage2/` | `rife_depth_refinement_starter.ipynb` |
| `tools/` | Генераторы обучающих ноутбуков (`build_*_notebook.py`) |
| `experiments/` | Прототипы optical-flow/blur (`testdelta*`, `testblur`, …) |
| `methods/` | `README.md` — нумерация методов 01–17 |
| `methods_gallery/` | *(локально, не в снимке)* экспорт методов 01–16, ego-артефакты, preview, `index.html` |
| `yolo/` | YOLO26 детекция/сегментация/семантика + конфиги |
| `analytics/` | Стратифицированная оценка ошибок (depth/semantic) + HTML-отчёт |
| `viewer/` | Локальный веб-просмотрщик результатов |
| `artifacts/` | *(локально, не в снимке)* `checkpoints/` — веса U-Net, `preview/`, eval JSON |
| `baseline_files/` | *(локально, не в снимке)* RIFE (ECCV2022-RIFE) + train_log |
| `dataset_sample/` | Маленькая выборка датасета (6 сэмплов + прекомпьюты) для GitHub/демо |
| `tests/` | `test_mega_parallax.py` |
| `data/` | `dataset/.gitkeep` — плейсхолдер (содержимое в .gitignore) |

### Корневые модули = реэкспорт из `lib/`
В корне лежат 8 файлов по ~300 байт (`consensus_kit.py`, `lidar_*.py`, `layered_parallax.py`,
`mega_parallax.py`, `static_mask.py`, `ego_mask_*.py`) — тонкие реэкспорты соответствующих
модулей `lib/`. Нужны, чтобы работал короткий `from consensus_kit import ...` из скриптов
и ноутбуков. Реализация — только в `lib/`.

### Внешние данные/артефакты (НЕ в git)
Задаются в `ya_paths.py`: `CV_ROOT=…/cv_dataset`, `DATASET_ROOT`, `BAKED_ROOT`,
`RIFE_ROOT`, `WARPS_ROOT`, `CONSENSUS_TEST_OUT`, `SUBMISSION_ROOT`, `STATIC_FAR_ROOT`.

---

## 3. Документация-источники
`README.md`, `docs/REPO_LAYOUT.md`, `docs/TEST_INFERENCE_PIPELINE.md`,
`docs/README_CV_DATASET.md`, `docs/README_YOLO.md`, `docs/CONSENSUS_V3.md`,
`methods/README.md`, `analytics/README.md`, `dataset_sample/README.md`.
