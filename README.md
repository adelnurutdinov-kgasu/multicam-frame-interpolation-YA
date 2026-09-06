# Интерполяция кадров на многокамерном датасете

Синтез кадра целевой камеры в промежуточный момент времени: на вход — surround-камеры
в моменты `t0` и `t1` плюс агрегированное облако точек LiDAR, на выход — предсказанный
кадр между ними.

**Пайплайн:** baseline-методы → LiDAR parallax → consensus U-Net → ego-маски →
test inference → submission.

**Стек:** Python, PyTorch (U-Net «consensus»), OpenCV (Farneback / DIS optical flow),
RIFE (нейросетевая интерполяция), Ultralytics YOLO26 (детекция / сегментация / семантика),
проекция LiDAR и карты глубины.

## Документация

| Файл | Содержание |
|------|------------|
| [PROJECT_MAP.md](PROJECT_MAP.md) | **Начать отсюда:** что за проект и в каком порядке идёт пайплайн |
| [docs/REPO_LAYOUT.md](docs/REPO_LAYOUT.md) | Структура репозитория и правила раскладки |
| [docs/TEST_INFERENCE_PIPELINE.md](docs/TEST_INFERENCE_PIPELINE.md) | Test inference end-to-end |
| [docs/CONSENSUS_V3.md](docs/CONSENSUS_V3.md) | Обучение consensus U-Net |
| [docs/README_CV_DATASET.md](docs/README_CV_DATASET.md) | Формат датасета |
| [docs/README_YOLO.md](docs/README_YOLO.md) | YOLO-аннотации |
| [analytics/README.md](analytics/README.md) | Стратифицированная оценка ошибок |

## Структура

```
lib/          реализация: consensus U-Net, LiDAR-глубина, parallax, маски
scripts/      точки входа по этапам: inference / training / ego / baselines / tuning / stage2
notebooks/    обучение и эксперименты
configs/      пути к данным и подобранные параметры
docs/         документация
analytics/    оценка ошибок + HTML-отчёт
yolo/         YOLO26-пайплайны для семантики
viewer/       локальный просмотрщик результатов
```

Восемь `.py` в корне (`consensus_kit.py`, `lidar_*.py`, `layered_parallax.py`,
`mega_parallax.py`, `static_mask.py`, `ego_mask_*.py`) — тонкие реэкспорты из `lib/`,
чтобы работал короткий `from consensus_kit import ...`. Реализации там нет.

## Запуск (test)

```bash
python scripts/inference/precompute_cv_split.py --split test
python scripts/inference/run_test_consensus_inference.py
python scripts/inference/run_test_blend_blur.py --force
python scripts/inference/export_submission_blend50.py
```

Ego-маски (ручной выбор через локальный picker):

```bash
python scripts/ego/export_mask_composer.py --test-only
python scripts/ego/serve_mask_picker.py      # http://127.0.0.1:8765/composer_test.html
python scripts/ego/import_mask_picker_selections.py
```

## Данные

Полный датасет (`final_dataset_v5_participants`, ~320 ГБ) в репозитории не лежит.
Пути задаются в [`ya_paths.py`](ya_paths.py) и [`configs/dataset.yaml`](configs/dataset.yaml),
переопределяются переменной окружения `YA_CV_DATASET`.

Формат сэмпла показан на маленькой выборке в [`dataset_sample/`](dataset_sample/):
6 сцен, для каждой — 6 камер в `t0` и `t1`, целевой кадр и `meta.json`.

Веса U-Net, экспорт галерей методов и клон RIFE тоже вне git — см.
[«Чего в репозитории нет и почему»](docs/REPO_LAYOUT.md#чего-в-репозитории-нет-и-почему).
