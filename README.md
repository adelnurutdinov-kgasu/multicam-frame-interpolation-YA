# YA- — интерполяция кадров (CV dataset)

Пайплайн: baselines → LiDAR parallax → consensus U-Net → ego-маски → test inference → submission.

## Документация

| Файл | Содержание |
|------|------------|
| [docs/REPO_LAYOUT.md](docs/REPO_LAYOUT.md) | **Структура репозитория** (что куда перенесено) |
| [docs/TEST_INFERENCE_PIPELINE.md](docs/TEST_INFERENCE_PIPELINE.md) | Test inference end-to-end |
| [docs/README_CV_DATASET.md](docs/README_CV_DATASET.md) | Формат датасета |
| [docs/README_YOLO.md](docs/README_YOLO.md) | YOLO-аннотации |

## Быстрый старт (test)

```bash
# из корня репозитория
python precompute_cv_split.py --split test
python run_test_consensus_inference.py
python run_test_blend_blur.py --force
python export_submission_blend50.py
```

Пути к данным: `ya_paths.py` или переменная окружения `YA_CV_DATASET`.

## Ego-маски (picker)

```bash
python serve_mask_picker.py
# http://127.0.0.1:8765/composer_test.html
python import_mask_picker_selections.py
```

## Основные папки

- `lib/` — библиотеки (consensus, lidar, parallax, ego)
- `scripts/` — CLI (inference, ego, baselines, tuning, stage2)
- `notebooks/` — обучение и stage2
- `methods_gallery/` — экспорт методов и ego-артефакты
- `artifacts/checkpoints/` — веса U-Net
- `configs/` — dataset.yaml, tuning JSON
