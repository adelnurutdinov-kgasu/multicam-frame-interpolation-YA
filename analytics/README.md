# Аналитика: ошибки по глубине и семантике

Сравнение методов **06_farneback_alpha**, **13_ensemble**, **15_layered_parallax**, **17_mega_parallax**:
- где больше/меньше ошибаемся по **глубине LiDAR** (5–6 bin-ов в метрах);
- по **семантическим слоям** (road, sky, vegetation, pole… из `semantic.jsonl`).

GPU: RIFE + torch для карт MAE.

## Запуск

```powershell
(colabus) cd <корень репозитория>

# все train-сэмплы (долго; checkpoint каждые 50)
python analytics/run_stratified_eval.py

# быстрый тест
python analytics/run_stratified_eval.py --num-samples 100 --split test

# продолжить после обрыва
python analytics/run_stratified_eval.py --resume

# догнать 17_mega_parallax, если отчёт был на 3 метода
python analytics/run_stratified_eval.py --backfill-new

# отчёт HTML
python analytics/build_report.py
```

Если **17_mega_parallax** в report.html с прочерками — запустите `--backfill-new`, затем снова `build_report.py`.

Откройте `analytics/out/report.html` — таблицы с heatmap (зелёный = меньше MAE).

## Выход

| Файл | Содержимое |
|------|------------|
| `analytics/out/stratified_results.json` | MAE/PSNR/pixels по depth и semantic |
| `analytics/out/stratified_checkpoint.json` | для `--resume` |
| `analytics/out/report.html` | визуализация |

## Конфиг `analytics/config.yaml`

- `depth_edges_m` — границы bin-ов глубины в метрах
- `semantic_classes` — фильтр классов (пусто = все)
- `device: cuda`
- tuned alpha/blur подтягиваются из `alpha_tune_best.json`, `blur_tune_best.json`

## Интерпретация

- **MAE** — средняя абсолютная ошибка яркости (0–255, меньше = лучше)
- **PSNR** — derived from MSE per stratum
- Сравнивайте методы в одной строке (road, 30-60m depth): у кого MAE ниже — там метод лучше
