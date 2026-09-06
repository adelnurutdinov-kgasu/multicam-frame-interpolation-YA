# Структура репозитория

Карта каталогов и правила, по которым разложен код. Что это за проект и в каком порядке
идёт пайплайн — в [`../PROJECT_MAP.md`](../PROJECT_MAP.md).

---

## Дерево

```
YA-/
├── README.md                     # индекс + быстрый старт
├── PROJECT_MAP.md                # проект и порядок этапов пайплайна
├── ya_paths.py                   # единая точка правды по путям (REPO + внешний cv_dataset)
│
├── lib/                          # переиспользуемые модули — вся реализация здесь
│   ├── consensus_kit.py              # датасет / модель / инференс consensus U-Net
│   ├── consensus_v3_*.py             # v3: аугментации, датасет, обучающий цикл
│   ├── consensus_v4_*.py             # v4: аугментации, датасет
│   ├── lidar_depth_map.py            # проекция LiDAR в камеру, карты глубины
│   ├── lidar_density_mask.py         # плотность LiDAR, trust-карты, блендинг с RIFE
│   ├── layered_parallax.py           # послойный parallax-варп
│   ├── mega_parallax.py              # parallax с семантикой (YOLO)
│   ├── static_mask.py                # статические маски по optical flow
│   └── ego_mask_extract.py           # извлечение масок кузова
│       ego_mask_policy.py            # политика пустых масок
│
├── scripts/                      # CLI, сгруппированы по этапам пайплайна
│   ├── inference/                    # precompute, test inference, submission, галереи
│   ├── training/                     # обучение consensus v2–v5, eval
│   ├── ego/                          # ego-маски: picker, импорт, валидация
│   ├── baselines/                    # сравнение baseline-методов, RIFE батчем
│   ├── tuning/                       # подбор alpha / blur / static-mask, свипы
│   └── stage2/                       # bake, multiview warps, refine, render depth
│
├── notebooks/
│   ├── training/                     # обучение consensus, LiDAR-blend, зоны плотности
│   └── stage2/                       # rife_depth_refinement_starter
│
├── configs/
│   ├── dataset.yaml                  # схема путей к внешнему датасету
│   └── tuning/                       # подобранные параметры (JSON / CSV)
│
├── docs/                         # эта документация
├── analytics/                    # стратифицированная оценка ошибок + HTML-отчёт
├── yolo/                         # YOLO26: детекция / сегментация / семантика
├── viewer/                       # локальный веб-просмотрщик результатов
├── methods/                      # нумерация методов 01–17
├── experiments/                  # ранние прототипы optical flow / blur
├── tools/                        # генераторы обучающих ноутбуков
├── tests/                        # pytest
├── dataset_sample/               # 6 сэмплов для запуска без внешнего датасета
└── data/dataset/                 # плейсхолдер под внешний датасет
```

---

## Правила раскладки

| Правило | Смысл |
|---------|-------|
| Реализация живёт только в `lib/` | Один модуль — одна ответственность; скрипты и ноутбуки его импортируют, но не дублируют |
| `scripts/` — только точки входа | Каждый файл запускается как `python scripts/<группа>/<имя>.py` и сам добавляет корень репозитория в `sys.path` |
| 8 модулей в корне — реэкспорт | `consensus_kit.py`, `lidar_*.py`, `layered_parallax.py`, `mega_parallax.py`, `static_mask.py`, `ego_mask_*.py` — файлы по ~300 байт, пробрасывающие одноимённые модули из `lib/`. Нужны, чтобы короткий `from consensus_kit import ...` работал из скриптов и ноутбуков |
| Пути к данным — через `ya_paths.py` | Единая точка правды: `configs/dataset.yaml` задаёт схему, `ya_paths.py` собирает пути и читает переменную окружения `YA_CV_DATASET`. Значение по умолчанию — путь машины разработки; в `experiments/` (ранние прототипы) пути ещё захардкожены |
| Подобранные параметры — в `configs/tuning/` | Результат свипов лежит рядом с кодом в JSON, а не в тексте скрипта |

---

## Чего в репозитории нет и почему

| Не хранится | Причина |
|-------------|---------|
| Сам датасет (`final_dataset_v5_participants`, ~320 ГБ) | Внешний; путь задаётся в `ya_paths.py`. Для демонстрации формата есть `dataset_sample/` |
| Веса U-Net (`artifacts/checkpoints/**.pt`) | Десятки мегабайт на файл |
| `methods_gallery/` | Тысячи HTML/JSON/кадров экспорта методов |
| `baseline_files/` | Клон стороннего кода RIFE (ECCV2022-RIFE) |
| Кэши прогонов, логи, превью | Генерируются локально; см. `.gitignore` |
