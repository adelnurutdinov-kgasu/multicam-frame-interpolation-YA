# YOLO: детекция и маски на датасете фото

Пайплайн на **Ultralytics YOLO26** (актуальная линейка, январь 2026). Раньше стоял YOLOv8 — он устарел.

## Два режима в репозитории

| Задача | Скрипт | Модель по умолчанию | Результат |
|--------|--------|---------------------|-----------|
| **Боксы** (где объект) | `yolo/detect_dataset.py` | `yolo26s.pt` | bbox, labels, CSV |
| **Маски** (контур объекта) | `yolo/segment_dataset.py` | `yolo26s-seg.pt` | полигоны, PNG-масски, overlay |

Размеры: `n` быстрее, `s` баланс, `m`/`l`/`x` точнее и тяжелее.

## Быстрый старт

```powershell
cd <корень репозитория>
pip install -U -r requirements-yolo.txt

python yolo/detect_dataset.py --dry-run
python yolo/detect_dataset.py

# маски вместо только боксов
python yolo/segment_dataset.py
```

Результаты: `runs/detect/...` или `runs/segment/...`.

### Сегментация (маски)

| Папка / файл | Содержимое |
|--------------|------------|
| `images/` | Визуализация с масками |
| `masks/<кадр>/` | Бинарная маска на каждый объект |
| `overlay/` | Цветная заливка поверх исходника |
| `labels_seg/` | Полигоны YOLO-seg (.txt) |
| `segments.jsonl`, `segments.csv` | Метаданные и площади |

Конфиг: **`yolo/config_seg.yaml`**.

## Какую модель выбрать (YOLO26)

- **Детекция:** `yolo26n.pt` → `yolo26s.pt` (дефолт) → `yolo26m.pt`
- **Маски:** `yolo26n-seg.pt` → `yolo26s-seg.pt` (дефолт) → `yolo26m-seg.pt`

YOLO26 vs YOLO11: NMS-free инференс, быстрее на CPU, лучше маски на `-seg`.  
Документация: [YOLO26](https://docs.ultralytics.com/models/yolo26/), [Segment](https://docs.ultralytics.com/tasks/segment/).

Обновить пакет перед первым запуском YOLO26:

```powershell
pip install -U ultralytics
```

## Другие способы поиска объектов (кроме боксов)

Кратко, что имеет смысл для дорожных сцен:

| Подход | Что даёт | Когда брать |
|--------|----------|-------------|
| **YOLO26 detect** (`yolo26s.pt`) | Боксы + класс COCO | Быстрый обзор, счётчики, фильтр кадров |
| **YOLO26-seg** (`yolo26s-seg.pt`) | **Маска на каждый объект** | Нужна форма машины/человека, площадь, точный контур |
| **YOLO26-sem** (`yolo26n-sem.pt`) | Пиксельные классы без отдельных инстансов | «Вся дорога / небо / тротуар», не «3 машины» |
| **SAM / SAM2** | Маски по клику или авто, **без имён классов** | Вырезать «что угодно», разметка, уточнение после YOLO |
| **Grounding DINO + SAM** | Маска по **тексту** («pole», «traffic sign») | Столбы и редкие классы, которых нет в COCO |
| **YOLO-World / YOLOE-26** | Детекция по **своим словам-классам** | «pole», «barrier» без полного дообучения |
| **Дообучение YOLO** | Свои классы + боксы/маски | Столбы, конусы, разметка под ваш датасет |

**Практичная связка для вашего кейса:**

1. `detect_dataset.py` — быстро найти людей/машины/знаки.  
2. `segment_dataset.py` — маски для тех же COCO-классов.  
3. Столбы — COCO не покрывает → текстовый поиск (Grounding DINO + SAM) или дообучение `-seg` на своих масках.

## Про столбы

В COCO нет класса **pole**. YOLO26 найдёт person, car, traffic light, stop sign и т.д., но не опоры.

Варианты: open-vocabulary (YOLOE / Grounding DINO), SAM с ручным промптом, своё обучение.

## Настройка

**`yolo/config.yaml`** — детекция, **`yolo/config_seg.yaml`** — маски.

- `classes` — фильтр COCO; пустой список = все 80 классов  
- `path_filters` — `all` или подстроки (`target`, `input/t1`)  
- `conf` — порог уверенности  

Пример для структуры `final_dataset_v5`: **`yolo/config_dataset_v5.yaml`**.

## Команды

```powershell
python yolo/detect_dataset.py --dataset D:\photos --model yolo26m.pt --device cuda:0
python yolo/segment_dataset.py --model yolo26m-seg.pt --limit 50
python yolo/detect_dataset.py --config yolo/config_dataset_v5.yaml
```

## Зависимости

`requirements-yolo.txt` — [Ultralytics](https://docs.ultralytics.com/).
