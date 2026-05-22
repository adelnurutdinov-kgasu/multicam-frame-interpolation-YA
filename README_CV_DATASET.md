# cv_dataset — команды для прогона и просмотра

Датасет: `C:\Users\adel\Downloads\cv_dataset\final_dataset_v5_participants` (~19 002 jpg)  
Аннотации пишутся рядом, **картинки не копируются**.

## 1. Прогон YOLO (запустите сами)

```powershell
cd C:\Users\adel\Documents\GitHub\YA-
pip install -r requirements-yolo.txt

python yolo/detect_dataset.py --config yolo/config_cv_dataset.yaml --dry-run
python yolo/detect_dataset.py --config yolo/config_cv_dataset.yaml
python yolo/segment_dataset.py --config yolo/config_cv_dataset_seg.yaml
```

Результаты:

| Путь | Содержимое |
|------|------------|
| `annotations/detect/labels/` | YOLO bbox `.txt` |
| `annotations/detect/detections.jsonl` | боксы для viewer |
| `annotations/segment/labels_seg/` | полигоны YOLO-seg |
| `annotations/segment/segments.jsonl` | полигоны для viewer |

`save_images: false`, `save_masks: false` — визуализации на диск не пишутся.

GPU: `--device cuda:0`  
Тест: `--limit 100`

## 2. Просмотр

```powershell
pip install -r requirements-viewer.txt
python viewer/app.py
```

http://127.0.0.1:7860 — overlay в памяти поверх оригиналов.

## 3. Структура кадра

```
final_dataset_v5_participants/
  train|test/<sample_id>/input/t0/front.jpg ...
```

Все камеры в прогоне. Только target: `path_filters: ["/target/"]` в конфиге.
