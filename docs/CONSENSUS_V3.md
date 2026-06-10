# Consensus V3

Stage-1 V2 остаётся базой (7 каналов YCrCb: `warp_mix` + `lidar_trust` + `mean(t0,t1)`). V3 добавляет **обучающие аугментации**, чтобы компенсировать малый объём данных (~1278 сэмплов).

## Аугментации (train)

| Приём | По умолчанию | Зачем |
|--------|--------------|--------|
| **virtual_mirror_2x** | вкл. | Каждый сэмпл дважды: как есть + flip H в canonical-space → ~2× эпох |
| **random hflip** | 0.5, если 2× выкл. | Симметрия сцены в canonical |
| **random crop** | 416×768 | Больше «патчей» с полного 544×1024 |
| **Y gain + chroma jitter** | ±10% Y, ±0.03 Cr/Cb | Освещение / баланс |
| **noise на warp** | σ≈0.012 | Робастность к артефактам warp |

**Val:** полный кадр, без аугментаций (как V2).

## Mirror policy (как V2)

- На диске — camera-space.
- `left_fwd` / `right_bwd` → flip H перед аугментациями.
- На инференсе — разворот выхода обратно.

## Регуляризация (fine-tune от Stage-1)

Если **best val на эпохе 1–2**, а train падает дальше — типичное переобучение. В скрипте по умолчанию:

| Приём | Параметр | Смысл |
|--------|-----------|--------|
| Anchor к warp | `--anchor-lambda 0.08` | штраф `\|pred − base_init\|²` — не уезжать от warp |
| Trust-weighted MSE | `--trust-floor 0.25` | меньше фита в «дырах» без LiDAR |
| EMA для val/best | `--ema-decay 0.999` | сглаженные веса для метрики |
| Weight decay | `--weight-decay 5e-4` | L2 на веса |
| Cosine LR | `--min-lr 1e-6` | затухание lr |
| Early stop | `--patience 8` | стоп, если val не улучшается |
| Cutout на warp | `--strong-aug` | сильнее входные аугментации |

Продолжить с **лучшего V3** (не Stage-1):

```powershell
python scripts/training/train_consensus_v3.py --continue-ckpt artifacts/checkpoints/consensus_v3_all/consensus_v3_best.pt --epochs 40 --batch-size 8 --lr 5e-5 --strong-aug
```

## Запуск обучения

```powershell
cd c:\Users\adel\Documents\GitHub\YA-
python scripts/training/train_consensus_v3.py --epochs 80 --batch-size 8 --patch-h 416 --patch-w 768
```

Чекпоинты: `artifacts/checkpoints/consensus_v3_all/consensus_v3_best.pt`

Продолжить с V2 весов:

```powershell
python scripts/training/train_consensus_v3.py --resume artifacts/checkpoints/consensus_v2_all/consensus_v2_all_best.pt
```

## Что ещё можно добавить (V3.1+)

- **Mixup** только в зоне `trust > 0.5` (осторожно с YCrCb).
- **Cutout** на warp-канале (закрыть дыры → учить inpaint).
- **RIFE как 8-й канал** или post-fusion (как на val sweep).
- **TTA на инференсе:** flip H + average (бесплатно +0.1–0.3 dB).
- **Псевдо-лейблы** на test без GT — не для метрик, только если разрешат.

## Отличие от V2

| | V2 | V3 |
|---|----|----|
| Аугментации | нет | mirror 2× + crop + color |
| Эффективный train N/эпоху | ~1074 | ~2148 |
| Код | ноутбук | `lib/consensus_v3_*.py` + скрипт |
