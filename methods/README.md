# Каталог методов (от простого к сложному)

| # | Папка | Метод | Источник |
|---|-------|-------|----------|
| 01 | `01_t0` | Копия кадра t0 | CPU baseline |
| 02 | `02_t1` | Копия кадра t1 | CPU baseline |
| 03 | `03_mean` | (t0 + t1) / 2 | CPU baseline |
| 04 | `04_global_blur` | Gaussian blur t1, σ=2 | testblur |
| 05 | `05_farneback_half` | Warp t1 на полпотока | testdeltadif.py |
| 06 | `06_farneback_alpha` | t0 на α·flow, t1 на (1−α)·flow | symmetric flow |
| 07 | `07_farneback_aggr` | Farneback + blur в зонах ошибки | testdeltadif.py |
| 08 | `08_dis_half` | DIS warp t1 на полпотока | testdeltadif.py |
| 09 | `09_adaptive_soft` | Farneback + adaptive blur (soft) | testdeltadifsoft.py |
| 10 | `10_adaptive_aggr` | Farneback + adaptive blur (aggr) | testdeltadifsoft.py |
| 11 | `11_dis_official` | DIS, оба кадра, α из timestamps | baseline_flow.py |
| 12 | `12_rife` | RIFE neural interpolation | baseline_flow.py |
| 13 | `13_ensemble` | 0.60·DIS + 0.40·RIFE | baseline_flow.py |
| 14 | `14_geo_perpixel` | LiDAR depth, warp обоих в target | layered_parallax.py |
| 15 | `15_layered_parallax` | 8 слоёв + tuned α + blur | layered_parallax.py |
| 16 | `16_phase_blocks` | Phase correlation по блокам | testdeltadif.py |
| 17 | `17_mega_parallax` | depth × semantic: warp каждого куска | mega_parallax.py |

Экспорт: `python export_all_methods.py --num-samples 8`

Результат: `methods_gallery/<метод>/<sample_id>.jpg` + `methods_gallery/_meta/` + `index.html`
