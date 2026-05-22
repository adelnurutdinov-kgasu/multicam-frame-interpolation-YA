import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm  # pip install tqdm если ещё нет

# ========== НАСТРОЙКИ ==========
BASE_PATH = "/Users/adel/Documents/GitHub/YA Интерполяция фото/final_dataset_v5_participants_small/train"
CAMERAS = ["front", "left_fwd", "left_bwd", "right_fwd", "right_bwd", "rear"]

# Параметры адаптивного размытия (soft, как в последнем скрипте)
ADAPTIVE_PARAMS = {
    "base_sigma": 0.3,
    "alpha": 0.4,
    "beta": 0.15,
    "max_sigma": 4.0,
    "sigma_levels": [0.3, 0.5, 1.0, 2.0, 4.0]
}
# ===============================

def warp_half(img, flow):
    h, w = flow.shape[:2]
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = x - flow_x / 2.0
    map_y = y - flow_y / 2.0
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)

def compensation_error(img_t0, img_t1, flow):
    h, w = flow.shape[:2]
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x_full = x - flow_x
    map_y_full = y - flow_y
    warped = cv2.remap(img_t1, map_x_full, map_y_full,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    return np.mean(np.abs(img_t0.astype(np.float32) - warped.astype(np.float32)), axis=2)

def adaptive_blur(img_t1, flow, comp_err, params):
    """Возвращает адаптивно размытое изображение на основе pred_flow и карты сигм."""
    flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    sigma_map = params["base_sigma"] + params["alpha"] * flow_mag + params["beta"] * comp_err
    sigma_map = np.clip(sigma_map, params["sigma_levels"][0], params["max_sigma"])

    # Интерполяция на полпотока
    pred_flow = warp_half(img_t1, flow)

    # Готовим размытые версии
    blurred_levels = []
    for s in params["sigma_levels"]:
        blurred = cv2.GaussianBlur(pred_flow, (0, 0), sigmaX=s, sigmaY=s)
        blurred_levels.append(blurred)

    pred_adaptive = np.zeros_like(pred_flow, dtype=np.float32)
    for i, s_val in enumerate(params["sigma_levels"][:-1]):
        s_next = params["sigma_levels"][i+1]
        mask = (sigma_map >= s_val) & (sigma_map < s_next)
        if not np.any(mask):
            continue
        alpha_mix = (sigma_map[mask] - s_val) / (s_next - s_val)
        alpha_mix = alpha_mix[..., np.newaxis]
        pred_adaptive[mask] = (blurred_levels[i][mask] * (1 - alpha_mix) +
                               blurred_levels[i+1][mask] * alpha_mix)
    mask_max = sigma_map >= params["sigma_levels"][-1]
    pred_adaptive[mask_max] = blurred_levels[-1][mask_max]
    return pred_adaptive.astype(np.uint8)

def calc_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64))**2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def score_from_psnr(psnr):
    return max(0.0, min(100.0, (psnr - 10) / 20.0 * 100))

# ---------- Сбор всех сэмплов ----------
samples = sorted([d for d in Path(BASE_PATH).iterdir() if d.is_dir()])
print(f"Найдено сэмплов: {len(samples)}")

# Хранилище метрик для каждого метода
metrics = {
    "raw_t1": [],
    "global_blur": [],
    "adaptive_soft": []
}

for sample_path in tqdm(samples, desc="Обработка сэмплов"):
    sample_id = sample_path.name
    for cam in CAMERAS:
        t0_path = sample_path / "input" / "t0" / f"{cam}.jpg"
        t1_path = sample_path / "input" / "t1" / f"{cam}.jpg"
        gt_path = sample_path / "target" / f"{cam}.jpg"

        if not (t0_path.exists() and t1_path.exists() and gt_path.exists()):
            print(f"  Пропуск {sample_id}/{cam} – нет файлов")
            continue

        img_t0 = cv2.imread(str(t0_path))
        img_t1 = cv2.imread(str(t1_path))
        img_gt = cv2.imread(str(gt_path))

        if img_t0 is None or img_t1 is None or img_gt is None:
            continue

        # Переводим в серый для потока
        gray0 = cv2.cvtColor(img_t0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2GRAY)

        # ---------- Метод 1: raw t1 ----------
        psnr_raw = calc_psnr(img_gt, img_t1)
        metrics["raw_t1"].append(score_from_psnr(psnr_raw))

        # ---------- Метод 2: глобальное размытие σ=2.0 ----------
        pred_blur = cv2.GaussianBlur(img_t1, (0, 0), sigmaX=2.0, sigmaY=2.0)
        psnr_blur = calc_psnr(img_gt, pred_blur)
        metrics["global_blur"].append(score_from_psnr(psnr_blur))

        # ---------- Метод 3: адаптивное размытие (soft) ----------
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray0, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        comp_err = compensation_error(img_t0, img_t1, flow)
        pred_adapt = adaptive_blur(img_t1, flow, comp_err, ADAPTIVE_PARAMS)
        psnr_adapt = calc_psnr(img_gt, pred_adapt)
        metrics["adaptive_soft"].append(score_from_psnr(psnr_adapt))

# ---------- Вывод итогов ----------
print("\n========== СРЕДНИЕ РЕЗУЛЬТАТЫ ==========")
for method, scores in metrics.items():
    if scores:
        avg_score = np.mean(scores)
        # Приблизительный средний PSNR (из score обратно не восстановить точно, поэтому выводим score)
        print(f"{method:25s}: avg score = {avg_score:.2f}  (по {len(scores)} примерам)")

# Если нужен и средний PSNR, соберём отдельно
print("\nСредний PSNR (по тем же примерам):")
# Для PSNR можно хранить параллельно, но проще пересчитать, собрав заново или сохранив в списки
# Но чтобы не усложнять, просто выведем предупреждение
print("Для точного PSNR нужно сохранять сами значения PSNR. В этой версии только score.")