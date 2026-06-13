import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def calculate_psnr(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError("Размеры изображений не совпадают")
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def score_from_psnr(psnr):
    return max(0.0, min(100.0, (psnr - 10) / 20.0 * 100))

base_path = "/Users/adel/Documents/Claude/Projects/Интерполяция фото/final_dataset_v5_participants_small/train"
sample_id = "2025-10-04_15_57_04_16_44_38_luka_1759586458800193000__000"
target_path = os.path.join(base_path, sample_id, "target", "left_fwd.jpg")
input_path = os.path.join(base_path, sample_id, "input", "t1", "left_fwd.jpg")

target = cv2.imread(target_path)
pred_raw = cv2.imread(input_path)

# Выбираем сигму, при которой был хороший прирост (например, 2.0)
sigma = 2.0
pred_blur = cv2.GaussianBlur(pred_raw, (0, 0), sigmaX=sigma, sigmaY=sigma)

# Вычисляем PSNR и score
psnr_raw = calculate_psnr(target, pred_raw)
psnr_blur = calculate_psnr(target, pred_blur)
print(f"Без размытия: PSNR={psnr_raw:.2f} dB, score={score_from_psnr(psnr_raw):.2f}")
print(f"С размытием σ={sigma}: PSNR={psnr_blur:.2f} dB, score={score_from_psnr(psnr_blur):.2f}")

# --- Визуализация ---
# Переводим BGR в RGB для отображения
target_rgb = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
pred_raw_rgb = cv2.cvtColor(pred_raw, cv2.COLOR_BGR2RGB)
pred_blur_rgb = cv2.cvtColor(pred_blur, cv2.COLOR_BGR2RGB)

# 1. Карта абсолютной разницы для размытого предсказания (по каналам RGB, усреднённая)
diff_blur = np.mean(np.abs(pred_blur.astype(np.float64) - target.astype(np.float64)), axis=2)

# 2. Карта разности квадратов ошибок: (pred_blur - GT)^2 - (pred_raw - GT)^2
# Отрицательные значения -> размытие уменьшило ошибку, положительные -> увеличило.
sq_error_raw = np.sum((pred_raw.astype(np.float64) - target.astype(np.float64))**2, axis=2)
sq_error_blur = np.sum((pred_blur.astype(np.float64) - target.astype(np.float64))**2, axis=2)
delta_sq_error = sq_error_blur - sq_error_raw   # если <0, то blur помог

# Создаём фигуру с подграфиками
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

# 1. GT
axes[0].imshow(target_rgb)
axes[0].set_title("Ground Truth (GT)")
axes[0].axis('off')

# 2. Размытое предсказание (σ=2.0)
axes[1].imshow(pred_blur_rgb)
axes[1].set_title(f"Predicted (t1 blurred σ={sigma})")
axes[1].axis('off')

# 3. Тепловая карта абсолютной ошибки (размытого предсказания)
im3 = axes[2].imshow(diff_blur, cmap='hot', interpolation='nearest')
axes[2].set_title("Absolute Error (|pred - GT|)")
axes[2].axis('off')
plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

# 4. Исходное t1 (без размытия)
axes[3].imshow(pred_raw_rgb)
axes[3].set_title("Predicted (t1 raw)")
axes[3].axis('off')

# 5. Тепловая карта разности квадратов ошибок (где blur помог/навредил)
# Нормируем для цветовой карты: отрицательные (помощь) -> синий, положительные (вред) -> красный
vmin, vmax = np.percentile(delta_sq_error, [5, 95])  # для лучшей контрастности
im5 = axes[4].imshow(delta_sq_error, cmap='RdBu_r', vmin=vmin, vmax=vmax, interpolation='nearest')
axes[4].set_title("Δ Squared Error (blur - raw)\nBlue = improvement, Red = degradation")
axes[4].axis('off')
plt.colorbar(im5, ax=axes[4], fraction=0.046, pad=0.04)

# 6. Наложение: GT с выделением областей, где размытие помогло (зелёный) или навредило (красный)
# Создаём маски
improve_mask = delta_sq_error < 0
degrade_mask = delta_sq_error > 0
overlay = target_rgb.copy()
# Увеличим яркость в каналах: улучшение -> зелёный оттенок, ухудшение -> красный
overlay[improve_mask] = np.clip(overlay[improve_mask] * 0.7 + [0, 100, 0], 0, 255)  # зеленый
overlay[degrade_mask] = np.clip(overlay[degrade_mask] * 0.7 + [100, 0, 0], 0, 255)  # красный
axes[5].imshow(overlay)
axes[5].set_title("Regions of improvement (green) / degradation (red)")
axes[5].axis('off')

plt.tight_layout()
plt.show()