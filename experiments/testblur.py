import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ================= НАСТРОЙКИ =================
base_path = "/Users/adel/Documents/Claude/Projects/Интерполяция фото/final_dataset_v5_participants_small/train"
sample_id = "2025-10-04_15_57_04_16_44_38_luka_1759586458800193000__000"
camera_name = "left_fwd"

sigma_full = 2.0          # сигма для глобального размытия
threshold_diff = 15.0     # порог абсолютной разницы t0/t1 для маски динамики
blur_dynamic_sigma = 3.0  # сигма размытия внутри динамической маски (можно сильнее, чем глобальное)
# ==============================================

# Пути
t0_path = os.path.join(base_path, sample_id, "input", "t0", f"{camera_name}.jpg")
t1_path = os.path.join(base_path, sample_id, "input", "t1", f"{camera_name}.jpg")
gt_path = os.path.join(base_path, sample_id, "target", f"{camera_name}.jpg")

img_t0 = cv2.imread(t0_path)
img_t1 = cv2.imread(t1_path)
img_gt = cv2.imread(gt_path)

if any(x is None for x in [img_t0, img_t1, img_gt]):
    raise FileNotFoundError("Ошибка загрузки изображений")

# Вспомогательные функции
def psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64))**2)
    if mse == 0: return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def score(psnr_val):
    return max(0.0, min(100.0, (psnr_val - 10) / 20.0 * 100))

# ---------- 1. Готовим предсказания ----------
pred_raw = img_t1.copy()

# Глобальное размытие
pred_full_blur = cv2.GaussianBlur(img_t1, (0,0), sigmaX=sigma_full, sigmaY=sigma_full)

# Маска динамики на основе разницы t0 и t1
diff_abs = np.mean(np.abs(img_t0.astype(np.float32) - img_t1.astype(np.float32)), axis=2)
dynamic_mask = (diff_abs > threshold_diff).astype(np.uint8) * 255
# Небольшое расширение маски (морфология), чтобы захватить края
kernel = np.ones((5,5), np.uint8)
dynamic_mask_dilated = cv2.dilate(dynamic_mask, kernel, iterations=1)

# Частичное размытие: копируем t1, затем внутри расширенной маски применяем размытие
pred_partial = img_t1.copy()
# Размываем всё изображение сильно, но наложим только по маске
blurred_strong = cv2.GaussianBlur(img_t1, (0,0), sigmaX=blur_dynamic_sigma, sigmaY=blur_dynamic_sigma)
mask_3ch = cv2.cvtColor(dynamic_mask_dilated, cv2.COLOR_GRAY2BGR) / 255.0
pred_partial = (pred_partial * (1 - mask_3ch) + blurred_strong * mask_3ch).astype(np.uint8)

# ---------- 2. Оценка метрик ----------
psnr_raw = psnr(img_gt, pred_raw)
psnr_full = psnr(img_gt, pred_full_blur)
psnr_partial = psnr(img_gt, pred_partial)

print(f"Raw t1:            PSNR={psnr_raw:.2f} dB, score={score(psnr_raw):.2f}")
print(f"Full blur (σ={sigma_full}):   PSNR={psnr_full:.2f} dB, score={score(psnr_full):.2f}")
print(f"Partial blur (σ={blur_dynamic_sigma}): PSNR={psnr_partial:.2f} dB, score={score(psnr_partial):.2f}")

# ---------- 3. Визуализация ----------
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.ravel()

# Приводим к RGB для отображения
gt_rgb = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
raw_rgb = cv2.cvtColor(pred_raw, cv2.COLOR_BGR2RGB)
full_rgb = cv2.cvtColor(pred_full_blur, cv2.COLOR_BGR2RGB)
partial_rgb = cv2.cvtColor(pred_partial, cv2.COLOR_BGR2RGB)

# Ряд 1: GT, предсказания
axes[0].imshow(gt_rgb)
axes[0].set_title("Ground Truth")
axes[0].axis('off')

axes[1].imshow(raw_rgb)
axes[1].set_title(f"Raw t1\nPSNR={psnr_raw:.1f}")
axes[1].axis('off')

axes[2].imshow(full_rgb)
axes[2].set_title(f"Full blur σ={sigma_full}\nPSNR={psnr_full:.1f}")
axes[2].axis('off')

axes[3].imshow(partial_rgb)
axes[3].set_title(f"Partial blur\nPSNR={psnr_partial:.1f}")
axes[3].axis('off')

# Ряд 2: Абсолютные ошибки (heatmap)
for i, (pred, title) in enumerate([(pred_raw, "Error raw"),
                                     (pred_full_blur, "Error full blur"),
                                     (pred_partial, "Error partial blur")]):
    error = np.mean(np.abs(pred.astype(np.float64) - img_gt.astype(np.float64)), axis=2)
    im = axes[4+i].imshow(error, cmap='hot', vmin=0, vmax=np.percentile(error, 95))
    axes[4+i].set_title(title)
    axes[4+i].axis('off')
    plt.colorbar(im, ax=axes[4+i], fraction=0.046, pad=0.04)

# Ряд 3: Сравнение с raw: где полное/частичное размытие улучшило (синий) или ухудшило (красный)
sq_error_raw = np.sum((pred_raw.astype(np.float64) - img_gt.astype(np.float64))**2, axis=2)
sq_error_full = np.sum((pred_full_blur.astype(np.float64) - img_gt.astype(np.float64))**2, axis=2)
sq_error_partial = np.sum((pred_partial.astype(np.float64) - img_gt.astype(np.float64))**2, axis=2)

delta_full = sq_error_full - sq_error_raw  # <0 => blur helped
delta_partial = sq_error_partial - sq_error_raw

vmax = max(np.percentile(np.abs(delta_full), 95), np.percentile(np.abs(delta_partial), 95))
vmin = -vmax

axes[8].imshow(delta_full, cmap='RdBu_r', vmin=vmin, vmax=vmax)
axes[8].set_title("Δ Error (full blur - raw)\nBlue=improve, Red=degrade")
axes[8].axis('off')

axes[9].imshow(delta_partial, cmap='RdBu_r', vmin=vmin, vmax=vmax)
axes[9].set_title("Δ Error (partial blur - raw)\nBlue=improve, Red=degrade")
axes[9].axis('off')

# Дополнительно: маска динамики
axes[10].imshow(dynamic_mask, cmap='gray')
axes[10].set_title(f"Dynamic mask (diff>{threshold_diff})")
axes[10].axis('off')

# Гистограмма разницы t0/t1 с порогом
axes[11].hist(diff_abs.ravel(), bins=100, range=(0, 100), color='gray', alpha=0.7)
axes[11].axvline(threshold_diff, color='red', linestyle='--')
axes[11].set_xlabel('|t0 - t1|')
axes[11].set_title("Diff histogram & threshold")

plt.tight_layout()
plt.show()