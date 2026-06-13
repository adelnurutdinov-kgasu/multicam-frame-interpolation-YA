import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ======= НАСТРОЙКИ =======
base_path = "/Users/adel/Documents/Claude/Projects/Интерполяция фото/final_dataset_v5_participants_small/train"
sample_id = "2025-10-04_15_57_04_16_44_38_luka_1759586458800193000__000"
camera_name = "left_fwd"  # можно менять на любую из 6 камер
# ==========================

# Пути к кадрам
t0_path = os.path.join(base_path, sample_id, "input", "t0", f"{camera_name}.jpg")
t1_path = os.path.join(base_path, sample_id, "input", "t1", f"{camera_name}.jpg")

img_t0 = cv2.imread(t0_path)
img_t1 = cv2.imread(t1_path)

if img_t0 is None or img_t1 is None:
    raise FileNotFoundError("Не удалось загрузить t0 или t1")

# Переводим BGR -> RGB для визуализации
t0_rgb = cv2.cvtColor(img_t0, cv2.COLOR_BGR2RGB)
t1_rgb = cv2.cvtColor(img_t1, cv2.COLOR_BGR2RGB)

# Абсолютная разница (усреднённая по каналам) - основная мера "изменения"
diff_abs = np.mean(np.abs(img_t0.astype(np.float32) - img_t1.astype(np.float32)), axis=2)

# Квадрат разницы (чувствителен к сильным выбросам)
diff_sq = np.mean((img_t0.astype(np.float32) - img_t1.astype(np.float32))**2, axis=2)

# Для сравнения: MSE между кадрами (если захотим оценить общую похожесть)
mse_full = np.mean((img_t0.astype(np.float32) - img_t1.astype(np.float32))**2)
psnr_full = 20 * np.log10(255.0 / np.sqrt(mse_full)) if mse_full > 0 else float('inf')

print(f"Общий PSNR между t0 и t1: {psnr_full:.2f} dB")

# ---------- Визуализация ----------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# 1. t0
axes[0,0].imshow(t0_rgb)
axes[0,0].set_title("t0")
axes[0,0].axis('off')

# 2. t1
axes[0,1].imshow(t1_rgb)
axes[0,1].set_title("t1")
axes[0,1].axis('off')

# 3. Абсолютная разница (heatmap)
im3 = axes[0,2].imshow(diff_abs, cmap='inferno', vmin=0, vmax=np.percentile(diff_abs, 95))
axes[0,2].set_title("Absolute Difference (|t0 - t1|)")
axes[0,2].axis('off')
plt.colorbar(im3, ax=axes[0,2], fraction=0.046, pad=0.04)

# 4. Разница, наложенная на t0 (полупрозрачная)
axes[1,0].imshow(t0_rgb)
im4 = axes[1,0].imshow(diff_abs, cmap='jet', alpha=0.6, vmin=0, vmax=np.percentile(diff_abs, 95))
axes[1,0].set_title("Overlay on t0")
axes[1,0].axis('off')
plt.colorbar(im4, ax=axes[1,0], fraction=0.046, pad=0.04)

# 5. Бинарная маска "динамики" по порогу (можно подобрать порог)
threshold = 15  # уровень яркости 0..255, эмпирически
dynamic_mask = diff_abs > threshold
axes[1,1].imshow(dynamic_mask, cmap='gray')
axes[1,1].set_title(f"Dynamic mask (thresh={threshold})")
axes[1,1].axis('off')

# 6. Гистограмма абсолютной разницы (помогает выбрать порог)
axes[1,2].hist(diff_abs.ravel(), bins=100, range=(0, 100), color='blue', alpha=0.7)
axes[1,2].axvline(threshold, color='red', linestyle='--', label=f'threshold={threshold}')
axes[1,2].set_xlabel('Absolute difference')
axes[1,2].set_ylabel('Pixel count')
axes[1,2].legend()
axes[1,2].set_title("Histogram of differences")

plt.tight_layout()
plt.show()