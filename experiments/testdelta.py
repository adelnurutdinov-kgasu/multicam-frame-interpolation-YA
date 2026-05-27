import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ========== НАСТРОЙКИ ==========
base_path = "/Users/adel/Documents/Claude/Projects/Интерполяция фото/final_dataset_v5_participants_small/train"
sample_id = "2025-10-04_15_57_04_16_44_38_luka_1759586458800193000__000"
camera_name = "left_fwd"

# Параметры адаптивного размытия
base_sigma = 0.1          # минимальное размытие для статики
alpha = 0.4               # коэффициент усиления от длины потока (пиксель -> сигма)
beta = 0.2                # коэффициент усиления от ошибки совмещения
max_sigma = 2.0           # ограничение сверху

# Уровни фиксированных размытий для интерполяции
sigma_levels = [0.1, 0.2, 0.3, 1.0, 2.0]
# ================================

t0_path = os.path.join(base_path, sample_id, "input", "t0", f"{camera_name}.jpg")
t1_path = os.path.join(base_path, sample_id, "input", "t1", f"{camera_name}.jpg")
gt_path = os.path.join(base_path, sample_id, "target", f"{camera_name}.jpg")

img_t0 = cv2.imread(t0_path)
img_t1 = cv2.imread(t1_path)
img_gt = cv2.imread(gt_path)

if any(x is None for x in [img_t0, img_t1, img_gt]):
    raise FileNotFoundError("Ошибка загрузки изображений")

h, w = img_t0.shape[:2]

# Grayscale для потока
gray0 = cv2.cvtColor(img_t0, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2GRAY)

# ----- 1. Оптический поток t1 -> t0 -----
flow = cv2.calcOpticalFlowFarneback(gray1, gray0, None,
                                    pyr_scale=0.5, levels=3, winsize=15,
                                    iterations=3, poly_n=5, poly_sigma=1.2,
                                    flags=0)
flow_x = flow[..., 0]
flow_y = flow[..., 1]
flow_mag = np.sqrt(flow_x**2 + flow_y**2)

# ----- 2. Интерполяция на полпотока (pred_flow) -----
y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
map_x_half = x_coords - flow_x / 2.0
map_y_half = y_coords - flow_y / 2.0
pred_flow = cv2.remap(img_t1, map_x_half, map_y_half,
                      interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)

# ----- 3. Оценка ошибки совмещения (compensation error) -----
# Перемещаем t1 на ПОЛНЫЙ поток к t0
map_x_full = x_coords - flow_x
map_y_full = y_coords - flow_y
warped_t1_to_t0 = cv2.remap(img_t1, map_x_full, map_y_full,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
# Абсолютная разница между настоящим t0 и восстановленным из t1
compensation_error = np.mean(np.abs(img_t0.astype(np.float32) -
                                   warped_t1_to_t0.astype(np.float32)), axis=2)

# ----- 4. Карта сигм -----
sigma_map = base_sigma + alpha * flow_mag + beta * compensation_error
sigma_map = np.clip(sigma_map, sigma_levels[0], max_sigma)

# ----- 5. Адаптивное размытие через уровни -----
# Готовим набор размытых копий исходного pred_flow (или img_t1? Лучше pred_flow, т.к. он уже интерполирован)
blurred_levels = []
for s in sigma_levels:
    if s == 0.5 and base_sigma < 0.5: s = base_sigma  # чтобы minimum был base_sigma
    blurred = cv2.GaussianBlur(pred_flow, (0, 0), sigmaX=s, sigmaY=s)
    blurred_levels.append(blurred)

# Для каждого пикселя находим, к каким уровням он ближе, и линейно интерполируем
# Простой метод: для каждого пикселя выбираем два ближайших уровня и смешиваем
pred_adaptive = np.zeros_like(pred_flow, dtype=np.float32)
for i, s_val in enumerate(sigma_levels[:-1]):
    s_next = sigma_levels[i+1]
    # Маска пикселей, sigma_map между s_val и s_next
    mask = (sigma_map >= s_val) & (sigma_map < s_next)
    if not np.any(mask):
        continue
    # Вес: 0 -> s_val, 1 -> s_next
    alpha_mix = (sigma_map[mask] - s_val) / (s_next - s_val)
    alpha_mix = alpha_mix[..., np.newaxis]  # для broadcasting по каналам
    # Смешиваем два уровня
    pred_adaptive[mask] = (blurred_levels[i][mask] * (1 - alpha_mix) +
                           blurred_levels[i+1][mask] * alpha_mix)
# Для пикселей с sigma_map >= max_sigma используем последний уровень
mask_max = sigma_map >= sigma_levels[-1]
pred_adaptive[mask_max] = blurred_levels[-1][mask_max]

pred_adaptive = pred_adaptive.astype(np.uint8)

# Дополнительные варианты для сравнения
pred_raw = img_t1
pred_global_blur = cv2.GaussianBlur(img_t1, (0, 0), sigmaX=2.0, sigmaY=2.0)

# ----- 6. Метрики -----
def calc_psnr(a, b):
    mse = np.mean((a.astype(np.float64)-b.astype(np.float64))**2)
    return 20*np.log10(255.0/np.sqrt(mse)) if mse>0 else float('inf')

def score(psnr):
    return max(0, min(100, (psnr-10)/20*100))

psnr_raw = calc_psnr(img_gt, pred_raw)
psnr_blur = calc_psnr(img_gt, pred_global_blur)
psnr_flow = calc_psnr(img_gt, pred_flow)
psnr_adapt = calc_psnr(img_gt, pred_adaptive)

print(f"Raw t1:            PSNR={psnr_raw:.2f} dB, score={score(psnr_raw):.2f}")
print(f"Global blur σ=2.0: PSNR={psnr_blur:.2f} dB, score={score(psnr_blur):.2f}")
print(f"Flow-interpolated: PSNR={psnr_flow:.2f} dB, score={score(psnr_flow):.2f}")
print(f"Adaptive blur:     PSNR={psnr_adapt:.2f} dB, score={score(psnr_adapt):.2f}")

# ----- 7. Визуализация -----
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
axes = axes.ravel()

gt_rgb = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
axes[0].imshow(gt_rgb)
axes[0].set_title("Ground Truth")
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(pred_raw, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"Raw t1\nPSNR={psnr_raw:.1f}")
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(pred_global_blur, cv2.COLOR_BGR2RGB))
axes[2].set_title(f"Global blur σ=2.0\nPSNR={psnr_blur:.1f}")
axes[2].axis('off')

axes[3].imshow(cv2.cvtColor(pred_adaptive, cv2.COLOR_BGR2RGB))
axes[3].set_title(f"Adaptive blur\nPSNR={psnr_adapt:.1f}")
axes[3].axis('off')

# Карта sigma
im_sig = axes[4].imshow(sigma_map, cmap='plasma', vmin=0, vmax=max_sigma)
axes[4].set_title("Sigma map (blur strength)")
axes[4].axis('off')
plt.colorbar(im_sig, ax=axes[4], fraction=0.046, pad=0.04)

# Карта ошибки совмещения
im_err = axes[5].imshow(compensation_error, cmap='hot', vmin=0, vmax=np.percentile(compensation_error, 95))
axes[5].set_title("Compensation error\n(|t0 - warped_t1|)")
axes[5].axis('off')
plt.colorbar(im_err, ax=axes[5], fraction=0.046, pad=0.04)

# Flow magnitude
im_flow = axes[6].imshow(flow_mag, cmap='plasma', vmin=0, vmax=np.percentile(flow_mag, 95))
axes[6].set_title("Flow magnitude")
axes[6].axis('off')
plt.colorbar(im_flow, ax=axes[6], fraction=0.046, pad=0.04)

# Δ squared error (adaptive - raw)
sq_raw = np.sum((pred_raw.astype(np.float64)-img_gt.astype(np.float64))**2, axis=2)
sq_adapt = np.sum((pred_adaptive.astype(np.float64)-img_gt.astype(np.float64))**2, axis=2)
delta = sq_adapt - sq_raw
vmax_delta = np.percentile(np.abs(delta), 95)
axes[7].imshow(delta, cmap='RdBu_r', vmin=-vmax_delta, vmax=vmax_delta)
axes[7].set_title("Δ Error (adaptive - raw)\nBlue=improvement, Red=degradation")
axes[7].axis('off')

plt.tight_layout()
plt.show()