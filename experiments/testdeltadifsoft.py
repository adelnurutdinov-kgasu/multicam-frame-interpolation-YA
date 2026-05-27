import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ========== НАСТРОЙКИ ==========
base_path = "/Users/adel/Documents/Claude/Projects/Интерполяция фото/final_dataset_v5_participants_small/train"
sample_id = "2025-10-04_15_57_04_16_44_38_luka_1759586458800193000__000"
camera_name = "left_fwd"

mode = 'soft'   # 'soft' или 'aggressive'
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
gray0 = cv2.cvtColor(img_t0, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2GRAY)

# ----- Поток Farneback -----
flow = cv2.calcOpticalFlowFarneback(gray1, gray0, None,
                                    pyr_scale=0.5, levels=3, winsize=15,
                                    iterations=3, poly_n=5, poly_sigma=1.2,
                                    flags=0)
flow_x = flow[..., 0]
flow_y = flow[..., 1]
flow_mag = np.sqrt(flow_x**2 + flow_y**2)

# ----- Интерполяция на полпотока -----
y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
map_x_half = x_coords - flow_x / 2.0
map_y_half = y_coords - flow_y / 2.0
pred_flow = cv2.remap(img_t1, map_x_half, map_y_half,
                      interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)

# ----- Ошибка компенсации -----
map_x_full = x_coords - flow_x
map_y_full = y_coords - flow_y
warped_to_t0 = cv2.remap(img_t1, map_x_full, map_y_full,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
comp_err = np.mean(np.abs(img_t0.astype(np.float32) - warped_to_t0.astype(np.float32)), axis=2)

# ----- Параметры размытия в зависимости от режима -----
if mode == 'aggressive':
    base_sigma = 0.5
    alpha = 0.8
    beta = 0.3
    max_sigma = 8.0
    sigma_levels = [0.5, 1.0, 2.0, 4.0, 8.0]
else:  # 'soft'
    base_sigma = 0.3
    alpha = 0.4
    beta = 0.15
    max_sigma = 4.0
    sigma_levels = [0.3, 0.5, 1.0, 2.0, 4.0]

# ----- Карта сигм -----
sigma_map = base_sigma + alpha * flow_mag + beta * comp_err
sigma_map = np.clip(sigma_map, sigma_levels[0], max_sigma)

# ----- Адаптивное размытие (интерполяция уровней) -----
blurred_levels = []
for s in sigma_levels:
    blurred_levels.append(cv2.GaussianBlur(pred_flow, (0, 0), sigmaX=s, sigmaY=s))

pred_adaptive = np.zeros_like(pred_flow, dtype=np.float32)
for i, s_val in enumerate(sigma_levels[:-1]):
    s_next = sigma_levels[i+1]
    mask = (sigma_map >= s_val) & (sigma_map < s_next)
    if not np.any(mask):
        continue
    alpha_mix = (sigma_map[mask] - s_val) / (s_next - s_val)
    alpha_mix = alpha_mix[..., np.newaxis]
    pred_adaptive[mask] = (blurred_levels[i][mask] * (1 - alpha_mix) +
                           blurred_levels[i+1][mask] * alpha_mix)
mask_max = sigma_map >= sigma_levels[-1]
pred_adaptive[mask_max] = blurred_levels[-1][mask_max]
pred_adaptive = pred_adaptive.astype(np.uint8)

# ----- Сравнительные варианты -----
pred_raw = img_t1
pred_global_blur = cv2.GaussianBlur(img_t1, (0,0), sigmaX=2.0, sigmaY=2.0)

# Для дополнительного сравнения: «лёгкий» вариант, если запущен aggressive
if mode == 'aggressive':
    # сделаем soft вручную для сравнения
    soft_sigma_map = 0.3 + 0.4*flow_mag + 0.15*comp_err
    soft_sigma_map = np.clip(soft_sigma_map, 0.3, 4.0)
    soft_levels = [0.3, 0.5, 1.0, 2.0, 4.0]
    soft_blurred = [cv2.GaussianBlur(pred_flow, (0,0), sigmaX=s, sigmaY=s) for s in soft_levels]
    pred_soft = np.zeros_like(pred_flow, dtype=np.float32)
    for i, s_val in enumerate(soft_levels[:-1]):
        s_next = soft_levels[i+1]
        mask = (soft_sigma_map >= s_val) & (soft_sigma_map < s_next)
        if not np.any(mask): continue
        alpha_mix = (soft_sigma_map[mask] - s_val) / (s_next - s_val)
        alpha_mix = alpha_mix[..., np.newaxis]
        pred_soft[mask] = (soft_blurred[i][mask]*(1-alpha_mix) + soft_blurred[i+1][mask]*alpha_mix)
    pred_soft[soft_sigma_map >= soft_levels[-1]] = soft_blurred[-1][soft_sigma_map >= soft_levels[-1]]
    pred_soft = pred_soft.astype(np.uint8)
else:
    pred_soft = pred_adaptive  # если уже soft, дублируем

# ----- Метрики -----
def calc_psnr(a, b):
    mse = np.mean((a.astype(np.float64)-b.astype(np.float64))**2)
    return 20*np.log10(255.0/np.sqrt(mse)) if mse>0 else float('inf')

def score(p):
    return max(0.0, min(100.0, (p-10)/20*100))

print(f"Mode: {mode}")
print(f"Raw t1:            PSNR={calc_psnr(img_gt, pred_raw):.2f} dB, score={score(calc_psnr(img_gt, pred_raw)):.2f}")
print(f"Global blur σ=2.0: PSNR={calc_psnr(img_gt, pred_global_blur):.2f} dB, score={score(calc_psnr(img_gt, pred_global_blur)):.2f}")
print(f"Flow-interpolated: PSNR={calc_psnr(img_gt, pred_flow):.2f} dB, score={score(calc_psnr(img_gt, pred_flow)):.2f}")
print(f"Adaptive blur ({mode}): PSNR={calc_psnr(img_gt, pred_adaptive):.2f} dB, score={score(calc_psnr(img_gt, pred_adaptive)):.2f}")
if mode == 'aggressive':
    print(f"Soft variant:      PSNR={calc_psnr(img_gt, pred_soft):.2f} dB, score={score(calc_psnr(img_gt, pred_soft)):.2f}")

# ----- Визуализация -----
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.ravel()
gt_rgb = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
axes[0].imshow(gt_rgb)
axes[0].set_title("Ground Truth")
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(pred_adaptive, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"Adaptive blur ({mode})\nPSNR={calc_psnr(img_gt, pred_adaptive):.2f}")
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(pred_soft, cv2.COLOR_BGR2RGB))
soft_psnr = calc_psnr(img_gt, pred_soft)
axes[2].set_title(f"Soft variant\nPSNR={soft_psnr:.2f}")
axes[2].axis('off')

# Карта сигм для основного варианта
im_sig = axes[3].imshow(sigma_map, cmap='plasma', vmin=0, vmax=max_sigma)
axes[3].set_title("Sigma map (main)")
axes[3].axis('off')
plt.colorbar(im_sig, ax=axes[3], fraction=0.046, pad=0.04)

# Ошибка адаптивного варианта
err_adapt = np.mean(np.abs(pred_adaptive.astype(np.float64)-img_gt.astype(np.float64)), axis=2)
im_err = axes[4].imshow(err_adapt, cmap='hot', vmin=0, vmax=np.percentile(err_adapt, 95))
axes[4].set_title("Error: adaptive")
axes[4].axis('off')
plt.colorbar(im_err, ax=axes[4], fraction=0.046, pad=0.04)

# Δ ошибки между soft и адаптивным (если aggressive), иначе просто flow mag
if mode == 'aggressive':
    sq_adapt = np.sum((pred_adaptive.astype(np.float64)-img_gt.astype(np.float64))**2, axis=2)
    sq_soft = np.sum((pred_soft.astype(np.float64)-img_gt.astype(np.float64))**2, axis=2)
    delta = sq_soft - sq_adapt
    vmax_d = np.percentile(np.abs(delta), 95)
    axes[5].imshow(delta, cmap='RdBu_r', vmin=-vmax_d, vmax=vmax_d)
    axes[5].set_title("Δ Squared Error (soft - adaptive)\nBlue=soft better")
    axes[5].axis('off')
else:
    axes[5].imshow(flow_mag, cmap='plasma', vmin=0, vmax=np.percentile(flow_mag, 95))
    axes[5].set_title("Flow magnitude")
    axes[5].axis('off')

plt.tight_layout()
plt.show()