import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ================= НАСТРОЙКИ =================
base_path = "/Users/adel/Documents/Claude/Projects/Интерполяция фото/final_dataset_v5_participants_small/train"
sample_id = "2025-10-04_15_57_04_16_44_38_luka_1759586458800193000__000"
camera_name = "left_fwd"

# Пороги для агрессивной стратегии
flow_thresh = 5.0       # пиксели
comp_error_thresh = 20   # яркость (0..255)
blur_sigma_fallback = 4.0
# =============================================

t0_path = os.path.join(base_path, sample_id, "input", "t0", f"{camera_name}.jpg")
t1_path = os.path.join(base_path, sample_id, "input", "t1", f"{camera_name}.jpg")
gt_path = os.path.join(base_path, sample_id, "target", f"{camera_name}.jpg")

img_t0 = cv2.imread(t0_path)
img_t1 = cv2.imread(t1_path)
img_gt = cv2.imread(gt_path)

if any(x is None for x in [img_t0, img_t1, img_gt]):
    raise FileNotFoundError("Ошибка загрузки изображений")

h, w = img_t0.shape[:2]

# Grayscale для методов потока
gray0 = cv2.cvtColor(img_t0, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2GRAY)

# ---------- Вспомогательные функции ----------
def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')

def score(p):
    return max(0.0, min(100.0, (p - 10) / 20.0 * 100))

# Функция warp'а по потоку (смещение на -flow/2)
def warp_half(img, flow):
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = x - flow_x / 2.0
    map_y = y - flow_y / 2.0
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)

# Оценка ошибки компенсации (warp t1 на полный поток к t0)
def compensation_error(flow):
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x_full = x - flow_x
    map_y_full = y - flow_y
    warped = cv2.remap(img_t1, map_x_full, map_y_full,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    return np.mean(np.abs(img_t0.astype(np.float32) - warped.astype(np.float32)), axis=2)

# ================== Метод 1: Farneback (классический) ==================
flow_farn = cv2.calcOpticalFlowFarneback(
    gray1, gray0, None, pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
pred_farn = warp_half(img_t1, flow_farn)

# ================== Метод 2: Farneback + агрессивное подавление ==================
flow_mag = np.sqrt(flow_farn[..., 0]**2 + flow_farn[..., 1]**2)
comp_err_farn = compensation_error(flow_farn)
bad_mask = (flow_mag > flow_thresh) | (comp_err_farn > comp_error_thresh)
pred_farn_agg = pred_farn.copy()
blurred_t1_strong = cv2.GaussianBlur(img_t1, (0, 0), sigmaX=blur_sigma_fallback, sigmaY=blur_sigma_fallback)
pred_farn_agg[bad_mask] = blurred_t1_strong[bad_mask]

# ================== Метод 3: DIS (Dense Inverse Search) ==================
try:
    # Проверим, доступен ли DIS (требует opencv-contrib)
    dis = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow_dis = dis.calc(gray1, gray0, None)
    pred_dis = warp_half(img_t1, flow_dis)
except AttributeError:
    print("DIS optical flow не найден в вашей сборке OpenCV. Будет использован Farneback.")
    flow_dis = flow_farn
    pred_dis = pred_farn

# ================== Метод 4: Фазовая корреляция по блокам ==================
block_size = 32
pred_phase = np.zeros_like(img_t1, dtype=np.float64)  # float64 для точности
count_map = np.zeros((h, w), dtype=np.float64)

for y0 in range(0, h - block_size + 1, block_size):
    for x0 in range(0, w - block_size + 1, block_size):
        block0 = gray0[y0:y0+block_size, x0:x0+block_size].astype(np.float32)
        block1 = gray1[y0:y0+block_size, x0:x0+block_size].astype(np.float32)
        try:
            shift, _ = cv2.phaseCorrelate(block0, block1)
        except:
            shift = (0.0, 0.0)
        dx, dy = -shift[0]/2.0, -shift[1]/2.0
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        patch_warped = cv2.warpAffine(img_t1[y0:y0+block_size, x0:x0+block_size],
                                      M, (block_size, block_size),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
        pred_phase[y0:y0+block_size, x0:x0+block_size] += patch_warped.astype(np.float64)
        count_map[y0:y0+block_size, x0:x0+block_size] += 1.0

# Усреднение с учётом трёх каналов
count_map_3ch = np.repeat(count_map[:, :, np.newaxis], 3, axis=2)
pred_phase = (pred_phase / count_map_3ch).astype(np.uint8)

# ================== Базовые варианты ==================
pred_raw = img_t1
pred_blur_global = cv2.GaussianBlur(img_t1, (0, 0), sigmaX=2.0, sigmaY=2.0)

# ================== Вычисление PSNR и score ==================
methods = {
    "Raw t1": pred_raw,
    "Global blur (2.0)": pred_blur_global,
    "Farneback half": pred_farn,
    "Farneback aggressive": pred_farn_agg,
    "DIS half": pred_dis,
    "Phase correlation blocks": pred_phase,
}

print("Метод                        PSNR (dB)   Score")
print("-" * 50)
for name, pred in methods.items():
    p = psnr(img_gt, pred)
    s = score(p)
    print(f"{name:30s} {p:8.2f}    {s:5.2f}")

# ================== Визуализация ==================
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(3, 4, figure=fig)

# Первая строка: Ground Truth и лучшие предсказания
ax_gt = fig.add_subplot(gs[0, 0])
ax_gt.imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
ax_gt.set_title("Ground Truth")
ax_gt.axis('off')

# Выберем три лучших по PSNR для показа (можно просто показать все)
sorted_methods = sorted(methods.items(), key=lambda x: psnr(img_gt, x[1]), reverse=True)
for i, (name, pred) in enumerate(sorted_methods[:3]):
    ax = fig.add_subplot(gs[0, i+1])
    ax.imshow(cv2.cvtColor(pred, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{name}\nPSNR={psnr(img_gt, pred):.2f}")
    ax.axis('off')

# Вторая строка: карты ошибок для трёх лучших
for i, (name, pred) in enumerate(sorted_methods[:3]):
    ax = fig.add_subplot(gs[1, i])
    err = np.mean(np.abs(pred.astype(np.float64) - img_gt.astype(np.float64)), axis=2)
    im = ax.imshow(err, cmap='hot', vmin=0, vmax=np.percentile(err, 95))
    ax.set_title(f"Error: {name}")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Третья строка: карты смещения и маски
ax_flow = fig.add_subplot(gs[2, 0])
ax_flow.imshow(flow_mag, cmap='plasma', vmin=0, vmax=np.percentile(flow_mag, 95))
ax_flow.set_title("Flow magnitude (Farneback)")
ax_flow.axis('off')
plt.colorbar(ax_flow.imshow(flow_mag, cmap='plasma', vmin=0, vmax=np.percentile(flow_mag, 95)), ax=ax_flow, fraction=0.046, pad=0.04)

ax_bad = fig.add_subplot(gs[2, 1])
ax_bad.imshow(bad_mask, cmap='gray')
ax_bad.set_title(f"Aggressive mask (flow>{flow_thresh} or err>{comp_error_thresh})")
ax_bad.axis('off')

# Сравнение ошибок: Farneback aggressive vs raw
ax_delta = fig.add_subplot(gs[2, 2])
sq_raw = np.sum((pred_raw.astype(np.float64)-img_gt.astype(np.float64))**2, axis=2)
sq_agg = np.sum((pred_farn_agg.astype(np.float64)-img_gt.astype(np.float64))**2, axis=2)
delta = sq_agg - sq_raw
vmax = np.percentile(np.abs(delta), 95)
im_delta = ax_delta.imshow(delta, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
ax_delta.set_title("Δ Error (aggr. - raw)\nBlue=better")
ax_delta.axis('off')
plt.colorbar(im_delta, ax=ax_delta, fraction=0.046, pad=0.04)

ax_phase = fig.add_subplot(gs[2, 3])
err_phase = np.mean(np.abs(pred_phase.astype(np.float64)-img_gt.astype(np.float64)), axis=2)
im_phase = ax_phase.imshow(err_phase, cmap='hot', vmin=0, vmax=np.percentile(err_phase, 95))
ax_phase.set_title("Error: Phase corr.")
ax_phase.axis('off')
plt.colorbar(im_phase, ax=ax_phase, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()