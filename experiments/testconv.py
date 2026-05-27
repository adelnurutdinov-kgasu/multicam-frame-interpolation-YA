import cv2
import numpy as np
import os

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

print("Исходный PSNR (без размытия): {:.2f} dB, score: {:.2f}".format(
    calculate_psnr(target, pred_raw), score_from_psnr(calculate_psnr(target, pred_raw))))

sigmas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
for s in sigmas:
    # Гауссово размытие (окно вычисляется автоматически, можно задать (0,0))
    blurred = cv2.GaussianBlur(pred_raw, (0, 0), sigmaX=s, sigmaY=s)
    psnr_val = calculate_psnr(target, blurred)
    score_val = score_from_psnr(psnr_val)
    print(f"σ={s:.1f}  PSNR: {psnr_val:.2f} dB  score: {score_val:.2f}")