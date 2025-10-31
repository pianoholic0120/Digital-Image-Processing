import os
import numpy as np
import cv2
import argparse

def _normalize01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    img -= img.min()
    m = img.max()
    if m > 1e-8:
        img /= m
    return img

def normalize_and_save(path, img_float):
    g_norm = img_float.copy()
    g_norm = g_norm - g_norm.min()
    if g_norm.max() > 1e-8:
        g_norm = g_norm / g_norm.max()
    g_u8 = (g_norm * 255).astype(np.uint8)
    cv2.imwrite(path, g_u8)

def draw_histogram(hist, hist_shape=(300, 256)):
    hist_img = np.zeros(hist_shape, dtype=np.uint8)
    cv2.normalize(hist, hist, 0, hist_shape[0], cv2.NORM_MINMAX)
    hist = hist.ravel()
    
    bin_width = int(np.ceil(hist_shape[1] / 256.0))
    for i in range(256):
        x1 = i * bin_width
        x2 = (i + 1) * bin_width
        y1 = hist_shape[0]
        y2 = hist_shape[0] - int(float(hist[i]))
        cv2.rectangle(hist_img, (x1, y1), (x2, y2), (255), -1)
        
    return hist_img

def _detect_horizontal_peak_rows(F_shift_mag: np.ndarray, center_suppress: int = 24, max_peaks: int = 8, min_sep: int = 8):
    h, w = F_shift_mag.shape
    cy = h // 2
    prof = F_shift_mag.mean(axis=1)
    prof_s = prof.copy()
    lo = max(0, cy - center_suppress)
    hi = min(h, cy + center_suppress + 1)
    prof_s[lo:hi] = prof.min()
    thr = np.percentile(prof_s, 99.0)
    cand = np.where(prof_s >= thr)[0].tolist()
    cand.sort(key=lambda r: prof_s[r], reverse=True)
    picked = []
    for r in cand:
        if len(picked) >= max_peaks:
            break
        if any(abs(r - p) < min_sep for p in picked):
            continue
        picked.append(r)
    return picked

def _build_smooth_row_notch_mask(shape, rows, sigma_rows: float = 2.5):
    h, w = shape
    y = np.arange(h, dtype=np.float32)[:, None]
    M = np.ones((h, w), dtype=np.float32)
    for r in rows:
        g = np.exp(-0.5 * ((y - float(r)) / float(sigma_rows)) ** 2)
        notch = 1.0 - g
        M *= notch
    return np.clip(M, 0.0, 1.0)

def run_part1_logic(img_gray):
    img_f = img_gray.astype(np.float32)
    M, N = img_f.shape
    
    F = np.fft.fft2(img_f)
    F_shift = np.fft.fftshift(F)

    H_shift = np.ones_like(img_f)
    crow, ccol = M // 2, N // 2
    
    notch_width = 4   
    dc_pass_height = 20 
    
    H_shift[:, ccol - notch_width : ccol + notch_width] = 0.0
    
    H_shift[crow - dc_pass_height : crow + dc_pass_height, 
            ccol - notch_width : ccol + notch_width] = 1.0

    G_shift = F_shift * H_shift
    G = np.fft.ifftshift(G_shift)
    g = np.fft.ifft2(G).real
    
    return g, H_shift

def run_part2_logic(img_p1_float):
    img_p1_norm = img_p1_float - img_p1_float.min()
    if img_p1_norm.max() > 1e-8:
        img_p1_norm = img_p1_norm / img_p1_norm.max()
    img_p1_u8 = (img_p1_norm * 255).astype(np.uint8)

    g_blur = cv2.GaussianBlur(img_p1_u8, (7, 7), 0)
    
    mask = cv2.subtract(img_p1_u8, g_blur)
    
    k = 2.0
    g_sharp = cv2.addWeighted(img_p1_u8, 1.0, mask, k, 0)
    
    print("Part 2: Utilizing Unsharp Masking (k=2.0) and 7x7 GaussianBlur.")
    return g_sharp

def run_part3_logic(img_p2_uint8):
    hist_before = cv2.calcHist([img_p2_uint8], [0], None, [256], [0, 256])
    
    img_p3_eq = cv2.equalizeHist(img_p2_uint8)
    
    hist_after = cv2.calcHist([img_p3_eq], [0], None, [256], [0, 256])
    
    hist_img_before = draw_histogram(hist_before)
    hist_img_after = draw_histogram(hist_after)
    
    return img_p3_eq, hist_img_before, hist_img_after

def sub_one(image_path, output_dir):
    print("Executing Subproblem 1...")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image at {image_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    g_p1, H_shift = run_part1_logic(img)
    
    normalize_and_save(os.path.join(output_dir, '1_filter.png'), H_shift)
    normalize_and_save(os.path.join(output_dir, '1_filtered_image.png'), g_p1)
    print("Subproblem 1: Filter and filtered image saved.")

def sub_two(image_path, output_dir):
    print("Executing Subproblem 2...")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image at {image_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    g_p1, _ = run_part1_logic(img)
    
    g_p2 = run_part2_logic(g_p1)
    
    cv2.imwrite(os.path.join(output_dir, '2_sharpened.png'), g_p2)
    print("Subproblem 2: Sharpened image saved.")

def sub_three(image_path, output_dir):
    print("Executing Subproblem 3...")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image at {image_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    g_p1, _ = run_part1_logic(img)
    g_p2 = run_part2_logic(g_p1)
    
    g_p3, hist_b, hist_a = run_part3_logic(g_p2)
    
    cv2.imwrite(os.path.join(output_dir, '3_equalized.png'), g_p3)
    cv2.imwrite(os.path.join(output_dir, '3_hist_before.png'), hist_b)
    cv2.imwrite(os.path.join(output_dir, '3_hist_after.png'), hist_a)
    print("Subproblem 3: Equalized image and histograms saved.")

def sub_four(image_path, output_dir):
    print("Executing Subproblem 4...")
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image at {image_path}")
    os.makedirs(output_dir, exist_ok=True)
    # 1) Adaptive frequency notch (smooth Gaussian row band-stops)
    img_f = img.astype(np.float32)
    F = np.fft.fft2(img_f)
    F_shift = np.fft.fftshift(F)
    mag = np.log1p(np.abs(F_shift))
    rows = _detect_horizontal_peak_rows(mag, center_suppress=24, max_peaks=10, min_sep=8)
    M = _build_smooth_row_notch_mask(img.shape, rows, sigma_rows=2.5)
    # soften the notch to avoid over-suppression (preserve structure)
    M_soft = 0.6 + 0.4 * M  # in [0.6,1]
    G_shift = F_shift * M_soft
    # normalize_and_save(os.path.join(output_dir, '4_mask.png'), M_soft)
    notch = np.fft.ifftshift(G_shift)
    notch_img = np.fft.ifft2(notch).real
    # normalize_and_save(os.path.join(output_dir, '4_notched.png'), notch_img)

    # 2) Row detrend (time-domain vertical smoothing subtraction)
    notch_u8 = (_normalize01(notch_img) * 255.0).astype(np.uint8)
    trend = cv2.blur(notch_u8, (1, 11))
    # partial subtraction to avoid edge-only result
    detrended = cv2.addWeighted(notch_u8, 1.0, trend, -0.3, 0)
    # cv2.imwrite(os.path.join(output_dir, '4_row_detrend.png'), detrended)

    # 3) Denoise (NLM then light bilateral)
    den = cv2.fastNlMeansDenoising(detrended, None, h=6, templateWindowSize=7, searchWindowSize=21)
    den = cv2.bilateralFilter(den, d=5, sigmaColor=20, sigmaSpace=7)
    # cv2.imwrite(os.path.join(output_dir, '4_denoised.png'), den)

    # 4) Multi-scale unsharp masking
    den_f = den.astype(np.float32)
    blur_s = cv2.GaussianBlur(den_f, (0, 0), 0.8)
    blur_l = cv2.GaussianBlur(den_f, (0, 0), 2.0)
    detail_s = den_f - blur_s
    detail_l = den_f - blur_l
    alpha_s, alpha_l = 0.8, 0.3
    enhance = den_f + alpha_s * detail_s + alpha_l * detail_l
    enhance_n = _normalize01(enhance)
    # cv2.imwrite(os.path.join(output_dir, '4_detail.png'), (enhance_n * 255).astype(np.uint8))

    # 5) CLAHE + mild gamma
    # blend sharpened with denoised to avoid cartooning
    sharp_u8 = (enhance_n * 255).astype(np.uint8)
    base_blend = cv2.addWeighted(den, 0.6, sharp_u8, 0.4, 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cla = clahe.apply(base_blend)
    gamma = 1.0  # keep tone neutral
    final = cla  # already uint8
    cv2.imwrite(os.path.join(output_dir, '4_my_procedure.png'), final)
    print("Subproblem 4: Saved final image.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./images/noisy_image.tif')
    parser.add_argument('--output', type=str, default='./output_b/')
    parser.add_argument('--subproblem', type=int, required=True, choices=[1, 2, 3, 4])
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input image not found at {args.input}")
        print("Please make sure 'noisy_image.tif' is in the 'images' directory or specify the correct path with --input.")
        return
        
    os.makedirs(args.output, exist_ok=True)
    
    if args.subproblem == 1:
        sub_one(args.input, args.output)
    elif args.subproblem == 2:
        sub_two(args.input, args.output)
    elif args.subproblem == 3:
        sub_three(args.input, args.output)
    elif args.subproblem == 4:
        sub_four(args.input, args.output)
    else:
        print("Invalid subproblem")

if __name__ == '__main__':
    main()