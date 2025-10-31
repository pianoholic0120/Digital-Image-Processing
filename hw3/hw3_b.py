import os
import numpy as np
import cv2  
import argparse
import sys

def normalize_and_save(path, img_float):
    g_norm = img_float.copy()
    g_min = g_norm.min()
    g_max = g_norm.max()
    
    g_norm = g_norm - g_min
    if (g_max - g_min) > 1e-8:
        g_norm = g_norm / (g_max - g_min)
    
    g_u8 = (g_norm * 255).astype(np.uint8)
    cv2.imwrite(path, g_u8)

def normalize_img(img):
    img_norm = img - img.min()
    if img_norm.max() > 1e-8:
        img_norm = img_norm / img_norm.max()
    return (img_norm * 255).astype(np.uint8)

def custom_normalize(data, out_min, out_max):
    data_norm = data.copy().astype(np.float32)
    data_min = data_norm.min()
    data_max = data_norm.max()
    
    if (data_max - data_min) > 1e-8:
        data_norm = (data_norm - data_min) / (data_max - data_min) * (out_max - out_min) + out_min
    else:
        data_norm = np.full_like(data_norm, out_min)
        
    return data_norm

def custom_calc_hist(img):
    hist, _ = np.histogram(img.ravel(), bins=np.arange(257))
    return hist.astype(np.float32)

def custom_gaussian_blur(img, ksize_tuple, sigma):
    img_f = img.astype(np.float32)
    M, N = img_f.shape
    
    k_h, k_w = ksize_tuple
    if sigma <= 0:
        if k_h == 0 and k_w == 0:
            raise ValueError("ksize=0 and sigma=0 is invalid combo")
        sigma_h = 0.3 * ((k_h - 1) * 0.5 - 1) + 0.8 if k_h > 0 else 0
        sigma_w = 0.3 * ((k_w - 1) * 0.5 - 1) + 0.8 if k_w > 0 else 0
        sigma = max(sigma_h, sigma_w)
        if sigma == 0: sigma = 1e-8

    if k_h == 0 and k_w == 0:
        k_h = int(np.ceil(sigma * 6)) | 1
        k_w = int(np.ceil(sigma * 6)) | 1

    pad_h, pad_w = k_h // 2, k_w // 2
    y, x = np.meshgrid(np.arange(-pad_h, pad_h + 1), np.arange(-pad_w, pad_w + 1), indexing='ij')
    
    exp_term = -(x**2 + y**2) / (2 * sigma**2)
    kernel = np.exp(exp_term)
    kernel = kernel / kernel.sum()

    kernel_padded = np.zeros((M, N), dtype=np.float32)
    kernel_padded[:k_h, :k_w] = kernel
    kernel_padded = np.roll(kernel_padded, -pad_h, axis=0)
    kernel_padded = np.roll(kernel_padded, -pad_w, axis=1)

    F_img = np.fft.fft2(img_f)
    F_kernel = np.fft.fft2(kernel_padded)
    
    F_blur = F_img * F_kernel
    
    blur_img = np.fft.ifft2(F_blur).real
    
    return blur_img

def custom_equalize_hist(img):
    hist = custom_calc_hist(img)
    cdf = hist.cumsum()
    
    cdf_m = np.ma.masked_equal(cdf, 0)
    if cdf_m.mask.all(): 
        return img.copy()
        
    cdf_min = cdf_m.min()
    num_pixels = cdf.max() 
    
    if num_pixels == cdf_min:
        return img.copy()
        
    cdf_norm = (cdf - cdf_min) * 255 / (num_pixels - cdf_min)
    cdf_lookup = cdf_norm.astype(np.uint8)
    
    img_eq = cdf_lookup[img]
    
    return img_eq.astype(np.uint8)

def custom_clahe(img, clip_limit=2.5, tile_grid_size=(8, 8)):
    if img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError("Input image must be a 2D uint8 grayscale image.")

    M, N = img.shape
    num_bins = 256

    tiles_y, tiles_x = tile_grid_size
    
    tile_h = M // tiles_y
    tile_w = N // tiles_x
    
    transform_functions = np.zeros((tiles_y, tiles_x, num_bins), dtype=np.uint8)

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y_start = ty * tile_h
            y_end = (ty + 1) * tile_h if ty < tiles_y - 1 else M
            x_start = tx * tile_w
            x_end = (tx + 1) * tile_w if tx < tiles_x - 1 else N

            tile = img[y_start:y_end, x_start:x_end]
            
            if tile.size == 0:
                continue

            hist = custom_calc_hist(tile)

            max_hist_height = (tile.size / num_bins) * clip_limit
            
            if max_hist_height < 1.0: 
                redistribute_amount = 0
            else:
                redistribute_amount = (hist[hist > max_hist_height] - max_hist_height).sum()
                hist[hist > max_hist_height] = max_hist_height 
            
            if redistribute_amount > 1e-6: 
                avg_redistribute_per_bin = redistribute_amount / num_bins
                hist += avg_redistribute_per_bin

            cdf = hist.cumsum()
            
            cdf_m = np.ma.masked_equal(cdf, 0)
            if cdf_m.mask.all():
                transform_functions[ty, tx, :] = np.arange(num_bins).astype(np.uint8)
                continue
            
            cdf_min = cdf_m.min()
            num_pixels_in_tile = cdf.max()

            if num_pixels_in_tile == cdf_min:
                 transform_functions[ty, tx, :] = np.arange(num_bins).astype(np.uint8)
                 continue

            cdf_norm = (cdf - cdf_min) * 255 / (num_pixels_in_tile - cdf_min)
            transform_functions[ty, tx, :] = cdf_norm.astype(np.uint8)

    output_img = np.zeros_like(img, dtype=np.uint8)
    
    for r in range(M):
        for c in range(N):
            ty_base = min(int(r / tile_h), tiles_y - 1)
            tx_base = min(int(c / tile_w), tiles_x - 1)
            
            r_f = (r + 0.5) / tile_h
            c_f = (c + 0.5) / tile_w
            
            ty1_f = max(0, ty_base)
            tx1_f = max(0, tx_base)
            
            ty2_f = min(tiles_y - 1, ty_base + 1)
            tx2_f = min(tiles_x - 1, tx_base + 1)

            
            interp_y_idx = r / tile_h
            interp_x_idx = c / tile_w
            
            y_low = int(np.floor(interp_y_idx))
            y_high = int(np.ceil(interp_y_idx))
            x_low = int(np.floor(interp_x_idx))
            x_high = int(np.ceil(interp_x_idx))
            
            y_low = np.clip(y_low, 0, tiles_y - 1)
            y_high = np.clip(y_high, 0, tiles_y - 1)
            x_low = np.clip(x_low, 0, tiles_x - 1)
            x_high = np.clip(x_high, 0, tiles_x - 1)

            if y_high == y_low: fy = 0.5
            else: fy = (interp_y_idx - y_low) / (y_high - y_low)
            
            if x_high == x_low: fx = 0.5
            else: fx = (interp_x_idx - x_low) / (x_high - x_low)
            
            tf_tl = transform_functions[y_low, x_low, img[r, c]]      # Top-Left
            tf_tr = transform_functions[y_low, x_high, img[r, c]]     # Top-Right
            tf_bl = transform_functions[y_high, x_low, img[r, c]]     # Bottom-Left
            tf_br = transform_functions[y_high, x_high, img[r, c]]    # Bottom-Right

            interp_top = tf_tl * (1 - fx) + tf_tr * fx
            interp_bottom = tf_bl * (1 - fx) + tf_br * fx
            
            final_pixel_val = interp_top * (1 - fy) + interp_bottom * fy
            
            output_img[r, c] = np.clip(final_pixel_val, 0, 255).astype(np.uint8)

    return output_img

def draw_histogram(hist, hist_shape=(300, 256)):
    hist_img = np.zeros(hist_shape, dtype=np.uint8)
    
    hist_norm = custom_normalize(hist, 0, hist_shape[0])
    hist_norm = hist_norm.ravel()
    
    bin_width = int(np.ceil(hist_shape[1] / 256.0))
    for i in range(256):
        x1 = i * bin_width
        x2 = (i + 1) * bin_width
        y1 = hist_shape[0]
        y2 = hist_shape[0] - int(float(hist_norm[i]))
        
        cv2.rectangle(hist_img, (x1, y1), (x2, y2), (255), -1)
        
    return hist_img

def single_scale_retinex(img, sigma=30):
    img = img.astype(np.float32) + 1.0  
    
    blur = custom_gaussian_blur(img, (0, 0), sigma)
    
    blur[blur < 1e-8] = 1e-8
    img[img < 1e-8] = 1e-8

    retinex = np.log(img) - np.log(blur)
    return retinex

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

    img_f = img_p1_u8.astype(np.float32)
    g_blur_f = custom_gaussian_blur(img_f, (7, 7), 0)
    
    mask_f = img_f - g_blur_f
    mask_f[mask_f < 0] = 0
    
    k = 2.0
    sharp_f = img_f * 1.0 + mask_f * k
    
    g_sharp = np.clip(sharp_f, 0, 255).astype(np.uint8)
    
    print("Part 2: Utilizing *Custom* Unsharp Masking (k=2.0) and *Custom* 7x7 GaussianBlur.")
    return g_sharp

def run_part3_logic(img_p2_uint8):
    hist_before = custom_calc_hist(img_p2_uint8)
    
    img_p3_eq = custom_equalize_hist(img_p2_uint8)
    
    hist_after = custom_calc_hist(img_p3_eq)
    
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

    # Step 1: Notch Removal 
    g_p1, _ = run_part1_logic(img)
    g_p1_norm = g_p1 - g_p1.min()
    if g_p1_norm.max() > 1e-8:
        g_p1_norm = g_p1_norm / g_p1_norm.max()
    g_p1_u8 = (g_p1_norm * 255).astype(np.uint8)

    # Step 2: Unsharp Mask 
    img_f = g_p1_u8.astype(np.float32)
    g_blur_f = custom_gaussian_blur(img_f, (7, 7), 0)
    mask_f = img_f - g_blur_f
    mask_f[mask_f < 0] = 0
    k = 2.0
    sharp_f = img_f + mask_f * k
    g_sharp = np.clip(sharp_f, 0, 255).astype(np.uint8)

    # Step 3: CLAHE 
    print("Subproblem 4: Using *Custom* CLAHE for adaptive histogram equalization.")
    g_clahe = custom_clahe(g_sharp, clip_limit=2.5, tile_grid_size=(8,8))

    # Step 4: Single-Scale Retinex 
    g_retinex = single_scale_retinex(g_clahe, sigma=30)
    g_retinex_norm = g_retinex - np.percentile(g_retinex, 1)
    g_retinex_norm = np.clip(g_retinex_norm, 0, None)
    if g_retinex_norm.max() > 1e-8:
        g_retinex_norm = g_retinex_norm / g_retinex_norm.max()
    retinex_img = (g_retinex_norm * 255).astype(np.uint8)

    # Step 5: Histogram Equalization 
    hist_eq = custom_equalize_hist(retinex_img)
    cv2.imwrite(os.path.join(output_dir, '4_hist_eq.png'), hist_eq)

    print("Subproblem 4: Applying *Custom* CLAHE again as per original logic.")
    final_clahe = custom_clahe(hist_eq, clip_limit=2.5, tile_grid_size=(16,16))

    cv2.imwrite(os.path.join(output_dir, '4_my_procedure.png'), final_clahe)

    print("Subproblem 4: Saved final enhanced image.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./images/noisy_image.tif')
    parser.add_argument('--output', type=str, default='./output_b/')
    parser.add_argument('--subproblem', type=int, required=True, choices=[1, 2, 3, 4])
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input image not found at {args.input}", file=sys.stderr)
        print("Please make sure 'noisy_image.tif' is in the 'images' directory or specify the correct path with --input.", file=sys.stderr)
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
        print(f"Invalid subproblem: {args.subproblem}", file=sys.stderr)

if __name__ == '__main__':
    main()