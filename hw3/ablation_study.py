import os
import numpy as np
import cv2
import sys

# Import functions from hw3_b.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw3_b import (
    run_part1_logic, custom_gaussian_blur, custom_clahe,
    custom_equalize_hist, single_scale_retinex, normalize_img
)

def compute_metrics(img1, img2=None):
    metrics = {}
    
    metrics['std'] = float(np.std(img1))
    
    metrics['mean'] = float(np.mean(img1))
    
    grad_y = np.gradient(img1.astype(np.float32), axis=0)
    grad_x = np.gradient(img1.astype(np.float32), axis=1)
    edge_strength = np.sqrt(grad_x**2 + grad_y**2)
    metrics['edge_strength'] = float(np.mean(edge_strength))
    
    if img2 is not None:
        diff = img1.astype(np.float32) - img2.astype(np.float32)
        metrics['mse'] = float(np.mean(diff**2))
        metrics['psnr'] = float(20 * np.log10(255.0 / (np.sqrt(metrics['mse']) + 1e-10)))
    
    return metrics

def ablation_study(image_path, output_dir):
    print("Starting Ablation Study...")
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image at {image_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    baseline = img.copy()
    results.append(('0_original', baseline, compute_metrics(baseline)))
    cv2.imwrite(os.path.join(output_dir, '0_original.png'), baseline)
    
    g_p1, _ = run_part1_logic(img)
    step1 = normalize_img(g_p1)
    results.append(('1_notch_only', step1, compute_metrics(step1, baseline)))
    cv2.imwrite(os.path.join(output_dir, '1_notch_only.png'), step1)
    
    img_f = step1.astype(np.float32)
    g_blur_f = custom_gaussian_blur(img_f, (7, 7), 0)
    mask_f = img_f - g_blur_f
    mask_f[mask_f < 0] = 0
    step2 = np.clip(img_f + mask_f * 2.0, 0, 255).astype(np.uint8)
    results.append(('2_notch_unsharp', step2, compute_metrics(step2, baseline)))
    cv2.imwrite(os.path.join(output_dir, '2_notch_unsharp.png'), step2)
    
    step3 = custom_clahe(step2, clip_limit=2.5, tile_grid_size=(8, 8))
    results.append(('3_notch_unsharp_clahe', step3, compute_metrics(step3, baseline)))
    cv2.imwrite(os.path.join(output_dir, '3_notch_unsharp_clahe.png'), step3)
    
    g_retinex = single_scale_retinex(step3, sigma=30)
    g_retinex_norm = g_retinex - np.percentile(g_retinex, 1)
    g_retinex_norm = np.clip(g_retinex_norm, 0, None)
    if g_retinex_norm.max() > 1e-8:
        g_retinex_norm = g_retinex_norm / g_retinex_norm.max()
    step4 = (g_retinex_norm * 255).astype(np.uint8)
    results.append(('4_add_retinex', step4, compute_metrics(step4, baseline)))
    cv2.imwrite(os.path.join(output_dir, '4_add_retinex.png'), step4)
    
    step5 = custom_equalize_hist(step4)
    results.append(('5_add_histeq', step5, compute_metrics(step5, baseline)))
    cv2.imwrite(os.path.join(output_dir, '5_add_histeq.png'), step5)
    
    step6 = custom_clahe(step5, clip_limit=2.5, tile_grid_size=(16, 16))
    results.append(('6_full_pipeline', step6, compute_metrics(step6, baseline)))
    cv2.imwrite(os.path.join(output_dir, '6_full_pipeline.png'), step6)
    
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print(f"{'Step':<25} {'Std':<10} {'Mean':<10} {'Edge Str':<12} {'MSE':<12} {'PSNR':<10}")
    print("-"*80)
    
    for name, img, metrics in results:
        std = metrics.get('std', 0)
        mean = metrics.get('mean', 0)
        edge = metrics.get('edge_strength', 0)
        mse = metrics.get('mse', 0)
        psnr = metrics.get('psnr', 0)
        print(f"{name:<25} {std:<10.2f} {mean:<10.2f} {edge:<12.2f} {mse:<12.2f} {psnr:<10.2f}")
    
    print("="*80)
    print("\nMetrics Explanation:")
    print("- Std: Standard deviation (higher = better contrast)")
    print("- Mean: Average intensity")
    print("- Edge Str: Mean gradient magnitude (higher = sharper edges)")
    print("- MSE: Mean squared error vs. original (lower = less distortion)")
    print("- PSNR: Peak signal-to-noise ratio (higher = better quality)")
    print("\nOutput images saved to:", output_dir)
    
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./images/noisy_image.tif')
    parser.add_argument('--output', type=str, default='./output_b/ablation/')
    args = parser.parse_args()
    
    ablation_study(args.input, args.output)

