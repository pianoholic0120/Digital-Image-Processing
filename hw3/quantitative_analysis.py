import os
import numpy as np
import cv2
import sys

def compute_image_metrics(img):
    metrics = {}
    
    metrics['mean'] = float(np.mean(img))
    metrics['std'] = float(np.std(img))
    metrics['min'] = float(np.min(img))
    metrics['max'] = float(np.max(img))
    
    metrics['contrast'] = metrics['std']
    
    grad_y = np.gradient(img.astype(np.float32), axis=0)
    grad_x = np.gradient(img.astype(np.float32), axis=1)
    edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    metrics['edge_strength'] = float(np.mean(edge_magnitude))
    metrics['edge_max'] = float(np.max(edge_magnitude))

    hist, _ = np.histogram(img, bins=256, range=(0, 256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]  # Remove zeros
    metrics['entropy'] = float(-np.sum(hist * np.log2(hist)))
    
    metrics['dynamic_range'] = float(metrics['max'] - metrics['min'])
    
    return metrics

def compare_images(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    diff = img1 - img2
    
    metrics = {}
    metrics['mse'] = float(np.mean(diff**2))
    metrics['mae'] = float(np.mean(np.abs(diff)))
    metrics['max_error'] = float(np.max(np.abs(diff)))
    
    if metrics['mse'] > 1e-10:
        metrics['psnr'] = float(20 * np.log10(255.0 / np.sqrt(metrics['mse'])))
    else:
        metrics['psnr'] = float('inf')
    
    identical_pixels = np.sum(np.abs(diff) < 0.5)
    total_pixels = diff.size
    metrics['identical_ratio'] = float(identical_pixels / total_pixels)
    
    return metrics

def analyze_part_a(output_dir):
    print("\n" + "="*80)
    print("PART A: QUANTITATIVE ANALYSIS")
    print("="*80)
    
    a3_path = os.path.join(output_dir, 'a3.png')
    a4_path = os.path.join(output_dir, 'a4.png')
    a5_path = os.path.join(output_dir, 'a5.png')
    
    if not all(os.path.exists(p) for p in [a3_path, a4_path, a5_path]):
        print("Error: Required Part A output files not found.")
        return None
    
    a3 = cv2.imread(a3_path, cv2.IMREAD_GRAYSCALE)
    a4 = cv2.imread(a4_path, cv2.IMREAD_GRAYSCALE)
    a5 = cv2.imread(a5_path, cv2.IMREAD_GRAYSCALE)
    
    print("\n1. Frequency Domain (a3) vs Spatial Domain (a4) Comparison:")
    print("-" * 80)
    comp_34 = compare_images(a3, a4)
    print(f"   MSE:           {comp_34['mse']:.6f}")
    print(f"   MAE:           {comp_34['mae']:.6f}")
    print(f"   Max Error:     {comp_34['max_error']:.2f}")
    print(f"   PSNR:          {comp_34['psnr']:.2f} dB" if comp_34['psnr'] != float('inf') else "   PSNR:          ∞ (identical)")
    print(f"   Identical:     {comp_34['identical_ratio']*100:.2f}%")
    
    print("\n2. With Odd Symmetry (a3) vs Without Odd Symmetry (a5) Comparison:")
    print("-" * 80)
    comp_35 = compare_images(a3, a5)
    print(f"   MSE:           {comp_35['mse']:.6f}")
    print(f"   MAE:           {comp_35['mae']:.6f}")
    print(f"   Max Error:     {comp_35['max_error']:.2f}")
    print(f"   PSNR:          {comp_35['psnr']:.2f} dB")
    
    print("\n3. Image Quality Metrics:")
    print("-" * 80)
    print(f"{'Metric':<15} {'a3 (Freq-4x4)':<15} {'a4 (Spatial)':<15} {'a5 (Freq-3x3)':<15}")
    print("-" * 80)
    
    metrics_a3 = compute_image_metrics(a3)
    metrics_a4 = compute_image_metrics(a4)
    metrics_a5 = compute_image_metrics(a5)
    
    print(f"{'Mean':<15} {metrics_a3['mean']:<15.2f} {metrics_a4['mean']:<15.2f} {metrics_a5['mean']:<15.2f}")
    print(f"{'Std Dev':<15} {metrics_a3['std']:<15.2f} {metrics_a4['std']:<15.2f} {metrics_a5['std']:<15.2f}")
    print(f"{'Edge Strength':<15} {metrics_a3['edge_strength']:<15.2f} {metrics_a4['edge_strength']:<15.2f} {metrics_a5['edge_strength']:<15.2f}")
    print(f"{'Entropy':<15} {metrics_a3['entropy']:<15.2f} {metrics_a4['entropy']:<15.2f} {metrics_a5['entropy']:<15.2f}")
    
    return {
        'comparison_34': comp_34,
        'comparison_35': comp_35,
        'metrics_a3': metrics_a3,
        'metrics_a4': metrics_a4,
        'metrics_a5': metrics_a5
    }

def analyze_part_b(output_dir):
    print("\n" + "="*80)
    print("PART B: QUANTITATIVE ANALYSIS")
    print("="*80)
    
    input_path = os.path.join(output_dir, '..', 'images', 'noisy_image.tif')
    if not os.path.exists(input_path):
        input_path = os.path.join(output_dir, '..', 'images', 'noisy_image.png')
    
    if not os.path.exists(input_path):
        print("Error: Input image not found.")
        return None
    
    original = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    stage_files = {
        'Original': input_path,
        'After Notch': os.path.join(output_dir, '1_filtered_image.png'),
        'After Unsharp': os.path.join(output_dir, '2_sharpened.png'),
        'After Hist Eq': os.path.join(output_dir, '3_equalized.png'),
        'Final': os.path.join(output_dir, '4_my_procedure.png')
    }
    
    results = {}
    
    print(f"\n{'Stage':<20} {'Mean':<10} {'Std Dev':<10} {'Contrast':<10} {'Edge Str':<12} {'Entropy':<10}")
    print("-" * 80)
    
    for stage_name, file_path in stage_files.items():
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping...")
            continue
        
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        metrics = compute_image_metrics(img)
        
        if stage_name != 'Original':
            comp = compare_images(original, img)
            metrics['mse_vs_original'] = comp['mse']
            metrics['psnr_vs_original'] = comp['psnr']
        
        results[stage_name] = metrics
        
        print(f"{stage_name:<20} {metrics['mean']:<10.2f} {metrics['std']:<10.2f} "
              f"{metrics['contrast']:<10.2f} {metrics['edge_strength']:<12.2f} {metrics['entropy']:<10.2f}")
    
    return results

def analyze_ablation(ablation_dir):
    print("\n" + "="*80)
    print("ABLATION STUDY: QUANTITATIVE ANALYSIS")
    print("="*80)
    
    stages = [
        ('0_original', 'Original'),
        ('1_notch_only', 'Notch Only'),
        ('2_notch_unsharp', 'Notch + Unsharp'),
        ('3_notch_unsharp_clahe', 'Notch + Unsharp + CLAHE'),
        ('4_add_retinex', '+ Retinex'),
        ('5_add_histeq', '+ Hist Eq'),
        ('6_full_pipeline', 'Full Pipeline')
    ]
    
    original_path = os.path.join(ablation_dir, '0_original.png')
    if not os.path.exists(original_path):
        print("Error: Ablation study images not found.")
        return None
    
    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    results = {}
    
    print(f"\n{'Stage':<25} {'Std Dev':<12} {'Edge Str':<12} {'Entropy':<10} {'MSE':<12} {'PSNR':<10}")
    print("-" * 80)
    
    for filename, stage_name in stages:
        file_path = os.path.join(ablation_dir, f'{filename}.png')
        if not os.path.exists(file_path):
            continue
        
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        metrics = compute_image_metrics(img)
        comp = compare_images(original, img)
        
        metrics['mse'] = comp['mse']
        metrics['psnr'] = comp['psnr']
        
        results[stage_name] = metrics
        
        print(f"{stage_name:<25} {metrics['std']:<12.2f} {metrics['edge_strength']:<12.2f} "
              f"{metrics['entropy']:<10.2f} {metrics['mse']:<12.2f} {metrics['psnr']:<10.2f}")
    
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_a', type=str, default='./output_a/')
    parser.add_argument('--output_b', type=str, default='./output_b/')
    parser.add_argument('--ablation', type=str, default='./output_b/ablation/')
    args = parser.parse_args()
    
    results_a = None
    results_b = None
    results_ablation = None
    
    if os.path.exists(args.output_a):
        results_a = analyze_part_a(args.output_a)
    
    if os.path.exists(args.output_b):
        results_b = analyze_part_b(args.output_b)
    
    if os.path.exists(args.ablation):
        results_ablation = analyze_ablation(args.ablation)
    
    output_file = os.path.join(args.output_b, 'quantitative_results.txt')
    with open(output_file, 'w') as f:
        f.write("QUANTITATIVE ANALYSIS RESULTS\n")
        f.write("="*80 + "\n\n")
        
        if results_a:
            f.write("PART A RESULTS\n")
            f.write("-"*80 + "\n")
            f.write(f"Frequency vs Spatial (a3 vs a4):\n")
            f.write(f"  MSE: {results_a['comparison_34']['mse']:.6f}\n")
            f.write(f"  Identical pixels: {results_a['comparison_34']['identical_ratio']*100:.2f}%\n\n")
        
        if results_ablation:
            f.write("ABLATION STUDY RESULTS\n")
            f.write("-"*80 + "\n")
            for stage, metrics in results_ablation.items():
                f.write(f"{stage}:\n")
                f.write(f"  Std Dev: {metrics['std']:.2f}\n")
                f.write(f"  Edge Strength: {metrics['edge_strength']:.2f}\n")
                f.write(f"  Entropy: {metrics['entropy']:.2f}\n\n")
    
    print(f"\nResults saved to: {output_file}")
    
    return results_a, results_b, results_ablation

if __name__ == '__main__':
    main()

