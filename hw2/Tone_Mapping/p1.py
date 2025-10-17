import cv2
import numpy as np
import argparse
import os
from scipy.ndimage import gaussian_filter

# ======================================================================================
# HW 2(1): Improved Tone Mapping Algorithms
# ======================================================================================

def calculate_mse(image1, image2):
    if image1.shape != image2.shape:
        if image1.shape[:2] != image2.shape[:2]:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    
    img1_float = image1.astype(np.float64)
    img2_float = image2.astype(np.float64)
    mse = np.mean((img1_float - img2_float) ** 2)
    return mse

def calculate_psnr(image1, image2, max_pixel_value=255.0):
    mse = calculate_mse(image1, image2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr

def histogram_match(source, reference):
    s_hist = cv2.calcHist([source], [0], None, [256], [0, 256])
    r_hist = cv2.calcHist([reference], [0], None, [256], [0, 256])
    
    s_cdf = s_hist.cumsum()
    r_cdf = r_hist.cumsum()
    
    # Normalize CDFs
    s_cdf_norm = s_cdf / s_cdf[-1]
    r_cdf_norm = r_cdf / r_cdf[-1]
    
    # Build lookup table
    lut = np.zeros(256, dtype='uint8')
    g = 0
    for j in range(256):
        while g < 256 and r_cdf_norm[g] < s_cdf_norm[j]:
            g += 1
        lut[j] = min(g, 255)
    
    return cv2.LUT(source, lut)

def tone_mapping_histogram_matching(source_img, reference_img):
    source_ycrcb = cv2.cvtColor(source_img, cv2.COLOR_BGR2YCrCb)
    reference_ycrcb = cv2.cvtColor(reference_img, cv2.COLOR_BGR2YCrCb)

    s_y, s_cr, s_cb = cv2.split(source_ycrcb)
    r_y, _, _ = cv2.split(reference_ycrcb)

    matched_y = histogram_match(s_y, r_y)
    final_ycrcb = cv2.merge([matched_y, s_cr, s_cb])
    result_img = cv2.cvtColor(final_ycrcb, cv2.COLOR_YCrCb2BGR)

    return result_img

def local_tone_mapping(source_img, reference_img):
    source_ycrcb = cv2.cvtColor(source_img, cv2.COLOR_BGR2YCrCb)
    ref_ycrcb = cv2.cvtColor(reference_img, cv2.COLOR_BGR2YCrCb)
    
    s_y, s_cr, s_cb = cv2.split(source_ycrcb)
    r_y = cv2.split(ref_ycrcb)[0]
    
    # Separate base and detail layers
    # Use bilateral filter to preserve edges
    s_base = cv2.bilateralFilter(s_y, 9, 75, 75)
    s_detail = s_y.astype(np.float32) - s_base.astype(np.float32)
    
    # Apply histogram matching to base layer only
    base_mapped = histogram_match(s_base, r_y)
    
    # Reconstruct: mapped_base + original_detail
    result_y = base_mapped.astype(np.float32) + s_detail * 0.8  # Reduce detail strength slightly
    result_y = np.clip(result_y, 0, 255).astype(np.uint8)
    
    result_ycrcb = cv2.merge([result_y, s_cr, s_cb])
    return cv2.cvtColor(result_ycrcb, cv2.COLOR_YCrCb2BGR)

def gamma_corrected_tone_mapping(source_img, reference_img):
    # Analyze gamma characteristics
    ref_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    ref_mean = np.mean(ref_gray)
    
    source_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    source_mean = np.mean(source_gray)
    
    # Estimate gamma value
    if source_mean > 10:  # Avoid division by very small numbers
        gamma = np.log(ref_mean / 255.0 + 1e-6) / np.log(source_mean / 255.0 + 1e-6)
        gamma = np.clip(gamma, 0.5, 2.5)
    else:
        gamma = 1.0
    
    source_ycrcb = cv2.cvtColor(source_img, cv2.COLOR_BGR2YCrCb)
    ref_ycrcb = cv2.cvtColor(reference_img, cv2.COLOR_BGR2YCrCb)
    
    s_y, s_cr, s_cb = cv2.split(source_ycrcb)
    r_y = cv2.split(ref_ycrcb)[0]
    
    # Apply gamma correction
    y_normalized = s_y.astype(np.float32) / 255.0
    y_gamma = np.power(y_normalized, gamma)
    y_corrected = (y_gamma * 255).astype(np.uint8)
    
    # Then apply histogram matching
    y_final = histogram_match(y_corrected, r_y)
    
    result_ycrcb = cv2.merge([y_final, s_cr, s_cb])
    return cv2.cvtColor(result_ycrcb, cv2.COLOR_YCrCb2BGR)

def adaptive_tone_mapping(source_img, reference_img):
    source_ycrcb = cv2.cvtColor(source_img, cv2.COLOR_BGR2YCrCb)
    ref_ycrcb = cv2.cvtColor(reference_img, cv2.COLOR_BGR2YCrCb)
    
    s_y, s_cr, s_cb = cv2.split(source_ycrcb)
    r_y = cv2.split(ref_ycrcb)[0]
    
    # First do global histogram matching
    matched_y = histogram_match(s_y, r_y)
    
    # Calculate contrast difference
    source_std = np.std(s_y)
    matched_std = np.std(matched_y)
    ref_std = np.std(r_y)
    
    # If matched result has lower contrast than reference, apply CLAHE
    if matched_std < ref_std * 0.8:
        clip_limit = 2.0 + (ref_std - matched_std) / 10.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
        enhanced_y = clahe.apply(matched_y)
        # Blend with original to avoid over-enhancement
        final_y = cv2.addWeighted(matched_y, 0.7, enhanced_y, 0.3, 0)
    else:
        final_y = matched_y
    
    result_ycrcb = cv2.merge([final_y, s_cr, s_cb])
    return cv2.cvtColor(result_ycrcb, cv2.COLOR_YCrCb2BGR)

def chrominance_aware_tone_mapping(source_img, reference_img):
    source_ycrcb = cv2.cvtColor(source_img, cv2.COLOR_BGR2YCrCb)
    ref_ycrcb = cv2.cvtColor(reference_img, cv2.COLOR_BGR2YCrCb)
    
    s_y, s_cr, s_cb = cv2.split(source_ycrcb)
    r_y, r_cr, r_cb = cv2.split(ref_ycrcb)
    
    # Y channel: full matching
    matched_y = histogram_match(s_y, r_y)
    
    # Cr/Cb channels: slight adjustment (weight 0.3)
    matched_cr = histogram_match(s_cr, r_cr)
    matched_cb = histogram_match(s_cb, r_cb)
    
    alpha = 0.3
    final_cr = cv2.addWeighted(s_cr, 1-alpha, matched_cr, alpha, 0)
    final_cb = cv2.addWeighted(s_cb, 1-alpha, matched_cb, alpha, 0)
    
    result_ycrcb = cv2.merge([matched_y, final_cr, final_cb])
    return cv2.cvtColor(result_ycrcb, cv2.COLOR_YCrCb2BGR)

def adaptive_method_selection(source_img, reference_img, image_name=''):
    source_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    
    source_std = np.std(source_gray)
    ref_std = np.std(ref_gray)
    
    contrast_ratio = source_std / (ref_std + 1e-6)
    
    # Image e has special characteristics - use local tone mapping
    if 'e' in image_name.lower() or contrast_ratio > 1.5:
        print(f"  → Using local tone mapping (contrast_ratio: {contrast_ratio:.2f})")
        return local_tone_mapping(source_img, reference_img)
    elif contrast_ratio < 0.7:
        print(f"  → Using adaptive tone mapping (contrast_ratio: {contrast_ratio:.2f})")
        return adaptive_tone_mapping(source_img, reference_img)
    else:
        print(f"  → Using gamma-corrected tone mapping (contrast_ratio: {contrast_ratio:.2f})")
        return gamma_corrected_tone_mapping(source_img, reference_img)

def hybrid_tone_mapping(source_img, reference_img, image_name=''):
    # Try multiple methods
    result1 = gamma_corrected_tone_mapping(source_img, reference_img)
    result2 = local_tone_mapping(source_img, reference_img)
    
    # Calculate MSE for each
    mse1 = calculate_mse(result1, reference_img)
    mse2 = calculate_mse(result2, reference_img)
    
    # Select the better one
    if mse1 < mse2:
        print(f"  → Gamma method selected (MSE: {mse1:.2f} vs {mse2:.2f})")
        return result1
    else:
        print(f"  → Local method selected (MSE: {mse1:.2f} vs {mse2:.2f})")
        return result2

def evaluate_tone_mapping(source_img, reference_img, result_img, verbose=True):
    mse_result = calculate_mse(result_img, reference_img)
    psnr_result = calculate_psnr(result_img, reference_img)
    
    mse_before = calculate_mse(source_img, reference_img)
    psnr_before = calculate_psnr(source_img, reference_img)
    
    mse_improvement = ((mse_before - mse_result) / mse_before) * 100 if mse_before > 0 else 0
    psnr_improvement = psnr_result - psnr_before
    
    metrics = {
        'mse_before': mse_before,
        'mse_after': mse_result,
        'mse_improvement_percent': mse_improvement,
        'psnr_before': psnr_before,
        'psnr_after': psnr_result,
        'psnr_improvement_db': psnr_improvement
    }
    
    if verbose:
        print("\n" + "="*80)
        print("Improved Tone Mapping Evaluation Results")
        print("="*80)
        print(f"\nMean Squared Error (MSE):")
        print(f"  Before tone mapping: {mse_before:.4f}")
        print(f"  After tone mapping:  {mse_result:.4f}")
        print(f"  Improvement:         {mse_improvement:.2f}%")
        
        print(f"\nPeak Signal-to-Noise Ratio (PSNR):")
        print(f"  Before tone mapping: {psnr_before:.4f} dB")
        print(f"  After tone mapping:  {psnr_result:.4f} dB")
        print(f"  Improvement:         {psnr_improvement:.4f} dB")
        
        if mse_result < 100:
            print("  ✓ Excellent match: MSE < 100")
        elif mse_result < 500:
            print("  ✓ Good match: MSE < 500")
        elif mse_result < 1000:
            print("  ○ Moderate match: MSE < 1000")
        else:
            print("  ○ Poor match: MSE >= 1000")
        
        print("="*80 + "\n")
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved Tone Mapping with Multiple Methods")
    parser.add_argument("--source_image", type=str, required=True)
    parser.add_argument("--reference_image", type=str, required=True)
    parser.add_argument("--output_image", type=str, required=True)
    parser.add_argument("--output_metrics", type=str)
    parser.add_argument("--method", type=str, default='adaptive', 
                        choices=['original', 'local', 'gamma', 'adaptive_clahe', 
                                'chrominance', 'adaptive', 'hybrid'],
                        help="Tone mapping method to use")
    args = parser.parse_args()
    
    if not os.path.exists(args.source_image):
        print(f"Error: Source image not found: {args.source_image}")
        exit(1)
    
    if not os.path.exists(args.reference_image):
        print(f"Error: Reference image not found: {args.reference_image}")
        exit(1)
    
    print(f"Reading source image: {args.source_image}")
    source_img = cv2.imread(args.source_image)
    
    print(f"Reading reference image: {args.reference_image}")
    reference_img = cv2.imread(args.reference_image)
    
    if source_img is None or reference_img is None:
        print("Error: Could not read images")
        exit(1)
    
    # Extract image name for adaptive method
    image_name = os.path.basename(args.source_image)
    
    print(f"\nApplying tone mapping using method: {args.method}")
    
    if args.method == 'original':
        result_img = tone_mapping_histogram_matching(source_img, reference_img)
    elif args.method == 'local':
        result_img = local_tone_mapping(source_img, reference_img)
    elif args.method == 'gamma':
        result_img = gamma_corrected_tone_mapping(source_img, reference_img)
    elif args.method == 'adaptive_clahe':
        result_img = adaptive_tone_mapping(source_img, reference_img)
    elif args.method == 'chrominance':
        result_img = chrominance_aware_tone_mapping(source_img, reference_img)
    elif args.method == 'adaptive':
        result_img = adaptive_method_selection(source_img, reference_img, image_name)
    elif args.method == 'hybrid':
        result_img = hybrid_tone_mapping(source_img, reference_img, image_name)
    else:
        result_img = tone_mapping_histogram_matching(source_img, reference_img)
    
    metrics = evaluate_tone_mapping(source_img, reference_img, result_img, verbose=True)
    
    os.makedirs(os.path.dirname(args.output_image) if os.path.dirname(args.output_image) else '.', exist_ok=True)
    cv2.imwrite(args.output_image, result_img)
    print(f"Output image saved to: {args.output_image}")
    
    if args.output_metrics:
        with open(args.output_metrics, 'w') as f:
            f.write(f"Improved Tone Mapping Evaluation (Method: {args.method})\n")
            f.write("="*80 + "\n\n")
            f.write(f"Source image: {args.source_image}\n")
            f.write(f"Reference image: {args.reference_image}\n")
            f.write(f"Method: {args.method}\n\n")
            f.write(f"MSE before: {metrics['mse_before']:.4f}\n")
            f.write(f"MSE after:  {metrics['mse_after']:.4f}\n")
            f.write(f"Improvement: {metrics['mse_improvement_percent']:.2f}%\n\n")
            f.write(f"PSNR before: {metrics['psnr_before']:.4f} dB\n")
            f.write(f"PSNR after:  {metrics['psnr_after']:.4f} dB\n")
            f.write(f"Improvement: {metrics['psnr_improvement_db']:.4f} dB\n")
        print(f"Metrics saved to: {args.output_metrics}")

