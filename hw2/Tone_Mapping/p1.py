import cv2
import numpy as np
import argparse
import os

# ======================================================================================
# HW 2(1): Chrominance-Aware Tone Mapping with Curve Visualization
# ======================================================================================

def calculate_mse(image1, image2):
    """Calculate Mean Squared Error between two images."""
    if image1.shape != image2.shape:
        if image1.shape[:2] != image2.shape[:2]:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    
    img1_float = image1.astype(np.float64)
    img2_float = image2.astype(np.float64)
    mse = np.mean((img1_float - img2_float) ** 2)
    return mse

def calculate_psnr(image1, image2, max_pixel_value=255.0):
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = calculate_mse(image1, image2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr

def histogram_match(source, reference):
    """
    Perform histogram matching to transfer the histogram of reference to source.
    Returns both the matched image and the LUT used for transformation.
    """
    s_hist = cv2.calcHist([source], [0], None, [256], [0, 256])
    r_hist = cv2.calcHist([reference], [0], None, [256], [0, 256])
    
    s_cdf = s_hist.cumsum()
    r_cdf = r_hist.cumsum()
    
    # Normalize CDFs
    s_cdf_norm = s_cdf / (s_cdf[-1] + 1e-10)
    r_cdf_norm = r_cdf / (r_cdf[-1] + 1e-10)
    
    # Build lookup table
    lut = np.zeros(256, dtype='uint8')
    g = 0
    for j in range(256):
        while g < 255 and r_cdf_norm[g] < s_cdf_norm[j]:
            g += 1
        lut[j] = g
    
    matched = cv2.LUT(source, lut)
    return matched, lut

def chrominance_aware_tone_mapping(source_img, reference_img):
    source_ycrcb = cv2.cvtColor(source_img, cv2.COLOR_BGR2YCrCb)
    ref_ycrcb = cv2.cvtColor(reference_img, cv2.COLOR_BGR2YCrCb)
    
    s_y, s_cr, s_cb = cv2.split(source_ycrcb)
    r_y, r_cr, r_cb = cv2.split(ref_ycrcb)
    matched_y, lut_y = histogram_match(s_y, r_y)
    final_cr, lut_cr = histogram_match(s_cr, r_cr)
    final_cb, lut_cb = histogram_match(s_cb, r_cb)
    
    alpha = 1.0
    
    result_ycrcb = cv2.merge([matched_y, final_cr, final_cb])
    result_img = cv2.cvtColor(result_ycrcb, cv2.COLOR_YCrCb2BGR)
    
    curves = {
        'lut_y': lut_y,    
        'lut_cr': lut_cr,
        'lut_cb': lut_cb,
        'alpha': alpha    # alpha = 1.0
    }
    
    return result_img, curves

def draw_curve_plot(lut, title, color, width=500, height=500):
    """
    Draw a single tone mapping curve using OpenCV.
    
    Args:
        lut: Lookup table (256 values)
        title: Title for the plot
        color: BGR color tuple for the curve
        width, height: Plot dimensions
    
    Returns:
        Image with the plotted curve
    """
    # Create white background
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Define margins
    margin_left = 60
    margin_right = 30
    margin_top = 50
    margin_bottom = 50
    
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    
    # Draw axes
    cv2.line(img, (margin_left, margin_top), (margin_left, height - margin_bottom), (0, 0, 0), 2)
    cv2.line(img, (margin_left, height - margin_bottom), (width - margin_right, height - margin_bottom), (0, 0, 0), 2)
    
    # Draw grid
    for i in range(0, 6):
        grid_y = margin_top + i * plot_height // 5
        grid_val = 255 - i * 51
        cv2.line(img, (margin_left, grid_y), (width - margin_right, grid_y), (200, 200, 200), 1)
        cv2.putText(img, str(grid_val), (5, grid_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        grid_x = margin_left + i * plot_width // 5
        if i <= 5:
            cv2.line(img, (grid_x, margin_top), (grid_x, height - margin_bottom), (200, 200, 200), 1)
            cv2.putText(img, str(i * 51), (grid_x - 10, height - margin_bottom + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Draw identity line (diagonal)
    for i in range(255):
        x1 = margin_left + int(i * plot_width / 255.0)
        y1 = height - margin_bottom - int(i * plot_height / 255.0)
        x2 = margin_left + int((i + 1) * plot_width / 255.0)
        y2 = height - margin_bottom - int((i + 1) * plot_height / 255.0)
        cv2.line(img, (x1, y1), (x2, y2), (150, 150, 150), 1, cv2.LINE_AA)
    
    # Draw the tone mapping curve
    for i in range(255):
        x1 = margin_left + int(i * plot_width / 255)
        y1 = height - margin_bottom - int(int(lut[i]) * plot_height / 255)
        x2 = margin_left + int((i + 1) * plot_width / 255)
        y2 = height - margin_bottom - int(int(lut[i + 1]) * plot_height / 255)
        cv2.line(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    
    # Add title
    cv2.putText(img, title, (width // 2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Add axis labels
    cv2.putText(img, 'Input Pixel Value', (width // 2 - 60, height - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Vertical text for Y axis (rotated effect with individual chars)
    y_label = "Output"
    for idx, char in enumerate(y_label):
        cv2.putText(img, char, (10, margin_top + 60 + idx * 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return img

def save_tone_mapping_curves(curves, output_path):
    """
    Visualize and save the tone mapping curves for Y, Cr, and Cb channels using OpenCV.
    """
    plot_size = 500
    
    # Create individual curve plots
    y_plot = draw_curve_plot(curves['lut_y'], 'Luminance (Y) Channel', (255, 0, 0), plot_size, plot_size)
    cr_plot = draw_curve_plot(curves['lut_cr'], 'Chrominance Red (Cr)', (0, 0, 255), plot_size, plot_size)
    cb_plot = draw_curve_plot(curves['lut_cb'], 'Chrominance Blue (Cb)', (0, 255, 0), plot_size, plot_size)
    
    # Create combined plot with all curves
    combined_img = np.ones((plot_size, plot_size, 3), dtype=np.uint8) * 255
    margin_left = 60
    margin_right = 30
    margin_top = 50
    margin_bottom = 50
    plot_width = plot_size - margin_left - margin_right
    plot_height = plot_size - margin_top - margin_bottom
    
    # Draw axes and grid for combined plot
    cv2.line(combined_img, (margin_left, margin_top), (margin_left, plot_size - margin_bottom), (0, 0, 0), 2)
    cv2.line(combined_img, (margin_left, plot_size - margin_bottom), 
            (plot_size - margin_right, plot_size - margin_bottom), (0, 0, 0), 2)
    
    for i in range(0, 6):
        grid_y = margin_top + i * plot_height // 5
        grid_val = 255 - i * 51
        cv2.line(combined_img, (margin_left, grid_y), (plot_size - margin_right, grid_y), (200, 200, 200), 1)
        cv2.putText(combined_img, str(grid_val), (5, grid_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        grid_x = margin_left + i * plot_width // 5
        if i <= 5:
            cv2.line(combined_img, (grid_x, margin_top), (grid_x, plot_size - margin_bottom), (200, 200, 200), 1)
            cv2.putText(combined_img, str(i * 51), (grid_x - 10, plot_size - margin_bottom + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Draw identity line
    for i in range(255):
        x1 = margin_left + int(i * plot_width / 255.0)
        y1 = plot_size - margin_bottom - int(i * plot_height / 255.0)
        x2 = margin_left + int((i + 1) * plot_width / 255.0)
        y2 = plot_size - margin_bottom - int((i + 1) * plot_height / 255.0)
        cv2.line(combined_img, (x1, y1), (x2, y2), (150, 150, 150), 1, cv2.LINE_AA)
    
    # Draw all three curves
    for i in range(255):
        x1 = margin_left + int(i * plot_width / 255)
        x2 = margin_left + int((i + 1) * plot_width / 255)
        
        # Y curve (blue)
        y1_y = plot_size - margin_bottom - int(int(curves['lut_y'][i]) * plot_height / 255)
        y2_y = plot_size - margin_bottom - int(int(curves['lut_y'][i + 1]) * plot_height / 255)
        cv2.line(combined_img, (x1, y1_y), (x2, y2_y), (255, 0, 0), 2, cv2.LINE_AA)
        
        # Cr curve (red)
        y1_cr = plot_size - margin_bottom - int(int(curves['lut_cr'][i]) * plot_height / 255)
        y2_cr = plot_size - margin_bottom - int(int(curves['lut_cr'][i + 1]) * plot_height / 255)
        cv2.line(combined_img, (x1, y1_cr), (x2, y2_cr), (0, 0, 255), 2, cv2.LINE_AA)
        
        # Cb curve (green)
        y1_cb = plot_size - margin_bottom - int(int(curves['lut_cb'][i]) * plot_height / 255)
        y2_cb = plot_size - margin_bottom - int(int(curves['lut_cb'][i + 1]) * plot_height / 255)
        cv2.line(combined_img, (x1, y1_cb), (x2, y2_cb), (0, 255, 0), 2, cv2.LINE_AA)
    
    # Add title and legend
    title_text = f'All Channels (alpha={curves["alpha"]:.2f})'
    cv2.putText(combined_img, title_text, (plot_size // 2 - 100, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Add legend
    legend_y = margin_top + 20
    cv2.line(combined_img, (plot_size - 150, legend_y), (plot_size - 120, legend_y), (255, 0, 0), 2)
    cv2.putText(combined_img, 'Y', (plot_size - 110, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    cv2.line(combined_img, (plot_size - 150, legend_y + 20), (plot_size - 120, legend_y + 20), (0, 0, 255), 2)
    cv2.putText(combined_img, 'Cr', (plot_size - 110, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    cv2.line(combined_img, (plot_size - 150, legend_y + 40), (plot_size - 120, legend_y + 40), (0, 255, 0), 2)
    cv2.putText(combined_img, 'Cb', (plot_size - 110, legend_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Combine all four plots into a 2x2 grid
    top_row = np.hstack([y_plot, cr_plot])
    bottom_row = np.hstack([cb_plot, combined_img])
    final_img = np.vstack([top_row, bottom_row])
    
    # Add overall title
    title_height = 60
    title_bar = np.ones((title_height, final_img.shape[1], 3), dtype=np.uint8) * 240
    cv2.putText(title_bar, 'Tone Mapping Curves (White-balanced -> Reference)', 
               (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    final_img = np.vstack([title_bar, final_img])
    
    # Save the combined image
    cv2.imwrite(output_path, final_img)
    print(f"Tone mapping curves saved to: {output_path}")

def evaluate_tone_mapping(source_img, reference_img, result_img, verbose=True):
    """Evaluate tone mapping quality using MSE and PSNR metrics."""
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
        print("Chrominance-Aware Tone Mapping Evaluation Results")
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
    parser = argparse.ArgumentParser(
        description="Chrominance-Aware Tone Mapping with Curve Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python p1.py --source_image input.png --reference_image ref.png --output_image output.png
  python p1.py --source_image input.png --reference_image ref.png --output_image output.png --output_curve curve.png --output_metrics metrics.txt
        """
    )
    parser.add_argument("--source_image", type=str, required=True,
                       help="Path to the white-balanced source image")
    parser.add_argument("--reference_image", type=str, required=True,
                       help="Path to the reference image")
    parser.add_argument("--output_image", type=str, required=True,
                       help="Path to save the tone-mapped output image")
    parser.add_argument("--output_curve", type=str, default=None,
                       help="Path to save the tone mapping curve visualization (optional)")
    parser.add_argument("--output_metrics", type=str, default=None,
                       help="Path to save evaluation metrics (optional)")
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.source_image):
        print(f"Error: Source image not found: {args.source_image}")
        exit(1)
    
    if not os.path.exists(args.reference_image):
        print(f"Error: Reference image not found: {args.reference_image}")
        exit(1)
    
    # Read images
    print(f"Reading source image: {args.source_image}")
    source_img = cv2.imread(args.source_image)
    
    print(f"Reading reference image: {args.reference_image}")
    reference_img = cv2.imread(args.reference_image)
    
    if source_img is None or reference_img is None:
        print("Error: Could not read images")
        exit(1)
    
    print(f"Source image shape: {source_img.shape}")
    print(f"Reference image shape: {reference_img.shape}")
    
    # Apply chrominance-aware tone mapping
    print(f"\nApplying chrominance-aware tone mapping...")
    result_img, curves = chrominance_aware_tone_mapping(source_img, reference_img)
    print(f"Tone mapping completed. Alpha value used: {curves['alpha']:.3f}")
    
    # Evaluate results
    metrics = evaluate_tone_mapping(source_img, reference_img, result_img, verbose=True)
    
    # Save output image
    os.makedirs(os.path.dirname(args.output_image) if os.path.dirname(args.output_image) else '.', exist_ok=True)
    cv2.imwrite(args.output_image, result_img)
    print(f"✓ Output image saved to: {args.output_image}")
    
    # Save tone mapping curves if requested
    if args.output_curve:
        curve_dir = os.path.dirname(args.output_curve)
        if curve_dir:
            os.makedirs(curve_dir, exist_ok=True)
        save_tone_mapping_curves(curves, args.output_curve)
        print(f"✓ Tone mapping curves saved to: {args.output_curve}")
    else:
        # Auto-generate curve filename if not specified
        base_name = os.path.splitext(args.output_image)[0]
        auto_curve_path = f"{base_name}_curve.png"
        save_tone_mapping_curves(curves, auto_curve_path)
    
    # Save metrics if requested
    if args.output_metrics:
        metrics_dir = os.path.dirname(args.output_metrics)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        with open(args.output_metrics, 'w') as f:
            f.write("Chrominance-Aware Tone Mapping Evaluation\n")
            f.write("="*80 + "\n\n")
            f.write(f"Source image: {args.source_image}\n")
            f.write(f"Reference image: {args.reference_image}\n")
            f.write(f"Output image: {args.output_image}\n")
            f.write(f"Alpha value: {curves['alpha']:.4f}\n\n")
            f.write("Performance Metrics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"MSE before:  {metrics['mse_before']:.4f}\n")
            f.write(f"MSE after:   {metrics['mse_after']:.4f}\n")
            f.write(f"Improvement: {metrics['mse_improvement_percent']:.2f}%\n\n")
            f.write(f"PSNR before: {metrics['psnr_before']:.4f} dB\n")
            f.write(f"PSNR after:  {metrics['psnr_after']:.4f} dB\n")
            f.write(f"Improvement: {metrics['psnr_improvement_db']:.4f} dB\n")
        print(f"✓ Metrics saved to: {args.output_metrics}")
    
    print(f"\n{'='*80}")
    print("Processing completed successfully!")
    print(f"{'='*80}\n")
