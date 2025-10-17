import cv2
import numpy as np
import os
import argparse

# ======================================================================================
# HW 2: Evaluate Auto White Balance (AWB) Algorithm
# ======================================================================================

def read_ground_truth_illuminant(rgb_file):
    try:
        with open(rgb_file, 'r') as f:
            line = f.readline().strip()
            r, g, b = map(float, line.split())
        return np.array([r, g, b])
    except Exception as e:
        print(f"Error reading {rgb_file}: {e}")
        return None

def estimate_illuminant_from_awb_image(original_img, awb_img):
    # Convert to float
    original_float = original_img.astype(np.float32)
    awb_float = awb_img.astype(np.float32)
    
    # Calculate the average gain applied to each channel
    # AWB applies: corrected = original * gain
    # So: gain = corrected / original (for non-zero pixels)
    
    # Use only bright pixels to estimate gains (avoid numerical instability)
    mask = (original_float > 50).all(axis=2)
    
    if np.sum(mask) == 0:
        mask = (original_float > 10).all(axis=2)
    
    gains = np.zeros(3)
    for i in range(3):
        original_channel = original_float[:, :, i][mask]
        awb_channel = awb_float[:, :, i][mask]
        
        if len(original_channel) > 0:
            gains[i] = np.median(awb_channel / (original_channel + 1e-6))
        else:
            gains[i] = 1.0
    
    # The estimated illuminant is inversely proportional to the gains
    # If gain = target_illuminant / source_illuminant
    # Then source_illuminant = target_illuminant / gain
    # Assuming target_illuminant = [1, 1, 1] (normalized white)
    estimated_illuminant = 1.0 / (gains + 1e-6)
    
    # Normalize the estimated illuminant
    estimated_illuminant = estimated_illuminant / np.max(estimated_illuminant) * 255.0
    
    # Return in RGB order (OpenCV uses BGR, so reverse)
    return estimated_illuminant[[2, 1, 0]]

def calculate_angular_error(estimated_illuminant, ground_truth_illuminant):
    # Normalize vectors
    est_norm = estimated_illuminant / (np.linalg.norm(estimated_illuminant) + 1e-10)
    gt_norm = ground_truth_illuminant / (np.linalg.norm(ground_truth_illuminant) + 1e-10)
    
    # Calculate cosine similarity
    cos_angle = np.dot(est_norm, gt_norm)
    
    # Clamp to [-1, 1] to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Calculate angular error in radians, then convert to degrees
    angular_error_rad = np.arccos(cos_angle)
    angular_error_deg = np.degrees(angular_error_rad)
    
    return angular_error_deg

def evaluate_awb_results(test_images_dir, awb_results_dir, output_file=None):
    image_names = ['a', 'b', 'c', 'd', 'e']
    
    results = []
    print("="*80)
    print("Auto White Balance (AWB) Evaluation - Angular Error Analysis")
    print("="*80)
    print()
    
    for name in image_names:
        # Read original image
        original_path = os.path.join(test_images_dir, f"{name}.tif")
        original_img = cv2.imread(original_path)
        
        if original_img is None:
            print(f"Warning: Could not read {original_path}")
            continue
        
        # Read AWB corrected image
        awb_path = os.path.join(awb_results_dir, f"{name}.png")
        awb_img = cv2.imread(awb_path)
        
        if awb_img is None:
            print(f"Warning: Could not read {awb_path}")
            continue
        
        # Read ground-truth illuminant
        rgb_file = os.path.join(test_images_dir, f"{name}.rgb")
        gt_illuminant = read_ground_truth_illuminant(rgb_file)
        
        if gt_illuminant is None:
            print(f"Warning: Could not read ground-truth illuminant for {name}")
            continue
        
        # Estimate illuminant from AWB result
        est_illuminant = estimate_illuminant_from_awb_image(original_img, awb_img)
        
        # Calculate angular error
        angular_error = calculate_angular_error(est_illuminant, gt_illuminant)
        
        # Store results
        result = {
            'image': name,
            'ground_truth_rgb': gt_illuminant,
            'estimated_rgb': est_illuminant,
            'angular_error_deg': angular_error
        }
        results.append(result)
        
        # Print results for this image
        print(f"Image: {name}.tif")
        print(f"  Ground-truth illuminant (RGB): [{gt_illuminant[0]:.2f}, {gt_illuminant[1]:.2f}, {gt_illuminant[2]:.2f}]")
        print(f"  Estimated illuminant (RGB):    [{est_illuminant[0]:.2f}, {est_illuminant[1]:.2f}, {est_illuminant[2]:.2f}]")
        print(f"  Angular error: {angular_error:.4f}°")
        print()
    
    # Calculate statistics
    if results:
        angular_errors = [r['angular_error_deg'] for r in results]
        mean_error = np.mean(angular_errors)
        std_error = np.std(angular_errors)
        min_error = np.min(angular_errors)
        max_error = np.max(angular_errors)
        
        print("="*80)
        print("Summary Statistics:")
        print("="*80)
        print(f"Mean angular error:     {mean_error:.4f}°")
        print(f"Std dev angular error:  {std_error:.4f}°")
        print(f"Min angular error:      {min_error:.4f}°")
        print(f"Max angular error:      {max_error:.4f}°")
        print()
        
        # Analysis
        print("="*80)
        print("Analysis:")
        print("="*80)
        if mean_error < 2.0:
            print("✓ Excellent performance: Mean angular error < 2°")
            print("  The AWB algorithm performs very well with high accuracy.")
        elif mean_error < 5.0:
            print("✓ Good performance: Mean angular error < 5°")
            print("  The AWB algorithm performs well with acceptable accuracy.")
        elif mean_error < 10.0:
            print("○ Moderate performance: Mean angular error < 10°")
            print("  The AWB algorithm works but there's room for improvement.")
        else:
            print("✗ Poor performance: Mean angular error >= 10°")
            print("  The AWB algorithm needs significant improvement.")
        
        print()
        if std_error < 2.0:
            print("✓ Consistent results: Low standard deviation")
        else:
            print("○ Inconsistent results: High standard deviation indicates varying performance")
        
        print()
        print("Individual image performance:")
        for r in results:
            if r['angular_error_deg'] < 5.0:
                status = "✓ Good"
            elif r['angular_error_deg'] < 10.0:
                status = "○ Moderate"
            else:
                status = "✗ Poor"
            print(f"  Image {r['image']}: {r['angular_error_deg']:.4f}° - {status}")
        
        print("="*80)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write("Auto White Balance (AWB) Evaluation Results\n")
                f.write("="*80 + "\n\n")
                
                for r in results:
                    f.write(f"Image: {r['image']}.tif\n")
                    f.write(f"  Ground-truth illuminant (RGB): [{r['ground_truth_rgb'][0]:.2f}, {r['ground_truth_rgb'][1]:.2f}, {r['ground_truth_rgb'][2]:.2f}]\n")
                    f.write(f"  Estimated illuminant (RGB):    [{r['estimated_rgb'][0]:.2f}, {r['estimated_rgb'][1]:.2f}, {r['estimated_rgb'][2]:.2f}]\n")
                    f.write(f"  Angular error: {r['angular_error_deg']:.4f}°\n\n")
                
                f.write("="*80 + "\n")
                f.write("Summary Statistics:\n")
                f.write("="*80 + "\n")
                f.write(f"Mean angular error:     {mean_error:.4f}°\n")
                f.write(f"Std dev angular error:  {std_error:.4f}°\n")
                f.write(f"Min angular error:      {min_error:.4f}°\n")
                f.write(f"Max angular error:      {max_error:.4f}°\n")
                
            print(f"\nResults saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AWB algorithm by computing angular error")
    parser.add_argument("--test_images_dir", type=str, required=True, 
                        help="Directory containing test images and .rgb files")
    parser.add_argument("--awb_results_dir", type=str, required=True, 
                        help="Directory containing AWB corrected images")
    parser.add_argument("--output_file", type=str, 
                        help="Optional output file to save evaluation results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.test_images_dir):
        print(f"Error: Test images directory not found: {args.test_images_dir}")
        exit(1)
    
    if not os.path.exists(args.awb_results_dir):
        print(f"Error: AWB results directory not found: {args.awb_results_dir}")
        exit(1)
    
    evaluate_awb_results(args.test_images_dir, args.awb_results_dir, args.output_file)

