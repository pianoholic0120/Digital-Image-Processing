import cv2
import numpy as np
import argparse
import os

# ======================================================================================
# HW 2(3): Auto White Balance (AWB) Algorithm
# Implementation based on "Shades of Gray and Colour Constancy" by Finlayson & Trezzi
# ======================================================================================

def shades_of_gray_illuminant(img, p=6, epsilon=1e-10):
    """
    Estimate illuminant using Shades of Gray algorithm (Minkowski p-norm).
    
    Paper: "Shades of Gray and Colour Constancy" by Finlayson & Trezzi (2004)
    
    The core idea is to compute the p-norm of each color channel:
    e_c = (∫ |f_c(x)|^p dx)^(1/p)
    
    Special cases:
    - p=1: Grey World assumption (mean)
    - p=2: L2 norm
    - p=6: Empirically found to be optimal in the paper
    - p→∞: Max RGB assumption (maximum value)
    
    Args:
        img: Input image in BGR format
        p: Minkowski norm order (default=6, optimal from paper)
        epsilon: Small value to avoid numerical issues
    
    Returns:
        illuminant_bgr: Estimated illuminant in BGR format (0-255 scale)
    """
    # Normalize image to [0, 1] range
    img_float = img.astype(np.float64) / 255.0
    
    # Add epsilon to avoid zero values
    img_float = np.maximum(img_float, epsilon)
    
    # Split channels (BGR format)
    b, g, r = cv2.split(img_float)
    
    # Compute Minkowski p-norm for each channel
    # e_c = (mean(|f_c|^p))^(1/p)
    if np.isinf(p):
        # Max RGB: use maximum value
        b_est = np.max(b)
        g_est = np.max(g)
        r_est = np.max(r)
    else:
        # General case: Minkowski p-norm
        b_est = np.power(np.mean(np.power(b, p)), 1.0/p)
        g_est = np.power(np.mean(np.power(g, p)), 1.0/p)
        r_est = np.power(np.mean(np.power(r, p)), 1.0/p)
    
    # Return in BGR format, scaled to [0, 255]
    illuminant_bgr = np.array([b_est, g_est, r_est]) * 255.0
    
    return illuminant_bgr


def grey_world_illuminant(img):
    """
    Grey World assumption: average of each channel is the illuminant.
    This is equivalent to Shades of Gray with p=1.
    """
    return shades_of_gray_illuminant(img, p=1)


def max_rgb_illuminant(img):
    """
    Max RGB assumption: maximum value of each channel is the illuminant.
    This is equivalent to Shades of Gray with p→∞.
    """
    return shades_of_gray_illuminant(img, p=np.inf)


def multi_scale_shades_of_gray(img, p_values=[1, 2, 4, 6, 8, 10], weights=None):
    """
    Combine multiple Shades of Gray estimates with different p values.
    This can provide more robust illuminant estimation.
    
    Args:
        img: Input image in BGR format
        p_values: List of p values to use
        weights: Weights for each p value (if None, use equal weights)
    
    Returns:
        illuminant_bgr: Combined illuminant estimate
    """
    if weights is None:
        weights = np.ones(len(p_values)) / len(p_values)
    else:
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize
    
    illuminants = []
    for p in p_values:
        illuminants.append(shades_of_gray_illuminant(img, p=p))
    
    illuminants = np.array(illuminants)
    combined = np.average(illuminants, axis=0, weights=weights)
    
    return combined


def von_kries_adaptation(img, illuminant_bgr, target_illuminant=None):
    """
    Apply von Kries chromatic adaptation to correct white balance.
    
    Args:
        img: Input image in BGR format
        illuminant_bgr: Estimated scene illuminant (0-255 scale)
        target_illuminant: Target illuminant (default: neutral [1,1,1])
    
    Returns:
        Corrected image
    """
    if illuminant_bgr is None:
        return img
    
    img_float = img.astype(np.float32)
    
    # Normalize illuminant to [0, 1]
    illuminant_normalized = illuminant_bgr / 255.0
    
    # Target is typically neutral (equal RGB)
    if target_illuminant is None:
        target_illuminant = np.array([1.0, 1.0, 1.0])
    
    # Compute gain factors
    gains = target_illuminant / (illuminant_normalized + 1e-10)
    
    # Split channels
    b, g, r = cv2.split(img_float)
    
    # Apply gains
    b_corrected = b * gains[0]
    g_corrected = g * gains[1]
    r_corrected = r * gains[2]
    
    # Merge and clip
    result = cv2.merge([b_corrected, g_corrected, r_corrected])
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def advanced_white_balance(img, method='shades_of_gray', p=6):
    """
    Main white balance function using Shades of Gray algorithm.
    
    Args:
        img: Input image in BGR format
        method: 'shades_of_gray', 'grey_world', 'max_rgb', or 'multi_scale'
        p: Minkowski norm order (only used for 'shades_of_gray')
    
    Returns:
        White-balanced image
    """
    print(f"Estimating illuminant using method: {method}")
    
    if method == 'grey_world':
        illuminant_bgr = grey_world_illuminant(img)
        print(f"Grey World illuminant: {illuminant_bgr}")
    elif method == 'max_rgb':
        illuminant_bgr = max_rgb_illuminant(img)
        print(f"Max RGB illuminant: {illuminant_bgr}")
    elif method == 'multi_scale':
        # Use weighted combination favoring p=6 (optimal from paper)
        illuminant_bgr = multi_scale_shades_of_gray(
            img, 
            p_values=[1, 2, 4, 6, 8, 10],
            weights=[0.1, 0.1, 0.15, 0.35, 0.2, 0.1]  # p=6 has highest weight
        )
        print(f"Multi-scale illuminant: {illuminant_bgr}")
    else:  # 'shades_of_gray'
        illuminant_bgr = shades_of_gray_illuminant(img, p=p)
        print(f"Shades of Gray (p={p}) illuminant: {illuminant_bgr}")
    
    # Apply von Kries adaptation
    result = von_kries_adaptation(img, illuminant_bgr)
    
    return result

def auto_white_balance(img, method='shades_of_gray', p=6):
    """
    Convenience function for automatic white balance.
    """
    return advanced_white_balance(img, method=method, p=p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto White Balance using Shades of Gray Algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default Shades of Gray (p=6, optimal from paper)
  python p3.py --input_image image.tif --output_dir results
  
  # Use Grey World (p=1)
  python p3.py --input_image image.tif --output_dir results --method grey_world
  
  # Use Max RGB (p→∞)
  python p3.py --input_image image.tif --output_dir results --method max_rgb
  
  # Use multi-scale approach (most robust)
  python p3.py --input_image image.tif --output_dir results --method multi_scale
  
  # Use custom p value
  python p3.py --input_image image.tif --output_dir results --method shades_of_gray --p 8
        """
    )
    parser.add_argument("--input_image", type=str, required=True, 
                       help="Path to the input image (.tif, .png, .jpg, etc.)")
    parser.add_argument("--output_dir", type=str, required=True, 
                       help="Directory to save the output image")
    parser.add_argument("--method", type=str, default='shades_of_gray',
                       choices=['shades_of_gray', 'grey_world', 'max_rgb', 'multi_scale'],
                       help="Illuminant estimation method (default: shades_of_gray)")
    parser.add_argument("--p", type=float, default=6,
                       help="Minkowski norm order for shades_of_gray method (default: 6)")
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input_image):
        print(f"Error: Input image not found at {args.input_image}")
        exit(1)
    
    # Read image
    img = cv2.imread(args.input_image)
    if img is None:
        print(f"Error: Could not read the image file {args.input_image}")
        exit(1)
    
    print(f"Processing image: {args.input_image}")
    print(f"Image size: {img.shape}")
    print(f"Method: {args.method}")
    if args.method == 'shades_of_gray':
        print(f"Minkowski norm p: {args.p}")
    print("-" * 60)
    
    # Apply white balance
    result = auto_white_balance(img, method=args.method, p=args.p)
    
    # Save result
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.basename(args.input_image)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(args.output_dir, f"{name}.png")
    
    cv2.imwrite(output_path, result)
    print("-" * 60)
    print(f"✓ Success! White balanced image saved to: {output_path}")