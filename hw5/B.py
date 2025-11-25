import cv2
import numpy as np
import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Manual Morphological Operations (Erosion, Dilation, Opening, Closing)")
    parser.add_argument('--input', type=str, required=True, help='Path to input image (e.g., noisy_rectangle.png)')
    parser.add_argument('--output_dir', type=str, default='output_images', help='Directory to save output images')
    return parser.parse_args()

def manual_erosion(image, kernel_size=5):
    h, w = image.shape
    pad_size = kernel_size // 2
    
    # Pad the image with zeros (black) to handle borders
    # For erosion, padding with max value usually avoids shrinking, 
    # but for object isolation on black background, 0 padding is standard.
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant', constant_values=0)
    
    # instead of using slow python for-loops over every pixel.
    shifted_views = []
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            # Extract the slice corresponding to the kernel position
            # Since we padded, we can extract slices of size (h, w)
            roi = padded_image[i:i+h, j:j+w]
            shifted_views.append(roi)
            
    # Stack them along a new axis
    stack = np.stack(shifted_views, axis=0)
    
    # Erosion: Minimum value in the neighborhood
    # If any pixel in the neighborhood is 0 (black), the result is 0.
    eroded_image = np.min(stack, axis=0)
    
    return eroded_image.astype(np.uint8)

def manual_dilation(image, kernel_size=5):
    h, w = image.shape
    pad_size = kernel_size // 2
    
    # Pad the image with zeros
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant', constant_values=0)
    
    shifted_views = []
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            roi = padded_image[i:i+h, j:j+w]
            shifted_views.append(roi)
            
    stack = np.stack(shifted_views, axis=0)
    
    # Dilation: Maximum value in the neighborhood
    # If any pixel in the neighborhood is 255 (white), the result is 255.
    dilated_image = np.max(stack, axis=0)
    
    return dilated_image.astype(np.uint8)

def manual_opening(image, kernel_size=5):
    img_eroded = manual_erosion(image, kernel_size)
    img_opened = manual_dilation(img_eroded, kernel_size)
    return img_opened

def manual_closing(image, kernel_size=2):
    img_dilated = manual_dilation(image, kernel_size)
    img_closed = manual_erosion(img_dilated, kernel_size)
    return img_closed

def main():
    args = parse_args()
    
    # Check input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
        
    # Prepare output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
        
    # Read Image
    # Read as grayscale
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not read image.")
        sys.exit(1)

    # Binarize Image
    # Ensure the image is strictly 0 and 255. 
    # Usually strictly > 0 or > 127 is treated as 255.
    _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    print(f"Processing image: {args.input}")
    print(f"Image shape: {bin_img.shape}")
    
    # Perform Operations
    # Kernel size is 3x3 as implied by typical "square" operations in such problems unless specified otherwise
    k_size = 3
    
    # Erosion
    img_C = manual_erosion(bin_img, kernel_size=k_size)
    path_C = os.path.join(args.output_dir, 'C_erosion.png')
    cv2.imwrite(path_C, img_C)
    print(f"(C) Saved Erosion result to: {path_C}")
    
    # Dilation
    img_D = manual_dilation(bin_img, kernel_size=k_size)
    path_D = os.path.join(args.output_dir, 'D_dilation.png')
    cv2.imwrite(path_D, img_D)
    print(f"(D) Saved Dilation result to: {path_D}")
    
    # Opening (Erosion -> Dilation)
    img_E = manual_opening(bin_img, kernel_size=k_size)
    path_E = os.path.join(args.output_dir, 'E_opening.png')
    cv2.imwrite(path_E, img_E)
    print(f"(E) Saved Opening result to: {path_E}")
    
    # Closing (Dilation -> Erosion)
    img_F = manual_closing(bin_img, kernel_size=k_size)
    path_F = os.path.join(args.output_dir, 'F_closing.png')
    cv2.imwrite(path_F, img_F)
    print(f"(F) Saved Closing result to: {path_F}")
    
    print("All operations completed successfully.")

if __name__ == "__main__":
    main()