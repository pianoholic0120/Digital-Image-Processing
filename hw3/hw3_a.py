import os
import numpy as np
import cv2
import argparse

# Show the Fourier spectrum of the test image “keyboard.”
def sub_one(image_path, output_dir):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image at {image_path}")
    os.makedirs(output_dir, exist_ok=True)

    F = np.fft.fft2(img.astype(np.float32))
    F_shift = np.fft.fftshift(F)
    mag = np.log1p(np.abs(F_shift))

    mag -= mag.min()
    mag /= (mag.max() + 1e-8)
    mag_uint8 = (mag * 255).astype(np.uint8)

    cv2.imwrite(os.path.join(output_dir, 'a1.png'), mag_uint8)
    print("Subproblem 1: Fourier spectrum saved to output/1.png")

# Enforce odd symmetry on the kernel. Show the kernel. (Vertical Sobel kernel)
def sub_two(image_path, output_dir):
    # Using top-left anchoring.
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image at {image_path}")
    os.makedirs(output_dir, exist_ok=True)

    # 3x3 vertical Sobel (odd-symmetric)
    h3 = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    # Processing the spatial filter as per Example 4.15:
    # Convert h to the smallest size that satisfies the odd symmetry requirement
    # by adding a leading row and column of 0's -> make it 4x4 (top row/col zeros)
    h4 = np.zeros((4, 4), dtype=np.float32)
    h4[1:, 1:] = h3

    # Visualize kernel as image for saving
    vis = h4.copy()
    vis_min, vis_max = vis.min(), vis.max()
    if vis_max - vis_min > 1e-8:
        vis = (vis - vis_min) / (vis_max - vis_min)
    else:
        vis = np.zeros_like(vis)
    vis = (vis * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, 'a2.png'), vis)
    print("Subproblem 2: Padded 4x4 kernel saved to output/2.png")

# Show the result of frequency domain filtering of the test image using the vertical Sober kernel.
def sub_three(image, output_path):
    # Frequency-domain filtering with vertical Sobel, enforcing odd symmetry (4x4 with leading zeros)
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image at {image}")
    os.makedirs(output_path, exist_ok=True)

    img_f = img.astype(np.float32)

    # Build 4x4 kernel with leading zeros then pad to image size at (0,0)
    h3 = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)
    h4 = np.zeros((4, 4), dtype=np.float32)
    h4[1:, 1:] = h3

    H = np.zeros_like(img_f)
    kh, kw = h4.shape
    H[:kh, :kw] = h4

    # FFT multiply
    F = np.fft.fft2(img_f)
    Hf = np.fft.fft2(H)
    
    # To perform CORRELATION (equivalent to cv2.filter2D), 
    # we must use the complex conjugate of the kernel's spectrum.
    # G = F * Hf         <-- This is CONVOLUTION
    G = F * np.conj(Hf)  # <-- This is CORRELATION
    
    g = np.fft.ifft2(G).real

    # Normalize to uint8 for saving
    g_norm = g - g.min()
    if g_norm.max() > 1e-8:
        g_norm = g_norm / g_norm.max()
    g_u8 = (g_norm * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_path, 'a3.png'), g_u8)
    print("Subproblem 3: Frequency-domain filtering (Correlation) saved to output/3.png")

# Compare the result in subproblem 3 with the result of space-domain filtering.
def sub_four(image, output_path):
    # Spatial-domain filtering (Correlation) with enforced odd symmetry using 4x4 kernel (leading zeros)
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image at {image}")
    os.makedirs(output_path, exist_ok=True)

    img_f = img.astype(np.float32)

    # 3x3 vertical Sobel and convert to 4x4 with leading zero row/column
    h3 = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)
    h4 = np.zeros((4, 4), dtype=np.float32)
    h4[1:, 1:] = h3

    # Use anchor (0,0) so spatial filtering aligns with the frequency-domain top-left placement
    # cv2.filter2D performs CORRELATION
    resp = cv2.filter2D(img_f, ddepth=-1, kernel=h4, anchor=(0, 0), borderType=cv2.BORDER_REPLICATE)

    # Normalize for visualization
    resp = resp - resp.min()
    if resp.max() > 1e-8:
        resp = resp / resp.max()
    resp_u8 = (resp * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_path, 'a4.png'), resp_u8)
    print("Subproblem 4: Spatial-domain filtering (Correlation) saved to output/4.png")
    print("Compare 3.png and 4.png. They should now be identical.")


# Show the result of frequency domain filtering without enforcing odd symmetry on the kernel.
def sub_five(image, output_path):
    # Frequency-domain filtering WITHOUT enforcing odd symmetry (use raw 3x3 kernel)
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image at {image}")
    os.makedirs(output_path, exist_ok=True)

    img_f = img.astype(np.float32)

    h3 = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    # Naively pad 3x3 kernel at (0,0) without the leading zero row/col adjustment
    H = np.zeros_like(img_f)
    kh, kw = h3.shape
    H[:kh, :kw] = h3

    F = np.fft.fft2(img_f)
    Hf = np.fft.fft2(H)

    # G = F * Hf
    G = F * np.conj(Hf)
    
    g = np.fft.ifft2(G).real

    g_norm = g - g.min()
    if g_norm.max() > 1e-8:
        g_norm = g_norm / g_norm.max()
    g_u8 = (g_norm * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_path, 'a5.png'), g_u8)
    print("Subproblem 5: Frequency-domain filtering (naive 3x3) saved to output/5.png")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./images/keyboard.tif', help="Path to the input image (e.g., ./images/keyboard.tif)")
    parser.add_argument('--output', type=str, default='./output_a/', help="Directory to save output images")
    parser.add_argument('--subproblem', type=int, required=True, choices=[1, 2, 3, 4, 5], help="Subproblem number to run (1-5)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input image not found at {args.input}")
        print("Please make sure 'keyboard.tif' is in the 'images' directory or specify the correct path with --input.")
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
    elif args.subproblem == 5:
        sub_five(args.input, args.output)
    else:
        # argparse 'choices' should prevent this, but good to have
        print("Invalid subproblem")

if __name__ == '__main__':
    main()
