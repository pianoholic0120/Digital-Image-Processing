import cv2
import numpy as np
import os

def manual_erosion(image, kernel_size):
    h, w = image.shape
    pad_size = kernel_size // 2
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant', constant_values=0)
    shifted_views = []
    for i in range(kernel_size):
        for j in range(kernel_size):
            roi = padded_image[i:i+h, j:j+w]
            shifted_views.append(roi)
    stack = np.stack(shifted_views, axis=0)
    eroded_image = np.min(stack, axis=0)
    return eroded_image.astype(np.uint8)

def manual_dilation(image, kernel_size):
    h, w = image.shape
    pad_size = kernel_size // 2
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant', constant_values=0)
    shifted_views = []
    for i in range(kernel_size):
        for j in range(kernel_size):
            roi = padded_image[i:i+h, j:j+w]
            shifted_views.append(roi)
    stack = np.stack(shifted_views, axis=0)
    dilated_image = np.max(stack, axis=0)
    return dilated_image.astype(np.uint8)

def manual_opening(image, kernel_size):
    img_eroded = manual_erosion(image, kernel_size)
    img_opened = manual_dilation(img_eroded, kernel_size)
    return img_opened

def manual_closing(image, kernel_size):
    img_dilated = manual_dilation(image, kernel_size)
    img_closed = manual_erosion(img_dilated, kernel_size)
    return img_closed

def create_comparison_image(images, labels, rows, cols):
    """Create a grid comparison image"""
    h, w = images[0].shape
    result = np.zeros((rows * h, cols * w), dtype=np.uint8)
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // cols
        col = idx % cols
        result[row*h:(row+1)*h, col*w:(col+1)*w] = img
        
        # Add text label (simple version - just save with filename)
    return result

# Read image
img = cv2.imread('noisy_rectangle.png', cv2.IMREAD_GRAYSCALE)
_, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Create output directory
os.makedirs('results/morphology_experiments', exist_ok=True)

# Test different kernel sizes
kernel_sizes = [3, 5, 7, 9, 11]

print("Generating experiments with different kernel sizes...")

# 1. Generate comparison images for different kernel sizes
for k_size in kernel_sizes:
    erosion = manual_erosion(bin_img, k_size)
    dilation = manual_dilation(bin_img, k_size)
    opening = manual_opening(bin_img, k_size)
    closing = manual_closing(bin_img, k_size)
    
    # Save individual results
    cv2.imwrite(f'results/morphology_experiments/k{k_size}_erosion.png', erosion)
    cv2.imwrite(f'results/morphology_experiments/k{k_size}_dilation.png', dilation)
    cv2.imwrite(f'results/morphology_experiments/k{k_size}_opening.png', opening)
    cv2.imwrite(f'results/morphology_experiments/k{k_size}_closing.png', closing)
    
    # Create side-by-side comparison (2x3 grid)
    h, w = bin_img.shape
    comparison = np.zeros((2*h, 3*w), dtype=np.uint8)
    
    # Row 1
    comparison[0:h, 0:w] = bin_img
    comparison[0:h, w:2*w] = erosion
    comparison[0:h, 2*w:3*w] = dilation
    
    # Row 2
    comparison[h:2*h, 0:w] = opening
    comparison[h:2*h, w:2*w] = closing
    diff_opening = cv2.absdiff(bin_img, opening)
    comparison[h:2*h, 2*w:3*w] = diff_opening
    
    cv2.imwrite(f'results/morphology_experiments/k{k_size}_comparison.png', comparison)
    
    print(f"  Kernel {k_size}x{k_size}: Done")

# 2. Find interesting regions to zoom in
# Find white noise points
inverted = 255 - bin_img
num_labels, labels = cv2.connectedComponents(inverted)
print(f"\nFound {num_labels-1} white noise components")

# Find a small white noise component
white_noise_sizes = []
for label in range(1, num_labels):
    size = np.sum(labels == label)
    white_noise_sizes.append((label, size))

white_noise_sizes.sort(key=lambda x: x[1])

# Find a region with interesting noise (around 50-200 pixels)
target_label = None
for label, size in white_noise_sizes:
    if 50 <= size <= 200:
        target_label = label
        break

if target_label:
    # Get bounding box
    y_coords, x_coords = np.where(labels == target_label)
    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()
    
    # Add padding
    padding = 30
    y_min = max(0, y_min - padding)
    y_max = min(bin_img.shape[0], y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(bin_img.shape[1], x_max + padding)
    
    # Extract region
    region = bin_img[y_min:y_max, x_min:x_max]
    
    # Apply operations on this region
    for k_size in [3, 5, 7]:
        region_erosion = manual_erosion(region, k_size)
        region_dilation = manual_dilation(region, k_size)
        region_opening = manual_opening(region, k_size)
        region_closing = manual_closing(region, k_size)
        
        # Create zoomed comparison (2x3 grid)
        h_reg, w_reg = region.shape
        comparison_reg = np.zeros((2*h_reg, 3*w_reg), dtype=np.uint8)
        
        # Row 1
        comparison_reg[0:h_reg, 0:w_reg] = region
        comparison_reg[0:h_reg, w_reg:2*w_reg] = region_erosion
        comparison_reg[0:h_reg, 2*w_reg:3*w_reg] = region_dilation
        
        # Row 2
        comparison_reg[h_reg:2*h_reg, 0:w_reg] = region_opening
        comparison_reg[h_reg:2*h_reg, w_reg:2*w_reg] = region_closing
        
        # Overlay: create RGB image
        overlay = np.zeros((h_reg, w_reg, 3), dtype=np.uint8)
        overlay[:, :, 0] = region  # Red
        overlay[:, :, 1] = region_opening  # Green
        overlay[:, :, 2] = region_closing  # Blue
        comparison_reg[h_reg:2*h_reg, 2*w_reg:3*w_reg] = cv2.cvtColor(overlay, cv2.COLOR_RGB2GRAY)
        
        cv2.imwrite(f'results/morphology_experiments/zoomed_region_k{k_size}.png', comparison_reg)
    
    print(f"\nZoomed region saved: ({x_min}, {y_min}) to ({x_max}, {y_max})")

# 3. Find black holes inside white rectangle
num_labels_black, labels_black = cv2.connectedComponents(bin_img)
print(f"\nFound {num_labels_black-1} black hole components")

# Find a small black hole
black_hole_sizes = []
for label in range(1, num_labels_black):
    size = np.sum(labels_black == label)
    black_hole_sizes.append((label, size))

black_hole_sizes.sort(key=lambda x: x[1])

# Find a region with interesting black hole
target_label_black = None
for label, size in black_hole_sizes:
    if 20 <= size <= 100:
        target_label_black = label
        break

if target_label_black:
    # Get bounding box
    y_coords, x_coords = np.where(labels_black == target_label_black)
    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()
    
    # Add padding
    padding = 30
    y_min = max(0, y_min - padding)
    y_max = min(bin_img.shape[0], y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(bin_img.shape[1], x_max + padding)
    
    # Extract region
    region_black = bin_img[y_min:y_max, x_min:x_max]
    
    # Apply operations
    for k_size in [3, 5, 7]:
        region_erosion = manual_erosion(region_black, k_size)
        region_dilation = manual_dilation(region_black, k_size)
        region_opening = manual_opening(region_black, k_size)
        region_closing = manual_closing(region_black, k_size)
        
        # Create zoomed comparison
        h_reg, w_reg = region_black.shape
        comparison_reg = np.zeros((2*h_reg, 3*w_reg), dtype=np.uint8)
        
        # Row 1
        comparison_reg[0:h_reg, 0:w_reg] = region_black
        comparison_reg[0:h_reg, w_reg:2*w_reg] = region_erosion
        comparison_reg[0:h_reg, 2*w_reg:3*w_reg] = region_dilation
        
        # Row 2
        comparison_reg[h_reg:2*h_reg, 0:w_reg] = region_opening
        comparison_reg[h_reg:2*h_reg, w_reg:2*w_reg] = region_closing
        
        # Overlay
        overlay = np.zeros((h_reg, w_reg, 3), dtype=np.uint8)
        overlay[:, :, 0] = region_black
        overlay[:, :, 1] = region_opening
        overlay[:, :, 2] = region_closing
        comparison_reg[h_reg:2*h_reg, 2*w_reg:3*w_reg] = cv2.cvtColor(overlay, cv2.COLOR_RGB2GRAY)
        
        cv2.imwrite(f'results/morphology_experiments/zoomed_blackhole_k{k_size}.png', comparison_reg)
    
    print(f"\nZoomed black hole region saved: ({x_min}, {y_min}) to ({x_max}, {y_max})")

# 4. Create side-by-side comparison of all kernel sizes
# Create a large comparison image
h, w = bin_img.shape
num_kernels = len(kernel_sizes)
comparison_all = np.zeros((4*h, (num_kernels+1)*w), dtype=np.uint8)

# First column: original (repeated 4 times)
for i in range(4):
    comparison_all[i*h:(i+1)*h, 0:w] = bin_img

# Other columns: different kernel sizes
for col_idx, k_size in enumerate(kernel_sizes, 1):
    erosion = manual_erosion(bin_img, k_size)
    dilation = manual_dilation(bin_img, k_size)
    opening = manual_opening(bin_img, k_size)
    closing = manual_closing(bin_img, k_size)
    
    comparison_all[0:h, col_idx*w:(col_idx+1)*w] = erosion
    comparison_all[h:2*h, col_idx*w:(col_idx+1)*w] = dilation
    comparison_all[2*h:3*h, col_idx*w:(col_idx+1)*w] = opening
    comparison_all[3*h:4*h, col_idx*w:(col_idx+1)*w] = closing

cv2.imwrite('results/morphology_experiments/all_kernels_comparison.png', comparison_all)

# 5. Statistics table
print("\nGenerating statistics...")
stats = []
for k_size in kernel_sizes:
    erosion = manual_erosion(bin_img, k_size)
    dilation = manual_dilation(bin_img, k_size)
    opening = manual_opening(bin_img, k_size)
    closing = manual_closing(bin_img, k_size)
    
    original_white = np.sum(bin_img == 255)
    erosion_white = np.sum(erosion == 255)
    dilation_white = np.sum(dilation == 255)
    opening_white = np.sum(opening == 255)
    closing_white = np.sum(closing == 255)
    
    stats.append({
        'kernel_size': k_size,
        'original_white': original_white,
        'erosion_white': erosion_white,
        'dilation_white': dilation_white,
        'opening_white': opening_white,
        'closing_white': closing_white,
        'erosion_removed': original_white - erosion_white,
        'dilation_added': dilation_white - original_white,
        'opening_removed': original_white - opening_white,
        'closing_added': closing_white - original_white
    })

# Save statistics
with open('results/morphology_experiments/statistics.txt', 'w') as f:
    f.write("Morphological Operations Statistics\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"{'Kernel':<10} {'Original':<12} {'Erosion':<12} {'Dilation':<12} {'Opening':<12} {'Closing':<12}\n")
    f.write(f"{'Size':<10} {'White Pixels':<12} {'White Pixels':<12} {'White Pixels':<12} {'White Pixels':<12} {'White Pixels':<12}\n")
    f.write("-" * 80 + "\n")
    for s in stats:
        f.write(f"{s['kernel_size']:<10} {s['original_white']:<12} {s['erosion_white']:<12} {s['dilation_white']:<12} {s['opening_white']:<12} {s['closing_white']:<12}\n")
    f.write("\n" + "=" * 80 + "\n")
    f.write("Changes from Original:\n")
    f.write(f"{'Kernel':<10} {'Erosion':<15} {'Dilation':<15} {'Opening':<15} {'Closing':<15}\n")
    f.write(f"{'Size':<10} {'Removed':<15} {'Added':<15} {'Removed':<15} {'Added':<15}\n")
    f.write("-" * 80 + "\n")
    for s in stats:
        f.write(f"{s['kernel_size']:<10} {s['erosion_removed']:<15} {s['dilation_added']:<15} {s['opening_removed']:<15} {s['closing_added']:<15}\n")

print("\nAll experiments completed!")
print("Results saved in results/morphology_experiments/")
