import cv2
import numpy as np
import os
import json
from PIL import Image, ImageDraw, ImageFont

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

def create_text_image(text, width, height, font_size=40):
    """Create an image with text"""
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((width - text_width) // 2, (height - text_height) // 2)
    draw.text(position, text, fill='black', font=font)
    return np.array(img)

def create_comparison_grid(images, labels, rows, cols, cell_h, cell_w, row_labels=None, col_labels=None):
    """Create a grid of images with labels"""
    # Add extra space for row and column labels if provided
    row_label_w = 100 if row_labels else 0
    col_label_h = 50 if col_labels else 0
    
    result = np.ones((rows * cell_h + col_label_h, cols * cell_w + row_label_w), dtype=np.uint8) * 255
    
    # Add column labels (operation names) at the top
    if col_labels:
        for col in range(cols):
            label_img = create_text_image(col_labels[col], cell_w, col_label_h, font_size=28)
            label_gray = cv2.cvtColor(label_img, cv2.COLOR_RGB2GRAY)
            result[0:col_label_h, row_label_w+col*cell_w:row_label_w+(col+1)*cell_w] = label_gray
    
    # Add row labels (kernel sizes) on the left
    if row_labels:
        for row in range(rows):
            label_img = create_text_image(row_labels[row], row_label_w, cell_h, font_size=24)
            label_gray = cv2.cvtColor(label_img, cv2.COLOR_RGB2GRAY)
            result[col_label_h+row*cell_h:col_label_h+(row+1)*cell_h, 0:row_label_w] = label_gray
    
    # Place images
    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // cols
        col = idx % cols
        
        # Resize image to fit cell
        img_resized = cv2.resize(img, (cell_w - 20, cell_h - 60))
        h, w = img_resized.shape
        
        # Place image (accounting for labels)
        y_offset = col_label_h + 50
        x_offset = row_label_w + 10
        result[y_offset+row*cell_h:y_offset+row*cell_h+h, x_offset+col*cell_w:x_offset+col*cell_w+w] = img_resized
    
    return result

# Read image
img = cv2.imread('noisy_rectangle.png', cv2.IMREAD_GRAYSCALE)
_, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Create output directory
os.makedirs('results/comprehensive_experiments', exist_ok=True)
os.makedirs('results/comprehensive_experiments/figures', exist_ok=True)

# Test kernel sizes as specified
kernel_sizes = [3, 5, 9, 15, 20, 30, 40, 50, 70, 80]

print("Generating comprehensive experiments...")

# Statistics collection
stats = {
    'kernel_sizes': kernel_sizes,
    'erosion': [],
    'dilation': [],
    'opening': [],
    'closing': []
}

original_white = np.sum(bin_img == 255)
original_total = bin_img.size

# 1. Generate results for all kernel sizes
for k_size in kernel_sizes:
    print(f"Processing kernel {k_size}x{k_size}...")
    
    erosion = manual_erosion(bin_img, k_size)
    dilation = manual_dilation(bin_img, k_size)
    opening = manual_opening(bin_img, k_size)
    closing = manual_closing(bin_img, k_size)
    
    # Calculate statistics
    erosion_white = np.sum(erosion == 255)
    dilation_white = np.sum(dilation == 255)
    opening_white = np.sum(opening == 255)
    closing_white = np.sum(closing == 255)
    
    stats['erosion'].append({
        'white_pixels': int(erosion_white),
        'removed': int(original_white - erosion_white),
        'removal_rate': float((original_white - erosion_white) / original_white * 100)
    })
    
    stats['dilation'].append({
        'white_pixels': int(dilation_white),
        'added': int(dilation_white - original_white),
        'addition_rate': float((dilation_white - original_white) / original_white * 100)
    })
    
    stats['opening'].append({
        'white_pixels': int(opening_white),
        'removed': int(original_white - opening_white),
        'removal_rate': float((original_white - opening_white) / original_white * 100)
    })
    
    stats['closing'].append({
        'white_pixels': int(closing_white),
        'added': int(closing_white - original_white),
        'addition_rate': float((closing_white - original_white) / original_white * 100)
    })
    
    # Save individual results for key kernel sizes
    if k_size in [3, 5, 9, 15, 30, 50, 80]:
        cv2.imwrite(f'results/comprehensive_experiments/k{k_size}_erosion.png', erosion)
        cv2.imwrite(f'results/comprehensive_experiments/k{k_size}_dilation.png', dilation)
        cv2.imwrite(f'results/comprehensive_experiments/k{k_size}_opening.png', opening)
        cv2.imwrite(f'results/comprehensive_experiments/k{k_size}_closing.png', closing)

# Save statistics
with open('results/comprehensive_experiments/statistics.json', 'w') as f:
    json.dump(stats, f, indent=2)

# 2. Create ablation study visualization
# New layout: columns = operations (Erosion, Dilation, Opening, Closing), rows = Original + kernel sizes
ablation_kernels = [3, 9, 15, 30, 50, 80]
cell_h, cell_w = 300, 300
rows = len(ablation_kernels) + 1  # Original + kernel sizes
cols = 4  # Four operations

ablation_images = []
ablation_labels = []  # Not used in new version, but kept for compatibility
row_labels = ['Original'] + [f'{k}×{k}' for k in ablation_kernels]
col_labels = ['Erosion', 'Dilation', 'Opening', 'Closing']

# First row: Original (repeated 4 times for 4 operations)
for i in range(4):
    ablation_images.append(bin_img)
    ablation_labels.append('')

# Other rows: different kernel sizes, each row has 4 operations
for k_size in ablation_kernels:
    erosion = manual_erosion(bin_img, k_size)
    dilation = manual_dilation(bin_img, k_size)
    opening = manual_opening(bin_img, k_size)
    closing = manual_closing(bin_img, k_size)
    
    ablation_images.extend([erosion, dilation, opening, closing])
    ablation_labels.extend(['', '', '', ''])

ablation_grid = create_comparison_grid(ablation_images, ablation_labels, rows, cols, cell_h, cell_w, 
                                       row_labels=row_labels, col_labels=col_labels)
cv2.imwrite('results/comprehensive_experiments/figures/ablation_study.png', ablation_grid)

# 3. Find and analyze random elements
inverted = 255 - bin_img
num_labels_white, labels_white = cv2.connectedComponents(inverted)
num_labels_black, labels_black = cv2.connectedComponents(bin_img)

print(f"\nFound {num_labels_white-1} white noise components")
print(f"Found {num_labels_black-1} black hole components")

# Analyze component sizes
white_sizes = []
for label in range(1, num_labels_white):
    size = np.sum(labels_white == label)
    white_sizes.append(size)

black_sizes = []
for label in range(1, num_labels_black):
    size = np.sum(labels_black == label)
    black_sizes.append(size)

# Save size statistics
size_stats = {
    'white_noise': {
        'count': len(white_sizes),
        'min': int(np.min(white_sizes)) if white_sizes else 0,
        'max': int(np.max(white_sizes)) if white_sizes else 0,
        'mean': float(np.mean(white_sizes)) if white_sizes else 0,
        'median': float(np.median(white_sizes)) if white_sizes else 0,
        'std': float(np.std(white_sizes)) if white_sizes else 0
    },
    'black_holes': {
        'count': len(black_sizes),
        'min': int(np.min(black_sizes)) if black_sizes else 0,
        'max': int(np.max(black_sizes)) if black_sizes else 0,
        'mean': float(np.mean(black_sizes)) if black_sizes else 0,
        'median': float(np.median(black_sizes)) if black_sizes else 0,
        'std': float(np.std(black_sizes)) if black_sizes else 0
    }
}

with open('results/comprehensive_experiments/component_statistics.json', 'w') as f:
    json.dump(size_stats, f, indent=2)

# 4. Create zoomed region analysis
target_label_white = None
for label in range(1, num_labels_white):
    size = np.sum(labels_white == label)
    if 50 <= size <= 200:
        target_label_white = label
        break

if target_label_white:
    y_coords, x_coords = np.where(labels_white == target_label_white)
    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()
    
    padding = 40
    y_min = max(0, y_min - padding)
    y_max = min(bin_img.shape[0], y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(bin_img.shape[1], x_max + padding)
    
    region = bin_img[y_min:y_max, x_min:x_max]
    
    # Create zoomed comparison for key kernels
    zoom_kernels = [3, 9, 15, 30, 50]
    zoom_images = []
    zoom_labels = []
    
    for k_size in zoom_kernels:
        region_erosion = manual_erosion(region, k_size)
        region_dilation = manual_dilation(region, k_size)
        region_opening = manual_opening(region, k_size)
        region_closing = manual_closing(region, k_size)
        
        zoom_images.extend([region, region_erosion, region_dilation, region_opening, region_closing])
        if k_size == zoom_kernels[0]:
            zoom_labels.extend(['Original', 'Erosion', 'Dilation', 'Opening', 'Closing'])
        else:
            zoom_labels.extend([f'{k_size}×{k_size}', '', '', '', ''])
    
    zoom_grid = create_comparison_grid(zoom_images, zoom_labels, len(zoom_kernels), 5, 300, 300)
    cv2.imwrite('results/comprehensive_experiments/figures/zoomed_analysis.png', zoom_grid)

# 5. Create statistics table image
# Generate a simple text-based statistics visualization
stats_text = f"""Morphological Operations Statistics
{'='*60}
Kernel Size | Erosion Removed | Dilation Added | Opening Removed | Closing Added
{'-'*60}
"""
for i, k_size in enumerate(kernel_sizes):
    stats_text += f"{k_size:10d} | {stats['erosion'][i]['removed']:14d} | {stats['dilation'][i]['added']:13d} | {stats['opening'][i]['removed']:14d} | {stats['closing'][i]['added']:13d}\n"

with open('results/comprehensive_experiments/statistics_table.txt', 'w') as f:
    f.write(stats_text)

print("\nAll experiments completed!")
print("Results saved in results/comprehensive_experiments/")
