import os
import numpy as np
import cv2

def create_comparison_grid(image_paths, labels, output_path, cols=2, spacing=10):
    images = []
    for path in image_paths:
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not read {path}")
                continue
            images.append(img)
        else:
            print(f"Warning: File not found: {path}")
    
    if len(images) == 0:
        print("Error: No valid images to display")
        return
    
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    
    resized = []
    for img in images:
        resized_img = cv2.resize(img, (max_w, max_h), interpolation=cv2.INTER_CUBIC)
        resized.append(resized_img)
    
    rows = (len(resized) + cols - 1) // cols
    
    grid_h = rows * max_h + (rows - 1) * spacing
    grid_w = cols * max_w + (cols - 1) * spacing
    
    label_height = 30
    grid = np.ones((grid_h + rows * label_height, grid_w, 3), dtype=np.uint8) * 255
    
    for idx, img in enumerate(resized):
        row = idx // cols
        col = idx % cols
        
        y_start = row * (max_h + spacing) + row * label_height
        x_start = col * (max_w + spacing)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        grid[y_start:y_start+max_h, x_start:x_start+max_w] = img_rgb
        
        if idx < len(labels):
            label_y = y_start + max_h + 5
            cv2.putText(grid, labels[idx], (x_start + 10, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.imwrite(output_path, grid)
    print(f"Comparison grid saved to: {output_path}")

def create_side_by_side(img1_path, img2_path, label1, label2, output_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        print(f"Error: Could not read images")
        return
    
    h = max(img1.shape[0], img2.shape[0])
    scale1 = h / img1.shape[0]
    scale2 = h / img2.shape[0]
    w1 = int(img1.shape[1] * scale1)
    w2 = int(img2.shape[1] * scale2)
    
    img1_resized = cv2.resize(img1, (w1, h))
    img2_resized = cv2.resize(img2, (w2, h))
    
    label_height = 40
    spacing = 20
    combined = np.ones((h + label_height, w1 + w2 + spacing, 3), dtype=np.uint8) * 255
    
    combined[0:h, 0:w1] = cv2.cvtColor(img1_resized, cv2.COLOR_GRAY2RGB)
    combined[0:h, w1+spacing:w1+spacing+w2] = cv2.cvtColor(img2_resized, cv2.COLOR_GRAY2RGB)
    
    cv2.putText(combined, label1, (10, h + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(combined, label2, (w1 + spacing + 10, h + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    cv2.imwrite(output_path, combined)
    print(f"Side-by-side comparison saved to: {output_path}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['grid', 'side'], default='grid')
    parser.add_argument('--images', nargs='+', help='Image paths')
    parser.add_argument('--labels', nargs='+', help='Labels for images')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--cols', type=int, default=2)
    
    args = parser.parse_args()
    
    if args.mode == 'grid':
        if not args.images or not args.labels:
            print("Error: --images and --labels required for grid mode")
        else:
            create_comparison_grid(args.images, args.labels, args.output, cols=args.cols)
    elif args.mode == 'side':
        if len(args.images) != 2 or len(args.labels) != 2:
            print("Error: Need exactly 2 images and 2 labels for side mode")
        else:
            create_side_by_side(args.images[0], args.images[1], 
                              args.labels[0], args.labels[1], args.output)

