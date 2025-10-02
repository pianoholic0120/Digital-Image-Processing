# Simple Edge-Aware Interpolation Technique for CFA Demosaicing
import cv2
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse

def load_bayer(path):
    raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return raw.astype(np.float32)

def edge_aware_interpolation(bayer, thres=10):
    h, w = bayer.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    # RGGB pattern assignment
    rgb[0::2, 0::2, 0] = bayer[0::2, 0::2]  # Red
    rgb[0::2, 1::2, 1] = bayer[0::2, 1::2]  # Green
    rgb[1::2, 0::2, 1] = bayer[1::2, 0::2]  # Green
    rgb[1::2, 1::2, 2] = bayer[1::2, 1::2]  # Blue

    # Edge-aware interpolation for Green channel
    G = rgb[:, :, 1]
    
    # Green at Red positions (0::2, 0::2)
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            if G[i, j] == 0:  # need interpolation
                # Get neighbors: G2(top), G4(left), G6(right), G8(bottom)
                neighbors = []
                if i > 0:  # G2 (top)
                    neighbors.append(('top', G[i-1, j]))
                if j > 0:  # G4 (left)
                    neighbors.append(('left', G[i, j-1]))
                if j < w-1:  # G6 (right)
                    neighbors.append(('right', G[i, j+1]))
                if i < h-1:  # G8 (bottom)
                    neighbors.append(('bottom', G[i+1, j]))
                
                if len(neighbors) >= 2:
                    # Calculate differences
                    d_v = 0  # vertical difference (G2 - G8)
                    d_h = 0  # horizontal difference (G4 - G6)
                    
                    top_val = None
                    bottom_val = None
                    left_val = None
                    right_val = None
                    
                    for direction, value in neighbors:
                        if direction == 'top':
                            top_val = value
                        elif direction == 'bottom':
                            bottom_val = value
                        elif direction == 'left':
                            left_val = value
                        elif direction == 'right':
                            right_val = value
                    
                    # Calculate differences if we have the pairs
                    if top_val is not None and bottom_val is not None:
                        d_v = abs(top_val - bottom_val)
                    if left_val is not None and right_val is not None:
                        d_h = abs(left_val - right_val)
                    
                    # Apply edge-aware logic
                    if d_v < thres and d_h < thres:
                        # All neighbors are similar - use all available
                        G[i, j] = np.mean([val for _, val in neighbors])
                    elif d_v > thres:
                        # Vertical edge - use horizontal neighbors
                        if left_val is not None and right_val is not None:
                            G[i, j] = (left_val + right_val) / 2
                        else:
                            G[i, j] = np.mean([val for _, val in neighbors])
                    else:
                        # Horizontal edge - use vertical neighbors
                        if top_val is not None and bottom_val is not None:
                            G[i, j] = (top_val + bottom_val) / 2
                        else:
                            G[i, j] = np.mean([val for _, val in neighbors])

    # Green at Blue positions (1::2, 1::2)
    for i in range(1, h, 2):
        for j in range(1, w, 2):
            if G[i, j] == 0:  # need interpolation
                # Get neighbors: G2(top), G4(left), G6(right), G8(bottom)
                neighbors = []
                if i > 0:  # G2 (top)
                    neighbors.append(('top', G[i-1, j]))
                if j > 0:  # G4 (left)
                    neighbors.append(('left', G[i, j-1]))
                if j < w-1:  # G6 (right)
                    neighbors.append(('right', G[i, j+1]))
                if i < h-1:  # G8 (bottom)
                    neighbors.append(('bottom', G[i+1, j]))
                
                if len(neighbors) >= 2:
                    # Calculate differences
                    d_v = 0  # vertical difference (G2 - G8)
                    d_h = 0  # horizontal difference (G4 - G6)
                    
                    top_val = None
                    bottom_val = None
                    left_val = None
                    right_val = None
                    
                    for direction, value in neighbors:
                        if direction == 'top':
                            top_val = value
                        elif direction == 'bottom':
                            bottom_val = value
                        elif direction == 'left':
                            left_val = value
                        elif direction == 'right':
                            right_val = value
                    
                    # Calculate differences if we have the pairs
                    if top_val is not None and bottom_val is not None:
                        d_v = abs(top_val - bottom_val)
                    if left_val is not None and right_val is not None:
                        d_h = abs(left_val - right_val)
                    
                    # Apply edge-aware logic
                    if d_v < thres and d_h < thres:
                        # All neighbors are similar - use all available
                        G[i, j] = np.mean([val for _, val in neighbors])
                    elif d_v > thres:
                        # Vertical edge - use horizontal neighbors
                        if left_val is not None and right_val is not None:
                            G[i, j] = (left_val + right_val) / 2
                        else:
                            G[i, j] = np.mean([val for _, val in neighbors])
                    else:
                        # Horizontal edge - use vertical neighbors
                        if top_val is not None and bottom_val is not None:
                            G[i, j] = (top_val + bottom_val) / 2
                        else:
                            G[i, j] = np.mean([val for _, val in neighbors])

    # Edge-aware interpolation for Red and Blue channels
    
    # Red at Green positions - edge-aware horizontal interpolation
    for i in range(0, h, 2):
        for j in range(1, w, 2):
            neighbors = []
            if j > 0:  # left neighbor
                neighbors.append(rgb[i, j-1, 0])
            if j < w-1:  # right neighbor
                neighbors.append(rgb[i, j+1, 0])
            if neighbors:
                rgb[i, j, 0] = np.mean(neighbors)

    for i in range(1, h, 2):
        for j in range(0, w, 2):
            neighbors = []
            if j > 0:  # left neighbor
                neighbors.append(rgb[i, j-1, 0])
            if j < w-1:  # right neighbor
                neighbors.append(rgb[i, j+1, 0])
            if neighbors:
                rgb[i, j, 0] = np.mean(neighbors)

    # Red at Blue positions - edge-aware diagonal interpolation
    for i in range(1, h, 2):
        for j in range(1, w, 2):
            if rgb[i, j, 0] == 0:  # need interpolation
                # Get diagonal neighbors
                neighbors = []
                if i > 0 and j > 0:  # top-left
                    neighbors.append(('top-left', rgb[i-1, j-1, 0]))
                if i > 0 and j < w-1:  # top-right
                    neighbors.append(('top-right', rgb[i-1, j+1, 0]))
                if i < h-1 and j > 0:  # bottom-left
                    neighbors.append(('bottom-left', rgb[i+1, j-1, 0]))
                if i < h-1 and j < w-1:  # bottom-right
                    neighbors.append(('bottom-right', rgb[i+1, j+1, 0]))
                
                if len(neighbors) >= 2:
                    # Calculate differences for diagonal neighbors
                    d_v = 0  # top-left vs bottom-right
                    d_h = 0  # bottom-left vs top-right
                    
                    top_left = None
                    top_right = None
                    bottom_left = None
                    bottom_right = None
                    
                    for direction, value in neighbors:
                        if direction == 'top-left':
                            top_left = value
                        elif direction == 'top-right':
                            top_right = value
                        elif direction == 'bottom-left':
                            bottom_left = value
                        elif direction == 'bottom-right':
                            bottom_right = value
                    
                    # Calculate differences
                    if top_left is not None and bottom_right is not None:
                        d_v = abs(top_left - bottom_right)  # top-left vs bottom-right
                    if bottom_left is not None and top_right is not None:
                        d_h = abs(bottom_left - top_right)  # bottom-left vs top-right
                    
                    # Apply edge-aware logic
                    if d_v < thres and d_h < thres:
                        # All neighbors are similar - use all available
                        rgb[i, j, 0] = np.mean([val for _, val in neighbors])
                    elif d_v > thres:
                        # Diagonal edge (top-left vs bottom-right) - use other diagonal
                        if bottom_left is not None and top_right is not None:
                            rgb[i, j, 0] = (bottom_left + top_right) / 2
                        else:
                            rgb[i, j, 0] = np.mean([val for _, val in neighbors])
                    else:
                        # Diagonal edge (bottom-left vs top-right) - use other diagonal
                        if top_left is not None and bottom_right is not None:
                            rgb[i, j, 0] = (top_left + bottom_right) / 2
                        else:
                            rgb[i, j, 0] = np.mean([val for _, val in neighbors])

    # Blue at Green positions - edge-aware horizontal interpolation
    for i in range(0, h, 2):
        for j in range(1, w, 2):
            neighbors = []
            if j > 0:  # left neighbor
                neighbors.append(rgb[i, j-1, 2])
            if j < w-1:  # right neighbor
                neighbors.append(rgb[i, j+1, 2])
            if neighbors:
                rgb[i, j, 2] = np.mean(neighbors)

    for i in range(1, h, 2):
        for j in range(0, w, 2):
            neighbors = []
            if j > 0:  # left neighbor
                neighbors.append(rgb[i, j-1, 2])
            if j < w-1:  # right neighbor
                neighbors.append(rgb[i, j+1, 2])
            if neighbors:
                rgb[i, j, 2] = np.mean(neighbors)

    # Blue at Red positions - edge-aware diagonal interpolation
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            if rgb[i, j, 2] == 0:  # need interpolation
                # Get diagonal neighbors
                neighbors = []
                if i > 0 and j > 0:  # top-left
                    neighbors.append(('top-left', rgb[i-1, j-1, 2]))
                if i > 0 and j < w-1:  # top-right
                    neighbors.append(('top-right', rgb[i-1, j+1, 2]))
                if i < h-1 and j > 0:  # bottom-left
                    neighbors.append(('bottom-left', rgb[i+1, j-1, 2]))
                if i < h-1 and j < w-1:  # bottom-right
                    neighbors.append(('bottom-right', rgb[i+1, j+1, 2]))
                
                if len(neighbors) >= 2:
                    # Calculate differences for diagonal neighbors
                    d_v = 0  # top-left vs bottom-right
                    d_h = 0  # bottom-left vs top-right
                    
                    top_left = None
                    top_right = None
                    bottom_left = None
                    bottom_right = None
                    
                    for direction, value in neighbors:
                        if direction == 'top-left':
                            top_left = value
                        elif direction == 'top-right':
                            top_right = value
                        elif direction == 'bottom-left':
                            bottom_left = value
                        elif direction == 'bottom-right':
                            bottom_right = value
                    
                    # Calculate differences
                    if top_left is not None and bottom_right is not None:
                        d_v = abs(top_left - bottom_right)  # top-left vs bottom-right
                    if bottom_left is not None and top_right is not None:
                        d_h = abs(bottom_left - top_right)  # bottom-left vs top-right
                    
                    # Apply edge-aware logic
                    if d_v < thres and d_h < thres:
                        # All neighbors are similar - use all available
                        rgb[i, j, 2] = np.mean([val for _, val in neighbors])
                    elif d_v > thres:
                        # Diagonal edge (top-left vs bottom-right) - use other diagonal
                        if bottom_left is not None and top_right is not None:
                            rgb[i, j, 2] = (bottom_left + top_right) / 2
                        else:
                            rgb[i, j, 2] = np.mean([val for _, val in neighbors])
                    else:
                        # Diagonal edge (bottom-left vs top-right) - use other diagonal
                        if top_left is not None and bottom_right is not None:
                            rgb[i, j, 2] = (top_left + bottom_right) / 2
                        else:
                            rgb[i, j, 2] = np.mean([val for _, val in neighbors])

    return np.clip(rgb, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, help="Path to the input image")
    parser.add_argument("--threshold", type=float, default=10.0, help="Threshold for edge detection (default: 10.0)")
    args = parser.parse_args()
    path = args.input_image
    thres = args.threshold
    bayer = load_bayer(path)
    rgb = edge_aware_interpolation(bayer, thres)
    output_file_name = path.split("/")[-1].split(".")[0]
    output_path = "./output/p2_50thre/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.imsave(output_path + output_file_name + ".png", rgb.astype(np.uint8))