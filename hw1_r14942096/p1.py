# Simple Interpolation Technique for CFA Demosaicing
import cv2
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse

def load_bayer(path):
    raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return raw.astype(np.float32)

def simple_interpolation(bayer):
    h, w = bayer.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    # Assign Bayer pattern (RGGB as illustrated in the course material)
    
    # RGGB pattern is as follows: 
    # ============
    # R G R G R G
    # G B G B G B
    # R G R G R G
    # G B G B G B
    # ============

    # Red pixels at (0,0), (0,2), (2,0), (2,2), ...
    rgb[0::2, 0::2, 0] = bayer[0::2, 0::2]  # Red
    # Green pixels at (0,1), (1,0), (1,2), (2,1), ...
    rgb[0::2, 1::2, 1] = bayer[0::2, 1::2]  # Green
    rgb[1::2, 0::2, 1] = bayer[1::2, 0::2]  # Green
    # Blue pixels at (1,1), (1,3), (3,1), (3,3), ...
    rgb[1::2, 1::2, 2] = bayer[1::2, 1::2]  # Blue

    # Interpolation for missing values with proper boundary handling
    
    # Green at Red positions (0::2, 0::2) - average available horizontal/vertical neighbors
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            neighbors = []
            if i > 0:  # top neighbor
                neighbors.append(rgb[i-1, j, 1])
            if i < h-1:  # bottom neighbor
                neighbors.append(rgb[i+1, j, 1])
            if j > 0:  # left neighbor
                neighbors.append(rgb[i, j-1, 1])
            if j < w-1:  # right neighbor
                neighbors.append(rgb[i, j+1, 1])
            
            if neighbors:
                rgb[i, j, 1] = np.mean(neighbors)

    # Green at Blue positions (1::2, 1::2) - average available horizontal/vertical neighbors
    for i in range(1, h, 2):
        for j in range(1, w, 2):
            neighbors = []
            if i > 0:  # top neighbor
                neighbors.append(rgb[i-1, j, 1])
            if i < h-1:  # bottom neighbor
                neighbors.append(rgb[i+1, j, 1])
            if j > 0:  # left neighbor
                neighbors.append(rgb[i, j-1, 1])
            if j < w-1:  # right neighbor
                neighbors.append(rgb[i, j+1, 1])
            
            if neighbors:
                rgb[i, j, 1] = np.mean(neighbors)

    # Red at Green positions - average available horizontal neighbors
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

    # Red at Blue positions - average available diagonal neighbors
    for i in range(1, h, 2):
        for j in range(1, w, 2):
            neighbors = []
            if i > 0 and j > 0:  # top-left
                neighbors.append(rgb[i-1, j-1, 0])
            if i > 0 and j < w-1:  # top-right
                neighbors.append(rgb[i-1, j+1, 0])
            if i < h-1 and j > 0:  # bottom-left
                neighbors.append(rgb[i+1, j-1, 0])
            if i < h-1 and j < w-1:  # bottom-right
                neighbors.append(rgb[i+1, j+1, 0])
            
            if neighbors:
                rgb[i, j, 0] = np.mean(neighbors)

    # Blue at Green positions - average available horizontal neighbors
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

    # Blue at Red positions - average available diagonal neighbors
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            neighbors = []
            if i > 0 and j > 0:  # top-left
                neighbors.append(rgb[i-1, j-1, 2])
            if i > 0 and j < w-1:  # top-right
                neighbors.append(rgb[i-1, j+1, 2])
            if i < h-1 and j > 0:  # bottom-left
                neighbors.append(rgb[i+1, j-1, 2])
            if i < h-1 and j < w-1:  # bottom-right
                neighbors.append(rgb[i+1, j+1, 2])
            
            if neighbors:
                rgb[i, j, 2] = np.mean(neighbors)

    return np.clip(rgb, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, help="Path to the input image")
    args = parser.parse_args()
    path = args.input_image
    bayer = load_bayer(path)
    rgb = simple_interpolation(bayer)
    output_file_name = path.split("/")[-1].split(".")[0]
    output_path = "./output/p1/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.imsave(output_path + output_file_name + ".png", rgb.astype(np.uint8))