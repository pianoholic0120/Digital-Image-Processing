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

    # Assign Bayer pattern (RGGB)
    rgb[0::2, 0::2, 0] = bayer[0::2, 0::2]  # Red
    rgb[0::2, 1::2, 1] = bayer[0::2, 1::2]  # Green
    rgb[1::2, 0::2, 1] = bayer[1::2, 0::2]  # Green
    rgb[1::2, 1::2, 2] = bayer[1::2, 1::2]  # Blue

    # --- Green channel interpolation (This part was correct) ---
    # Green at Red positions (average 4 neighbors)
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            neighbors = []
            if i > 0: neighbors.append(bayer[i-1, j])
            if i < h-1: neighbors.append(bayer[i+1, j])
            if j > 0: neighbors.append(bayer[i, j-1])
            if j < w-1: neighbors.append(bayer[i, j+1])
            if neighbors: rgb[i, j, 1] = np.mean(neighbors)

    # Green at Blue positions (average 4 neighbors)
    for i in range(1, h, 2):
        for j in range(1, w, 2):
            neighbors = []
            if i > 0: neighbors.append(bayer[i-1, j])
            if i < h-1: neighbors.append(bayer[i+1, j])
            if j > 0: neighbors.append(bayer[i, j-1])
            if j < w-1: neighbors.append(bayer[i, j+1])
            if neighbors: rgb[i, j, 1] = np.mean(neighbors)

    # --- Red channel interpolation ---
    # Red at Green positions on Red rows (average horizontal neighbors)
    for i in range(0, h, 2):
        for j in range(1, w, 2):
            neighbors = []
            if j > 0: neighbors.append(rgb[i, j-1, 0])
            if j < w-1: neighbors.append(rgb[i, j+1, 0])
            if neighbors: rgb[i, j, 0] = np.mean(neighbors)

    # Red at Green positions on Blue rows (average vertical neighbors)
    for i in range(1, h, 2):
        for j in range(0, w, 2):
            neighbors = []
            if i > 0: neighbors.append(rgb[i-1, j, 0])      # Use vertical neighbor
            if i < h-1: neighbors.append(rgb[i+1, j, 0])  # Use vertical neighbor
            if neighbors: rgb[i, j, 0] = np.mean(neighbors)

    # Red at Blue positions (average diagonal neighbors)
    for i in range(1, h, 2):
        for j in range(1, w, 2):
            neighbors = []
            if i > 0 and j > 0: neighbors.append(rgb[i-1, j-1, 0])
            if i > 0 and j < w-1: neighbors.append(rgb[i-1, j+1, 0])
            if i < h-1 and j > 0: neighbors.append(rgb[i+1, j-1, 0])
            if i < h-1 and j < w-1: neighbors.append(rgb[i+1, j+1, 0])
            if neighbors: rgb[i, j, 0] = np.mean(neighbors)

    # --- Blue channel interpolation ---
    # Blue at Green positions on Red rows (average vertical neighbors) 
    for i in range(0, h, 2):
        for j in range(1, w, 2):
            neighbors = []
            if i > 0: neighbors.append(rgb[i-1, j, 2])      # Use vertical neighbor
            if i < h-1: neighbors.append(rgb[i+1, j, 2])  # Use vertical neighbor
            if neighbors: rgb[i, j, 2] = np.mean(neighbors)

    # Blue at Green positions on Blue rows (average horizontal neighbors)
    for i in range(1, h, 2):
        for j in range(0, w, 2):
            neighbors = []
            if j > 0: neighbors.append(rgb[i, j-1, 2])
            if j < w-1: neighbors.append(rgb[i, j+1, 2])
            if neighbors: rgb[i, j, 2] = np.mean(neighbors)

    # Blue at Red positions (average diagonal neighbors)
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            neighbors = []
            if i > 0 and j > 0: neighbors.append(rgb[i-1, j-1, 2])
            if i > 0 and j < w-1: neighbors.append(rgb[i-1, j+1, 2])
            if i < h-1 and j > 0: neighbors.append(rgb[i+1, j-1, 2])
            if i < h-1 and j < w-1: neighbors.append(rgb[i+1, j+1, 2])
            if neighbors: rgb[i, j, 2] = np.mean(neighbors)

    return np.clip(rgb, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, help="Path to the input image")
    args = parser.parse_args()
    path = args.input_image
    bayer = load_bayer(path)
    rgb = simple_interpolation(bayer)
    output_file_name = path.split("/")[-1].split(".")[0]
    output_path = "./images/p1/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.imsave(output_path + output_file_name + ".png", rgb.astype(np.uint8))