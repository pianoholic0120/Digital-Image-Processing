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
    
    # --- Step 0: Separate the known R, G, B values ---
    R = np.zeros((h, w), dtype=np.float32)
    G = np.zeros((h, w), dtype=np.float32)
    B = np.zeros((h, w), dtype=np.float32)
    
    R[0::2, 0::2] = bayer[0::2, 0::2]
    G[0::2, 1::2] = bayer[0::2, 1::2]
    G[1::2, 0::2] = bayer[1::2, 0::2]
    B[1::2, 1::2] = bayer[1::2, 1::2]

    # --- Step 1: Interpolate Green channel with edge-aware logic (Your original logic was fine here) ---
    for i in range(h):
        for j in range(w):
            # We only need to interpolate G at R or B locations
            is_R_loc = (i % 2 == 0 and j % 2 == 0)
            is_B_loc = (i % 2 == 1 and j % 2 == 1)
            
            if is_R_loc or is_B_loc:
                g_top = G[i-1, j] if i > 0 else 0
                g_bottom = G[i+1, j] if i < h-1 else 0
                g_left = G[i, j-1] if j > 0 else 0
                g_right = G[i, j+1] if j < w-1 else 0
                
                # Default to a simple average for boundary pixels
                valid_neighbors = [n for n in [g_top, g_bottom, g_left, g_right] if n > 0]
                if len(valid_neighbors) < 4:
                    G[i, j] = np.mean(valid_neighbors) if valid_neighbors else 0
                    continue

                d_v = abs(g_top - g_bottom)
                d_h = abs(g_left - g_right)
                
                if d_v < d_h:
                    G[i, j] = (g_top + g_bottom) / 2
                elif d_h < d_v:
                    G[i, j] = (g_left + g_right) / 2
                else: # d_v == d_h
                    G[i, j] = (g_top + g_bottom + g_left + g_right) / 4

    # --- Step 2: Calculate Color Differences (R-G, B-G) where available ---
    R_minus_G = R - G  # Will be non-zero only at R locations
    B_minus_G = B - G  # Will be non-zero only at B locations

    # --- Step 3: Interpolate the Color Differences ---
    # Interpolate R-G at G locations
    for i in range(h):
        for j in range(w):
            is_G_on_R_row = (i % 2 == 0 and j % 2 == 1)
            is_G_on_B_row = (i % 2 == 1 and j % 2 == 0)
            
            if is_G_on_R_row: # Horizontal interpolation for R-G
                left = R_minus_G[i, j-1] if j > 0 else 0
                right = R_minus_G[i, j+1] if j < w-1 else 0
                R_minus_G[i, j] = (left + right) / 2
            elif is_G_on_B_row: # Vertical interpolation for R-G
                top = R_minus_G[i-1, j] if i > 0 else 0
                bottom = R_minus_G[i+1, j] if i < h-1 else 0
                R_minus_G[i, j] = (top + bottom) / 2
    
    # Interpolate B-G at G locations (similar logic)
    for i in range(h):
        for j in range(w):
            is_G_on_R_row = (i % 2 == 0 and j % 2 == 1)
            is_G_on_B_row = (i % 2 == 1 and j % 2 == 0)
            
            if is_G_on_R_row: # Vertical interpolation for B-G
                top = B_minus_G[i-1, j] if i > 0 else 0
                bottom = B_minus_G[i+1, j] if i < h-1 else 0
                B_minus_G[i, j] = (top + bottom) / 2
            elif is_G_on_B_row: # Horizontal interpolation for B-G
                left = B_minus_G[i, j-1] if j > 0 else 0
                right = B_minus_G[i, j+1] if j < w-1 else 0
                B_minus_G[i, j] = (left + right) / 2
    
    # Interpolate R-G at B locations & B-G at R locations (average of 4 diagonal neighbors)
    for i in range(h):
        for j in range(w):
            if i % 2 == 1 and j % 2 == 1: # At B location, interpolate R-G
                tl = R_minus_G[i-1, j-1] if i>0 and j>0 else 0
                tr = R_minus_G[i-1, j+1] if i>0 and j<w-1 else 0
                bl = R_minus_G[i+1, j-1] if i<h-1 and j>0 else 0
                br = R_minus_G[i+1, j+1] if i<h-1 and j<w-1 else 0
                R_minus_G[i,j] = (tl+tr+bl+br)/4
            elif i % 2 == 0 and j % 2 == 0: # At R location, interpolate B-G
                tl = B_minus_G[i-1, j-1] if i>0 and j>0 else 0
                tr = B_minus_G[i-1, j+1] if i>0 and j<w-1 else 0
                bl = B_minus_G[i+1, j-1] if i<h-1 and j>0 else 0
                br = B_minus_G[i+1, j+1] if i<h-1 and j<w-1 else 0
                B_minus_G[i,j] = (tl+tr+bl+br)/4
                
    # --- Step 4: Reconstruct full R and B channels ---
    final_R = R_minus_G + G
    final_B = B_minus_G + G
    final_G = G # G is already complete

    # Combine channels into a final RGB image
    rgb = cv2.merge([final_R, final_G, final_B])
    
    return np.clip(rgb, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, help="Path to the input image")
    parser.add_argument("--threshold", type=float, default=100.0, help="Threshold for edge detection (default: 10.0)")
    args = parser.parse_args()
    path = args.input_image
    thres = args.threshold
    bayer = load_bayer(path)
    rgb = edge_aware_interpolation(bayer, thres)
    output_file_name = path.split("/")[-1].split(".")[0]
    output_path = "./images/p2_"+str(thres)+"thre/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.imsave(output_path + output_file_name + ".png", rgb.astype(np.uint8))