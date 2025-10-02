# Advanced Interpolation Technique for CFA Demosaicing
import cv2
import os
import numpy as np
import argparse
import math
import matplotlib.pyplot as plt

def load_bayer(path):
    raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return raw.astype(np.float32)

def save_rgb(path, rgb_image):
    rgb_image = np.clip(rgb_image, 0, 255)
    
    # Convert to uint8, which is the standard for image saving.
    rgb_image = rgb_image.astype(np.uint8)
    
    # Create the output directory if it does not exist.
    output_dir = os.path.dirname(path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save the image using Matplotlib as required.
    plt.imsave(path, rgb_image)

def convolve2d(image, kernel):
    # Get dimensions
    kh, kw = kernel.shape
    ih, iw = image.shape

    # Calculate padding size
    pad_h, pad_w = kh // 2, kw // 2

    # Create a padded image with reflection padding to handle borders
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    # Initialize the output image
    output = np.zeros_like(image, dtype=np.float32)

    # Perform convolution by iterating through each pixel
    for i in range(ih):
        for j in range(iw):
            # Extract the region of interest from the padded image
            region = padded_image[i:i+kh, j:j+kw]
            # Perform element-wise multiplication and sum
            output[i, j] = np.sum(region * kernel)

    return output

def stochastic_demosaicing(bayer):
    h, w = bayer.shape
    # Pad the CFA for easier neighborhood access at the borders
    CFA = np.pad(bayer, ((2, 2), (2, 2)), mode='reflect') 

    R_mask = np.zeros_like(bayer, dtype=bool); R_mask[0::2, 0::2] = True
    Gr_mask = np.zeros_like(bayer, dtype=bool); Gr_mask[0::2, 1::2] = True # G in Red rows
    Gb_mask = np.zeros_like(bayer, dtype=bool); Gb_mask[1::2, 0::2] = True # G in Blue rows
    G_mask = Gr_mask | Gb_mask
    B_mask = np.zeros_like(bayer, dtype=bool); B_mask[1::2, 1::2] = True
    
    G_final = bayer.copy()
    
    # Iterate over each pixel that needs interpolation (R and B locations)
    for i in range(h):
        for j in range(w):
            # Process only at R or B pixel locations
            if R_mask[i, j] or B_mask[i, j]:
                pi, pj = i + 2, j + 2 # Indices for the padded CFA

                # Calculate edge indicators in four directions (North, South, East, West)
                E_N = abs(CFA[pi-1, pj] - CFA[pi+1, pj]) + abs(CFA[pi, pj] - CFA[pi-2, pj])
                E_S = abs(CFA[pi+1, pj] - CFA[pi-1, pj]) + abs(CFA[pi, pj] - CFA[pi+2, pj])
                E_E = abs(CFA[pi, pj+1] - CFA[pi, pj-1]) + abs(CFA[pi, pj] - CFA[pi, pj+2])
                E_W = abs(CFA[pi, pj-1] - CFA[pi, pj+1]) + abs(CFA[pi, pj] - CFA[pi, pj-2])
                
                indicators = [E_N, E_S, E_E, E_W]
                
                # Calculate weights for each direction using a stochastic policy
                mu = np.mean(indicators)
                sigma_d = mu * (math.pi / 2)**0.5
                if sigma_d == 0: sigma_d = 1e-6 # Avoid division by zero

                weights = [math.erfc(ind / (sigma_d * math.sqrt(2))) for ind in indicators]
                
                weights_sum = sum(weights)
                if weights_sum == 0: # If all weights are zero, use equal weighting
                    weights = [0.25, 0.25, 0.25, 0.25]
                else:
                    weights = [w / weights_sum for w in weights]

                # Estimate Green value from each of the four directions
                G_N = (CFA[pi-1, pj] + CFA[pi+1, pj])/2 + (CFA[pi,pj] - CFA[pi-2, pj])/2
                G_S = (CFA[pi+1, pj] + CFA[pi-1, pj])/2 + (CFA[pi,pj] - CFA[pi+2, pj])/2
                G_E = (CFA[pi, pj+1] + CFA[pi, pj-1])/2 + (CFA[pi,pj] - CFA[pi, pj+2])/2
                G_W = (CFA[pi, pj-1] + CFA[pi, pj+1])/2 + (CFA[pi,pj] - CFA[pi, pj-2])/2
                
                estimates = [G_N, G_S, G_E, G_W]
                
                # Calculate the final interpolated G value as a weighted average
                G_final[i, j] = sum(w * e for w, e in zip(weights, estimates))

    R_final = np.zeros_like(bayer, dtype=np.float32)
    B_final = np.zeros_like(bayer, dtype=np.float32)

    # Calculate color differences (e.g., R-G) only where R and B are known
    R_minus_G = np.zeros_like(bayer, dtype=np.float32)
    B_minus_G = np.zeros_like(bayer, dtype=np.float32)
    R_minus_G[R_mask] = bayer[R_mask] - G_final[R_mask]
    B_minus_G[B_mask] = bayer[B_mask] - G_final[B_mask]
    
    # Pad the color difference images for interpolation
    R_m_G_pad = np.pad(R_minus_G, 1, 'reflect')
    B_m_G_pad = np.pad(B_minus_G, 1, 'reflect')
    
    # Interpolate R and B at G locations using directional interpolation
    for i in range(h):
        for j in range(w):
            pi, pj = i + 1, j + 1
            if Gr_mask[i,j]: # G in a Red row
                # Interpolate R horizontally, B vertically
                r_diff = (R_m_G_pad[pi, pj-1] + R_m_G_pad[pi, pj+1]) / 2.0
                b_diff = (B_m_G_pad[pi-1, pj] + B_m_G_pad[pi+1, pj]) / 2.0
                R_final[i,j] = G_final[i,j] + r_diff
                B_final[i,j] = G_final[i,j] + b_diff
            elif Gb_mask[i,j]: # G in a Blue row
                # Interpolate R vertically, B horizontally
                r_diff = (R_m_G_pad[pi-1, pj] + R_m_G_pad[pi+1, pj]) / 2.0
                b_diff = (B_m_G_pad[pi, pj-1] + B_m_G_pad[pi, pj+1]) / 2.0
                R_final[i,j] = G_final[i,j] + r_diff
                B_final[i,j] = G_final[i,j] + b_diff

    # Interpolate R at B locations and B at R locations (diagonally)
    kernel_diag = np.array([[0.25, 0, 0.25], [0, 0, 0], [0.25, 0, 0.25]])
    R_at_B_diff = convolve2d(R_minus_G, kernel_diag)
    B_at_R_diff = convolve2d(B_minus_G, kernel_diag)

    R_final[B_mask] = G_final[B_mask] + R_at_B_diff[B_mask]
    B_final[R_mask] = G_final[R_mask] + B_at_R_diff[B_mask]

    # Restore original known values from the Bayer pattern
    R_final[R_mask] = bayer[R_mask]
    B_final[B_mask] = bayer[B_mask]

    # Combine the three channels into a final RGB image
    rgb_final = np.stack([R_final, G_final, B_final], axis=-1)
    return rgb_final

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CFA Demosaicing using the Stochastic Interpolation method."
    )
    parser.add_argument(
        "--input_image", 
        type=str, 
        required=True, 
        help="Path to the input Bayer pattern image."
    )
    args = parser.parse_args()
    bayer_image = load_bayer(args.input_image)
    rgb_result = stochastic_demosaicing(bayer_image)
    input_file_name = os.path.basename(args.input_image)
    output_file_name = f"{os.path.splitext(input_file_name)[0]}.png"
    output_dir = "./output/p3/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, output_file_name)
    save_rgb(output_path, rgb_result)