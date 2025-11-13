import os
import numpy as np
import cv2
import argparse

# The complete, filtered backprojection (to obtain the reconstructed image f(x,y)) consists of the following steps:
# 1. Compute the 1-D Fourier transform of each projection
# 2. Multiply each Fourier transform by the filter function |ω| multiplied by Hamming window (c = 0.54)
# 3. Obtain the inverse 1-D Fourier transform of each resulting filtered transform
# 4. Integrate (sum) all the 1-D inverse transforms from Step 3

# The source image is of 600x600 pixels with black background and a central white square of 300x300 pixels
def generate_input_image():
    print("Generating input image...")
    image = np.zeros((600, 600), dtype=np.float32)
    # 300 - 150 = 150 # 300 + 150 = 450
    image[150:450, 150:450] = 1.0
    return image

# (Output 1) Generating fan projections of the rectangle image with Δa = Δb = 1°
# => converting each fan ray to the corresponding parallel ray using Eq. (5-133): p(nγ, mγ) = g(Dsin(nγ), (m + n)γ)
# => using the filtered backprojection approach developed earlier for parallel rays.
# (Output 2-4) show the results using 0.5°, 0.25°, and 0.125° increments of Δa and Δb. A Hamming window was used in all cases.
    
def generate_parallel_sinogram(image, thetas):
    height, width = image.shape
    center_x = (width - 1) / 2
    center_y = (height - 1) / 2

    diagonal = np.ceil(np.sqrt(height**2 + width**2))
    num_bins = int(diagonal)
    if num_bins % 2 == 0:
        num_bins += 1
    
    s_bins = np.linspace(-num_bins // 2, num_bins // 2, num_bins)
    
    sinogram = np.zeros((num_bins, len(thetas)))
    
    y_coords, x_coords = np.indices(image.shape)
    x_coords_centered = x_coords - center_x
    y_coords_centered = center_y - y_coords
    
    bin_edges = np.linspace(-num_bins // 2 - 0.5, num_bins // 2 + 0.5, num_bins + 1)
    
    for i, theta_deg in enumerate(thetas):
        # if (i+1) % (max(1, len(thetas) // 10)) == 0 or i == len(thetas) - 1:
            # print(f"    ... projection {i+1}/{len(thetas)} ({theta_deg:.2f} deg)")
            
        theta_rad = np.deg2rad(theta_deg)
        # s = x * cos(theta) + y * sin(theta)
        s_projection = x_coords_centered * np.cos(theta_rad) + y_coords_centered * np.sin(theta_rad)
        
        hist, _ = np.histogram(
            s_projection.ravel(),
            bins=bin_edges,
            weights=image.ravel()
        )
        sinogram[:, i] = hist
        
    return sinogram, s_bins

def filter_sinogram(sinogram):
    # print("  Filtering sinogram...")
    num_bins, num_thetas = sinogram.shape
    
    projections_fft = np.fft.fft(sinogram, axis=0)
    projections_fft_shifted = np.fft.fftshift(projections_fft, axes=0)
    
    omega_shifted = np.fft.fftshift(np.fft.fftfreq(num_bins))
    
    filter_ramp = np.abs(omega_shifted)
    
    filter_hamming = np.hamming(num_bins)
    
    filter_combined = filter_ramp * filter_hamming
    
    filter_2d = np.tile(filter_combined[:, np.newaxis], (1, num_thetas))
    filtered_fft_shifted = projections_fft_shifted * filter_2d
    
    filtered_fft = np.fft.ifftshift(filtered_fft_shifted, axes=0)
    filtered_sinogram = np.fft.ifft(filtered_fft, axis=0)
    
    return filtered_sinogram.real

def backproject(filtered_sinogram, thetas, s_bins, output_shape=(600, 600)):
    # print("  Back-projecting...")
    num_bins, num_thetas = filtered_sinogram.shape
    reconstructed_image = np.zeros(output_shape, dtype=np.float32)
    
    height, width = output_shape
    center_x = (width - 1) / 2
    center_y = (height - 1) / 2
    
    y_coords, x_coords = np.indices(output_shape)
    x_coords_centered = x_coords - center_x
    y_coords_centered = center_y - y_coords
    
    for i, theta_deg in enumerate(thetas):
        # if (i+1) % (max(1, len(thetas) // 10)) == 0 or i == len(thetas) - 1:
            # print(f"    ... back-projecting {i+1}/{len(thetas)} ({theta_deg:.2f} deg)")
            
        theta_rad = np.deg2rad(theta_deg)
        
        projection = filtered_sinogram[:, i]
        
        s_projection = x_coords_centered * np.cos(theta_rad) + y_coords_centered * np.sin(theta_rad)
        
        interpolated_projection = np.interp(
            s_projection,
            s_bins,
            projection
        )
        
        reconstructed_image += interpolated_projection
        
    delta_theta_rad = np.deg2rad(np.abs(thetas[1] - thetas[0]))
    reconstructed_image *= delta_theta_rad
    
    return reconstructed_image

def filtered_backprojection(input_image, output_path):
    os.makedirs(output_path, exist_ok=True)
    
    input_8bit = (np.clip(input_image, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_path, "input_image_600x600.png"), input_8bit)

    angle_increments = [1.0, 0.5, 0.25, 0.125]
    
    # Step 1: Generate all reconstructions
    reconstructed_images = {}
    for delta_angle in angle_increments:
        print(f"\n--- Processing: angle increment = {delta_angle}° ---")
        
        thetas = np.arange(0, 180, delta_angle)
        
        sinogram, s_bins = generate_parallel_sinogram(input_image, thetas)
        filtered_sinogram = filter_sinogram(sinogram)
        reconstructed_image = backproject(filtered_sinogram, thetas, s_bins, input_image.shape)
        
        rec_min = np.min(reconstructed_image)
        rec_max = np.max(reconstructed_image)
        if rec_max > rec_min:
            reconstructed_normalized = (reconstructed_image - rec_min) / (rec_max - rec_min)
        else:
            reconstructed_normalized = np.zeros_like(reconstructed_image)
            
        reconstructed_8bit = (reconstructed_normalized * 255).astype(np.uint8)
        
        filename = f"reconstruction_{delta_angle}deg.png"
        filepath = os.path.join(output_path, filename)
        cv2.imwrite(filepath, reconstructed_8bit)
        print(f"  Saved reconstruction to: {filepath}")
        
        # Store normalized reconstruction for sinogram generation
        reconstructed_images[delta_angle] = reconstructed_normalized
    
    # Step 2: Generate sinograms from original and reconstructed images with consistent dimensions
    print(f"\n--- Generating sinograms from original and reconstructed images ---")
    
    # Calculate fixed dimensions for all sinograms
    # Use the finest angle increment for consistent height
    finest_angle = min(angle_increments)
    target_num_angles = int(180 / finest_angle)  # Maximum number of angles
    
    # Calculate fixed num_bins based on input image size
    diag = np.ceil(np.sqrt(input_image.shape[0]**2 + input_image.shape[1]**2))
    target_num_bins = int(diag)
    if target_num_bins % 2 == 0:
        target_num_bins += 1
    
    print(f"  Target sinogram dimensions: width={target_num_bins} (ρ), height={target_num_angles} (θ)")
    
    # Use the finest angle increment for all sinograms to ensure consistent height
    thetas = np.arange(0, 180, finest_angle)
    
    # Generate sinogram from original input image
    print(f"\n  Generating sinogram for original input image...")
    sinogram_original, s_bins = generate_parallel_sinogram(input_image, thetas)
    
    # Normalize original sinogram
    sino_min = np.min(sinogram_original)
    sino_max = np.max(sinogram_original)
    if sino_max > sino_min:
        sinogram_normalized = (sinogram_original - sino_min) / (sino_max - sino_min)
    else:
        sinogram_normalized = np.zeros_like(sinogram_original)
    
    sinogram_8bit = (sinogram_normalized * 255).astype(np.uint8)
    
    # Transpose: (num_bins, num_thetas) -> (num_thetas, num_bins)
    # This makes rows = angles (θ), columns = distances (ρ)
    sinogram_display = sinogram_8bit.T
    
    # Verify dimensions match target
    assert sinogram_display.shape[1] == target_num_bins, f"Width mismatch: {sinogram_display.shape[1]} != {target_num_bins}"
    assert sinogram_display.shape[0] == target_num_angles, f"Height mismatch: {sinogram_display.shape[0]} != {target_num_angles}"
    
    sino_filename = "sinogram_original.png"
    sino_filepath = os.path.join(output_path, sino_filename)
    cv2.imwrite(sino_filepath, sinogram_display)
    print(f"  Saved sinogram to: {sino_filepath} (shape: {sinogram_display.shape})")
    
    # Generate sinograms from reconstructed images
    for delta_angle in angle_increments:
        print(f"\n  Generating sinogram for reconstruction_{delta_angle}deg...")
        
        # Generate sinogram from reconstructed image
        sinogram, s_bins = generate_parallel_sinogram(reconstructed_images[delta_angle], thetas)
        
        # Normalize sinogram
        sino_min = np.min(sinogram)
        sino_max = np.max(sinogram)
        if sino_max > sino_min:
            sinogram_normalized = (sinogram - sino_min) / (sino_max - sino_min)
        else:
            sinogram_normalized = np.zeros_like(sinogram)
        
        sinogram_8bit = (sinogram_normalized * 255).astype(np.uint8)
        
        # Transpose: (num_bins, num_thetas) -> (num_thetas, num_bins)
        # This makes rows = angles (θ), columns = distances (ρ)
        sinogram_display = sinogram_8bit.T
        
        # Verify dimensions match target
        assert sinogram_display.shape[1] == target_num_bins, f"Width mismatch: {sinogram_display.shape[1]} != {target_num_bins}"
        assert sinogram_display.shape[0] == target_num_angles, f"Height mismatch: {sinogram_display.shape[0]} != {target_num_angles}"
        
        sino_filename = f"sinogram_{delta_angle}deg.png"
        sino_filepath = os.path.join(output_path, sino_filename)
        cv2.imwrite(sino_filepath, sinogram_display)
        print(f"  Saved sinogram to: {sino_filepath} (shape: {sinogram_display.shape})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="./output_part_a")
    args = parser.parse_args()
    input_image = generate_input_image()
    filtered_backprojection(input_image, args.output_path)
    print("\n--- Finished ---")