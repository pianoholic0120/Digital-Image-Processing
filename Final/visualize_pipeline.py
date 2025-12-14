#!/usr/bin/env python3
"""
Visualize each step of the preprocessing pipeline
"""

import cv2
import numpy as np
import os
import sys
import argparse

def gamma_correction(image, gamma=2.2):
    """Apply gamma correction to linearize the image"""
    # Normalize to [0, 1]
    normalized = image.astype(np.float32) / 255.0
    # Apply gamma correction
    corrected = np.power(normalized, gamma)
    # Convert back to [0, 255]
    return (corrected * 255.0).astype(np.uint8)

def fixed_gain_exposure_compensation(image, target_mean=128.0):
    """Apply fixed gain exposure compensation"""
    # Convert to grayscale for mean calculation
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Calculate mean
    mean_luma = np.mean(gray)
    
    # Calculate fixed gain
    epsilon = 1e-6
    gain = target_mean / (mean_luma + epsilon)
    
    # Clamp gain to safe range [0.8, 1.2]
    gain = np.clip(gain, 0.8, 1.2)
    
    # Apply gain to all channels
    if len(image.shape) == 3:
        result = (image.astype(np.float32) * gain).clip(0, 255).astype(np.uint8)
    else:
        result = (image.astype(np.float32) * gain).clip(0, 255).astype(np.uint8)
    
    return result, gain

def grayscale_conversion(image):
    """Convert to grayscale using BT.709 weights"""
    if len(image.shape) == 3:
        # BT.709: Y = 0.2126*R + 0.7152*G + 0.0722*B
        b, g, r = cv2.split(image)
        gray = (0.2126 * r.astype(np.float32) + 
                0.7152 * g.astype(np.float32) + 
                0.0722 * b.astype(np.float32)).clip(0, 255).astype(np.uint8)
        return gray
    else:
        return image.copy()

def bilateral_filter_denoising(image, d=5, sigmaColor=20, sigmaSpace=20):
    """Apply bilateral filter for edge-preserving denoising"""
    return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

def load_pcalib(pcalib_file):
    """Load inverse camera response function from pcalib.txt (matching DSO implementation)"""
    with open(pcalib_file, 'r') as f:
        line = f.readline().strip()
        values = [float(x) for x in line.split()]
    
    G = np.array(values, dtype=np.float32)
    
    # Normalize G to [0, 255] range (matching DSO Undistort.cpp line 109)
    # G[i] = 255.0 * (G[i] - min) / (max - min)
    min_val = G[0]
    max_val = G[-1]
    if max_val > min_val:
        G_normalized = 255.0 * (G - min_val) / (max_val - min_val)
    else:
        G_normalized = G.copy()
    
    return G_normalized

def apply_photometric_undistortion(image, pcalib_lut, vignette_mask_inv=None):
    """
    Apply photometric undistortion (matching DSO PhotometricUndistorter::processFrame)
    Steps:
    1. Apply CRF: data[i] = G[image_in[i]]
    2. Apply vignette: data[i] *= vignetteMapInv[i] (if available)
    """
    h, w = image.shape[:2]
    result = np.zeros((h, w), dtype=np.float32)
    
    # Step 1: Apply CRF lookup table (matching DSO line 250)
    for i in range(h):
        for j in range(w):
            pixel_val = int(image[i, j])
            pixel_val = np.clip(pixel_val, 0, 255)
            result[i, j] = pcalib_lut[pixel_val]
    
    # Step 2: Apply vignette correction (matching DSO line 256)
    # data[i] *= vignetteMapInv[i]
    if vignette_mask_inv is not None:
        if vignette_mask_inv.shape[:2] != (h, w):
            vignette_resized = cv2.resize(vignette_mask_inv, (w, h))
        else:
            vignette_resized = vignette_mask_inv
        
        result = result * vignette_resized
    
    # Convert back to uint8 for display
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def load_vignette(vignette_file):
    """
    Load vignette mask from vignette.png (matching DSO implementation)
    Returns vignetteMapInv (inverse vignette map) for direct multiplication
    """
    vignette = cv2.imread(vignette_file, cv2.IMREAD_UNCHANGED)
    if vignette is None:
        return None
    
    h, w = vignette.shape[:2]
    
    # Read as 16-bit or 8-bit (matching DSO lines 135-136)
    if vignette.dtype == np.uint16:
        vignette_data = vignette.astype(np.float32)
    else:
        vignette_data = vignette.astype(np.float32)
    
    # Find maximum value (matching DSO lines 151-153, 169-171)
    maxV = np.max(vignette_data)
    
    if maxV > 0:
        # Normalize: vignetteMap[i] = vignette[i] / maxV (matching DSO lines 156, 174)
        vignette_map = vignette_data / maxV
        
        # Compute inverse: vignetteMapInv[i] = 1.0f / vignetteMap[i] (matching DSO line 189)
        vignette_map_inv = 1.0 / (vignette_map + 1e-10)  # Add small epsilon to avoid division by zero
        
        return vignette_map_inv
    else:
        return None

def load_camera_calibration(camera_file):
    """Load camera calibration parameters from camera.txt (matching DSO format)"""
    with open(camera_file, 'r') as f:
        lines = f.readlines()
    
    # Parse RadTan model
    # Format: RadTan fx fy cx cy k1 k2 p1 p2 [k3?]
    #         width height
    #         rectification_mode or output_calibration
    #         output_width output_height
    if len(lines) >= 2:
        model_line = lines[0].strip()
        if model_line.startswith('RadTan'):
            parts = model_line.split()
            fx = float(parts[1])
            fy = float(parts[2])
            cx = float(parts[3])
            cy = float(parts[4])
            k1 = float(parts[5])
            k2 = float(parts[6])
            p1 = float(parts[7])
            p2 = float(parts[8])
            
            size_line = lines[1].strip().split()
            width = int(size_line[0])
            height = int(size_line[1])
            
            # Check rectification mode (line 2)
            if len(lines) >= 3:
                rect_mode = lines[2].strip()
                if rect_mode == "none":
                    # Passthrough mode - no geometric undistortion
                    return None, None, None
            
            camera_matrix = np.array([[fx, 0, cx],
                                      [0, fy, cy],
                                      [0, 0, 1]], dtype=np.float32)
            dist_coeffs = np.array([k1, k2, p1, p2], dtype=np.float32)
            
            return camera_matrix, dist_coeffs, (width, height)
    
    return None, None, None

def apply_geometric_undistortion(image, camera_matrix, dist_coeffs):
    """Apply geometric undistortion"""
    if camera_matrix is None or dist_coeffs is None:
        return image.copy()
    
    h, w = image.shape[:2]
    
    # Get optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    
    # Undistort
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    return undistorted

def visualize_pipeline(video_file, frame_idx=0, output_dir="pipeline_visualization", 
                      camera_file=None, pcalib_file=None, vignette_file=None):
    """Visualize each step of the preprocessing pipeline"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {video_file}")
        return
    
    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Cannot read frame {frame_idx}")
        return
    
    print(f"Processing frame {frame_idx} from {video_file}")
    print(f"Original frame size: {frame.shape}")
    
    # Step 1: Raw
    step1_raw = frame.copy()
    cv2.imwrite(os.path.join(output_dir, "01_raw.png"), step1_raw)
    print("Step 1: Raw - saved")
    
    # Step 2: Gamma correction
    step2_gamma = gamma_correction(step1_raw, gamma=2.2)
    cv2.imwrite(os.path.join(output_dir, "02_gamma_correction.png"), step2_gamma)
    print("Step 2: Gamma correction - saved")
    
    # Step 3: Exposure compensation
    step3_exposure, gain = fixed_gain_exposure_compensation(step2_gamma)
    cv2.imwrite(os.path.join(output_dir, "03_exposure_compensation.png"), step3_exposure)
    print(f"Step 3: Exposure compensation (gain={gain:.4f}) - saved")
    
    # Step 4: Grayscale conversion
    step4_grayscale = grayscale_conversion(step3_exposure)
    cv2.imwrite(os.path.join(output_dir, "04_grayscale_conversion.png"), step4_grayscale)
    print("Step 4: Grayscale conversion - saved")
    
    # Step 5: Light denoising (bilateral filter)
    step5_denoised = bilateral_filter_denoising(step4_grayscale)
    cv2.imwrite(os.path.join(output_dir, "05_bilateral_denoising.png"), step5_denoised)
    print("Step 5: Bilateral filter denoising - saved")
    
    # Step 6: Undistortion (photometric + geometric)
    # Matching DSO implementation: photometric first, then geometric
    step6_undistorted = step5_denoised.copy()
    
    # Load vignette mask inverse (if available)
    vignette_mask_inv = None
    if vignette_file and os.path.exists(vignette_file):
        print(f"Loading vignette from: {vignette_file}")
        vignette_mask_inv = load_vignette(vignette_file)
        if vignette_mask_inv is not None:
            print("Loaded vignette mask (inverse)")
    
    # Apply photometric undistortion if pcalib.txt is available
    # This matches DSO PhotometricUndistorter::processFrame
    if pcalib_file and os.path.exists(pcalib_file):
        print(f"Loading pcalib from: {pcalib_file}")
        pcalib_lut = load_pcalib(pcalib_file)
        step6_undistorted = apply_photometric_undistortion(step6_undistorted, pcalib_lut, vignette_mask_inv)
        print("Applied photometric undistortion (CRF + Vignette)")
    
    # Apply geometric undistortion if camera.txt is available
    # This matches DSO Undistort::undistort (after photometric correction)
    if camera_file and os.path.exists(camera_file):
        print(f"Loading camera calibration from: {camera_file}")
        camera_matrix, dist_coeffs, size = load_camera_calibration(camera_file)
        if camera_matrix is not None:
            step6_undistorted = apply_geometric_undistortion(step6_undistorted, camera_matrix, dist_coeffs)
            print("Applied geometric undistortion")
    
    cv2.imwrite(os.path.join(output_dir, "06_undistortion.png"), step6_undistorted)
    print("Step 6: Undistortion (photometric + geometric) - saved")
    
    # Create a side-by-side comparison image
    create_comparison_image(output_dir, step1_raw, step2_gamma, step3_exposure, 
                           step4_grayscale, step5_denoised, step6_undistorted)
    
    print(f"\nAll steps saved to: {output_dir}/")
    print("Comparison image saved to: comparison.png")

def create_comparison_image(output_dir, raw, gamma, exposure, grayscale, denoised, undistorted):
    """Create a side-by-side comparison of all steps"""
    # Resize all images to same size for comparison
    h, w = raw.shape[:2]
    
    # Resize grayscale images to match
    if len(gamma.shape) == 2:
        gamma = cv2.cvtColor(gamma, cv2.COLOR_GRAY2BGR)
    if len(exposure.shape) == 2:
        exposure = cv2.cvtColor(exposure, cv2.COLOR_GRAY2BGR)
    if len(grayscale.shape) == 2:
        grayscale_colored = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
    else:
        grayscale_colored = grayscale
    if len(denoised.shape) == 2:
        denoised_colored = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    else:
        denoised_colored = denoised
    if len(undistorted.shape) == 2:
        undistorted_colored = cv2.cvtColor(undistorted, cv2.COLOR_GRAY2BGR)
    else:
        undistorted_colored = undistorted
    
    # Resize all to same dimensions
    target_size = (w, h)
    gamma_resized = cv2.resize(gamma, target_size)
    exposure_resized = cv2.resize(exposure, target_size)
    grayscale_resized = cv2.resize(grayscale_colored, target_size)
    denoised_resized = cv2.resize(denoised_colored, target_size)
    undistorted_resized = cv2.resize(undistorted_colored, target_size)
    
    # Create 2x3 grid
    top_row = np.hstack([raw, gamma_resized, exposure_resized])
    bottom_row = np.hstack([grayscale_resized, denoised_resized, undistorted_resized])
    comparison = np.vstack([top_row, bottom_row])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)
    
    labels = ["1. Raw", "2. Gamma", "3. Exposure", 
              "4. Grayscale", "5. Denoised", "6. Undistorted"]
    
    label_positions = [
        (10, 30), (w + 10, 30), (2*w + 10, 30),
        (10, h + 30), (w + 10, h + 30), (2*w + 10, h + 30)
    ]
    
    for i, (label, pos) in enumerate(zip(labels, label_positions)):
        row = i // 3
        col = i % 3
        x = pos[0] + col * w
        y = pos[1] + row * h
        cv2.putText(comparison, label, (x, y), font, font_scale, color, thickness)
    
    cv2.imwrite(os.path.join(output_dir, "comparison.png"), comparison)
    print("Comparison image created")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize preprocessing pipeline steps")
    parser.add_argument("--video", type=str, default="raw.mp4", help="Input video file")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to process")
    parser.add_argument("--output", type=str, default="pipeline_visualization", help="Output directory")
    parser.add_argument("--camera", type=str, help="Camera calibration file (camera.txt)")
    parser.add_argument("--pcalib", type=str, help="Photometric calibration file (pcalib.txt)")
    parser.add_argument("--vignette", type=str, help="Vignette mask file (vignette.png)")
    
    args = parser.parse_args()
    
    # Try to find calibration files in common locations
    if args.camera is None:
        # Try loop/camera.txt
        if os.path.exists("loop/camera.txt"):
            args.camera = "loop/camera.txt"
        elif os.path.exists("12_04_wall1/camera.txt"):
            args.camera = "12_04_wall1/camera.txt"
    
    if args.pcalib is None:
        # Try loop/pcalib.txt
        if os.path.exists("loop/pcalib.txt"):
            args.pcalib = "loop/pcalib.txt"
        elif os.path.exists("12_04_wall1/pcalib.txt"):
            args.pcalib = "12_04_wall1/pcalib.txt"
    
    if args.vignette is None:
        # Try loop/vignette.png
        if os.path.exists("loop/vignette.png"):
            args.vignette = "loop/vignette.png"
        elif os.path.exists("12_04_wall1/vignette.png"):
            args.vignette = "12_04_wall1/vignette.png"
    
    visualize_pipeline(args.video, args.frame, args.output, 
                      args.camera, args.pcalib, args.vignette)

