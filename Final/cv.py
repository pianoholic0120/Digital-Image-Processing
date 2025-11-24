import cv2
import numpy as np
from utils.usb_baseline_pipeline import USBBaselinePipeline
from utils.build import build
import os 
import argparse as arg
import sys

def load_calibration():
    calib_path = "./calib.npz"
    if not os.path.exists(calib_path):
        print(f"Warning: Calibration file '{calib_path}' not found.")
        return None, None, None, None

    data = np.load(calib_path)
    camera_matrix = data.get("camera_matrix", None) 
    dist_coeffs = data.get("dist_coeffs", None)
    vignette_mask = data.get("vignette_mask", None)
    crf_lut = data.get("crf_lut", None)

    return camera_matrix, dist_coeffs, vignette_mask, crf_lut


def main():
    build()
    parser = arg.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./hall_with_intrinsics/origin/images")
    parser.add_argument("--output_path", type=str, default="./hall_with_intrinsics/improved/images")
    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = args.output_path

    if not os.path.exists(input_path):
        print(f"Error: Input path '{input_path}' does not exist.")
        return

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output directory: {output_path}")

    camera_matrix, dist_coeffs, vignette_mask, crf_lut = load_calibration()
    
    pipeline = USBBaselinePipeline(
        gamma=2.2,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        vignette_mask=vignette_mask,
        crf_lut=crf_lut,
        exposure_smooth_alpha=0.1,
        brightness_boost=0.2,        # Increase brightness (0.0-1.0 range)
        contrast_enhancement=1.2,     # Increase contrast (1.0 = no change, >1.0 = more contrast)
        use_clahe=False,                # Enable CLAHE for local contrast enhancement
        clahe_clip_limit=2.0,          # CLAHE clip limit (higher = more contrast)
        clahe_tile_grid_size=(8, 8)    # CLAHE tile size
    )

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = sorted([f for f in os.listdir(input_path) if f.lower().endswith(valid_extensions)])

    print(f"Found {len(image_files)} images in {input_path}")

    for i, filename in enumerate(image_files):
        full_input_path = os.path.join(input_path, filename)
        full_output_path = os.path.join(output_path, filename)

        frame = cv2.imread(full_input_path)
        
        if frame is None:
            print(f"Warning: Could not read image {filename}. Skipping.")
            continue

        baseline = pipeline.process(frame)
        
        # Debug: Check output image properties
        if i == 0:  # Only print for first image
            print(f"Debug - Input shape: {frame.shape}, dtype: {frame.dtype}, min: {frame.min()}, max: {frame.max()}")
            print(f"Debug - Output shape: {baseline.shape}, dtype: {baseline.dtype}, min: {baseline.min()}, max: {baseline.max()}")
        
        # Ensure output is uint8 for saving
        if baseline.dtype != np.uint8:
            baseline = np.clip(baseline, 0, 255).astype(np.uint8)
        
        success = cv2.imwrite(full_output_path, baseline)
        if not success:
            print(f"Error: Failed to save image {filename}")
            continue
        
        print(f"[{i+1}/{len(image_files)}] Processed and saved: {filename}")

        # cv2.imshow("Processing (Press 'q' to stop)", baseline)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     print("Processing interrupted by user.")
        #     break

    cv2.destroyAllWindows()
    print("Batch processing completed.")

if __name__ == "__main__":
    main()