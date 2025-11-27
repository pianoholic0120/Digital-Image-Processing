import numpy as np
import cv2
import os
import time

# --- Global Calibration Data (From Zhang) ---
# Defining these globally is more efficient than redefining them inside the loop
K = np.array([
    [647.20561101,   0.0,         325.37416519],
    [  0.0,         647.96778345, 253.87801536],
    [  0.0,           0.0,           1.0      ]
], dtype=np.float64)

DIST = np.array([-0.17280594, -0.23860071,
                 -0.00133931,  0.00224144,
                 0.03647576], dtype=np.float64)

def undistort_image_memory(img):
    """
    Performs undistortion and cropping in memory.
    Returns the undistorted image matrix.
    """
    h, w = img.shape[:2]
    
    # Calculate new optimal camera matrix
    # alpha=1 retains all pixels (may have black borders)
    # alpha=0 crops the image to valid area only
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, DIST, (w,h), 1, (w,h))
    
    # Undistort
    dst = cv2.undistort(img, K, DIST, None, newcameramtx)

    # Crop the image (Remove black borders)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    
    return dst

def enhance_for_colmap(image):
    """
    Applies CLAHE (Low Light Fix), Denoising, and Sharpening.
    Optimized for COLMAP feature extraction.
    """
    # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This improves low light details without blowing out bright spots
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # ClipLimit=3.0 is aggressive enough for low light but prevents too much noise
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 2. Denoising
    # Essential because CLAHE amplifies sensor noise in the dark
    denoised = cv2.fastNlMeansDenoisingColored(enhanced_img, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21)

    # 3. Sharpening
    # Helps SIFT/ORB find edges in the now-denoised image
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    return sharpened

def main():
    # --- Configuration ---
    input_folder = "/Users/arthurlin/Desktop/DIP/Final/psyduck/baseline/images"
    output_folder = "/Users/arthurlin/Desktop/DIP/Final/COLMAP/input"
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = os.listdir(input_folder)
    print(f"Found {len(files)} files in {input_folder}...")
    
    start_time = time.time()
    count = 0

    for filename in files:
        if filename.lower().endswith(valid_extensions):
            
            # 1. Read Image
            input_path = os.path.join(input_folder, filename)
            img = cv2.imread(input_path)
            
            if img is None:
                print(f"Error: Could not read {filename}")
                continue

            # 2. Pipeline: Undistort -> Enhance
            # We pass the image object, not the filename
            undistorted = undistort_image_memory(img)
            final_img = enhance_for_colmap(undistorted)

            # 3. Save Image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, final_img)
            
            count += 1
            if count % 10 == 0:
                print(f"Processed {count} images...")

    total_time = time.time() - start_time
    print(f"--- Finished processing {count} images in {total_time:.2f} seconds ---")

if __name__ == '__main__':
    main()