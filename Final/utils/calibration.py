import cv2
import numpy as np
import os
import glob
import random

# --- 1. CONFIGURATION ---

IMAGE_DIR = "/Users/arthurlin/Desktop/DIP/Final/hall_with_iphone13/calibrations"

# Number of internal corners on the chessboard pattern
GRID_PATTERN = (14, 14)      # (cols, rows) of internal corners

# Size of one square block (in mm, or any consistent unit)
BLOCK_SIZE_MM = 12.0

# Limit number of images to speed up calibration
MAX_IMAGES = 1000

# --- 2. CORNER REFINEMENT & CHESSBOARD FLAGS ---

# Criteria for cornerSubPix refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Flags for more robust chessboard detection
cb_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
# You *can* add FAST_CHECK for speed, but it may skip some good images:
# cb_flags |= cv2.CALIB_CB_FAST_CHECK

# Prepare single 3D object point grid (Z=0)
objp = np.zeros((GRID_PATTERN[0] * GRID_PATTERN[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:GRID_PATTERN[0], 0:GRID_PATTERN[1]].T.reshape(-1, 2)
objp *= BLOCK_SIZE_MM

# Lists of 3D and 2D points for all accepted images
objpoints = []
imgpoints = []

# --- 3. LOAD & (OPTIONALLY) SUBSAMPLE IMAGES ---

print(f"Looking for images in: {IMAGE_DIR}")
images = glob.glob(os.path.join(IMAGE_DIR, '*.jpg')) + glob.glob(os.path.join(IMAGE_DIR, '*.png'))
print(f"Found {len(images)} images.")

if not images:
    print("Error: No images found. Check your IMAGE_DIR path.")
    exit()

if len(images) > MAX_IMAGES:
    images = random.sample(images, MAX_IMAGES)
    print(f"Subsampling to {len(images)} images for calibration.")

img_size = None
found_corners_count = 0

# --- 4. DETECT CORNERS WITH STRICT FILTERS ---

for i, fname in enumerate(images):
    print(f"[{i+1}/{len(images)}] Processing {os.path.basename(fname)}")

    img = cv2.imread(fname)
    if img is None:
        print(f"  ...Warning: Could not read {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img_size is None:
        img_size = gray.shape[::-1]  # (width, height)

    # Detect chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, GRID_PATTERN, cb_flags)

    if not ret:
        print(f"  ...Could not find corners in {os.path.basename(fname)}")
        continue

    # Refine corner positions
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # --- Geometric quality checks ---

    xs = corners_refined[:, 0, 0]
    ys = corners_refined[:, 0, 1]
    board_w = xs.max() - xs.min()
    board_h = ys.max() - ys.min()

    img_w, img_h = img_size  # (width, height)

    # Require the board to occupy at least 30% of image width & height
    if board_w < 0.3 * img_w or board_h < 0.3 * img_h:
        print("  ...Board too small in image, skipping.")
        continue

    # Require the board not to be too close to image borders (5% margin)
    margin_x = 0.05 * img_w
    margin_y = 0.05 * img_h
    if (xs.min() < margin_x or xs.max() > img_w - margin_x or
        ys.min() < margin_y or ys.max() > img_h - margin_y):
        print("  ...Board too close to image border, skipping.")
        continue

    # If it passes all checks, accept it
    found_corners_count += 1
    objpoints.append(objp)
    imgpoints.append(corners_refined)

cv2.destroyAllWindows()

if found_corners_count == 0:
    print("\nError: No valid chessboard views were accepted. "
          "Check GRID_PATTERN, image quality, or relax filters.")
    exit()

print(f"\nAccepted {found_corners_count}/{len(images)} images after strict filtering.")
print("Running calibration with 8-coefficient rational model...")

# --- 5. CALIBRATION WITH 8 DISTORTION COEFFS ---

# Enables k4, k5, k6 in addition to k1,k2,k3,p1,p2
flags = cv2.CALIB_RATIONAL_MODEL

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_size, None, None, flags=flags
)

# --- 6. DISPLAY RESULTS ---

print("\n--- Calibration Results ---")

if ret:
    print("\n✅ Calibration Successful!")
    print(f"RMS Reprojection Error (from calibrateCamera): {ret:.6f} pixels")

    print("\nCamera Intrinsic Matrix (mtx):")
    print(camera_matrix)
    print(f"  fx: {camera_matrix[0, 0]:.4f}")
    print(f"  fy: {camera_matrix[1, 1]:.4f}")
    print(f"  cx: {camera_matrix[0, 2]:.4f}")
    print(f"  cy: {camera_matrix[1, 2]:.4f}")

    print("\nDistortion Coefficients (dist):")
    print(dist_coeffs[0])
    print(f"Number of distortion parameters: {len(dist_coeffs[0])}")

    # OpenCV order: [k1, k2, p1, p2, k3, k4, k5, k6, ...]
    print("\nCoefficients Breakdown (first 8):")
    print(f"  k1: {dist_coeffs[0][0]:.6f}  (Radial)")
    print(f"  k2: {dist_coeffs[0][1]:.6f}  (Radial)")
    print(f"  p1: {dist_coeffs[0][2]:.6f}  (Tangential)")
    print(f"  p2: {dist_coeffs[0][3]:.6f}  (Tangential)")
    print(f"  k3: {dist_coeffs[0][4]:.6f}  (Radial)")
    print(f"  k4: {dist_coeffs[0][5]:.6f}  (Radial, rational)")
    print(f"  k5: {dist_coeffs[0][6]:.6f}  (Radial, rational)")
    print(f"  k6: {dist_coeffs[0][7]:.6f}  (Radial, rational)")

    # --- Optional: detailed reprojection error on subset ---

    mean_error = 0
    num_for_error = min(50, len(objpoints))
    for i in range(num_for_error):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    mean_error /= num_for_error

    print(f"\nAvg Reprojection Error (on {num_for_error} images): {mean_error:.6f} pixels")

    # --- 7. SAVE & UNDISTORT EXAMPLE ---

    np.savez(
        "calibration_data_rational.npz",
        mtx=camera_matrix, dist=dist_coeffs, rvecs=rvecs, tvecs=tvecs
    )
    print("\nCalibration data saved to 'calibration_data_rational.npz'")

    test_img = cv2.imread(images[0])
    h, w = test_img.shape[:2]

    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    undistorted_img = cv2.undistort(
        test_img, camera_matrix, dist_coeffs, None, new_camera_mtx
    )

    cv2.imwrite("undistorted_example_rational.png", undistorted_img)
    print("Saved 'undistorted_example_rational.png' for you to check.")

else:
    print("\n❌ Calibration Failed!")
