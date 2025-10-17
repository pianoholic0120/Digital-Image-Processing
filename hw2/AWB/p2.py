import cv2
import numpy as np
import argparse
import os

# ======================================================================================
# HW 2(2): Auto White Balance (AWB) White Patch Algorithm
# ======================================================================================

def white_patch_awb(img, option=1):
    """
    Args:
        img (np.ndarray): input image in BGR format, numpy array.

    Returns:
        np.ndarray: white balanced image.
    """
    img_float = img.astype(np.float32)
    b, g, r = cv2.split(img_float)
    max_b = np.max(b)
    max_g = np.max(g)
    max_r = np.max(r)
    if option == 1:
    # === Option 1: 255.0 as the reference ===
        max_val = 255.0
        gain_b = max_val / max_b
        gain_g = max_val / max_g
        gain_r = max_val / max_r
        b_corrected = b * gain_b
        g_corrected = g * gain_g
        r_corrected = r * gain_r
        result_img = cv2.merge([b_corrected, g_corrected, r_corrected])
    elif option == 2:
    # === Option 2: G_max as the reference ===
        max_val = max_g
        gain_b = max_val / max_b
        gain_r = max_val / max_r
        b_corrected = b * gain_b
        r_corrected = r * gain_r
        result_img = cv2.merge([b_corrected, g, r_corrected])
    else:
        raise ValueError(f"Invalid option: {option}")
    result_img = np.clip(result_img, 0, 255).astype(np.uint8)

    return result_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, help="Path to the input image")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--option", type=int, help="Option for the white patch algorithm", default=1)
    args = parser.parse_args()
    path = args.input_image
    img = cv2.imread(path)
    result = white_patch_awb(img, args.option)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_image = os.path.join(args.output_dir, os.path.basename(path).split(".")[0] + ".png")
    cv2.imwrite(output_image, result)