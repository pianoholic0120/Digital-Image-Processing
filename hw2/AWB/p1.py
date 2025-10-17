import cv2
import numpy as np
import argparse
import os

# ======================================================================================
# HW 2(1): Auto White Balance (AWB) Grey World Algorithm
# ======================================================================================

def grey_world_awb(img, option=3):
    """
    Args:
        img (np.ndarray): input image in BGR format, numpy array.

    Returns:
        np.ndarray: white balanced image.
    """
    img_float = img.astype(np.float32)
    b, g, r = cv2.split(img_float)
    avg_b = np.mean(b)
    avg_g = np.mean(g)
    avg_r = np.mean(r)
    if option == 1:
    # === Option 1: Average g channel as the reference ===
        gain_b = avg_g / avg_b
        gain_r = avg_g / avg_r

        b_corrected = b * gain_b
        r_corrected = r * gain_r
        result_img = cv2.merge([b_corrected, g, r_corrected])
    elif option == 2:
    # === Option 2: Average all channels as the reference ===
        avg_gray = (avg_b + avg_g + avg_r) / 3.0
        gain_b = avg_gray / avg_b
        gain_g = avg_gray / avg_g
        gain_r = avg_gray / avg_r
        b_corrected = b * gain_b
        g_corrected = g * gain_g
        r_corrected = r * gain_r
        result_img = cv2.merge([b_corrected, g_corrected, r_corrected])
    elif option == 3:
    # === Option 3: 127.5 as the reference ===
        gain_b = 127.5 / avg_b
        gain_g = 127.5 / avg_g
        gain_r = 127.5 / avg_r
        b_corrected = b * gain_b
        g_corrected = g * gain_g
        r_corrected = r * gain_r    
        result_img = cv2.merge([b_corrected, g_corrected, r_corrected])
    else:
        raise ValueError(f"Invalid option: {option}")
    result_img = np.clip(result_img, 0, 255).astype(np.uint8)

    return result_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, help="Path to the input image")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--option", type=int, help="Option for the grey world algorithm", default=3)
    args = parser.parse_args()
    path = args.input_image
    img = cv2.imread(path)
    result = grey_world_awb(img, args.option)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_image = os.path.join(args.output_dir, os.path.basename(path).split(".")[0] + ".png")
    cv2.imwrite(output_image, result)