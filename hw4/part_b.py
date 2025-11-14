import os
import numpy as np
import argparse
import colour  

MATRIX_SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
])

MATRIX_XYZ_TO_SRGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252]
])

# CAT02 matrix (2-degree observer)
M_CAT02 = np.array([
    [ 0.7328,  0.4296, -0.1624],
    [-0.7036,  1.6975,  0.0061],
    [ 0.0030,  0.0136,  0.9834]
])
M_CAT02_INV = np.linalg.inv(M_CAT02)

# Bradford matrix (2-degree observer)
M_BRADFORD = np.array([
    [ 0.8951,  0.2664, -0.1614],
    [-0.7502,  1.7135,  0.0367],
    [ 0.0389, -0.0685,  1.0296]
])
M_BRADFORD_INV = np.linalg.inv(M_BRADFORD)

def srgb_to_linear(img):
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(img):
    return np.where(img <= 0.0031308, img * 12.92, 1.055 * np.power(img, 1/2.4) - 0.055)

def linear_RGB_to_XYZ(img_rgb):
    return np.dot(img_rgb, MATRIX_SRGB_TO_XYZ.T)

def XYZ_to_linear_RGB(img_xyz):
    return np.dot(img_xyz, MATRIX_XYZ_TO_SRGB.T)

def _lab_f(t):
    delta = 6 / 29
    delta_cubed = delta ** 3
    term_1 = (1 / 3) * (delta ** 2)
    term_2 = 4 / 29
    return np.where(t > delta_cubed, np.cbrt(t), term_1 * t + term_2)

def XYZ_to_Lab(img_xyz, xyz_white_ref):
    xyz_white_ref = np.asarray(xyz_white_ref).flatten().reshape(1, 3)
    
    ratios = img_xyz / xyz_white_ref
    f_ratios = _lab_f(ratios)
    
    L = 116 * f_ratios[..., 1] - 16
    a = 500 * (f_ratios[..., 0] - f_ratios[..., 1])
    b = 200 * (f_ratios[..., 1] - f_ratios[..., 2])
    
    return np.stack([L, a, b], axis=-1)

def read_white_point(filepath):
    with open(filepath, 'r') as f:
        values = f.readline().strip().split()
        if len(values) != 3:
            raise ValueError(f"White point file {filepath} has incorrect format.")
        wp = np.array([float(v) / 255.0 for v in values])
        wp = np.clip(wp, 0.0, 1.0)
        return wp

def chromatic_adaptation(img_linear_rgb, wp_source_xyz, wp_target_xyz, M_CAT, M_CAT_inv):
    wp_source_xyz = np.asarray(wp_source_xyz).flatten()
    wp_target_xyz = np.asarray(wp_target_xyz).flatten()
    
    img_xyz = linear_RGB_to_XYZ(img_linear_rgb)
    
    LMS_source = np.dot(M_CAT, wp_source_xyz)
    LMS_target = np.dot(M_CAT, wp_target_xyz)
    
    adaptation_vector = LMS_target / (LMS_source + 1e-6)
    
    LMS_img = np.dot(img_xyz, M_CAT.T)
    
    LMS_adapted = LMS_img * adaptation_vector
    
    XYZ_adapted = np.dot(LMS_adapted, M_CAT_inv.T)
    
    linear_rgb_adapted = XYZ_to_linear_RGB(XYZ_adapted)
    
    return np.clip(linear_rgb_adapted, 0, 1)

def CAT02_Adaptation(source_image_linear, source_wp_xyz, target_wp_xyz):
    print("Applying CAT02 Adaptation...")
    return chromatic_adaptation(source_image_linear, source_wp_xyz, target_wp_xyz, M_CAT02, M_CAT02_INV)

def Bradford_Chromatic_Adaptation(source_image_linear, source_wp_xyz, target_wp_xyz):
    print("Applying Bradford Adaptation...")
    return chromatic_adaptation(source_image_linear, source_wp_xyz, target_wp_xyz, M_BRADFORD, M_BRADFORD_INV)

def evaluate_performance(adapted_linear_rgb, target_linear_rgb, target_wp_xyz_ref):
    print("Evaluating performance...")
    target_wp_xyz_ref = np.asarray(target_wp_xyz_ref).flatten()
    
    adapted_xyz = linear_RGB_to_XYZ(adapted_linear_rgb)
    target_xyz = linear_RGB_to_XYZ(target_linear_rgb)

    adapted_lab = XYZ_to_Lab(adapted_xyz, target_wp_xyz_ref)
    target_lab = XYZ_to_Lab(target_xyz, target_wp_xyz_ref)

    delta_E = colour.delta_E(adapted_lab, target_lab, method='CIE 2000')

    return np.mean(delta_E)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform Chromatic Adaptation (CAT02 or Bradford)")
    parser.add_argument("--source_image", type=str, required=True, help="Path to the source .tif image")
    parser.add_argument("--target_image", type=str, required=True, help="Path to the target .tif image")
    parser.add_argument("--method", type=str, required=True, choices=["CAT02", "Bradford"], help="Adaptation method")
    args = parser.parse_args()

    output_path = f"./output_part_b_{args.method}"
    os.makedirs(output_path, exist_ok=True)

    print(f"Loading source image: {args.source_image}")
    print(f"Loading target image: {args.target_image}")
    
    source_image = colour.read_image(args.source_image, bit_depth='float32')
    target_image = colour.read_image(args.target_image, bit_depth='float32')
    

    source_wp_path = os.path.splitext(args.source_image)[0] + ".rgb"
    target_wp_path = os.path.splitext(args.target_image)[0] + ".rgb"
    print(f"Loading source white point: {source_wp_path}")
    print(f"Loading target white point: {target_wp_path}")
    
    source_wp_rgb = read_white_point(source_wp_path)
    target_wp_rgb = read_white_point(target_wp_path)

    source_wp_xyz = linear_RGB_to_XYZ(source_wp_rgb)
    target_wp_xyz = linear_RGB_to_XYZ(target_wp_rgb)

    print(f"Source WP (Linear RGB): {source_wp_rgb} -> (XYZ): {source_wp_xyz}")
    print(f"Target WP (Linear RGB): {target_wp_rgb} -> (XYZ): {target_wp_xyz}")

    if args.method == "CAT02":
        adapted_linear_rgb = CAT02_Adaptation(source_image, source_wp_xyz, target_wp_xyz)
    else:
        adapted_linear_rgb = Bradford_Chromatic_Adaptation(source_image, source_wp_xyz, target_wp_xyz)

    adapted_srgb = linear_to_srgb(adapted_linear_rgb)
    adapted_srgb = np.clip(adapted_srgb, 0, 1)

    output_filename = os.path.join(output_path, f"{args.method}_adapted_sRGB.png")
    colour.write_image(adapted_srgb, output_filename, bit_depth='uint8')
    print(f"Saved {args.method} adapted image to: {output_filename}")

    source_srgb_view = linear_to_srgb(source_image)
    target_srgb_view = linear_to_srgb(target_image)
    colour.write_image(source_srgb_view, os.path.join(output_path, "source_view_sRGB.png"), bit_depth='uint8')
    colour.write_image(target_srgb_view, os.path.join(output_path, "target_view_sRGB.png"), bit_depth='uint8')

    delta_e = evaluate_performance(adapted_linear_rgb, target_image, target_wp_xyz)
    
    print("---" * 10)
    print(f"Method: {args.method}")
    print(f"Mean CIEDE2000 (ΔE*00): {delta_e:.4f}")
    print("---" * 10)