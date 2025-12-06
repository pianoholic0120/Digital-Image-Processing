import numpy as np
import cv2

def create_manual_pcalib(width, height, output_folder="."):
    x = np.linspace(0, 1, 256)
    gamma_correction = np.power(x, 2.2) 
    
    gamma_correction = gamma_correction / gamma_correction[-1]
    
    with open(f"{output_folder}/pcalib.txt", "w") as f:
        line = " ".join([f"{v:.6f}" for v in gamma_correction])
        f.write(line)

    print(f"Generated pcalib.txt (Gamma 2.2) in {output_folder}")

    x_axis = np.linspace(-1, 1, width)
    y_axis = np.linspace(-1 * (height/width), 1 * (height/width), height)
    xx, yy = np.meshgrid(x_axis, y_axis)
    radius_squared = xx**2 + yy**2
    
    strength = 0.2  # 0.2 represents the edge is 20% darker than the center (conservative estimate)
    vignette = 1.0 - strength * radius_squared
    vignette = np.clip(vignette, 0, 1)

    vignette_uint16 = (vignette * 65535).astype(np.uint16)
    cv2.imwrite(f"{output_folder}/vignette.png", vignette_uint16)
    
    print(f"Generated vignette.png (Strength {strength}) in {output_folder}")

create_manual_pcalib(width=640, height=480) 