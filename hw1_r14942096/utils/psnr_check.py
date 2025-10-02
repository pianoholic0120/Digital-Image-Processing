import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse

def psnr(img1, img2):
    """Compute Peak Signal-to-Noise Ratio between two images."""
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", type=str, help="Path to the reconstructed image")
    parser.add_argument("--gt", type=str, help="Path to the ground truth image")
    args = parser.parse_args()
    
    if args.recon is None or args.gt is None:
        print("Usage: python psnr_check.py --recon reconstructed.png --gt ground_truth.png")
        sys.exit(1)
    
    rec = cv2.imread(args.recon)
    gt = cv2.imread(args.gt)

    if rec.shape != gt.shape:
        print("Error: Images must have the same shape")
        sys.exit(1)

    score = psnr(rec, gt)
    print(f"PSNR: {score:.2f} dB")