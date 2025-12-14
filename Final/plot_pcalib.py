#!/usr/bin/env python3
"""
Plot pcalib.txt (inverse camera response function)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def plot_pcalib(pcalib_file):
    # Read pcalib.txt
    with open(pcalib_file, 'r') as f:
        line = f.readline().strip()
        values = [float(x) for x in line.split()]
    
    # Pixel values (0-255)
    pixel_values = np.arange(256)
    
    # Intensity values from pcalib.txt
    intensity_values = np.array(values)
    
    # Normalize intensity to [0, 255] range for comparison
    intensity_normalized = intensity_values
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Full range
    ax1 = axes[0]
    ax1.plot(pixel_values, intensity_normalized, 'b-', linewidth=2, label='Inverse CRF')
    ax1.plot(pixel_values, pixel_values, 'r--', linewidth=1.5, label='Linear (y=x)')
    ax1.set_xlabel('Pixel Value (Input)', fontsize=12)
    ax1.set_ylabel('Linear Intensity (Output)', fontsize=12)
    ax1.set_title('Camera Response Function (Inverse) - Full Range', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_xlim([0, 255])
    ax1.set_ylim([0, 260])
    
    # Plot 2: Zoomed in to see non-linearity
    ax2 = axes[1]
    ax2.plot(pixel_values, intensity_normalized, 'b-', linewidth=2, label='Inverse CRF')
    ax2.plot(pixel_values, pixel_values, 'r--', linewidth=1.5, label='Linear (y=x)')
    ax2.set_xlabel('Pixel Value (Input)', fontsize=12)
    ax2.set_ylabel('Linear Intensity (Output)', fontsize=12)
    ax2.set_title('Camera Response Function (Inverse) - Zoomed View', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_xlim([0, 100])
    ax2.set_ylim([0, 50])
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(os.path.dirname(pcalib_file), 'pcalib_plot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Also show the plot
    plt.show()
    
    # Analyze linearity
    print("\n=== Linearity Analysis ===")
    print(f"Min intensity: {intensity_normalized[0]:.6f}")
    print(f"Max intensity: {intensity_normalized[255]:.6f}")
    print(f"Range: {intensity_normalized[255] - intensity_normalized[0]:.6f}")
    
    # Calculate deviation from linear
    linear_values = pixel_values
    deviation = intensity_normalized - linear_values
    max_deviation = np.max(np.abs(deviation))
    mean_deviation = np.mean(np.abs(deviation))
    
    print(f"\nDeviation from linear (y=x):")
    print(f"  Max absolute deviation: {max_deviation:.6f}")
    print(f"  Mean absolute deviation: {mean_deviation:.6f}")
    
    # Check if it's approximately linear
    if max_deviation < 1.0:
        print("\n✓ The function is approximately LINEAR (deviation < 1.0)")
    else:
        print(f"\n✗ The function is NON-LINEAR (max deviation = {max_deviation:.2f})")
        print("  This indicates the camera applies gamma encoding or other non-linear response.")
    
    # Calculate approximate gamma if non-linear
    if max_deviation > 1.0:
        # Try to estimate gamma by fitting y = x^gamma
        # For inverse CRF, if original is y = x^gamma, then inverse is y = x^(1/gamma)
        # We can estimate by looking at mid-range values
        mid_idx = 128
        if intensity_normalized[mid_idx] > 0 and pixel_values[mid_idx] > 0:
            # If linear: intensity = pixel
            # If gamma-encoded: intensity ≈ pixel^gamma
            # For inverse: intensity ≈ pixel^(1/gamma)
            # So: intensity/pixel ≈ pixel^(1/gamma - 1)
            # This is complex, let's just check the shape
            print(f"\n  At pixel=128: intensity={intensity_normalized[mid_idx]:.2f}, expected linear={128:.2f}")
            print(f"  Ratio: {intensity_normalized[mid_idx]/128:.4f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pcalib_file = sys.argv[1]
    else:
        pcalib_file = "/Users/arthurlin/Desktop/DIP/Final/loop/pcalib.txt"
    
    if not os.path.exists(pcalib_file):
        print(f"Error: File not found: {pcalib_file}")
        sys.exit(1)
    
    plot_pcalib(pcalib_file)

