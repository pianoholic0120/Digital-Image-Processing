# Digital Image Processing Homework 3

## Overview

This homework implements frequency domain filtering and comprehensive image enhancement pipelines. It consists of two main parts:

- **Part A**: Frequency domain filtering using vertical Sobel kernel with odd symmetry requirements
- **Part B**: Multi-stage image enhancement for noisy Jupiter images captured by NASA's Juno spacecraft

## Requirements

- Python 3.8+
- NumPy
- OpenCV (opencv-python)
- Standard Python library only

**Important**: Only image I/O functions from OpenCV are used. All other image operations (filtering, convolution, histogram operations, etc.) are manually implemented.

## Quick Start

### Part A: Frequency Domain Filtering

```bash
# Run all subproblems for Part A
python3 hw3_a.py --subproblem 1 --input ./images/keyboard.tif --output ./output_a/
python3 hw3_a.py --subproblem 2 --input ./images/keyboard.tif --output ./output_a/
python3 hw3_a.py --subproblem 3 --input ./images/keyboard.tif --output ./output_a/
python3 hw3_a.py --subproblem 4 --input ./images/keyboard.tif --output ./output_a/
python3 hw3_a.py --subproblem 5 --input ./images/keyboard.tif --output ./output_a/
```

Or use the shell script:
```bash
bash hw3_a.sh
```

### Part B: Image Enhancement

```bash
# Run all subproblems for Part B
python3 hw3_b.py --subproblem 1 --input ./images/noisy_image.tif --output ./output_b/
python3 hw3_b.py --subproblem 2 --input ./images/noisy_image.tif --output ./output_b/
python3 hw3_b.py --subproblem 3 --input ./images/noisy_image.tif --output ./output_b/
python3 hw3_b.py --subproblem 4 --input ./images/noisy_image.tif --output ./output_b/
```

Or use the shell script:
```bash
bash hw3_b.sh
```

## Additional Utilities

### Ablation Study

Run systematic ablation study to understand the contribution of each processing step:

```bash
python3 ablation_study.py --input ./images/noisy_image.tif --output ./output_b/ablation/
```

This generates:
- Intermediate results for each step combination
- Quantitative metrics (contrast, edge strength, PSNR)
- Metrics comparison table

### Create Comparison Images

Generate side-by-side or grid comparisons:

```bash
# Side-by-side comparison
python3 create_comparison.py --mode side \
    --images output_a/a3.png output_a/a4.png \
    --labels "Frequency Domain" "Spatial Domain" \
    --output comparison.png

# Grid comparison
python3 create_comparison.py --mode grid --cols 2 \
    --images img1.png img2.png img3.png img4.png \
    --labels "Step 1" "Step 2" "Step 3" "Step 4" \
    --output grid.png
```

## Output Files

### Part A Outputs (output_a/)
- `a1.png`: Fourier spectrum of keyboard image
- `a2.png`: 4×4 padded kernel visualization
- `a3.png`: Frequency-domain filtering result
- `a4.png`: Spatial-domain filtering result (should match a3.png)
- `a5.png`: Frequency-domain filtering without odd symmetry

### Part B Outputs (output_b/)
- `1_filter.png`: Designed frequency-domain notch filter
- `1_filtered_image.png`: Image after notch filtering
- `2_sharpened.png`: Image after unsharp masking
- `3_equalized.png`: Image after histogram equalization
- `3_hist_before.png`: Histogram before equalization
- `3_hist_after.png`: Histogram after equalization
- `4_hist_eq.png`: Intermediate result (after histogram equalization)
- `4_my_procedure.png`: Final enhanced image (custom pipeline)

## Report

See `REPORT.md` for detailed documentation including:
- Theory and methodology
- Implementation details
- Results and analysis
- Ablation study findings
- Conclusion and future work

## Implementation Notes

### Manual Implementations

All image processing operations (except I/O) are manually implemented:

1. **2D Correlation/Filtering**: Custom `filter2d_correlation()` function
2. **Gaussian Blur**: Frequency-domain implementation using FFT
3. **Histogram Operations**: Manual histogram calculation, equalization, and CLAHE
4. **Retinex Algorithm**: Single-scale Retinex for illumination correction

### Key Features

- **Odd Symmetry**: Proper kernel padding for frequency-domain equivalence
- **Correlation vs. Convolution**: Correct handling using complex conjugate
- **Multi-stage Pipeline**: Comprehensive enhancement combining multiple techniques
- **Adaptive Processing**: CLAHE for local contrast enhancement
- **Illumination Correction**: Retinex for non-uniform lighting

## License

This project is for educational purposes only.

---

For detailed information, please refer to `REPORT.md`.

