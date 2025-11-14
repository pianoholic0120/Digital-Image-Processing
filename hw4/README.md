# Digital Image Processing - Homework 4

## Image Reconstruction and Color Appearance Model Analysis

**Author:** R14942096 林祐群  
**Course:** Digital Image Processing  
**Assignment:** Homework 4

---

## Overview

This project implements and analyzes two fundamental image processing techniques:

1. **Part A: Filtered Backprojection Image Reconstruction** - Reconstruction of images from Radon transform projections using filtered backprojection algorithm with varying angular sampling rates and Hamming window filtering.

2. **Part B: Chromatic Adaptation and Color Constancy** - Implementation and comparison of CAT02 and Bradford chromatic adaptation transforms for color appearance modeling under different illuminants.

---

## Project Structure

```
hw4/
├── part_a.py                          # Filtered backprojection implementation
├── part_b.py                          # Chromatic adaptation implementation
├── report.md                          # Comprehensive analysis report (Markdown)
├── report.html                        # Generated HTML report (run export_html.sh)
├── style.css                          # Academic styling for HTML report
├── export_html.sh                     # Script to generate HTML from Markdown
├── images/                            # Input images
│   ├── source_image.tif              # Source image for chromatic adaptation
│   ├── source_image.rgb              # Source white point (RGB values)
│   ├── target_image.tif              # Target image for chromatic adaptation
│   └── target_image.rgb               # Target white point (RGB values)
├── output_part_a/                     # Part A results (with Hamming window)
│   ├── input_image_600x600.png
│   ├── sinogram_original.png
│   ├── sinogram_*.deg.png            # Sinograms for different angle increments
│   └── reconstruction_*.deg.png      # Reconstructions for different angle increments
├── output_part_a_no_hamming/          # Part A results (without Hamming window)
│   └── [same structure as output_part_a]
├── output_part_b_CAT02/               # Part B CAT02 results
│   ├── CAT02_adapted_sRGB.png
│   ├── source_view_sRGB.png
│   └── target_view_sRGB.png
└── output_part_b_Bradford/            # Part B Bradford results
    ├── Bradford_adapted_sRGB.png
    ├── source_view_sRGB.png
    └── target_view_sRGB.png
```

---

## Requirements

### Python Packages

- **NumPy** (>= 1.20.0) - Numerical computations
- **OpenCV** (cv2, >= 4.5.0) - Image I/O operations
- **Colour** (>= 0.4.0) - Color science library (for CIEDE2000 calculation)

### System Requirements

- Python 3.8 or higher
- Bash shell (for running export script)
- Pandoc (for HTML report generation, optional)

### Installation

```bash
# Install required packages
pip install numpy opencv-python colour-science

# Optional: Install Pandoc for HTML report generation
# macOS: brew install pandoc
# Ubuntu/Debian: sudo apt-get install pandoc
```

---

## Part A: Filtered Backprojection Image Reconstruction

### Description

Implements the filtered backprojection algorithm to reconstruct images from their Radon transform (sinogram). The algorithm includes:

- Parallel-beam projection generation
- Frequency-domain filtering with ramp filter |ω|
- Hamming window application for noise reduction
- Backprojection and image reconstruction

### Algorithm Steps

1. **Projection Generation**: Compute parallel-beam projections at angles θ ∈ [0°, 180°)
2. **Frequency Domain Filtering**: Apply ramp filter |ω| multiplied by Hamming window
3. **Inverse Fourier Transform**: Convert filtered projections back to spatial domain
4. **Backprojection**: Integrate all filtered projections to reconstruct the image

### Usage

```bash
# Run with Hamming window (default)
python part_a.py --use_hamming true

# Run without Hamming window
python part_a.py --use_hamming false
```

### Parameters

- **Angular Sampling**: Tested with increments of 1.0°, 0.5°, 0.25°, and 0.125°
- **Input Image**: 600×600 pixel image with 300×300 white square
- **Hamming Window**: c = 0.54 (standard Hamming window coefficient)

### Output

The script generates:

- **Reconstructions**: `reconstruction_{angle_increment}deg.png` - Reconstructed images for each angular sampling rate
- **Sinograms**: `sinogram_{angle_increment}deg.png` - Sinogram visualizations for each reconstruction
- **Original Sinogram**: `sinogram_original.png` - Sinogram of the input image

### Results Summary

| Angle Increment | Projections | PSNR (dB) | SSIM    |
|:---------------:|:-----------:|:---------:|:-------:|
| 1.0°            | 180         | 13.42     | 0.8131  |
| 0.5°            | 360         | 16.41     | 0.9098  |
| 0.25°           | 720         | 20.16     | 0.9653  |
| 0.125°          | 1440        | 23.15     | 0.9832  |

**Hamming Window Impact**: Provides 4.9-8.6 dB PSNR improvement across all sampling rates.

---

## Part B: Chromatic Adaptation and Color Constancy

### Description

Implements and compares two chromatic adaptation transforms (CAT02 and Bradford) for mapping colors from a source illuminant to a target illuminant, maintaining perceptual color constancy.

### Chromatic Adaptation Methods

#### CAT02 Transform

The CAT02 (CIECAM02) transform uses a cone response matrix optimized for the 2° standard observer:

$$
[L, M, S]^T = \mathbf{M}_{CAT02} [X, Y, Z]^T
$$

#### Bradford Transform

The Bradford transform uses a historically developed cone response matrix:

$$
[L, M, S]^T = \mathbf{M}_{Bradford} [X, Y, Z]^T
$$

### Adaptation Pipeline

1. Linear RGB → XYZ (using sRGB transformation matrix)
2. XYZ → LMS (using CAT02 or Bradford matrix)
3. Apply adaptation vector (LMS ratios)
4. LMS → XYZ (using inverse matrix)
5. XYZ → Linear RGB
6. Linear RGB → sRGB (gamma correction)

### Usage

```bash
# Run CAT02 adaptation
python part_b.py --source_image ./images/source_image.tif \
                 --target_image ./images/target_image.tif \
                 --method CAT02

# Run Bradford adaptation
python part_b.py --source_image ./images/source_image.tif \
                 --target_image ./images/target_image.tif \
                 --method Bradford
```

### Parameters

- **Source Image**: Input image under source illuminant (linear RGB)
- **Target Image**: Reference image under target illuminant (linear RGB)
- **White Points**: Provided in `.rgb` files (normalized RGB values)
- **Method**: Either "CAT02" or "Bradford"

### Output

For each method, the script generates:

- **Adapted Image**: `{method}_adapted_sRGB.png` - Color-adapted result
- **Source View**: `source_view_sRGB.png` - Source image in sRGB
- **Target View**: `target_view_sRGB.png` - Target image in sRGB

### Performance Metrics

| Method   | Mean CIEDE2000 ($\Delta E^{*}00$) | PSNR vs Target | SSIM vs Target |
|:--------:|:---------------------------------:|:--------------:|:--------------:|
| CAT02    | 5.54                              | 12.24 dB       | 0.4721         |
| Bradford | 5.55                              | 12.23 dB       | 0.4721         |

Both methods demonstrate equivalent performance with negligible difference (0.01 $\Delta E^{*}00$).

---

## Report Generation

### Markdown Report

The comprehensive analysis report is written in Markdown format (`report.md`) and includes:

- Detailed methodology explanations
- Mathematical formulations
- Quantitative analysis with tables
- Visual comparisons with embedded images
- Discussion of results and trade-offs
- Conclusions and implementation details

### HTML Report Generation

To generate an HTML report from the Markdown file:

```bash
bash export_html.sh
```

This will:

1. Convert `report.md` to `report.html` using Pandoc
2. Apply academic styling from `style.css`
3. Embed MathJax for mathematical formula rendering
4. Embed all images and resources

### PDF Export

After generating the HTML report:

1. Open `report.html` in a web browser
2. Use browser's print function (Ctrl+P / Cmd+P)
3. Select "Save as PDF" or print to PDF
4. Ensure "Background graphics" is enabled for proper styling

**Note**: The CSS includes print-specific optimizations to ensure proper layout in PDF format, including side-by-side image arrangements.

---

## Implementation Details

### Part A: Filtered Backprojection

**Key Features:**

- Manual sinogram generation using histogram-based projection
- FFT-based frequency domain filtering
- Proper handling of angle normalization in backprojection
- Consistent sinogram dimensions for comparison
- Support for Hamming window filtering

**Computational Complexity:**

- Sinogram generation: O(N² × M) where N is image size, M is number of angles
- Filtering: O(M × N log N) for FFT operations
- Backprojection: O(N² × M) for interpolation and accumulation

### Part B: Chromatic Adaptation

**Key Features:**

- Manual color space conversions (RGB↔XYZ, XYZ↔Lab)
- Proper white point handling and normalization
- Correct adaptation vector broadcasting
- CIEDE2000 color difference calculation

**Color Space Conversion:**

- sRGB to XYZ matrix: Standard ITU-R BT.709 matrix
- CAT02 matrix: CIECAM02 optimized for 2° observer
- Bradford matrix: Historical standard for 2° observer

---

## Technical Specifications

### Color Space Conversions

**sRGB to Linear RGB:**
- Linear segment: $R_{linear} = R_{sRGB} / 12.92$ for $R_{sRGB} \leq 0.04045$
- Gamma segment: $R_{linear} = ((R_{sRGB} + 0.055) / 1.055)^{2.4}$ otherwise

**Linear RGB to XYZ:**
- Uses standard ITU-R BT.709 transformation matrix

**XYZ to Lab:**
- Manual implementation using CIE standard formulas
- White point normalization for perceptual uniformity

### Filtering

**Ramp Filter:**
- Frequency domain: $H(\omega) = |\omega|$

**Hamming Window:**
- $W(\omega) = 0.54 + 0.46\cos(2\pi\omega)$
- Applied to reduce high-frequency noise and artifacts

---

## Results and Analysis

### Part A Findings

1. **Angular Sampling**: Reconstruction quality improves logarithmically with finer angular sampling. With 0.125° increments (1440 projections) and Hamming window filtering, near-perfect reconstruction is achieved (SSIM > 0.98, PSNR = 23.15 dB).

2. **Hamming Window**: Essential for noise reduction and artifact suppression. Provides 4.9-8.6 dB PSNR improvement across all sampling rates, with increasing benefit at finer sampling.

3. **Optimal Configuration**: For practical applications, 0.25° angular increment (720 projections) with Hamming window provides excellent quality (SSIM = 0.965, PSNR = 20.16 dB) with reasonable computational cost.

### Part B Findings

1. **Method Equivalence**: Both CAT02 and Bradford transforms demonstrate equivalent performance ($\Delta E^{*}00 \approx 5.5$) for the tested illuminant pair, with negligible difference (0.01 $\Delta E^{*}00$).

2. **Adaptation Success**: Both methods successfully adapt colors from a cool, blue-dominant source illuminant to a warmer target illuminant, demonstrating effective chromatic adaptation.

3. **CIEDE2000 Interpretation**: $\Delta E^{*}00 \approx 5.5$ indicates "noticeable but acceptable" color difference, confirming successful but not perfect adaptation.

---

## File Format Specifications

### White Point Files (.rgb)

Format: Three space-separated floating-point values representing RGB values in the range [0, 255].

Example:
```
66.83 100.36 228.09
```

The script automatically normalizes these values to [0, 1] range and converts to XYZ color space.

### Image Formats

- **Input**: TIFF files (`.tif`) - Linear RGB format
- **Output**: PNG files (`.png`) - sRGB format for display

---

## Troubleshooting

### Common Issues

**Issue**: Images not loading in HTML report
- **Solution**: Ensure all image paths are relative to the report.md file location
- **Solution**: Check that output directories exist and contain the expected images

**Issue**: Math formulas not rendering in HTML
- **Solution**: Ensure internet connection for MathJax CDN
- **Solution**: Check browser console for JavaScript errors

**Issue**: PDF layout problems (images stacking vertically)
- **Solution**: Use the updated CSS with print-specific styles
- **Solution**: Ensure browser print settings include background graphics
- **Solution**: Check that flex containers use `flex-wrap: nowrap` in print mode

**Issue**: Pandoc warnings about math formulas
- **Solution**: These are typically warnings, not errors. MathJax will render formulas correctly in the browser.

---

## References

1. Kak, A. C., & Slaney, M. (2001). *Principles of Computerized Tomographic Imaging*. SIAM.

2. Fairchild, M. D. (2013). *Color Appearance Models* (3rd ed.). John Wiley & Sons.

3. Sharma, G., Wu, W., & Dalal, E. N. (2005). The CIEDE2000 color-difference formula: Implementation notes, supplementary test data, and mathematical observations. *Color Research & Application*, 30(1), 21-30.

4. Luo, M. R., Cui, G., & Rigg, B. (2001). The development of the CIE 2000 colour-difference formula: CIEDE2000. *Color Research & Application*, 26(5), 340-350.

---

## License

This project is part of a course assignment and is intended for educational purposes.

---

## Contact

For questions or issues regarding this implementation, please refer to the course materials or contact the course instructor.

