# Digital Image Processing - Homework 5

## Overview

This project implements and analyzes two fundamental image processing topics: JPEG image compression and morphological operations. The work includes comparative analysis of Variable Length Coding (VLC) versus Arithmetic Encoding for AC coefficients, JPEG bitstream structure analysis, and comprehensive morphological operation experiments with various kernel sizes.

## Project Structure

```
hw5/
├── A_part_a.py                    # Part A.1: VLC encoding implementation
├── A_part_b.py                    # Part A.2: Arithmetic encoding implementation
├── B.py                           # Part B: Morphological operations (erosion, dilation, opening, closing)
├── generate_comprehensive_experiments.py  # Comprehensive morphology experiments
├── generate_quantitative_charts.py        # Quantitative analysis visualization
├── report.md                      # Main report document (Markdown)
├── export_html.sh                 # Script to generate HTML report
├── style.css                      # CSS stylesheet for report
├── image_hex.txt                  # JPEG hex dump for bitstream analysis
├── noisy_rectangle.tif            # Input image for morphological operations
└── results/                       # Output directory
    ├── A.txt                      # Arithmetic encoding result
    ├── C_erosion.png
    ├── D_dilation.png
    ├── E_opening.png
    └── F_closing.png
```

## Requirements

- Python 3.7+
- NumPy
- OpenCV (cv2)
- Pillow (PIL)
- Pandoc (for report generation)
- Decimal module (standard library)

## Installation

```bash
pip install numpy opencv-python pillow
```

For report generation, install Pandoc:
- macOS: `brew install pandoc`
- Linux: `sudo apt-get install pandoc`
- Windows: Download from [pandoc.org](https://pandoc.org/installing.html)

## Usage

### Part A: JPEG Image Compression

#### A.1 Variable Length Coding (VLC)

Run the VLC encoding implementation:

```bash
python A_part_a.py
```

This script implements Huffman encoding for AC coefficients using run-length encoding and predefined JPEG Huffman tables.

#### A.2 Arithmetic Encoding

Run the arithmetic encoding implementation:

```bash
python A_part_b.py
```

This script implements arithmetic encoding for AC coefficients using symbol probability distributions. The output is saved to `results/A.txt`.

### Part B: Morphological Operations

#### Basic Operations

Run morphological operations on the input image:

```bash
python B.py --input noisy_rectangle.tif --kernel_size 9 --output_dir results
```

Available operations:
- Erosion
- Dilation
- Opening (erosion followed by dilation)
- Closing (dilation followed by erosion)

#### Comprehensive Experiments

Generate comprehensive experiments with multiple kernel sizes:

```bash
python generate_comprehensive_experiments.py
```

This script performs morphological operations with kernel sizes: 3×3, 5×5, 9×9, 15×15, 20×20, 30×30, 40×40, 50×50, 70×70, 80×80.

Output includes:
- Individual processed images for each operation and kernel size
- Ablation study visualization
- Quantitative analysis charts
- Component statistics (white noise and black holes)
- Zoomed region analysis

#### Quantitative Analysis

Generate quantitative analysis charts:

```bash
python generate_quantitative_charts.py
```

## Report Generation

Generate HTML report from Markdown:

```bash
./export_html.sh
```

This generates `report.html` with embedded resources. To convert to PDF:
1. Open `report.html` in a web browser
2. Use the browser's Print function (Cmd+P / Ctrl+P)
3. Select "Save as PDF"

## Key Features

### Part A: JPEG Compression Analysis

- **VLC Implementation**: Complete Huffman encoding pipeline with run-length encoding, SIZE determination, and amplitude encoding
- **Arithmetic Encoding**: High-precision arithmetic encoding with cumulative distribution functions
- **Comparative Analysis**: Compression ratio comparison (VLC: 9.33:1, Arithmetic: 10.5:1)
- **Bitstream Analysis**: Detailed JPEG file structure analysis including SOI, APP0, DQT, SOF0, DHT, SOS, ECS, and EOI segments

### Part B: Morphological Operations

- **Manual Implementation**: Custom implementations of erosion, dilation, opening, and closing without using OpenCV's built-in functions
- **Comprehensive Ablation Study**: Systematic evaluation across 10 different kernel sizes
- **Quantitative Analysis**: Pixel-level statistics including white pixel counts, change rates, and operation efficiency ratios
- **Qualitative Analysis**: Visual inspection of operation effects with zoomed region studies
- **Component Analysis**: Connected components analysis for white noise and black hole identification

## Experimental Results

### Morphological Operations Efficiency

- **Opening**: Removes 0.06%-5.34% of pixels (selective noise removal)
- **Closing**: Adds 0.03%-11.07% of pixels (selective hole filling)
- **Erosion**: Removes 2.01%-45.93% of pixels (non-selective)
- **Dilation**: Adds 2.08%-113.87% of pixels (non-selective)

### Optimal Kernel Size

Based on empirical analysis, the optimal kernel range is **15×15 to 30×30** for balanced precision and coverage while preserving main structure.

## Technical Details

### Arithmetic Encoding

- Uses high-precision Decimal arithmetic (100 decimal places)
- Implements interval subdivision based on cumulative distribution functions
- Generates binary output representing the final interval

### Morphological Operations

- Zero-padding for border handling
- Vectorized operations using NumPy array slicing
- Efficient kernel application through shifted views

## References

1. ITU-T Recommendation T.81 (1992). *Information technology – Digital compression and coding of continuous-tone still images – Requirements and guidelines*. (JPEG standard)
2. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
3. Witten, I. H., Neal, R. M., & Cleary, J. G. (1987). Arithmetic coding for data compression. *Communications of the ACM*, 30(6), 520-540.
4. Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press.

## Author

R14942096 林祐群

## Date

November 26, 2025

