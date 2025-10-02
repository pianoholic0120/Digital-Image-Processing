# CFA Demosaicing Methods Comparison Study

## Project Overview

This project implements and compares three different CFA (Color Filter Array) demosaicing methods with comprehensive performance analysis and visualization. The research focuses on evaluating the performance of different interpolation strategies in image reconstruction quality.

## Method Introduction

### 1. P1 - Simple Interpolation Method
- **Algorithm**: Basic neighborhood averaging interpolation
- **Features**: Simple computation, fast processing
- **Use Cases**: Basic applications, resource-constrained environments

### 2. P2 - Edge-Aware Interpolation Method
- **Algorithm**: Adaptive interpolation based on edge detection
- **Features**: Considers image edge information, reduces interpolation artifacts
- **Parameters**: Supports multiple threshold settings (3, 5, 10, 20, 30, 50)
- **Use Cases**: Applications requiring edge detail preservation

### 3. P3 - Stochastic Interpolation Method
- **Algorithm**: Intelligent interpolation based on stochastic policy
- **Features**: Uses multi-directional edge indicators and weight calculation
- **Advantages**: Significantly improves image quality
- **Use Cases**: High-quality image reconstruction requirements

## Project Structure

```
hw1_r14942096/
├── images/
│   ├── raw_image/          # Original Bayer images
│   │   ├── A.tiff
│   │   ├── B.png
│   │   ├── C.png
│   │   ├── D.png
│   │   └── E.png
│   └── ground_truth/       # Ground truth images
│       ├── A.png
│       ├── B.png
│       ├── C.png
│       ├── D.png
│       └── E.png
├── output/                 # Output results
│   ├── p1/                # P1 method results
│   ├── p2_*thre/          # P2 method results (different thresholds)
│   ├── p3/                 # P3 method results
│   └── analysis_plots/     # Analysis plots
├── utils/                 # Utility scripts
│   ├── psnr_check.py      # PSNR calculation tool
│   └── psnr_check.sh      # PSNR check script
├── p1.py                  # P1 method implementation
├── p2.py                  # P2 method implementation
├── p3.py                  # P3 method implementation
├── test_improvements.py   # Complete analysis script
├── analysis_report.md     # Detailed analysis report
└── README.md              # This file
```

## Quick Start

### Requirements
- Python 3.6+
- OpenCV (cv2)
- NumPy
- Matplotlib
- Pandas

### Install Dependencies
```bash
pip install opencv-python numpy matplotlib pandas
```

### Run Individual Methods
```bash
# Run P1 method
python3 p1.py --input_image images/raw_image/A.tiff

# Run P2 method (specify threshold)
python3 p2.py --input_image images/raw_image/A.tiff --threshold 10

# Run P3 method
python3 p3.py --input_image images/raw_image/A.tiff
```

### Run Complete Analysis
```bash
# Run all methods and generate analysis report
python3 test_improvements.py
```

## Performance Results

### Key Findings
- **P3 method performs best**: Average PSNR 31.14 dB
- **Improvement over P1**: +14.40 dB (85% improvement)
- **Improvement over best P2**: +12.16 dB (73% improvement)

### Method Rankings
| Rank | Method | Avg PSNR | Std Dev | Best Performance | Worst Performance |
|------|--------|----------|---------|------------------|-------------------|
| 1 | P3 (Stochastic) | 31.14 dB | 4.13 dB | 36.79 dB | 25.62 dB |
| 2 | P1 (Simple) | 16.75 dB | 1.74 dB | 18.94 dB | 14.12 dB |
| 3 | P2_50 | 16.73 dB | 1.75 dB | 18.98 dB | 14.12 dB |
| 4 | P2_30 | 16.71 dB | 1.75 dB | 18.97 dB | 14.12 dB |
| 5 | P2_20 | 16.70 dB | 1.75 dB | 18.96 dB | 14.12 dB |
| 6 | P2_10 | 16.69 dB | 1.74 dB | 18.94 dB | 14.11 dB |
| 7 | P2_5 | 16.69 dB | 1.74 dB | 18.92 dB | 14.10 dB |
| 8 | P2_3 | 16.68 dB | 1.74 dB | 18.92 dB | 14.10 dB |

### Image Difficulty Analysis
| Image | Avg PSNR | Std Dev | Difficulty Level |
|-------|----------|---------|------------------|
| D | 21.18 dB | 6.31 dB | Easiest |
| C | 19.22 dB | 5.59 dB | Easy |
| E | 18.50 dB | 4.69 dB | Medium |
| B | 17.53 dB | 3.27 dB | Hard |
| A | 16.12 dB | 5.67 dB | Hardest |

## Output Files Description

### Analysis Plots
- `psnr_comparison.png` - PSNR comparison bar chart
- `psnr_heatmap.png` - Method vs Image heatmap
- `method_average.png` - Method average performance chart
- `psnr_distribution.png` - PSNR distribution box plot
- `image_average.png` - Image average performance chart

### Data Files
- `detailed_results.csv` - Complete PSNR data
- `method_statistics.csv` - Method statistics data
- `image_statistics.csv` - Image statistics data

## Technical Details

### P3 Method Core Algorithm
1. **Multi-directional edge detection**: Calculate edge indicators for 4 main directions
2. **Weight calculation**: Use stochastic policy to calculate directional weights
3. **Intelligent interpolation**: Weighted average interpolation based on weights
4. **Color difference interpolation**: Separate processing of R-G and B-G color differences

### Key Improvements
- Edge-aware Green channel interpolation
- Adaptive weight allocation
- Multi-directional gradient analysis
- Color difference preservation strategy

## Usage Recommendations

### Method Selection Guidelines
- **High quality requirements**: Use P3 method
- **Computational resource constraints**: Use P1 method
- **Edge details important**: Try P2 method (but limited improvement)

### Parameter Tuning
- In P2 method, threshold has minimal impact on performance
- P3 method requires no parameter tuning, stable performance
- Difficult images (like A) may require special handling

## Future Improvements

1. **Algorithm Optimization**
   - Improve P2 method's edge detection algorithm
   - Optimize P3 method's computational efficiency
   - Develop adaptive parameter adjustment

2. **Performance Enhancement**
   - Parallel processing
   - GPU acceleration implementation
   - Memory usage optimization

3. **Feature Extensions**
   - Support for more Bayer patterns
   - Add other evaluation metrics
   - Real-time processing capabilities

## References

- CFA Demosaicing Algorithm Comparison Study
- Stochastic Interpolation Applications in Image Reconstruction
- Edge-Aware Interpolation Method Optimization Strategies

## Contact Information

- Student ID: r14942096
- Project Type: Digital Image Processing Assignment
- Completion Date: October 2024

---

*This project demonstrates the performance comparison of different CFA demosaicing methods, providing practical reference implementations for image processing research.*
