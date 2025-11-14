# Digital Image Processing (DIP) - Course Projects

**Student**: 林祐群 (Arthur Lin)  
**Student ID**: R14942096  
**Institution**: National Taiwan University  
**Course**: Digital Image Processing  
**Semester**: Fall 2025

---

## Repository Overview

This repository contains comprehensive implementations, analyses, and reports for Digital Image Processing coursework. Each homework demonstrates advanced understanding of fundamental image processing techniques through rigorous implementation, experimentation, and evaluation.

## Project Structure

```
DIP/
├── hw1_r14942096/          # CFA Demosaicing Methods (Submitted Version)
│   ├── p1.py, p2.py, p3.py
│   ├── images/
│   ├── report.pdf
│   └── README.md
├── hw2/                     # Auto White Balance & Tone Mapping
│   ├── AWB/                 # AWB implementations
│   ├── Tone_Mapping/        # Tone mapping implementation
│   ├── output/              # Experimental results
│   ├── report.md            # Comprehensive report
│   └── README.md
├── hw3/                     # Frequency Domain Filtering & Image Enhancement
│   ├── hw3_a.py            # Frequency domain filtering
│   ├── hw3_b.py            # Image enhancement pipeline
│   ├── output_a/, output_b/
│   ├── REPORT.md           # Technical report
│   └── README.md
├── hw4/                     # Image Reconstruction & Color Appearance Model
│   ├── part_a.py           # Filtered backprojection reconstruction
│   ├── part_b.py           # Chromatic adaptation transforms
│   ├── output_part_a/, output_part_b_*/
│   ├── report.md           # Comprehensive report
│   └── README.md
├── Reading/                 # Course readings and papers
│   └── [Research papers and materials]
└── README.md               # This file
```

---

## Homework Assignments

### Homework 1: CFA Demosaicing Methods Comparison

**Topic**: Color Filter Array (CFA) Demosaicing Algorithm Implementation and Evaluation

**Implemented Methods**:
1. **P1 - Simple Interpolation**: Basic bilinear interpolation
2. **P2 - Edge-Aware Interpolation**: Adaptive interpolation with edge detection
3. **P3 - Stochastic Interpolation**: Advanced multi-directional weighted interpolation

**Key Results**:
- P3 method achieves **31.14 dB** average PSNR
- **85% improvement** over simple interpolation
- Comprehensive analysis across 5 test images

**Technologies**: Python, OpenCV, NumPy, Matplotlib

**Details**: See [hw1_r14942096/README.md](hw1_r14942096/README.md)

---

### Homework 2: Auto White Balance and Tone Mapping

**Topic**: Illuminant Estimation, Chromatic Adaptation, and Tone Reproduction

**Implemented Methods**:

**Auto White Balance (6 variants)**:
1. **Grey World** (3 options) - Statistical mean-based
2. **White Patch** (2 options) - Maximum value-based
3. **Shades of Gray** (multi-scale) - Minkowski p-norm with Von Kries adaptation

**Tone Mapping**:
- Histogram Matching in YCrCb color space
- Chrominance-aware tone adjustment
- Comprehensive MSE/PSNR evaluation

**Key Results**:
- Multi-scale Shades of Gray: **3.28°** mean angular error (best AWB)
- Grey World Option 3: **193.6** average MSE (best tone mapping)
- **93-99%** MSE reduction through tone mapping

**Key Finding**: 
> Best AWB performance does not guarantee best tone mapping results, highlighting the independence of chromatic adaptation and luminance adjustment.

**Technologies**: Python, OpenCV, NumPy, Color Space Transformations

**Details**: See [hw2/README.md](hw2/README.md)

---

### Homework 3: Frequency Domain Filtering and Image Enhancement

**Topic**: Frequency-Spatial Domain Equivalence and Multi-Stage Image Enhancement

**Part A - Frequency Domain Filtering**:
- 2D Discrete Fourier Transform visualization
- Odd symmetry kernel design (4×4 padded Sobel kernel)
- Frequency-spatial domain equivalence verification
- Correlation implementation using complex conjugate

**Part B - Image Enhancement Pipeline**:
- Notch filtering for periodic noise removal
- Unsharp masking for detail enhancement
- Histogram equalization and CLAHE
- Single-scale Retinex for illumination correction
- Comprehensive ablation study

**Key Results**:
- Frequency-spatial equivalence: **98.77%** pixel match, **47.75 dB** PSNR
- Odd symmetry violation increases MSE by **455×**
- Enhancement pipeline achieves **764%** edge strength increase
- All operations manually implemented (except image I/O)

**Technologies**: Python, NumPy, OpenCV (I/O only)

**Details**: See [hw3/README.md](hw3/README.md)

---

### Homework 4: Image Reconstruction and Color Appearance Model

**Topic**: Filtered Backprojection Reconstruction and Chromatic Adaptation Transforms

**Part A - Filtered Backprojection Image Reconstruction**:
- Radon transform and sinogram generation
- Parallel-beam projection computation
- Frequency-domain ramp filtering (|ω|)
- Hamming window for noise reduction
- Backprojection algorithm implementation
- Angular sampling rate analysis

**Part B - Chromatic Adaptation and Color Constancy**:
- CAT02 chromatic adaptation transform (CIECAM02)
- Bradford chromatic adaptation transform
- Color space conversions (RGB, XYZ, LMS)
- CIEDE2000 color difference calculation
- Illuminant adaptation and color constancy

**Key Results**:
- Reconstruction quality: **23.15 dB** PSNR, **0.9832** SSIM with 0.125° sampling
- Hamming window provides **4.9-8.6 dB** PSNR improvement across all sampling rates
- CAT02 and Bradford show equivalent performance: **5.54** vs **5.55** ΔE*00
- Angular sampling analysis demonstrates logarithmic quality improvement

**Technologies**: Python, NumPy, OpenCV, Colour Science Library

**Details**: See [hw4/README.md](hw4/README.md)

---

## Research Topics & Methods

### Image Enhancement
- Demosaicing algorithms
- White balance correction
- Tone mapping and tone reproduction
- Histogram matching
- Frequency domain filtering
- Multi-stage enhancement pipelines

### Color Science
- Color constancy and illuminant estimation
- Chromatic adaptation (Von Kries, CAT02, Bradford transforms)
- Color space conversions (RGB, XYZ, YCrCb, LMS)
- Ground-truth illuminant evaluation
- Color appearance models (CIECAM02)
- CIEDE2000 color difference metrics

### Frequency Domain Processing
- 2D Discrete Fourier Transform
- Odd symmetry requirements for kernels
- Frequency-spatial domain equivalence
- Notch filtering for periodic noise
- Correlation vs. convolution
- Radon transform and sinogram generation
- Filtered backprojection reconstruction
- Ramp filtering and Hamming window

### Evaluation Metrics
- **PSNR** (Peak Signal-to-Noise Ratio)
- **MSE** (Mean Squared Error)
- **SSIM** (Structural Similarity Index)
- **Angular Error** (Illuminant estimation accuracy)
- **CIEDE2000** (ΔE*00 color difference)
- Statistical analysis (mean, std dev, per-image performance)
- Edge strength and contrast metrics

### Algorithm Design
- Edge-aware processing
- Multi-scale approaches
- Adaptive parameter selection
- Statistical estimators (mean, max, p-norms)
- Manual algorithm implementation

---

## Technical Stack

### Programming Languages
- **Python 3.8+**: Primary implementation language

### Core Libraries
- **OpenCV** (cv2): Image I/O, processing, color space conversion
- **NumPy**: Numerical computations, array operations, FFT
- **Matplotlib**: Visualization and plotting
- **Pandas**: Data analysis and statistics
- **Colour Science**: CIEDE2000 calculation, color science utilities

### Development Tools
- **Shell Scripts**: Batch processing automation
- **Markdown**: Documentation and reporting
- **Git**: Version control

---

## Performance Highlights

### HW1: Demosaicing
```
Method          Avg PSNR    Improvement over P1
─────────────────────────────────────────────────
P3 (Best)       31.14 dB    +14.40 dB (85%)
P1 (Baseline)   16.75 dB    -
P2 variants     16.68-16.73 dB  +0.0-0.02 dB
```

### HW2: AWB & Tone Mapping
```
Method              AWB Error   Tone MSE    Overall Rating
──────────────────────────────────────────────────────────
Multi-scale         3.28°      292.0       Best AWB
Grey World Opt3     3.97°      193.6       Best Balance
White Patch Opt1    5.98°      211.2       Moderate
```

### HW3: Frequency Domain & Enhancement
```
Metric                          Result
─────────────────────────────────────────────────
Frequency-Spatial Match         98.77% pixels
Frequency-Spatial PSNR          47.75 dB
Odd Symmetry Violation Impact   455× MSE increase
Edge Strength Improvement       764% increase
```

### HW4: Image Reconstruction & Color Appearance
```
Part A - Reconstruction         Result
─────────────────────────────────────────────────
Best PSNR (0.125° sampling)    23.15 dB
Best SSIM (0.125° sampling)     0.9832
Hamming Window Improvement      4.9-8.6 dB
─────────────────────────────────────────────────
Part B - Chromatic Adaptation   Result
─────────────────────────────────────────────────
CAT02 ΔE*00                     5.54
Bradford ΔE*00                  5.55
Method Equivalence              Negligible difference
```

---

## Key Learning Outcomes

### Technical Skills
1. **Algorithm Implementation**: From research papers to working code
2. **Image Processing**: Demosaicing, AWB, tone mapping, frequency domain filtering
3. **Color Science**: Color spaces, chromatic adaptation, color constancy
4. **Evaluation Methodology**: Quantitative metrics, statistical analysis
5. **Optimization**: Performance tuning, parameter selection
6. **Manual Implementation**: Custom algorithms without library dependencies

### Analytical Skills
1. **Critical Analysis**: Understanding algorithm assumptions and limitations
2. **Comparative Evaluation**: Multi-method performance comparison
3. **Visual Inspection**: Qualitative assessment complementing quantitative metrics
4. **Research Synthesis**: Connecting theory with experimental results
5. **Ablation Studies**: Systematic component contribution analysis

### Software Engineering
1. **Modular Design**: Reusable functions and utilities
2. **Batch Processing**: Automated testing across datasets
3. **Documentation**: Comprehensive README and reports
4. **Code Quality**: Clear structure, error handling, command-line interfaces

---

## Documentation Standards

Each homework includes:
- **Comprehensive README**: Quick start, usage, results
- **Technical Report**: Methodology, implementation, analysis
- **Source Code**: Well-documented Python implementations
- **Evaluation Scripts**: Automated testing and metrics
- **Visual Results**: Output images and visualizations
- **Performance Data**: Quantitative results in CSV/TXT format

---

## Quick Start

### Prerequisites

```bash
# Install Python 3.8+
brew install python3

# Install required libraries
pip3 install opencv-python numpy matplotlib pandas
```

### Run Experiments

```bash
# Navigate to specific homework
cd hw1_r14942096    # or hw2, hw3, hw4

# Read the README
cat README.md

# Run batch processing (each homework has automated scripts)
# See individual README for specific commands
```

---

## Repository Statistics

### Code Metrics (Approximate)
- **Total Lines of Code**: ~4,000+ lines
- **Python Files**: 35+ implementations
- **Test Images**: 15+ across all homeworks
- **Output Results**: 200+ images and data files
- **Documentation**: 5,000+ lines

### Implementation Coverage
- 3 Demosaicing algorithms (HW1)
- 6 AWB algorithm variants (HW2)
- 1 Tone mapping algorithm (HW2)
- Frequency domain filtering pipeline (HW3)
- Multi-stage image enhancement pipeline (HW3)
- Filtered backprojection reconstruction (HW4)
- 2 Chromatic adaptation transforms (HW4)
- Multiple evaluation utilities
- Comprehensive analysis scripts

---

## Research Insights

### HW1 Insights
1. **Stochastic methods** significantly outperform simple interpolation
2. **Edge-aware processing** is crucial for artifact reduction
3. **Multi-directional analysis** improves interpolation quality

### HW2 Insights
1. **Multi-scale approaches** provide superior robustness
2. **AWB and tone mapping** are functionally independent
3. **Scene characteristics** significantly impact algorithm performance
4. **Statistical assumptions** determine method suitability

### HW3 Insights
1. **Odd symmetry** is essential for frequency-spatial domain equivalence
2. **Multi-stage pipelines** enable cumulative improvements
3. **Manual implementation** provides deep understanding of algorithms
4. **Ablation studies** reveal individual component contributions

### HW4 Insights
1. **Angular sampling rate** critically affects reconstruction quality
2. **Hamming window** provides substantial noise reduction in filtered backprojection
3. **CAT02 and Bradford** demonstrate equivalent performance for chromatic adaptation
4. **Radon transform** enables image reconstruction from projections
5. **CIEDE2000** provides perceptually uniform color difference measurement

### Cross-Homework Observations
1. **Simple methods** often provide baseline performance
2. **Advanced methods** require parameter tuning but offer substantial improvements
3. **Evaluation metrics** must align with application goals
4. **Visual inspection** complements quantitative analysis
5. **Theoretical understanding** enables effective algorithm design

---

## Reading Materials

The `Reading/` directory contains relevant research papers and course materials covering:
- Color filter array demosaicing techniques
- Color constancy and white balance algorithms
- Tone mapping and HDR imaging
- Image quality assessment metrics
- Frequency domain filtering methods
- Image reconstruction and Radon transform
- Color appearance models and chromatic adaptation

---

## Comparative Analysis

### Algorithm Families

| Family | HW1 Application | HW2 Application | HW3 Application | HW4 Application | Key Principle |
|--------|----------------|-----------------|-----------------|----------------|---------------|
| Statistical | Mean-based interpolation | Grey World, Shades of Gray | - | - | Average properties |
| Edge-based | P2 adaptive | - | Sobel filtering | - | Gradient analysis |
| Optimization | P3 stochastic | - | - | - | Weighted combination |
| Transform | - | Von Kries adaptation | DFT/FFT | Radon, CAT02, Bradford | Diagonal/spectral mapping |
| Histogram | - | Tone mapping | Histogram equalization | - | CDF matching |
| Frequency | - | - | Notch filtering | Ramp filtering | Spectral filtering |
| Reconstruction | - | - | - | Backprojection | Projection integration |

### Performance Patterns

**Consistent Observations**:
1. Simple methods provide baselines but limited quality
2. Advanced methods significantly outperform baselines
3. Multi-scale/adaptive approaches improve robustness
4. Scene characteristics strongly influence results
5. No single method optimal for all cases
6. Theoretical foundations enable reliable implementations

---

## Learning Journey

### Progression

**HW1 → HW2 → HW3 → HW4**:
- From **spatial interpolation** to **color correction** to **frequency domain** to **image reconstruction**
- From **single metric (PSNR)** to **dual metrics** to **comprehensive analysis** to **multi-metric evaluation**
- From **3 methods** to **6+ variants** to **multi-stage pipelines** to **transform comparison**
- From **single-stage** to **multi-stage pipeline** to **ablation studies** to **theoretical validation**

### Skills Developed
1. Image processing algorithm implementation
2. Color science and chromatic adaptation
3. Frequency domain processing and FFT
4. Image reconstruction from projections
5. Statistical analysis and evaluation
6. Technical writing and reporting
7. Software engineering practices
8. Manual algorithm implementation
9. Transform theory and implementation

---

## Future Work

### Potential Extensions
1. **Deep Learning Integration**: CNN-based demosaicing, AWB, and enhancement
2. **Real-Time Processing**: GPU acceleration, optimization
3. **Advanced Metrics**: Perceptual quality (SSIM, LPIPS)
4. **Interactive Tools**: GUI for parameter tuning
5. **Larger Datasets**: Comprehensive evaluation on diverse images
6. **Multi-Scale Retinex**: Improved illumination correction
7. **3D Reconstruction**: Extension to 3D filtered backprojection
8. **Advanced CAT Models**: Implementation of additional chromatic adaptation transforms

### Research Directions
1. Joint optimization of multi-stage pipelines
2. Scene-adaptive algorithm selection
3. Physics-based color constancy
4. HDR imaging and advanced tone mapping
5. Mobile deployment and edge computing
6. Advanced frequency domain techniques
7. Computed tomography applications
8. Perceptual color difference optimization

---

## Support & Feedback

For questions or suggestions regarding this coursework:
- Check individual homework README files
- Review comprehensive reports (report.md, REPORT.md)
- Consult analysis documents and output results

---

## Version History

- **v4.0** (Dec 2025): HW4 - Image Reconstruction and Color Appearance Model
- **v3.0** (Nov 2025): HW3 - Frequency Domain Filtering and Image Enhancement
- **v2.0** (Oct 2025): HW2 - Auto White Balance and Tone Mapping
- **v1.0** (Sep 2025): HW1 - CFA Demosaicing Methods

---

## Highlights

### Technical Achievements
- 12+ algorithm implementations
- 10+ evaluation scripts
- 200+ output images and visualizations
- Comprehensive documentation (5,000+ lines)
- Reproducible experimental pipelines
- Manual implementation of complex algorithms
- Transform theory validation

### Research Contributions
- Multi-scale AWB approach validation
- AWB-tone mapping independence discovery
- Scene-dependent performance analysis
- Frequency-spatial domain equivalence verification
- Multi-stage enhancement pipeline design
- Filtered backprojection reconstruction analysis
- Chromatic adaptation transform comparison
- Practical algorithm selection guidelines

### Academic Quality
- Rigorous experimental methodology
- Proper citation of research papers
- Statistical significance testing
- Visual and quantitative validation
- Critical analysis and discussion
- Comprehensive ablation studies

---

**This repository represents a comprehensive study of fundamental digital image processing techniques, from low-level sensor data reconstruction to high-level color correction, tone reproduction, frequency domain processing, image reconstruction, and color appearance modeling.**

---

*Last Updated: December 2025*  
*Course: Digital Image Processing, National Taiwan University*
