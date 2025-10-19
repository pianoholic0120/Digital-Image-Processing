# Digital Image Processing (DIP) - Course Projects

**Student**: 林祐群 (Arthur Lin)  
**Student ID**: R14942096  
**Institution**: National Taiwan University  
**Course**: Digital Image Processing  
**Semester**: Fall 2025

---

## 📚 Repository Overview

This repository contains comprehensive implementations, analyses, and reports for Digital Image Processing coursework. Each homework demonstrates advanced understanding of fundamental image processing techniques through rigorous implementation, experimentation, and evaluation.

## 📂 Project Structure

```
DIP/
├── hw1/                          # CFA Demosaicing Methods
│   ├── [Implementation files]
│   └── README.md
├── hw1_r14942096/               # Submitted version
│   ├── report.pdf
│   └── [Complete deliverables]
├── hw2/                          # Auto White Balance & Tone Mapping
│   ├── AWB/                     # AWB implementations
│   ├── Tone_Mapping/            # Tone mapping implementation
│   ├── output/                  # Experimental results
│   ├── report.md                # Comprehensive report
│   └── README.md
├── Reading/                      # Course readings and papers
└── README.md                    # This file
```

---

## 🎓 Homework Assignments

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

📁 **Details**: See [hw1/README.md](hw1/README.md)

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

📁 **Details**: See [hw2/README.md](hw2/README.md)

---

## 🔬 Research Topics & Methods

### Image Enhancement
- Demosaicing algorithms
- White balance correction
- Tone mapping and tone reproduction
- Histogram matching

### Color Science
- Color constancy and illuminant estimation
- Chromatic adaptation (Von Kries transformation)
- Color space conversions (RGB, YCrCb)
- Ground-truth illuminant evaluation

### Evaluation Metrics
- **PSNR** (Peak Signal-to-Noise Ratio)
- **MSE** (Mean Squared Error)
- **Angular Error** (Illuminant estimation accuracy)
- Statistical analysis (mean, std dev, per-image performance)

### Algorithm Design
- Edge-aware processing
- Multi-scale approaches
- Adaptive parameter selection
- Statistical estimators (mean, max, p-norms)

---

## 🛠️ Technical Stack

### Programming Languages
- **Python 3.8+**: Primary implementation language

### Core Libraries
- **OpenCV** (cv2): Image I/O, processing, color space conversion
- **NumPy**: Numerical computations, array operations
- **Matplotlib**: Visualization and plotting
- **Pandas**: Data analysis and statistics

### Development Tools
- **Shell Scripts**: Batch processing automation
- **Markdown**: Documentation and reporting
- **Git**: Version control

---

## 📊 Performance Highlights

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

---

## 🎯 Key Learning Outcomes

### Technical Skills
1. **Algorithm Implementation**: From research papers to working code
2. **Image Processing**: Demosaicing, AWB, tone mapping
3. **Color Science**: Color spaces, chromatic adaptation, color constancy
4. **Evaluation Methodology**: Quantitative metrics, statistical analysis
5. **Optimization**: Performance tuning, parameter selection

### Analytical Skills
1. **Critical Analysis**: Understanding algorithm assumptions and limitations
2. **Comparative Evaluation**: Multi-method performance comparison
3. **Visual Inspection**: Qualitative assessment complementing quantitative metrics
4. **Research Synthesis**: Connecting theory with experimental results

### Software Engineering
1. **Modular Design**: Reusable functions and utilities
2. **Batch Processing**: Automated testing across datasets
3. **Documentation**: Comprehensive README and reports
4. **Code Quality**: Clear structure, error handling, command-line interfaces

---

## 📖 Documentation Standards

Each homework includes:
- ✅ **Comprehensive README**: Quick start, usage, results
- ✅ **Technical Report**: Methodology, implementation, analysis
- ✅ **Source Code**: Well-documented Python implementations
- ✅ **Evaluation Scripts**: Automated testing and metrics
- ✅ **Visual Results**: Output images and visualizations
- ✅ **Performance Data**: Quantitative results in CSV/TXT format

---

## 🚀 Quick Start (Any Homework)

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
cd hw1    # or hw2

# Read the README
cat README.md

# Run batch processing (each homework has automated scripts)
# See individual README for specific commands
```

---

## 📈 Repository Statistics

### Code Metrics (Approximate)
- **Total Lines of Code**: ~2,000+ lines
- **Python Files**: 20+ implementations
- **Test Images**: 10+ (5 per homework)
- **Output Results**: 100+ images and data files
- **Documentation**: 3,000+ lines

### Implementation Coverage
- ✅ 3 Demosaicing algorithms (HW1)
- ✅ 6 AWB algorithm variants (HW2)
- ✅ 1 Tone mapping algorithm (HW2)
- ✅ Multiple evaluation utilities
- ✅ Comprehensive analysis scripts

---

## 🔍 Research Insights

### HW1 Insights
1. **Stochastic methods** significantly outperform simple interpolation
2. **Edge-aware processing** is crucial for artifact reduction
3. **Multi-directional analysis** improves interpolation quality

### HW2 Insights
1. **Multi-scale approaches** provide superior robustness
2. **AWB and tone mapping** are functionally independent
3. **Scene characteristics** significantly impact algorithm performance
4. **Statistical assumptions** determine method suitability

### Cross-Homework Observations
1. **Simple methods** often provide baseline performance
2. **Advanced methods** require parameter tuning but offer substantial improvements
3. **Evaluation metrics** must align with application goals
4. **Visual inspection** complements quantitative analysis

---

## 📚 Reading Materials

The `Reading/` directory contains relevant research papers and course materials covering:
- Color filter array demosaicing techniques
- Color constancy and white balance algorithms
- Tone mapping and HDR imaging
- Image quality assessment metrics

---

## 🎨 Visual Results Gallery

### HW1: Demosaicing Results
- Original Bayer patterns
- P1 simple interpolation outputs
- P2 edge-aware results (6 threshold variants)
- P3 stochastic interpolation outputs
- Comparative analysis visualizations

### HW2: AWB & Tone Mapping Results
- 30 white-balanced images (6 methods × 5 images)
- 30 tone-mapped outputs
- 30 tone curve visualizations
- Method comparison figures
- Complete pipeline demonstrations

---

## 💡 Best Practices Demonstrated

### Code Quality
- Modular function design
- Comprehensive error handling
- Command-line interfaces with argparse
- Batch processing automation
- Clear code documentation

### Experimental Methodology
- Ground-truth comparison
- Multiple evaluation metrics
- Statistical significance analysis
- Per-image and aggregate results
- Visual validation

### Documentation
- Academic-quality reports
- Clear usage instructions
- Performance summaries
- Troubleshooting guides
- Code availability

---

## 🔧 Utilities and Tools

### Evaluation Scripts
- `psnr_check.py` (HW1): PSNR calculation for demosaicing
- `evaluate_AWB.py` (HW2): Angular error for AWB
- `p1.py` (Tone_Mapping): MSE/PSNR for tone mapping

### Batch Processing
- Shell scripts for automated processing
- Support for multiple input images
- Organized output directory structure

### Analysis Tools
- Performance comparison scripts
- Statistical analysis utilities
- Visualization generators

---

## 📊 Comparative Analysis

### Algorithm Families

| Family | HW1 Application | HW2 Application | Key Principle |
|--------|----------------|-----------------|---------------|
| Statistical | Mean-based interpolation | Grey World, Shades of Gray | Average properties |
| Edge-based | P2 adaptive | - | Gradient analysis |
| Optimization | P3 stochastic | - | Weighted combination |
| Transform | - | Von Kries adaptation | Diagonal mapping |
| Histogram | - | Tone mapping | CDF matching |

### Performance Patterns

**Consistent Observations**:
1. Simple methods provide baselines but limited quality
2. Advanced methods significantly outperform baselines
3. Multi-scale/adaptive approaches improve robustness
4. Scene characteristics strongly influence results
5. No single method optimal for all cases

---

## 🎓 Learning Journey

### Progression

**HW1 → HW2**:
- From **spatial interpolation** to **color correction**
- From **single metric (PSNR)** to **dual metrics (Angular Error + MSE/PSNR)**
- From **3 methods** to **6+ variants**
- From **single-stage** to **multi-stage pipeline**

### Skills Developed
1. Image processing algorithm implementation
2. Color science and chromatic adaptation
3. Statistical analysis and evaluation
4. Technical writing and reporting
5. Software engineering practices

---

## 🔮 Future Work

### Potential Extensions
1. **Deep Learning Integration**: CNN-based demosaicing and AWB
2. **Real-Time Processing**: GPU acceleration, optimization
3. **Advanced Metrics**: Perceptual quality (SSIM, LPIPS)
4. **Interactive Tools**: GUI for parameter tuning
5. **Larger Datasets**: Comprehensive evaluation on diverse images

### Research Directions
1. Joint optimization of multi-stage pipelines
2. Scene-adaptive algorithm selection
3. Physics-based color constancy
4. HDR imaging and advanced tone mapping
5. Mobile deployment and edge computing

---

## 📞 Support & Feedback

For questions or suggestions regarding this coursework:
- Check individual homework README files
- Review comprehensive reports (report.md)
- Consult analysis documents (COMPREHENSIVE_ANALYSIS.md)

---

## 📝 Version History

- **v2.0** (Oct 2025): HW2 - Auto White Balance and Tone Mapping
- **v1.0** (Sep 2025): HW1 - CFA Demosaicing Methods

---

## 🏆 Highlights

### Technical Achievements
- ✅ 10+ algorithm implementations
- ✅ 6+ evaluation scripts
- ✅ 130+ output images and visualizations
- ✅ Comprehensive documentation (3,000+ lines)
- ✅ Reproducible experimental pipelines

### Research Contributions
- ✅ Multi-scale AWB approach validation
- ✅ AWB-tone mapping independence discovery
- ✅ Scene-dependent performance analysis
- ✅ Practical algorithm selection guidelines

### Academic Quality
- ✅ Rigorous experimental methodology
- ✅ Proper citation of research papers
- ✅ Statistical significance testing
- ✅ Visual and quantitative validation
- ✅ Critical analysis and discussion

---

**This repository represents a comprehensive study of fundamental digital image processing techniques, from low-level sensor data reconstruction to high-level color correction and tone reproduction.**

---

*Last Updated: October 19, 2025*  
*Course: Digital Image Processing, National Taiwan University*
