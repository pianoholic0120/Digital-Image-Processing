# DSO-SLAM: Direct Sparse Odometry for macOS

Enhanced DSO (Direct Sparse Odometry) implementation with macOS (Apple Silicon) support, OpenCV 4 compatibility, real-time USB camera input, video file support, and dual-mode reconstruction pipeline.

## Features

- **Real-time Visual Odometry**: Monocular SLAM with direct sparse tracking
- **Multiple Input Sources**: USB camera, video files, and image sequences
- **Dual-Mode Reconstruction**: Compare raw and processed pipeline results side-by-side
- **Image Preprocessing Pipeline**: Photometric calibration, gamma correction, exposure compensation, vignetting removal, and geometric undistortion
- **Automatic Data Export**: Point clouds, camera poses, quantitative metrics, and videos
- **macOS Optimized**: Full support for Apple Silicon (ARM64) architecture

## Requirements

### System
- macOS (tested on Apple Silicon)
- CMake 3.10+
- C++14 compiler

### Dependencies
```bash
brew install eigen suitesparse boost opencv
```

## Installation

### 1. Build Pangolin
```bash
cd Pangolin
mkdir build && cd build
cmake ..
make -j4
cd ../..
```

### 2. Build DSO
```bash
cd dso
mkdir build && cd build
cmake ..
make -j4
```

The executable will be available at `dso/build/bin/dso_dataset`.

## Usage

### Camera Calibration

#### Geometric Calibration
Generate camera calibration file using `utils/calibration.py`:
```bash
python utils/calibration.py
```

This creates `camera.txt` with camera intrinsics and distortion coefficients.

#### Photometric Calibration
Generate photometric calibration using `online_photometric_calibration`:
```bash
cd online_photometric_calibration/build
./bin/online_pcalib_demo --input-dir /path/to/images --output-dir /path/to/output --no-wait
```

This creates:
- `pcalib.txt`: Inverse camera response function (256 values)
- `vignette.png`: Vignetting mask (16-bit grayscale PNG)

### Running DSO-SLAM

#### USB Camera Mode
```bash
cd dso/build
bin/dso_dataset camera=0 calib=/path/to/camera.txt gamma=/path/to/pcalib.txt vignette=/path/to/vignette.png dual=1
```

**Interactive Controls:**
- Press `s` to start processing
- Press `e` to stop and save results

**Options:**
- `save_video=1`: Enable video recording (default: 0, disabled to reduce frame drops)
- `dual=0`: Raw path only (photometric calibration only)
- `dual=1`: Both raw and pipeline paths (side-by-side comparison)
- `dual=2`: Pipeline path only (full preprocessing pipeline)

#### Video File Mode
```bash
bin/dso_dataset video=/path/to/video.mp4 calib=/path/to/camera.txt gamma=/path/to/pcalib.txt vignette=/path/to/vignette.png dual=1
```

#### Image Sequence Mode
```bash
bin/dso_dataset files=/path/to/images calib=/path/to/camera.txt gamma=/path/to/pcalib.txt vignette=/path/to/vignette.png
```

### Command-Line Arguments

| Argument | Description | Options |
|----------|-------------|---------|
| `camera=N` | USB camera device index | `0` (default), `1`, `2`, ... |
| `video=XXX` | Path to input video file | `.mp4`, `.avi`, `.mov`, etc. |
| `files=XXX` | Path to image folder or ZIP archive | - |
| `calib=XXX` | Path to camera calibration file | Required |
| `gamma=XXX` | Photometric calibration file (`pcalib.txt`) | Optional, recommended |
| `vignette=XXX` | Vignetting mask image (`vignette.png`) | Optional, recommended |
| `dual=N` | Dual-mode reconstruction | `0`=raw only, `1`=both, `2`=pipeline only |
| `save_video=N` | Save video in camera mode | `0`=disabled (default), `1`=enabled |
| `clahe=N` | Enable CLAHE in pipeline | `0`=disabled (default), `1`=enabled |
| `preset=N` | Processing preset | `0`=default, `1`=real-time, `2`=fast |
| `mode=N` | Photometric mode | `0`=with calib, `1`=no calib, `2`=no distortion |

### Preprocessing Pipeline

The pipeline path (`dual=1` or `dual=2`) applies the following processing steps in order:

1. **Gamma Correction**: Linearize gamma-compressed RGB image (γ=2.2)
2. **Fixed Gain Exposure Compensation**: Apply uniform scaling factor computed from first frame
3. **Grayscale Conversion**: Convert to single-channel using BT.709 weights
4. **Bilateral Filter Denoising**: Light edge-preserving noise reduction
5. **Photometric Undistortion**: Apply CRF inverse (`pcalib.txt`) and vignetting correction (`vignette.png`)
6. **Geometric Undistortion**: Apply lens distortion correction using camera intrinsics

**Note**: The raw path (`dual=0`) only applies photometric undistortion (CRF + vignette) without the preprocessing pipeline.

### Output

Results are saved based on the `dual` mode:

#### Dual Mode (`dual=1`)
- `dso_output/raw/`: Raw path reconstruction
  - `camera_poses.txt`: Camera trajectory (TUM format)
  - `point_cloud.ply`: 3D point cloud with colors
  - `quantitative_metrics.txt`: Comprehensive evaluation metrics
  - `output_video.mp4`: Processed video (if available)
- `dso_output/pipeline/`: Pipeline path reconstruction
  - Same files as above

#### Single Mode (`dual=0` or `dual=2`)
- `dso_output/raw/` (for `dual=0`) or `dso_output/pipeline/` (for `dual=2`)
  - Same files as above

#### Camera Mode Additional Output
- `dso_output/recorded_camera_video.mp4`: Complete recorded video (only if `save_video=1`)

### Quantitative Metrics

The `quantitative_metrics.txt` file includes:

**Trajectory Quality:**
- Translation Smoothness
- Rotation Smoothness
- Tracking Robustness
- Temporal Consistency
- Trajectory Drift
- Relative Pose Error (RPE)
- Scale Drift

**Point Cloud Quality:**
- Total Points
- Density (points/m³)
- Uniformity
- Map Coverage Ratio
- Completeness

**System Performance:**
- Processing Speed (FPS)
- Average Latency per Frame
- Keyframe Ratio

## Project Structure

```
.
├── dso/                           # DSO source code
│   ├── src/
│   │   ├── main_dso_pangolin.cpp  # Main executable
│   │   ├── util/                  # CameraReader, DataExporter, PipelineProcessor
│   │   └── IOWrapper/             # Input/Output wrappers
│   │       └── Pangolin/          # DualPangolinDSOViewer, PangolinDSOViewer
│   └── build/                     # Build directory
├── Pangolin/                      # Visualization library
├── online_photometric_calibration/ # Photometric calibration tool
├── utils/                         # Image processing utilities
│   ├── calibration.py            # Camera geometric calibration
│   └── video_to_images.py        # Video to image sequence converter
├── main.sh                        # Automated pipeline script
└── visualize_pipeline.py         # Pipeline visualization tool
```

## Automated Pipeline

Use `main.sh` to automate the entire process from video to reconstruction:

```bash
./main.sh /path/to/input/folder
```

This script will:
1. Find video files in the input folder
2. Convert video to image sequence
3. Run photometric calibration
4. Generate camera calibration file (if missing)
5. Run DSO reconstruction
6. Save results to `input_folder/dso_output/`

## Key Features

### Real-time Processing
- Thread-safe camera access
- Interactive keyboard controls
- Real-time 3D visualization with Pangolin
- Side-by-side comparison in dual mode

### Image Preprocessing
- **Gamma Correction**: Linearizes gamma-compressed images for direct methods
- **Fixed Gain Exposure**: Maintains photometric consistency across frames
- **Bilateral Filtering**: Edge-preserving denoising without gradient loss
- **Photometric Calibration**: CRF and vignetting correction
- **Geometric Undistortion**: Lens distortion correction

### Data Export
- Point cloud export from all keyframes
- Complete camera trajectory (all frames, not just keyframes)
- Quantitative metrics calculation
- Video export of processed frames
- Automatic metrics calculation from exported files

## Troubleshooting

**Build errors:**
- Ensure all dependencies are installed via Homebrew
- Check Boost version (1.89+ required)
- Verify CMake finds all libraries

**Camera not detected:**
- Check camera permissions in System Preferences → Security & Privacy
- Try different camera indices (`camera=0`, `camera=1`, etc.)
- Ensure camera is not used by another application

**Poor tracking quality:**
- Use accurate camera calibration (geometric + photometric)
- Ensure sufficient scene texture
- Use photometric calibration files (`pcalib.txt`, `vignette.png`)
- Avoid excessive motion during initialization
- Use appropriate preset for hardware capabilities

**Frame drops in camera mode:**
- Disable video recording: `save_video=0` (default)
- Use `preset=2` for faster processing
- Reduce image resolution in camera settings

**Pangolin viewer not closing:**
- The viewer should close automatically after export
- Press `Ctrl+C` if it hangs (data is already saved)

## License

GNU General Public License Version 3 (GPLv3)

Based on DSO: https://github.com/JakobEngel/dso

## References

- **Direct Sparse Odometry**, J. Engel, V. Koltun, D. Cremers, arXiv:1607.02565, 2016
- **A Photometrically Calibrated Benchmark For Monocular Visual Odometry**, J. Engel, V. Usenko, D. Cremers, arXiv:1607.02555, 2016
