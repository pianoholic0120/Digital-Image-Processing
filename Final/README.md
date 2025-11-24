# DSO-SLAM: Direct Sparse Odometry for macOS

Enhanced DSO (Direct Sparse Odometry) implementation with macOS (Apple Silicon) support, OpenCV 4 compatibility, and real-time USB camera input.

## Features

- **Real-time Visual Odometry**: Monocular SLAM with direct sparse tracking
- **USB Camera Support**: Live camera feed processing with interactive controls
- **Image Preprocessing**: Photometric calibration, vignetting removal, and enhancement pipeline
- **Data Export**: Automatic export of point clouds, camera poses, and video
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

Generate calibration file using `calibration.py`:
```bash
python calibration.py
```

This creates `calib.npz` with camera intrinsics, distortion coefficients, vignette mask, and CRF LUT.

### Image Preprocessing

Enhance images for better SLAM performance:
```bash
python cv.py --input_path ./path/to/input/images --output_path ./path/to/output/images
```

The preprocessing pipeline includes:
- Gamma correction
- Photometric calibration
- Exposure compensation
- Vignetting removal
- Undistortion
- Brightness/contrast enhancement

### Running DSO-SLAM

#### USB Camera Mode
```bash
cd dso/build
bin/dso_dataset camera=0 calib=/path/to/camera.txt preset=0 mode=2
```

**Interactive Controls:**
- Press `s` to start processing
- Press `e` to stop and save results

#### Image Sequence Mode
```bash
bin/dso_dataset files=/path/to/images calib=/path/to/camera.txt preset=0 mode=2
```

### Command-Line Arguments

| Argument | Description | Options |
|----------|-------------|---------|
| `camera=N` | USB camera device index | `0` (default), `1`, `2`, ... |
| `files=XXX` | Path to image folder or ZIP archive | - |
| `calib=XXX` | Path to camera calibration file | Required |
| `preset=N` | Processing preset | `0`=default, `1`=real-time, `2`=fast |
| `mode=N` | Photometric mode | `0`=with calib, `1`=no calib, `2`=no distortion |
| `gamma=XXX` | Photometric gamma calibration file | Optional |
| `vignette=XXX` | Vignetting mask image | Optional |

### Output

Results are saved in `dso_output/`:
- `camera_poses.txt`: Camera trajectory (TUM format: timestamp tx ty tz qx qy qz qw)
- `point_cloud.ply`: 3D point cloud with colors
- `output_video.mp4`: Processed video (if available)
- `result.txt`: DSO internal results

## Project Structure

```
.
├── dso/                    # DSO source code
│   ├── src/
│   │   ├── main_dso_pangolin.cpp
│   │   ├── util/           # CameraReader, DataExporter
│   │   └── IOWrapper/      # Input/Output wrappers
│   └── build/              # Build directory
├── Pangolin/               # Visualization library
├── utils/                  # Image processing utilities
│   ├── usb_baseline_pipeline.py
│   └── build.py
├── calibration.py          # Camera calibration script
└── cv.py                   # Image preprocessing script
```

## Key Features

### Real-time Processing
- Thread-safe camera access
- Interactive keyboard controls
- Real-time 3D visualization

### Image Enhancement
- Photometric calibration pipeline
- Adaptive exposure compensation
- Conservative brightness/contrast enhancement (maintains photometric consistency for SLAM)

### Data Export
- Point cloud export from all keyframes
- Complete camera trajectory (all frames, not just keyframes)
- Video export of processed frames

## Troubleshooting

**Build errors:**
- Ensure all dependencies are installed via Homebrew
- Check Boost version (1.89+ required)
- Verify CMake finds all libraries

**Camera not detected:**
- Check camera permissions in System Preferences
- Try different camera indices
- Ensure camera is not used by another application

**Poor tracking quality:**
- Use accurate camera calibration
- Ensure sufficient scene texture
- Avoid excessive image enhancement (maintains photometric consistency)
- Use appropriate preset for hardware capabilities

**Scale drift in turns:**
- Reduce brightness/contrast enhancement parameters
- Ensure consistent lighting conditions
- Use conservative preprocessing settings

## License

GNU General Public License Version 3 (GPLv3)

Based on DSO: https://github.com/JakobEngel/dso

## References

- **Direct Sparse Odometry**, J. Engel, V. Koltun, D. Cremers, arXiv:1607.02565, 2016
- **A Photometrically Calibrated Benchmark For Monocular Visual Odometry**, J. Engel, V. Usenko, D. Cremers, arXiv:1607.02555, 2016
