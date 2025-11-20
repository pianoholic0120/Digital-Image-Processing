# DSO-SLAM: Direct Sparse Odometry with Real-time Camera Support

This project is an enhanced version of DSO (Direct Sparse Odometry) adapted for macOS (Apple Silicon) with OpenCV 4 support and real-time USB camera input capabilities.

## Overview

DSO-SLAM is a direct sparse visual odometry system that performs real-time 3D reconstruction and camera pose estimation from monocular camera input. This implementation extends the original DSO with the following features:

- **OpenCV 4 Compatibility**: Updated to work with OpenCV 4 API changes
- **macOS Support**: Optimized for Apple Silicon (ARM64) architecture
- **Real-time USB Camera Input**: Support for live camera feed processing
- **Data Export**: Automatic export of point clouds, camera poses, and video
- **Interactive Controls**: Keyboard controls for camera mode (start/stop processing)

## Features

### Real-time Processing
- Process images from USB cameras or pre-recorded image sequences
- Interactive keyboard controls: press 's' to start, 'e' to stop and save
- Real-time 3D visualization using Pangolin
- Automatic data export upon completion

### Data Export
- **Point Cloud**: Exported in PLY format with color information
- **Camera Poses**: TUM format trajectory file with all processed frames
- **Video**: MP4 video export of captured frames

### Visualization
- 3D point cloud rendering
- Camera trajectory visualization
- Real-time feature point display
- Interactive GUI controls

## Requirements

### System Requirements
- macOS (tested on Apple Silicon)
- CMake 3.10 or higher
- C++14 compatible compiler

### Dependencies

#### Required
- **Eigen3**: Linear algebra library
  ```bash
  brew install eigen
  ```

- **SuiteSparse**: Sparse matrix library
  ```bash
  brew install suitesparse
  ```

- **Boost**: C++ libraries (version 1.89+)
  ```bash
  brew install boost
  ```

#### Optional but Recommended
- **OpenCV 4**: Image processing and camera support
  ```bash
  brew install opencv
  ```

- **Pangolin**: 3D visualization library
  - Included as a submodule in this repository
  - Build instructions provided below

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd DIP/Final
```

### 2. Build Pangolin

Pangolin is required for visualization. Build it first:

```bash
cd Pangolin
mkdir build && cd build
cmake ..
make -j4
cd ../..
```

### 3. Build DSO

```bash
cd dso
mkdir build && cd build
cmake ..
make -j4
```

The build process will create:
- `lib/libdso.a`: Static library
- `bin/dso_dataset`: Main executable

## Usage

### Camera Calibration

Before running DSO, you need a camera calibration file. Use the provided `calibration.py` script or create a `camera.txt` file manually.

#### Camera Calibration File Format

```
fx fy cx cy 0
width height
none
width height
```

Where:
- `fx, fy, cx, cy`: Camera intrinsic parameters (normalized to image dimensions)
- `width, height`: Image resolution

### Running with USB Camera

To process live camera feed:

```bash
cd dso/build
bin/dso_dataset camera=0 calib=/path/to/camera.txt preset=0 mode=2
```

**Interactive Controls:**
- Press **'s'** to start processing
- Press **'e'** to stop processing and save all data

**Arguments:**
- `camera=N`: Camera device index (0 for default camera)
- `calib=XXX`: Path to camera calibration file
- `preset=0`: Processing preset (0=default, 1=real-time, 2=fast)
- `mode=2`: Photometric mode (0=with calibration, 1=no calibration, 2=no distortion)

### Running with Image Sequence

To process a pre-recorded image sequence:

```bash
bin/dso_dataset files=/path/to/images calib=/path/to/camera.txt preset=0 mode=2
```

**Arguments:**
- `files=XXX`: Path to image folder or ZIP archive
- `calib=XXX`: Path to camera calibration file
- Other options same as above

### Output Files

After processing completes, the following files are saved in `dso_output/`:

- `camera_poses.txt`: Camera trajectory in TUM format (timestamp tx ty tz qx qy qz qw)
- `point_cloud.ply`: 3D point cloud in PLY format
- `output_video.mp4`: Video of captured frames (if available)
- `result.txt`: DSO internal result file

## Project Structure

```
.
├── dso/                    # Main DSO source code
│   ├── src/               # Source files
│   │   ├── main_dso_pangolin.cpp  # Main executable
│   │   ├── util/          # Utilities including CameraReader and DataExporter
│   │   └── IOWrapper/     # Input/Output wrappers
│   ├── build/             # Build directory
│   └── CMakeLists.txt     # Build configuration
├── Pangolin/              # Pangolin visualization library
├── calibration.py         # Camera calibration script
└── README.md             # This file
```

## Key Modifications

### macOS Compatibility
- Fixed OpenGL compatibility issues for macOS
- Implemented main thread requirements for GUI operations
- Added thread-safe camera access
- Fixed Boost library linking for Homebrew installations

### Real-time Camera Support
- Added `CameraReader` class for USB camera input
- Implemented automatic image resizing to match calibration
- Added keyboard controls for interactive operation

### Data Export
- Implemented point cloud export from Pangolin viewer
- Enhanced camera pose export to include all frames (not just keyframes)
- Added video export functionality

## Troubleshooting

### Build Issues

**Boost not found:**
- Ensure Boost 1.89+ is installed via Homebrew
- Check that `BOOST_ROOT` is set correctly in CMakeLists.txt

**OpenGL errors on macOS:**
- The project includes fixes for macOS OpenGL compatibility
- Ensure you're building on the main thread for GUI operations

**Camera not detected:**
- Check camera permissions in System Preferences
- Try different camera indices (0, 1, 2, etc.)
- Verify camera is not being used by another application

### Runtime Issues

**Program crashes on macOS:**
- Ensure GUI operations run on the main thread
- Check that all dependencies are properly linked
- Verify camera calibration file format

**Poor tracking quality:**
- Ensure accurate camera calibration
- Use appropriate preset for your hardware
- Check lighting conditions and scene texture

## License

This project is based on DSO (Direct Sparse Odometry), which is licensed under the GNU General Public License Version 3 (GPLv3).

Original DSO repository: https://github.com/JakobEngel/dso

## References

- **Direct Sparse Odometry**, J. Engel, V. Koltun, D. Cremers, arXiv:1607.02565, 2016
- **A Photometrically Calibrated Benchmark For Monocular Visual Odometry**, J. Engel, V. Usenko, D. Cremers, arXiv:1607.02555, 2016

## Acknowledgments

This project extends the original DSO implementation with macOS support, OpenCV 4 compatibility, and real-time camera input capabilities.

