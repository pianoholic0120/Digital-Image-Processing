#!/bin/bash

# DSO 3D Reconstruction Full Process Script
# Usage: ./main.sh <input folder path>
# Example: ./main.sh /Users/arthurlin/Desktop/DIP/Final/12_04_wall1/raw

set -e  # Exit on error

INPUT_DIR="$1"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if the input folder exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input folder does not exist: $INPUT_DIR"
    exit 1
fi

echo "=========================================="
echo "DSO 3D Reconstruction Full Process"
echo "=========================================="
echo "Input folder: $INPUT_DIR"
echo "=========================================="
echo ""

# Step 1: Find video files
echo "[1/5] Find video files..."
VIDEO_FILE=""
for ext in mp4 mov avi mkv MOV MP4 AVI MKV; do
    found=$(find "$INPUT_DIR" -maxdepth 1 -type f -iname "*.${ext}" | head -1)
    if [ -n "$found" ]; then
        VIDEO_FILE="$found"
        break
    fi
done

if [ -z "$VIDEO_FILE" ]; then
    echo "Warning: No video file found, skipping video to image step"
    echo "       Assuming images already exist in $INPUT_DIR/images/"
    IMAGES_DIR="$INPUT_DIR/images"
else
    echo "Found video: $VIDEO_FILE"
    
    # Step 2: Convert video to images
    echo ""
    echo "[2/5] Convert video to images..."
    IMAGES_DIR="$INPUT_DIR/images"
    mkdir -p "$IMAGES_DIR"
    
    python3 "$SCRIPT_DIR/utils/video_to_images.py" \
        --video "$VIDEO_FILE" \
        --output "$IMAGES_DIR"
    
    if [ ! -d "$IMAGES_DIR" ] || [ -z "$(ls -A $IMAGES_DIR/*.png 2>/dev/null)" ]; then
        echo "Error: Image conversion failed or no images generated"
        exit 1
    fi
    
    IMAGE_COUNT=$(ls -1 "$IMAGES_DIR"/*.png 2>/dev/null | wc -l | tr -d ' ')
    echo "Successfully generated $IMAGE_COUNT images"
fi

# Check if images exist
if [ ! -d "$IMAGES_DIR" ] || [ -z "$(ls -A $IMAGES_DIR/*.png 2>/dev/null)" ]; then
    echo "Error: Images not found in $IMAGES_DIR"
    exit 1
fi

# Get the size of the first image
FIRST_IMAGE=$(ls "$IMAGES_DIR"/*.png | head -1)
if [ -z "$FIRST_IMAGE" ]; then
    echo "Error: No image file found"
    exit 1
fi

# Use Python to get the image size
IMAGE_SIZE=$(python3 -c "
import cv2
img = cv2.imread('$FIRST_IMAGE')
if img is not None:
    h, w = img.shape[:2]
    print(f'{w} {h}')
else:
    print('640 480')  # Default value
")
IMAGE_WIDTH=$(echo $IMAGE_SIZE | cut -d' ' -f1)
IMAGE_HEIGHT=$(echo $IMAGE_SIZE | cut -d' ' -f2)
echo "Image size: ${IMAGE_WIDTH}x${IMAGE_HEIGHT}"

# Step 3: Execute photometric calibration
echo ""
echo "[3/5] Execute photometric calibration..."
ONLINE_CALIB_DIR="$SCRIPT_DIR/online_photometric_calibration"
ONLINE_CALIB_BUILD="$ONLINE_CALIB_DIR/build"

if [ ! -f "$ONLINE_CALIB_BUILD/bin/online_pcalib_demo" ]; then
    echo "Error: Photometric calibration program not found, please compile first:"
    echo "      cd $ONLINE_CALIB_DIR/build && cmake .. && make -j4"
    exit 1
fi

cd "$ONLINE_CALIB_BUILD"

IMAGE_COUNT=$(ls -1 "$IMAGES_DIR"/*.png 2>/dev/null | wc -l | tr -d ' ')
# if [ "$IMAGE_COUNT" -gt 1000 ]; then
#     echo "There are many images ($IMAGE_COUNT images), only processing the first 1000 for calibration"
#     END_INDEX=1000
# else
END_INDEX=-1  # Process all images
# fi

echo "Starting photometric calibration (this may take a while)..."
./bin/online_pcalib_demo \
    -i "$IMAGES_DIR" \
    --calibration-mode batch \
    --image-width "$IMAGE_WIDTH" \
    --image-height "$IMAGE_HEIGHT" \
    --end-image-index "$END_INDEX" \
    --no-wait \
    -o "$INPUT_DIR" 2>&1 | tee "$INPUT_DIR/calibration.log"

# Wait a moment to ensure files are written
sleep 2

# Check if the calibration files are generated
if [ ! -f "$INPUT_DIR/pcalib.txt" ] || [ ! -f "$INPUT_DIR/vignette.png" ]; then
    echo "Error: Photometric calibration files not fully generated"
    echo "       Expected: $INPUT_DIR/pcalib.txt and $INPUT_DIR/vignette.png"
    echo "       Check log: $INPUT_DIR/calibration.log"
    exit 1
fi

echo "Photometric calibration completed successfully!"
echo "  - pcalib.txt: $(wc -l < "$INPUT_DIR/pcalib.txt" | tr -d ' ') lines"
if [ -f "$INPUT_DIR/vignette.png" ]; then
    echo "  - vignette.png: $(file "$INPUT_DIR/vignette.png" | cut -d: -f2)"
fi

# Step 4: Generate camera.txt (if not exists)
echo ""
echo "[4/5] Check/generate camera.txt..."
CAMERA_FILE="$INPUT_DIR/camera.txt"

if [ ! -f "$CAMERA_FILE" ]; then
    echo "Generate default camera.txt (FOV model)..."
    # Use FOV model, parameters need to be adjusted according to the actual camera
    cat > "$CAMERA_FILE" << EOF
1.019232 1.359512 0.515960 0.524358 0.0
$IMAGE_WIDTH $IMAGE_HEIGHT
1.019232 1.359512 0.515960 0.524358 0.0
$IMAGE_WIDTH $IMAGE_HEIGHT
EOF
    echo "Generated default camera.txt"
    echo "Note: You may need to adjust camera.txt according to the actual camera parameters"
else
    echo "Using existing camera.txt"
fi

# Step 5: Execute DSO 3D Reconstruction
echo ""
echo "[5/5] Execute DSO 3D Reconstruction..."
DSO_BUILD_DIR="$SCRIPT_DIR/dso/build"
DSO_OUTPUT_DIR="$INPUT_DIR/dso_output"

if [ ! -f "$DSO_BUILD_DIR/bin/dso_dataset" ]; then
    echo "Compile DSO..."
    cd "$DSO_BUILD_DIR"
    cmake .. > /dev/null 2>&1
    make -j4 > /dev/null 2>&1
fi

if [ ! -f "$DSO_BUILD_DIR/bin/dso_dataset" ]; then
    echo "Error: DSO compilation failed or dso_dataset not found"
    exit 1
fi

mkdir -p "$DSO_OUTPUT_DIR"

cd "$DSO_OUTPUT_DIR"  # DSO will save the results in the current directory
"$DSO_BUILD_DIR/bin/dso_dataset" \
    files="$IMAGES_DIR" \
    calib="$CAMERA_FILE" \
    gamma="$INPUT_DIR/pcalib.txt" \
    vignette="$INPUT_DIR/vignette.png" \
    preset=0 \
    mode=0 2>&1 | tee "$DSO_OUTPUT_DIR/dso.log"

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
echo "Input folder: $INPUT_DIR"
echo "Image directory: $IMAGES_DIR"
echo "DSO output: $DSO_OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  - pcalib.txt: $INPUT_DIR/pcalib.txt"
echo "  - vignette.png: $INPUT_DIR/vignette.png"
echo "  - camera.txt: $CAMERA_FILE"
echo "  - DSO results: $DSO_OUTPUT_DIR/"
echo "=========================================="