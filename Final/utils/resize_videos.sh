#!/bin/bash

# Script to resize all videos in a directory to 1280x1024
# Usage: ./resize_videos.sh <input_dir> [output_dir]

INPUT_DIR="${1:-/Users/arthurlin/Desktop/DIP/Final/video/C3VD}"
OUTPUT_DIR="${2:-${INPUT_DIR}_resized}"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Video Resize Script"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Target resolution: 1280x1024"
echo "=========================================="
echo ""

# Process all MP4 files
for video in "$INPUT_DIR"/*.mp4; do
    if [ ! -f "$video" ]; then
        continue
    fi
    
    filename=$(basename "$video")
    output_path="$OUTPUT_DIR/$filename"
    temp_path="$OUTPUT_DIR/${filename}.tmp"
    
    echo "Processing: $filename"
    
    # Get current resolution
    current_res=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$video" 2>/dev/null)
    echo "  Current resolution: $current_res"
    
    # Check if already 1280x1024
    if [ "$current_res" = "1280x1024" ]; then
        echo "  ✓ Already 1280x1024, skipping..."
        echo ""
        continue
    fi
    
    # If output is same as input, use temp file first (with .mp4 extension)
    if [ "$INPUT_DIR" = "$OUTPUT_DIR" ]; then
        temp_path="${video%.mp4}_resized.mp4"
    else
        temp_path="$output_path"
    fi
    
    # Resize video to 1280x1024 using ffmpeg
    # Using scale filter with padding to maintain aspect ratio
    echo "  Resizing to 1280x1024..."
    ffmpeg -i "$video" \
        -vf "scale=1280:1024:force_original_aspect_ratio=decrease,pad=1280:1024:(ow-iw)/2:(oh-ih)/2:black" \
        -c:v libx264 \
        -preset medium \
        -crf 23 \
        -c:a copy \
        -y \
        "$temp_path" 2>&1 | grep -E "(Duration|Stream|Output|error|frame=|size=)" | tail -3 || true
    
    # If successful and output is same as input, replace original with resized version
    if [ -f "$temp_path" ]; then
        if [ "$INPUT_DIR" = "$OUTPUT_DIR" ]; then
            # Backup original
            mv "$video" "${video%.mp4}_original.mp4"
            # Replace with resized
            mv "$temp_path" "$output_path"
            echo "  ✓ Original backed up to ${filename%.mp4}_original.mp4"
        fi
        echo "  Resize completed"
    else
        echo "  ✗ ERROR: Failed to create output file"
        echo ""
        continue
    fi
    
    # Verify output
    if [ -f "$output_path" ]; then
        new_res=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$output_path" 2>/dev/null)
        file_size=$(du -h "$output_path" | cut -f1)
        echo "  ✓ Output resolution: $new_res"
        echo "  ✓ File size: $file_size"
    else
        echo "  ✗ ERROR: Failed to create output file"
    fi
    
    echo ""
done

echo "=========================================="
echo "All videos processed!"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

