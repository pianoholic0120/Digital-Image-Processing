#!/usr/bin/env python3
"""
Resize dataset files (video, camera.txt, vignette.png) to meet DSO requirements.

DSO Requirements:
- Resolution must be a multiple of powers of 2 for pyramid construction
- Coarsest pyramid level must have > 5000 pixels (wlvl*hlvl > 5000)
- Recommended resolutions: 1280x1024, 640x512, 320x256, etc.

This script will:
1. Resize video to DSO-compatible resolution
2. Update camera.txt with new resolution and adjusted intrinsics
3. Resize vignette.png to match new resolution
4. Overwrite original files (with backup option)
"""

import cv2
import numpy as np
import os
import sys
import argparse
import shutil
from pathlib import Path


def find_dso_compatible_resolution(width, height, min_pixels=5000):
    """
    Find a DSO-compatible resolution that is:
    - A multiple of powers of 2
    - Close to the original aspect ratio
    - Has enough pyramid levels (coarsest level > min_pixels)
    
    Args:
        width: Original width
        height: Original height
        min_pixels: Minimum pixels for coarsest pyramid level (default: 5000)
    
    Returns:
        (target_width, target_height, pyramid_levels)
    """
    # Common DSO-compatible resolutions (sorted by total pixels, descending)
    # Format: (width, height, max_pyramid_levels)
    # All resolutions are multiples of powers of 2
    common_resolutions = [
        (1280, 1024, 5),  # 1280x1024 -> 640x512 -> 320x256 -> 160x128 -> 80x64 -> 40x32
        (1024, 768, 5),   # 1024x768 -> 512x384 -> 256x192 -> 128x96 -> 64x48 -> 32x24
        (1024, 512, 5),   # 1024x512 -> 512x256 -> 256x128 -> 128x64 -> 64x32 -> 32x16
        (768, 512, 5),    # 768x512 -> 384x256 -> 192x128 -> 96x64 -> 48x32 -> 24x16
        (640, 512, 5),    # 640x512 -> 320x256 -> 160x128 -> 80x64 -> 40x32 -> 20x16
        (640, 480, 5),    # 640x480 -> 320x240 -> 160x120 -> 80x60 -> 40x30 -> 20x15
        (512, 512, 5),    # 512x512 -> 256x256 -> 128x128 -> 64x64 -> 32x32 -> 16x16
        (512, 384, 5),    # 512x384 -> 256x192 -> 128x96 -> 64x48 -> 32x24 -> 16x12
        (480, 360, 4),    # 480x360 -> 240x180 -> 120x90 -> 60x45 -> 30x22
        (320, 256, 4),    # 320x256 -> 160x128 -> 80x64 -> 40x32 -> 20x16
        (320, 240, 4),    # 320x240 -> 160x120 -> 80x60 -> 40x30 -> 20x15
    ]
    
    # Calculate aspect ratio
    aspect_ratio = width / height
    
    # Find the best matching resolution
    best_res = None
    best_score = float('inf')
    
    for res_w, res_h, pyr_levels in common_resolutions:
        # Check if coarsest level has enough pixels
        coarsest_w = res_w // (2 ** (pyr_levels - 1))
        coarsest_h = res_h // (2 ** (pyr_levels - 1))
        
        if coarsest_w * coarsest_h < min_pixels:
            continue
        
        # Calculate aspect ratio difference
        res_aspect = res_w / res_h
        aspect_diff = abs(aspect_ratio - res_aspect)
        
        # Calculate size difference (prefer larger resolutions)
        size_diff = abs(width - res_w) + abs(height - res_h)
        
        # Combined score (lower is better)
        score = aspect_diff * 1000 + size_diff / 1000
        
        if score < best_score:
            best_score = score
            best_res = (res_w, res_h, pyr_levels)
    
    # If no good match found, calculate a custom resolution
    if best_res is None:
        # Find a resolution that's a multiple of powers of 2
        # Round width to nearest multiple of 64 (or 32 for smaller images)
        if width >= 640:
            base_w = 64
        elif width >= 320:
            base_w = 32
        else:
            base_w = 16
        
        # Round to nearest multiple
        target_w = ((width + base_w // 2) // base_w) * base_w
        if target_w < 320:
            target_w = 320
        
        # Calculate height maintaining aspect ratio, also multiple of 2
        target_h = int(round(target_w / aspect_ratio))
        # Round to nearest multiple of 2
        target_h = ((target_h + 1) // 2) * 2
        if target_h < 240:
            target_h = 240
        
        # Ensure both are multiples of powers of 2 (at least divisible by 2)
        # Round down to ensure divisibility
        # Find largest power of 2 that divides both
        def largest_power_of_2_divisor(n):
            """Find largest power of 2 that divides n"""
            power = 1
            while n % (power * 2) == 0:
                power *= 2
            return power
        
        # Ensure we can build at least 3 pyramid levels
        min_divisor = 4  # Need at least 2^2 = 4
        target_w = (target_w // min_divisor) * min_divisor
        target_h = (target_h // min_divisor) * min_divisor
        
        # Calculate pyramid levels
        pyr_levels = 1
        wlvl, hlvl = target_w, target_h
        while wlvl % 2 == 0 and hlvl % 2 == 0 and wlvl * hlvl > min_pixels and pyr_levels < 6:
            wlvl //= 2
            hlvl //= 2
            pyr_levels += 1
        
        best_res = (target_w, target_h, pyr_levels)
    
    return best_res


def read_camera_txt(camera_path):
    """
    Read camera.txt file.
    Format:
    Line 1: fx fy cx cy k (5 values)
    Line 2: width height (2 values)
    """
    with open(camera_path, 'r') as f:
        lines = f.readlines()
    
    # Parse first line (intrinsics)
    intrinsics = list(map(float, lines[0].strip().split()))
    fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
    
    # Parse second line (resolution)
    resolution = list(map(int, lines[1].strip().split()))
    width, height = resolution[0], resolution[1]
    
    return fx, fy, cx, cy, width, height


def write_camera_txt(camera_path, fx, fy, cx, cy, width, height, k=0.0):
    """
    Write camera.txt file with updated values.
    """
    with open(camera_path, 'w') as f:
        f.write(f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f} {k}\n")
        f.write(f"{width} {height}\n")


def resize_video(video_path, output_path, target_width, target_height):
    """
    Resize video to target resolution using ffmpeg.
    Maintains aspect ratio with padding.
    """
    import subprocess
    import tempfile
    
    # Create a temporary file with .mp4 extension so ffmpeg can recognize the format
    # Use a unique name to avoid conflicts
    video_path_obj = Path(video_path)
    temp_path = video_path_obj.parent / f"{video_path_obj.stem}_resize_temp.mp4"
    
    # Use ffmpeg to resize
    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-vf', f'scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:black',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-c:a', 'copy',
        '-y',
        str(temp_path)
    ]
    
    try:
        # Run ffmpeg and capture output
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        # Move temp file to final location
        if temp_path.exists():
            # If output path is the same as input, we need to replace it
            if str(output_path) == str(video_path):
                # Remove original and rename temp
                os.remove(video_path)
                temp_path.rename(output_path)
            else:
                # Just move temp to output
                shutil.move(str(temp_path), str(output_path))
            return True
        else:
            print(f"Error: Temporary file was not created")
            return False
            
    except subprocess.CalledProcessError as e:
        # Extract useful error message from stderr
        error_lines = e.stderr.split('\n')
        # Find the actual error line (usually contains "Error")
        error_msg = None
        for line in error_lines:
            if 'Error' in line or 'error' in line:
                error_msg = line.strip()
                break
        
        if error_msg:
            print(f"Error resizing video: {error_msg}")
        else:
            print(f"Error resizing video: {e.stderr[-500:]}")  # Last 500 chars
        
        # Clean up temp file if it exists
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except:
                pass
        return False
    
    return False


def resize_vignette(vignette_path, output_path, target_width, target_height):
    """
    Resize vignette.png to target resolution.
    Vignette is typically a 16-bit grayscale image.
    """
    # Read vignette (may be 8-bit or 16-bit)
    vignette = cv2.imread(str(vignette_path), cv2.IMREAD_UNCHANGED)
    
    if vignette is None:
        print(f"Warning: Could not read vignette.png")
        return False
    
    # Resize using INTER_LINEAR for smooth scaling
    resized = cv2.resize(vignette, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    
    # Save with same bit depth
    if len(vignette.shape) == 2:
        # Grayscale
        cv2.imwrite(str(output_path), resized)
    else:
        # Color (unlikely for vignette, but handle it)
        cv2.imwrite(str(output_path), resized)
    
    return True


def update_camera_intrinsics(fx, fy, cx, cy, old_width, old_height, new_width, new_height):
    """
    Scale camera intrinsics to match new resolution.
    """
    scale_x = new_width / old_width
    scale_y = new_height / old_height
    
    # Scale focal lengths
    new_fx = fx * scale_x
    new_fy = fy * scale_y
    
    # Scale principal point
    new_cx = cx * scale_x
    new_cy = cy * scale_y
    
    return new_fx, new_fy, new_cx, new_cy


def process_dataset(input_dir, backup=True):
    """
    Process a dataset directory to meet DSO requirements.
    
    Args:
        input_dir: Path to dataset directory
        backup: Whether to backup original files
    """
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        print(f"Error: Directory does not exist: {input_dir}")
        return False
    
    print("=" * 60)
    print("DSO Dataset Resize Tool")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print()
    
    # Find video file
    video_files = list(input_dir.glob("*.mp4"))
    if not video_files:
        print("Error: No .mp4 video file found in directory")
        return False
    
    if len(video_files) > 1:
        print(f"Warning: Multiple video files found, using: {video_files[0].name}")
    
    video_path = video_files[0]
    print(f"Video file: {video_path.name}")
    
    # Get video resolution
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return False
    
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"Current video resolution: {video_width}x{video_height}")
    
    # Find DSO-compatible resolution
    target_width, target_height, pyramid_levels = find_dso_compatible_resolution(
        video_width, video_height
    )
    
    print(f"Target resolution: {target_width}x{target_height}")
    print(f"Pyramid levels: {pyramid_levels}")
    print(f"Coarsest level: {target_width // (2**(pyramid_levels-1))}x{target_height // (2**(pyramid_levels-1))}")
    print()
    
    # Check if already correct resolution
    if video_width == target_width and video_height == target_height:
        print("Video already at target resolution. Checking other files...")
    else:
        # Backup video if needed
        if backup:
            backup_path = video_path.with_suffix('.mp4.original')
            if not backup_path.exists():
                print(f"Backing up video to: {backup_path.name}")
                shutil.copy2(video_path, backup_path)
        
        # Resize video
        print("Resizing video...")
        if not resize_video(video_path, video_path, target_width, target_height):
            print("Error: Failed to resize video")
            return False
        print("✓ Video resized successfully")
    
    # Process camera.txt
    camera_path = input_dir / "camera.txt"
    if camera_path.exists():
        print("\nProcessing camera.txt...")
        
        # Backup if needed
        if backup:
            backup_path = camera_path.with_suffix('.txt.original')
            if not backup_path.exists():
                print(f"Backing up camera.txt to: {backup_path.name}")
                shutil.copy2(camera_path, backup_path)
        
        # Read current camera parameters
        fx, fy, cx, cy, cam_width, cam_height = read_camera_txt(camera_path)
        print(f"Current camera: fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}")
        print(f"Current resolution in camera.txt: {cam_width}x{cam_height}")
        
        # Update intrinsics
        new_fx, new_fy, new_cx, new_cy = update_camera_intrinsics(
            fx, fy, cx, cy, cam_width, cam_height, target_width, target_height
        )
        
        # Write updated camera.txt
        write_camera_txt(camera_path, new_fx, new_fy, new_cx, new_cy, target_width, target_height)
        print(f"Updated camera: fx={new_fx:.4f}, fy={new_fy:.4f}, cx={new_cx:.4f}, cy={new_cy:.4f}")
        print(f"Updated resolution: {target_width}x{target_height}")
        print("✓ camera.txt updated")
    else:
        print("\nWarning: camera.txt not found, skipping...")
    
    # Process vignette.png
    vignette_path = input_dir / "vignette.png"
    if vignette_path.exists():
        print("\nProcessing vignette.png...")
        
        # Backup if needed
        if backup:
            backup_path = vignette_path.with_suffix('.png.original')
            if not backup_path.exists():
                print(f"Backing up vignette.png to: {backup_path.name}")
                shutil.copy2(vignette_path, backup_path)
        
        # Get current vignette size
        vignette = cv2.imread(str(vignette_path), cv2.IMREAD_UNCHANGED)
        if vignette is not None:
            v_width, v_height = vignette.shape[1], vignette.shape[0]
            print(f"Current vignette size: {v_width}x{v_height}")
            
            if v_width == target_width and v_height == target_height:
                print("Vignette already at target resolution")
            else:
                # Resize vignette
                if resize_vignette(vignette_path, vignette_path, target_width, target_height):
                    print(f"✓ Vignette resized to {target_width}x{target_height}")
                else:
                    print("Error: Failed to resize vignette")
        else:
            print("Warning: Could not read vignette.png")
    else:
        print("\nWarning: vignette.png not found, skipping...")
    
    print()
    print("=" * 60)
    print("Processing complete!")
    print("=" * 60)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Resize dataset files to meet DSO requirements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 resize_to_dso.py /path/to/dataset
  python3 resize_to_dso.py /path/to/dataset --no-backup
        """
    )
    
    parser.add_argument(
        'input_dir',
        type=str,
        help='Input dataset directory (should contain .mp4, camera.txt, vignette.png)'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not backup original files (default: backup original files)'
    )
    
    args = parser.parse_args()
    
    # Process dataset
    success = process_dataset(args.input_dir, backup=not args.no_backup)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

