import cv2
import os
import argparse

def video_to_frames(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Can't open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}, FPS: {fps}")

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_filename = os.path.join(output_dir, f"frame_{saved_idx:05d}.png")
        cv2.imwrite(frame_filename, frame)
        saved_idx += 1
        frame_idx += 1

    cap.release()
    print(f"Saved {saved_idx} images to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split video into images")
    parser.add_argument("--video", type=str, required=True, help="Video file path")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    video_to_frames(args.video, args.output)
