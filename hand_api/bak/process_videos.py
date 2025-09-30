import cv2
import os
from pathlib import Path
import sys # Import sys to check for successful video opening

def extract_frames_from_videos(video_dir_path_str, output_base_dir_str):
    """
    Reads all .mp4 videos from a directory, extracts frames as .jpg images,
    and saves them into corresponding subdirectories in the output base directory.

    Args:
        video_dir_path_str (str): Path to the directory containing the video files.
        output_base_dir_str (str): Path to the base directory where image folders will be created.
    """
    video_dir = Path(video_dir_path_str)
    output_base_dir = Path(output_base_dir_str)

    if not video_dir.is_dir():
        print(f"Error: Input video directory not found: {video_dir}")
        return

    print(f"Searching for videos in: {video_dir}")
    video_files = sorted(list(video_dir.glob('*.mp4'))) # Use glob to find mp4 files and sort them

    if not video_files:
        print(f"No .mp4 files found in {video_dir}")
        return

    print(f"Found {len(video_files)} video files. Starting frame extraction...")

    for video_path in video_files:
        episode_name = video_path.stem # Get filename without extension (e.g., "episode_000000")
        output_episode_dir = output_base_dir / episode_name

        # Create the output directory for this episode's frames
        # parents=True creates any necessary parent directories
        # exist_ok=True doesn't raise an error if the directory already exists
        output_episode_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing video: {video_path.name}")
        print(f"Outputting frames to: {output_episode_dir}")

        # Open the video file
        # Use str(video_path) as cv2.VideoCapture expects a string
        cap = cv2.VideoCapture(str(video_path))

        # Check if video opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}", file=sys.stderr)
            continue # Skip to the next video

        frame_count = 0
        while True:
            # Read frame by frame
            success, frame = cap.read()

            # If frame is read correctly, success is True
            if not success:
                # print("Reached end of video or failed to read frame.")
                break # Exit the loop if no frame is captured

            # Construct the output frame filename (e.g., frame_000000.jpg, frame_000001.jpg, ...)
            # Using zfill for zero-padding ensures correct alphabetical sorting
            # Adjust the padding number (e.g., 6) if you expect more frames
            frame_filename = f"frame_{frame_count:06d}.jpg"
            output_frame_path = output_episode_dir / frame_filename

            # Save frame as JPEG file
            # Use str(output_frame_path) as cv2.imwrite expects a string
            cv2.imwrite(str(output_frame_path), frame)

            frame_count += 1

            # Optional: Print progress every N frames
            if frame_count % 100 == 0:
                 print(f"  Extracted {frame_count} frames...", end='\r')


        # Release the video capture object
        cap.release()
        print(f"  Finished processing. Extracted {frame_count} frames.")

    print("\n-----------------------------")
    print("Frame extraction complete.")
    print(f"Total videos processed: {len(video_files)}")
    print(f"Output base directory: {output_base_dir}")
    print("-----------------------------")


# --- Configuration ---
INPUT_VIDEO_DIR = "/home/admin123/projects/lerobot-plus/lerobot-joycon_plus_data/datasets/pick_put_hand/videos/chunk-000/observation.images.webcam"
OUTPUT_IMAGE_DIR = "/home/admin123/projects/lerobot-plus/lerobot-joycon_plus_data/datasets/pick_put_hand/images/observation.images.webcam"

# --- Run the extraction ---
if __name__ == "__main__":
    extract_frames_from_videos(INPUT_VIDEO_DIR, OUTPUT_IMAGE_DIR)