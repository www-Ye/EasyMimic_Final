#!/usr/bin/env python3
"""
Video frame interpolation utility for filling in skipped frames.
Supports multiple interpolation methods including linear blending and optical flow.
"""

import cv2
import numpy as np
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import torch
import torchvision.transforms.functional as F

def linear_interpolate_frames(frame1, frame2, num_intermediate=1):
    """
    在两帧之间进行简单的线性插值，生成中间帧。

    参数:
        frame1: 第一帧（numpy 数组）
        frame2: 第二帧（numpy 数组）
        num_intermediate: 需要生成的中间帧数量

    返回:
        插值生成的中间帧列表
    """
    interpolated_frames = []  # 用于存储插值帧的列表

    # 依次生成每一帧插值
    for i in range(1, num_intermediate + 1):
        alpha = i / (num_intermediate + 1)  # 计算当前插值权重
        # 线性插值公式：(1-alpha)*frame1 + alpha*frame2
        interpolated = (1 - alpha) * frame1 + alpha * frame2
        # 转换为uint8类型并加入结果列表
        interpolated_frames.append(interpolated.astype(np.uint8))

    return interpolated_frames  # 返回所有插值帧


def optical_flow_interpolate(frame1, frame2, num_intermediate=1):
    """
    Use optical flow for more sophisticated frame interpolation.
    
    Args:
        frame1: First frame (numpy array)
        frame2: Second frame (numpy array)
        num_intermediate: Number of intermediate frames to generate
        
    Returns:
        List of interpolated frames
    """
    # Convert to grayscale for optical flow
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    interpolated_frames = []
    h, w = frame1.shape[:2]
    
    for i in range(1, num_intermediate + 1):
        alpha = i / (num_intermediate + 1)
        
        # Create flow map for current interpolation position
        flow_map = flow * alpha
        
        # Create mesh grid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply flow
        new_x = (x - flow_map[..., 0]).astype(np.float32)
        new_y = (y - flow_map[..., 1]).astype(np.float32)
        
        # Warp frame1 according to flow
        warped = cv2.remap(frame1, new_x, new_y, cv2.INTER_LINEAR)
        
        # Blend with linear interpolation as fallback
        linear_blend = (1 - alpha) * frame1 + alpha * frame2
        
        # Combine warped and linear (helps with occlusions)
        interpolated = 0.7 * warped + 0.3 * linear_blend
        interpolated_frames.append(interpolated.astype(np.uint8))
    
    return interpolated_frames


def interpolate_video(input_path, output_path, skip_frames, method='linear', show_progress=True):
    """
    Interpolate a video that was rendered with frame skipping.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        skip_frames: The skip_frames value used during rendering
        method: Interpolation method ('linear' or 'optical_flow')
        show_progress: Show progress bar
        
    Returns:
        Success status and message
    """
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False, f"Failed to open video: {input_path}"
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output video with H264 codec for better compatibility
    # Try different codecs in order of preference
    codecs = ['h264', 'mp4v', 'avc1', 'x264']
    out = None
    
    for codec in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out.isOpened():
                print(f"Using codec: {codec}")
                break
            else:
                out.release()
                out = None
        except:
            continue
    
    if out is None or not out.isOpened():
        return False, f"Failed to create output video with any available codec"
    
    # Process frames
    frame_count = 0
    prev_frame = None
    
    if show_progress:
        pbar = tqdm(total=total_frames * skip_frames, desc="Interpolating frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if prev_frame is not None and skip_frames > 1:
            # Generate intermediate frames
            if method == 'linear':
                interpolated = linear_interpolate_frames(prev_frame, frame, skip_frames - 1)
            elif method == 'optical_flow':
                interpolated = optical_flow_interpolate(prev_frame, frame, skip_frames - 1)
            else:
                raise ValueError(f"Unknown interpolation method: {method}")
            
            # Write previous frame
            out.write(prev_frame)
            frame_count += 1
            
            # Write interpolated frames
            for interp_frame in interpolated:
                out.write(interp_frame)
                frame_count += 1
                
        elif prev_frame is None:
            # First frame
            out.write(frame)
            frame_count += 1
        
        prev_frame = frame.copy()
        
        if show_progress:
            pbar.update(skip_frames)
    
    # Write last frame
    if prev_frame is not None and skip_frames == 1:
        out.write(prev_frame)
        frame_count += 1
    
    # Cleanup
    cap.release()
    out.release()
    
    if show_progress:
        pbar.close()
    
    # Ensure the output is a proper MP4 file using FFmpeg if available
    if os.path.exists(output_path):
        try:
            # Try to use FFmpeg to ensure proper MP4 format
            import subprocess
            temp_output = output_path.replace('.mp4', '_temp.mp4')
            
            # Check if ffmpeg is available
            result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
            if result.returncode == 0:
                # Use FFmpeg to re-encode with H264
                cmd = [
                    'ffmpeg', '-i', output_path,  # Input
                    '-c:v', 'libx264',            # H264 codec
                    '-preset', 'fast',            # Fast encoding
                    '-crf', '22',                 # Quality (lower = better, 22 is good)
                    '-pix_fmt', 'yuv420p',        # Pixel format for compatibility
                    '-movflags', 'faststart',     # Optimize for streaming
                    '-y',                         # Overwrite
                    temp_output
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    # Replace original with re-encoded version
                    os.replace(temp_output, output_path)
                    print("✓ Re-encoded with FFmpeg for better compatibility")
                else:
                    print("⚠ FFmpeg re-encoding failed, using original output")
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
        except Exception as e:
            print(f"⚠ Could not optimize MP4 format: {e}")
    
    return True, f"Successfully interpolated {frame_count} frames"


def interpolate_episode_videos(dataset_path, skip_frames, method='linear', dry_run=False):
    """
    Interpolate all episode videos in a dataset.
    
    Args:
        dataset_path: Path to the dataset
        skip_frames: The skip_frames value used during rendering
        method: Interpolation method
        dry_run: If True, only show what would be done
    """
    dataset_path = Path(dataset_path)
    video_dir = dataset_path / "videos" / "chunk-000" / "observation.images.webcam"
    
    if not video_dir.exists():
        print(f"Video directory not found: {video_dir}")
        return
    
    # Find all MP4 files
    video_files = sorted(video_dir.glob("episode_*.mp4"))
    print(f"Found {len(video_files)} video files to process")
    
    if dry_run:
        print("Dry run - no files will be modified")
        for video_file in video_files:
            print(f"  Would process: {video_file.name}")
        return
    
    # Create backup directory
    backup_dir = video_dir / "original_skip_frames"
    backup_dir.mkdir(exist_ok=True)
    
    success_count = 0
    for video_file in video_files:
        print(f"\nProcessing {video_file.name}...")
        
        # Backup original
        backup_path = backup_dir / video_file.name
        if not backup_path.exists():
            import shutil
            shutil.copy2(video_file, backup_path)
            print(f"  Backed up to: {backup_path}")
        
        # Interpolate to temporary file
        temp_path = video_file.with_suffix('.tmp.mp4')
        success, message = interpolate_video(
            str(video_file), 
            str(temp_path), 
            skip_frames, 
            method
        )
        
        if success:
            # Replace original with interpolated
            temp_path.replace(video_file)
            print(f"  ✓ {message}")
            success_count += 1
        else:
            print(f"  ✗ {message}")
            if temp_path.exists():
                temp_path.unlink()
    
    print(f"\nCompleted: {success_count}/{len(video_files)} videos interpolated successfully")


def main():
    parser = argparse.ArgumentParser(description="Interpolate frames in videos rendered with skip_frames")
    parser.add_argument("dataset_path", help="Path to the dataset directory")
    parser.add_argument("--skip-frames", type=int, required=True, 
                        help="The skip_frames value used during rendering")
    parser.add_argument("--method", choices=['linear', 'optical_flow'], default='linear',
                        help="Interpolation method (default: linear)")
    parser.add_argument("--dry-run", action='store_true',
                        help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    interpolate_episode_videos(
        args.dataset_path,
        args.skip_frames,
        args.method,
        args.dry_run
    )


if __name__ == "__main__":
    main()