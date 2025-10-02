# hand_single_process.py
import os
import numpy as np
import shutil
from glob import glob
from tqdm import tqdm
import json
import torch
import datetime
import cv2
import pandas as pd
import argparse
import sys

from aitviewer.headless import HeadlessRenderer
# Assuming these imports are correct and available in your environment
from hamer_api import detect_track_video, hamer_motion_estimation, hamer_infiller, hamer_slam, compute_all_finger_distances
from lib.eval_utils.custom_utils import world2cam_convert, cam2world_convert, load_slam_cam
from lib.vis.run_vis2 import run_vis2_on_video, run_vis2_on_video_cam, run_vis2_on_video_cam_random_color
from hamer.utils.process import get_mano_faces, run_mano, run_mano_left
from hamer.utils.rotation import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis, angle_axis_to_quaternion

def quaternion_to_rpy(quaternion):
    """Convert quaternion to roll, pitch, yaw Euler angles."""
    qw, qx, qy, qz = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    roll = torch.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    sinp = 2 * (qw * qy - qz * qx)
    pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * (torch.pi / 2), torch.asin(sinp))
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return torch.stack([roll, pitch, yaw], dim=-1)

# --- Helper function to find longest non-NaN sequence (FIXED for dimensionality) ---
def find_longest_non_nan_sequence(data_tensor):
    """
    Finds the start and end indices of the longest contiguous sub-sequence
    where the values are not NaN. Requires sequence length >= 2.

    Args:
        data_tensor: A 1D or 2D (shape [T, 1]) PyTorch tensor.

    Returns:
        A tuple (start_index, end_index) of the longest sequence (length >= 2),
        or None if no such sequence is found. Indices are inclusive.
    """
    if data_tensor is None or data_tensor.numel() < 2: # Need at least 2 elements
        return None

    # --- Ensure data_tensor is effectively 1D for processing ---
    if data_tensor.ndim == 2 and data_tensor.shape[1] == 1:
        print(f"Input data_tensor has shape {data_tensor.shape}, squeezing to 1D.")
        data_tensor = data_tensor.squeeze(1)
    elif data_tensor.ndim > 1:
        print(f"Warning: Input data_tensor has unexpected dimensions {data_tensor.shape}. Flattening.")
        data_tensor = data_tensor.flatten()
    # Now data_tensor should be 1D

    is_not_nan = ~torch.isnan(data_tensor)
    if not is_not_nan.any():
        return None # No non-NaN values

    # Convert to long type
    is_not_nan_long = is_not_nan.long() # Should be 1D now

    # --- Manually prepend 0 ---
    prepended_value = torch.tensor([0], dtype=torch.long, device=data_tensor.device) # This is 1D [1]


    # Concatenate (both should be 1D now)
    try:
        data_with_prepend = torch.cat((prepended_value, is_not_nan_long), dim=0) # Explicitly use dim=0 for 1D
    except RuntimeError as e:
        print("!!! Error during torch.cat even after potential squeeze !!!")
        print("Shape of prepended_value:", prepended_value.shape)
        print("Shape of is_not_nan_long:", is_not_nan_long.shape)
        raise e # Re-raise the error

    # Calculate differences
    change_points = torch.diff(data_with_prepend)

    # Find indices where sequences start (change is 1) and end (change is -1)
    starts_in_diff = torch.where(change_points == 1)[0]
    ends_in_diff = torch.where(change_points == -1)[0]

    # Correct indices to match the original (squeezed) data_tensor
    starts_original = starts_in_diff
    ends_original_calc = ends_in_diff

    # Handle edge cases:
    if data_tensor.numel() > 0 and is_not_nan[0]: # Check if first element is non-NaN
        if starts_original.numel() == 0 or starts_original[0] != 0:
             starts_original = torch.cat((torch.tensor([0], device=starts_original.device), starts_original))

    if len(starts_original) > len(ends_original_calc):
        ends_original_calc = torch.cat((ends_original_calc, torch.tensor([len(data_tensor)], device=ends_original_calc.device)))

    # Ensure starts and ends have the same length for zipping
    min_len = min(len(starts_original), len(ends_original_calc))
    starts_original = starts_original[:min_len]
    ends_original_calc = ends_original_calc[:min_len]

    best_len = 0
    best_seq = None

    if starts_original.numel() > 0: # Check if any sequences were found
        for start, end_calc in zip(starts_original, ends_original_calc):
            length = end_calc - start # Calculate length
            if length >= 2 and length > best_len: # Check if length is >= 2 and longest so far
                best_len = length
                best_seq = (start.item(), end_calc.item() - 1) # Inclusive end index

    return best_seq


def process_hand_video(input_episode_path, enable_visualization=False, new_base_path=None, output_episode_name=None):
    """
    Processes a single episode's image directory.

    Args:
        input_episode_path: Path to the directory containing images for one episode.
        enable_visualization: Whether to generate visualization video.
        new_base_path: The root directory for the processed dataset output.
        output_episode_name: The desired sequential name for the output (e.g., episode_000000).

    Returns:
        The number of frames successfully processed and saved for this episode (>= 0).
        Returns 0 if the episode is skipped (no hands, no valid sequence, etc.).
        Exits with status 1 if critical data corruption (NaN in state) is detected.
    """
    print(f"--- Processing Input: {input_episode_path} ---")
    print(f"--- Output Name Target: {output_episode_name} ---")

    num_frames_processed = 0 # Default return value for skipped episodes

    try:
        # --- Stage 1: Detection and Tracking ---
        start_idx, end_idx, seq_folder, imgfiles = detect_track_video(input_episode_path)
        if not imgfiles:
                print(f"Skipping {input_episode_path}: No image files found by detect_track_video.")
                print(f"---EPISODE_LENGTH---:{num_frames_processed}")
                return num_frames_processed

        # --- Stage 2: Motion Estimation ---
        frame_chunks_all, img_focal = hamer_motion_estimation(input_episode_path, start_idx, end_idx, seq_folder)
        valid_hand = list(frame_chunks_all.keys())
        if not valid_hand:
            print(f"Skipping {input_episode_path}: No valid hands detected by hamer_motion_estimation.")
            print(f"---EPISODE_LENGTH---:{num_frames_processed}")
            return num_frames_processed

        # --- Stage 3: SLAM ---
        slam_path = os.path.join(seq_folder, f"SLAM/hamer_slam_w_scale_{start_idx}_{end_idx}.npz")
        if not os.path.exists(slam_path):
            hamer_slam(input_episode_path, start_idx, end_idx, seq_folder)
        # Check again if SLAM ran successfully
        if not os.path.exists(slam_path):
            print(f"Skipping {input_episode_path}: SLAM file not found even after running hamer_slam.")
            print(f"---EPISODE_LENGTH---:{num_frames_processed}")
            return num_frames_processed
        R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)

        # --- Stage 4: Infilling (Hand Pose Estimation) ---
        pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hamer_infiller(input_episode_path, start_idx, end_idx, frame_chunks_all, seq_folder)
        print("pred_trans shape:", pred_trans.shape) # Shape: [2, T, 3]
        print("pred_rot shape:", pred_rot.shape)     # Shape: [2, T, 3]
        print("pred_hand_pose shape:", pred_hand_pose.shape) # Shape: [2, T, 45]
        print("pred_betas shape:", pred_betas.shape) # Shape: [2, T, 10]
        print("pred_valid shape:", pred_valid.shape) # Shape: [2, T]

        # Determine which hand to use
        if len(valid_hand) == 1:
            use_hand_idx = valid_hand[0]
            print(f"Using single detected hand: {'left' if use_hand_idx == 0 else 'right'} (index {use_hand_idx})")
        elif len(valid_hand) == 2:
            # Defaulting to right hand if both are detected
            use_hand_idx = 1
            print(f"Detected both hands. Defaulting to use right hand (index {use_hand_idx}).")
        else:
            # This case should have been caught earlier, but double-check
            print(f"Skipping {input_episode_path}: Unexpected valid_hand content: {valid_hand}")
            print(f"---EPISODE_LENGTH---:{num_frames_processed}")
            return num_frames_processed

        # Select data for the chosen hand
        # Squeeze the first dimension (batch=1) which comes from world2cam_convert later
        # Note: pred_* tensors have shape [2, T, D], select the hand first -> [T, D]
        hand_pred_trans = pred_trans[use_hand_idx]
        hand_pred_rot = pred_rot[use_hand_idx]
        hand_pred_hand_pose = pred_hand_pose[use_hand_idx]
        hand_pred_betas = pred_betas[use_hand_idx]

        # --- Stage 5: Coordinate Conversion (World to Camera) ---
        # Wrap the selected hand data to fit world2cam_convert input format
        data_world_single_hand = {
            "init_root_orient": hand_pred_rot.unsqueeze(0), # Add batch dim [1, T, 3]
            "init_hand_pose": hand_pred_hand_pose.unsqueeze(0), # [1, T, 45]
            "init_trans": hand_pred_trans.unsqueeze(0),      # [1, T, 3]
            "init_betas": hand_pred_betas.unsqueeze(0)       # [1, T, 10]
        }
        hand_label = "right" if use_hand_idx == 1 else "left"
        data_cam_single_hand = world2cam_convert(R_w2c_sla_all, t_w2c_sla_all, data_world_single_hand, hand_label)

        # Extract camera-space data and remove batch dimension
        cam_trans = data_cam_single_hand["init_trans"].squeeze(0)       # [T, 3]
        cam_root_orient_aa = data_cam_single_hand["init_root_orient"].squeeze(0) # [T, 3] Angle-axis
        cam_hand_pose = data_cam_single_hand["init_hand_pose"].squeeze(0)   # [T, 45]
        cam_betas = data_cam_single_hand["init_betas"].squeeze(0)       # [T, 10]

        # --- Stage 6: Calculate Open Distance and Find Longest Sequence ---
        # Need MANO joints in camera space to compute distances
        mano_fn = run_mano if use_hand_idx == 1 else run_mano_left
        # Add batch dim back for run_mano functions
        pred_glob_cam = mano_fn(cam_trans.unsqueeze(0),
                                cam_root_orient_aa.unsqueeze(0),
                                cam_hand_pose.unsqueeze(0),
                                betas=cam_betas.unsqueeze(0))
        cam_joints = pred_glob_cam['joints'].squeeze(0) # [T, 21, 3]

        fingd_dict = compute_all_finger_distances(cam_joints)
        thumb_to_index_distance = fingd_dict["thumb_to_index_distance"] # [T]
        open_distance = (thumb_to_index_distance * 100).cpu().detach() # [T]

        midpoint_trans = get_midpoint(cam_joints) # [T, 3]
        midpoint_orient_aa = cam_root_orient_aa

        cam_trans = midpoint_trans
        cam_root_orient_aa = midpoint_orient_aa

        longest_seq = find_longest_non_nan_sequence(open_distance)

        if longest_seq is None:
            print(f"Skipping {input_episode_path}: No valid non-NaN sequence (length >= 2) found in 'open_distance'.")
            print(f"---EPISODE_LENGTH---:{num_frames_processed}")
            return num_frames_processed

        k_start, k_end = longest_seq # Inclusive indices
        seq_len = k_end - k_start + 1
        print(f"Found longest non-NaN sequence in open_distance: index {k_start} to {k_end} (inclusive), length {seq_len}")

        # --- Stage 7: Prepare State, Action, MANO data based on sequence ---
        # State/MANO/Image use indices k_start to k_end - 1
        # Action uses indices k_start + 1 to k_end
        # Number of pairs = (k_end - 1) - k_start + 1 = k_end - k_start
        num_pairs = k_end - k_start
        if num_pairs <= 0:
                print(f"Skipping {input_episode_path}: Longest sequence length ({seq_len}) is too short to form state-action pairs.")
                print(f"---EPISODE_LENGTH---:{num_frames_processed}")
                return num_frames_processed

        # Convert root orientation to RPY for state/action
        cam_root_orient_quat = angle_axis_to_quaternion(cam_root_orient_aa) # [T, 4]
        cam_root_orient_rpy = quaternion_to_rpy(cam_root_orient_quat)    # [T, 3]

        state_indices = torch.arange(k_start, k_end)     # Indices k_start to k_end-1
        action_indices = torch.arange(k_start + 1, k_end + 1) # Indices k_start+1 to k_end

        # Slice data
        state_trans = cam_trans[state_indices]
        state_rpy = cam_root_orient_rpy[state_indices]
        state_open = open_distance[state_indices]#.unsqueeze(-1) # Ensure [N, 1] shape

        action_trans = cam_trans[action_indices]
        action_rpy = cam_root_orient_rpy[action_indices]
        action_open = open_distance[action_indices]#.unsqueeze(-1) # Ensure [N, 1] shape

        mano_trans = cam_trans[state_indices]
        mano_root_aa = cam_root_orient_aa[state_indices]
        mano_hand_pose = cam_hand_pose[state_indices]
        mano_betas = cam_betas[state_indices] # Assuming betas are constant or using the state's beta

        print("state_trans", state_trans.shape)
        print("state_rpy", state_rpy.shape)
        print("state_open", state_open.shape)
        # Concatenate features
        state_frames_data = torch.cat([state_trans, state_rpy, state_open], dim=-1) # [N, 3+3+1=7]
        action_frames_data = torch.cat([action_trans, action_rpy, action_open], dim=-1) # [N, 7]
        mano_data = torch.cat([mano_trans, mano_root_aa, mano_hand_pose, mano_betas], dim=-1) # [N, 3+3+45+10=61]

        imgfiles_sequence = imgfiles[k_start:k_end] # List slicing is exclusive at end

        print("Sliced state_frames_data shape:", state_frames_data.shape)
        print("Sliced action_frames_data shape:", action_frames_data.shape)
        print("Sliced mano_data shape:", mano_data.shape)
        print("Sliced imgfiles count:", len(imgfiles_sequence))
        # breakpoint()

        # --- CRITICAL NaN Check ---
        if torch.isnan(state_frames_data).any():
            nan_mask = torch.isnan(state_frames_data)
            rows_with_nan = torch.where(nan_mask.any(dim=1))[0].tolist()
            print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"CRITICAL ERROR: NaN values detected in 'state_frames_data' for input: {input_episode_path}")
            print(f"Output target was: {output_episode_name}")
            print(f"Affected rows (0-based indices within the sliced data): {rows_with_nan}")
            print(f"Original sequence used: indices {k_start} to {k_end-1}")
            print(f"STOPPING EXECUTION.")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            sys.exit(1) # Hard stop for critical error

        # --- Stage 8: Interpolation ---
        interp_factor = 4
        state_frames_interp = linear_interpolation(state_frames_data, interp_factor, output_episode_name)
        action_frames_interp = linear_interpolation(action_frames_data, interp_factor, output_episode_name)
        mano_frames_interp = linear_interpolation(mano_data, interp_factor, output_episode_name)

        # Check for NaN after interpolation
        if torch.isnan(state_frames_interp).any():
            print(f"CRITICAL ERROR: NaN values detected in 'state_frames_interp' AFTER interpolation for input: {input_episode_path}")
            print(f"Output target was: {output_episode_name}")
            sys.exit(1)

        # Interpolate image files and timestamps
        imgfiles_interp = interpolate_imgfiles(imgfiles_sequence, interp_factor, output_episode_name)
        original_timestamps = extract_timestamps_from_imgfiles(imgfiles_sequence)

        if len(original_timestamps) == 0 and len(imgfiles_sequence) > 0:
            print(f"Warning: Could not extract or generate timestamps for {input_episode_path}, using frame indices.")
            # Use dummy timestamps if needed, or handle downstream
            timestamps_interp = np.arange(len(imgfiles_interp)) * (1.0 / 30.0) # Assume 30fps
        elif len(original_timestamps) > 0:
            timestamps_interp = interpolate_timestamps(original_timestamps, interp_factor, output_episode_name)
            # Adjust timestamps
            if len(timestamps_interp) > 0:
                start_time = timestamps_interp[0]
                timestamps_interp = timestamps_interp - start_time
                fps = 30.0
                interval = 1.0 / fps
                for i in range(1, len(timestamps_interp)):
                    timestamps_interp[i] = i * interval
        else: # No images, no timestamps
            timestamps_interp = np.array([])


        num_frames_interp = state_frames_interp.shape[0]
        print("Interpolated image file count:", len(imgfiles_interp))
        print("Interpolated state frame count:", num_frames_interp)

        # Final check for consistency and minimum length
        if num_frames_interp == 0:
                print(f"Skipping {input_episode_path}: Resulted in 0 frames after processing and interpolation.")
                num_frames_processed = 0
                print(f"---EPISODE_LENGTH---:{num_frames_processed}")
                return num_frames_processed

        if len(imgfiles_interp) != num_frames_interp:
            print(f"Warning: Mismatch between interpolated image count ({len(imgfiles_interp)}) and frame count ({num_frames_interp}) for {input_episode_path}. Adjusting to minimum.")
            min_len = min(len(imgfiles_interp), num_frames_interp)
            if min_len == 0:
                    print(f"Skipping {input_episode_path}: Mismatch resulted in 0 length.")
                    num_frames_processed = 0
                    print(f"---EPISODE_LENGTH---:{num_frames_processed}")
                    return num_frames_processed

            imgfiles_interp = imgfiles_interp[:min_len]
            state_frames_interp = state_frames_interp[:min_len]
            action_frames_interp = action_frames_interp[:min_len]
            mano_frames_interp = mano_frames_interp[:min_len]
            timestamps_interp = timestamps_interp[:min_len]
            num_frames_interp = min_len # Update frame count
            print(f"Adjusted length: {num_frames_interp}")


        # --- Stage 9: Save Parquet Data ---
        state_frames_array = state_frames_interp.numpy()
        action_frames_array = action_frames_interp.numpy()
        # mano_frames_array = mano_frames_interp.numpy() # Optionally save MANO params
        timestamps_array = timestamps_interp

        # xyzrpy+gripper distance [1:]
        # xyzrpy+gripper distance [:-1]
        data_dict = {
            "action": action_frames_array.tolist(),
            "observation.state": state_frames_array.tolist(),
            "timestamp": timestamps_array.tolist(),
            "frame_index": list(range(num_frames_interp)),
            # episode_index/task_index might need to be passed or handled later
            "episode_index": [0] * num_frames_interp,
            "index": list(range(num_frames_interp)),
            "task_index": [0] * num_frames_interp,
        }
        chunk_dir = os.path.join(new_base_path, 'data', 'chunk-000')
        os.makedirs(chunk_dir, exist_ok=True)

        df = pd.DataFrame(data_dict)
        parquet_file_path = os.path.join(chunk_dir, f'{output_episode_name}.parquet')
        df.to_parquet(parquet_file_path, engine='pyarrow')
        print(f"DataFrame successfully saved to {parquet_file_path}")

        # --- Stage 10: Visualization (Optional) ---
        if enable_visualization:
            print("Starting visualization...")
            # Get faces
            faces = get_mano_faces()
            # Simplified face definition from original code
            faces_new = np.array([[92, 38, 234],[234, 38, 239],[38, 122, 239],[239, 122, 279],[122, 118, 279],[279, 118, 215],[118, 117, 215],[215, 117, 214],[117, 119, 214],[214, 119, 121],[119, 120, 121],[121, 120, 78],[120, 108, 78],[78, 108, 79]])
            faces_right = np.concatenate([faces, faces_new], axis=0)
            faces_left = faces_right[:, [0, 2, 1]]

            vis_mano_fn = run_mano if use_hand_idx == 1 else run_mano_left
            vis_faces = faces_right if use_hand_idx == 1 else faces_left

            # Use interpolated MANO parameters for visualization
            vis_pred_glob = vis_mano_fn(mano_frames_interp[:, :3].unsqueeze(0),    # trans
                                        mano_frames_interp[:, 3:6].unsqueeze(0),    # root_orient_aa
                                        mano_frames_interp[:, 6:51].unsqueeze(0),   # hand_pose
                                        betas=mano_frames_interp[:, 51:].unsqueeze(0)) # betas
            vis_verts = vis_pred_glob['vertices'][0] # [N_interp, 778, 3]

            hand_dict = {
                'vertices': vis_verts.unsqueeze(0), # [1, N_interp, 778, 3]
                'faces': vis_faces,
            }
            left_dict_vis = hand_dict if use_hand_idx == 0 else None
            right_dict_vis = hand_dict if use_hand_idx == 1 else None

            output_video_dir = os.path.join(new_base_path, "videos", "chunk-000", "observation.images.webcam")
            os.makedirs(output_video_dir, exist_ok=True)
            print(f"Visualization output path: {output_video_dir}")

            run_vis2_on_video_cam_random_color(
                left_dict_vis,
                right_dict_vis,
                output_video_dir,
                img_focal,
                imgfiles_interp, # Use interpolated image paths
                R_w2c=torch.eye(3).repeat(num_frames_interp, 1, 1),
                t_w2c=torch.zeros(num_frames_interp, 3),
                valid_hand=[use_hand_idx], # Only visualize the used hand
                v=None,
                video_name=output_episode_name # Use the sequential name
            )
            print("Visualization finished.")

        # If everything succeeded, set the number of processed frames
        num_frames_processed = num_frames_interp

    except Exception as e:
        # Catch unexpected errors during processing of this episode
        print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"UNEXPECTED ERROR occurred while processing input: {input_episode_path}")
        print(f"Output target was: {output_episode_name}")
        print(f"Error type: {type(e)}")
        print(f"Error message: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
        print(f"SKIPPING this episode due to unexpected error.")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        num_frames_processed = 0 # Ensure 0 is returned for skipped episode

    # Print the final length for this episode for the shell script to capture
    print(f"---EPISODE_LENGTH---:{num_frames_processed}")
    return num_frames_processed


# --- Interpolation and Utility functions (modified slightly for robustness) ---
def linear_interpolation(data, factor=2, episode_name=""):
    if data is None or data.shape[0] < 2:
        # print(f"Warning [{episode_name}]: Less than 2 frames ({data.shape[0] if data is not None else 0}) for linear interpolation. Returning original data.")
        return data if data is not None else torch.empty((0, data.shape[1] if data is not None else 0)) # Return empty tensor of correct dim if needed

    n, d = data.shape
    if torch.isnan(data).any():
         print(f"Warning [{episode_name}]: NaN detected in data passed to linear_interpolation. Interpolation might propagate NaNs.")

    interp_n = (n - 1) * factor + 1
    interp_data = torch.zeros((interp_n, d), dtype=data.dtype, device=data.device)

    for i in range(n):
        interp_data[i * factor] = data[i]

    for i in range(n - 1):
        start_frame = data[i]
        end_frame = data[i+1]
        if torch.isnan(start_frame).any() or torch.isnan(end_frame).any():
            # print(f"Warning [{episode_name}]: NaN detected between frame {i} and {i+1} during interpolation. Filling with previous frame.")
            for j in range(1, factor):
                 interp_data[i * factor + j] = start_frame # Fill with previous
            continue

        for j in range(1, factor):
            alpha = j / factor
            interp_data[i * factor + j] = (1 - alpha) * start_frame + alpha * end_frame

    return interp_data

def interpolate_imgfiles(imgfiles, factor, episode_name=""):
    n = len(imgfiles)
    if n < 2:
        # print(f"Warning [{episode_name}]: Less than 2 image files ({n}) for interpolation. Returning original list.")
        return imgfiles
    interp_n = (n - 1) * factor + 1
    interp_imgfiles = [None] * interp_n
    for i in range(n):
        interp_imgfiles[i * factor] = imgfiles[i]
    for i in range(n - 1):
        for j in range(1, factor):
            interp_imgfiles[i * factor + j] = imgfiles[i]
    return interp_imgfiles

def extract_timestamps_from_imgfiles(imgfiles):
    timestamps = []
    if not imgfiles: return np.array([])
    time_found = True
    for img_path in imgfiles:
        filename = os.path.basename(img_path).split('.')[0]
        import re
        match = re.search(r'(\d+)', filename)
        if match:
            timestamps.append(float(match.group(1)))
        else:
            time_found = False; break
    if not time_found:
        fps = 30.0; interval = 1.0 / fps
        timestamps = [i * interval for i in range(len(imgfiles))]
    return np.array(timestamps)

def interpolate_timestamps(timestamps, factor, episode_name=""):
    n = len(timestamps)
    if n < 2:
        # print(f"Warning [{episode_name}]: Less than 2 timestamps ({n}) for interpolation. Returning original array.")
        return timestamps
    interp_n = (n - 1) * factor + 1
    interp_timestamps = np.zeros(interp_n)
    for i in range(n):
        interp_timestamps[i * factor] = timestamps[i]
    for i in range(n - 1):
        for j in range(1, factor):
            alpha = j / factor
            interp_timestamps[i * factor + j] = (1 - alpha) * timestamps[i] + alpha * timestamps[i + 1]
    return interp_timestamps

def get_midpoint(joints):
    """
    Calculate the midpoint between the tips of the thumb and index finger.
    
    Args:
        joints: A tensor representing hand joints with shape [T, 21, 3] where T is time steps,
                21 is the number of joints, and 3 represents 3D coordinates.
    
    Returns:
        A tensor with shape [T, 3] containing the midpoint coordinates between the thumb and index finger tips.
    """
    if len(joints.shape) == 4:  # [batch_size, time_steps, 21, 3]
        joints = joints.squeeze(0)  # Remove batch dimension

    # MANO joint indices for thumb and index finger tips
    thumb_tip_idx = 4   # Thumb tip index
    index_tip_idx = 8   # Index finger tip index

    # Extract tip coordinates
    thumb_tip = joints[:, thumb_tip_idx]  # [T, 3]
    index_tip = joints[:, index_tip_idx]  # [T, 3]

    # Compute midpoint
    midpoint = (thumb_tip + index_tip) / 2.0  # [T, 3]

    return midpoint



# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a single hand video episode.")
    parser.add_argument("--file_path", required=True, help="Path to the source episode image directory.")
    parser.add_argument("--new_base_path", required=True, help="Path to the new base output directory.")
    parser.add_argument("--episode_name", required=True, help="Sequential output name for the episode (e.g., episode_000000).")
    parser.add_argument("--enable_visualization", action='store_true', help="Enable visualization output.")

    args = parser.parse_args()

    # Call the main processing function
    process_hand_video(
        input_episode_path=args.file_path,
        enable_visualization=args.enable_visualization,
        new_base_path=args.new_base_path,
        output_episode_name=args.episode_name
    )