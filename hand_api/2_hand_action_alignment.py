#!/usr/bin/env python3
"""
Batch update script for hand dataset.

This script updates the data portion of all episodes in a dataset in batch, without modifying videos or other data.
It recalculates and updates all parquet files using the data generation logic from hand_single_process_api.py.

Usage:
python update_handdataset.py --dataset_path /path/to/dataset
"""

import os
import sys
import math
import glob
import shutil
import argparse
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

# Import required functions
from hawor_api import (
    detect_track_video, hawor_motion_estimation, hawor_infiller,
    hawor_slam, compute_all_finger_distances, computer_thumb_index_thenar_midpoint
)
from lib.eval_utils.custom_utils import world2cam_convert, load_slam_cam
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from hawor.utils.rotation import (
    angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis,
    angle_axis_to_quaternion
)

def clean_nan_sequence(data_tensor):
    """
    清理张量中的NaN值，使用前向填充方法
    
    Args:
        data_tensor: 输入张量，可能包含NaN值
        
    Returns:
        清理后的张量，NaN值被替换为前一个有效值
    """
    if data_tensor is None or data_tensor.numel() == 0:
        return data_tensor
    
    # 确保是2D张量 [T, 1]
    if data_tensor.ndim == 1:
        data_tensor = data_tensor.unsqueeze(1)
    
    cleaned_tensor = data_tensor.clone()
    
    # 对每一列进行前向填充
    for col in range(cleaned_tensor.shape[1]):
        col_data = cleaned_tensor[:, col]
        
        # 找到第一个非NaN值作为初始值
        first_valid_idx = None
        for i in range(len(col_data)):
            if not torch.isnan(col_data[i]).any():
                first_valid_idx = i
                break
        
        if first_valid_idx is None:
            # 如果所有值都是NaN，用0填充
            cleaned_tensor[:, col] = 0.0
            continue
        
        # 前向填充
        last_valid_value = col_data[first_valid_idx]
        for i in range(len(col_data)):
            if torch.isnan(col_data[i]).any():
                cleaned_tensor[i, col] = last_valid_value
            else:
                last_valid_value = col_data[i]
    
    return cleaned_tensor

def clear_existing_data(dataset_path, backup=True):
    """
    Clear existing data files before regeneration.
    
    Args:
        dataset_path: Root path of the dataset.
        backup: Whether to backup before clearing.
    
    Returns:
        Number of files cleared.
    """
    dataset_path = Path(dataset_path)
    data_path = dataset_path / "data" / "chunk-000"
    
    if not data_path.exists():
        print(f"Data path does not exist: {data_path}")
        return 0
    
    # Find all parquet files
    parquet_files = list(data_path.glob("*.parquet"))
    
    if not parquet_files:
        print("No existing parquet files found to clear.")
        return 0
    
    cleared_count = 0
    
    for parquet_file in parquet_files:
        if backup:
            # Create backup with timestamp
            backup_path = parquet_file.with_suffix(f'.parquet.backup_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}')
            try:
                shutil.copy2(parquet_file, backup_path)
                print(f"Backed up: {parquet_file.name} -> {backup_path.name}")
            except Exception as e:
                print(f"Warning: Failed to backup {parquet_file.name}: {e}")
        
        # Delete the original file
        try:
            parquet_file.unlink()
            print(f"Deleted: {parquet_file.name}")
            cleared_count += 1
        except Exception as e:
            print(f"Error: Failed to delete {parquet_file.name}: {e}")
    
    print(f"Cleared {cleared_count} existing data files.")
    return cleared_count

def quaternion_to_rpy(quaternion):
    """Convert quaternion to roll, pitch, yaw Euler angles."""
    qw, qx, qy, qz = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    roll = torch.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    sinp = 2 * (qw * qy - qz * qx)
    pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * (torch.pi / 2), torch.asin(sinp))
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return torch.stack([roll, pitch, yaw], dim=-1)

def rotation_matrix_to_rpy(rotation_matrix, order='ZYX'):
    """
    将旋转矩阵转换为欧拉角。
    支持两种顺序：
        - 'ZYX'（默认）：roll-pitch-yaw（即绕Z、Y、X轴，常用于RPY）
        - 'XYZ'：欧拉角（即绕X、Y、Z轴）
    输入:
        rotation_matrix: (..., 3, 3) 的张量
        order: 'ZYX' 或 'XYZ'
    输出:
        euler: (..., 3) 的张量，分别为对应顺序的欧拉角
    """
    # 保证输入为张量
    if not torch.is_tensor(rotation_matrix):
        rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32)
    # 展平为二维
    orig_shape = rotation_matrix.shape
    rotation_matrix = rotation_matrix.reshape(-1, 3, 3)

    if order.upper() == 'ZYX':
        # RPY（roll-pitch-yaw，ZYX顺序）
        r = torch.atan2(rotation_matrix[:, 2, 1], rotation_matrix[:, 2, 2])
        p = torch.asin(-rotation_matrix[:, 2, 0].clamp(-1.0, 1.0))
        y = torch.atan2(rotation_matrix[:, 1, 0], rotation_matrix[:, 0, 0])
        euler = torch.stack([r, p, y], dim=-1)
    elif order.upper() == 'XYZ':
        # XYZ欧拉角
        x = torch.atan2(-rotation_matrix[:, 1, 2], rotation_matrix[:, 2, 2])
        y = torch.asin(rotation_matrix[:, 0, 2].clamp(-1.0, 1.0))
        z = torch.atan2(-rotation_matrix[:, 0, 1], rotation_matrix[:, 0, 0])
        euler = torch.stack([x, y, z], dim=-1)
    else:
        raise ValueError("order参数仅支持'ZYX'或'XYZ'")

    # 恢复原始批次维度
    if len(orig_shape) > 2:
        euler = euler.reshape(*orig_shape[:-2], 3)
    return euler

def find_longest_non_nan_sequence(data_tensor):
    """Find the longest non-NaN sequence in a tensor."""
    if data_tensor is None or data_tensor.numel() < 2:
        return None

    if data_tensor.ndim == 2 and data_tensor.shape[1] == 1:
        data_tensor = data_tensor.squeeze(1)
    elif data_tensor.ndim > 1:
        data_tensor = data_tensor.flatten()

    # 检查是否包含NaN值
    if torch.isnan(data_tensor).any():
        print(f"Warning: Input tensor contains {torch.isnan(data_tensor).sum()} NaN values")
        # 尝试清理NaN值
        data_tensor = clean_nan_sequence(data_tensor.unsqueeze(1)).squeeze(1)

    is_not_nan = ~torch.isnan(data_tensor)
    if not is_not_nan.any():
        print("Error: All values are NaN after cleaning")
        return None

    # 如果所有值都是有效的，返回整个序列
    if is_not_nan.all():
        return (0, len(data_tensor) - 1)

    is_not_nan_long = is_not_nan.long()
    prepended_value = torch.tensor([0], dtype=torch.long, device=data_tensor.device)

    try:
        data_with_prepend = torch.cat((prepended_value, is_not_nan_long), dim=0)
    except RuntimeError as e:
        print("!!! Error during torch.cat !!!")
        print("Shape of prepended_value:", prepended_value.shape)
        print("Shape of is_not_nan_long:", is_not_nan_long.shape)
        raise e

    change_points = torch.diff(data_with_prepend)
    starts_in_diff = torch.where(change_points == 1)[0]
    ends_in_diff = torch.where(change_points == -1)[0]

    starts_original = starts_in_diff
    ends_original_calc = ends_in_diff

    if data_tensor.numel() > 0 and is_not_nan[0]:
        if starts_original.numel() == 0 or starts_original[0] != 0:
            starts_original = torch.cat((torch.tensor([0], device=starts_original.device), starts_original))

    if len(starts_original) > len(ends_original_calc):
        ends_original_calc = torch.cat((ends_original_calc, torch.tensor([len(data_tensor)], device=ends_original_calc.device)))

    min_len = min(len(starts_original), len(ends_original_calc))
    starts_original = starts_original[:min_len]
    ends_original_calc = ends_original_calc[:min_len]

    best_len = 0
    best_seq = None

    if starts_original.numel() > 0:
        for start, end_calc in zip(starts_original, ends_original_calc):
            length = end_calc - start
            if length >= 2 and length > best_len:
                best_len = length
                best_seq = (start.item(), end_calc.item() - 1)

    return best_seq

def compute_rotation_svd_robust(points_a: torch.Tensor, points_b: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    """Compute rotation matrix between two point sets using weighted SVD."""
    # 检查输入是否包含非有限值
    if not torch.isfinite(points_a).all() or not torch.isfinite(points_b).all():
        print("Warning: Input contains non-finite values, cleaning...")
        # 使用更智能的清理方法
        points_a = clean_nan_sequence(points_a)
        points_b = clean_nan_sequence(points_b)
        
        # 再次检查
        if not torch.isfinite(points_a).all() or not torch.isfinite(points_b).all():
            print("Error: Failed to clean non-finite values, using identity matrix")
            # 返回单位矩阵作为fallback
            if points_a.dim() == 2:
                return torch.eye(3, device=points_a.device, dtype=points_a.dtype)
            else:
                return torch.eye(3, device=points_a.device, dtype=points_a.dtype).unsqueeze(0).expand(points_a.shape[0], -1, -1)

    if points_a.dim() == 2:
        points_a = points_a.unsqueeze(0)
        points_b = points_b.unsqueeze(0)
        single_batch = True
    else:
        single_batch = False

    batch_size, n_points, _ = points_a.shape

    if weights is None:
        weights = torch.ones(batch_size, n_points, device=points_a.device, dtype=points_a.dtype)
    elif weights.dim() == 1:
        weights = weights.unsqueeze(0).expand(batch_size, -1)

    weights = weights / weights.sum(dim=1, keepdim=True)

    centroid_a = torch.sum(points_a * weights.unsqueeze(-1), dim=1, keepdim=True)
    centroid_b = torch.sum(points_b * weights.unsqueeze(-1), dim=1, keepdim=True)

    points_a_centered = points_a - centroid_a
    points_b_centered = points_b - centroid_b

    points_a_weighted = points_a_centered * torch.sqrt(weights).unsqueeze(-1)
    points_b_weighted = points_b_centered * torch.sqrt(weights).unsqueeze(-1)

    H = torch.bmm(points_a_weighted.transpose(1, 2), points_b_weighted)

    try:
        U, S, Vt = torch.linalg.svd(H)
    except torch._C._LinAlgError as e:
        print(f"SVD failed, using pseudo-inverse: {e}")
        try:
            H_pinv = torch.linalg.pinv(H)
            U, S, Vt = torch.linalg.svd(H_pinv)
        except Exception as e2:
            print(f"Pseudo-inverse also failed: {e2}, using identity matrix")
            if single_batch:
                return torch.eye(3, device=points_a.device, dtype=points_a.dtype)
            else:
                return torch.eye(3, device=points_a.device, dtype=points_a.dtype).unsqueeze(0).expand(batch_size, -1, -1)

    rotation_matrix = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))

    det = torch.linalg.det(rotation_matrix)
    correction = torch.eye(3, device=points_a.device, dtype=points_a.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    correction = correction.clone()
    correction[:, 2, 2] = det

    rotation_matrix = torch.bmm(rotation_matrix, correction)

    if single_batch:
        rotation_matrix = rotation_matrix.squeeze(0)

    return rotation_matrix

def linear_interpolation(data, factor=2, episode_name=""):
    """Linear interpolation for data."""
    if data is None or data.shape[0] < 2:
        return data if data is not None else torch.empty((0, data.shape[1] if data is not None else 0))

    n, d = data.shape
    if torch.isnan(data).any():
        print(f"Warning [{episode_name}]: NaN detected in data passed to linear_interpolation.")

    interp_n = (n - 1) * factor + 1
    interp_data = torch.zeros((interp_n, d), dtype=data.dtype, device=data.device)

    for i in range(n):
        interp_data[i * factor] = data[i]

    for i in range(n - 1):
        start_frame = data[i]
        end_frame = data[i + 1]
        if torch.isnan(start_frame).any() or torch.isnan(end_frame).any():
            for j in range(1, factor):
                interp_data[i * factor + j] = start_frame
            continue

        for j in range(1, factor):
            alpha = j / factor
            interp_data[i * factor + j] = (1 - alpha) * start_frame + alpha * end_frame

    return interp_data

def update_episode_data(episode_path, coordinate_transformed=True,
                        eef_type="wrist_joint", interpolation_factor=4):
    """
    Update data for a single episode.

    Args:
        episode_path: Path to the episode.
        coordinate_transformed: Whether to transform coordinates.
        eef_type: End-effector type.
        interpolation_factor: Interpolation factor.

    Returns:
        Updated data dict.
    """
    print(f"Processing episode: {episode_path}")

    try:
        # 1. Detection and tracking
        start_idx, end_idx, seq_folder, imgfiles = detect_track_video(episode_path)
        if not imgfiles:
            print(f"Skipping {episode_path}: No image files found")
            return None

        # 2. Motion estimation
        frame_chunks_all, img_focal = hawor_motion_estimation(episode_path, start_idx, end_idx, seq_folder)
        valid_hand = list(frame_chunks_all.keys())
        if not valid_hand:
            print(f"Skipping {episode_path}: No valid hand detected")
            return None

        # 3. SLAM
        slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
        if not os.path.exists(slam_path):
            # 优先尝试复用旧SLAM（仅当当前序列是旧序列的尾部截断时）
            try:
                slam_dir = os.path.join(seq_folder, "SLAM")
                os.makedirs(slam_dir, exist_ok=True)
                candidate_files = [f for f in os.listdir(slam_dir) if f.startswith("hawor_slam_w_scale_") and f.endswith(".npz")]

                def parse_start_end(filename: str):
                    try:
                        stem = filename.replace("hawor_slam_w_scale_", "").replace(".npz", "")
                        parts = stem.split("_")
                        s = int(parts[0])
                        e = int(parts[1])
                        return s, e
                    except Exception:
                        return None

                # 只考虑 start_idx 相同且旧 end_idx >= 新 end_idx 的文件，优先选择 end_idx 最大的
                reusable = []
                for fname in candidate_files:
                    se = parse_start_end(fname)
                    if se is None:
                        continue
                    s, e = se
                    if s == start_idx and e >= end_idx:
                        reusable.append((e, fname))
                reusable.sort(reverse=True)

                reused = False
                for _, fname in reusable:
                    old_path = os.path.join(slam_dir, fname)
                    try:
                        pred_cam = dict(np.load(old_path, allow_pickle=True))
                        tstamp = pred_cam.get('tstamp')
                        disps = pred_cam.get('disps')
                        traj = pred_cam.get('traj')
                        img_focal = pred_cam.get('img_focal')
                        img_center = pred_cam.get('img_center')
                        scale = pred_cam.get('scale')

                        if tstamp is None or disps is None or traj is None:
                            print(f"Skip reuse {fname}: missing keys")
                            continue

                        # 仅保留 tstamp < len(imgfiles) 的关键帧，保证与新图像序列对齐
                        max_valid_idx = len(imgfiles) - 1
                        valid_mask = tstamp <= max_valid_idx
                        if valid_mask.sum() < 2:
                            print(f"Skip reuse {fname}: too few valid keyframes after truncation")
                            continue

                        tstamp_new = tstamp[valid_mask]
                        disps_new = disps[valid_mask]
                        traj_new = traj[valid_mask]

                        # 另存为当前期望文件名
                        np.savez(slam_path,
                                 tstamp=tstamp_new,
                                 disps=disps_new,
                                 traj=traj_new,
                                 img_focal=img_focal,
                                 img_center=img_center,
                                 scale=scale)
                        print(f"复用并截断旧SLAM成功: {fname} -> {os.path.basename(slam_path)} (frames={len(imgfiles)})")
                        reused = True
                        break
                    except Exception as e:
                        print(f"Failed to reuse {fname}: {e}")

                if not reused:
                    # 无可复用旧文件，再运行SLAM
                    hawor_slam(episode_path, start_idx, end_idx, seq_folder, enable_camera_jitter=True)
            except Exception as e:
                print(f"SLAM复用流程异常，回退到重跑：{e}")
                hawor_slam(episode_path, start_idx, end_idx, seq_folder, enable_camera_jitter=True)

        if not os.path.exists(slam_path):
            print(f"Skipping {episode_path}: SLAM file not found")
            return None

        R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)

        # 4. Hand pose estimation
        pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(
            episode_path, start_idx, end_idx, frame_chunks_all, seq_folder
        )

        def print_nan_indices(tensor, name):
            if torch.isnan(tensor).any():
                non_nan_indices = torch.nonzero(~torch.isnan(tensor), as_tuple=False)
                print(f"{name} 非NaN数据索引如下：")
                print(non_nan_indices)
                return min(non_nan_indices[:,0]), max(non_nan_indices[:,0])
            else:
                return 0,len(tensor)-1


        # clean step1 
        start_idx_trans,end_idx_trans=print_nan_indices(pred_trans[1], "pred_trans")
        start_idx_rot,end_idx_rot=print_nan_indices(pred_rot[1], "pred_rot")
        start_idx_hand_pose,end_idx_hand_pose=print_nan_indices(pred_hand_pose[1], "pred_hand_pose")
        start_idx_betas,end_idx_betas=print_nan_indices(pred_betas[1], "pred_betas")
        
        # 检查是否有有效数据
        if any(idx is None for idx in [start_idx_trans, start_idx_rot, start_idx_hand_pose, start_idx_betas]):
            print(f"Skipping {episode_path}: Some data contains all NaN values")
            return None

        start_idx = max(start_idx_trans, start_idx_rot, start_idx_hand_pose, start_idx_betas)
        end_idx = min(end_idx_trans, end_idx_rot, end_idx_hand_pose, end_idx_betas)

        print(f"start_idx: {start_idx}, end_idx: {end_idx}")
        pred_trans=pred_trans[:,start_idx:end_idx,:]
        pred_rot=pred_rot[:,start_idx:end_idx,:]
        pred_hand_pose=pred_hand_pose[:,start_idx:end_idx,:]
        pred_betas=pred_betas[:,start_idx:end_idx,:]

        R_w2c_sla_all = R_w2c_sla_all[start_idx:end_idx, :]
        t_w2c_sla_all = t_w2c_sla_all[start_idx:end_idx, :]
        R_c2w_sla_all = R_c2w_sla_all[start_idx:end_idx, :]
        t_c2w_sla_all = t_c2w_sla_all[start_idx:end_idx, :]


        # Select hand
        if len(valid_hand) == 1:
            use_hand_idx = valid_hand[0]
        elif len(valid_hand) == 2:
            use_hand_idx = 1  # Default to right hand
        else:
            print(f"Skipping {episode_path}: Unexpected hand detection result: {valid_hand}")
            return None

        # Select hand data
        hand_pred_trans = pred_trans[use_hand_idx]
        hand_pred_rot = pred_rot[use_hand_idx]
        hand_pred_hand_pose = pred_hand_pose[use_hand_idx]
        hand_pred_betas = pred_betas[use_hand_idx]

        # 5. Coordinate transform
        data_world_single_hand = {
            "init_root_orient": hand_pred_rot.unsqueeze(0),
            "init_hand_pose": hand_pred_hand_pose.unsqueeze(0),
            "init_trans": hand_pred_trans.unsqueeze(0),
            "init_betas": hand_pred_betas.unsqueeze(0)
        }

        hand_label = "right" if use_hand_idx == 1 else "left"
        data_cam_single_hand = world2cam_convert(R_w2c_sla_all, t_w2c_sla_all, data_world_single_hand, hand_label)

        cam_trans_wrist = data_cam_single_hand["init_trans"].squeeze(0)
        cam_root_orient_aa = data_cam_single_hand["init_root_orient"].squeeze(0)
        cam_hand_pose = data_cam_single_hand["init_hand_pose"].squeeze(0)
        cam_betas = data_cam_single_hand["init_betas"].squeeze(0)

        # 6. Compute open/close distance
        mano_fn = run_mano if use_hand_idx == 1 else run_mano_left
        pred_glob_cam = mano_fn(
            cam_trans_wrist.unsqueeze(0),
            cam_root_orient_aa.unsqueeze(0),
            cam_hand_pose.unsqueeze(0),
            betas=cam_betas.unsqueeze(0)
        )
        cam_joints = pred_glob_cam['joints'].squeeze(0)
        
        # 检查关节坐标是否包含NaN
        if torch.isnan(cam_joints).any():
            print(f"Warning: cam_joints contains {torch.isnan(cam_joints).sum()} NaN values")
            print(f"cam_joints shape: {cam_joints.shape}")

        fingd_dict = compute_all_finger_distances(cam_joints)
        thumb_to_index_distance = fingd_dict["thumb_to_index_distance"]
        
        # 检查手指距离是否包含NaN值
        if torch.isnan(thumb_to_index_distance).any():
            print(f"Warning: thumb_to_index_distance contains {torch.isnan(thumb_to_index_distance).sum()} NaN values")
            print(f"thumb_to_index_distance shape: {thumb_to_index_distance.shape}")
            # 使用前向填充方法清理NaN值
            thumb_to_index_distance = clean_nan_sequence(thumb_to_index_distance)
        
        open_distance = (thumb_to_index_distance * 100).detach()
        
        # 先获取open_distance的最大值和最小值，再进行线性映射
        open_min = open_distance.min()
        open_max = open_distance.max()
        print(f"open_distance range: [{open_min:.4f}, {open_max:.4f}]")

        # 将open_distance中最小的30%设置为最小值（基于排序）
        n = open_distance.shape[0]
        print(f"open_distance shape: {open_distance.shape}")
        tail_num = int(n * 0.3)
        # 修正：确保tail_num不超过n且大于0，并且指定dim参数
        if 0 < tail_num <= n:
            # 指定dim=0，确保在正确的维度上排序
            _, indices = torch.topk(open_distance, tail_num, largest=False, dim=0)
            open_distance[indices] = open_min

        # 线性映射至[14,90]
        open_distance = (open_distance - open_min) / (open_max - open_min) * (90 - 14) + 14

        if coordinate_transformed:
            # cam_to_base_matrix = torch.tensor([
            #     [-0.48238306, -0.37595801,  0.7911777,   0.01460186],
            #     [-0.84657845,  0.4320609,  -0.31085096,  0.20225367],
            #     [-0.22497004, -0.81974323, -0.5266968,   0.41582841],
            #     [ 0.0,          0.0,          0.0,          1.0]
            # ], dtype=cam_trans_wrist.dtype, device=cam_trans_wrist.device)

            # cam_to_base_matrix = torch.tensor([
            #     [ 0.00824967, -0.99953842,  0.02923837, 0.215],
            #     [-0.99247578, -0.01175646, -0.12187538,  -0.085 ],
            #     [-0.12216287,  0.02801294,  0.99211467,  -0.585 ],
            #     [ 0.0        ,  0.0        ,  0.0        ,  1.0 ]
            # ], dtype=torch.float32)

            # R = cam_to_base_matrix[:3, :3].to(cam_trans_wrist.device)
            # t = cam_to_base_matrix[:3, 3].to(cam_trans_wrist.device)

            R = torch.tensor([
            [0.99969799, -0.01875905,  0.0158754],
            [0.01373589,  0.9621979,   0.27200465],
            [-0.02037782, -0.27170444,  0.96216498]
            ], dtype=torch.float32)

            R = R @ torch.tensor([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], dtype=torch.float32) 

            t = torch.tensor([0.085, 0.215, 0.585])
            
            base_trans_wrist = torch.matmul(cam_trans_wrist, R.T) + t
            base_joints = torch.matmul(cam_joints.to(cam_trans_wrist.device), R.T) + t
            
            # 先计算rpy再进行修正
            thumb_pip_base = base_joints[:, 3, :].unsqueeze(1)
            thumb_mcp_base = base_joints[:, 4, :].unsqueeze(1)
            index_mcp_base = base_joints[:, 5, :].unsqueeze(1)
            thumb_index_midpoint_base = computer_thumb_index_thenar_midpoint(base_joints).unsqueeze(1)

            thumb_pip_local = torch.tensor([0.035, 0.025, 0.0], dtype=torch.float32)
            thumb_mcp_local = torch.tensor([0., 0.025, 0.0], dtype=torch.float32)
            index_mcp_local = torch.tensor([0., -0.025, 0.0], dtype=torch.float32)
            thumb_index_midpoint_local = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

            thumb_pip_local_batch = thumb_pip_local.unsqueeze(0).expand(base_joints.shape[0], -1).unsqueeze(1)
            thumb_mcp_local_batch = thumb_mcp_local.unsqueeze(0).expand(base_joints.shape[0], -1).unsqueeze(1)
            index_mcp_local_batch = index_mcp_local.unsqueeze(0).expand(base_joints.shape[0], -1).unsqueeze(1)
            thumb_index_midpoint_local_batch = thumb_index_midpoint_local.unsqueeze(0).expand(base_joints.shape[0], -1).unsqueeze(1)

            points_local = torch.cat([thumb_pip_local_batch, thumb_mcp_local_batch, index_mcp_local_batch, thumb_index_midpoint_local_batch], dim=1)
            points_base = torch.cat([thumb_pip_base, thumb_mcp_base, index_mcp_base, thumb_index_midpoint_base], dim=1)
            print(f"points_local.shape:{points_local.shape}")
            print(f"points_base.shape:{points_base.shape}")
            R_thenar = compute_rotation_svd_robust(points_local, points_base, weights=None)

            # 由于标定变化，先使用0 
            base_root_r = -(math.pi + rotation_matrix_to_rpy(R_thenar, order='ZYX')[:, 0])

            # 保证base_root_r在[-pi, pi]区间
            base_root_r = (base_root_r + math.pi) % (2 * math.pi) - math.pi

            # !!!!!!!!!!!!!!!!!!!!!!!!左手为-，右手为+!!!!!!!!!!!!!!!!!!!!!!!
            base_root_r = base_root_r + 1.6
            # 保证base_root_r在[0, pi]区间
            base_root_r = base_root_r % (math.pi/2)
           
            # !!!!!!!!!!!!!!!!!!!!!!!!注意区分左右手!!!!!!!!!!!!!!!!!!!!!!!
            base_root_p = -rotation_matrix_to_rpy(R_thenar, order='ZYX')[:, 1]
            # 控制base_root_p在(0, pi/2)区间
            base_root_p = base_root_p.clamp(min=1e-6, max=math.pi/2 - 1e-6)
            base_root_y = torch.zeros_like(base_root_p)
            base_root_rpy = torch.stack([base_root_r, base_root_p, base_root_y], dim=1)

            cam_root_rotmat = angle_axis_to_rotation_matrix(cam_root_orient_aa.to(cam_trans_wrist.device))
            base_rotmat = torch.matmul(cam_root_rotmat, R.T)
            base_root_orient_aa = rotation_matrix_to_angle_axis(base_rotmat)

            # 调整base_trans_wrist和base_joints的x轴坐标,由于摄像头安装高度不同导致误差
            base_trans_wrist[:,0] = base_trans_wrist[:,0] + 0.14
            base_joints[:,:,0] = base_joints[:,:,0] + 0.14

            # 调整base_trans_wrist和base_joints的y轴坐标,由于摄像头安装高度不同导致误差
            base_trans_wrist[:,1] = base_trans_wrist[:,1] + 0.1
            base_joints[:,:,1] = base_joints[:,:,1] + 0.1

            # 调整base_trans_wrist和base_joints的z轴坐标,由于摄像头安装高度不同导致误差
            base_trans_wrist[:,-1] = base_trans_wrist[:,-1] - 0.28
            base_joints[:,:,-1] = base_joints[:,:,-1] - 0.28


        else:
            base_trans_wrist = cam_trans_wrist
            base_joints = cam_joints
            base_root_orient_aa = cam_root_orient_aa


        # 8. Prepare data
        if eef_type == "wrist_joint":
            cam_trans = cam_trans_wrist
            base_trans = base_trans_wrist
            # base_trans[:,2] = base_trans[:,2] - 0.08
            
            cam_root_orient_quat = angle_axis_to_quaternion(cam_root_orient_aa)
            cam_root_orient_rpy = quaternion_to_rpy(cam_root_orient_quat).to(cam_trans.device)
            base_root_orient_quat = angle_axis_to_quaternion(base_root_orient_aa)
            base_root_orient_rpy = quaternion_to_rpy(base_root_orient_quat).to(cam_trans.device)

        elif eef_type == "thenar_middle_joint":
            cam_trans = computer_thumb_index_thenar_midpoint(cam_joints)
            base_trans = computer_thumb_index_thenar_middle_joint(base_joints)
            
            cam_root_orient_quat = angle_axis_to_quaternion(cam_root_orient_aa)
            cam_root_orient_rpy = quaternion_to_rpy(cam_root_orient_quat).to(cam_trans.device)
            base_root_orient_rpy = base_root_rpy

            # 需要对z轴进行线性映射至[-0.01, max(0.16, max(base_trans_wrist[:,-1]))]
            first_frame_z = base_trans[0, -1]

            # 如果first_frame_z小于0.176，则将所有z值线性缩放，使得first_frame_z变为0.176，其余点等比例缩放
            if float(first_frame_z) < 0.13:
                scale = 0.13 / float(first_frame_z) if abs(float(first_frame_z)) > 1e-8 else 1.0
                print(f"first_frame_z={first_frame_z:.5f} < 0.2，进行缩放，缩放比例为 {scale:.5f}")
                base_trans[:, -1] = base_trans[:, -1] * scale
            # else:
            #     min_z = base_trans[:, -1].min()
            #     max_z = base_trans[:, -1].max()
            #     target_min = -0.03
            #     target_max = max(0.176, float(max_z))
            #     print(f"z轴线性映射: 原始min={min_z:.5f}, max={max_z:.5f}，目标区间=[{target_min}, {target_max}]")
            #     if abs(max_z - min_z) < 1e-8:
            #         base_trans[:, -1] = target_min
            #     else:
            #         base_trans[:, -1] = (base_trans[:, -1] - min_z) / (max_z - min_z) * (target_max - target_min) + target_min

        # smooth
        def exponential_smoothing(tensor, alpha=0.1):
            """
            指数移动平均平滑
            
            Args:
                tensor: 形状为 [L, N] 的 tensor
                alpha: 平滑因子 (0 < alpha < 1)，越小越平滑
            
            Returns:
                平滑后的 tensor
            """
            L, N = tensor.shape
            smoothed = torch.zeros_like(tensor)
            smoothed[0] = tensor[0]  # 第一个元素保持不变
            
            for i in range(1, L):
                smoothed[i] = alpha * tensor[i] + (1 - alpha) * smoothed[i-1]
            
            return smoothed 

        base_trans = exponential_smoothing(base_trans, alpha=0.2)
        base_root_orient_rpy = exponential_smoothing(base_root_orient_rpy, alpha=0.1)


        # 9. Find longest valid sequence
        longest_seq_trans_x = find_longest_non_nan_sequence(base_trans[:,0].cpu())
        longest_seq_trans_y = find_longest_non_nan_sequence(base_trans[:,1].cpu())
        longest_seq_trans_z = find_longest_non_nan_sequence(base_trans[:,2].cpu())

        longest_seq_rpy_r = find_longest_non_nan_sequence(base_root_orient_rpy[:,0].cpu())
        longest_seq_rpy_p = find_longest_non_nan_sequence(base_root_orient_rpy[:,1].cpu())
        longest_seq_rpy_y = find_longest_non_nan_sequence(base_root_orient_rpy[:,2].cpu())
        
        base_trans_x=clean_nan_sequence(base_trans[:,0])
        base_trans_y=clean_nan_sequence(base_trans[:,1])
        base_trans_z=clean_nan_sequence(base_trans[:,2])
        base_root_orient_rpy_r=clean_nan_sequence(base_root_orient_rpy[:,0])
        base_root_orient_rpy_p=clean_nan_sequence(base_root_orient_rpy[:,1])
        base_root_orient_rpy_y=clean_nan_sequence(base_root_orient_rpy[:,2])

        base_trans=torch.cat([base_trans_x,base_trans_y,base_trans_z],dim=-1)
        base_root_orient_rpy=torch.cat([base_root_orient_rpy_r,base_root_orient_rpy_p,base_root_orient_rpy_y],dim=-1)

        print(f"longest_seq_trans_x: {longest_seq_trans_x[0]}, {longest_seq_trans_x[1]}")
        print(f"longest_seq_trans_y: {longest_seq_trans_y[0]}, {longest_seq_trans_y[1]}")
        print(f"longest_seq_trans_z: {longest_seq_trans_z[0]}, {longest_seq_trans_z[1]}")

        print(f"longest_seq_rpy_r: {longest_seq_rpy_r[0]}, {longest_seq_rpy_r[1]}")
        print(f"longest_seq_rpy_p: {longest_seq_rpy_p[0]}, {longest_seq_rpy_p[1]}")
        print(f"longest_seq_rpy_y: {longest_seq_rpy_y[0]}, {longest_seq_rpy_y[1]}")

        longest_seq_trans_start = max(longest_seq_trans_x[0], longest_seq_trans_y[0], longest_seq_trans_z[0])
        longest_seq_trans_end = min(longest_seq_trans_x[1], longest_seq_trans_y[1], longest_seq_trans_z[1])

        longest_seq_rpy_start = max(longest_seq_rpy_r[0], longest_seq_rpy_p[0], longest_seq_rpy_y[0])
        longest_seq_rpy_end = min(longest_seq_rpy_r[1], longest_seq_rpy_p[1], longest_seq_rpy_y[1])

        if longest_seq_trans_start is None or longest_seq_trans_end is None or longest_seq_rpy_start is None or longest_seq_rpy_end is None:
            print(f"Skipping {episode_path}: No valid non-NaN sequence found")
            return None

        k_start = max(longest_seq_trans_start, longest_seq_rpy_start)
        k_end = min(longest_seq_trans_end, longest_seq_rpy_end)

        print(f"k_start: {k_start}, k_end: {k_end}")

        seq_len = k_end - k_start + 1
        print(f"Found longest non-NaN sequence: index {k_start} to {k_end} (inclusive), length {seq_len}")

        state_indices = torch.arange(k_start, k_end)
        action_indices = torch.arange(k_start + 1, k_end + 1)

        device = cam_trans.device
        state_trans = base_trans[state_indices].to(device)
        state_rpy = base_root_orient_rpy[state_indices].to(device)
        state_open = open_distance[state_indices].to(device)

        action_trans = base_trans[action_indices].to(device)
        action_rpy = base_root_orient_rpy[action_indices].to(device)
        action_open = open_distance[action_indices].to(device)

        state_frames_data = torch.cat([state_trans, state_rpy, state_open], dim=-1).cpu()
        action_frames_data = torch.cat([action_trans, action_rpy, action_open], dim=-1).cpu()

        if torch.isnan(state_frames_data).any():
            print(f"Error: state_frames_data contains NaN")
            return None

        if torch.isnan(action_frames_data).any():
            print(f"Error: action_frames_data contains NaN")
            return None

        if interpolation_factor > 0:
            state_frames_interp = linear_interpolation(state_frames_data, interpolation_factor, "episode")
            action_frames_interp = linear_interpolation(action_frames_data, interpolation_factor, "episode")
        else:
            state_frames_interp = state_frames_data
            action_frames_interp = action_frames_data

        if torch.isnan(state_frames_interp).any():
            print(f"Error: state_frames_interp contains NaN after interpolation")
            return None

        if interpolation_factor > 0:
            original_start_time = (start_idx+k_start) * (1.0 / 30.0) * interpolation_factor  # 原始起始时间
        else:
            original_start_time = (start_idx+k_start) * (1.0 / 30.0)

        num_frames_interp = state_frames_interp.shape[0]
        print(f"original_start_time: {original_start_time}")
        timestamps_interp = np.arange(num_frames_interp) * (1.0 / 30.0)
        timestamps_interp = timestamps_interp + float(original_start_time)

        data_dict = {
            "action": action_frames_interp.numpy().tolist(),
            "observation.state": state_frames_interp.numpy().tolist(),
            "timestamp": timestamps_interp.tolist(),
            "frame_index": list(range(num_frames_interp)),
            "episode_index": [0] * num_frames_interp,
            "index": list(range(num_frames_interp)),
            "task_index": [0] * num_frames_interp,
        }

        print(f"Successfully generated data: {num_frames_interp} frames")
        return data_dict

    except Exception as e:
        print(f"Error processing episode: {e}")
        import traceback
        traceback.print_exc()
        return None

def discover_episodes(dataset_path):
    """
    Discover all episodes in the dataset.

    Args:
        dataset_path: Root path of the dataset.

    Returns:
        List of episode names.
    """
    dataset_path = Path(dataset_path)
    images_path = dataset_path / "images" / "observation.images.webcam"

    if not images_path.exists():
        print(f"Error: Image path does not exist: {images_path}")
        return []

    episode_dirs = [d for d in images_path.iterdir() if d.is_dir() and d.name.startswith('episode_')]
    episode_names = [d.name for d in episode_dirs]
    episode_names.sort()

    print(f"Found {len(episode_names)} episodes:")
    for name in episode_names:
        print(f"  - {name}")

    return episode_names

def batch_update_dataset(dataset_path, coordinate_transformed=True,
                        eef_type="wrist_joint", interpolation_factor=4, backup=True,
                        start_from=None, max_episodes=None, skip_failed=True):
    """
    Batch update all episodes in the dataset.

    Args:
        dataset_path: Root path of the dataset.
        coordinate_transformed: Whether to transform coordinates.
        eef_type: End-effector type.
        interpolation_factor: Interpolation factor.
        backup: Whether to backup original files.
        start_from: Start from which episode (optional).
        max_episodes: Max number of episodes to process (optional).
        skip_failed: Whether to skip failed episodes.
    """
    dataset_path = Path(dataset_path)
    print("=" * 60)
    print("CLEARING EXISTING DATA...")
    print("=" * 60)
    cleared_count = clear_existing_data(dataset_path, backup=backup)
    if cleared_count == 0:
        print("No existing data to clear.")
    print("=" * 60)

    episode_names = discover_episodes(dataset_path)
    if not episode_names:
        print("No episodes found")
        return

    if start_from:
        try:
            start_idx = episode_names.index(start_from)
            episode_names = episode_names[start_idx:]
            print(f"Starting from {start_from}")
        except ValueError:
            print(f"Warning: Start episode {start_from} not found")

    if max_episodes:
        episode_names = episode_names[:max_episodes]
        print(f"Limiting to {max_episodes} episodes")

    total_episodes = len(episode_names)
    successful_updates = 0
    failed_updates = 0
    skipped_episodes = 0

    print(f"\nStarting batch update for {total_episodes} episodes...")
    print("=" * 60)

    for i, episode_name in enumerate(tqdm(episode_names, desc="Update Progress")):
        print(f"\n[{i+1}/{total_episodes}] Processing: {episode_name}")

        episode_path = dataset_path / "images" / "observation.images.webcam" / episode_name
        parquet_path = dataset_path / "data" / "chunk-000" / f"{episode_name}.parquet"

        if backup:
            backup_path = parquet_path.with_suffix('.parquet.backup')
            if not backup_path.exists():
                shutil.copy2(parquet_path, backup_path)
                print(f"Backed up: {backup_path}")

        try:
            updated_data = update_episode_data(
                episode_path=str(episode_path),
                coordinate_transformed=coordinate_transformed,
                eef_type=eef_type,
                interpolation_factor=interpolation_factor
            )

            if updated_data is not None:
                df = pd.DataFrame(updated_data)
                df.to_parquet(parquet_path, engine='pyarrow')
                print(f"✓ Updated: {episode_name}")
                successful_updates += 1
            else:
                print(f"✗ Update failed: {episode_name}")
                failed_updates += 1
                if not skip_failed:
                    print("skip_failed=False, stopping.")
                    break

        except Exception as e:
            print(f"✗ Exception while processing {episode_name}: {e}")
            failed_updates += 1
            if not skip_failed:
                print("skip_failed=False, stopping.")
                break

    print("\n" + "=" * 60)
    print("Batch update complete!")
    print(f"Total: {total_episodes} episodes")
    print(f"Success: {successful_updates}")
    print(f"Failed: {failed_updates}")
    print(f"Skipped: {skipped_episodes}")

    if successful_updates == total_episodes:
        print("🎉 All episodes updated successfully!")
    elif successful_updates > 0:
        print(f"⚠️ Some episodes updated successfully ({successful_updates}/{total_episodes})")
    else:
        print("❌ No episodes updated successfully")

def main():
    parser = argparse.ArgumentParser(description="Batch update all episode data in the dataset")
    parser.add_argument("--dataset_path", type=str,default=r"/home/admin123/projects_lerobot/Easymimic/lerobot-joycon_plus/datasets/pinkduck/human/human_23_pink_duck_0831_true_full_hand_thenar_middle_joint_0_2_clear", help="Root path of the dataset")
    parser.add_argument("--coordinate_transformed", type=bool, default=True, help="Whether to transform coordinates")
    parser.add_argument("--eef_type", type=str, default="thenar_middle_joint",
                        choices=["wrist_joint", "tip_middle_joint", "thenar_middle_joint"],
                        help="End-effector type")
    parser.add_argument("--interpolation_factor", type=int, default=0, help="Interpolation factor")
    parser.add_argument("--backup", action='store_true', default=False, help="Whether to backup original files")
    parser.add_argument("--start_from", type=str, default=None, help="Start from specified episode (e.g. episode_000000)")
    parser.add_argument("--max_episodes", type=int, default=None, help="Max number of episodes to process")
    parser.add_argument("--skip_failed", action='store_true', default=True, help="Skip failed episodes and continue")

    args = parser.parse_args()

    batch_update_dataset(
        dataset_path=args.dataset_path,
        coordinate_transformed=args.coordinate_transformed,
        eef_type=args.eef_type,
        interpolation_factor=args.interpolation_factor,
        backup=args.backup,
        start_from=args.start_from,
        max_episodes=args.max_episodes,
        skip_failed=args.skip_failed
    )

if __name__ == "__main__":
    main()