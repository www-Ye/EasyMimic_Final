import os
import numpy as np
import shutil
from glob import glob
from tqdm import tqdm
import json
import torch
import datetime
import cv2

from aitviewer.headless import HeadlessRenderer
from hawor_api import detect_track_video, hawor_motion_estimation, hawor_infiller, hawor_slam, compute_all_finger_distances
from lib.eval_utils.custom_utils import world2cam_convert, cam2world_convert, load_slam_cam
from lib.vis.run_vis2 import run_vis2_on_video, run_vis2_on_video_cam, run_vis2_on_video_cam_random_color
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from hawor.utils.rotation import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis, angle_axis_to_quaternion

def quaternion_to_rpy(quaternion):
    """Convert quaternion to roll, pitch, yaw Euler angles.
    
    Args:
        quaternion: tensor of shape (..., 4) in (w, x, y, z) format
        
    Returns:
        rpy: tensor of shape (..., 3) containing roll, pitch, yaw in radians
    """
    # Extract quaternion components
    qw, qx, qy, qz = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    # Handle gimbal lock edge cases
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * (torch.pi / 2),  # use 90 degrees if out of range
        torch.asin(sinp)
    )
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    
    return torch.stack([roll, pitch, yaw], dim=-1)

def process_hand_video(output_image_dir, enable_visualization=False):
    # print(123556)
    # breakpoint()
    
    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(output_image_dir)
    frame_chunks_all, img_focal = hawor_motion_estimation(output_image_dir, start_idx, end_idx, seq_folder)

    
    valid_hand = list(frame_chunks_all.keys())

    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    if not os.path.exists(slam_path):
        hawor_slam(output_image_dir, start_idx, end_idx, seq_folder)
    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)
    
    
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(output_image_dir, start_idx, end_idx, frame_chunks_all, seq_folder)
    print("pred_trans", pred_trans.shape)
    print("pred_rot", pred_rot.shape)
    print("pred_hand_pose", pred_hand_pose.shape)
    print("pred_betas", pred_betas.shape)
    print("pred_valid", pred_valid.shape)
    # breakpoint()

    # # 世界坐标系到相机坐标系(w2c)
    # R_w2c_sla = torch.from_numpy(extrinsics[serial][:3, :3]).to(torch.float32)  # 旋转矩阵(3x3)
    # t_w2c_sla = torch.from_numpy(extrinsics[serial][:3, 3]).to(torch.float32)   # 平移向量(3x1)

    # # 相机坐标系到世界坐标系(c2w)
    # extrinsics_inverse = np.linalg.inv(extrinsics[serial])
    # R_c2w_sla = torch.from_numpy(extrinsics_inverse[:3, :3]).to(torch.float32)  # 旋转矩阵(3x3)
    # t_c2w_sla = torch.from_numpy(extrinsics_inverse[:3, 3]).to(torch.float32)   # 平移向量(3x1)

    # # 添加批次维度
    # batch_size = len(image_timestamps)  # 您需要的批次大小
    # R_c2w_sla_all = R_c2w_sla.unsqueeze(0).expand(batch_size, 3, 3)  # 形状变为 [b,3,3]
    # t_c2w_sla_all = t_c2w_sla.unsqueeze(0).expand(batch_size, 3)      # 形状变为 [b,3]

    # R_w2c_sla_all = R_w2c_sla.unsqueeze(0).expand(batch_size, 3, 3)  # 形状变为 [b,3,3]
    # t_w2c_sla_all = t_w2c_sla.unsqueeze(0).expand(batch_size, 3)      # 形状变为 [b,3]

    vis_start = 0
    vis_end = pred_trans.shape[1] - 1

    # 创建世界坐标系数据结构
    data_world_right = {
        "init_root_orient": pred_rot[1:2],  # 右手，索引1
        "init_hand_pose": pred_hand_pose[1:2],
        "init_trans": pred_trans[1:2], 
        "init_betas": pred_betas[1:2]
    }

    data_world_left = {
        "init_root_orient": pred_rot[0:1],  # 左手，索引0
        "init_hand_pose": pred_hand_pose[0:1],
        "init_trans": pred_trans[0:1],
        "init_betas": pred_betas[0:1]
    }

    # data_right = data_world_right
    # data_left = data_world_left

    # 转换到相机坐标系
    data_cam_right = world2cam_convert(R_w2c_sla_all, t_w2c_sla_all, data_world_right, "right")
    data_cam_left = world2cam_convert(R_w2c_sla_all, t_w2c_sla_all, data_world_left, "left")
    print(f"相机坐标中right hand init_root_orient的形状: {data_cam_right['init_root_orient'].shape}")
    print(f"相机坐标中right hand init_hand_pose的形状: {data_cam_right['init_hand_pose'].shape}")
    print(f"相机坐标中right hand init_trans的形状: {data_cam_right['init_trans'].shape}")
    print(f"相机坐标中right hand init_betas的形状: {data_cam_right['init_betas'].shape}")

    print(f"相机坐标中left hand init_root_orient的形状: {data_cam_left['init_root_orient'].shape}")
    print(f"相机坐标中left hand init_hand_pose的形状: {data_cam_left['init_hand_pose'].shape}")
    print(f"相机坐标中left hand init_trans的形状: {data_cam_left['init_trans'].shape}")
    print(f"相机坐标中left hand init_betas的形状: {data_cam_left['init_betas'].shape}")

    # 直接使用相机坐标系数据
    data_right = data_cam_right
    data_left = data_cam_left

    # vis sequence for this video
    hand2idx = {
        "right": 1,
        "left": 0
    }
    vis_start = 0
    # vis_end = pred_trans.shape[1] - 1
    vis_end = pred_trans.shape[1]

    # get faces
    faces = get_mano_faces()
    faces_new = np.array([[92, 38, 234],
            [234, 38, 239],
            [38, 122, 239],
            [239, 122, 279],
            [122, 118, 279],
            [279, 118, 215],
            [118, 117, 215],
            [215, 117, 214],
            [117, 119, 214],
            [214, 119, 121],
            [119, 120, 121],
            [121, 120, 78],
            [120, 108, 78],
            [78, 108, 79]])
    faces_right = np.concatenate([faces, faces_new], axis=0)

    # 获取右手顶点
    hand = 'right'
    hand_idx = hand2idx[hand]
    pred_glob_r = run_mano(data_right["init_trans"][:, vis_start:vis_end], 
                        data_right["init_root_orient"][:, vis_start:vis_end], 
                        data_right["init_hand_pose"][:, vis_start:vis_end], 
                        betas=data_right["init_betas"][:, vis_start:vis_end])
    right_verts = pred_glob_r['vertices'][0]
    right_dict = {
            'vertices': right_verts.unsqueeze(0),
            'faces': faces_right,
        }

    # 获取左手顶点
    faces_left = faces_right[:,[0,2,1]]
    hand = 'left'
    hand_idx = hand2idx[hand]
    pred_glob_l = run_mano_left(data_left["init_trans"][:, vis_start:vis_end], 
                            data_left["init_root_orient"][:, vis_start:vis_end], 
                            data_left["init_hand_pose"][:, vis_start:vis_end], 
                            betas=data_left["init_betas"][:, vis_start:vis_end])
    left_verts = pred_glob_l['vertices'][0]
    left_dict = {
            'vertices': left_verts.unsqueeze(0),
            'faces': faces_left,
        }
    # print(pred_glob_l)
    # breakpoint()

    hand_data = {0: (data_left, pred_glob_l), 1: (data_right, pred_glob_r)}

    # # 如果需要可视化，才执行可视化相关代码
    # if enable_visualization:
    #     output_pth = os.path.join(seq_folder, f"cam_vis_{vis_start}_{vis_end}")
    #     if not os.path.exists(output_pth):
    #         os.makedirs(output_pth)
    #     image_names = imgfiles[vis_start:vis_end]
    #     print(f"visualizing camera space from {vis_start} to {vis_end}")

    #     print("valid_hand", valid_hand)
    #     # 在相机坐标系下可视化时，使用单位旋转矩阵和零平移向量
    #     run_vis2_on_video_cam(left_dict, right_dict, output_pth, img_focal, image_names, 
    #                         R_w2c=torch.eye(3).repeat(vis_end-vis_start,1,1),  
    #                         t_w2c=torch.zeros(vis_end-vis_start,3), valid_hand=valid_hand, v=None)
    # # breakpoint()

    if len(valid_hand) == 1:
        use_hand_idx = valid_hand[0]
        print(use_hand_idx)
    else:
        print(frame_chunks_all)
        print(valid_hand)
        use_hand_idx = 1
        # breakpoint()

    use_hand_data, use_hand_pred = hand_data[use_hand_idx]
    # print(use_hand_pred["joints"].shape)
    fingd_dict = compute_all_finger_distances(use_hand_pred["joints"])
    thumb_to_index_distance = fingd_dict["thumb_to_index_distance"]
    thumb_to_middle_distance = fingd_dict["thumb_to_middle_distance"]
    # open_distance = (thumb_to_index_distance + thumb_to_middle_distance) / 2.
    open_distance = thumb_to_index_distance * 100
    open_distance = open_distance.cpu().detach()
    # print(open_distance)
    
    root_rot_quat = angle_axis_to_quaternion(use_hand_data["init_root_orient"]).squeeze(0)
    root_trans = use_hand_data["init_trans"].squeeze(0)
    root_rot = use_hand_data["init_root_orient"].squeeze(0)
    hand_pose = use_hand_data["init_hand_pose"].squeeze(0)
    hand_betas = use_hand_data["init_betas"].squeeze(0)
    
    # 切换 4元数 到 RPY
    root_rot_rpy = quaternion_to_rpy(root_rot_quat)
    
    # print(angle_axis_to_quaternion(root_rot))
    # print(root_trans)
    # print(root_trans.shape, root_rot_quat.shape, open_distance.shape)
    # imgfiles
    
    # Find the first index where open_distance is not NaN
    k = 0
    for i in range(len(open_distance)):
        if not torch.isnan(open_distance[i]):
            k = i
            break
    print(f"First non-NaN index: {k}")
    
    # new_imgfiles 变成 new_video
    # xyzrpy
    new_imgfiles =  imgfiles[k:-1]
    state_frames_data = torch.cat([root_trans[k:-1], root_rot_rpy[k:-1], open_distance[k:-1]], dim=-1)
    action_frames_data = torch.cat([root_trans[k+1:], root_rot_rpy[k+1:], open_distance[k+1:]], dim=-1)
    mano_data = torch.cat([root_trans[k:-1], root_rot[k:-1], hand_pose[k:-1], hand_betas[k:-1]], dim=-1)
    print("root_trans", root_trans.shape)
    print("root_rot", root_rot.shape)
    print("hand_pose", hand_pose.shape)
    print("hand_betas", hand_betas.shape)
    print("mano_data", mano_data.shape)
    # breakpoint()
    # 线性插值 
    # new_imgfiles
    # state_frames_data
    # action_frames_data

    # breakpoint()

    def linear_interpolation(data, factor=2):
        """
        线性插值：在每两帧之间插入factor-1个新帧
        
        Args:
            data: 原始数据, shape [n, d]
            factor: 插值因子，表示每两帧之间插入几个新帧
            
        Returns:
            插值后的数据, shape [(n-1)*factor + 1, d]
        """
        n, d = data.shape
        interp_n = (n - 1) * factor + 1
        interp_data = torch.zeros((interp_n, d), dtype=data.dtype, device=data.device)
        
        # 设置原始帧
        for i in range(n):
            interp_data[i * factor] = data[i]
        
        # 在相邻帧之间进行线性插值
        for i in range(n - 1):
            for j in range(1, factor):
                alpha = j / factor
                interp_data[i * factor + j] = (1 - alpha) * data[i] + alpha * data[i + 1]
        
        return interp_data
    
    # 对数据进行线性插值
    interp_factor = 4  
    state_frames_interp = linear_interpolation(state_frames_data, interp_factor)
    action_frames_interp = linear_interpolation(action_frames_data, interp_factor)
    mano_frames_interp = linear_interpolation(mano_data, interp_factor)
    
    # 为每个插值帧分配对应的图像
    def interpolate_imgfiles(imgfiles, factor):
        """
        为每个插值后的帧创建对应的图像文件引用
        
        Args:
            imgfiles: 原始图像文件列表
            factor: 插值因子
            
        Returns:
            扩展后的图像文件列表，与插值后的帧一一对应
        """
        n = len(imgfiles)
        interp_n = (n - 1) * factor + 1
        interp_imgfiles = [None] * interp_n
        
        # 设置原始帧的图像
        for i in range(n):
            interp_imgfiles[i * factor] = imgfiles[i]
        
        # 为插值帧分配最近的原始图像
        for i in range(n - 1):
            for j in range(1, factor):
                # 使用前一帧的图像
                interp_imgfiles[i * factor + j] = imgfiles[i]
                
                # 可选：使用距离最近的原始图像
                # if j < factor / 2:
                #     interp_imgfiles[i * factor + j] = imgfiles[i]  # 使用前一帧图像
                # else:
                #     interp_imgfiles[i * factor + j] = imgfiles[i+1]  # 使用后一帧图像
        
        return interp_imgfiles
    
    # 创建或提取时间戳
    def extract_timestamps_from_imgfiles(imgfiles):
        """
        从图像文件名中提取时间戳，如果文件名中没有时间戳，则生成等间隔的时间戳
        
        Args:
            imgfiles: 图像文件列表
            
        Returns:
            时间戳列表，与图像文件一一对应
        """
        timestamps = []
        
        # 尝试从文件名中提取时间戳
        time_found = False
        for img_path in imgfiles:
            # 提取文件名（不含路径和扩展名）
            filename = os.path.basename(img_path).split('.')[0]
            
            # 尝试查找时间戳格式，例如：frame_123456789.jpg
            # 假设时间戳是文件名中的数字部分
            import re
            match = re.search(r'(\d+)', filename)
            if match:
                time_found = True
                timestamps.append(float(match.group(1)))
            else:
                time_found = False
                break
                
        # 如果无法从文件名提取时间戳，创建等间隔时间戳
        if not time_found:
            # 默认FPS为30，时间间隔为1/30秒
            fps = 30.0
            interval = 1.0 / fps
            timestamps = [i * interval for i in range(len(imgfiles))]
            
        return np.array(timestamps)
        
    def interpolate_timestamps(timestamps, factor):
        """
        对时间戳进行线性插值
        
        Args:
            timestamps: 原始时间戳
            factor: 插值因子
            
        Returns:
            插值后的时间戳
        """
        n = len(timestamps)
        interp_n = (n - 1) * factor + 1
        interp_timestamps = np.zeros(interp_n)
        
        # 设置原始帧的时间戳
        for i in range(n):
            interp_timestamps[i * factor] = timestamps[i]
        
        # 在相邻时间戳之间进行线性插值
        for i in range(n - 1):
            for j in range(1, factor):
                alpha = j / factor
                interp_timestamps[i * factor + j] = (1 - alpha) * timestamps[i] + alpha * timestamps[i + 1]
        
        return interp_timestamps
    
    # 提取原始时间戳
    original_timestamps = extract_timestamps_from_imgfiles(new_imgfiles)
    # 对时间戳进行插值
    timestamps_interp = interpolate_timestamps(original_timestamps, interp_factor)
    
    # 调整时间戳，使第一帧从0开始
    if len(timestamps_interp) > 0:
        start_time = timestamps_interp[0]
        timestamps_interp = timestamps_interp - start_time
        
        # 重新调整时间间隔为30FPS
        fps = 30.0
        interval = 1.0 / fps
        for i in range(1, len(timestamps_interp)):
            timestamps_interp[i] = i * interval
    
    # 对imgfiles进行对应的扩展
    imgfiles_interp = interpolate_imgfiles(new_imgfiles, interp_factor)


    # generate random color video
    # 获取右手顶点
    if enable_visualization:
        print("action_frames_interp.shape", action_frames_interp.shape)
        print("state_frames_interp.shape", state_frames_interp.shape)
        print("mano_frames_interp.shape", mano_frames_interp.shape)
        print("imgfiles_interp.shape", len(imgfiles_interp))
        # breakpoint()
        right_dict = None
        if 1 in valid_hand:
            hand = 'right'
            hand_idx = hand2idx[hand]
            pred_glob_r = run_mano(mano_frames_interp[:, :3].unsqueeze(0), 
                                mano_frames_interp[:, 3:6].unsqueeze(0), 
                                mano_frames_interp[:,6:51].unsqueeze(0), 
                                betas=mano_frames_interp[:, 51:].unsqueeze(0))
            right_verts = pred_glob_r['vertices'][0]
            right_dict = {
                    'vertices': right_verts.unsqueeze(0),
                    'faces': faces_right,
                }

        left_dict = None
        # 获取左手顶点
        if 0 in valid_hand:
            faces_left = faces_right[:,[0,2,1]]
            hand = 'left'
            hand_idx = hand2idx[hand]
            pred_glob_l = run_mano_left(mano_frames_interp[:, :3].unsqueeze(0), 
                                    mano_frames_interp[:, 3:6].unsqueeze(0), 
                                    mano_frames_interp[:, 6:51].unsqueeze(0), 
                                    betas=mano_frames_interp[:, 51:].unsqueeze(0))
            left_verts = pred_glob_l['vertices'][0]
            left_dict = {
                    'vertices': left_verts.unsqueeze(0),
                    'faces': faces_left,
                }

        # 如果需要可视化，才执行可视化相关代码
    
        output_pth = os.path.join(seq_folder, f"cam_vis_random_color")
        if not os.path.exists(output_pth):
            os.makedirs(output_pth)
        image_names = imgfiles_interp
        # print(f"visualizing camera space from {vis_start} to {vis_end}")

        print("valid_hand", valid_hand)
        # 在相机坐标系下可视化时，使用单位旋转矩阵和零平移向量
        run_vis2_on_video_cam_random_color(left_dict, right_dict, output_pth, img_focal, image_names, 
                            R_w2c=torch.eye(3).repeat(len(image_names),1,1),  
                            t_w2c=torch.zeros(len(image_names),3), valid_hand=valid_hand, v=None)
    
    # 使用插值后的数据
    state_frames_array = state_frames_interp.numpy()
    action_frames_array = action_frames_interp.numpy()
    mano_frames_array = mano_data.numpy()
    timestamps_array = timestamps_interp
    
    print("state with timestamps")
    for i in range(min(5, state_frames_array.shape[0])):
        print(f"Frame {i}: timestamp={timestamps_array[i]:.6f}, data={state_frames_array[i]}")

file_path = "/home/admin123/projects/lerobot-plus/lerobot-joycon_plus_data/datasets/pick_put_hand/images/observation.images.webcam/episode_000000"
process_hand_video(file_path, enable_visualization=False )