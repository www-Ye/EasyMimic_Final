import os
import numpy as np
import shutil
from glob import glob
from tqdm import tqdm
import json
import torch

from aitviewer.headless import HeadlessRenderer
from hawor_api import detect_track_video, hawor_motion_estimation, hawor_infiller, hawor_slam, compute_all_finger_distances
from lib.eval_utils.custom_utils import world2cam_convert, cam2world_convert, load_slam_cam
from lib.vis.run_vis2 import run_vis2_on_video, run_vis2_on_video_cam, run_vis2_on_video_cam_3d
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from hawor.utils.rotation import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis, angle_axis_to_quaternion

def process_hand_video(output_image_dir):

    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(output_image_dir)
    frame_chunks_all, img_focal = hawor_motion_estimation(output_image_dir, start_idx, end_idx, seq_folder)

    valid_hand = list(frame_chunks_all.keys())

    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    if not os.path.exists(slam_path):
        hawor_slam(output_image_dir, start_idx, end_idx, seq_folder)
    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(output_image_dir, start_idx, end_idx, frame_chunks_all, seq_folder)

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

    output_pth = os.path.join(seq_folder, f"cam_vis_{vis_start}_{vis_end}")
    if not os.path.exists(output_pth):
        os.makedirs(output_pth)
    image_names = imgfiles[vis_start:vis_end]
    print(f"visualizing camera space from {vis_start} to {vis_end}")

    print("valid_hand", valid_hand)
    # 在相机坐标系下可视化时，使用单位旋转矩阵和零平移向量
    # run_vis2_on_video_cam(left_dict, right_dict, output_pth, img_focal, image_names, 
    #                     R_w2c=torch.eye(3).repeat(vis_end-vis_start,1,1),  
    #                     t_w2c=torch.zeros(vis_end-vis_start,3), valid_hand=valid_hand, v=None)
    # breakpoint()

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
    open_distance = (thumb_to_index_distance + thumb_to_middle_distance) / 2.
    open_distance = open_distance.cpu().detach()
    # print(open_distance)

    root_rot_quat = angle_axis_to_quaternion(use_hand_data["init_root_orient"]).squeeze(0)
    root_trans = use_hand_data["init_trans"].squeeze(0)
    root_rot = use_hand_data["init_root_orient"].squeeze(0)
    hand_pose = use_hand_data["init_hand_pose"].squeeze(0)
    hand_betas = use_hand_data["init_betas"].squeeze(0)

    # print(angle_axis_to_quaternion(root_rot))
    # print(root_trans)
    # print(root_trans.shape, root_rot_quat.shape, open_distance.shape)
    state_frames_data = torch.cat([root_trans, root_rot_quat, open_distance], dim=-1)
    action_frames_data = torch.cat([root_trans, root_rot_quat, open_distance], dim=-1)
    mano_data = torch.cat([root_trans, root_rot, hand_pose, hand_betas], dim=-1)
    print(mano_data.shape)
    # print(state_frames_data.shape)

    # breakpoint()

    # Convert list to numpy array and save
    # if state_frames_data and action_frames_data:
    state_frames_array = state_frames_data.numpy()
    action_frames_array = action_frames_data.numpy()
    mano_frames_array = mano_data.numpy()

process_hand_video("/home/admin123/projects/lerobot-plus/lerobot-joycon_plus_data/datasets/pick_put_hand/images/observation.images.webcam/episode_000000")