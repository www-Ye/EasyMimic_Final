import os
import numpy as np
import shutil
from glob import glob
from rh20t_api.configurations import load_conf
from rh20t_api.scene import RH20TScene
from tqdm import tqdm
import json
import torch

from aitviewer.headless import HeadlessRenderer
from hawor_api import detect_track_video, hawor_motion_estimation, hawor_infiller, hawor_slam, compute_all_finger_distances
from lib.eval_utils.custom_utils import world2cam_convert, cam2world_convert, load_slam_cam
from lib.vis.run_vis2 import run_vis2_on_video, run_vis2_on_video_cam, run_vis2_on_video_cam_3d
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from hawor.utils.rotation import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis, angle_axis_to_quaternion

def extract_image_paths_for_serial(image_paths, serial):
    """
    Extract image paths for a specific serial from the return value of get_image_path_pairs.
    Handles various possible return structures.
    
    Args:
        image_paths: The return value from scene.get_image_path_pairs()
        serial (str): The camera serial to filter for
        
    Returns:
        list: A list of image paths for the specified serial
    """
    paths_for_serial = []
    
    # Case 1: image_paths is a dictionary with serials as keys
    if isinstance(image_paths, dict) and serial in image_paths:
        value = image_paths[serial]
        if isinstance(value, list):
            paths_for_serial.extend(value)
        else:
            paths_for_serial.append(value)
        return paths_for_serial
    
    # Case 2: image_paths is a list of paths or tuples/lists of paths
    if isinstance(image_paths, list):
        for item in image_paths:
            if isinstance(item, (list, tuple)):
                # If item is a list/tuple, check each element
                for path in item:
                    if serial in str(path):
                        paths_for_serial.append(path)
            elif serial in str(item):
                paths_for_serial.append(item)
        return paths_for_serial
    
    # Case 3: image_paths is a dictionary with other keys
    if isinstance(image_paths, dict):
        for key, value in image_paths.items():
            if isinstance(value, list):
                for path in value:
                    if serial in str(path):
                        paths_for_serial.append(path)
            elif serial in str(value):
                paths_for_serial.append(value)
        return paths_for_serial
    
    # Case 4: image_paths is a single path string
    if isinstance(image_paths, str) and serial in image_paths:
        paths_for_serial.append(image_paths)
        
    return paths_for_serial

def process_rh20t_data(task_id="task_0008", text_content="Grab the block and place it at the designated location"):
    """
    Process RH20T data for a specified task ID.
    
    Args:
        task_id (str): Task ID to process (e.g., "task_0008")
    """
    # Configuration
    data_root = "/share/zsp/datasets/motion_raw/hoi/RH20T/data/RH20T_cfg1/"
    output_root = "/share/zsp/datasets/motiondb/motionlib-hoi/RH20T_human/"
    robot_config_path = "configs/configs.json"

    v = HeadlessRenderer()
    
    # Create necessary directories if they don't exist
    mano_feature_dir = os.path.join(output_root, "feature", "mano", task_id)
    state_feature_dir = os.path.join(output_root, "feature", "state_tcp_gripper", task_id)
    action_feature_dir = os.path.join(output_root, "feature", "action_tcp_gripper", task_id)
    images_dir = os.path.join(output_root, "images", "images", task_id)
    text_dir = os.path.join(output_root, "text", "task_description", task_id)
    serials_text_dir = os.path.join(output_root, "text", "serials", task_id)
    
    os.makedirs(mano_feature_dir, exist_ok=True)
    os.makedirs(state_feature_dir, exist_ok=True)
    os.makedirs(action_feature_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(serials_text_dir, exist_ok=True)
    
    # Load robot configurations
    robot_configs = load_conf(robot_config_path)
    
    # Find all task folders
    task_pattern = os.path.join(data_root, f"{task_id}*")
    task_folders = [f for f in glob(task_pattern) 
                   if os.path.isdir(f) and "human" in f]
    
    print(f"Found {len(task_folders)} task folders")
    
    for task_folder in tqdm(task_folders):
        folder_name = os.path.basename(task_folder)
        print(f"Processing folder: {folder_name}")
        
        # Load the scene
        scene = RH20TScene(task_folder, robot_configs)
        
        # Get start and end timestamps
        start_timestamp = scene.start_timestamp
        end_timestamp = scene.end_timestamp
        
        in_hand_serials = scene.in_hand_serials
        print("in_hand_serials", scene.in_hand_serials)
        serials = scene.serials

        serials_dict = {"in_hand_serials": in_hand_serials, "serials": serials}

        extrinsics = scene.extrinsics
        intrinsics = scene.intrinsics
        # print(scene.extrinsics)
        # print(scene.intrinsics)
        # breakpoint()
        # Process each camera serial
        for serial in scene.serials:
            print(f"  Processing serial: {serial}")

            # Create output directory for images
            output_image_dir = os.path.join(images_dir, f"{folder_name}_{serial}")
            os.makedirs(output_image_dir, exist_ok=True)
            
            # First, collect all timestamps that have corresponding images
            image_timestamps_set = set()
            image_path_map = {}  # Map from timestamp to image path
            
            print(f"    Scanning for images with serial {serial}...")
            for timestamp in range(start_timestamp, end_timestamp + 1):
                image_paths = scene.get_image_path_pairs(timestamp, image_types=["color"])
                serial_paths = extract_image_paths_for_serial(image_paths, serial)
                
                # If we found a valid image path that exists
                for path in serial_paths:
                    if os.path.exists(path):
                        # Extract actual timestamp from the filename
                        img_filename = os.path.basename(path)
                        # Assuming filename format like "1630848118404.jpg"
                        actual_timestamp = int(os.path.splitext(img_filename)[0])
                        
                        # Store using actual timestamp from filename
                        image_timestamps_set.add(actual_timestamp)
                        image_path_map[actual_timestamp] = path
                        break
            # Copy all found images to the output directory
            total_copied = 0
            for timestamp, path in image_path_map.items():
                img_filename = os.path.basename(path)
                output_img_path = os.path.join(output_image_dir, img_filename)
                if not os.path.exists(output_img_path):  # Avoid duplicates
                    shutil.copy(path, output_img_path)
                    total_copied += 1
            
            print(f"    Copied {total_copied} color images to {output_image_dir}")
            
            # Convert set to sorted list for ordered processing
            image_timestamps = sorted(list(image_timestamps_set))
            print(f"    Found {len(image_timestamps)} frames with images for serial {serial}")
            # print(image_timestamps)
            # print(len(image_timestamps))
            
            start_idx, end_idx, seq_folder, imgfiles = detect_track_video(output_image_dir)
            frame_chunks_all, img_focal = hawor_motion_estimation(output_image_dir, start_idx, end_idx, seq_folder)
            # print(valid_hand)
            # print(frame_chunks_all)
            valid_hand = list(frame_chunks_all.keys())
            # breakpoint()
            # print(extrinsics[serial])
            # slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
            # if not os.path.exists(slam_path):
            #     hawor_slam(output_image_dir, start_idx, end_idx, seq_folder)
            # slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
            pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(output_image_dir, start_idx, end_idx, frame_chunks_all, seq_folder, extrinsics[serial])

            # 世界坐标系到相机坐标系(w2c)
            R_w2c_sla = torch.from_numpy(extrinsics[serial][:3, :3]).to(torch.float32)  # 旋转矩阵(3x3)
            t_w2c_sla = torch.from_numpy(extrinsics[serial][:3, 3]).to(torch.float32)   # 平移向量(3x1)

            # 相机坐标系到世界坐标系(c2w)
            extrinsics_inverse = np.linalg.inv(extrinsics[serial])
            R_c2w_sla = torch.from_numpy(extrinsics_inverse[:3, :3]).to(torch.float32)  # 旋转矩阵(3x3)
            t_c2w_sla = torch.from_numpy(extrinsics_inverse[:3, 3]).to(torch.float32)   # 平移向量(3x1)

            # 添加批次维度
            batch_size = len(image_timestamps)  # 您需要的批次大小
            R_c2w_sla_all = R_c2w_sla.unsqueeze(0).expand(batch_size, 3, 3)  # 形状变为 [b,3,3]
            t_c2w_sla_all = t_c2w_sla.unsqueeze(0).expand(batch_size, 3)      # 形状变为 [b,3]

            R_w2c_sla_all = R_w2c_sla.unsqueeze(0).expand(batch_size, 3, 3)  # 形状变为 [b,3,3]
            t_w2c_sla_all = t_w2c_sla.unsqueeze(0).expand(batch_size, 3)      # 形状变为 [b,3]

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

            mano_output_file = os.path.join(mano_feature_dir, f"{folder_name}_{serial}.npy")
            np.save(mano_output_file, mano_frames_array)
            print(f"    Saved mano features to {mano_output_file}, shape: {mano_frames_array.shape}")

            state_output_file = os.path.join(state_feature_dir, f"{folder_name}_{serial}.npy")
            np.save(state_output_file, state_frames_array)
            print(f"    Saved state features to {state_output_file}, shape: {state_frames_array.shape}")

            action_output_file = os.path.join(action_feature_dir, f"{folder_name}_{serial}.npy")
            np.save(action_output_file, action_frames_array)
            print(f"    Saved action features to {action_output_file}, shape: {action_frames_array.shape}")
                
            
            # Create task description text file
            text_file_path = os.path.join(text_dir, f"{folder_name}_{serial}.txt")
            with open(text_file_path, 'w') as f:
                f.write(text_content)
            print(f"    Created task description file at {text_file_path}")

            serials_file_path = os.path.join(serials_text_dir, f"{folder_name}_{serial}.txt")
            with open(serials_file_path, 'w') as f:
                f.write(json.dumps(serials_dict))
            print(f"    Created in serials file at {serials_file_path}")
    
    print("Processing complete")

# Run the function with the specified task ID
if __name__ == "__main__":
    
    # Use command line argument for task_id if provided, otherwise default to "task_0008"
    task_id = "task_0008"
    
    process_rh20t_data(task_id)