import os
import numpy as np
import shutil
from glob import glob
from rh20t_api.configurations import load_conf
from rh20t_api.scene import RH20TScene
from tqdm import tqdm
import json

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
    output_root = "/share/zsp/datasets/motiondb/motionlib-hoi/RH20T_robot/"
    robot_config_path = "configs/configs.json"
    
    # Create necessary directories if they don't exist
    state_feature_dir = os.path.join(output_root, "feature", "state_tcp_gripper", task_id)
    action_feature_dir = os.path.join(output_root, "feature", "action_tcp_gripper", task_id)
    images_dir = os.path.join(output_root, "images", "images", task_id)
    text_dir = os.path.join(output_root, "text", "task_description", task_id)
    # in_hand_text_dir = os.path.join(output_root, "text", "in_hand_serials", task_id)
    serials_text_dir = os.path.join(output_root, "text", "serials", task_id)
    
    os.makedirs(state_feature_dir, exist_ok=True)
    os.makedirs(action_feature_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)
    # os.makedirs(in_hand_text_dir, exist_ok=True)
    os.makedirs(serials_text_dir, exist_ok=True)
    
    # Load robot configurations
    robot_configs = load_conf(robot_config_path)
    
    # Find all task folders
    task_pattern = os.path.join(data_root, f"{task_id}*")
    task_folders = [f for f in glob(task_pattern) 
                   if os.path.isdir(f) and "human" not in f]
    
    print(f"Found {len(task_folders)} task folders")
    
    for task_folder in tqdm(task_folders):
        folder_name = os.path.basename(task_folder)
        print(f"Processing folder: {folder_name}")
        
        # try:
        # Load the scene
        scene = RH20TScene(task_folder, robot_configs)
        
        # Get start and end timestamps
        start_timestamp = scene.start_timestamp
        end_timestamp = scene.end_timestamp
        
        in_hand_serials = scene.in_hand_serials
        print("in_hand_serials", scene.in_hand_serials)
        serials = scene.serials

        serials_dict = {"in_hand_serials": in_hand_serials, "serials": serials}
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
            # breakpoint()
            
            
            # Process all frames and concatenate tcp and gripper data
            state_frames_data = []
            action_frames_data = []
            # processed_timestamps = []
            
            for actual_timestamp in image_timestamps:
                # try:
                # Get TCP data
                # 每个相机坐标系中的夹爪笛卡尔姿态，{序列号: [{"timestamp": ..., "tcp": ..., "robot_ft": ...}]}
                # 其中"tcp"值是xyz+四元数（7维）夹爪笛卡尔姿态
                tcp_data = scene.get_tcp_aligned(actual_timestamp, serial)
                # print(tcp_data)
                
                # Get gripper command data
                # 最大110
                gripper_info = scene.get_gripper_info(actual_timestamp)
                gripper_command = scene.get_gripper_command(actual_timestamp)
                # print(scene.get_gripper(actual_timestamp, command_or_info="command"), scene.get_gripper(actual_timestamp, command_or_info="info"))
                # print()
                # print(gripper_info, gripper_command)
                # breakpoint()
                
                # Ensure data are numpy arrays
                # if not isinstance(tcp_data, np.ndarray):
                #     tcp_data = np.array(tcp_data)
                # if not isinstance(gripper_command, np.ndarray):
                gripper_info = np.array([1. * gripper_info / 1000.])
                gripper_command = np.array([1. * gripper_command / 1000.])
                # gripper_info = np.array([1. * gripper_info / 110.])
                # gripper_command = np.array([1. * gripper_command / 110.])
                
                # Concatenate TCP and gripper data
                state_data = np.concatenate([tcp_data, gripper_info])
                action_data = np.concatenate([tcp_data, gripper_command])
                # print(state_data)
                # print(action_data)
                # print(f"{folder_name}_{serial}.npy")
                # breakpoint()
                
                
                # Append to frames data
                state_frames_data.append(state_data)
                action_frames_data.append(action_data)
                # processed_timestamps.append(actual_timestamp)

                # except Exception as e:
                #     print(f"    Error processing frame at timestamp {timestamp}: {e}")
                #     continue
            # print(state_frames_data)
            # breakpoint()
            
            # Convert list to numpy array and save
            if state_frames_data and action_frames_data:
                state_frames_array = np.array(state_frames_data)
                action_frames_array = np.array(action_frames_data)
                # print(state_frames_array.shape)
                # print(action_frames_array.shape)
                # breakpoint()
                state_output_file = os.path.join(state_feature_dir, f"{folder_name}_{serial}.npy")
                np.save(state_output_file, state_frames_array)
                print(f"    Saved state features to {state_output_file}, shape: {state_frames_array.shape}")

                action_output_file = os.path.join(action_feature_dir, f"{folder_name}_{serial}.npy")
                np.save(action_output_file, action_frames_array)
                print(f"    Saved action features to {action_output_file}, shape: {action_frames_array.shape}")
                
            # else:
            #     print(f"    No frames data collected for {folder_name}_{serial}")
            #     continue
            
            # Create task description text file
            text_file_path = os.path.join(text_dir, f"{folder_name}_{serial}.txt")
            with open(text_file_path, 'w') as f:
                f.write(text_content)
            print(f"    Created task description file at {text_file_path}")

            serials_file_path = os.path.join(serials_text_dir, f"{folder_name}_{serial}.txt")
            with open(serials_file_path, 'w') as f:
                f.write(json.dumps(serials_dict))
            print(f"    Created in serials file at {serials_file_path}")

            # breakpoint()
                
        # except Exception as e:
        #     print(f"Error processing folder {folder_name}: {e}")
        #     continue
    
    print("Processing complete")

# Run the function with the specified task ID
if __name__ == "__main__":
    
    # Use command line argument for task_id if provided, otherwise default to "task_0008"
    task_id = "task_0008"
    
    process_rh20t_data(task_id)