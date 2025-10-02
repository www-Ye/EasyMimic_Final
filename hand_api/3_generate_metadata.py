import os
import glob
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

def compute_action_statistics(actions_data):
    """
    Calculate action statistics for a single episode
    
    Args:
        actions_data: numpy array containing action data
        
    Returns:
        dictionary containing statistics
    """
    if actions_data is None or len(actions_data) == 0:
        return None
    
    # Action data format: [trans_x, trans_y, trans_z, rot_r, rot_p, rot_y, open]
    # Calculate statistics for each dimension separately
    
    stats = {
        "num_frames": len(actions_data),
        "translation": {},
        "rotation": {},
        "gripper": {},
        "overall": {}
    }
    
    # Translation statistics (first 3 columns: x, y, z)
    if actions_data.shape[1] >= 3:
        trans_data = actions_data[:, :3]
        stats["translation"] = {
            "x": {
                "mean": float(np.mean(trans_data[:, 0])),
                "std": float(np.std(trans_data[:, 0])),
                "min": float(np.min(trans_data[:, 0])),
                "max": float(np.max(trans_data[:, 0])),
                "range": float(np.max(trans_data[:, 0]) - np.min(trans_data[:, 0]))
            },
            "y": {
                "mean": float(np.mean(trans_data[:, 1])),
                "std": float(np.std(trans_data[:, 1])),
                "min": float(np.min(trans_data[:, 1])),
                "max": float(np.max(trans_data[:, 1])),
                "range": float(np.max(trans_data[:, 1]) - np.min(trans_data[:, 1]))
            },
            "z": {
                "mean": float(np.mean(trans_data[:, 2])),
                "std": float(np.std(trans_data[:, 2])),
                "min": float(np.min(trans_data[:, 2])),
                "max": float(np.max(trans_data[:, 2])),
                "range": float(np.max(trans_data[:, 2]) - np.min(trans_data[:, 2]))
            }
        }
        
        # Calculate translation magnitude
        trans_magnitudes = np.linalg.norm(trans_data, axis=1)
        stats["translation"]["magnitude"] = {
            "mean": float(np.mean(trans_magnitudes)),
            "std": float(np.std(trans_magnitudes)),
            "min": float(np.min(trans_magnitudes)),
            "max": float(np.max(trans_magnitudes)),
            "range": float(np.max(trans_magnitudes) - np.min(trans_magnitudes))
        }
    
    # Rotation statistics (middle 3 columns: roll, pitch, yaw)
    if actions_data.shape[1] >= 6:
        rot_data = actions_data[:, 3:6]
        stats["rotation"] = {
            "roll": {
                "mean": float(np.mean(rot_data[:, 0])),
                "std": float(np.std(rot_data[:, 0])),
                "min": float(np.min(rot_data[:, 0])),
                "max": float(np.max(rot_data[:, 0])),
                "range": float(np.max(rot_data[:, 0]) - np.min(rot_data[:, 0]))
            },
            "pitch": {
                "mean": float(np.mean(rot_data[:, 1])),
                "std": float(np.std(rot_data[:, 1])),
                "min": float(np.min(rot_data[:, 1])),
                "max": float(np.max(rot_data[:, 1])),
                "range": float(np.max(rot_data[:, 1]) - np.min(rot_data[:, 1]))
            },
            "yaw": {
                "mean": float(np.mean(rot_data[:, 2])),
                "std": float(np.std(rot_data[:, 2])),
                "min": float(np.min(rot_data[:, 2])),
                "max": float(np.max(rot_data[:, 2])),
                "range": float(np.max(rot_data[:, 2]) - np.min(rot_data[:, 2]))
            }
        }
        
        # Calculate rotation magnitude
        rot_magnitudes = np.linalg.norm(rot_data, axis=1)
        stats["rotation"]["magnitude"] = {
            "mean": float(np.mean(rot_magnitudes)),
            "std": float(np.std(rot_magnitudes)),
            "min": float(np.min(rot_magnitudes)),
            "max": float(np.max(rot_magnitudes)),
            "range": float(np.max(rot_magnitudes) - np.min(rot_magnitudes))
        }
    
    # Gripper statistics (last column: open/close)
    if actions_data.shape[1] >= 7:
        gripper_data = actions_data[:, 6]
        stats["gripper"] = {
            "mean": float(np.mean(gripper_data)),
            "std": float(np.std(gripper_data)),
            "min": float(np.min(gripper_data)),
            "max": float(np.max(gripper_data)),
            "range": float(np.max(gripper_data) - np.min(gripper_data)),
            "open_frames": int(np.sum(gripper_data > 50)),  # Assume >50 is open state
            "close_frames": int(np.sum(gripper_data <= 50))  # Assume <=50 is closed state
        }
    
    # Overall action statistics
    if actions_data.shape[1] >= 7:
        # Calculate action change rate (difference between consecutive frames)
        action_diffs = np.diff(actions_data, axis=0)
        action_change_magnitudes = np.linalg.norm(action_diffs, axis=1)
        
        stats["overall"] = {
            "total_action_change": float(np.sum(action_change_magnitudes)),
            "mean_action_change": float(np.mean(action_change_magnitudes)),
            "max_action_change": float(np.max(action_change_magnitudes)),
            "action_smoothness": float(1.0 / (1.0 + np.mean(action_change_magnitudes)))  # Smoothness metric
        }
    
    return stats

def compute_dataset_summary_statistics(all_episode_stats):
    """
    Calculate summary statistics for the entire dataset
    
    Args:
        all_episode_stats: Dictionary of statistics for all episodes
        
    Returns:
        A dictionary containing the overall statistical summary of the dataset
    """
    if not all_episode_stats:
        return {}
    
    # Collect statistics from all episodes
    all_trans_x_means = []
    all_trans_y_means = []
    all_trans_z_means = []
    all_rot_r_means = []
    all_rot_p_means = []
    all_rot_y_means = []
    all_gripper_means = []
    all_frame_counts = []
    all_action_smoothness = []
    
    for episode_name, stats in all_episode_stats.items():
        if "translation" in stats and "x" in stats["translation"]:
            all_trans_x_means.append(stats["translation"]["x"]["mean"])
            all_trans_y_means.append(stats["translation"]["y"]["mean"])
            all_trans_z_means.append(stats["translation"]["z"]["mean"])
        
        if "rotation" in stats and "roll" in stats["rotation"]:
            all_rot_r_means.append(stats["rotation"]["roll"]["mean"])
            all_rot_p_means.append(stats["rotation"]["pitch"]["mean"])
            all_rot_y_means.append(stats["rotation"]["yaw"]["mean"])
        
        if "gripper" in stats:
            all_gripper_means.append(stats["gripper"]["mean"])
        
        if "num_frames" in stats:
            all_frame_counts.append(stats["num_frames"])
        
        if "overall" in stats and "action_smoothness" in stats["overall"]:
            all_action_smoothness.append(stats["overall"]["action_smoothness"])
    
    # Calculate overall summary statistics
    summary = {
        "total_episodes": len(all_episode_stats),
        "frame_statistics": {
            "total_frames": sum(all_frame_counts) if all_frame_counts else 0,
            "mean_frames_per_episode": float(np.mean(all_frame_counts)) if all_frame_counts else 0,
            "std_frames_per_episode": float(np.std(all_frame_counts)) if all_frame_counts else 0,
            "min_frames": min(all_frame_counts) if all_frame_counts else 0,
            "max_frames": max(all_frame_counts) if all_frame_counts else 0
        }
    }
    
    # Translation summary statistics
    if all_trans_x_means:
        summary["translation_summary"] = {
            "x": {
                "mean": float(np.mean(all_trans_x_means)),
                "std": float(np.std(all_trans_x_means)),
                "min": float(np.min(all_trans_x_means)),
                "max": float(np.max(all_trans_x_means))
            },
            "y": {
                "mean": float(np.mean(all_trans_y_means)),
                "std": float(np.std(all_trans_y_means)),
                "min": float(np.min(all_trans_y_means)),
                "max": float(np.max(all_trans_y_means))
            },
            "z": {
                "mean": float(np.mean(all_trans_z_means)),
                "std": float(np.std(all_trans_z_means)),
                "min": float(np.min(all_trans_z_means)),
                "max": float(np.max(all_trans_z_means))
            }
        }
    
    # Rotation summary statistics
    if all_rot_r_means:
        summary["rotation_summary"] = {
            "roll": {
                "mean": float(np.mean(all_rot_r_means)),
                "std": float(np.std(all_rot_r_means)),
                "min": float(np.min(all_rot_r_means)),
                "max": float(np.max(all_rot_r_means))
            },
            "pitch": {
                "mean": float(np.mean(all_rot_p_means)),
                "std": float(np.std(all_rot_p_means)),
                "min": float(np.min(all_rot_p_means)),
                "max": float(np.max(all_rot_p_means))
            },
            "yaw": {
                "mean": float(np.mean(all_rot_y_means)),
                "std": float(np.std(all_rot_y_means)),
                "min": float(np.min(all_rot_y_means)),
                "max": float(np.max(all_rot_y_means))
            }
        }
    
    # Gripper summary statistics
    if all_gripper_means:
        summary["gripper_summary"] = {
            "mean": float(np.mean(all_gripper_means)),
            "std": float(np.std(all_gripper_means)),
            "min": float(np.min(all_gripper_means)),
            "max": float(np.max(all_gripper_means))
        }
    
    # Action smoothness summary
    if all_action_smoothness:
        summary["action_smoothness_summary"] = {
            "mean": float(np.mean(all_action_smoothness)),
            "std": float(np.std(all_action_smoothness)),
            "min": float(np.min(all_action_smoothness)),
            "max": float(np.max(all_action_smoothness))
        }
    
    return summary

# --- CLI Arguments ---
parser = argparse.ArgumentParser(description="Generate LeRobot v2 metadata (episodes/info/stats/action_statistics/modality)")
parser.add_argument(
    "--dataset-path",
    type=str,
    required=True,
    help="Root directory of the dataset (folder containing data/, meta/, videos/, etc. subdirectories)",
)
parser.add_argument(
    "--chunk-name",
    type=str,
    default="chunk-000",
    help="Data chunk name (default: chunk-000)",
)
args = parser.parse_args()

# --- Path construction ---
dataset_path = args.dataset_path
chunk_name = args.chunk_name
data_path = os.path.join(dataset_path, "data", chunk_name)
meta_path = os.path.join(dataset_path, "meta")

# --- Check if data path exists ---
if not os.path.isdir(data_path):
    print(f"Error: Data path does not exist: {data_path}")
    exit()

# --- Create meta directory ---
os.makedirs(meta_path, exist_ok=True)
print(f"Metadata directory '{meta_path}' ensured to exist.")

# --- Find Parquet files ---
# Use glob to find all episode files and sort them to ensure episode_index is processed in order
parquet_files = sorted(glob.glob(os.path.join(data_path, "episode_*.parquet")))

if not parquet_files:
    print(f"Error: No episode_*.parquet files found in {data_path}.")
    exit()

print(f"Found {len(parquet_files)} episode Parquet files.")

# --- Initialize variables ---
total_episodes = 0
total_frames = 0
episodes_data = [] # for episodes.jsonl
tasks_data = []    # for tasks.jsonl
action_stats = {"min": None, "max": None, "mean": None, "std": None, "dim": None, "q01": None, "q99": None}
state_stats = {"min": None, "max": None, "mean": None, "std": None, "dim": None, "q99": None}
first_action_shape = None # to get dimension
first_state_shape = None # to get state dimension

# To collect all data for statistical calculation
all_actions = []
all_states = []

# To collect action statistics for all episodes
all_episode_stats = {}

# --- Process each Parquet file ---
print("Processing Parquet files and generating metadata...")
for file_path in tqdm(parquet_files, desc="Processing Episodes"):
    try:
        # Extract episode_index
        filename = os.path.basename(file_path)
        episode_index_str = filename.replace("episode_", "").replace(".parquet", "")
        episode_index = int(episode_index_str)
        print("file_path", file_path)
        # Read Parquet file
        df = pd.read_parquet(file_path)
        print("df", df.head())
        print("df.columns", df.columns)
        print("df['action'].shape", df['action'].shape)

        # Get trajectory length
        length = len(df)
        print("length", length)

        # Accumulate statistics
        total_episodes += 1
        total_frames += length

        # --- Prepare episodes.jsonl data ---
        # Assume task name is fixed to "pick_put"
        episode_info = {
            "episode_index": episode_index,
            "tasks": ["[Robot] pull the drawer and then push the drawer."], # According to your example
            "length": length
        }
        episodes_data.append(episode_info)

        # --- Prepare tasks.jsonl data ---
        # Assume task name is "pick_put", ID is 0
        task_info = {
             "episode_index": episode_index,
             "task_name": "pick_put",
             "task_id": 0 # Assume there is only one task, ID is 0
        }
        tasks_data.append(task_info)

        # --- Collect action and state data for stats.json ---
        # Process action data
        if 'action' in df.columns and len(df) > 0:
            try:
                # Get action data
                episode_actions = []
                for idx, action_data in enumerate(df['action']):
                    if isinstance(action_data, (list, np.ndarray)):
                        episode_actions.append(np.array(action_data, dtype=np.float32))
                    elif isinstance(action_data, str):
                        try:
                            parsed_action = json.loads(action_data.replace("'", "\""))
                            if isinstance(parsed_action, list):
                                episode_actions.append(np.array(parsed_action, dtype=np.float32))
                        except json.JSONDecodeError:
                            print(f"Warning: Could not parse action string for frame {idx} in episode {episode_index}.")
                            continue

                if episode_actions:
                    episode_actions = np.array(episode_actions)
                    all_actions.append(episode_actions)
                    
                    # Get dimension info from the first action
                    if first_action_shape is None:
                        first_action_shape = episode_actions[0].shape
                        print(f"Got action dimension from episode {episode_index}: {first_action_shape}")
                    
                    # Calculate action statistics for this episode
                    episode_stats = compute_action_statistics(episode_actions)
                    if episode_stats:
                        episode_name = f"episode_{episode_index:06d}"
                        all_episode_stats[episode_name] = episode_stats
                        print(f"  📊 Calculated action statistics for {episode_name}")

            except Exception as e:
                print(f"Warning: Error processing action data for episode {episode_index}: {e}")

        # Process observation.state data
        if 'observation.state' in df.columns and len(df) > 0:
            try:
                # Get state data
                episode_states = []
                for idx, state_data in enumerate(df['observation.state']):
                    if isinstance(state_data, (list, np.ndarray)):
                        episode_states.append(np.array(state_data, dtype=np.float32))
                    elif isinstance(state_data, str):
                        try:
                            parsed_state = json.loads(state_data.replace("'", "\""))
                            if isinstance(parsed_state, list):
                                episode_states.append(np.array(parsed_state, dtype=np.float32))
                        except json.JSONDecodeError:
                            print(f"Warning: Could not parse state string for frame {idx} in episode {episode_index}.")
                            continue
                
                if episode_states:
                    episode_states = np.array(episode_states)
                    all_states.append(episode_states)
                    
                    # Get dimension info from the first state
                    if first_state_shape is None:
                        first_state_shape = episode_states[0].shape
                        print(f"Got state dimension from episode {episode_index}: {first_state_shape}")
                        
            except Exception as e:
                print(f"Warning: Error processing state data for episode {episode_index}: {e}")


    except Exception as e:
        print(f"\nError processing file {file_path}: {e}")
        continue # Skip problematic files

# --- Write episodes.jsonl ---
episodes_jsonl_path = os.path.join(meta_path, "episodes.jsonl")
try:
    with open(episodes_jsonl_path, 'w', encoding='utf-8') as f:
        for item in episodes_data:
            # Ensure writing a compact JSON string and add a newline
            json.dump(item, f, separators=(',', ':'))
            f.write('\n')
    print(f"Successfully wrote: {episodes_jsonl_path}")
except Exception as e:
    print(f"Error writing to {episodes_jsonl_path}: {e}")

# --- Write info.json ---
info_data = {
    "codebase_version": "v2.0",
    "robot_type": "so100",
    "total_episodes": total_episodes,
    "total_frames": total_frames,
    "total_tasks": 1,
    "total_videos": 2*total_episodes,
    "total_chunks": 1,
    "chunks_size": 1000,
    "fps": 30,
    "splits": {
        "train": f"0:{total_episodes}"
    },
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "features": {
        "action": {
            "dtype": "float32",
            "shape": [7],
            "names": [
                "right_shoulder_rotation_joint",
                "right_shoulder_pitch_joint",
                "right_ellbow_joint",
                "right_wrist_pitch_joint",
                "right_wrist_jaw_joint",
                "right_wrist_roll_joint",
                "right_gripper_joint"
            ]
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [7],
            "names": [
                "right_shoulder_rotation_joint",
                "right_shoulder_pitch_joint",
                "right_ellbow_joint",
                "right_wrist_pitch_joint",
                "right_wrist_jaw_joint",
                "right_wrist_roll_joint",
                "right_gripper_joint"
            ]
        },
        "observation.images.webcam": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": 30.0,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        },
        "observation.images.phone": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": 30.0,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [1],
            "names": None
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [1],
            "names": None
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [1],
            "names": None
        },
        "index": {
            "dtype": "int64",
            "shape": [1],
            "names": None
        },
        "task_index": {
            "dtype": "int64",
            "shape": [1],
            "names": None
        }
    }
}
print(info_data)
info_json_path = os.path.join(meta_path, "info.json")
try:
    with open(info_json_path, 'w', encoding='utf-8') as f:
        json.dump(info_data, f, indent=4) # Use indent to make it more readable
    print(f"Successfully wrote: {info_json_path}")
    # Print calculated statistics
    print("\n--- Calculated Statistics ---")
    print(f"Total Episodes (total_episodes): {info_data['total_episodes']}")
    print(f"Total Frames (total_frames): {info_data['total_frames']}")
    print(f"Total Videos (total_videos): {info_data['total_videos']}")
except Exception as e:
    print(f"Error writing to {info_json_path}: {e}")


# --- Calculate statistics and write to stats.json ---
print("\nCalculating action and state statistics...")

def compute_stats(arr, shape):
    stats = {}
    stats["mean"] = np.mean(arr, axis=0).tolist()
    stats["std"] = np.std(arr, axis=0).tolist()
    stats["min"] = np.min(arr, axis=0).tolist()
    stats["max"] = np.max(arr, axis=0).tolist()
    stats["q01"] = np.percentile(arr, 1, axis=0).tolist()
    stats["q99"] = np.percentile(arr, 99, axis=0).tolist()
    stats["dim"] = list(shape) if shape else None
    return stats

# Calculate action statistics
if all_actions:
    try:
        # Concatenate action data from all episodes
        all_actions_concat = np.concatenate(all_actions, axis=0)  # [total_frames, action_dim]
        print(f"Total frames of Action data: {all_actions_concat.shape[0]}, Dimension: {all_actions_concat.shape[1]}")
        action_stats = compute_stats(all_actions_concat, first_action_shape)
        print(f"Action statistics calculation complete, dimension: {action_stats['dim']}")
    except Exception as e:
        print(f"Error calculating action statistics: {e}")
        action_stats = {"min": None, "max": None, "mean": None, "std": None, "dim": None, "q01": None, "q99": None}
else:
    print("Warning: No action data found")

# Calculate state statistics
if all_states:
    try:
        all_states_concat = np.concatenate(all_states, axis=0)
        print(f"Total frames of State data: {all_states_concat.shape[0]}, Dimension: {all_states_concat.shape[1]}")
        state_stats = compute_stats(all_states_concat, first_state_shape)
        print(f"State statistics calculation complete, dimension: {state_stats['dim']}")
    except Exception as e:
        print(f"Error calculating state statistics: {e}")
        state_stats = {"min": None, "max": None, "mean": None, "std": None, "dim": None, "q01": None, "q99": None}
else:
    print("Warning: No state data found")

# Build stats data structure
stats_data = {
    "action": {k: v for k, v in action_stats.items() if k != "dim"},
    "observation.state": {k: v for k, v in state_stats.items() if k != "dim"}
}

stats_json_path = os.path.join(meta_path, "stats.json")
try:
    with open(stats_json_path, 'w', encoding='utf-8') as f:
        def numpy_encoder(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        json.dump(stats_data, f, indent=4, default=numpy_encoder)
    print(f"Successfully wrote: {stats_json_path}")

    # Print statistics summary
    print("\n--- Action Statistics Summary ---")
    if action_stats["dim"]:
        print(f"Dimension: {action_stats['dim']}")
        print(f"Mean range: {min(action_stats['mean']):.6f} ~ {max(action_stats['mean']):.6f}")
        print(f"Standard deviation range: {min(action_stats['std']):.6f} ~ {max(action_stats['std']):.6f}")
        print(f"1% quantile range: {min(action_stats['q01']):.6f} ~ {max(action_stats['q01']):.6f}")
        print(f"99% quantile range: {min(action_stats['q99']):.6f} ~ {max(action_stats['q99']):.6f}")
    else:
        print("No Action statistics data")

    print("\n--- State Statistics Summary ---")
    if state_stats["dim"]:
        print(f"Dimension: {state_stats['dim']}")
        print(f"Mean range: {min(state_stats['mean']):.6f} ~ {max(state_stats['mean']):.6f}")
        print(f"Standard deviation range: {min(state_stats['std']):.6f} ~ {max(state_stats['std']):.6f}")
        print(f"1% quantile range: {min(state_stats['q01']):.6f} ~ {max(state_stats['q01']):.6f}")
        print(f"99% quantile range: {min(state_stats['q99']):.6f} ~ {max(state_stats['q99']):.6f}")
    else:
        print("No State statistics data")

except Exception as e:
    print(f"Error writing to {stats_json_path}: {e}")

# --- Generate action statistics JSON file ---
if all_episode_stats:
    print("\nGenerating action statistics file...")
    
    # Calculate overall dataset summary statistics
    dataset_summary_stats = compute_dataset_summary_statistics(all_episode_stats)
    
    # Build action statistics data structure
    action_statistics_data = {
        "dataset_info": {
            "dataset_path": dataset_path,
            "dataset_name": os.path.basename(os.path.normpath(dataset_path)),
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "processing_timestamp": pd.Timestamp.now().isoformat()
        },
        "dataset_summary": dataset_summary_stats,
        "episode_statistics": all_episode_stats
    }
    
    # Save action statistics to JSON file
    action_stats_path = os.path.join(meta_path, "action_statistics.json")
    try:
        with open(action_stats_path, 'w', encoding='utf-8') as f:
            json.dump(action_statistics_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Action statistics saved to: {action_stats_path}")
        
        # Print statistics summary
        print(f"\n--- Action Statistics Summary ---")
        print(f"Total episodes: {dataset_summary_stats.get('total_episodes', 0)}")
        print(f"Total frames: {dataset_summary_stats.get('frame_statistics', {}).get('total_frames', 0)}")
        print(f"Mean frames/episode: {dataset_summary_stats.get('frame_statistics', {}).get('mean_frames_per_episode', 0):.1f}")
        
        if "gripper_summary" in dataset_summary_stats:
            gripper_summary = dataset_summary_stats["gripper_summary"]
            print(f"Gripper mean value range: [{gripper_summary['min']:.1f}, {gripper_summary['max']:.1f}]")
        
        if "action_smoothness_summary" in dataset_summary_stats:
            smoothness_summary = dataset_summary_stats["action_smoothness_summary"]
            print(f"Action smoothness range: [{smoothness_summary['min']:.4f}, {smoothness_summary['max']:.4f}]")
            
    except Exception as e:
        print(f"Warning: Error saving action statistics to {action_stats_path}: {e}")
else:
    print("\nWarning: No action data found, cannot generate action statistics")

print("\nMetadata file generation complete!")

# Generate modality.json file
modality_json_path = os.path.join(meta_path, "modality.json")
modality_json_data = {
    "state": {
        "single_arm_eef_xyz": {
            "start": 0,
            "end": 3
        },
        "single_arm_eef_rpy": {
            "start": 3,
            "end": 6
        },
        "gripper": {
            "start": 6,
            "end": 7
        }
    },
    "action": {
        "single_arm_eef_xyz": {
            "start": 0,
            "end": 3
        },
        "single_arm_eef_rpy": {
            "start": 3,
            "end": 6
        },
        "gripper": {
            "start": 6,
            "end": 7
        }
    },
    "video": {
        "webcam": {
            "original_key": "observation.images.webcam"
        },
        "phone": {
            "original_key": "observation.images.phone"
        }
    },
    "annotation": {
        "human.task_description": {
            "original_key": "task_index"
        }
    }
}

try:
    with open(modality_json_path, 'w', encoding='utf-8') as f:
        json.dump(modality_json_data, f, indent=4, ensure_ascii=False)
    print(f"✓ modality.json saved to: {modality_json_path}")
except Exception as e:
    print(f"Warning: Error saving modality.json to {modality_json_path}: {e}")
