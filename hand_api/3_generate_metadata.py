import os
import glob
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm  # 用于显示进度条

def compute_action_statistics(actions_data):
    """
    计算单个episode的动作统计值
    
    Args:
        actions_data: 包含动作数据的numpy数组
        
    Returns:
        包含统计值的字典
    """
    if actions_data is None or len(actions_data) == 0:
        return None
    
    # 动作数据格式: [trans_x, trans_y, trans_z, rot_r, rot_p, rot_y, open]
    # 分别计算每个维度的统计值
    
    stats = {
        "num_frames": len(actions_data),
        "translation": {},
        "rotation": {},
        "gripper": {},
        "overall": {}
    }
    
    # 平移统计 (前3列: x, y, z)
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
        
        # 计算平移幅度
        trans_magnitudes = np.linalg.norm(trans_data, axis=1)
        stats["translation"]["magnitude"] = {
            "mean": float(np.mean(trans_magnitudes)),
            "std": float(np.std(trans_magnitudes)),
            "min": float(np.min(trans_magnitudes)),
            "max": float(np.max(trans_magnitudes)),
            "range": float(np.max(trans_magnitudes) - np.min(trans_magnitudes))
        }
    
    # 旋转统计 (中间3列: roll, pitch, yaw)
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
        
        # 计算旋转幅度
        rot_magnitudes = np.linalg.norm(rot_data, axis=1)
        stats["rotation"]["magnitude"] = {
            "mean": float(np.mean(rot_magnitudes)),
            "std": float(np.std(rot_magnitudes)),
            "min": float(np.min(rot_magnitudes)),
            "max": float(np.max(rot_magnitudes)),
            "range": float(np.max(rot_magnitudes) - np.min(rot_magnitudes))
        }
    
    # 夹爪统计 (最后一列: open/close)
    if actions_data.shape[1] >= 7:
        gripper_data = actions_data[:, 6]
        stats["gripper"] = {
            "mean": float(np.mean(gripper_data)),
            "std": float(np.std(gripper_data)),
            "min": float(np.min(gripper_data)),
            "max": float(np.max(gripper_data)),
            "range": float(np.max(gripper_data) - np.min(gripper_data)),
            "open_frames": int(np.sum(gripper_data > 50)),  # 假设>50为打开状态
            "close_frames": int(np.sum(gripper_data <= 50))  # 假设<=50为关闭状态
        }
    
    # 整体动作统计
    if actions_data.shape[1] >= 7:
        # 计算动作变化率 (相邻帧之间的差异)
        action_diffs = np.diff(actions_data, axis=0)
        action_change_magnitudes = np.linalg.norm(action_diffs, axis=1)
        
        stats["overall"] = {
            "total_action_change": float(np.sum(action_change_magnitudes)),
            "mean_action_change": float(np.mean(action_change_magnitudes)),
            "max_action_change": float(np.max(action_change_magnitudes)),
            "action_smoothness": float(1.0 / (1.0 + np.mean(action_change_magnitudes)))  # 平滑度指标
        }
    
    return stats

def compute_dataset_summary_statistics(all_episode_stats):
    """
    计算整个数据集的统计摘要
    
    Args:
        all_episode_stats: 所有episode的统计值字典
        
    Returns:
        包含数据集整体统计摘要的字典
    """
    if not all_episode_stats:
        return {}
    
    # 收集所有episode的统计值
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
    
    # 计算整体统计摘要
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
    
    # 平移统计摘要
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
    
    # 旋转统计摘要
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
    
    # 夹爪统计摘要
    if all_gripper_means:
        summary["gripper_summary"] = {
            "mean": float(np.mean(all_gripper_means)),
            "std": float(np.std(all_gripper_means)),
            "min": float(np.min(all_gripper_means)),
            "max": float(np.max(all_gripper_means))
        }
    
    # 动作平滑度摘要
    if all_action_smoothness:
        summary["action_smoothness_summary"] = {
            "mean": float(np.mean(all_action_smoothness)),
            "std": float(np.std(all_action_smoothness)),
            "min": float(np.min(all_action_smoothness)),
            "max": float(np.max(all_action_smoothness))
        }
    
    return summary

# --- CLI 参数 ---
parser = argparse.ArgumentParser(description="Generate LeRobot v2 metadata (episodes/info/stats/action_statistics/modality)")
parser.add_argument(
    "--dataset-path",
    type=str,
    required=True,
    help="数据集根目录（包含 data/ meta/ videos/ 等子目录的文件夹）",
)
parser.add_argument(
    "--chunk-name",
    type=str,
    default="chunk-000",
    help="数据 chunk 名称（默认: chunk-000）",
)
args = parser.parse_args()

# --- 路径构建 ---
dataset_path = args.dataset_path
chunk_name = args.chunk_name
data_path = os.path.join(dataset_path, "data", chunk_name)
meta_path = os.path.join(dataset_path, "meta")

# --- 检查数据路径是否存在 ---
if not os.path.isdir(data_path):
    print(f"错误：数据路径不存在: {data_path}")
    exit()

# --- 创建 meta 目录 ---
os.makedirs(meta_path, exist_ok=True)
print(f"元数据目录 '{meta_path}' 已确保存在。")

# --- 查找 Parquet 文件 ---
# 使用 glob 查找所有 episode 文件并排序，确保 episode_index 按顺序处理
parquet_files = sorted(glob.glob(os.path.join(data_path, "episode_*.parquet")))

if not parquet_files:
    print(f"错误：在 {data_path} 中没有找到 episode_*.parquet 文件。")
    exit()

print(f"找到 {len(parquet_files)} 个 episode Parquet 文件。")

# --- 初始化变量 ---
total_episodes = 0
total_frames = 0
episodes_data = [] # 用于 episodes.jsonl
tasks_data = []    # 用于 tasks.jsonl
action_stats = {"min": None, "max": None, "mean": None, "std": None, "dim": None, "q01": None, "q99": None}
state_stats = {"min": None, "max": None, "mean": None, "std": None, "dim": None, "q99": None}
first_action_shape = None # 用于获取维度
first_state_shape = None # 用于获取state维度

# 用于收集所有数据进行统计计算
all_actions = []
all_states = []

# 用于收集所有episode的动作统计值
all_episode_stats = {}

# --- 处理每个 Parquet 文件 ---
print("正在处理 Parquet 文件并生成元数据...")
for file_path in tqdm(parquet_files, desc="处理 Episodes"):
    try:
        # 提取 episode_index
        filename = os.path.basename(file_path)
        episode_index_str = filename.replace("episode_", "").replace(".parquet", "")
        episode_index = int(episode_index_str)
        print("file_path", file_path)
        # 读取 Parquet 文件
        df = pd.read_parquet(file_path)
        print("df", df.head())
        print("df.columns", df.columns)
        print("df['action'].shape", df['action'].shape)

        # 获取轨迹长度
        length = len(df)
        print("length", length)

        # 累加统计信息
        total_episodes += 1
        total_frames += length

        # --- 准备 episodes.jsonl 数据 ---
        # 假设任务名称固定为 "pick_put"
        episode_info = {
            "episode_index": episode_index,
            "tasks": ["[Robot] pull the drawer and then push the drawer."], # 根据你的例子
            "length": length
        }
        episodes_data.append(episode_info)

        # --- 准备 tasks.jsonl 数据 ---
        # 假设任务名称为 "pick_put"，ID 为 0
        task_info = {
             "episode_index": episode_index,
             "task_name": "pick_put",
             "task_id": 0 # 假设只有一个任务，ID为0
        }
        tasks_data.append(task_info)

        # --- 收集action和state数据用于stats.json ---
        # 处理action数据
        if 'action' in df.columns and len(df) > 0:
            try:
                # 获取action数据
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
                            print(f"警告：无法解析 episode {episode_index} 第 {idx} 帧的动作字符串。")
                            continue

                if episode_actions:
                    episode_actions = np.array(episode_actions)
                    all_actions.append(episode_actions)
                    
                    # 获取第一个action的维度信息
                    if first_action_shape is None:
                        first_action_shape = episode_actions[0].shape
                        print(f"从 episode {episode_index} 获取到动作维度: {first_action_shape}")
                    
                    # 计算该episode的动作统计值
                    episode_stats = compute_action_statistics(episode_actions)
                    if episode_stats:
                        episode_name = f"episode_{episode_index:06d}"
                        all_episode_stats[episode_name] = episode_stats
                        print(f"  📊 计算了 {episode_name} 的动作统计值")

            except Exception as e:
                print(f"警告：处理 episode {episode_index} 的action数据时出错: {e}")

        # 处理observation.state数据
        if 'observation.state' in df.columns and len(df) > 0:
            try:
                # 获取state数据
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
                            print(f"警告：无法解析 episode {episode_index} 第 {idx} 帧的状态字符串。")
                            continue
                
                if episode_states:
                    episode_states = np.array(episode_states)
                    all_states.append(episode_states)
                    
                    # 获取第一个state的维度信息
                    if first_state_shape is None:
                        first_state_shape = episode_states[0].shape
                        print(f"从 episode {episode_index} 获取到状态维度: {first_state_shape}")
                        
            except Exception as e:
                print(f"警告：处理 episode {episode_index} 的state数据时出错: {e}")


    except Exception as e:
        print(f"\n处理文件 {file_path} 时出错: {e}")
        continue # 跳过有问题的文件

# --- 写入 episodes.jsonl ---
episodes_jsonl_path = os.path.join(meta_path, "episodes.jsonl")
try:
    with open(episodes_jsonl_path, 'w', encoding='utf-8') as f:
        for item in episodes_data:
            # 确保写入的是紧凑的 JSON 字符串，并添加换行符
            json.dump(item, f, separators=(',', ':'))
            f.write('\n')
    print(f"成功写入: {episodes_jsonl_path}")
except Exception as e:
    print(f"写入 {episodes_jsonl_path} 时出错: {e}")

# # --- 写入 tasks.jsonl ---
# tasks_jsonl_path = os.path.join(meta_path, "tasks.jsonl")
# try:
#     with open(tasks_jsonl_path, 'w', encoding='utf-8') as f:
#         for item in tasks_data:
#             json.dump(item, f, separators=(',', ':'))
#             f.write('\n')
#     print(f"成功写入: {tasks_jsonl_path}")
# except Exception as e:
#     print(f"写入 {tasks_jsonl_path} 时出错: {e}")

# --- 写入 info.json ---
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
        json.dump(info_data, f, indent=4) # 使用 indent 使其更易读
    print(f"成功写入: {info_json_path}")
    # 打印计算出的统计信息
    print("\n--- 计算得到的统计信息 ---")
    print(f"总 Episodes (total_episodes): {info_data['total_episodes']}")
    print(f"总帧数 (total_frames): {info_data['total_frames']}")
    print(f"总视频数 (total_videos): {info_data['total_videos']}")
except Exception as e:
    print(f"写入 {info_json_path} 时出错: {e}")


# --- 计算统计信息并写入 stats.json ---
print("\n正在计算action和state的统计信息...")

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

# 计算action统计信息
if all_actions:
    try:
        # 将所有episode的action数据合并
        all_actions_concat = np.concatenate(all_actions, axis=0)  # [total_frames, action_dim]
        print(f"Action数据总帧数: {all_actions_concat.shape[0]}, 维度: {all_actions_concat.shape[1]}")
        action_stats = compute_stats(all_actions_concat, first_action_shape)
        print(f"Action统计信息计算完成，维度: {action_stats['dim']}")
    except Exception as e:
        print(f"计算action统计信息时出错: {e}")
        action_stats = {"min": None, "max": None, "mean": None, "std": None, "dim": None, "q01": None, "q99": None}
else:
    print("警告：未找到任何action数据")

# 计算state统计信息
if all_states:
    try:
        all_states_concat = np.concatenate(all_states, axis=0)
        print(f"State数据总帧数: {all_states_concat.shape[0]}, 维度: {all_states_concat.shape[1]}")
        state_stats = compute_stats(all_states_concat, first_state_shape)
        print(f"State统计信息计算完成，维度: {state_stats['dim']}")
    except Exception as e:
        print(f"计算state统计信息时出错: {e}")
        state_stats = {"min": None, "max": None, "mean": None, "std": None, "dim": None, "q01": None, "q99": None}
else:
    print("警告：未找到任何state数据")

# 构建stats数据结构
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
    print(f"成功写入: {stats_json_path}")

    # 打印统计信息摘要
    print("\n--- Action统计信息摘要 ---")
    if action_stats["dim"]:
        print(f"维度: {action_stats['dim']}")
        print(f"均值范围: {min(action_stats['mean']):.6f} ~ {max(action_stats['mean']):.6f}")
        print(f"标准差范围: {min(action_stats['std']):.6f} ~ {max(action_stats['std']):.6f}")
        print(f"1%分位数范围: {min(action_stats['q01']):.6f} ~ {max(action_stats['q01']):.6f}")
        print(f"99%分位数范围: {min(action_stats['q99']):.6f} ~ {max(action_stats['q99']):.6f}")
    else:
        print("无Action统计数据")

    print("\n--- State统计信息摘要 ---")
    if state_stats["dim"]:
        print(f"维度: {state_stats['dim']}")
        print(f"均值范围: {min(state_stats['mean']):.6f} ~ {max(state_stats['mean']):.6f}")
        print(f"标准差范围: {min(state_stats['std']):.6f} ~ {max(state_stats['std']):.6f}")
        print(f"1%分位数范围: {min(state_stats['q01']):.6f} ~ {max(state_stats['q01']):.6f}")
        print(f"99%分位数范围: {min(state_stats['q99']):.6f} ~ {max(state_stats['q99']):.6f}")
    else:
        print("无State统计数据")

except Exception as e:
    print(f"写入 {stats_json_path} 时出错: {e}")

# --- 生成动作统计值JSON文件 ---
if all_episode_stats:
    print("\n正在生成动作统计值文件...")
    
    # 计算数据集整体统计摘要
    dataset_summary_stats = compute_dataset_summary_statistics(all_episode_stats)
    
    # 构建动作统计数据结构
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
    
    # 保存动作统计值到JSON文件
    action_stats_path = os.path.join(meta_path, "action_statistics.json")
    try:
        with open(action_stats_path, 'w', encoding='utf-8') as f:
            json.dump(action_statistics_data, f, indent=2, ensure_ascii=False)
        print(f"✓ 动作统计值已保存到: {action_stats_path}")
        
        # 打印统计摘要
        print(f"\n--- 动作统计摘要 ---")
        print(f"总episode数: {dataset_summary_stats.get('total_episodes', 0)}")
        print(f"总帧数: {dataset_summary_stats.get('frame_statistics', {}).get('total_frames', 0)}")
        print(f"平均帧数/episode: {dataset_summary_stats.get('frame_statistics', {}).get('mean_frames_per_episode', 0):.1f}")
        
        if "gripper_summary" in dataset_summary_stats:
            gripper_summary = dataset_summary_stats["gripper_summary"]
            print(f"夹爪平均值范围: [{gripper_summary['min']:.1f}, {gripper_summary['max']:.1f}]")
        
        if "action_smoothness_summary" in dataset_summary_stats:
            smoothness_summary = dataset_summary_stats["action_smoothness_summary"]
            print(f"动作平滑度范围: [{smoothness_summary['min']:.4f}, {smoothness_summary['max']:.4f}]")
            
    except Exception as e:
        print(f"警告：保存动作统计值到 {action_stats_path} 时出错: {e}")
else:
    print("\n警告：未找到任何动作数据，无法生成动作统计值")

print("\n元数据文件生成完毕！")

# 生成 modality.json 文件
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
    print(f"✓ modality.json 已保存到: {modality_json_path}")
except Exception as e:
    print(f"警告：保存 modality.json 到 {modality_json_path} 时出错: {e}")
