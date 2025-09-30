# 读取robot数据集，并依据gripper的夹爪距离对原始数据长度进行截断，认为当松开夹爪后任务完成

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from typing import Dict, List, Tuple
import argparse


def detect_task_completion_by_gripper(gripper_data: np.ndarray, 
                                    open_threshold: float = 50.0,
                                    min_episode_length: int = 50,
                                    delay_frames: int = 10) -> int:
    """
    基于gripper数据检测任务完成点，并在截断点后延迟20帧

    Args:
        gripper_data: gripper关节数据 (1D array)
        open_threshold: 判断夹爪打开/关闭的阈值
        min_episode_length: 最小episode长度
        delay_frames: 截断后延迟的帧数，默认为20

    Returns:
        任务完成点的索引，如果未检测到则返回原始长度
    """
    # 找到夹爪从关闭到打开的变化点
    is_open = gripper_data > open_threshold

    # 找到最后一个从关闭到打开的转换点
    # 这通常表示任务完成（松开物体）
    transitions = np.diff(is_open.astype(int))
    open_transitions = np.where(transitions == 1)[0]  # 从0到1的转换

    if len(open_transitions) > 0:
        # 取最后一个打开转换点
        completion_point = open_transitions[-1] + 1

        # 延迟delay_frames帧，但不超过数据长度
        completion_point_delayed = min(completion_point + delay_frames, len(gripper_data))

        # 确保截断后的长度不小于最小长度
        if completion_point_delayed >= min_episode_length:
            return completion_point_delayed

    # 如果没有检测到明显的任务完成点，返回原始长度
    return len(gripper_data)


def process_episode_data(episode_path: str, 
                        open_threshold: float = 50.0,
                        min_episode_length: int = 50) -> Tuple[pd.DataFrame, int]:
    """
    处理单个episode数据，检测任务完成点并截断数据
    
    Returns:
        (截断后的数据, 原始长度)
    """
    # 读取episode数据
    df = pd.read_parquet(episode_path)
    print(f"episode_path: {episode_path}")
    print(f"df.shape: {df.shape}")
    print(f"df.columns: {df.columns}")
    print(f"observation.state: {df['observation.state']}")
    print(f"observation.state.shape: {df['observation.state'].shape}")

    # 提取gripper数据（action列可能为字符串或list/ndarray，需解析）
    if 'action' in df.columns:
        # 解析action数据，确保为二维np.ndarray
        action_list = []
        for idx, action_data in enumerate(df['action']):
            if isinstance(action_data, (list, np.ndarray)):
                action_list.append(np.array(action_data, dtype=np.float32))
            elif isinstance(action_data, str):
                try:
                    parsed_action = json.loads(action_data.replace("'", "\""))
                    if isinstance(parsed_action, list):
                        action_list.append(np.array(parsed_action, dtype=np.float32))
                    else:
                        print(f"警告：第{idx}帧 action 解析后不是list，跳过。")
                        action_list.append(np.zeros(7, dtype=np.float32))
                except json.JSONDecodeError:
                    print(f"警告：无法解析 episode {episode_path} 第 {idx} 帧的动作字符串。")
                    action_list.append(np.zeros(7, dtype=np.float32))
            else:
                print(f"警告：第{idx}帧 action 类型未知，跳过。")
                action_list.append(np.zeros(7, dtype=np.float32))
        action_array = np.stack(action_list, axis=0)
        print(f"action_array.shape: {action_array.shape}")
        if action_array.shape[1] >= 7:
            gripper_data = action_array[:, 6]  # 第7列是gripper

            # 检测任务完成点
            completion_point = detect_task_completion_by_gripper(
                gripper_data, open_threshold, min_episode_length
            )

            # 截断数据
            if completion_point < len(df):
                df_truncated = df.iloc[:completion_point].copy()
                print(f"截断episode {episode_path}: {len(df)} -> {len(df_truncated)} frames")
                return df_truncated, len(df)
    
    return df, len(df)


def recalculate_stats_from_truncated_data(dataset_path: str, truncation_info: Dict) -> Dict:
    """
    基于截断后的数据重新计算统计信息
    """
    print("正在重新计算截断后数据的统计信息...")
    
    # 收集所有截断后的数据
    all_action_data = []
    all_observation_data = []
    
    data_path = Path(dataset_path) / "data"
    
    for chunk_dir in data_path.iterdir():
        if chunk_dir.is_dir():
            for episode_file in chunk_dir.glob("episode_*.parquet"):
                episode_name = episode_file.stem
                
                if episode_name in truncation_info:
                    # 读取截断后的数据
                    df = pd.read_parquet(episode_file)
                    
                    # 提取action数据
                    if 'action' in df.columns:
                        action_list = []
                        for action_data in df['action']:
                            if isinstance(action_data, (list, np.ndarray)):
                                action_list.append(np.array(action_data, dtype=np.float32))
                            elif isinstance(action_data, str):
                                try:
                                    parsed_action = json.loads(action_data.replace("'", "\""))
                                    if isinstance(parsed_action, list):
                                        action_list.append(np.array(parsed_action, dtype=np.float32))
                                except:
                                    continue
                        if action_list:
                            all_action_data.extend(action_list)
                    
                    # 提取observation.state数据
                    if 'observation.state' in df.columns:
                        obs_list = []
                        for obs_data in df['observation.state']:
                            if isinstance(obs_data, (list, np.ndarray)):
                                obs_list.append(np.array(obs_data, dtype=np.float32))
                            elif isinstance(obs_data, str):
                                try:
                                    parsed_obs = json.loads(obs_data.replace("'", "\""))
                                    if isinstance(parsed_obs, list):
                                        obs_list.append(np.array(parsed_obs, dtype=np.float32))
                                except:
                                    continue
                        if obs_list:
                            all_observation_data.extend(obs_list)
    
    # 计算统计信息
    new_stats = {}
    
    if all_action_data:
        action_array = np.array(all_action_data)
        new_stats['action'] = calculate_statistics(action_array, 'action')
        print(f"重新计算了action统计信息，数据形状: {action_array.shape}")
    
    if all_observation_data:
        obs_array = np.array(all_observation_data)
        new_stats['observation.state'] = calculate_statistics(obs_array, 'observation.state')
        print(f"重新计算了observation.state统计信息，数据形状: {obs_array.shape}")
    
    return new_stats


def calculate_statistics(data_array: np.ndarray, data_type: str) -> Dict:
    """
    计算数据的统计信息
    """
    stats = {}
    
    # 计算基本统计量
    stats['mean'] = np.mean(data_array, axis=0).tolist()
    stats['std'] = np.std(data_array, axis=0).tolist()
    stats['min'] = np.min(data_array, axis=0).tolist()
    stats['max'] = np.max(data_array, axis=0).tolist()
    
    # 计算分位数
    stats['q01'] = np.percentile(data_array, 1, axis=0).tolist()
    stats['q99'] = np.percentile(data_array, 99, axis=0).tolist()
    
    return stats


def update_dataset_metadata(dataset_path: str, 
                          original_stats: Dict,
                          truncation_info: Dict) -> Dict:
    """
    更新数据集元数据，反映截断后的统计信息
    """
    # 计算新的统计信息
    new_total_frames = sum(info['truncated_length'] for info in truncation_info.values())
    new_mean_frames = new_total_frames / len(truncation_info)
    
    # 重新计算基于截断后数据的统计信息
    new_stats = recalculate_stats_from_truncated_data(dataset_path, truncation_info)
    
    # 更新统计信息
    updated_stats = original_stats.copy()
    
    # 处理不同的元数据格式
    if 'dataset_info' in updated_stats:
        # 自定义格式（action_statistics.json）
        updated_stats['dataset_info']['total_frames'] = new_total_frames
        updated_stats['dataset_info']['processing_timestamp'] = pd.Timestamp.now().isoformat()
        
        if 'dataset_summary' in updated_stats:
            updated_stats['dataset_summary']['frame_statistics'] = {
                'total_frames': new_total_frames,
                'mean_frames_per_episode': new_mean_frames,
                'std_frames_per_episode': np.std([info['truncated_length'] for info in truncation_info.values()]),
                'min_frames': min(info['truncated_length'] for info in truncation_info.values()),
                'max_frames': max(info['truncated_length'] for info in truncation_info.values())
            }
    else:
        # lerobot标准格式（stats.json），用新的统计信息替换
        updated_stats = new_stats.copy()
        
        # 添加截断信息
        updated_stats['truncation_info'] = {
            'method': 'gripper_based_task_completion',
            'open_threshold': 50.0,
            'min_episode_length': 50,
            'episodes_processed': len(truncation_info),
            'total_frames_removed': sum(info['original_length'] - info['truncated_length'] 
                                      for info in truncation_info.values()),
            'episode_details': truncation_info,
            'new_total_frames': new_total_frames,
            'new_mean_frames': new_mean_frames
        }
    
    return updated_stats


def update_episodes_jsonl(episodes_path: Path, truncation_info: Dict) -> None:
    """
    更新episodes.jsonl文件，反映截断后的episode长度
    
    Args:
        episodes_path: episodes.jsonl文件路径
        truncation_info: 截断信息字典
    """
    # 读取原始episodes数据
    episodes_data = []
    with open(episodes_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                episodes_data.append(json.loads(line))
    
    # 更新每个episode的长度
    for episode in episodes_data:
        episode_index = episode.get('episode_index')
        episode_name = f"episode_{episode_index:06d}"
        
        if episode_name in truncation_info:
            # 更新截断后的长度
            episode['length'] = truncation_info[episode_name]['truncated_length']
            print(f"更新episode {episode_index}: {truncation_info[episode_name]['original_length']} -> {episode['length']} frames")
        else:
            print(f"警告: 未找到episode {episode_index} 的截断信息")
    
    # 保存更新后的episodes数据
    with open(episodes_path, 'w') as f:
        for episode in episodes_data:
            f.write(json.dumps(episode) + '\n')


def process_robot_dataset(dataset_path: str,
                         open_threshold: float = 50.0,
                         min_episode_length: int = 50,
                         backup: bool = True) -> None:
    """
    处理robot数据集，基于gripper数据截断episodes
    
    Args:
        dataset_path: 数据集路径
        open_threshold: gripper打开阈值
        min_episode_length: 最小episode长度
        backup: 是否创建备份
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise ValueError(f"数据集路径不存在: {dataset_path}")
    
    # 创建备份
    if backup:
        backup_path = dataset_path.parent / f"{dataset_path.name}_backup"
        if backup_path.exists():
            shutil.rmtree(backup_path)
        shutil.copytree(dataset_path, backup_path)
        print(f"已创建备份: {backup_path}")
    
    # 读取原始元数据
    meta_path = dataset_path / "meta"
    info_path = meta_path / "info.json"
    stats_path = meta_path / "action_statistics.json"
    
    # 检查是否有stats.json（lerobot标准格式）
    if not stats_path.exists():
        stats_path = meta_path / "stats.json"
    
    if not info_path.exists():
        raise ValueError(f"info.json不存在: {info_path}")
    
    with open(info_path, 'r') as f:
        info_data = json.load(f)
    
    original_stats = {}
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            original_stats = json.load(f)
    
    # 处理所有episodes
    data_path = dataset_path / "data"
    truncation_info = {}
    
    for chunk_dir in data_path.iterdir():
        if chunk_dir.is_dir():
            print(f"处理chunk: {chunk_dir.name}")
            
            for episode_file in chunk_dir.glob("episode_*.parquet"):
                episode_name = episode_file.stem
                print(f"处理episode: {episode_name}")
                
                # 处理episode数据
                df_truncated, original_length = process_episode_data(
                    str(episode_file), open_threshold, min_episode_length
                )
                
                # 保存截断后的数据
                df_truncated.to_parquet(episode_file, index=False)
                
                # 记录截断信息
                truncation_info[episode_name] = {
                    'original_length': original_length,
                    'truncated_length': len(df_truncated),
                    'frames_removed': original_length - len(df_truncated)
                }
    
    # 更新元数据
    updated_stats = update_dataset_metadata(
        str(dataset_path), original_stats, truncation_info
    )
    
    # 保存更新后的统计信息
    if stats_path.exists():
        with open(stats_path, 'w') as f:
            json.dump(updated_stats, f, indent=2, ensure_ascii=False)
        print(f"已更新统计信息: {stats_path}")
    else:
        print("警告: 未找到统计信息文件，跳过更新")
    
    # 更新info.json中的总帧数
    info_data['total_frames'] = sum(info['truncated_length'] for info in truncation_info.values())
    with open(info_path, 'w') as f:
        json.dump(info_data, f, indent=2, ensure_ascii=False)
    
    # 更新episodes.jsonl文件
    episodes_path = meta_path / "episodes.jsonl"
    if episodes_path.exists():
        update_episodes_jsonl(episodes_path, truncation_info)
        print(f"已更新episodes信息: {episodes_path}")
    else:
        print("警告: 未找到episodes.jsonl文件，跳过更新")
    
    # 打印总结
    total_original = sum(info['original_length'] for info in truncation_info.values())
    total_truncated = sum(info['truncated_length'] for info in truncation_info.values())
    total_removed = total_original - total_truncated
    
    print(f"\n处理完成!")
    print(f"原始总帧数: {total_original}")
    print(f"截断后总帧数: {total_truncated}")
    print(f"移除帧数: {total_removed}")
    print(f"压缩率: {total_removed/total_original*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="基于gripper数据截断robot数据集")
    parser.add_argument("dataset_path", help="数据集路径")
    parser.add_argument("--open_threshold", type=float, default=50.0,
                       help="gripper打开阈值 (默认: 50.0)")
    parser.add_argument("--min_episode_length", type=int, default=50,
                       help="最小episode长度 (默认: 50)")
    parser.add_argument("--no_backup", action="store_true",
                       help="不创建备份")
    
    args = parser.parse_args()
    
    try:
        process_robot_dataset(
            dataset_path=args.dataset_path,
            open_threshold=args.open_threshold,
            min_episode_length=args.min_episode_length,
            backup=not args.no_backup
        )
    except Exception as e:
        print(f"处理失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


"""
使用示例:

1. 基本使用:
python 2_update_robotdataset.py /path/to/robot/dataset

2. 自定义参数:
python 2_update_robotdataset.py /path/to/robot/dataset --open_threshold 60.0 --min_episode_length 100

3. 不创建备份:
python 2_update_robotdataset.py /path/to/robot/dataset --no_backup

脚本功能说明:
- 自动检测gripper从关闭到打开的变化点，认为这是任务完成的标志
- 截断数据，只保留任务完成前的部分
- 自动创建备份（除非指定--no_backup）
- 重新计算基于截断后数据的统计信息（stats.json）
- 更新数据集元数据，包括统计信息和截断记录
- 更新episodes.jsonl文件中的episode长度信息
- 支持自定义gripper打开阈值和最小episode长度
- 兼容lerobot标准格式和自定义格式的元数据文件

注意事项:
- 确保数据集路径正确且包含meta/info.json文件
- 建议先在小数据集上测试参数设置
- 脚本会修改原始数据，建议先备份重要数据
"""