import argparse
import logging
import time
from pathlib import Path
from typing import List
import os
import shutil
import numpy as np

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.control_utils import (
    control_loop,
    has_method,
    init_keyboard_listener,
    init_policy,
    log_control_info,
    record_episode,
    reset_environment,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
    stop_recording,
    warmup_record,
)
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.utils.utils import init_hydra_config, init_logging, log_say, none_or_int

import mujoco
import mujoco.viewer
import numpy as np
from lerobot_kinematics import lerobot_IK, lerobot_FK,get_robot

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import itertools


########################################################################################
# Control modes
########################################################################################



def update_traj_geoms(mjmodel, mjdata, positions):
    """
    动态更新预定义的轨迹点 geom 的位置和颜色
    :param mjmodel: mujoco.MjModel
    :param mjdata: mujoco.MjData
    :param positions: 轨迹点列表，每个元素为 [x, y, z]
    """
    max_points = 8  # 和XML里定义的数量一致
    n = min(len(positions), max_points)
    # 获取所有轨迹点geom的id
    geom_ids = [mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_GEOM, f"traj_point{i}") for i in range(max_points)]
    # 先全部隐藏
    for i, gid in enumerate(geom_ids):
        mjmodel.geom_pos[gid] = [0, 0, -10]  # 隐藏到视野外
        # mjmodel.geom_rgba[gid] = [0, 0, 1, 0.0]  # 透明
    # 显示最近的n个点
    for i in range(n):
        gid = geom_ids[i]
        mjmodel.geom_pos[gid] = positions[-n + i]  # 按顺序显示
        if i == n - 1:
            # 当前点用红色
            mjmodel.geom_rgba[gid] = [1, 0, 0, 1]
            mjmodel.geom_size[gid][0] = 0.03  # 当前点大一点
        else:
            # 轨迹点用蓝色
            mjmodel.geom_rgba[gid] = [0, 0, 1, 0.5]
            mjmodel.geom_size[gid][0] = 0.03

# TODO 2.3.7版本的可视化API需要手动修改，没有viewer.user_scn和viewer.add_marker！！！！！！！
def draw_trajectory_visualization(mjdata, positions):
    # 只显示最近8个点
    max_marker = min(8, len(positions))
    for i in range(max_marker):
        idx = -max_marker + i  # 取最近的点
        pos = positions[idx]
        mjdata.marker.pos[i] = pos
        mjdata.marker.size[i] = [0.01, 0.01, 0.01]
        mjdata.marker.rgba[i] = [0.0, 0.0, 1.0, 0.5]  # 蓝色
        mjdata.marker.objtype[i] = mujoco.mjtObj.mjOBJ_UNKNOWN
        mjdata.marker.objid[i] = -1
        mjdata.marker.active[i] = 1
        mjdata.marker.type[i] = mujoco.mjtGeom.mjGEOM_SPHERE
    # 当前点（红色，放在最后一个marker槽）
    if len(positions) > 0:
        mjdata.marker.pos[max_marker-1] = positions[-1]
        mjdata.marker.size[max_marker-1] = [0.03, 0.03, 0.03]
        mjdata.marker.rgba[max_marker-1] = [1.0, 0.0, 0.0, 1.0]
        mjdata.marker.objtype[max_marker-1] = mujoco.mjtObj.mjOBJ_UNKNOWN
        mjdata.marker.objid[max_marker-1] = -1
        mjdata.marker.active[max_marker-1] = 1
        mjdata.marker.type[max_marker-1] = mujoco.mjtGeom.mjGEOM_SPHERE
    # 关闭多余的marker
    for i in range(max_marker, mjdata.marker.pos.shape[0]):
        mjdata.marker.active[i] = 0


def replay(
    root: Path,
    repo_id: str,
    episode: int,
    fps: int | None = None,
    play_sounds: bool = True,
    local_files_only: bool = False,
    use_simulation: bool = True,
    xml_path: str = None,
):
    # TODO(rcadene, aliberts): refactor with control_loop, once `dataset` is an instance of LeRobotDataset
    # TODO(rcadene): Add option to record logs

    print("in replay")

    dataset = LeRobotDataset(repo_id, root=root, episodes=[episode], local_files_only=True)
    actions = dataset.hf_dataset.select_columns("action")

    # print(dataset)

    # if not robot.is_connected:
    #     robot.connect()

    # Initialize MuJoCo simulation if requested
    mjmodel = None
    mjdata = None
    viewer = None
    qpos_indices = None
    
    if use_simulation:
        # Set up the MuJoCo render backend
        os.environ["MUJOCO_GL"] = "egl"
        
        # Define joint names for the robot
        JOINT_NAMES = ['shoulder_rotation_joint', 'shoulder_pitch_joint', 'ellbow_joint', 
                      'wrist_pitch_joint', 'wrist_jaw_joint', 'wrist_roll_joint', 'gripper_joint']
        
        # Use provided XML path or default
        if xml_path is None:
            xml_path = "./lerobot/common/robot_devices/controllers/scene_plus.xml"
        
        try:
            mjmodel = mujoco.MjModel.from_xml_path(xml_path)
            qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in JOINT_NAMES])
            mjdata = mujoco.MjData(mjmodel)

            init_qpos = np.array([0.0, -3.1, 3.0, 0.0, 0.0, 1.57, -0.157])
            target_qpos = init_qpos.copy()  # Copy the initial joint positions
            init_gpos = lerobot_FK(init_qpos[:6], robot=get_robot('so100_plus'))
            target_gpos = init_gpos.copy()
            target_gpos_last = init_gpos.copy()
            target_gpos_visualization = init_gpos.copy()
            
            # Initialize trajectory visualization
            manager = multiprocessing.Manager()
            trajectory_positions = manager.list()  # Store trajectory positions (shared list)
            trajectory_rotations = []  # Store trajectory rotations
            # 检查可视化环境并启动3D可视化进程
            print("Checking visualization environment...")
            has_display = 'DISPLAY' in os.environ and os.environ['DISPLAY']
            print(f"DISPLAY environment: {os.environ.get('DISPLAY', 'Not set')}")
            
            if has_display:
                vis_proc = start_trajectory_visualization_process(trajectory_positions)
                if vis_proc is None:
                    print("Falling back to static plot generation")
            else:
                print("No display environment detected, will save static plots instead")
                vis_proc = None
            
            print(f"MuJoCo simulation initialized with XML: {xml_path}")
            print(f"Initial gpos: {[f'{x:.3f}' for x in init_gpos]}")
            print("Trajectory visualization enabled")

        except Exception as e:
            print(f"Failed to initialize MuJoCo simulation: {e}")
            print("Falling back to robot-only replay")
            use_simulation = False

    log_say("Replaying episode", play_sounds, blocking=True)
    
    try:
        if use_simulation and mjmodel is not None:
            # Launch MuJoCo viewer for simulation
            with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
                for idx in range(dataset.num_frames):

                    # if idx == 0:  # Only add once at the beginning
                    #     try:
                    #         # Add a marker to mjdata
                    #         marker = mjdata.marker
                    #         marker.pos[0] = [0.0, 0.0, 3.0]
                    #         marker.size[0] = [0.02, 0.02, 0.02]
                    #         marker.rgba[0] = [1.0, 1.0, 0.0, 1.0]
                    #         marker.objtype[0] = mujoco.mjtObj.mjOBJ_UNKNOWN
                    #         marker.objid[0] = -1
                    #         marker.active[0] = 1
                    #         marker.type[0] = mujoco.mjtGeom.mjGEOM_SPHERE
                    #     except Exception as e:
                    #         print(f"Warning: Failed to add origin marker to viewer: {e}")

                    start_episode_t = time.perf_counter()
                    action = actions[idx]["action"]
                    
                    # Convert action to numpy array if needed
                    if hasattr(action, 'numpy'):
                        target_gpos = action.numpy()
                    else:
                        target_gpos = np.array(action)
                    

                    print(f'Frame {idx}: target_gpos: {[f"{x:.3f}" for x in target_gpos]}')

                    # Update simulation directly with joint positions
                    fd_qpos = mjdata.qpos[qpos_indices][:6]
                    qpos_inv, ik_success = lerobot_IK(fd_qpos[:6],target_gpos[:6],robot=get_robot('so100_plus'))

                    target_qpos[6:] = (target_gpos[-1]-24.31) / 75.38

                    if ik_success:
                        target_qpos = np.concatenate((qpos_inv[:6], target_qpos[6:]))
                        mjdata.qpos[qpos_indices] = target_qpos
                        mujoco.mj_step(mjmodel, mjdata)
                        target_gpos_last = target_gpos.copy()
                        target_gpos_visualization = target_gpos.copy()
                    else:
                        print(f"IK failed at frame {idx}")
                        target_gpos_visualization = target_gpos.copy()
                        target_gpos = target_gpos_last.copy()

                    # Add current position to trajectory
                    trajectory_positions.append(target_gpos_visualization[:3].copy())  # manager.list()自动同步
                    trajectory_rotations.append(target_gpos_visualization[3:6].copy())

                    # Draw trajectory and rotation visualization
                    # print(f"trajectory_positions: {len(trajectory_positions)}")
                    # update_traj_geoms(mjmodel, mjdata, trajectory_positions)
                        
                    # Update viewer
                    with viewer.lock():
                        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mjdata.time % 2)
                    viewer.sync()

                    dt_s = time.perf_counter() - start_episode_t
                    busy_wait(1 / fps - dt_s)
                    time.sleep(0.02)  # 增加延迟，单位为秒

                    # dt_s = time.perf_counter() - start_episode_t
                    # log_control_info(robot, dt_s, fps=fps)
                    
                    # Check if viewer is still running
                    if not viewer.is_running():
                        print("Viewer closed, stopping replay")
                        break
                        
                print(f"Replay completed. Total trajectory points: {len(trajectory_positions)}")
                
                # 保存轨迹图片作为备用
                trajectory_list = list(trajectory_positions)  # 转换为普通list
                if trajectory_list:
                    visualize_trajectory_simple(trajectory_list, f"trajectory_episode_{episode}.png")
                
                # 关闭可视化进程
                if vis_proc is not None:
                    vis_proc.terminate()
                    vis_proc.join()
                    print("Trajectory visualization process terminated.")
                
        else:
            raise NotImplementedError("Real robot replay is not implemented yet")
                
    except KeyboardInterrupt:
        print("Replay interrupted by user")
    except Exception as e:
        print(f"Error during replay: {e}")
    finally:
        if viewer is not None:
            viewer.close()
        print("Replay completed")


def visualize_trajectory_positions_3d(trajectory_positions, interval=100):
    """
    使用matplotlib 3D动画实时可视化trajectory_positions。
    :param trajectory_positions: manager.list() of [x, y, z]
    :param interval: 动画帧间隔（毫秒）
    """
    try:
        import matplotlib
        
        # 测试GUI环境是否真正可用
        def test_gui_available():
            try:
                # 测试tkinter是否能创建窗口
                import tkinter as tk
                root = tk.Tk()
                root.withdraw()  # 隐藏窗口
                root.geometry("1x1")  # 最小窗口
                root.update()  # 强制更新
                root.destroy()
                return True
            except Exception:
                return False
        
        # 检查显示环境
        if 'DISPLAY' not in os.environ or not os.environ['DISPLAY']:
            print("Warning: No DISPLAY environment variable found. Skipping GUI visualization.")
            return
        
        # 测试GUI是否真正可用
        if not test_gui_available():
            print("Warning: GUI environment test failed. Skipping visualization.")
            return
        
        # 尝试使用GUI后端
        gui_backend_found = False
        backends_to_try = ['TkAgg', 'Qt5Agg', 'Qt4Agg']
        
        for backend in backends_to_try:
            try:
                matplotlib.use(backend)
                # 测试是否能创建figure
                test_fig = matplotlib.pyplot.figure(figsize=(1, 1))
                matplotlib.pyplot.close(test_fig)
                gui_backend_found = True
                print(f"Successfully using matplotlib backend: {backend}")
                break
            except Exception as e:
                print(f"Backend {backend} failed: {e}")
                continue
        
        if not gui_backend_found:
            print("Warning: No working GUI backend found. Skipping visualization.")
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')    
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Trajectory Visualization')

        # x:0.202,0.421
        # y:-0.35,0.35
        # z:0.14,0.3
        # 绘制一个具有透明度的立方体
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # 定义立方体的8个顶点
        cube_x = [0.202, 0.421]
        cube_y = [-0.35, 0.35]
        cube_z = [0.14, 0.3]
        
        # 8个顶点
        vertices = [
            [cube_x[0], cube_y[0], cube_z[0]],
            [cube_x[1], cube_y[0], cube_z[0]],
            [cube_x[1], cube_y[1], cube_z[0]],
            [cube_x[0], cube_y[1], cube_z[0]],
            [cube_x[0], cube_y[0], cube_z[1]],
            [cube_x[1], cube_y[0], cube_z[1]],
            [cube_x[1], cube_y[1], cube_z[1]],
            [cube_x[0], cube_y[1], cube_z[1]],
        ]
        # 立方体的6个面，每个面由4个顶点组成
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # 左面
        ]
        # 创建Poly3DCollection对象
        cube = Poly3DCollection(faces, facecolors=(0, 1, 0, 0.2), edgecolors='g', linewidths=1, alpha=0.2)
        ax.add_collection3d(cube)

        # 固定x轴范围
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        # 绘制原点坐标系
        axis_len = 0.1  # 坐标轴长度
        ax.quiver(0, 0, 0, axis_len, 0, 0, color='r', linewidth=2, arrow_length_ratio=0.2)
        ax.quiver(0, 0, 0, 0, axis_len, 0, color='g', linewidth=2, arrow_length_ratio=0.2)
        ax.quiver(0, 0, 0, 0, 0, axis_len, color='b', linewidth=2, arrow_length_ratio=0.2)
        ax.text(axis_len, 0, 0, 'X', color='r')
        ax.text(0, axis_len, 0, 'Y', color='g')
        ax.text(0, 0, axis_len, 'Z', color='b')

        line, = ax.plot([], [], [], 'b-', lw=2, label='Trajectory')
        point, = ax.plot([], [], [], 'ro', markersize=8, label='Current')

        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
            return line, point

        def update(frame):
            arr = np.array(trajectory_positions)
            if arr.shape[0] == 0:
                return line, point
            xs = arr[:, 0]
            ys = arr[:, 1]
            zs = arr[:, 2]
            line.set_data(xs, ys)
            line.set_3d_properties(zs)
            point.set_data([xs[-1]], [ys[-1]])
            point.set_3d_properties([zs[-1]])

            return line, point

        ani = animation.FuncAnimation(
            fig, update, frames=itertools.count(),
            init_func=init, interval=interval, blit=False
        )
        ax.legend()
        
        # 确保窗口显示并保持打开
        plt.show(block=True)
        
    except Exception as e:
        print(f"Error in trajectory visualization: {e}")
        import traceback
        traceback.print_exc()

def start_trajectory_visualization_process(trajectory_positions, interval=100):
    """
    启动一个新进程进行3D轨迹可视化。
    :param trajectory_positions: list of [x, y, z]
    :param interval: 动画帧间隔（毫秒）
    """
    try:
        p = multiprocessing.Process(target=visualize_trajectory_positions_3d, args=(trajectory_positions, interval))
        p.start()
        return p
    except Exception as e:
        print(f"Failed to start visualization process: {e}")
        print("Trajectory visualization disabled")
        return None

def visualize_trajectory_simple(trajectory_positions, save_path="trajectory_plot.png"):
    """
    简单的轨迹可视化，保存为图片文件
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # 无GUI后端
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        
        if len(trajectory_positions) == 0:
            print("No trajectory data to visualize")
            return
            
        arr = np.array(trajectory_positions)
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制轨迹
        xs, ys, zs = arr[:, 0], arr[:, 1], arr[:, 2]
        ax.plot(xs, ys, zs, 'b-', linewidth=2, label='Trajectory')
        ax.scatter(xs[-1], ys[-1], zs[-1], c='red', s=100, label='End Point')
        ax.scatter(xs[0], ys[0], zs[0], c='green', s=100, label='Start Point')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Robot Trajectory ({len(trajectory_positions)} points)')
        ax.legend()
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Trajectory plot saved to: {save_path}")
        
    except Exception as e:
        print(f"Error saving trajectory plot: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Set common options for all the subparsers
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--robot-path",
        type=str,
        default="lerobot/configs/robot/koch.yaml",
        help="Path to robot yaml file used to instantiate the robot using `make_robot` factory function.",
    )
    base_parser.add_argument(
        "--robot-overrides",
        type=str,
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )

    parser_calib = subparsers.add_parser("calibrate", parents=[base_parser])
    parser_calib.add_argument(
        "--arms",
        type=str,
        nargs="*",
        help="List of arms to calibrate (e.g. `--arms left_follower right_follower left_leader`)",
    )

    parser_teleop = subparsers.add_parser("teleoperate", parents=[base_parser])
    parser_teleop.add_argument(
        "--fps", type=none_or_int, default=None, help="Frames per second (set to None to disable)"
    )
    parser_teleop.add_argument(
        "--display-cameras",
        type=int,
        default=1,
        help="Display all cameras on screen (set to 1 to display or 0).",
    )

    parser_record = subparsers.add_parser("record", parents=[base_parser])
    task_args = parser_record.add_mutually_exclusive_group(required=True)
    parser_record.add_argument(
        "--fps", type=none_or_int, default=None, help="Frames per second (set to None to disable)"
    )
    task_args.add_argument(
        "--single-task",
        type=str,
        help="A short but accurate description of the task performed during the recording.",
    )
    # TODO(aliberts): add multi-task support
    # task_args.add_argument(
    #     "--multi-task",
    #     type=int,
    #     help="You will need to enter the task performed at the start of each episode.",
    # )
    parser_record.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory where the dataset will be stored (e.g. 'dataset/path').",
    )
    parser_record.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/test",
        help="Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).",
    )
    parser_record.add_argument(
        "--local-files-only",
        type=int,
        default=0,
        help="Use local files only. By default, this script will try to fetch the dataset from the hub if it exists.",
    )
    parser_record.add_argument(
        "--warmup-time-s",
        type=int,
        default=10,
        help="Number of seconds before starting data collection. It allows the robot devices to warmup and synchronize.",
    )
    parser_record.add_argument(
        "--episode-time-s",
        type=int,
        default=60,
        help="Number of seconds for data recording for each episode.",
    )
    parser_record.add_argument(
        "--reset-time-s",
        type=int,
        default=60,
        help="Number of seconds for resetting the environment after each episode.",
    )
    parser_record.add_argument("--num-episodes", type=int, default=50, help="Number of episodes to record.")
    parser_record.add_argument(
        "--run-compute-stats",
        type=int,
        default=1,
        help="By default, run the computation of the data statistics at the end of data collection. Compute intensive and not required to just replay an episode.",
    )
    parser_record.add_argument(
        "--push-to-hub",
        type=int,
        default=1,
        help="Upload dataset to Hugging Face hub.",
    )
    parser_record.add_argument(
        "--tags",
        type=str,
        nargs="*",
        help="Add tags to your dataset on the hub.",
    )
    parser_record.add_argument(
        "--num-image-writer-processes",
        type=int,
        default=0,
        help=(
            "Number of subprocesses handling the saving of frames as PNGs. Set to 0 to use threads only; "
            "set to ≥1 to use subprocesses, each using threads to write images. The best number of processes "
            "and threads depends on your system. We recommend 4 threads per camera with 0 processes. "
            "If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses."
        ),
    )
    parser_record.add_argument(
        "--num-image-writer-threads-per-camera",
        type=int,
        default=4,
        help=(
            "Number of threads writing the frames as png images on disk, per camera. "
            "Too many threads might cause unstable teleoperation fps due to main thread being blocked. "
            "Not enough threads might cause low camera fps."
        ),
    )
    parser_record.add_argument(
        "--resume",
        type=int,
        default=0,
        help="Resume recording on an existing dataset.",
    )
    parser_record.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        type=str,
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`."
        ),
    )
    parser_record.add_argument(
        "--policy-overrides",
        type=str,
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )


    parser_replay = subparsers.add_parser("replay", parents=[base_parser])
    parser_replay.add_argument(
        "--fps", type=none_or_int, default=None, help="Frames per second (set to None to disable)"
    )
    parser_replay.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory where the dataset will be stored (e.g. 'dataset/path').",
    )
    parser_replay.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/test",
        help="Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).",
    )
    parser_replay.add_argument(
        "--local-files-only",
        type=int,
        default=0,
        help="Use local files only. By default, this script will try to fetch the dataset from the hub if it exists.",
    )
    parser_replay.add_argument("--episode", type=int, default=0, help="Index of the episode to replay.")
    parser_replay.add_argument(
        "--use-simulation",
        type=int,
        default=1,
        help="Use MuJoCo simulation for replay.",
    )
    parser_replay.add_argument(
        "--xml-path",
        type=str,
        default=None,
        help="Path to MuJoCo XML file for simulation.",
    )

    args = parser.parse_args()

    init_logging()

    control_mode = args.mode
    robot_path = args.robot_path
    robot_overrides = args.robot_overrides
    kwargs = vars(args)
    del kwargs["mode"]
    del kwargs["robot_path"]
    del kwargs["robot_overrides"]

    robot_cfg = init_hydra_config(robot_path, robot_overrides)
    # robot = make_robot(robot_cfg)

    # if control_mode == "calibrate":
    #     calibrate(robot, **kwargs)

    # elif control_mode == "teleoperate":
    #     teleoperate(robot, **kwargs)

    # elif control_mode == "record":
    #     record(robot, **kwargs)

    if control_mode == "replay":
        replay(**kwargs)

    # if robot.is_connected:
    #     # Disconnect manually to avoid a "Core dump" during process
    #     # termination due to camera threads not properly exiting.
    #     robot.disconnect()
