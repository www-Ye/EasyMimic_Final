import logging
import time
import traceback
from contextlib import nullcontext
from copy import copy
from functools import cache

import cv2
import torch
import tqdm
from deepdiff import DeepDiff
from termcolor import colored

from lerobot.common.datasets.image_writer import safe_stop_image_writer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import get_features_from_robot
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config, set_global_seed
from lerobot.scripts.eval import get_pretrained_policy_path
import numpy as np

from scipy.spatial.transform import Rotation as R
from typing import Tuple

EULER_CONVENTION = "XYZ" 

def apply_relative_transform(
    absolute_pose: np.ndarray, 
    relative_pose: np.ndarray
) -> np.ndarray:
    """
    Apply relative transformation to absolute pose.

    Given an absolute pose (pose A in world) and a transformation relative to that pose 
    (pose B relative to A), compute the absolute pose of the second pose in world coordinates 
    (pose B in world).

    Args:
        absolute_pose: NumPy array with shape (6,), representing the absolute pose of the first 
                       [x, y, z, roll, pitch, yaw] (units: meters, radians).
        relative_pose: NumPy array with shape (6,), representing the transformation relative to the first pose
                       [dx, dy, dz, droll, dpitch, dyaw].
                       dx, dy, dz are defined in the coordinate system of the first pose.
                       droll, dpitch, dyaw define the rotation from the first pose to
                       the second pose.

    Returns:
        NumPy array with shape (6,), representing the absolute pose of the second pose in world coordinates
        [x, y, z, roll, pitch, yaw].

    Raises:
        ValueError: If input array shapes are incorrect.
    """
    if absolute_pose.shape != (6,) or relative_pose.shape != (6,):
        raise ValueError(f"Input poses must have shape (6,), but got {absolute_pose.shape} and {relative_pose.shape}")

    # 1. Decompose poses
    abs_xyz = absolute_pose[:3]
    abs_rpy = absolute_pose[3:]
    # abs_rpy[:2] = - abs_rpy[:2]
    
    rel_xyz = relative_pose[:3] # Translation, defined in abs_pose coordinate system
    rel_rpy = relative_pose[3:] # Rotation, from abs_pose to new_pose

    # 2. Handle rotation
    try:
        # Convert absolute RPY to Rotation object
        rot_abs = R.from_euler(EULER_CONVENTION, abs_rpy)
        
        # Convert relative RPY to Rotation object
        rot_rel = R.from_euler(EULER_CONVENTION, rel_rpy)
    except ValueError as e:
        print(f"Error creating Rotation object. Check EULER_CONVENTION ('{EULER_CONVENTION}') and RPY values.")
        print(f"Absolute RPY: {abs_rpy}")
        print(f"Relative RPY: {rel_rpy}")
        raise e

    # Calculate new absolute rotation: combined rotation R_new = R_abs * R_rel
    # Note: Scipy Rotation's * operator performs rotation composition
    rot_new_abs = rot_abs * rot_rel

    # Convert new absolute rotation back to RPY
    new_abs_rpy = rot_new_abs.as_euler(EULER_CONVENTION)
    # new_abs_rpy[:2] = - new_abs_rpy[:2]

    # 3. Handle position
    # Rotate relative displacement (rel_xyz) from abs_pose coordinate system to world coordinate system
    world_delta_xyz = rot_abs.apply(rel_xyz)

    # Add world coordinate displacement to absolute position
    new_abs_xyz = abs_xyz + world_delta_xyz

    # 4. Combine new absolute pose
    new_absolute_pose = np.concatenate((new_abs_xyz, new_abs_rpy))

    return new_absolute_pose

def log_control_info(robot: Robot, dt_s, episode_index=None, frame_index=None, fps=None):
    log_items = []
    if episode_index is not None:
        log_items.append(f"ep:{episode_index}")
    if frame_index is not None:
        log_items.append(f"frame:{frame_index}")

    def log_dt(shortname, dt_val_s):
        nonlocal log_items, fps
        info_str = f"{shortname}:{dt_val_s * 1000:5.2f} ({1/ dt_val_s:3.1f}hz)"
        if fps is not None:
            actual_fps = 1 / dt_val_s
            if actual_fps < fps - 1:
                info_str = colored(info_str, "yellow")
        log_items.append(info_str)

    # total step time displayed in milliseconds and its frequency
    log_dt("dt", dt_s)

    # TODO(aliberts): move robot-specific logs logic in robot.print_logs()
    if not robot.robot_type.startswith("stretch"):
        for name in robot.leader_arms:
            key = f"read_leader_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRlead", robot.logs[key])

        for name in robot.follower_arms:
            key = f"write_follower_{name}_goal_pos_dt_s"
            if key in robot.logs:
                log_dt("dtWfoll", robot.logs[key])

            key = f"read_follower_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRfoll", robot.logs[key])

        for name in robot.cameras:
            key = f"read_camera_{name}_dt_s"
            if key in robot.logs:
                log_dt(f"dtR{name}", robot.logs[key])

    info_str = " ".join(log_items)
    logging.info(info_str)


@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


def has_method(_object: object, method_name: str):
    return hasattr(_object, method_name) and callable(getattr(_object, method_name))


def predict_action(observation, policy, device, use_amp):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # based on the current observation
        action = policy.select_action(observation)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action


def predict_action_gr00t(observation, policy, device, use_amp, action_space='abs', video_mode: str = 'dual'):
    observation = copy(observation)
    
    img_webcam = cv2.cvtColor(observation["observation.images.webcam"].numpy(), cv2.COLOR_BGR2RGB)
    img_phone = None
    video_count = 0
    if img_webcam is not None:
        video_count += 1
    if video_mode == 'dual':
        if "observation.images.phone" in observation and observation["observation.images.phone"] is not None:
            img_phone = cv2.cvtColor(observation["observation.images.phone"].numpy(), cv2.COLOR_BGR2RGB)
            if img_phone is not None:
                video_count += 1
        else:
            img_phone = None
    elif video_mode == 'webcam':
        img_phone = None
    else:
        img_phone = None

    
    action = policy.get_action(img_webcam, img_phone, observation["observation.state"].numpy(), action_space)

    # # Remove batch dimension
    # action = action.squeeze(0)

    # # Move to cpu, if not already the case
    # action = action.to("cpu")

    return action


def init_keyboard_listener():
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def init_policy(pretrained_policy_name_or_path, policy_overrides):
    """Instantiate the policy and load fps, device and use_amp from config yaml"""
    pretrained_policy_path = get_pretrained_policy_path(pretrained_policy_name_or_path)
    hydra_cfg = init_hydra_config(pretrained_policy_path / "config.yaml", policy_overrides)
    policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=pretrained_policy_path)

    # Check device is available
    device = get_safe_torch_device(hydra_cfg.device, log=True)
    use_amp = hydra_cfg.use_amp
    policy_fps = hydra_cfg.env.fps

    policy.eval()
    policy.to(device)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_global_seed(hydra_cfg.seed)
    return policy, policy_fps, device, use_amp

import time
def reset_robot_to_init_gpos(robot):
    # grip qpos:[-0.157,1.5]-> distance:[12.48,137.34]
    INIT_GPOS = np.array([0.21, 0.000, 0.17608294, 1.55464093, -0.02126498, 0.10906027, 12.])

    for _ in range(1000):
        robot.send_action(torch.from_numpy(INIT_GPOS))
        time.sleep(0.01)
    

def warmup_record(
    robot,
    events,
    enable_teleoperation,
    warmup_time_s,
    display_cameras,
    fps,
):
    control_loop(
        robot=robot,
        control_time_s=warmup_time_s,
        display_cameras=display_cameras,
        events=events,
        fps=fps,
        teleoperate=enable_teleoperation,
    )


def record_episode(
    robot,
    dataset,
    events,
    episode_time_s,
    display_cameras,
    policy,
    device,
    use_amp,
    fps,
):
    control_loop(
        robot=robot,
        control_time_s=episode_time_s,
        display_cameras=display_cameras,
        dataset=dataset,
        events=events,
        policy=policy,
        device=device,
        use_amp=use_amp,
        fps=fps,
        teleoperate=policy is None,
    )


@safe_stop_image_writer
def control_loop(
    robot,
    control_time_s=None,
    teleoperate=False,
    display_cameras=False,
    dataset: LeRobotDataset | None = None,
    events=None,
    policy=None,
    device=None,
    use_amp=None,
    fps=None,
):
    # TODO(rcadene): Add option to record logs
    if not robot.is_connected:
        robot.connect()

    if events is None:
        events = {"exit_early": False}

    if control_time_s is None:
        control_time_s = float("inf")

    if teleoperate and policy is not None:
        raise ValueError("When `teleoperate` is True, `policy` should be None.")

    if dataset is not None and fps is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset['fps']} != {fps}).")

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if teleoperate:
            observation, action = robot.teleop_step(record_data=True)
        else:
            observation, current_state = robot.capture_observation()
            # current_state = observation["observation.state"].numpy()

            if policy is not None:
                # video_mode can be threaded from robot configuration if desired; default to 'dual'
                pred_action = predict_action_gr00t(
                    observation,
                    policy,
                    device,
                    use_amp,
                    robot.action_space,
                    video_mode='dual',
                )
                # print(f"pred_action: {pred_action}")
                # print(f"sim_state: {sim_state}")
                # breakpoint()
                # Action can eventually be clipped using `max_relative_target`,
                # so action actually sent is saved in the dataset.
                ACTION_HORIZON = 12
                MODALITY_KEYS = ["single_arm_eef_xyz", "single_arm_eef_rpy", "gripper"]
                # print(f"action: {pred_action}")
                for i in range(ACTION_HORIZON):
                    concat_action = np.concatenate(
                        [np.atleast_1d(pred_action[f"action.{key}"][i]) for key in MODALITY_KEYS],
                        axis=0,
                    )
                    if "relative" in robot.action_space:
                        print(f"current_state: {current_state}")
                        print(f"relative_action: {concat_action}")
                        next_gpos = apply_relative_transform(current_state[:6].copy(), concat_action[:6].copy())
                        next_action = np.concatenate((next_gpos, concat_action[6:]), axis=0)
                        # next_action = current_state
                        print(f"next_action: {next_action}")
                        assert concat_action.shape == (7,), concat_action.shape
                        breakpoint()
                        action = robot.send_action(torch.from_numpy(next_action))
                    else:
                        # print(f"concat_action: {concat_action}")
                        # breakpoint()
                        action = robot.send_action(torch.from_numpy(concat_action))

                    # breakpoint()
                    observation, _ = robot.capture_observation()
                    time.sleep(0.05)

                    # action = robot.send_action(pred_action)
                    action = {"action": action}

                    if dataset is not None:
                        frame = {**observation, **action}
                        dataset.add_frame(frame)

                    if display_cameras and not is_headless():
                        image_keys = [key for key in observation if "image" in key]
                        for key in image_keys:
                            cv2.imshow(key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)

        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        log_control_info(robot, dt_s, fps=fps)

        timestamp = time.perf_counter() - start_episode_t
        if events["exit_early"] or robot.button_control != 0:
            events["exit_early"] = False
            break


def reset_environment(robot, events, reset_time_s):
    # TODO(rcadene): refactor warmup_record and reset_environment
    # TODO(alibets): allow for teleop during reset
    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    timestamp = 0
    start_vencod_t = time.perf_counter()

    # Wait if necessary
    with tqdm.tqdm(total=reset_time_s, desc="Waiting") as pbar:
        while timestamp < reset_time_s:
            time.sleep(1)
            timestamp = time.perf_counter() - start_vencod_t
            pbar.update(1)
            if events["exit_early"]:
                events["exit_early"] = False
                break


def stop_recording(robot, listener, display_cameras):
    robot.disconnect()

    if not is_headless():
        if listener is not None:
            listener.stop()

        if display_cameras:
            cv2.destroyAllWindows()


def sanity_check_dataset_name(repo_id, policy):
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, use_videos: bool
) -> None:
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, get_features_from_robot(robot, use_videos)),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )
