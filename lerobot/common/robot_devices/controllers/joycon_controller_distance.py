import logging
import math
import struct
import threading
import time
import os

from joyconrobotics import JoyconRobotics
import numpy as np

from lerobot_kinematics import lerobot_IK, lerobot_FK, get_robot
from joyconrobotics import JoyconRobotics
import mujoco

np.set_printoptions(linewidth=200)
os.environ["MUJOCO_GL"] = "egl"
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
JOINT_NAMES_PLUS = ['shoulder_rotation_joint', 'shoulder_pitch_joint', 'ellbow_joint', 'wrist_pitch_joint', 'wrist_jaw_joint',  'wrist_roll_joint', 'gripper_joint']
class JoyConController:
    def __init__(
        self,
        name,
        initial_position=None,
        timeout = 60,
        *args,
        **kwargs,
    ):
        self.name = name
        self.xml_path = "./lerobot/common/robot_devices/controllers/scene.xml"
        self.mjmodel = mujoco.MjModel.from_xml_path(self.xml_path)
        self.qpos_indices = np.array([self.mjmodel.jnt_qposadr[self.mjmodel.joint(n).id] for n in JOINT_NAMES])
        self.mjdata = mujoco.MjData(self.mjmodel)
        
        self.robot = get_robot('so100')
        self.glimit = [[0.125, -0.4,  0.046, -3.1, -1.5, -1.5], 
                       [0.380,  0.4,  0.23,  3.1,  1.5,  1.5]]
        
        self.init_qpos = np.array([0.0, -3.14, 3.14, 0.0, -1.57, -0.157])
        self.target_qpos = self.init_qpos.copy() 
        self.init_gpos = lerobot_FK(self.init_qpos[1:5], robot=self.robot)
        self.target_gpos = self.init_gpos.copy() 
        
        self.mjdata.qpos[self.qpos_indices] = self.init_qpos   
        mujoco.mj_step(self.mjmodel, self.mjdata) 
        
        self.max_retries = timeout
        self.retries = 0
        self.offset_position_m = self.init_gpos[0:3]
        while self.retries < self.max_retries:
            try:
                self.joyconrobotics = JoyconRobotics(
                    device=name, 
                    horizontal_stick_mode='yaw_diff', 
                    close_y=True, 
                    limit_dof=True, 
                    offset_position_m=self.offset_position_m, 
                    glimit=self.glimit,
                    dof_speed=[1.0, 1.0, 1.0, 1.0, 1.0, 0.5], 
                    common_rad=False,
                    lerobot=True,
                    pitch_down_double=True
                )
                break  # 连接成功，跳出循环
            except Exception as e:
                self.retries += 1
                print(f"Failed to connect to {name} Joycon: {e}")
                print(f"Retrying ({self.retries}/{self.max_retries}) in 1 second...")
                time.sleep(1)  # 连接失败后等待 1 秒重试
        else:
            print("Failed to connect after several attempts.")
        
        self.target_gpos_last = self.init_gpos.copy() 
        self.joint_angles_last = self.init_qpos.copy() 
        
    def get_command(self, present_pose):
        
        target_pose, gripper_state, button_control = self.joyconrobotics.get_control()
        # print("target_pose:", [f"{x:.3f}" for x in target_pose])
        
        for i in range(6):
            if target_pose[i] < self.glimit[0][i]:
                target_pose[i] = self.glimit[0][i]  
            elif target_pose[i] > self.glimit[1][i]:
                target_pose[i] = self.glimit[1][i]  
            else:
                 target_pose[i]
                 
        x = target_pose[0] # init_gpos[0] + 
        z = target_pose[2] # init_gpos[2] + 
        _, _, _, roll, pitch, yaw = target_pose
        y = 0.01
        pitch = -pitch 
        roll = roll - math.pi/2 # lerobo末端旋转90度
        
        # 双臂朝中间偏
        # if self.name == 'left':
        #     yaw = yaw - 0.4
        # elif self.name == 'right':
        #     yaw = yaw + 0.4
        
        target_gpos = np.array([x, y, z, roll, pitch, 0.0])
        fd_qpos_mucojo = self.mjdata.qpos[self.qpos_indices][1:5]
        
        qpos_inv_mujoco, IK_success = lerobot_IK(fd_qpos_mucojo, target_gpos, robot=self.robot)
        if IK_success:
            self.target_qpos = np.concatenate(([yaw,], qpos_inv_mujoco[:4], [gripper_state,])) 

            self.mjdata.qpos[self.qpos_indices] = self.target_qpos
            mujoco.mj_step(self.mjmodel, self.mjdata)
            
            self.target_gpos_last = target_gpos.copy() 
            joint_angles = self.target_qpos
            
            joint_angles = np.rad2deg(self.target_qpos)
            joint_angles[1] = -joint_angles[1]
            joint_angles[0] = -joint_angles[0]
            joint_angles[4] = -joint_angles[4]
            self.joint_angles_last = joint_angles.copy() 

        else:
            self.target_gpos = self.target_gpos_last.copy()
            self.joyconrobotics.set_position = self.target_gpos[0:3]
            joint_angles = self.joint_angles_last
        
        # if button_control != 0:
        #     self.joyconrobotics.reset_joycon()
            
        return joint_angles, button_control
    

from scipy.spatial.transform import Rotation as R # 引入 Rotation

class JoyConController_plus:
    def __init__(
        self,
        name,
        initial_position=None,
        timeout = 60,
        *args,
        **kwargs,
    ):
        self.name = name
        self.xml_path = "./lerobot/common/robot_devices/controllers/scene_plus.xml"
        self.mjmodel = mujoco.MjModel.from_xml_path(self.xml_path)
        self.qpos_indices = np.array([self.mjmodel.jnt_qposadr[self.mjmodel.joint(n).id] for n in JOINT_NAMES_PLUS])
        self.mjdata = mujoco.MjData(self.mjmodel)
        
        self.robot = get_robot('so100_plus')
        self.glimit = [[0.210, -0.4, -0.047, -3.1, -1.45, -1.5], 
                       [0.420,  0.4,  0.30,   3.1,  1.45,  1.5]]
        
        self.init_qpos = np.array([0.0, -3.1, 3.0, 0.0, 0.0, 1.57, -0.157])
        self.target_qpos = self.init_qpos.copy() 
        self.init_gpos = lerobot_FK(self.init_qpos[:6], robot=self.robot)
        self.target_gpos = self.init_gpos.copy()
        # print(f"self.target_gpos: {self.target_gpos}")

        # gripper_state = self.target_gpos[-1]
        # fd_qpos_mucojo = self.mjdata.qpos[self.qpos_indices][:6]
        # qpos_inv_mujoco, IK_success = lerobot_IK(fd_qpos_mucojo, self.target_gpos[:6], robot=self.robot)
        # if IK_success:
        #     self.target_qpos = np.concatenate((qpos_inv_mujoco[:6], [gripper_state,]))
        self.target_gpos_for_ik = [0.21772567056163034, 0.0, 0.2, 0.0, 0.0, 0.0]
        
        self.mjdata.qpos[self.qpos_indices] = self.init_qpos   
        mujoco.mj_step(self.mjmodel, self.mjdata) 
        
        self.max_retries = timeout
        self.retries = 0
        self.offset_position_m = self.init_gpos[0:3]
        while self.retries < self.max_retries:
            try:
                self.joyconrobotics = JoyconRobotics(
                                      device=self.name, 
                                      glimit = self.glimit,
                                      offset_position_m=self.offset_position_m, 
                                      pitch_down_double = True,
                                      gripper_close = -0.15,
                                      gripper_open = 0.5
                                      )
                break  # 连接成功，跳出循环
            except Exception as e:
                self.retries += 1
                print(f"Failed to connect to {name} Joycon: {e}")
                print(f"Retrying ({self.retries}/{self.max_retries}) in 1 second...")
                time.sleep(1)  # 连接失败后等待 1 秒重试
        else:
            print("Failed to connect after several attempts.")
        
        self.target_gpos_last = self.init_gpos.copy()  
        self.joint_angles_last = self.init_qpos.copy()     

        self.gripper_length = 40.11 # mm

    def set_mujoco_pos(self, pos):
        self.mjdata.qpos[self.qpos_indices] = pos
        mujoco.mj_step(self.mjmodel, self.mjdata)

    def get_sim_eef(self):
        qpos = self.mjdata.qpos[self.qpos_indices]
        eef_pos = lerobot_FK(qpos[:6], robot=self.robot)
        distance = 75.38 * qpos[-1] + 24.31 # mm为单位
        # gripper_state = qpos[-1]
        eef_state = np.concatenate((eef_pos, [distance,]))
        return eef_state

    def get_eef(self, joint_angles, action_space="eef_rpy"):   # "aa"
        joint_angles[0] = -joint_angles[0]  # 第0个关节角度再次取反
        joint_angles[1] = -joint_angles[1]  # 第1个关节角度再次取反
        joint_angles[3] = -joint_angles[3]  # 第3个关节角度再次取反
        qpos = np.deg2rad(joint_angles)
        distance = 75.38 * qpos[-1] + 24.31 # mm为单位
        # gripper_state = qpos[-1]
        eef_pos = lerobot_FK(qpos[:6], robot=self.robot)
        # if "aa" in action_space:
        #     eef_pos[-3:] = rpy2aa(eef_pos[-3:])
        if "eef_rpy_midpoint" in action_space:
            eef_pos = self.get_midpoint(eef_pos, qpos[-1], L=self.gripper_length)  # mm为单位
        eef_state = np.concatenate((eef_pos, [distance,]))
        
        return eef_state

    def eef_action_to_joint_angles(self, eef_action, action_space="eef_rpy"):
        eef_action = np.asarray(eef_action)
        target_gpos = eef_action[:6]
        # if "aa" in action_space:
        #     target_gpos[-3:] = aa2rpy(target_gpos[-3:])

        x, y, z, roll, pitch, yaw = target_gpos
        pitch_transformed = -pitch
        roll_transformed = -(roll - math.pi/2)

        pose_transformed_for_limiting = np.array([x, y, z, roll_transformed, pitch_transformed, yaw])

        # 5. 应用限制
        pose_limited_transformed = np.zeros_like(pose_transformed_for_limiting)
        for i in range(6):
            pose_limited_transformed[i] = np.clip(
                pose_transformed_for_limiting[i],
                self.glimit[0][i],
                self.glimit[1][i]
            )

        x_lim_t, y_lim_t, z_lim_t, roll_lim_t, pitch_lim_t, yaw_lim_t = pose_limited_transformed

        # 7. 逆变换 (恢复 IK 期望的 RPY 格式)
        # 逆变换 pitch: pitch_original = -pitch_transformed
        pitch_limited_original = -pitch_lim_t
        # 逆变换 roll: roll_original = -roll_transformed + math.pi/2
        roll_limited_original = -(roll_lim_t) + math.pi/2

        target_gpos = np.array([
            x_lim_t, y_lim_t, z_lim_t,
            roll_limited_original, pitch_limited_original, yaw_lim_t # 使用调整后的 yaw
        ])

        distance = eef_action[-1]
        gripper_state = (distance - 24.31) / 75.38

        # 获取夹爪角度的回归
        gripper_angles = self.get_gripper_angles(distance)

        # gripper_state = eef_action[-1]

        # 从夹爪中点计算末端执行器位置
        if "eef_rpy_midpoint" in action_space:
            target_gpos = self.get_midpoint(target_gpos, gripper_angles, L=self.gripper_length, inverse=True)

        else:

            fd_qpos_mucojo = self.mjdata.qpos[self.qpos_indices][:6]
            
            qpos_inv_mujoco, IK_success = lerobot_IK(fd_qpos_mucojo, target_gpos, robot=self.robot)
            
            if IK_success:
                
                self.target_qpos = np.concatenate((qpos_inv_mujoco[:6], [gripper_state,])) 
                # print(f'{self.target_qpos=}')
                self.mjdata.qpos[self.qpos_indices] = self.target_qpos
                mujoco.mj_step(self.mjmodel, self.mjdata)
                self.target_gpos_last = target_gpos.copy() 

                joint_angles = np.rad2deg(self.target_qpos)
                joint_angles[0] = -joint_angles[0]
                joint_angles[1] = -joint_angles[1]
                joint_angles[3] = -joint_angles[3]
                self.joint_angles_last = joint_angles.copy()

            else:
                self.target_gpos = self.target_gpos_last.copy()
                joint_angles = self.joint_angles_last

            return joint_angles
    
    # def pos_to_mujoco(self, pos):
    
    def get_command(self, present_pose, action_space="eef_rpy"):    # present_pose以joint_angles形式给出
        
        target_pose, gripper_state, button_control = self.joyconrobotics.get_control()
        # print("target_pose:", [f"{x:.3f}" for x in target_pose])
        
        for i in range(6):
            if target_pose[i] < self.glimit[0][i]:
                target_pose[i] = self.glimit[0][i]  
            elif target_pose[i] > self.glimit[1][i]:
                target_pose[i] = self.glimit[1][i]  
            else:
                 target_pose[i]
                 
        x = target_pose[0] # init_gpos[0] + 
        y = target_pose[1]
        z = target_pose[2] # init_gpos[2] + 
        _, _, _, roll, pitch, yaw = target_pose
        pitch = -pitch 
        roll = -(roll - math.pi/2) # lerobo末端旋转90度
        
        target_gpos = np.array([x, y, z, roll, pitch, yaw])

        # 遥操作时有点问题，控制夹爪时，夹爪角度会变化，导致中点计算偏差，
        # 原本逻辑，控制末端执行其的位置和姿态，然后控制夹爪状态
        # 现在逻辑，控制夹爪中点位置和姿态，但是夹爪状态是独立的现在耦合在一起了！！！！！
        if "eef_rpy_midpoint" in action_space:
            gripper_angles = 1
            target_gpos = self.get_midpoint(target_gpos, gripper_angles, L=self.gripper_length, inverse=True)
        
        print(f"target_gpos: {target_gpos}")
        fd_qpos_mucojo = self.mjdata.qpos[self.qpos_indices][:6]
        # print(self.mjdata.qpos[self.qpos_indices])
        
        qpos_inv_mujoco, IK_success = lerobot_IK(fd_qpos_mucojo, target_gpos, robot=self.robot)
        
        if IK_success:
            
            self.target_qpos = np.concatenate((qpos_inv_mujoco[:6], [gripper_state,])) 
            # print("target_qpos", self.target_qpos)
            self.mjdata.qpos[self.qpos_indices] = self.target_qpos
            mujoco.mj_step(self.mjmodel, self.mjdata)
            self.target_gpos_last = target_gpos.copy() 

            joint_angles = np.rad2deg(self.target_qpos)
            joint_angles[0] = -joint_angles[0]
            joint_angles[1] = -joint_angles[1]
            joint_angles[3] = -joint_angles[3]
            self.joint_angles_last = joint_angles.copy()

            distance = 75.38 * gripper_state + 24.31 # mm为单位
            target_gpos = np.concatenate((target_gpos, [distance,]))

        else:
            self.target_gpos = self.target_gpos_last.copy()
            self.joyconrobotics.set_position = self.target_gpos[0:3]
            joint_angles = self.joint_angles_last

            distance = 75.38 * gripper_state + 24.31 # mm为单位
            target_gpos = np.concatenate((self.target_gpos, [distance,]))
            
        return joint_angles, button_control, target_gpos
    
    def get_delta_command(self, present_pose_angles_deg):
        """
        根据 Joycon 的 delta 输入计算新的目标关节角度。

        Args:
            present_pose_angles_deg (np.ndarray): 当前机械臂的关节角度 (单位: 度),
                                                 格式应与 self.joint_angles_last 一致 (带符号翻转)。

        Returns:
            tuple: (joint_angles, button_control, target_gpos_with_distance)
                   - joint_angles (np.ndarray): 计算出的目标关节角度 (度, 带符号翻转)。
                   - button_control (int): 来自 Joycon 的按钮控制状态。
                   - target_gpos_with_distance (np.ndarray): IK 尝试的目标位姿 (x,y,z,r,p,y in rad)
                                                            加上计算出的夹爪距离 (mm)。
                                                            如果 IK 失败，则是上一次成功的目标位姿。
        """
        # 1. FK: 从当前关节角度 (度, 带翻转) 计算当前 EEF 位姿 (米, 弧度)
        current_angles_deg_flipped = np.array(present_pose_angles_deg)
        # 准备 FK 输入：反转符号，转为弧度
        qpos_rad_for_fk = current_angles_deg_flipped.copy()
        qpos_rad_for_fk[0] = -qpos_rad_for_fk[0] # 恢复原始符号
        qpos_rad_for_fk[1] = -qpos_rad_for_fk[1] # 恢复原始符号
        qpos_rad_for_fk[3] = -qpos_rad_for_fk[3] # 恢复原始符号
        qpos_rad_for_fk = np.deg2rad(qpos_rad_for_fk[:6]) # 取前6个，转为弧度

        current_eef_pose_rad = lerobot_FK(qpos_rad_for_fk, robot=self.robot) # (x,y,z, r,p,y in radians)
        print('-' * 50)
        print("current_eef_pose_rad", current_eef_pose_rad)
        # current_eef_pose_rad = self.target_gpos_for_ik
        current_pos_w = current_eef_pose_rad[:3]
        current_orient_rad_w = current_eef_pose_rad[3:] # [roll, pitch, yaw]

        # 2. 获取 Delta Action (工具坐标系)
        # delta_pose = [dx_t, dy_t, dz_t, droll_t, dpitch_t, dyaw_t]
        delta_pose, gripper_state, button_control = self.joyconrobotics.get_control(return_delta=True)

        print("delta_pose", delta_pose)

        delta_pose = np.array([0.0, 0.0, 0.01, 0, 0, 0])
        print("curr delta_pose", delta_pose)
        
        delta_pos_t = np.array(delta_pose[:3])
        delta_orient_rad_t = np.array(delta_pose[3:]) # [droll, dpitch, dyaw] 绕工具轴

        distance = 75.38 * gripper_state + 24.31 # mm为单位
        print(delta_pose)
        print(distance)
        delta_action = np.concatenate((delta_pose, [distance,]))

        # 3. 计算新的目标 EEF 位姿 (世界坐标系)

        # --- a. 处理旋转 ---
        # 将当前世界姿态和工具 delta 姿态转换为 Rotation 对象
        # !!关键: 确认 FK 和 IK 使用的欧拉角顺序 ('xyz', 'zyx', etc.)
        # 这里假设是 'xyz'
        R_world_tool_current = R.from_euler('xyz', current_orient_rad_w, degrees=False)

        # 将工具坐标系下的 delta 旋转也看作是 'xyz' 欧拉角序列定义的旋转
        # 这是一个内在旋转 (intrinsic rotation)
        R_tool_delta = R.from_euler('xyz', delta_orient_rad_t, degrees=False)
        # 组合旋转：先应用当前旋转，再应用工具坐标系下的 delta 旋转
        R_world_tool_new = R_world_tool_current * R_tool_delta
        # 获取新的世界坐标系下的欧拉角
        # !!关键: 确保输出的欧拉角顺序与 IK 需要的一致
        next_orient_rad_w = R_world_tool_new.as_euler('xyz', degrees=False)

        # --- b. 处理平移 ---
        # 将工具坐标系下的 delta 平移转换到世界坐标系
        # delta_pos_w = R_world_tool_current.apply(delta_pos_t) # R_current * delta_t
        # !! 注意： R.apply 是将向量从“本体坐标系”转换到“外部坐标系”
        # 如果 delta_pos_t 是在 tool frame 下的向量，我们需要用 R_world_tool_current 将其旋转到 world frame
        delta_pos_w = R_world_tool_current.apply(delta_pos_t)
        # 计算新的世界坐标系下的位置
        next_pos_w = current_pos_w + delta_pos_w

        # --- 组合新的目标位姿 ---
        next_target_eef_pose_rad = np.concatenate((next_pos_w, next_orient_rad_w))

        # 4. 应用变换和限制 (为 IK 做准备)
        # 注意：这里的变换和限制逻辑现在应用于 next_target_eef_pose_rad (世界坐标系)
        # 如果你的限制 self.glimit 是定义在某个变换后的坐标系下，你需要相应调整
        # 以下代码假设变换和限制仍按原样应用于世界坐标系的 r,p,y
        x, y, z = next_target_eef_pose_rad[:3]
        roll, pitch, yaw = next_target_eef_pose_rad[3:] # 世界坐标系的 roll, pitch, yaw

        # # a. 坐标变换 (如果你的 IK 或限位需要特定变换)
        # pitch_transformed = -pitch
        # roll_transformed = -(roll - math.pi/2)

        # pose_transformed_for_limiting = np.array([x, y, z, roll_transformed, pitch_transformed, yaw])

        # # b. 应用限制 (在变换后的坐标系下)
        # pose_limited_transformed = np.zeros_like(pose_transformed_for_limiting)
        # for i in range(6):
        #     pose_limited_transformed[i] = np.clip(
        #         pose_transformed_for_limiting[i],
        #         self.glimit[0][i],
        #         self.glimit[1][i]
        #     )

        # x_lim_t, y_lim_t, z_lim_t, roll_lim_t, pitch_lim_t, yaw_lim_t = pose_limited_transformed

        # # c. 逆变换 (恢复 IK 期望的 RPY 格式 - 世界坐标系)
        # pitch_limited_original = -pitch_lim_t
        # roll_limited_original = -(roll_lim_t) + math.pi/2

        # target_gpos_for_ik = np.array([
        #     x_lim_t, y_lim_t, z_lim_t,
        #     roll_limited_original, pitch_limited_original, yaw_lim_t
        # ])
        # # print(f"Target gpos for IK (rad): {[f'{v:.3f}' for v in target_gpos_for_ik]}")
        target_gpos_for_ik = [x, y, z, roll, pitch, yaw]

        # target_gpos_for_ik = np.array([0.21186902, 0.00181367, 0.17290198, 1.533, 0.026, 0.018])
        # target_gpos_for_ik = np.array([0.21186902, 0.00181367, 0.17290198, 1.533, 0.026, 0.018])

        # target_gpos_for_ik = [ 0.21, 0.0, 0.17608294, 1.55084124, -0.00889929, 0.00473278]
        # target_gpos_for_ik = [ 0.20, 0.0, 0.17608294, 1.35084124, -0.00889929, 0.00473278]
        # target_gpos_for_ik = [ 0.21, 0.0, 0.17608294, 1.55084124, -0.00889929, 0.00473278]
        # target_gpos = [ 0.41, 0.0, 0.17608294, 1.55084124, -0.00889929, 0.00473278] # x
        # target_gpos = [ 0.21, 0.2, 0.17608294, 1.55084124, -0.00889929, 0.00473278] # y
        # target_gpos = [ 0.21, 0.0, 0.37608294, 1.55084124, -0.00889929, 0.00473278] # z
        # target_gpos = [ 0.21, 0.0, 0.17608294, 1.85084124, -0.00889929, 0.00473278] # roll
        # target_gpos = [ 0.21, 0.0, 0.17608294, 1.55084124, 0.29110071, 0.00473278] # pitch
        # target_gpos = [ 0.21, 0.0, 0.17608294, 1.55084124, -0.00889929, 0.30473278] # yaw

        self.target_gpos_for_ik = target_gpos_for_ik.copy()
        # target_gpos_for_ik = [0.21772567056163034, 0.0, 0.3, 0.0, 0.0, 0.0]
        target_gpos_for_ik = [0.26, 0.0, 0.27, 0.0, 0.0, 0.0]
        print("current_eef_pose_rad", current_eef_pose_rad)
        print("target_gpos_for_ik", target_gpos_for_ik)

        # 5. 执行逆运动学 (IK)
        fd_qpos_mucojo = self.mjdata.qpos[self.qpos_indices][:6]
        qpos_inv_mujoco, IK_success = lerobot_IK(fd_qpos_mucojo, target_gpos_for_ik, robot=self.robot)

        # 6. 处理 IK 结果 (与之前逻辑相同)
        if IK_success:
            # print("IK Success")
            self.target_qpos = np.concatenate((qpos_inv_mujoco[:6], [gripper_state,]))
            # print(f"New target_qpos (rad): {[f'{v:.3f}' for v in self.target_qpos]}")
            self.mjdata.qpos[self.qpos_indices] = self.target_qpos
            mujoco.mj_step(self.mjmodel, self.mjdata)
            self.target_gpos_last = target_gpos_for_ik.copy()
            joint_angles = np.rad2deg(self.target_qpos)
            joint_angles[0] = -joint_angles[0]
            joint_angles[1] = -joint_angles[1]
            joint_angles[3] = -joint_angles[3]
            self.joint_angles_last = joint_angles.copy()
            distance = 75.38 * gripper_state + 24.31
            target_gpos_output = np.concatenate((target_gpos_for_ik, [distance,]))
        else:
            # print("IK Failed")
            joint_angles = self.joint_angles_last.copy()
            distance = 75.38 * gripper_state + 24.31
            target_gpos_output = np.concatenate((self.target_gpos_last, [distance,]))

        breakpoint()
        return joint_angles, button_control, delta_action
    
    def get_midpoint(self, eef_pos, grip_angles, L=10, inverse=False): 
        """
        计算机械臂末端执行器位置加上一个固定杆后的端点位置，或进行逆运算。

        Args:
            eef_pos (np.ndarray): 末端执行器位姿 (x, y, z, roll, pitch, yaw) 或端点位姿 (当inverse=True时)
            grip_angles (np.ndarray): 夹爪角度 in radians
            L (float): 固定杆的长度，单位为米
            inverse (bool): 是否进行逆运算，True表示从端点位置计算执行器位置

        Returns:
            np.ndarray: 杆端点位置和姿态 [x, y, z, roll, pitch, yaw] (当inverse=False时)
                       或执行器位置和姿态 [x, y, z, roll, pitch, yaw] (当inverse=True时)
        """
        # 从欧拉角创建旋转矩阵 (使用ZYX顺序)
        R_final = R.from_euler('zyx', eef_pos[3:], degrees=False)

        L_effective = L * math.cos(grip_angles)
        
        # 定义杆在局部坐标系下的位移向量
        rod_local = np.array([0, 0, L_effective])
        
        if inverse:
            # 逆运算：从端点位置计算执行器位置
            # 将位移转换到全局坐标系并从端点位置减去
            rod_displacement = R_final.apply(rod_local)
            eef_tip_pos = eef_pos[:3] - rod_displacement
            
            # 姿态保持不变
            eef_tip_angles = eef_pos[3:]    
            
            # 返回执行器的完整位姿
            return np.concatenate([eef_tip_pos, eef_tip_angles])
        else:
            # 正运算：从执行器位置计算端点位置
            # 将位移转换到全局坐标系并加上末端执行器位置
            rod_displacement = R_final.apply(rod_local)
            rod_tip_pos = eef_pos[:3] + rod_displacement
            
            # 姿态保持不变
            rod_tip_angles = eef_pos[3:]    
            
            # 返回杆端点的完整位姿
            return np.concatenate([rod_tip_pos, rod_tip_angles])
    
    # 从夹爪距离计算夹爪角度，参数待计算
    def get_gripper_angles(self, gripper_distance):
        # 夹爪距离与夹爪角度的关系
        # 夹爪距离 = 75.38 * 夹爪角度 + 24.31
        # 夹爪角度 = (夹爪距离 - 24.31) / 75.38
        return (gripper_distance - 24.31) / 75.38