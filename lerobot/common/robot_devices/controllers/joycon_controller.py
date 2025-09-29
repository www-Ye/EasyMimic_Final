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

    def set_mujoco_pos(self, pos):
        self.mjdata.qpos[self.qpos_indices] = pos
        mujoco.mj_step(self.mjmodel, self.mjdata)

    def get_sim_eef(self):
        qpos = self.mjdata.qpos[self.qpos_indices]
        eef_pos = lerobot_FK(qpos[:6], robot=self.robot)
        gripper_state = qpos[-1]
        eef_state = np.concatenate((eef_pos, [gripper_state,]))
        return eef_state

    def get_eef(self, joint_angles, action_space="eef_rpy"):   # "aa"
        joint_angles[0] = -joint_angles[0]  # 第0个关节角度再次取反
        joint_angles[1] = -joint_angles[1]  # 第1个关节角度再次取反
        joint_angles[3] = -joint_angles[3]  # 第3个关节角度再次取反
        qpos = np.deg2rad(joint_angles)
        # distance = 75.38 * qpos[-1] + 24.31 # mm为单位
        gripper_state = qpos[-1]
        eef_pos = lerobot_FK(qpos[:6], robot=self.robot)
        # if "aa" in action_space:
        #     eef_pos[-3:] = rpy2aa(eef_pos[-3:])
        eef_state = np.concatenate((eef_pos, [gripper_state,]))
        
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

        # distance = eef_action[-1]
        # gripper_state = (distance - 24.31) / 75.38
        gripper_state = eef_action[-1]

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
        y = target_pose[1]
        z = target_pose[2] # init_gpos[2] + 
        _, _, _, roll, pitch, yaw = target_pose
        pitch = -pitch 
        roll = -(roll - math.pi/2) # lerobo末端旋转90度
        
        # 双臂朝中间偏
        # if self.name == 'left':
        #     yaw = yaw + 0.4
        # elif self.name == 'right':
        # yaw = yaw + 1.57
        
        target_gpos = np.array([x, y, z, roll, pitch, yaw])
        
        # target_gpos = [ 0.21, 0.0, 0.17608294, 1.55084124, -0.00889929, 0.00473278]
        # target_gpos = [ 0.41, 0.0, 0.17608294, 1.55084124, -0.00889929, 0.00473278] # x
        # target_gpos = [ 0.21, 0.2, 0.17608294, 1.55084124, -0.00889929, 0.00473278] # y
        # target_gpos = [ 0.21, 0.0, 0.37608294, 1.55084124, -0.00889929, 0.00473278] # z
        # target_gpos = [ 0.21, 0.0, 0.17608294, 1.85084124, -0.00889929, 0.00473278] # roll
        # target_gpos = [ 0.21, 0.0, 0.17608294, 1.55084124, 0.29110071, 0.00473278] # pitch
        # target_gpos = [ 0.21, 0.0, 0.17608294, 1.55084124, -0.00889929, 0.30473278] # yaw
        # print("target_gpos", target_gpos)
        # breakpoint()
        # print(f'{target_gpos=}')
        fd_qpos_mucojo = self.mjdata.qpos[self.qpos_indices][:6]
        # print(self.mjdata.qpos[self.qpos_indices])
        
        qpos_inv_mujoco, IK_success = lerobot_IK(fd_qpos_mucojo, target_gpos, robot=self.robot)
        
        if IK_success:
            
            self.target_qpos = np.concatenate((qpos_inv_mujoco[:6], [gripper_state,])) 
            # print("target_qpos", self.target_qpos)
            self.mjdata.qpos[self.qpos_indices] = self.target_qpos
            mujoco.mj_step(self.mjmodel, self.mjdata)
            self.target_gpos_last = target_gpos.copy() 

            # distance = 75.38 * self.target_qpos[-1] + 24.31 # mm为单位
            joint_angles = np.rad2deg(self.target_qpos)
            joint_angles[0] = -joint_angles[0]
            joint_angles[1] = -joint_angles[1]
            joint_angles[3] = -joint_angles[3]
            self.joint_angles_last = joint_angles.copy()

            target_gpos = np.concatenate((target_gpos, [gripper_state,]))

            print(f"target_gpos: {target_gpos}")

        else:
            self.target_gpos = self.target_gpos_last.copy()
            self.joyconrobotics.set_position = self.target_gpos[0:3]
            joint_angles = self.joint_angles_last
            target_gpos = np.concatenate((self.target_gpos, [gripper_state,]))
        # breakpoint()
        # if button_control != 0:
        #     self.joyconrobotics.reset_joycon()
            
        return joint_angles, button_control, target_gpos