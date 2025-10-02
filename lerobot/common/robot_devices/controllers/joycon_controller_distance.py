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
                break  # Connection successful, exit loop
            except Exception as e:
                self.retries += 1
                print(f"Failed to connect to {name} Joycon: {e}")
                print(f"Retrying ({self.retries}/{self.max_retries}) in 1 second...")
                time.sleep(1)  # Wait 1 second before retry after connection failure
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
        roll = roll - math.pi/2 # Rotate lerobot end effector by 90 degrees
        
        # Dual arm bias towards center
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
    

from scipy.spatial.transform import Rotation as R # Import Rotation

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
                break  # Connection successful, exit loop
            except Exception as e:
                self.retries += 1
                print(f"Failed to connect to {name} Joycon: {e}")
                print(f"Retrying ({self.retries}/{self.max_retries}) in 1 second...")
                time.sleep(1)  # Wait 1 second before retry after connection failure
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
        distance = 75.38 * qpos[-1] + 24.31 # In mm units
        # gripper_state = qpos[-1]
        eef_state = np.concatenate((eef_pos, [distance,]))
        return eef_state

    def get_eef(self, joint_angles, action_space="eef_rpy"):   # "aa"
        joint_angles[0] = -joint_angles[0]  
        joint_angles[1] = -joint_angles[1]  
        joint_angles[3] = -joint_angles[3]  
        qpos = np.deg2rad(joint_angles)
        distance = 75.38 * qpos[-1] + 24.31 # In mm units
        # gripper_state = qpos[-1]
        eef_pos = lerobot_FK(qpos[:6], robot=self.robot)
        # if "aa" in action_space:
        #     eef_pos[-3:] = rpy2aa(eef_pos[-3:])
        if "eef_rpy_midpoint" in action_space:
            eef_pos = self.get_midpoint(eef_pos, qpos[-1], L=self.gripper_length)  # In mm units
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

        # 5. Apply limits
        pose_limited_transformed = np.zeros_like(pose_transformed_for_limiting)
        for i in range(6):
            pose_limited_transformed[i] = np.clip(
                pose_transformed_for_limiting[i],
                self.glimit[0][i],
                self.glimit[1][i]
            )

        x_lim_t, y_lim_t, z_lim_t, roll_lim_t, pitch_lim_t, yaw_lim_t = pose_limited_transformed

        # 7. Inverse transform (restore RPY format expected by IK)
        # Inverse transform pitch: pitch_original = -pitch_transformed
        pitch_limited_original = -pitch_lim_t
        # Inverse transform roll: roll_original = -roll_transformed + math.pi/2
        roll_limited_original = -(roll_lim_t) + math.pi/2

        target_gpos = np.array([
            x_lim_t, y_lim_t, z_lim_t,
            roll_limited_original, pitch_limited_original, yaw_lim_t # Use adjusted yaw
        ])

        distance = eef_action[-1]
        gripper_state = (distance - 24.31) / 75.38

        # Get gripper angle regression
        gripper_angles = self.get_gripper_angles(distance)

        # gripper_state = eef_action[-1]

        # Calculate end effector position from gripper midpoint
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
    
    def get_command(self, present_pose, action_space="eef_rpy"):   
        
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
        roll = -(roll - math.pi/2) # Rotate lerobot end effector by 90 degrees
        
        target_gpos = np.array([x, y, z, roll, pitch, yaw])

        # There's an issue with teleoperation, when controlling gripper, gripper angle changes, causing midpoint calculation deviation
        # Original logic: control end effector position and orientation, then control gripper state
        # Current logic: control gripper midpoint position and orientation, but gripper state is independent, now coupled together!!!!!
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

            distance = 75.38 * gripper_state + 24.31 # In mm units
            target_gpos = np.concatenate((target_gpos, [distance,]))

        else:
            self.target_gpos = self.target_gpos_last.copy()
            self.joyconrobotics.set_position = self.target_gpos[0:3]
            joint_angles = self.joint_angles_last

            distance = 75.38 * gripper_state + 24.31 # In mm units
            target_gpos = np.concatenate((self.target_gpos, [distance,]))
            
        return joint_angles, button_control, target_gpos
    
    def get_delta_command(self, present_pose_angles_deg):
        """
        Calculates the new target joint angles based on the delta input from the Joycon.

        Args:
            present_pose_angles_deg (np.ndarray): The current joint angles of the robot arm (in degrees),
                             in the same format as self.joint_angles_last (with sign flips).

        Returns:
            tuple: (joint_angles, button_control, target_gpos_with_distance)
               - joint_angles (np.ndarray): The calculated target joint angles (degrees, with sign flips).
               - button_control (int): The button control state from the Joycon.
               - target_gpos_with_distance (np.ndarray): The target pose (x,y,z,r,p,y in rad) attempted by IK,
                                    plus the calculated gripper distance (mm).
                                    If IK fails, this is the last successful target pose.
        """
        current_angles_deg_flipped = np.array(present_pose_angles_deg)
        qpos_rad_for_fk = current_angles_deg_flipped.copy()
        qpos_rad_for_fk[0] = -qpos_rad_for_fk[0] 
        qpos_rad_for_fk[1] = -qpos_rad_for_fk[1] 
        qpos_rad_for_fk[3] = -qpos_rad_for_fk[3] 
        qpos_rad_for_fk = np.deg2rad(qpos_rad_for_fk[:6]) 

        current_eef_pose_rad = lerobot_FK(qpos_rad_for_fk, robot=self.robot) # (x,y,z, r,p,y in radians)
        print('-' * 50)
        print("current_eef_pose_rad", current_eef_pose_rad)
        # current_eef_pose_rad = self.target_gpos_for_ik
        current_pos_w = current_eef_pose_rad[:3]
        current_orient_rad_w = current_eef_pose_rad[3:] # [roll, pitch, yaw]

        # delta_pose = [dx_t, dy_t, dz_t, droll_t, dpitch_t, dyaw_t]
        delta_pose, gripper_state, button_control = self.joyconrobotics.get_control(return_delta=True)

        print("delta_pose", delta_pose)

        delta_pose = np.array([0.0, 0.0, 0.01, 0, 0, 0])
        print("curr delta_pose", delta_pose)
        
        delta_pos_t = np.array(delta_pose[:3])
        delta_orient_rad_t = np.array(delta_pose[3:]) # [droll, dpitch, dyaw] around tool axes

        distance = 75.38 * gripper_state + 24.31 # In mm units
        print(delta_pose)
        print(distance)
        delta_action = np.concatenate((delta_pose, [distance,]))

        # 3. Calculate new target EEF pose (world coordinate system)

        # --- a. Handle rotation ---
        # Convert current world pose and tool delta pose to Rotation objects
        # !!Key: Confirm euler angle order used by FK and IK ('xyz', 'zyx', etc.)
        # Assuming 'xyz' here
        R_world_tool_current = R.from_euler('xyz', current_orient_rad_w, degrees=False)

        # Convert tool coordinate system delta rotation as 'xyz' euler angle sequence defined rotation
        # This is an intrinsic rotation
        R_tool_delta = R.from_euler('xyz', delta_orient_rad_t, degrees=False)
        # Apply current rotation first, then apply delta rotation in tool coordinate system
        R_world_tool_new = R_world_tool_current * R_tool_delta
        # Get new euler angles in world coordinate system
        # !!Key: Ensure output euler angle order matches what IK needs
        next_orient_rad_w = R_world_tool_new.as_euler('xyz', degrees=False)

        # --- b. Handle translation ---
        # Convert tool coordinate system delta translation to world coordinate system
        # delta_pos_w = R_world_tool_current.apply(delta_pos_t) # R_current * delta_t
        # !! Note: R.apply converts vector from "body coordinate system" to "external coordinate system"
        # If delta_pos_t is a vector in tool frame, we need to use R_world_tool_current to rotate it to world frame
        delta_pos_w = R_world_tool_current.apply(delta_pos_t)
        # Calculate new position in world coordinate system
        next_pos_w = current_pos_w + delta_pos_w

        # --- Combine new target pose ---
        next_target_eef_pose_rad = np.concatenate((next_pos_w, next_orient_rad_w))

        # 4. Apply transformation and limits (prepare for IK)
        # Note: The transformation and limit logic here is now applied to next_target_eef_pose_rad (world coordinate system)
        # If your limits self.glimit are defined in some transformed coordinate system, you need to adjust accordingly
        # The following code assumes transformation and limits are still applied to world coordinate system r,p,y as before
        x, y, z = next_target_eef_pose_rad[:3]
        roll, pitch, yaw = next_target_eef_pose_rad[3:] # World coordinate system roll, pitch, yaw

        target_gpos_for_ik = [x, y, z, roll, pitch, yaw]

        self.target_gpos_for_ik = target_gpos_for_ik.copy()
        # target_gpos_for_ik = [0.21772567056163034, 0.0, 0.3, 0.0, 0.0, 0.0]
        target_gpos_for_ik = [0.26, 0.0, 0.27, 0.0, 0.0, 0.0]
        print("current_eef_pose_rad", current_eef_pose_rad)
        print("target_gpos_for_ik", target_gpos_for_ik)

        # 5. Execute inverse kinematics (IK)
        fd_qpos_mucojo = self.mjdata.qpos[self.qpos_indices][:6]
        qpos_inv_mujoco, IK_success = lerobot_IK(fd_qpos_mucojo, target_gpos_for_ik, robot=self.robot)

        # 6. Handle IK results (same logic as before)
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
        Calculate the endpoint position after adding a fixed rod to the robotic arm end effector position, or perform inverse calculation.

        Args:
            eef_pos (np.ndarray): End effector pose (x, y, z, roll, pitch, yaw) or endpoint pose (when inverse=True)
            grip_angles (np.ndarray): Gripper angles in radians
            L (float): Length of the fixed rod, in meters
            inverse (bool): Whether to perform inverse calculation, True means calculate executor position from endpoint position

        Returns:
            np.ndarray: Rod endpoint position and pose [x, y, z, roll, pitch, yaw] (when inverse=False)
                       or executor position and pose [x, y, z, roll, pitch, yaw] (when inverse=True)
        """
        # Create rotation matrix from euler angles (using ZYX order)
        R_final = R.from_euler('zyx', eef_pos[3:], degrees=False)

        L_effective = L * math.cos(grip_angles)
        
        # Define rod displacement vector in local coordinate system
        rod_local = np.array([0, 0, L_effective])
        
        if inverse:
            # Inverse calculation: calculate executor position from endpoint position
            # Convert displacement to global coordinate system and subtract from endpoint position
            rod_displacement = R_final.apply(rod_local)
            eef_tip_pos = eef_pos[:3] - rod_displacement
            
            # Pose remains unchanged
            eef_tip_angles = eef_pos[3:]    
            
            # Return complete pose of executor
            return np.concatenate([eef_tip_pos, eef_tip_angles])
        else:
            # Forward calculation: calculate endpoint position from executor position
            # Convert displacement to global coordinate system and add to end effector position
            rod_displacement = R_final.apply(rod_local)
            rod_tip_pos = eef_pos[:3] + rod_displacement
            
            # Pose remains unchanged
            rod_tip_angles = eef_pos[3:]    
            
            # Return complete pose of rod endpoint
            return np.concatenate([rod_tip_pos, rod_tip_angles])
    
    # Calculate gripper angles from gripper distance, parameters to be calculated
    def get_gripper_angles(self, gripper_distance):
        # Relationship between gripper distance and gripper angle
        # Gripper distance = 75.38 * gripper angle + 24.31
        # Gripper angle = (gripper distance - 24.31) / 75.38
        return (gripper_distance - 24.31) / 75.38