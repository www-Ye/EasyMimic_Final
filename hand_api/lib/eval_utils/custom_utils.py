import copy
import numpy as np
import torch

from hawor.utils.process import run_mano, run_mano_left
from hawor.utils.rotation import angle_axis_to_quaternion, rotation_matrix_to_angle_axis, quaternion_to_rotation_matrix, angle_axis_to_rotation_matrix
from scipy.interpolate import interp1d

def world2cam_convert(R_w2c_sla, t_w2c_sla, data_world, handedness, input_type=None):
    """
    将手部姿态数据从世界坐标系转换到相机坐标系。
    
    参数：
        R_w2c_sla: 世界到相机的旋转矩阵 (T, 3, 3)
        t_w2c_sla: 世界到相机的平移向量 (T, 3)
        data_world: 包含世界坐标系下手部姿态数据的字典
        handedness: 'right'或'left'指示哪只手
        input_type: 'matrix'或'axis_angle'指定输入表示形式，或None自动检测
    """
    # 检查输入类型
    if input_type is None:
        if len(data_world["init_root_orient"].shape) == 3 and data_world["init_root_orient"].shape[-1] == 3:
            input_type = "axis_angle"
        elif len(data_world["init_root_orient"].shape) == 4 and data_world["init_root_orient"].shape[-2:] == (3, 3):
            input_type = "matrix"
        else:
            raise ValueError("无法自动检测输入类型，请明确指定")
    
    # 根据输入类型处理根方向
    if input_type == "axis_angle":
        world_root_orient = copy.deepcopy(data_world["init_root_orient"])
        world_rot_mat = angle_axis_to_rotation_matrix(world_root_orient)
    else:  # 矩阵
        world_rot_mat = copy.deepcopy(data_world["init_root_orient"])
        world_root_orient = rotation_matrix_to_angle_axis(world_rot_mat)
    
    # 根据输入类型处理手部姿态
    if input_type == "axis_angle":
        world_hand_pose = copy.deepcopy(data_world["init_hand_pose"])
    else:  # 矩阵
        world_hand_pose = rotation_matrix_to_angle_axis(data_world["init_hand_pose"])
    
    # 获取世界坐标系下的平移
    world_trans = data_world["init_trans"]  # (B, T, 3)
    
    # 在世界坐标系下运行MANO获取关节位置
    if handedness == "right":
        outputs = run_mano(world_trans, world_root_orient, world_hand_pose, betas=data_world["init_betas"])
    elif handedness == "left":
        outputs = run_mano_left(world_trans, world_root_orient, world_hand_pose, betas=data_world["init_betas"])

    # 注意：在你的代码中，R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
    # 并且 t_w2c_sla_all = -torch.einsum("bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)
    
    world_root_loc = outputs["joints"][..., 0, :].cpu()  # (B, T, 3)
    # 将世界坐标系的根关节位置转换到相机坐标系
    cam_root_loc = torch.einsum("tij,btj->bti", R_w2c_sla, world_root_loc)
    # 注意：在这种情况下，t_w2c_sla已经包含了-R_w2c * t_c2w的结果
    cam_root_loc = cam_root_loc + t_w2c_sla[None, :, :]  # 添加平移
    offset = world_trans - world_root_loc 
    # 计算相机坐标系下的平移
    cam_trans = cam_root_loc + offset
    
    # 将旋转矩阵从世界坐标系转换到相机坐标系
    cam_rot_mat = torch.einsum("tij,btjk->btik", R_w2c_sla, world_rot_mat)
    cam_root_orient = rotation_matrix_to_angle_axis(cam_rot_mat)
    
    # 准备输出数据
    data_cam = {
        "init_root_orient": cam_root_orient,  # (B, T, 3)
        "init_hand_pose": world_hand_pose,  # (B, T, 15, 3) - 手部姿态在局部坐标系中不变
        "init_trans": cam_trans,  # (B, T, 3)
        "init_betas": data_world["init_betas"]  # (B, T, 10)
    }
    
    return data_cam


def cam2world_convert(R_c2w_sla, t_c2w_sla, data_out, handedness, input_type=None):
    """
    Convert hand pose data from camera coordinates to world coordinates.
    
    Args:
        R_c2w_sla: Camera-to-world rotation matrix (T, 3, 3)
        t_c2w_sla: Camera-to-world translation vector (T, 3)
        data_out: Dictionary containing hand pose data in camera coordinates
        handedness: 'right' or 'left' to indicate which hand
        input_type: 'matrix' or 'axis_angle' to specify input representation, or None to auto-detect
        
    Returns:
        Dictionary with hand pose data in world coordinates
    """
    # Check input type for root orientation
    if input_type is None:
        # Auto-detect based on shape
        if len(data_out["init_root_orient"].shape) == 3 and data_out["init_root_orient"].shape[-1] == 3:
            input_type = "axis_angle"
        elif len(data_out["init_root_orient"].shape) == 4 and data_out["init_root_orient"].shape[-2:] == (3, 3):
            input_type = "matrix"
        else:
            raise ValueError("Cannot auto-detect input type, please specify explicitly")
    # Convert root orientation to rotation matrix if needed
    if input_type == "axis_angle":
        init_rot_mat = angle_axis_to_rotation_matrix(data_out["init_root_orient"])
    else:  # matrix
        init_rot_mat = copy.deepcopy(data_out["init_root_orient"])
    
    # Transform rotation matrix from camera to world
    init_rot_mat = torch.einsum("tij,btjk->btik", R_c2w_sla, init_rot_mat)
    init_rot = rotation_matrix_to_angle_axis(init_rot_mat)
    
    # Convert hand pose to axis-angle if it's in matrix form
    if input_type == "axis_angle":
        data_out_init_hand_pose = copy.deepcopy(data_out["init_hand_pose"])
    else:  # matrix
        data_out_init_hand_pose = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
    
    # Run MANO to get joint positions in camera space
    init_trans = data_out["init_trans"]  # (B, T, 3)
    if input_type == "axis_angle":
        mano_input_orient = data_out["init_root_orient"]
    else:
        mano_input_orient = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
        
    if handedness == "right":
        outputs = run_mano(init_trans, mano_input_orient, data_out_init_hand_pose, betas=data_out["init_betas"])
    elif handedness == "left":
        outputs = run_mano_left(init_trans, mano_input_orient, data_out_init_hand_pose, betas=data_out["init_betas"])
    
    root_loc = outputs["joints"][..., 0, :].cpu()  # (B, T, 3)
    offset = init_trans - root_loc  # It is a constant, no matter what the rotation is.
    
    # Transform translation from camera to world
    init_trans_world = (
        torch.einsum("tij,btj->bti", R_c2w_sla, root_loc)
        + t_c2w_sla[None, :]
        + offset
    )
    
    # Prepare output
    data_world = {
        "init_root_orient": init_rot,  # (B, T, 3)
        "init_hand_pose": data_out_init_hand_pose,  # (B, T, 15, 3)
        "init_trans": init_trans_world,  # (B, T, 3)
        "init_betas": data_out["init_betas"]  # (B, T, 10)
    }
    
    return data_world

# def cam2world_convert(R_c2w_sla, t_c2w_sla, data_out, handedness):
#     init_rot_mat = copy.deepcopy(data_out["init_root_orient"])
#     init_rot_mat = torch.einsum("tij,btjk->btik", R_c2w_sla, init_rot_mat)
#     init_rot = rotation_matrix_to_angle_axis(init_rot_mat)
#     init_rot_quat = angle_axis_to_quaternion(init_rot)
#     # data_out["init_root_orient"] = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
#     # data_out["init_hand_pose"] = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
#     data_out_init_root_orient = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
#     data_out_init_hand_pose = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])

#     init_trans = data_out["init_trans"] # (B, T, 3)
#     if handedness == "right":
#         outputs = run_mano(data_out["init_trans"], data_out_init_root_orient, data_out_init_hand_pose, betas=data_out["init_betas"])
#     elif handedness == "left":
#         outputs = run_mano_left(data_out["init_trans"], data_out_init_root_orient, data_out_init_hand_pose, betas=data_out["init_betas"])
#     root_loc = outputs["joints"][..., 0, :].cpu()  # (B, T, 3)
#     offset = init_trans - root_loc  # It is a constant, no matter what the rotation is.
#     init_trans = (
#         torch.einsum("tij,btj->bti", R_c2w_sla, root_loc)
#         + t_c2w_sla[None, :]
#         + offset
#     )

#     data_world = {
#         "init_root_orient": init_rot, # (B, T, 3)
#         "init_hand_pose": data_out_init_hand_pose, # (B, T, 15, 3)
#         "init_trans": init_trans,  # (B, T, 3)
#         "init_betas": data_out["init_betas"]  # (B, T, 10)
#     }

#     return data_world

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def load_slam_cam(fpath):
    print(f"Loading cameras from {fpath}...")
    pred_cam = dict(np.load(fpath, allow_pickle=True))
    pred_traj = pred_cam['traj']
    t_c2w_sla = torch.tensor(pred_traj[:, :3]) * pred_cam['scale']
    pred_camq = torch.tensor(pred_traj[:, 3:])
    R_c2w_sla = quaternion_to_matrix(pred_camq[:,[3,0,1,2]])
    R_w2c_sla = R_c2w_sla.transpose(-1, -2)
    t_w2c_sla = -torch.einsum("bij,bj->bi", R_w2c_sla, t_c2w_sla)
    return R_w2c_sla, t_w2c_sla, R_c2w_sla, t_c2w_sla


def interpolate_bboxes(bboxes):
    T = bboxes.shape[0]  

    zero_indices = np.where(np.all(bboxes == 0, axis=1))[0]

    non_zero_indices = np.where(np.any(bboxes != 0, axis=1))[0]

    if len(zero_indices) == 0:
        return bboxes

    interpolated_bboxes = bboxes.copy()
    for i in range(5):  
        interp_func = interp1d(non_zero_indices, bboxes[non_zero_indices, i], kind='linear', fill_value="extrapolate")
        interpolated_bboxes[zero_indices, i] = interp_func(zero_indices)
    
    return interpolated_bboxes