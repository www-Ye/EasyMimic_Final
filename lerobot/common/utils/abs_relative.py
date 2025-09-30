import torch
import numpy as np
import pytorch3d.transforms as pt
from typing import Tuple, Union, Literal

# 确定欧拉角约定 (例如: 'XYZ', 'ZYX', 'xyz' 等)
# RPY 通常对应于 ZYX 旋转顺序 (绕固定轴 Z, 再 Y, 再 X)
# 或者等效于 xyz 旋转顺序 (绕移动轴 x, 再 y, 再 z)
# pytorch3d 使用的是 extrinsic 约定 (大写) 或 intrinsic 约定 (小写)
# 我们这里假设使用 'XYZ' extrinsic 约定，如果你的 RPY 来源不同，需要调整 convention
EULER_CONVENTION = "XYZ"

def calculate_relative_transform(
    xyz: Union[np.ndarray, torch.Tensor],
    rpy: Union[np.ndarray, torch.Tensor],
    mode: Literal["delta", "relative"],
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    计算一系列 XYZ 位置和 RPY 姿态的相对变换。

    可以计算每个元素相对于前一个元素的变换 ('delta')，
    或者计算每个元素相对于第一个元素的变换 ('relative')。

    Args:
        xyz: 位置数据，形状为 (N, 3)，N 是序列长度。可以是 NumPy 数组或 PyTorch 张量。
        rpy: 欧拉角姿态数据 (RPY, 假定单位为弧度)，形状为 (N, 3)。
             必须与 xyz 类型相同。
        mode: 计算模式。
              - 'delta': 计算 t 相对于 t-1 的变换 (结果长度 N-1)。
              - 'relative': 计算 t 相对于 0 的变换 (结果长度 N，第一个为恒等变换)。

    Returns:
        A tuple (relative_xyz, relative_rpy):
        - relative_xyz: 相对位置，形状为 (N-1, 3) 或 (N, 3)，类型与输入相同。
                        表示在参考帧坐标系下的位移。
        - relative_rpy: 相对姿态 (RPY)，形状为 (N-1, 3) 或 (N, 3)，类型与输入相同。

    Raises:
        ValueError: 如果输入形状不正确或模式无效。
        TypeError: 如果 xyz 和 rpy 类型不匹配。
    """
    if not isinstance(xyz, type(rpy)):
        raise TypeError(f"xyz (type {type(xyz)}) and rpy (type {type(rpy)}) must have the same type.")

    # print(xyz.shape, rpy.shape)
    if xyz.shape != rpy.shape or len(xyz.shape) != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz and rpy must have shape (N, 3), but got {xyz.shape} and {rpy.shape}")

    n = xyz.shape[0]
    if n == 0:
        # 返回空的适当类型
        empty_shape = (0, 3)
        if isinstance(xyz, torch.Tensor):
            return torch.empty(empty_shape, dtype=xyz.dtype, device=xyz.device), \
                   torch.empty(empty_shape, dtype=rpy.dtype, device=rpy.device)
        else: # numpy
            return np.empty(empty_shape, dtype=xyz.dtype), \
                   np.empty(empty_shape, dtype=rpy.dtype)
            
    if n == 1 and mode == "delta":
        # delta 模式下，长度为1无法计算相对变换
        empty_shape = (0, 3)
        if isinstance(xyz, torch.Tensor):
            return torch.empty(empty_shape, dtype=xyz.dtype, device=xyz.device), \
                   torch.empty(empty_shape, dtype=rpy.dtype, device=rpy.device)
        else: # numpy
            return np.empty(empty_shape, dtype=xyz.dtype), \
                   np.empty(empty_shape, dtype=rpy.dtype)

    is_torch = isinstance(xyz, torch.Tensor)

    if is_torch:
        # --- PyTorch Implementation ---
        # 确保是 float 类型以进行旋转计算
        xyz_t = xyz.float()
        rpy_t = rpy.float()
        device = xyz_t.device

        # 1. 将 RPY 转换为旋转矩阵
        # (N, 3, 3)
        rot_matrices = pt.euler_angles_to_matrix(rpy_t, convention=EULER_CONVENTION)

        if mode == "delta":
            # 计算 t 时刻相对于 t-1 时刻的变换
            # 结果长度为 N-1

            # 参考帧的旋转矩阵 (t-1 时刻) (N-1, 3, 3)
            ref_rot_matrices = rot_matrices[:-1]
            # 当前帧的旋转矩阵 (t 时刻) (N-1, 3, 3)
            cur_rot_matrices = rot_matrices[1:]

            # 参考帧的位置 (t-1 时刻) (N-1, 3)
            ref_xyz = xyz_t[:-1]
            # 当前帧的位置 (t 时刻) (N-1, 3)
            cur_xyz = xyz_t[1:]

            # 计算相对旋转矩阵: R_rel = R_ref^{-1} * R_cur
            # R_ref^{-1} 等于 R_ref.T (转置)
            # (N-1, 3, 3)
            relative_rot_matrices = torch.matmul(ref_rot_matrices.transpose(-1, -2), cur_rot_matrices)

            # 计算相对位置: xyz_rel = R_ref^{-1} * (xyz_cur - xyz_ref)
            # (N-1, 3, 1)
            world_diff = (cur_xyz - ref_xyz).unsqueeze(-1)
            # (N-1, 3)
            relative_xyz_t = torch.matmul(ref_rot_matrices.transpose(-1, -2), world_diff).squeeze(-1)

            # 将相对旋转矩阵转换回 RPY
            relative_rpy_t = pt.matrix_to_euler_angles(relative_rot_matrices, convention=EULER_CONVENTION)

            # 转换回原始 dtype
            relative_xyz = relative_xyz_t.to(xyz.dtype)
            relative_rpy = relative_rpy_t.to(rpy.dtype)

        elif mode == "relative":
            # 计算 t 时刻相对于 0 时刻的变换
            # 结果长度为 N

            # 参考帧 (0 时刻)
            ref_rot_matrix_0 = rot_matrices[0]      # (3, 3)
            ref_xyz_0 = xyz_t[0]                   # (3,)

            cur_xyz = xyz_t[1:]

            # 参考帧的逆旋转 (3, 3)
            ref_rot_inv_0 = ref_rot_matrix_0.T

            cur_rot_matrices = rot_matrices[1:]

            # 计算相对旋转矩阵: R_rel = R_0^{-1} * R_t
            # (N, 3, 3)
            relative_rot_matrices = torch.matmul(ref_rot_inv_0, cur_rot_matrices) # 利用广播

            # 计算相对位置: xyz_rel = R_0^{-1} * (xyz_t - xyz_0)
            # (N, 3, 1)
            world_diff = (cur_xyz - ref_xyz_0).unsqueeze(-1)
            # (N, 3)
            relative_xyz_t = torch.matmul(ref_rot_inv_0, world_diff).squeeze(-1) # 利用广播

            # 将相对旋转矩阵转换回 RPY
            relative_rpy_t = pt.matrix_to_euler_angles(relative_rot_matrices, convention=EULER_CONVENTION)

            # # 第一个元素应该是恒等变换 (0 位置, 0 旋转) - 理论上计算结果就应接近这个
            # # 可以选择强制设为0以确保数值精度
            # relative_xyz_t[0] = 0.0
            # relative_rpy_t[0] = 0.0

            # 转换回原始 dtype
            relative_xyz = relative_xyz_t.to(xyz.dtype)
            relative_rpy = relative_rpy_t.to(rpy.dtype)

        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 'delta' or 'relative'.")

        return relative_xyz, relative_rpy

    else:
        # --- NumPy Implementation ---
        from scipy.spatial.transform import Rotation as R

        # 确保是 float 类型
        xyz_np = xyz.astype(float)
        rpy_np = rpy.astype(float)

        # 1. 将 RPY 转换为旋转对象 (内部使用四元数或矩阵)
        # scipy 的 Euler 约定需要小写 'xyz', 'zyx' 等表示 intrinsic
        # 或大写 'XYZ', 'ZYX' 表示 extrinsic
        # 假设 EULER_CONVENTION = "XYZ" (extrinsic)
        # 注意：scipy RPY->Rotation 的 'xyz' 可能与 pytorch3d 不同，确保一致性！
        # 如果 RPY 来自 ROS，通常是 intrinsic zyx
        # RPY (roll-pitch-yaw) 最常见的约定是绕固定轴 Z-Y-X (ZYX extrinsic)
        # 或者等效地绕移动轴 x-y-z (xyz intrinsic)
        # 我们继续假设 EULER_CONVENTION = "XYZ" extrinsic for scipy as well
        try:
            rotations = R.from_euler(EULER_CONVENTION, rpy_np)
        except ValueError as e:
             print(f"Error creating Rotation object. Check EULER_CONVENTION ('{EULER_CONVENTION}') and RPY values.")
             print(f"RPY data sample: {rpy_np[:5]}")
             raise e
             
        rot_matrices_np = rotations.as_matrix() # (N, 3, 3)

        if mode == "delta":
            if n < 2: return np.empty((0, 3), dtype=xyz.dtype), np.empty((0, 3), dtype=rpy.dtype)
            # 参考帧 (t-1)
            ref_rots = rotations[:-1]
            ref_xyz_np = xyz_np[:-1]
            # 当前帧 (t)
            cur_rots = rotations[1:]
            cur_xyz_np = xyz_np[1:]

            # 计算相对旋转: R_rel = R_ref^{-1} * R_cur
            relative_rots = ref_rots.inv() * cur_rots # 使用 Rotation 对象的乘法和逆

            # 计算相对位置: xyz_rel = R_ref^{-1} * (xyz_cur - xyz_ref)
            world_diff_np = cur_xyz_np - ref_xyz_np
            # 使用 R_ref^{-1} (即 ref_rots.inv()) 来旋转差值向量
            relative_xyz_np = ref_rots.inv().apply(world_diff_np) # apply 方法用于旋转向量

            # 转回 RPY
            relative_rpy_np = relative_rots.as_euler(EULER_CONVENTION)

            # 转回原始 dtype
            relative_xyz = relative_xyz_np.astype(xyz.dtype)
            relative_rpy = relative_rpy_np.astype(rpy.dtype)

        elif mode == "relative":
            # 参考帧 (0)
            ref_rot_0 = rotations[0]
            ref_xyz_np_0 = xyz_np[0]

            cur_rots = rotations[1:]
            cur_xyz_np = xyz_np[1:]

            # 计算相对旋转: R_rel = R_0^{-1} * R_t
            relative_rots = ref_rot_0.inv() * cur_rots

            # 计算相对位置: xyz_rel = R_0^{-1} * (xyz_t - xyz_0)
            world_diff_np = cur_xyz_np - ref_xyz_np_0
            relative_xyz_np = ref_rot_0.inv().apply(world_diff_np)

            # 转回 RPY
            relative_rpy_np = relative_rots.as_euler(EULER_CONVENTION)

            # # 强制第一个元素为0
            # relative_xyz_np[0] = 0.0
            # relative_rpy_np[0] = 0.0
            
            # 转回原始 dtype
            relative_xyz = relative_xyz_np.astype(xyz.dtype)
            relative_rpy = relative_rpy_np.astype(rpy.dtype)
            
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 'delta' or 'relative'.")

        return relative_xyz, relative_rpy

# --- 示例用法 ---
if __name__ == '__main__':
    # 1. 使用你的 PyTorch Tensor 数据
    print("--- PyTorch Example ---")
    xyz_pt = torch.tensor([[0.2468, 0.0218, 0.1196],
                           [0.2478, 0.0243, 0.1287],
                           [0.2500, 0.0250, 0.1300]], dtype=torch.float64)
    rpy_pt = torch.tensor([[1.5500, 0.1000, 0.4770],
                           [1.5220, 0.0830, 0.4740],
                           [1.5000, 0.0700, 0.4700]], dtype=torch.float64) # 假设是弧度

    print("Input XYZ:\n", xyz_pt)
    print("Input RPY:\n", rpy_pt)

    # 计算 t vs t-1
    rel_xyz_delta_pt, rel_rpy_delta_pt = calculate_relative_transform(xyz_pt, rpy_pt, mode="delta")
    print("\nDelta (t vs t-1) XYZ:\n", rel_xyz_delta_pt)
    print("Delta (t vs t-1) RPY:\n", rel_rpy_delta_pt)

    # 计算 t vs 0
    rel_xyz_rel_pt, rel_rpy_rel_pt = calculate_relative_transform(xyz_pt, rpy_pt, mode="relative")
    print("\nRelative (t vs 0) XYZ:\n", rel_xyz_rel_pt)
    print("Relative (t vs 0) RPY:\n", rel_rpy_rel_pt)
    
    # 2. 使用你的 NumPy Array 数据 (假设 action 数据是序列)
    print("\n--- NumPy Example ---")
    # 注意：你的 action 数据 shape 看起来是 (17, 3)，适合直接用
    xyz_np = np.array([[0.24785347, 0.02430044, 0.12868929],
                       [0.24777243, 0.02430988, 0.12884052],
                       [0.24770507, 0.02428775, 0.12890854]]) # 取前3个示例
    rpy_np = np.array([[1.52199996, 0.083     , 0.47400001],
                       [1.52199996, 0.089     , 0.47      ],
                       [1.52199996, 0.095     , 0.465     ]]) # 取前3个示例

    print("Input XYZ:\n", xyz_np)
    print("Input RPY:\n", rpy_np)

    rel_xyz_delta_np, rel_rpy_delta_np = calculate_relative_transform(xyz_np, rpy_np, mode="delta")
    print("\nDelta (t vs t-1) XYZ:\n", rel_xyz_delta_np)
    print("Delta (t vs t-1) RPY:\n", rel_rpy_delta_np)

    rel_xyz_rel_np, rel_rpy_rel_np = calculate_relative_transform(xyz_np, rpy_np, mode="relative")
    print("\nRelative (t vs 0) XYZ:\n", rel_xyz_rel_np)
    print("Relative (t vs 0) RPY:\n", rel_rpy_rel_np)

    # 处理边界情况 N=1
    print("\n--- Edge Case N=1 ---")
    xyz_pt_n1 = xyz_pt[0:1]
    rpy_pt_n1 = rpy_pt[0:1]
    print("Input N=1 XYZ:\n", xyz_pt_n1)
    
    rel_xyz_d, rel_rpy_d = calculate_relative_transform(xyz_pt_n1, rpy_pt_n1, mode="delta")
    print("Delta N=1 XYZ shape:", rel_xyz_d.shape) # Expect (0, 3)
    
    rel_xyz_r, rel_rpy_r = calculate_relative_transform(xyz_pt_n1, rpy_pt_n1, mode="relative")
    print("Relative N=1 XYZ:\n", rel_xyz_r) # Expect [[0., 0., 0.]]
    print("Relative N=1 RPY:\n", rel_rpy_r) # Expect [[0., 0., 0.]]