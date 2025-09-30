import torch
import numpy as np
import cv2
from typing import Tuple, Optional

class CameraJitter:
    """
    相机抖动工具类，用于为SLAM生成更多关键帧对
    """
    
    def __init__(self, 
                 translation_std: float = 0.002,  # 平移标准差 (相对于图像尺寸)
                 rotation_std: float = 0.01,      # 旋转标准差 (弧度)
                 focal_std: float = 0.005,        # 焦距变化标准差 (相对值)
                 center_std: float = 1.0,         # 光心偏移标准差 (像素)
                 enable_geometric_jitter: bool = True,  # 启用几何抖动
                 enable_photometric_jitter: bool = True, # 启用光度抖动
                 jitter_probability: float = 0.8,  # 抖动概率
                 seed: Optional[int] = None):
        """
        初始化相机抖动参数
        
        Args:
            translation_std: 虚拟相机平移的标准差
            rotation_std: 虚拟相机旋转的标准差 
            focal_std: 焦距变化的标准差
            center_std: 光心偏移的标准差
            enable_geometric_jitter: 是否启用几何抖动
            enable_photometric_jitter: 是否启用光度抖动
            jitter_probability: 每帧应用抖动的概率
            seed: 随机种子
        """
        self.translation_std = translation_std
        self.rotation_std = rotation_std
        self.focal_std = focal_std
        self.center_std = center_std
        self.enable_geometric_jitter = enable_geometric_jitter
        self.enable_photometric_jitter = enable_photometric_jitter
        self.jitter_probability = jitter_probability
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    def apply_intrinsic_jitter(self, intrinsics: torch.Tensor, image_shape: Tuple[int, int]) -> torch.Tensor:
        """
        对相机内参添加抖动
        
        Args:
            intrinsics: [fx, fy, cx, cy] 内参张量
            image_shape: (height, width) 图像尺寸
            
        Returns:
            抖动后的内参张量
        """
        if not self.enable_geometric_jitter or np.random.random() > self.jitter_probability:
            return intrinsics.clone()
            
        jittered_intrinsics = intrinsics.clone()
        h, w = image_shape
        
        # 焦距抖动 (保持长宽比)
        focal_noise = np.random.normal(0, self.focal_std)
        jittered_intrinsics[0] *= (1 + focal_noise)  # fx
        jittered_intrinsics[1] *= (1 + focal_noise)  # fy
        
        # 光心抖动
        center_noise_x = np.random.normal(0, self.center_std)
        center_noise_y = np.random.normal(0, self.center_std)
        jittered_intrinsics[2] += center_noise_x  # cx
        jittered_intrinsics[3] += center_noise_y  # cy
        
        # 确保光心在图像范围内
        jittered_intrinsics[2] = torch.clamp(jittered_intrinsics[2], 0, w-1)
        jittered_intrinsics[3] = torch.clamp(jittered_intrinsics[3], 0, h-1)
        
        return jittered_intrinsics
    
    def apply_image_jitter(self, image: torch.Tensor) -> torch.Tensor:
        """
        对图像添加微小的几何变换和光度变化
        
        Args:
            image: [C, H, W] 图像张量
            
        Returns:
            抖动后的图像张量
        """
        if np.random.random() > self.jitter_probability:
            return image.clone()
            
        jittered_image = image.clone()
        
        # 几何抖动：微小的仿射变换
        if self.enable_geometric_jitter:
            jittered_image = self._apply_geometric_transform(jittered_image)
        
        # 光度抖动：亮度和对比度变化
        if self.enable_photometric_jitter:
            jittered_image = self._apply_photometric_transform(jittered_image)
            
        return jittered_image
    
    def _apply_geometric_transform(self, image: torch.Tensor) -> torch.Tensor:
        """应用微小的几何变换"""
        C, H, W = image.shape
        
        # 生成微小的仿射变换矩阵
        # 平移
        dx = np.random.normal(0, self.translation_std * W)
        dy = np.random.normal(0, self.translation_std * H)
        
        # 旋转
        angle = np.random.normal(0, self.rotation_std * 180 / np.pi)  # 转换为度
        
        # 缩放 (非常微小)
        scale = 1 + np.random.normal(0, 0.002)
        
        # 构建变换矩阵
        center = (W // 2, H // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += dx
        M[1, 2] += dy
        
        # 应用变换
        image_np = image.permute(1, 2, 0).numpy().astype(np.uint8)
        transformed = cv2.warpAffine(image_np, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return torch.from_numpy(transformed).permute(2, 0, 1).float()
    
    def _apply_photometric_transform(self, image: torch.Tensor) -> torch.Tensor:
        """应用光度变换"""
        # 亮度变化
        brightness_factor = 1 + np.random.normal(0, 0.05)
        
        # 对比度变化
        contrast_factor = 1 + np.random.normal(0, 0.03)
        
        # 应用变换
        mean_intensity = image.mean()
        jittered = contrast_factor * (image - mean_intensity) + mean_intensity * brightness_factor
        
        # 限制到有效范围
        jittered = torch.clamp(jittered, 0, 255)
        
        return jittered
    
    def generate_virtual_keyframes(self, image: torch.Tensor, intrinsics: torch.Tensor, 
                                 num_virtual: int = 2) -> list:
        """
        为单帧图像生成多个虚拟关键帧
        
        Args:
            image: [C, H, W] 原始图像
            intrinsics: [4] 原始内参
            num_virtual: 生成的虚拟帧数量
            
        Returns:
            [(image, intrinsics), ...] 虚拟帧列表
        """
        virtual_frames = []
        h, w = image.shape[1], image.shape[2]
        
        for _ in range(num_virtual):
            # 生成抖动后的图像和内参
            jittered_image = self.apply_image_jitter(image)
            jittered_intrinsics = self.apply_intrinsic_jitter(intrinsics, (h, w))
            
            virtual_frames.append((jittered_image, jittered_intrinsics))
            
        return virtual_frames
    
    def should_add_virtual_frames(self, frame_idx: int, total_frames: int, 
                                motion_score: float = 0.0) -> bool:
        """
        判断是否需要为当前帧添加虚拟帧
        
        Args:
            frame_idx: 当前帧索引
            total_frames: 总帧数
            motion_score: 运动评分 (0-1, 越低表示运动越少)
            
        Returns:
            是否需要添加虚拟帧
        """
        # 为运动较少的帧添加更多虚拟帧
        if motion_score < 0.3:  # 低运动阈值
            return True
            
        # 在视频的开始和结束部分添加虚拟帧
        if frame_idx < total_frames * 0.1 or frame_idx > total_frames * 0.9:
            return True
            
        # 随机添加一些虚拟帧
        return np.random.random() < 0.3