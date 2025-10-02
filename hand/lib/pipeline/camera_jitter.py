import torch
import numpy as np
import cv2
from typing import Tuple, Optional

class CameraJitter:
    """
    Camera jitter utility class for generating more keyframe pairs for SLAM
    """
    
    def __init__(self, 
                 translation_std: float = 0.002,  # Translation standard deviation (relative to image size)
                 rotation_std: float = 0.01,      # Rotation standard deviation (radians)
                 focal_std: float = 0.005,        # Focal length change standard deviation (relative value)
                 center_std: float = 1.0,         # Principal point offset standard deviation (pixels)
                 enable_geometric_jitter: bool = True,  # Enable geometric jittering
                 enable_photometric_jitter: bool = True, # Enable photometric jittering
                 jitter_probability: float = 0.8,  # Jittering probability
                 seed: Optional[int] = None):
        """
        Initialize camera jitter parameters
        
        Args:
            translation_std: Standard deviation for virtual camera translation
            rotation_std: Standard deviation for virtual camera rotation
            focal_std: Standard deviation for focal length changes
            center_std: Standard deviation for principal point offset
            enable_geometric_jitter: Whether to enable geometric jittering
            enable_photometric_jitter: Whether to enable photometric jittering
            jitter_probability: Probability of applying jitter per frame
            seed: Random seed
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
        Add jitter to camera intrinsics
        
        Args:
            intrinsics: [fx, fy, cx, cy] intrinsic parameter tensor
            image_shape: (height, width) image dimensions
            
        Returns:
            Jittered intrinsic parameter tensor
        """
        if not self.enable_geometric_jitter or np.random.random() > self.jitter_probability:
            return intrinsics.clone()
            
        jittered_intrinsics = intrinsics.clone()
        h, w = image_shape
        
        # Focal length jitter (maintain aspect ratio)
        focal_noise = np.random.normal(0, self.focal_std)
        jittered_intrinsics[0] *= (1 + focal_noise)  # fx
        jittered_intrinsics[1] *= (1 + focal_noise)  # fy
        
        # Principal point jitter
        center_noise_x = np.random.normal(0, self.center_std)
        center_noise_y = np.random.normal(0, self.center_std)
        jittered_intrinsics[2] += center_noise_x  # cx
        jittered_intrinsics[3] += center_noise_y  # cy
        
        # Ensure principal point stays within image bounds
        jittered_intrinsics[2] = torch.clamp(jittered_intrinsics[2], 0, w-1)
        jittered_intrinsics[3] = torch.clamp(jittered_intrinsics[3], 0, h-1)
        
        return jittered_intrinsics
    
    def apply_image_jitter(self, image: torch.Tensor) -> torch.Tensor:
        """
        Add minor geometric transformations and photometric changes to image
        
        Args:
            image: [C, H, W] image tensor
            
        Returns:
            Jittered image tensor
        """
        if np.random.random() > self.jitter_probability:
            return image.clone()
            
        jittered_image = image.clone()
        
        # Geometric jitter: minor affine transformations
        if self.enable_geometric_jitter:
            jittered_image = self._apply_geometric_transform(jittered_image)
        
        # Photometric jitter: brightness and contrast changes
        if self.enable_photometric_jitter:
            jittered_image = self._apply_photometric_transform(jittered_image)
            
        return jittered_image
    
    def _apply_geometric_transform(self, image: torch.Tensor) -> torch.Tensor:
        """Apply minor geometric transformations"""
        C, H, W = image.shape
        
        # Generate minor affine transformation matrix
        # Translation
        dx = np.random.normal(0, self.translation_std * W)
        dy = np.random.normal(0, self.translation_std * H)
        
        # Rotation
        angle = np.random.normal(0, self.rotation_std * 180 / np.pi)  # Convert to degrees
        
        # Scaling (very minor)
        scale = 1 + np.random.normal(0, 0.002)
        
        # Build transformation matrix
        center = (W // 2, H // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += dx
        M[1, 2] += dy
        
        # Apply transformation
        image_np = image.permute(1, 2, 0).numpy().astype(np.uint8)
        transformed = cv2.warpAffine(image_np, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return torch.from_numpy(transformed).permute(2, 0, 1).float()
    
    def _apply_photometric_transform(self, image: torch.Tensor) -> torch.Tensor:
        """Apply photometric transformations"""
        # Brightness change
        brightness_factor = 1 + np.random.normal(0, 0.05)
        
        # Contrast change
        contrast_factor = 1 + np.random.normal(0, 0.03)
        
        # Apply transformation
        mean_intensity = image.mean()
        jittered = contrast_factor * (image - mean_intensity) + mean_intensity * brightness_factor
        
        # Clamp to valid range
        jittered = torch.clamp(jittered, 0, 255)
        
        return jittered
    
    def generate_virtual_keyframes(self, image: torch.Tensor, intrinsics: torch.Tensor, 
                                 num_virtual: int = 2) -> list:
        """
        Generate multiple virtual keyframes for a single frame image
        
        Args:
            image: [C, H, W] original image
            intrinsics: [4] original intrinsics
            num_virtual: number of virtual frames to generate
            
        Returns:
            [(image, intrinsics), ...] list of virtual frames
        """
        virtual_frames = []
        h, w = image.shape[1], image.shape[2]
        
        for _ in range(num_virtual):
            # Generate jittered image and intrinsics
            jittered_image = self.apply_image_jitter(image)
            jittered_intrinsics = self.apply_intrinsic_jitter(intrinsics, (h, w))
            
            virtual_frames.append((jittered_image, jittered_intrinsics))
            
        return virtual_frames
    
    def should_add_virtual_frames(self, frame_idx: int, total_frames: int, 
                                motion_score: float = 0.0) -> bool:
        """
        Determine whether to add virtual frames for the current frame
        
        Args:
            frame_idx: current frame index
            total_frames: total number of frames
            motion_score: motion score (0-1, lower indicates less motion)
            
        Returns:
            whether virtual frames should be added
        """
        # Add more virtual frames for frames with less motion
        if motion_score < 0.3:  # Low motion threshold
            return True
            
        # Add virtual frames at the beginning and end of the video
        if frame_idx < total_frames * 0.1 or frame_idx > total_frames * 0.9:
            return True
            
        # Randomly add some virtual frames
        return np.random.random() < 0.3