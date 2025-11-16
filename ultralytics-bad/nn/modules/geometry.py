# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Geometry prior utilities for RGB-D fusion in YOLOv12.

This module provides lightweight geometry prior extraction from depth maps,
including surface normals, edge strength, and quality estimation using Sobel operators.

Classes:
    GeometryPriorGenerator: Extracts geometry priors (normals, edges, gradients) from depth maps.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryPriorGenerator(nn.Module):
    """
    Generate geometry priors (normals, edges, gradients, quality) from depth maps.
    
    This module uses Sobel operators to extract geometric features from depth images,
    which can enhance RGB-D fusion by providing structural information. Inspired by
    DFormer and RemDet's information preservation principles.
    
    Features:
        - Lightweight: No learnable parameters, only Sobel convolutions
        - Numerically stable: Gradient clamping prevents NaN/Inf propagation
        - Efficient: Compact mode reduces output channels from 7 to 5
        
    Args:
        eps (float): Small value to prevent division by zero. Default: 1e-6
        grad_clip (float): Maximum gradient value for Sobel output. Default: 5.0
        smooth_kernel (int): Kernel size for optional depth smoothing. Default: 3
        compact_mode (bool): If True, output 5 channels (normals+edge) instead of 7. Default: True
        
    Input:
        depth: Tensor of shape [B, 1, H, W] - normalized depth map in [0, 1]
        
    Output:
        dict with keys:
            - "geo_prior": [B, 5, H, W] if compact_mode else [B, 7, H, W]
            - "normal": [B, 3, H, W] - surface normal vectors (nx, ny, nz)
            - "edge": [B, 1, H, W] - edge strength map
            - "gradient": [B, 2, H, W] - raw gradients (grad_x, grad_y)
            - "quality": [B, 1, H, W] - quality score based on local variance
            
    Example:
        >>> geo_gen = GeometryPriorGenerator()
        >>> depth = torch.rand(2, 1, 64, 64)  # Batch of 2 depth maps
        >>> priors = geo_gen(depth)
        >>> print(priors["geo_prior"].shape)  # torch.Size([2, 5, 64, 64])
        >>> print(priors["normal"].shape)     # torch.Size([2, 3, 64, 64])
    """

    def __init__(
        self,
        eps: float = 1e-6,
        grad_clip: float = 5.0,
        smooth_kernel: int = 3,
        compact_mode: bool = True,
    ) -> None:
        """Initialize geometry prior generator with Sobel kernels and smoothing."""
        super().__init__()
        
        if smooth_kernel % 2 == 0 or smooth_kernel < 1:
            raise ValueError(f"smooth_kernel must be odd and positive, got {smooth_kernel}")
        
        self.eps = eps
        self.grad_clip = grad_clip
        self.compact_mode = compact_mode
        
        # Sobel kernels for gradient extraction (registered as buffers for automatic device handling)
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], 
                                [-2.0, 0.0, 2.0], 
                                [-1.0, 0.0, 1.0]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], 
                                [0.0, 0.0, 0.0], 
                                [1.0, 2.0, 1.0]], dtype=torch.float32)
        
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False)
        
        # Optional smoothing kernel to suppress noise (mimic median filter effect)
        if smooth_kernel > 1:
            kernel = torch.ones((1, 1, smooth_kernel, smooth_kernel), dtype=torch.float32)
            kernel /= smooth_kernel * smooth_kernel
            self.register_buffer("smooth_kernel", kernel, persistent=False)
        else:
            self.register_buffer("smooth_kernel", torch.tensor([]), persistent=False)

    def _normalize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Normalize depth to [0, 1] per batch to stabilize gradients across different scenes.
        
        📚 八股知识点: 数据归一化
        为什么需要归一化？
        1. 不同场景的深度范围差异大（室内0-5m vs 室外0-100m）
        2. Sobel梯度对绝对值敏感，归一化后梯度尺度统一
        3. 防止后续ReLU/Sigmoid饱和
        """
        d_min = depth.amin(dim=(-2, -1), keepdim=True)  # [B, 1, 1, 1]
        d_max = depth.amax(dim=(-2, -1), keepdim=True)
        scale = (d_max - d_min).clamp(min=self.eps)
        depth_norm = (depth - d_min) / scale
        
        # Optional smoothing to suppress speckle noise
        if self.smooth_kernel.numel() > 0:
            padding = self.smooth_kernel.shape[-1] // 2
            depth_norm = F.conv2d(depth_norm, self.smooth_kernel, padding=padding)
        
        return depth_norm

    def _compute_gradients(self, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute depth gradients using Sobel operators.
        
        📚 八股知识点: Sobel算子
        Sobel公式: Gx = [[-1,0,1],[-2,0,2],[-1,0,1]] * I
        优点：
        1. 平滑+微分，抗噪声能力强
        2. 对角线权重为2，考虑邻域
        3. 计算高效，无参数
        
        为什么用Sobel而非Laplacian？
        - Sobel提供方向性（可计算法向）
        - Laplacian仅提供二阶导，对噪声更敏感
        """
        grad_x = F.conv2d(depth, self.sobel_x, padding=1)
        grad_y = F.conv2d(depth, self.sobel_y, padding=1)
        
        # Clamp gradients to prevent extreme values in missing depth regions
        # 📌 改进点: ultralytics12缺少此步骤，导致梯度爆炸
        grad_x = torch.clamp(grad_x, -self.grad_clip, self.grad_clip)
        grad_y = torch.clamp(grad_y, -self.grad_clip, self.grad_clip)
        
        return grad_x, grad_y

    def _compute_normals(self, grad_x: torch.Tensor, grad_y: torch.Tensor) -> torch.Tensor:
        """
        Compute surface normal vectors from gradients.
        
        📚 八股知识点: 法向估计
        公式: n = normalize([-∂z/∂x, -∂z/∂y, 1])
        物理意义: 法向向量垂直于表面，指向观察者
        应用: 光照估计、平面检测、几何约束
        
        为什么要归一化？
        - 法向是单位向量（长度为1）
        - 便于后续点积计算（如Lambert光照）
        - 保持数值稳定性
        """
        ones = torch.ones_like(grad_x)
        normal = torch.cat([-grad_x, -grad_y, ones], dim=1)  # [B, 3, H, W]
        return F.normalize(normal, p=2, dim=1, eps=self.eps)

    def _compute_edges(self, grad_x: torch.Tensor, grad_y: torch.Tensor) -> torch.Tensor:
        """
        Compute edge strength as gradient magnitude, normalized to [0, 1].
        
        📚 八股知识点: 边缘检测
        公式: ||∇d|| = sqrt((∂d/∂x)² + (∂d/∂y)²)
        边缘强度高的地方：
        - 物体边界（深度突变）
        - 表面折痕
        - 遮挡区域
        
        为什么归一化到[0,1]？
        - 便于与其他特征融合
        - 避免数值范围差异
        - 可直接用作权重
        """
        edge = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + self.eps)  # [B, 1, H, W]
        
        # Normalize to [0, 1] per batch
        e_min = edge.amin(dim=(-2, -1), keepdim=True)
        e_max = edge.amax(dim=(-2, -1), keepdim=True)
        edge_norm = (edge - e_min) / (e_max - e_min + self.eps)
        
        return edge_norm

    def _compute_quality(self, depth: torch.Tensor, window_size: int = 5) -> torch.Tensor:
        """
        Estimate depth quality based on local variance (lower variance = higher quality).
        
        📚 八股知识点: 深度质量估计
        公式: quality = exp(-σ²), σ² = E[(d - μ)²]
        高质量区域特征：
        - 表面平滑（低方差）
        - 无噪点（无跳变）
        - 连续性好
        
        应用:
        - 自适应融合权重（好深度→高权重）
        - SOLR联动（好深度→强化小目标）
        - 可视化诊断
        """
        padding = window_size // 2
        # Compute local mean and variance
        mean = F.avg_pool2d(depth, window_size, stride=1, padding=padding)
        mean_sq = F.avg_pool2d(depth.pow(2), window_size, stride=1, padding=padding)
        variance = (mean_sq - mean.pow(2)).clamp(min=0.0)
        
        # Quality decreases exponentially with variance
        quality = torch.exp(-variance)
        return quality

    def forward(self, depth: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Generate geometry priors from depth map.
        
        Args:
            depth: [B, 1, H, W] normalized depth in [0, 1]
            
        Returns:
            Dictionary containing:
                - geo_prior: [B, 5/7, H, W] - concatenated priors
                - normal: [B, 3, H, W] - surface normals
                - edge: [B, 1, H, W] - edge strength
                - gradient: [B, 2, H, W] - raw gradients
                - quality: [B, 1, H, W] - quality score
                
        Raises:
            ValueError: If depth shape is not [B, 1, H, W]
        """
        if depth.ndim != 4 or depth.shape[1] != 1:
            raise ValueError(f"Expected depth of shape [B, 1, H, W], got {depth.shape}")
        
        # Step 1: Normalize depth to stabilize gradients
        depth_norm = self._normalize_depth(depth)
        
        # Step 2: Extract Sobel gradients
        grad_x, grad_y = self._compute_gradients(depth_norm)
        
        # Step 3: Compute geometry features
        normals = self._compute_normals(grad_x, grad_y)  # [B, 3, H, W]
        edge = self._compute_edges(grad_x, grad_y)       # [B, 1, H, W]
        gradient = torch.cat([grad_x, grad_y], dim=1)    # [B, 2, H, W]
        quality = self._compute_quality(depth_norm)      # [B, 1, H, W]
        
        # Step 4: Concatenate priors
        if self.compact_mode:
            # Compact: normals(3) + edge(1) + quality(1) = 5 channels
            geo_prior = torch.cat([normals, edge, quality], dim=1)
        else:
            # Full: normals(3) + edge(1) + gradient(2) + quality(1) = 7 channels
            geo_prior = torch.cat([normals, edge, gradient, quality], dim=1)
        
        return {
            "geo_prior": geo_prior,
            "normal": normals,
            "edge": edge,
            "gradient": gradient,
            "quality": quality,
        }


# 📚 八股扩展: 思考题
"""
1. 为什么GeometryPriorGenerator不用可学习参数？
   答: (1) Sobel算子是经典边缘检测算子,已经过验证有效
       (2) 无参数意味着不需要训练,可即插即用
       (3) 减少过拟合风险,提升泛化性
       (4) 降低计算复杂度,不增加反向传播负担

2. 如果深度图有大面积缺失,会发生什么？
   答: (1) 缺失区域梯度为0,法向为[0,0,1](垂直向外)
       (2) 质量评分会很低(方差大)
       (3) 通过quality权重自动抑制缺失区域的影响
       (4) 可在数据预处理阶段用inpainting填补

3. 如何验证GeometryPriorGenerator的正确性？
   答: (1) 可视化: 将normal/edge/quality保存为图片检查
       (2) 数值检验: 法向模长应接近1, edge应在[0,1]
       (3) 边界测试: 输入全0/全1深度图,检查输出合理性
       (4) 对比验证: 与OpenCV的Sobel结果对比

4. compact_mode什么时候用？
   答: (1) 推理阶段优先用compact(节省显存和计算)
       (2) 训练初期用compact(快速验证架构)
       (3) 需要详细诊断时用full mode(保留gradient)
       (4) 根据ablation实验决定最终配置
"""
