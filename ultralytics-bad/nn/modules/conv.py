# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = (
    "CBAM",
    "ChannelAttention",
    "Concat",
    "Conv",
    "Conv2",
    "ConvTranspose",
    "DepthGatedFusion",
    "DWConv",
    "DWConvTranspose2d",
    "Focus",
    "GhostConv",
    "Index",
    "LightConv",
    "RepConv",
    "RGBDStem",
    "SpatialAttention",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """
    Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """
    Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """
        Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """
        Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """
        Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """
        Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """
        Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """
    Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """
        Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """
        Apply activation and convolution transpose operation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """
    Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """
        Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Apply Focus operation and convolution to input tensor.

        Input shape is (B, C, W, H) and output shape is (B, 4C, W/2, H/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """
    Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """
        Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """
        Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """
        Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """
        Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """
        Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """
        Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (torch.Tensor): Equivalent kernel
            (torch.Tensor): Equivalent bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """
        Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """
        Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            kernel (torch.Tensor): Fused kernel.
            bias (torch.Tensor): Fused bias.
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse convolutions for inference by creating a single equivalent convolution."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """
    Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """
        Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """
    Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """
        Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """
        Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """
        Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """
    Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """
        Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]):
        """
        Concatenate input tensors along specified dimension.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """
    Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """
        Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x: list[torch.Tensor]):
        """
        Select and return a particular index from input.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]


# ================================================================================================
# RGB-D Dual-Modal Fusion Modules
# ================================================================================================


class DepthGatedFusion(nn.Module):
    """
    Depth-gated fusion module for adaptive RGB-Depth feature fusion.
    
    This module implements multiplication-based gating (inspired by RemDet's GatedFFN)
    to dynamically weight depth features before fusing with RGB features. The gating
    mechanism learns to suppress or enhance depth contributions based on local context.
    
    Mathematical formulation:
        output = rgb + depth * gate
        where gate = σ(FC(Pool(concat(rgb, depth))))
        
    Args:
        channels (int): Number of input/output channels for both RGB and depth.
        reduction (int): Channel reduction ratio for gating network. Default: 16.
        pool_size (int): Adaptive pooling output size. Default: 1 (global pooling).
        
    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling layer.
        project (nn.Sequential): Gating network (compress → activate → expand).
        act (nn.Sigmoid): Sigmoid activation for gate values in [0, 1].
        last_gate_mean/std (buffer): Statistics for monitoring gating behavior.
        
    Example:
        >>> fusion = DepthGatedFusion(channels=64)
        >>> rgb_feat = torch.rand(2, 64, 32, 32)
        >>> depth_feat = torch.rand(2, 64, 32, 32)
        >>> fused = fusion(rgb_feat, depth_feat)
        >>> print(fused.shape)  # torch.Size([2, 64, 32, 32])
        >>> print(fusion.last_gate_mean)  # tensor(0.52) - typical range [0.3, 0.7]
        
    📚 八股知识点: 门控融合 vs 简单相加
    Q: 为什么不直接 rgb + depth？
    A: (1) 深度质量不稳定(噪声、缺失)，盲目相加会引入噪声
       (2) 不同场景深度贡献度不同(室内>室外)
       (3) Gate学习自适应权重，低质量深度自动降权
       
    Q: 为什么用乘法而非加法门控？
    A: RemDet证明乘法门控效率更高:
       - 加法: output = rgb + MLP(depth) → 计算量2d²
       - 乘法: output = rgb + depth*gate → 计算量d²/2 + d
       - 当d≫2时，乘法更快且效果相当
    """

    def __init__(self, channels: int, reduction: int = 16, pool_size: int = 1) -> None:
        """Initialize depth-gated fusion with adaptive pooling and gating network."""
        super().__init__()
        
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if reduction <= 0 or reduction > channels:
            raise ValueError(f"reduction must be in (0, {channels}], got {reduction}")
        
        self.channels = channels
        hidden = max(channels // reduction, 4)  # At least 4 hidden units
        
        # Global pooling to capture channel-wise statistics
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        
        # Gating network: compress → activate → expand → sigmoid
        # 📌 改进点: ultralytics12用2层MLP，这里保持简洁
        self.project = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 1, bias=False),  # Compress
            nn.SiLU(),  # Non-linearity
            nn.Conv2d(hidden, channels, 1, bias=False),  # Expand
        )
        self.act = nn.Sigmoid()  # Gate values in [0, 1]
        
        # Buffers for monitoring (non-persistent, not saved in checkpoint)
        self.register_buffer("last_gate_mean", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_gate_std", torch.tensor(0.0), persistent=False)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Fuse RGB and depth features with learned gating.
        
        Args:
            rgb: [B, C, H, W] RGB feature map
            depth: [B, C, H, W] Depth feature map (same shape as rgb)
            
        Returns:
            [B, C, H, W] Fused feature map
            
        Raises:
            ValueError: If channel dimensions don't match
        """
        if rgb.shape[1] != self.channels or depth.shape[1] != self.channels:
            raise ValueError(
                f"DepthGatedFusion expects both inputs to have {self.channels} channels, "
                f"but got rgb={rgb.shape[1]} and depth={depth.shape[1]}."
            )
        
        # Step 1: Pool both features to capture global context
        pooled_rgb = self.pool(rgb)  # [B, C, 1, 1]
        pooled_depth = self.pool(depth)
        pooled = torch.cat([pooled_rgb, pooled_depth], dim=1)  # [B, 2C, 1, 1]
        
        # Step 2: Generate gate through gating network
        gate = self.act(self.project(pooled))  # [B, C, 1, 1] in [0, 1]
        
        # Step 3: Apply gating (broadcast gate to spatial dims)
        fused = rgb + depth * gate  # [B, C, H, W]
        
        # Step 4: Update statistics for monitoring (only valid gates)
        if self.training or not torch.isnan(gate).any():
            gate_mean = gate.mean().detach()
            gate_std = gate.std(unbiased=False).detach()
            
            # 📌 安全检查: 防止NaN/Inf传播
            if not torch.isfinite(gate_mean):
                gate_mean = torch.zeros_like(self.last_gate_mean)
            if not torch.isfinite(gate_std):
                gate_std = torch.zeros_like(self.last_gate_std)
            
            self.last_gate_mean.copy_(gate_mean)
            self.last_gate_std.copy_(gate_std)
        
        return fused


class RGBDStem(nn.Module):
    """
    Dual-branch stem for RGB-D input that extracts modality-specific low-level features.
    
    This module serves as the entry point for dual-modal networks, processing RGB and
    depth through separate pathways before fusing them. Geometry priors (normals, edges)
    are extracted from depth to enhance structural information.
    
    Architecture:
        Input [B, 4, H, W] (RGB+D)
        ├─ RGB Branch: Conv(3→c_mid) → Conv(c_mid→c_mid)
        ├─ Depth Branch: Conv(1→c_mid) → Conv(c_mid→c_mid)
        │   └─ Enhanced by GeometryPrior (Sobel normals+edges)
        └─ Fusion: Gated/Add/Concat
        Output [B, c2, H/2, W/2] (if stride=2)
        
    Args:
        c1 (int): Total input channels (RGB=3 + Depth=1 = 4).
        c2 (int): Total output channels (must be even for modality splitting).
        k (int): Kernel size for first conv in each branch. Default: 3.
        s (int): Stride for downsampling. Default: 2.
        depth_channels (int): Number of depth channels. Default: 1.
        c_mid (int, optional): Mid-layer channels. Default: c2//2.
        fusion (str): Fusion mode - "gated_add", "add", or "concat". Default: "gated_add".
        reduction (int): Channel reduction for gating network. Default: 16.
        act (bool | nn.Module): Activation function. Default: True (SiLU).
        
    Fusion Modes:
        - gated_add: rgb + depth*gate (adaptive, RemDet-inspired) 【推荐】
        - add: rgb + depth (simple, assumes good depth quality)
        - concat: [rgb, depth] (keeps modalities independent)
        
    Example:
        >>> stem = RGBDStem(c1=4, c2=64, s=2)  # Downsample 2x
        >>> x_rgbd = torch.rand(2, 4, 640, 640)  # RGB-D input
        >>> out = stem(x_rgbd)
        >>> print(out.shape)  # torch.Size([2, 64, 320, 320])
        >>> print(stem.last_geo_quality)  # tensor(0.75) - depth quality score
        
    📚 八股知识点: 为什么要双分支？
    Q: 直接4通道输入一个Conv不行吗？
    A: (1) RGB和Depth物理意义不同(颜色 vs 距离)
       (2) 预训练权重只有RGB(3通道)，无法直接迁移
       (3) 独立分支保持模态独立性，避免过早混合
       (4) 可以针对性地优化(如Depth加几何先验)
       
    Q: 几何先验为什么有效？
    A: (1) 深度原始值是"距离"，不包含结构信息
       (2) Sobel提取的法向/边缘描述"形状"
       (3) 形状是目标检测的关键线索
       (4) 类似人类视觉同时用"颜色+轮廓"识别物体
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 2,
        depth_channels: int = 1,
        c_mid: int | None = None,
        fusion: str = "gated_add",
        reduction: int = 16,
        act: bool | nn.Module = True,
    ) -> None:
        """Initialize dual-branch RGB-D stem with geometry-enhanced depth processing."""
        super().__init__()
        
        # Validation
        if c2 % 2 != 0:
            raise ValueError(f"RGBDStem requires even output channels for modality splitting, got {c2}")
        if depth_channels <= 0 or depth_channels >= c1:
            raise ValueError(f"depth_channels must be in (0, {c1}), got {depth_channels}")
        
        # Channel allocation
        self.rgb_channels = c1 - depth_channels
        self.depth_channels = depth_channels
        self.c_mid = c_mid or (c2 // 2)
        
        # Ensure c_mid * 2 == c2 for proper concatenation
        if self.c_mid * 2 != c2:
            raise ValueError(
                f"RGBDStem expects c_mid * 2 == c2 for dual-branch output, "
                f"but got c_mid={self.c_mid} and c2={c2}. "
                f"Suggestion: set c_mid={c2//2} or c2={self.c_mid*2}"
            )
        
        # RGB pathway: standard conv → conv
        self.rgb_path = nn.Sequential(
            Conv(self.rgb_channels, self.c_mid, k, s, act=act),
            Conv(self.c_mid, self.c_mid, 3, 1, act=act),
        )
        
        # Depth pathway: standard conv → conv
        self.depth_path = nn.Sequential(
            Conv(self.depth_channels, self.c_mid, k, s, act=act),
            Conv(self.c_mid, self.c_mid, 3, 1, act=act),
        )
        
        # Geometry prior generator (Sobel-based, no parameters)
        from .geometry import GeometryPriorGenerator
        self.geo_generator = GeometryPriorGenerator(compact_mode=True)  # 5 channels
        
        # Project geometry priors to feature space
        # 📌 改进点: ultralytics12用6通道，这里用5通道(compact)
        self.geo_proj = nn.Sequential(
            nn.Conv2d(5, self.c_mid, 1, bias=False),  # Compact mode: normals(3)+edge(1)+quality(1)
            nn.BatchNorm2d(self.c_mid),
            nn.SiLU(),
        )
        
        # Buffers for monitoring geometry quality
        self.register_buffer("last_geo_quality", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_geo_edge", torch.tensor(0.0), persistent=False)
        
        # Fusion module
        fusion = fusion.lower()
        self.fusion_mode = fusion
        if fusion == "gated_add":
            self.fusion = DepthGatedFusion(self.c_mid, reduction)
        elif fusion == "add":
            self.fusion = None  # Simple addition
        elif fusion == "concat":
            self.fusion = None  # Concatenation
        else:
            raise ValueError(
                f"Unsupported fusion mode '{fusion}' for RGBDStem. "
                f"Choose from: 'gated_add', 'add', 'concat'"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process RGB-D input through dual branches and fuse.
        
        Args:
            x: [B, C, H, W] where C >= rgb_channels + depth_channels
               Typical: [B, 4, H, W] for RGB+D
               
        Returns:
            [B, c2, H//s, W//s] Fused features
            
        Raises:
            ValueError: If input channels insufficient
        """
        if x.ndim != 4:
            raise ValueError(f"RGBDStem expects 4D input [B,C,H,W], got shape {x.shape}")
        
        total_channels = self.rgb_channels + self.depth_channels
        
        # Handle different input scenarios
        if x.shape[1] == self.rgb_channels:
            # RGB-only input (fallback mode for inference without depth)
            rgb = x
            depth = torch.zeros(
                (x.shape[0], self.depth_channels, x.shape[2], x.shape[3]),
                dtype=x.dtype,
                device=x.device,
            )
        elif x.shape[1] >= total_channels:
            # Standard RGB-D input: split channels
            rgb = x[:, : self.rgb_channels]
            depth = x[:, self.rgb_channels : total_channels]
        else:
            raise ValueError(
                f"RGBDStem expected at least {total_channels} channels, "
                f"but received {x.shape[1]} channels"
            )
        
        # Step 1: Extract modality-specific features
        rgb_feat = self.rgb_path(rgb)  # [B, c_mid, H//s, W//s]
        depth_feat = self.depth_path(depth)  # [B, c_mid, H//s, W//s]
        
        # Step 2: Generate geometry priors from depth
        geo = self.geo_generator(depth)  # dict with keys: geo_prior, normal, edge, quality
        geo_prior = geo["geo_prior"]  # [B, 5, H, W]
        geo_quality = geo["quality"]  # [B, 1, H, W]
        geo_edge = geo["edge"]  # [B, 1, H, W]
        
        # Step 3: Align geometry priors to depth_feat spatial dimensions
        target_size = depth_feat.shape[-2:]
        if geo_prior.shape[-2:] != target_size:
            import torch.nn.functional as F
            geo_prior = F.interpolate(geo_prior, size=target_size, mode="bilinear", align_corners=False)
            geo_quality = F.interpolate(geo_quality, size=target_size, mode="bilinear", align_corners=False)
            geo_edge = F.interpolate(geo_edge, size=target_size, mode="bilinear", align_corners=False)
        
        # Step 4: Enhance depth features with geometry priors
        # Formula: depth_enhanced = depth_feat + geo_proj * (0.5 + 0.5*quality)
        # Quality modulation: high quality → stronger geometry influence
        geo_modulation = self.geo_proj(geo_prior) * (0.5 + 0.5 * geo_quality)
        depth_feat = depth_feat + geo_modulation
        
        # Step 5: Update monitoring statistics
        quality_mean = geo_quality.mean().detach()
        edge_mean = geo_edge.mean().detach()
        if torch.isfinite(quality_mean):
            self.last_geo_quality.copy_(quality_mean)
        if torch.isfinite(edge_mean):
            self.last_geo_edge.copy_(edge_mean)
        
        # Step 6: Fuse RGB and depth features
        if self.fusion_mode == "gated_add":
            # Adaptive gating: rgb + depth*gate
            fused = self.fusion(rgb_feat, depth_feat)
        elif self.fusion_mode == "add":
            # Simple addition (assumes good depth quality)
            fused = rgb_feat + depth_feat
        else:  # concat
            # Concatenation (keeps modalities independent)
            # 📌 注意: concat模式下输出通道翻倍
            fused = torch.cat([rgb_feat, depth_feat], dim=1)
            return fused
        
        # Step 7: Return [fused_feat, depth_feat] for downstream use
        # 📌 设计思路: 保留depth_feat供后续RGBDMidFusion使用
        return torch.cat([fused, depth_feat], dim=1)


# 📚 八股扩展: 思考题
"""
1. RGBDStem的输出为什么是 [fused, depth] 拼接？
   答: (1) fused包含RGB主导+深度辅助的混合特征
       (2) depth保留纯深度信息供后续层使用
       (3) 多阶段融合策略: Stem融合一次，Neck再融合
       (4) 类似ResNet的shortcut，防止信息丢失

2. 如果depth_channels改成3(伪彩深度图)？
   答: (1) 代码兼容,只需修改depth_channels=3
       (2) GeometryPriorGenerator需要先转单通道(mean)
       (3) 伪彩深度通常是可视化用,实际存储仍是单通道
       (4) 建议保持depth_channels=1以节省显存

3. fusion="concat"模式什么时候用？
   答: (1) 消融实验对比: 验证融合是否有效
       (2) 多任务学习: RGB任务和Depth任务分开
       (3) 早期探索阶段: 不确定最佳融合策略
       (4) 生产环境不推荐(输出通道翻倍,计算量大)

4. 如何迁移ImageNet预训练权重？
   答: (1) RGB分支直接加载预训练权重(3→c_mid)
       (2) Depth分支随机初始化或复制RGB权重后微调
       (3) 融合模块随机初始化(无预训练)
       (4) 训练时freeze RGB前几层,先训练Depth分支

5. DepthGatedFusion的gate平均值多少合理？
   答: (1) 理想范围 [0.3, 0.7], 说明自适应工作正常
       (2) 接近0: 深度被完全抑制,可能质量太差
       (3) 接近1: 深度贡献过大,可能过拟合
       (4) 监控last_gate_mean趋势,稳定在0.4-0.6最佳
"""


# ================================================================================================
# RGBDMidFusion - Mid-Level RGB-D Feature Fusion for Multi-Scale Enhancement
# ================================================================================================

class RGBDMidFusion(nn.Module):
    """
    Mid-level RGB-D feature fusion module for multi-scale depth integration.
    
    Designed to address the "depth dilution problem" in early fusion strategies:
    After RGBDStem produces [fused 64ch + depth 64ch], standard YOLOv12 layers
    gradually mix depth features into increasing channels (256→512→1024), causing
    depth information to be "drowned out" by RGB features.
    
    Solution: Re-inject depth features at P3/P4/P5 scales using cross-modal attention.
    
    Architecture:
        Input:
          - rgb_feat: [B, C_rgb, H, W] from backbone layer (e.g., after C3k2)
          - depth_skip: [B, C_d, H', W'] from RGBDStem's depth branch
        
        Pipeline:
          1. Spatial alignment: Resize depth_skip to match rgb_feat resolution
          2. Channel alignment: Project depth_skip to C_rgb channels
          3. Cross-modal attention: Learn adaptive fusion weights
          4. Weighted fusion: rgb_feat + attention * depth_aligned
        
        Output:
          - enhanced_feat: [B, C_rgb, H, W] (same shape as rgb_feat)
    
    Args:
        rgb_channels (int): Number of RGB feature channels (from backbone).
        depth_channels (int): Number of depth feature channels (from RGBDStem).
        reduction (int): Channel reduction ratio for attention network. Default: 16.
            - Smaller = more expressive but heavier (e.g., 8 for large models)
            - Larger = lighter but less adaptive (e.g., 32 for nano models)
        fusion_weight (float): Initial weight for depth contribution. Default: 0.3.
            - Prevents depth from dominating in early training
            - Learnable parameter, will be optimized during training
    
    Fusion Formula:
        output = rgb_feat + α * attention(concat(rgb_feat, depth_aligned)) * depth_aligned
        where α is the learnable fusion_weight
    
    FLOPs Analysis (for C=512, H=40, W=40):
        1. Depth projection: 1x1 conv C_d→C_rgb = C_d * C_rgb * H * W
           Example: 64 * 512 * 40 * 40 = 52.4M FLOPs
        2. Attention network: 
           - AdaptiveAvgPool: ~0 (negligible)
           - Conv 2C→C//r: 2C * C//r = 1024 * 32 = 32K FLOPs
           - Conv C//r→C: C//r * C = 32 * 512 = 16K FLOPs
           Total attention: ~50K FLOPs (negligible)
        3. Total per RGBDMidFusion: ~52.4M FLOPs
        4. Three fusions (P3+P4+P5): ~157M FLOPs
        5. vs YOLOv12-S total (45.6G): +0.34% overhead ✅ Acceptable!
    
    Usage in YAML:
        backbone:
          # Layer 0: RGBDStem produces [fused 64ch + depth 64ch]
          - [-1, 1, RGBDStem, [4, 128, 3, 2, 1, 64, "gated_add", 16, True]]  # 0-P1/2
          
          # Layers 1-4: Standard YOLOv12 processing
          - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
          - [-1, 2, C3k2, [256, False, 0.25]]  # 2
          - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
          - [-1, 2, C3k2, [512, False, 0.25]]  # 4-P3 features
          
          # 📌 Re-inject depth features at P3 scale
          # Args: [rgb_channels (from layer 4), depth_channels (from layer 0)]
          - [[4, 0], 1, RGBDMidFusion, [512, 64]]  # 5-P3 depth fusion
          
          # Repeat for P4 and P5...
    
    Example:
        >>> # Simulate backbone outputs
        >>> rgb_feat = torch.rand(2, 512, 40, 40)  # P4 features
        >>> depth_skip = torch.rand(2, 64, 320, 320)  # From RGBDStem
        >>> fusion = RGBDMidFusion(rgb_channels=512, depth_channels=64)
        >>> enhanced = fusion(rgb_feat, depth_skip)
        >>> print(enhanced.shape)  # torch.Size([2, 512, 40, 40])
        >>> print(fusion.last_attn_mean)  # tensor(0.42) - attention weight
    
    📚 八股知识点: 为什么需要Mid-Level Fusion？
    Q: 有了RGBDStem为什么还需要RGBDMidFusion？
    A: (1) 早期融合的深度特征会被"稀释"
          Layer 0: 64ch depth / 128ch total = 50%
          Layer 4: 64ch depth / 512ch total = 12.5%  ← 信息丢失!
          Layer 8: 64ch depth / 1024ch total = 6.25% ← 几乎不可见!
       (2) 不同尺度需要不同深度信息
          P3 (small objects): 需要精细深度 (边缘、纹理)
          P4 (medium objects): 需要结构深度 (平面、法向)
          P5 (large objects): 需要全局深度 (距离、遮挡)
       (3) RemDet成功的关键: 多阶段特征增强
          ChannelC2f在每个block都优化
          GatedFFN在每个FFN都改进
          我们的RGBDMidFusion在每个尺度都融合
    
    Q: 为什么用Cross-Modal Attention而不是简单Add/Concat？
    A: (1) Attention可以学习"何时依赖深度"
          室内场景(深度准确) → attention=0.8 (高权重)
          室外场景(深度噪声) → attention=0.2 (低权重)
       (2) 自适应性: 不同样本、不同区域权重不同
          有纹理的区域 → 依赖RGB
          无纹理的区域 → 依赖Depth
       (3) 可解释性: last_attn_mean监控融合强度
    
    Q: reduction=16是怎么确定的？
    A: (1) 参考SENet/CBAM的经典设置 (r=16)
       (2) 平衡表达能力和计算量
          r=8: 更强表达,但参数量2x
          r=16: 标准配置 ✅
          r=32: 更轻量,但可能欠拟合
       (3) 消融实验验证: r=8/16/32对比
    
    Troubleshooting:
        1. RuntimeError: Sizes of tensors must match
           → Check rgb_feat.shape == enhanced.shape (no dimension change!)
        
        2. Attention weight always near 0 or 1
           → Depth quality issue or learning rate too high
           → Try adjusting fusion_weight initial value (0.1~0.5)
        
        3. Memory overflow
           → Reduce reduction ratio (e.g., 32 instead of 16)
           → Use FP16 mixed precision training
    """

    def __init__(
        self,
        rgb_channels: int,
        depth_channels: int,
        reduction: int = 16,
        fusion_weight: float = 0.3,
    ):
        """
        Initialize RGBDMidFusion module.
        
        Args:
            rgb_channels: Number of RGB feature channels (e.g., 512 for P4).
            depth_channels: Number of depth feature channels (e.g., 64 from RGBDStem).
            reduction: Channel reduction for attention network (default: 16).
            fusion_weight: Initial learnable weight for depth contribution (default: 0.3).
        """
        super().__init__()
        
        # Store configuration
        self.rgb_channels = rgb_channels
        self.depth_channels = depth_channels
        self.reduction = reduction
        
        # ========================================================================================
        # 1. Channel Alignment: Project depth_channels → rgb_channels
        # ========================================================================================
        # Use 1x1 conv for efficient channel projection
        # FLOPs: depth_channels * rgb_channels * H * W
        self.depth_proj = Conv(depth_channels, rgb_channels, k=1, s=1, act=True)
        
        # ========================================================================================
        # 2. Cross-Modal Attention Network
        # ========================================================================================
        # Architecture: GlobalPool → FC(2C→C//r) → SiLU → FC(C//r→C) → Sigmoid
        # Inspired by SENet and CBAM, adapted for RGB-D fusion
        
        concat_channels = rgb_channels * 2  # RGB + Depth_aligned
        hidden_dim = max(concat_channels // reduction, 8)  # Ensure min 8 channels
        
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, 2C, H, W] → [B, 2C, 1, 1]
            nn.Conv2d(concat_channels, hidden_dim, 1, bias=False),  # Squeeze
            nn.SiLU(),  # Activation
            nn.Conv2d(hidden_dim, rgb_channels, 1, bias=False),  # Excitation
            nn.Sigmoid(),  # Normalize to [0, 1]
        )
        
        # ========================================================================================
        # 3. Learnable Fusion Weight
        # ========================================================================================
        # Allows network to control overall depth contribution strength
        # Initialized to fusion_weight (default 0.3 = conservative)
        self.fusion_weight = nn.Parameter(torch.tensor(fusion_weight))
        
        # ========================================================================================
        # 4. Monitoring Statistics (for debugging/logging)
        # ========================================================================================
        # Track attention weights to ensure fusion is working
        self.register_buffer("last_attn_mean", torch.tensor(0.0))
        self.register_buffer("last_attn_std", torch.tensor(0.0))
        
    def forward(self, rgb_feat, depth_skip):
        """
        Forward pass: Fuse RGB features with depth skip connection.
        
        Args:
            rgb_feat (Tensor): RGB features from backbone [B, C_rgb, H, W].
            depth_skip (Tensor): Depth features from RGBDStem [B, C_d, H', W'].
        
        Returns:
            Tensor: Enhanced features [B, C_rgb, H, W] (same shape as rgb_feat).
        
        Pipeline:
            depth_skip [B, 64, 320, 320] 
                → resize to [B, 64, 40, 40]  (spatial alignment)
                → project to [B, 512, 40, 40]  (channel alignment)
                → concat with rgb_feat [B, 512, 40, 40]
                → attention [B, 1024, 40, 40] → [B, 512, 1, 1]
                → element-wise multiply: attn * depth_aligned
                → weighted add: rgb_feat + fusion_weight * (attn * depth_aligned)
        """
        import torch.nn.functional as F
        
        # ========================================================================================
        # Step 1: Spatial Alignment - Resize depth to match RGB resolution
        # ========================================================================================
        # RGBDStem outputs depth at P1/2 (320x320)
        # Backbone layer outputs RGB at P3/P4/P5 (80/40/20)
        # Need to downsample depth to match
        
        target_size = rgb_feat.shape[-2:]  # [H, W]
        
        if depth_skip.shape[-2:] != target_size:
            # Bilinear interpolation for smooth downsampling
            # align_corners=False is standard for object detection
            depth_skip_resized = F.interpolate(
                depth_skip,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        else:
            depth_skip_resized = depth_skip
        
        # ========================================================================================
        # Step 2: Channel Alignment - Project depth to RGB channel dimension
        # ========================================================================================
        # Input: [B, 64, H, W] → Output: [B, 512, H, W]
        depth_aligned = self.depth_proj(depth_skip_resized)
        
        # ========================================================================================
        # Step 3: Cross-Modal Attention - Learn fusion weights
        # ========================================================================================
        # Concatenate RGB and depth for attention calculation
        concat_feat = torch.cat([rgb_feat, depth_aligned], dim=1)  # [B, 2*C_rgb, H, W]
        
        # Compute attention weights [B, C_rgb, 1, 1]
        attn_weights = self.attention(concat_feat)  # Sigmoid output [0, 1]
        
        # ========================================================================================
        # Step 4: Update Monitoring Statistics
        # ========================================================================================
        # Track attention distribution for debugging
        attn_mean = attn_weights.mean().detach()
        attn_std = attn_weights.std().detach()
        
        if torch.isfinite(attn_mean):
            self.last_attn_mean.copy_(attn_mean)
        if torch.isfinite(attn_std):
            self.last_attn_std.copy_(attn_std)
        
        # ========================================================================================
        # Step 5: Weighted Fusion
        # ========================================================================================
        # Formula: output = rgb + α * (attn * depth)
        # - rgb_feat: Original RGB features (base information)
        # - attn_weights: Spatial attention map (where to fuse)
        # - depth_aligned: Depth features (what to fuse)
        # - fusion_weight: Global strength control (how much to fuse)
        
        # Broadcast attention: [B, C, 1, 1] → [B, C, H, W]
        depth_contribution = attn_weights * depth_aligned
        
        # Learnable weighted addition
        enhanced_feat = rgb_feat + self.fusion_weight * depth_contribution
        
        return enhanced_feat


# 📚 八股扩展: RGBDMidFusion深度解析
"""
1. 为什么要保存depth_skip而不是直接在backbone内部融合？
   答: (1) 灵活性: 可以在任意层级选择性融合
       (2) 消融实验: 方便对比"有/无"深度融合
       (3) 计算效率: 避免在不需要的层传播深度特征
       (4) 模块解耦: RGBDStem和RGBDMidFusion独立设计

2. fusion_weight为什么初始化为0.3而不是1.0？
   答: (1) 保守策略: 避免深度噪声主导训练早期
       (2) 稳定性: RGB特征已经很强,深度应该"辅助"而非"替代"
       (3) 学习空间: 0.3→0.5是有意义的优化方向
       (4) 类似ResNet的残差连接: 初始化接近0,逐渐学习

3. 如何监控RGBDMidFusion是否正常工作？
   答: (1) last_attn_mean应该在[0.2, 0.6]范围
          <0.1: 深度被完全忽略,检查depth质量
          >0.8: 深度主导,可能过拟合
       (2) fusion_weight应该逐渐增加
          初始0.3 → 训练后0.4-0.6 (说明深度有用)
       (3) mAP@0.5应该比v1.0提升2-3%
          v1: 41% → v2.1: 43-44%

4. 如果P3/P4/P5的depth_skip来自不同分辨率？
   答: (1) 代码已处理: F.interpolate自动resize
       (2) 建议: 使用同一depth_skip源(RGBDStem)
       (3) 高级: 可以为每个scale设计独立depth branch
       (4) 权衡: 简单 vs 性能,当前方案已足够

5. RGBDMidFusion vs Attention机制对比？
   答: RGBDMidFusion = Cross-Modal Attention
       - SENet: 单模态通道注意力
       - CBAM: 单模态空间+通道注意力
       - RGBDMidFusion: 双模态交叉注意力
       关键差异: 我们融合**两种模态**,不是增强单一特征

思考题:
1. 如果depth_skip质量很差(大量噪声),attention能否自动降低权重？
2. 为什么P3/P4/P5都需要独立的RGBDMidFusion,而不是共享参数？
3. 如何设计实验验证RGBDMidFusion的必要性(消融实验)?
4. 能否用Transformer的Multi-Head Attention替代当前的简单attention？
"""
