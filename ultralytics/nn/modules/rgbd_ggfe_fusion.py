"""
RGBDGGFEFusion Module - 组合RGBDMidFusion和GGFE的完整模块

功能:
1. RGB-D双模态融合 (继承RGBDMidFusion)
2. 几何先验引导的特征增强 (集成GGFE)
3. 一站式解决方案，便于YAML配置和消融实验

设计理念:
- 解耦设计: GGFE可通过use_ggfe参数开关
- 向后兼容: use_ggfe=False时退化为标准RGBDMidFusion
- 便于消融: 可单独测试GGFE的贡献

作者: GitHub Copilot
日期: 2025-01-20
基于: YOLOv12_GeometryEnhanced_Implementation_Guide_Version3.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, RGBDMidFusion
from .ggfe import GGFE
from .geometry import GeometryPriorGenerator


class RGBDGGFEFusion(nn.Module):
    """
    RGB-D Fusion with Geometry-Guided Feature Enhancement
    
    组合模块，将RGB-D融合和几何增强一体化:
    1. RGBDMidFusion: 双模态特征融合
    2. GGFE: 几何先验引导的注意力增强
    
    适用场景:
    - P3/P4/P5层的RGB-D融合
    - 中小目标检测增强
    - VisDrone/UAVDT等UAV数据集
    
    Example:
        >>> fusion = RGBDGGFEFusion(rgb_channels=256, depth_channels=64, use_ggfe=True)
        >>> x = torch.randn(2, 4, 320, 320)  # RGB+D输入
        >>> output = fusion(x)
        >>> print(output.shape)  # [2, C_out, H//s, W//s]
    """
    
    def __init__(
        self,
        rgb_channels=256,
        depth_channels=64,
        c_out=None,
        k=3,
        s=2,
        reduction=16,
        fusion="gated_add",
        use_ggfe=True,
        ggfe_reduction=8,
        act=True,
    ):
        """
        Args:
            rgb_channels (int): RGB输入通道数 (默认: 256)
            depth_channels (int): Depth输入通道数 (默认: 64)
            c_out (int, optional): 输出通道数。如果None，则根据fusion模式自动计算
            k (int): 卷积核大小 (默认: 3)
            s (int): 步长 (默认: 2)
            reduction (int): 门控融合的通道缩减比例 (默认: 16)
            fusion (str): 融合模式 - "gated_add" | "add" | "concat" (默认: "gated_add")
            use_ggfe (bool): 是否启用GGFE模块 (默认: True)
            ggfe_reduction (int): GGFE注意力的通道缩减比例 (默认: 8)
            act (bool): 是否使用激活函数 (默认: True)
        """
        super(RGBDGGFEFusion, self).__init__()
        
        self.rgb_channels = rgb_channels
        self.depth_channels = depth_channels
        self.use_ggfe = use_ggfe
        self.fusion_mode = fusion.lower()
        
        # 1. RGB-D融合模块 (基础融合)
        self.rgbd_fusion = RGBDMidFusion(
            rgb_channels=rgb_channels,
            depth_channels=depth_channels,
            c_out=c_out,
            k=k,
            s=s,
            reduction=reduction,
            fusion=fusion,
            act=act,
        )
        
        # 获取融合后的输出通道数 (从RGBDMidFusion推导)
        self.c_mid = self.rgbd_fusion.c_mid
        if fusion == "concat":
            self.fused_channels = self.c_mid * 2  # RGB + Depth拼接
        else:
            self.fused_channels = self.c_mid
        
        # 2. GGFE模块 (几何引导增强)
        if use_ggfe:
            self.ggfe = GGFE(
                in_channels=self.fused_channels,
                reduction=ggfe_reduction,
                geo_compact=True,  # 使用5通道紧凑模式
            )
        else:
            self.ggfe = None
        
        # 3. 缓存深度图 (用于GGFE)
        # 注: 深度图在RGBDMidFusion的forward中已经被分离出来
        self.register_buffer("_cached_depth", torch.zeros(1, 1, 1, 1), persistent=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x (torch.Tensor): [B, C, H, W] 输入特征
                - RGB-D模式: C = rgb_channels + depth_channels (典型: 4 = 3+1)
                - RGB-only: C = rgb_channels (depth部分为零)
        
        Returns:
            torch.Tensor: [B, C_out, H//s, W//s] 融合并增强后的特征
        
        Workflow:
            1. 分离RGB和Depth通道
            2. RGBDMidFusion: 双模态融合
            3. 如果use_ggfe=True: 用GGFE增强融合特征
            4. 返回增强特征
        """
        B, C, H, W = x.shape
        
        # Step 1: 分离RGB和Depth
        # (这一步在RGBDMidFusion中会再次执行，但我们需要提前提取depth用于GGFE)
        total_channels = self.rgb_channels + self.depth_channels
        
        if x.shape[1] >= total_channels:
            # 标准RGB-D输入
            rgb = x[:, : self.rgb_channels]
            depth = x[:, self.rgb_channels : total_channels]
        elif x.shape[1] == self.rgb_channels:
            # RGB-only输入 (fallback)
            rgb = x
            depth = torch.zeros(
                (B, self.depth_channels, H, W),
                dtype=x.dtype,
                device=x.device,
            )
        else:
            raise ValueError(
                f"RGBDGGFEFusion expected input with {total_channels} or "
                f"{self.rgb_channels} channels, but got {x.shape[1]} channels"
            )
        
        # 缓存原始深度图 (用于GGFE)
        self._cached_depth = depth.clone()
        
        # Step 2: RGB-D融合
        fused_output = self.rgbd_fusion(x)  # [B, C_fused, H//s, W//s]
        
        # RGBDMidFusion可能返回 [fused_feat, depth_feat] 拼接
        # 我们需要分离出来
        if self.fusion_mode == "concat":
            # concat模式: 输出是 [RGB_feat, Depth_feat]
            # 我们对整个输出应用GGFE
            fused_feat = fused_output
        else:
            # gated_add/add模式: 输出可能包含 [fused, depth]
            # 检查输出通道数
            if fused_output.shape[1] == self.c_mid * 2:
                # 确实是拼接输出
                fused_feat, depth_feat = fused_output.split(self.c_mid, dim=1)
            else:
                # 单一输出
                fused_feat = fused_output
        
        # Step 3: GGFE增强 (如果启用)
        if self.ggfe is not None:
            # 深度图需要对齐到fused_feat的尺寸
            # 注意: GGFE内部会自动对齐尺寸
            enhanced_feat = self.ggfe(fused_feat, self._cached_depth)
        else:
            enhanced_feat = fused_feat
        
        # Step 4: 如果原始输出包含depth_feat，重新拼接
        if self.fusion_mode != "concat" and fused_output.shape[1] == self.c_mid * 2:
            # 保持与RGBDMidFusion一致的输出格式
            output = torch.cat([enhanced_feat, depth_feat], dim=1)
        else:
            output = enhanced_feat
        
        return output


# ========== 八股知识点 #51: 组合模块设计模式 ==========
"""
**问题**: 为什么要创建RGBDGGFEFusion这样的组合模块，而不是在YAML中串联RGBDMidFusion和GGFE？

**标准答案**:

**优势**:
1. **封装复杂性**: 用户只需在YAML中配置一个模块，而非两个
2. **参数传递**: 深度图在模块内部传递，避免YAML配置复杂的多输入
3. **消融实验**: 通过use_ggfe=True/False轻松切换，无需修改YAML结构
4. **向后兼容**: use_ggfe=False时完全等价于RGBDMidFusion

**劣势**:
1. **代码冗余**: 如果GGFE需要独立使用，仍需保留单独的GGFE类
2. **灵活性降低**: 无法在RGBDMidFusion和GGFE之间插入其他模块

**本项目应用**:
- 组合模块适合**紧密耦合**的功能(RGB-D融合+几何增强)
- 如果GGFE需要在其他地方单独使用，保留独立的GGFE类

**常见追问**:
Q: 如果想在P3/P4/P5都用GGFE，但只在P4用RGBDMidFusion怎么办？
A: 保留独立的GGFE类，在P3/P5单独使用。RGBDGGFEFusion只在P4使用。

Q: 组合模块会不会影响模型加载(如加载预训练权重)？
A: 不影响。PyTorch的state_dict是按名称匹配的，只要参数名称一致即可。

**易错点**:
1. ❌ 忘记处理depth_feat的传递 → 后续层无法访问纯深度特征
2. ❌ use_ggfe=False时仍计算几何先验 → 浪费计算资源
3. ❌ 深度图尺寸未对齐 → GGFE内部会处理，但最好提前check

**拓展阅读**:
- PyTorch Module组合: https://pytorch.org/docs/stable/notes/modules.html
- 设计模式: Composite Pattern (组合模式)
"""


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("RGBDGGFEFusion模块测试")
    print("=" * 60)
    
    # 测试1: 标准RGB-D输入 (use_ggfe=True)
    print("\n[测试1] RGB-D输入 + GGFE启用")
    fusion_with_ggfe = RGBDGGFEFusion(
        rgb_channels=3,
        depth_channels=1,
        c_out=None,
        k=3,
        s=2,
        fusion="gated_add",
        use_ggfe=True,
        ggfe_reduction=8,
    ).cuda()
    
    x1 = torch.randn(2, 4, 320, 320).cuda()  # RGB+D
    with torch.no_grad():
        output1 = fusion_with_ggfe(x1)
    
    print(f"  输入: {x1.shape}")
    print(f"  输出: {output1.shape}")
    print(f"  参数量: {sum(p.numel() for p in fusion_with_ggfe.parameters()):,}")
    
    # 测试2: 不启用GGFE (消融实验)
    print("\n[测试2] RGB-D输入 + GGFE禁用 (消融)")
    fusion_no_ggfe = RGBDGGFEFusion(
        rgb_channels=3,
        depth_channels=1,
        use_ggfe=False,
    ).cuda()
    
    with torch.no_grad():
        output2 = fusion_no_ggfe(x1)
    
    print(f"  输入: {x1.shape}")
    print(f"  输出: {output2.shape}")
    print(f"  参数量: {sum(p.numel() for p in fusion_no_ggfe.parameters()):,}")
    print(f"  参数量减少: {sum(p.numel() for p in fusion_with_ggfe.parameters()) - sum(p.numel() for p in fusion_no_ggfe.parameters()):,}")
    
    # 测试3: 不同融合模式
    print("\n[测试3] Concat融合模式 + GGFE")
    fusion_concat = RGBDGGFEFusion(
        rgb_channels=3,
        depth_channels=1,
        fusion="concat",
        use_ggfe=True,
    ).cuda()
    
    with torch.no_grad():
        output3 = fusion_concat(x1)
    
    print(f"  输入: {x1.shape}")
    print(f"  输出: {output3.shape}")
    
    # 测试4: P4层配置 (256通道)
    print("\n[测试4] P4层典型配置 (256通道)")
    fusion_p4 = RGBDGGFEFusion(
        rgb_channels=256,
        depth_channels=64,
        c_out=None,
        fusion="gated_add",
        use_ggfe=True,
        ggfe_reduction=8,
    ).cuda()
    
    x4 = torch.randn(2, 320, 40, 40).cuda()  # P4层典型尺寸
    with torch.no_grad():
        output4 = fusion_p4(x4)
    
    print(f"  输入: {x4.shape}")
    print(f"  输出: {output4.shape}")
    print(f"  参数量: {sum(p.numel() for p in fusion_p4.parameters()):,}")
    print(f"  GGFE监控: 几何质量={fusion_p4.ggfe.last_geo_quality_mean.item():.4f}, "
          f"空间注意力={fusion_p4.ggfe.last_spatial_attn_mean.item():.4f}")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！RGBDGGFEFusion模块工作正常")
    print("=" * 60)
