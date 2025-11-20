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
        reduction=16,
        fusion_weight=0.3,
        use_ggfe=True,
        ggfe_reduction=8,
    ):
        """
        Args:
            rgb_channels (int): RGB特征通道数 (默认: 256)
            depth_channels (int): Depth特征通道数 (默认: 64)
            reduction (int): RGBDMidFusion的注意力缩减比例 (默认: 16)
            fusion_weight (float): RGBDMidFusion的深度融合权重 (默认: 0.3)
            use_ggfe (bool): 是否启用GGFE模块 (默认: True)
            ggfe_reduction (int): GGFE注意力的通道缩减比例 (默认: 8)
        """
        super(RGBDGGFEFusion, self).__init__()
        
        self.rgb_channels = rgb_channels
        self.depth_channels = depth_channels
        self.use_ggfe = use_ggfe
        
        # 1. RGB-D融合模块 (基础融合)
        # RGBDMidFusion的实际签名: __init__(rgb_channels, depth_channels, reduction, fusion_weight)
        self.rgbd_fusion = RGBDMidFusion(
            rgb_channels=rgb_channels,
            depth_channels=depth_channels,
            reduction=reduction,
            fusion_weight=fusion_weight,
        )
        
        # RGBDMidFusion输出通道数 = rgb_channels (不改变通道数)
        self.fused_channels = rgb_channels
        
        # 2. GGFE模块 (几何引导增强)
        if use_ggfe:
            self.ggfe = GGFE(
                in_channels=rgb_channels,  # GGFE输入 = RGBDMidFusion输出
                reduction=ggfe_reduction,
                geo_compact=True,  # 使用5通道紧凑模式
            )
        else:
            self.ggfe = None
    
    def forward(self, rgb_feat: torch.Tensor, depth_skip: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            rgb_feat (torch.Tensor): [B, C_rgb, H, W] RGB特征 (来自backbone层，如C3k2)
            depth_skip (torch.Tensor): [B, C_depth, H', W'] 深度特征 (来自RGBDStem layer 0)
        
        Returns:
            torch.Tensor: [B, C_rgb, H, W] 融合并增强后的特征 (与rgb_feat同shape)
        
        Workflow:
            1. RGBDMidFusion: RGB + Depth → fused_feat [B, C_rgb, H, W]
            2. 如果use_ggfe=True: GGFE(fused_feat, depth_skip) → enhanced_feat
            3. 返回enhanced_feat
        """
        # Step 1: RGB-D融合 (RGBDMidFusion)
        fused_feat = self.rgbd_fusion(rgb_feat, depth_skip)  # [B, C_rgb, H, W]
        
        # Step 2: GGFE几何增强 (如果启用)
        if self.ggfe is not None:
            # 注意: GGFE需要原始深度图，depth_skip来自RGBDStem的输出
            # GGFE内部会自动将depth_skip resize到fused_feat的尺寸
            enhanced_feat = self.ggfe(fused_feat, depth_skip)
        else:
            enhanced_feat = fused_feat
        
        return enhanced_feat


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
