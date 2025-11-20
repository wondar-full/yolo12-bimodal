# 多维度改进方案 - Phase 1：快速验证（1-2 周）

## 🎯 目标

用**最小代码改动**验证几何信息的价值，快速看到效果！

---

## 📋 实验计划

### **Exp 1: 几何先验计算 + 简单融合**

**核心代码**（只需 50 行！）：

```python
# 在ultralytics/nn/modules/geometry.py中添加
import torch
import torch.nn.functional as F

class GeometricPriorExtractor:
    """从深度图计算几何先验（无需训练）"""

    @staticmethod
    def compute_normals(depth):
        """
        计算表面法向量
        Args:
            depth: [B, 1, H, W]
        Returns:
            normals: [B, 3, H, W]
        """
        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=depth.dtype, device=depth.device).view(1,1,3,3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=depth.dtype, device=depth.device).view(1,1,3,3)

        grad_x = F.conv2d(depth, sobel_x, padding=1)
        grad_y = F.conv2d(depth, sobel_y, padding=1)

        # 法向量 = (-dz/dx, -dz/dy, 1)
        normals = torch.cat([-grad_x, -grad_y, torch.ones_like(depth)], dim=1)
        normals = F.normalize(normals, p=2, dim=1)
        return normals

    @staticmethod
    def compute_edges(depth):
        """
        计算深度边缘
        Args:
            depth: [B, 1, H, W]
        Returns:
            edges: [B, 1, H, W]
        """
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=depth.dtype, device=depth.device).view(1,1,3,3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=depth.dtype, device=depth.device).view(1,1,3,3)

        grad_x = F.conv2d(depth, sobel_x, padding=1)
        grad_y = F.conv2d(depth, sobel_y, padding=1)
        edges = torch.sqrt(grad_x**2 + grad_y**2)
        return edges

    @staticmethod
    def compute_confidence(depth, threshold=0.1):
        """
        深度置信度评估
        Args:
            depth: [B, 1, H, W]
        Returns:
            confidence: [B, 1, H, W]
        """
        # 基于局部方差的置信度
        depth_std = F.avg_pool2d(depth**2, kernel_size=5, stride=1, padding=2) - \
                    F.avg_pool2d(depth, kernel_size=5, stride=1, padding=2)**2
        confidence = torch.exp(-depth_std / threshold)
        return confidence
```

**修改 RGBDMidFusion**：

```python
# 在ultralytics/nn/modules/conv.py的RGBDMidFusion中
from ultralytics.nn.modules.geometry import GeometricPriorExtractor

class RGBDMidFusion(nn.Module):
    def __init__(self, rgb_channels, depth_channels, reduction=16, fusion="gated_add"):
        super().__init__()
        self.geo_extractor = GeometricPriorExtractor()  # 新增

        # 几何先验投影（4通道 -> depth_channels）
        self.geo_proj = nn.Sequential(
            nn.Conv2d(4, depth_channels, 1),  # 4 = 3(normals) + 1(edges)
            nn.BatchNorm2d(depth_channels),
            nn.SiLU(inplace=True)
        )

        # 原有代码...
        self.fusion = fusion
        # ...

    def forward(self, rgb_feat, depth_feat):
        """
        Args:
            rgb_feat: [B, C, H, W] - RGB特征
            depth_feat: [B, 1, H, W] - 深度图（原始）
        """
        # 1. 计算几何先验（无参数，纯数学运算）
        normals = self.geo_extractor.compute_normals(depth_feat)  # [B,3,H,W]
        edges = self.geo_extractor.compute_edges(depth_feat)      # [B,1,H,W]
        confidence = self.geo_extractor.compute_confidence(depth_feat)  # [B,1,H,W]

        # 2. 拼接几何先验
        geo_prior = torch.cat([normals, edges], dim=1)  # [B,4,H,W]
        geo_prior = geo_prior * confidence  # 质量加权

        # 3. 投影到深度特征空间
        geo_feat = self.geo_proj(geo_prior)  # [B, depth_channels, H, W]

        # 4. 调整尺寸（如果需要）
        if geo_feat.shape[2:] != rgb_feat.shape[2:]:
            geo_feat = F.interpolate(geo_feat, size=rgb_feat.shape[2:], mode='bilinear')

        # 5. 原有的融合逻辑
        if self.fusion == "gated_add":
            # 用几何增强的深度特征替代原始depth_feat
            return self._gated_add_fusion(rgb_feat, geo_feat)
        # ...
```

**训练命令**：

```bash
# 在服务器上运行（100 epochs快速验证）
python train_depth_solr_v2.py --cfg n --data visdrone-rgbd.yaml \
    --epochs 100 \
    --name exp_geometry_prior_test \
    --device 0
```

**预期结果**：

- AP@0.5:0.95: 19.2% → **20-21%** (+0.8-1.8%)
- AP_s: 9.9% → **11-12%** (+1-2%)
- AP_m: 29.6% → **31-32%** (+1-2%)

**判断标准**：

- ✅ **成功**：AP 提升 ≥1% → 继续 Exp 2
- ❌ **失败**：AP 无变化 → 说明几何信息无效，放弃此方向

---

### **Exp 2: 添加 GGFE 模块**

**如果 Exp 1 成功，进一步增强几何引导**

**实现位置**：

- 在 Backbone 的 P3 层后插入 GGFE 模块
- 只改动一个位置，最小化风险

**代码**（在`ultralytics/nn/modules/conv.py`中新增）：

```python
class GGFE(nn.Module):
    """
    Geometry-Guided Feature Enhancement
    轻量级版本：只用空间注意力
    """
    def __init__(self, in_channels=512):
        super().__init__()
        self.geo_extractor = GeometricPriorExtractor()

        # 几何先验投影
        self.geo_proj = nn.Conv2d(4, in_channels, 1)

        # 几何注意力（只用空间，不用通道）
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, depth):
        """
        Args:
            rgb_feat: [B, C, H, W] - RGB特征
            depth: [B, 1, H', W'] - 深度图
        """
        # 调整深度图尺寸
        depth = F.interpolate(depth, size=rgb_feat.shape[2:], mode='bilinear')

        # 计算几何先验
        normals = self.geo_extractor.compute_normals(depth)
        edges = self.geo_extractor.compute_edges(depth)
        geo_prior = torch.cat([normals, edges], dim=1)  # [B,4,H,W]

        # 投影
        geo_feat = self.geo_proj(geo_prior)  # [B, C, H, W]

        # 几何注意力
        attn_map = self.attn(geo_feat)  # [B, 1, H, W]

        # 增强RGB特征
        rgb_enhanced = rgb_feat * (1 + attn_map)

        return rgb_enhanced  # 残差连接在外面做
```

**修改 YAML 配置**：

```yaml
# 在yolo12-rgbd-v2.1-universal.yaml的Backbone部分
# Layer 5 (P3层) 后添加GGFE
backbone:
  # ...
  - [-1, 3, C2f, [512]] # 4-P3/8
  - [[4, 0], 1, RGBDMidFusion, [512, 64]] # 5-P3 depth fusion
  - [-1, 1, GGFE, [512]] # 6-GGFE (新增) ← 只在P3添加，测试效果
  # ...
```

**训练命令**：

```bash
python train_depth_solr_v2.py --cfg n --data visdrone-rgbd.yaml \
    --epochs 300 \
    --name exp_ggfe_p3 \
    --device 0
```

**预期结果**：

- AP@0.5:0.95: 20-21% → **21-22%** (+1%)
- AP_s: 11-12% → **12-13%** (+1%)

---

### **Exp 3: SADF 尺度感知融合**

**如果 Exp 2 成功，添加小目标专属优化**

**实现**（在`ultralytics/nn/modules/conv.py`）：

```python
class SADF(nn.Module):
    """
    Scale-Aware Depth Fusion
    简化版：只调整权重，不增加复杂结构
    """
    def __init__(self):
        super().__init__()
        # 小目标在浅层权重更大
        self.scale_weights = nn.Parameter(torch.tensor([2.0, 1.0, 0.5]))

    def forward(self, feats):
        """
        Args:
            feats: List[[B, C, H, W]] - P3/P4/P5三个尺度特征
        Returns:
            weighted_feats: List[[B, C, H, W]]
        """
        weighted_feats = []
        for i, feat in enumerate(feats):
            weighted_feat = feat * self.scale_weights[i]
            weighted_feats.append(weighted_feat)
        return weighted_feats
```

**修改位置**：在 Neck 输出后

**训练命令**：

```bash
python train_depth_solr_v2.py --cfg n --data visdrone-rgbd.yaml \
    --epochs 300 \
    --name exp_sadf \
    --device 0
```

**预期结果**：

- AP_s: 12-13% → **13-14%** (+1%)

---

## 📊 Phase 1 预期总提升

| 阶段            | 改进内容 | AP@0.5:0.95    | AP_s       | AP_m       | 训练时间 |
| --------------- | -------- | -------------- | ---------- | ---------- | -------- |
| **Baseline**    | 无       | 19.2%          | 9.9%       | 29.6%      | -        |
| **Exp 1**       | 几何先验 | 20-21%         | 11-12%     | 31-32%     | 4 天     |
| **Exp 2**       | +GGFE    | 21-22%         | 12-13%     | 32-33%     | 10 天    |
| **Exp 3**       | +SADF    | **21.5-22.5%** | **13-14%** | **32-33%** | 10 天    |
| **RemDet-Tiny** | 目标     | **21.8%**      | **12.7%**  | **33.0%**  | -        |

**Phase 1 完成后**：

- ✅ **接近 RemDet** - AP 差距缩小到 0.3-1.3%
- ✅ **验证方向正确** - 几何信息有效
- ✅ **为 Phase 2 打基础** - 确定哪些模块最有效

---

## ⚙️ 实施步骤

### **Step 1: 修改代码**（1 天）

1. 在`ultralytics/nn/modules/geometry.py`添加`GeometricPriorExtractor`
2. 修改`ultralytics/nn/modules/conv.py`的`RGBDMidFusion`
3. 本地测试代码语法正确性

### **Step 2: 快速验证**（4 天）

运行 Exp 1（100 epochs），检查结果

### **Step 3: 完整训练**（10 天）

如果 Exp 1 成功，运行 Exp 2 和 Exp 3

---

## 🎯 成功标准

- ✅ **Exp 1 成功**：AP 提升 ≥1%
- ✅ **Exp 2 成功**：AP 再提升 ≥0.5%
- ✅ **Exp 3 成功**：AP_s 提升 ≥1%

**如果三个实验都成功，Phase 1 预期达到 AP≥21.5%，接近 RemDet-Tiny 的 21.8%！**

---

**准备好开始了吗？我可以立即帮您生成完整的代码文件！** 🚀
