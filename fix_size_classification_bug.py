#!/usr/bin/env python3
"""
修复尺度分类Bug - 归一化坐标 vs 像素坐标单位不匹配

🐛 问题描述:
ultralytics12/ultralytics/models/yolo/detect/val.py 中的 _process_batch() 函数
使用归一化bbox坐标计算面积,但与像素面积阈值比较,导致所有目标都被错误分类。

根本原因:
    归一化面积范围: 0 ~ 1
    像素面积阈值: 32² = 1024, 96² = 9216
    比较: 0.0025 < 1024 → 几乎所有目标都是small!

🎯 修复方案:
将归一化坐标转换为像素坐标后再计算面积

作者: AI Assistant
日期: 2025-01-29
版本: v1.0
"""

import torch

def analyze_bug():
    """分析Bug的严重性"""
    print("=" * 80)
    print("🐛 Bug分析报告")
    print("=" * 80)
    
    # 模拟数据
    img_size = 640
    
    # 测试用例1: 小目标 (32×32像素)
    bbox_small = torch.tensor([0.1, 0.1, 0.15, 0.15])  # 归一化
    width_norm = bbox_small[2] - bbox_small[0]  # 0.05
    height_norm = bbox_small[3] - bbox_small[1]  # 0.05
    area_norm = width_norm * height_norm  # 0.0025
    
    width_pixel = width_norm * img_size  # 32
    height_pixel = height_norm * img_size  # 32
    area_pixel = width_pixel * height_pixel  # 1024
    
    print(f"\n测试用例1: 小目标 (32×32像素)")
    print(f"  归一化bbox: {bbox_small.tolist()}")
    print(f"  归一化面积: {area_norm:.6f}")
    print(f"  像素面积: {area_pixel:.0f} pixels²")
    print(f"  ❌ 错误判断: area_norm ({area_norm:.6f}) < 1024? → {area_norm < 1024} (small)")
    print(f"  ✅ 正确判断: area_pixel ({area_pixel:.0f}) < 1024? → {area_pixel < 1024} (small)")
    
    # 测试用例2: 中等目标 (64×64像素)
    bbox_medium = torch.tensor([0.2, 0.2, 0.3, 0.3])
    width_norm = bbox_medium[2] - bbox_medium[0]  # 0.1
    height_norm = bbox_medium[3] - bbox_medium[1]  # 0.1
    area_norm = width_norm * height_norm  # 0.01
    
    width_pixel = width_norm * img_size  # 64
    height_pixel = height_norm * img_size  # 64
    area_pixel = width_pixel * height_pixel  # 4096
    
    print(f"\n测试用例2: 中等目标 (64×64像素)")
    print(f"  归一化bbox: {bbox_medium.tolist()}")
    print(f"  归一化面积: {area_norm:.6f}")
    print(f"  像素面积: {area_pixel:.0f} pixels²")
    print(f"  ❌ 错误判断: area_norm ({area_norm:.6f}) < 1024? → {area_norm < 1024} (错误分为small)")
    print(f"  ✅ 正确判断: area_pixel ({area_pixel:.0f}) ∈ [1024, 9216)? → {1024 <= area_pixel < 9216} (medium)")
    
    # 测试用例3: 大目标 (128×128像素)
    bbox_large = torch.tensor([0.1, 0.1, 0.3, 0.3])
    width_norm = bbox_large[2] - bbox_large[0]  # 0.2
    height_norm = bbox_large[3] - bbox_large[1]  # 0.2
    area_norm = width_norm * height_norm  # 0.04
    
    width_pixel = width_norm * img_size  # 128
    height_pixel = height_norm * img_size  # 128
    area_pixel = width_pixel * height_pixel  # 16384
    
    print(f"\n测试用例3: 大目标 (128×128像素)")
    print(f"  归一化bbox: {bbox_large.tolist()}")
    print(f"  归一化面积: {area_norm:.6f}")
    print(f"  像素面积: {area_pixel:.0f} pixels²")
    print(f"  ❌ 错误判断: area_norm ({area_norm:.6f}) < 1024? → {area_norm < 1024} (错误分为small)")
    print(f"  ✅ 正确判断: area_pixel ({area_pixel:.0f}) >= 9216? → {area_pixel >= 9216} (large)")
    
    print("\n" + "=" * 80)
    print("📊 统计影响")
    print("=" * 80)
    print("由于几乎所有归一化面积都 < 1024:")
    print("  - Small类别: 包含所有目标 (错误)")
    print("  - Medium类别: 几乎为空")
    print("  - Large类别: 几乎为空")
    print("结果: Small mAP > Medium mAP > Large mAP (完全错误的分布)")
    print("=" * 80)

def generate_fix():
    """生成修复代码"""
    print("\n" + "=" * 80)
    print("🔧 修复代码")
    print("=" * 80)
    
    fix_code = """
# 修复位置: ultralytics12/ultralytics/models/yolo/detect/val.py
# 函数: _process_batch()
# 行号: ~298-303

# ❌ 原始代码 (Bug版本)
widths = (batch["bboxes"][..., 2] - batch["bboxes"][..., 0]).cpu()
heights = (batch["bboxes"][..., 3] - batch["bboxes"][..., 1]).cpu()
areas = widths * heights  # 归一化面积!
small_mask = areas < 32.0**2  # 错误比较!
medium_mask = (areas >= 32.0**2) & (areas < 96.0**2)
large_mask = areas >= 96.0**2

# ✅ 修复代码 (正确版本)
# 获取图像尺寸
img_shape = batch["img"].shape  # [B, C, H, W]
img_h, img_w = img_shape[2], img_shape[3]  # 通常是640×640

# 计算像素级宽高和面积
widths = (batch["bboxes"][..., 2] - batch["bboxes"][..., 0]) * img_w  # 转换为像素
heights = (batch["bboxes"][..., 3] - batch["bboxes"][..., 1]) * img_h  # 转换为像素
areas = (widths * heights).cpu()  # 像素面积

# 正确的尺度判断
small_mask = areas < 32.0**2  # 32×32 = 1024 pixels²
medium_mask = (areas >= 32.0**2) & (areas < 96.0**2)  # 1024 ~ 9216 pixels²
large_mask = areas >= 96.0**2  # >= 9216 pixels²

# 预测框也需要相同修复
pred_widths = (preds["bboxes"][..., 2] - preds["bboxes"][..., 0]) * img_w
pred_heights = (preds["bboxes"][..., 3] - preds["bboxes"][..., 1]) * img_h
pred_areas = pred_widths * pred_heights
area_small = pred_areas.new_tensor(32.0**2)
area_medium = pred_areas.new_tensor(96.0**2)
pred_small_mask = pred_areas < area_small
pred_medium_mask = (pred_areas >= area_small) & (pred_areas < area_medium)
pred_large_mask = pred_areas >= area_medium
"""
    
    print(fix_code)
    print("=" * 80)

def verify_fix():
    """验证修复后的效果"""
    print("\n" + "=" * 80)
    print("✅ 修复验证")
    print("=" * 80)
    
    img_size = 640
    
    # 模拟batch数据
    batch_bboxes = torch.tensor([
        [0.1, 0.1, 0.15, 0.15],  # 32×32 (small)
        [0.2, 0.2, 0.3, 0.3],    # 64×64 (medium)
        [0.4, 0.4, 0.6, 0.6],    # 128×128 (large)
    ])
    
    # ❌ 错误的归一化面积计算
    widths_norm = batch_bboxes[:, 2] - batch_bboxes[:, 0]
    heights_norm = batch_bboxes[:, 3] - batch_bboxes[:, 1]
    areas_norm = widths_norm * heights_norm
    
    small_mask_wrong = areas_norm < 1024
    medium_mask_wrong = (areas_norm >= 1024) & (areas_norm < 9216)
    large_mask_wrong = areas_norm >= 9216
    
    print("❌ 错误分类 (归一化面积):")
    print(f"  归一化面积: {areas_norm.tolist()}")
    print(f"  Small mask: {small_mask_wrong.tolist()}  → {small_mask_wrong.sum().item()} targets")
    print(f"  Medium mask: {medium_mask_wrong.tolist()} → {medium_mask_wrong.sum().item()} targets")
    print(f"  Large mask: {large_mask_wrong.tolist()} → {large_mask_wrong.sum().item()} targets")
    
    # ✅ 正确的像素面积计算
    widths_pixel = (batch_bboxes[:, 2] - batch_bboxes[:, 0]) * img_size
    heights_pixel = (batch_bboxes[:, 3] - batch_bboxes[:, 1]) * img_size
    areas_pixel = widths_pixel * heights_pixel
    
    small_mask_correct = areas_pixel < 1024
    medium_mask_correct = (areas_pixel >= 1024) & (areas_pixel < 9216)
    large_mask_correct = areas_pixel >= 9216
    
    print("\n✅ 正确分类 (像素面积):")
    print(f"  像素面积: {areas_pixel.tolist()}")
    print(f"  Small mask: {small_mask_correct.tolist()}  → {small_mask_correct.sum().item()} target")
    print(f"  Medium mask: {medium_mask_correct.tolist()} → {medium_mask_correct.sum().item()} target")
    print(f"  Large mask: {large_mask_correct.tolist()} → {large_mask_correct.sum().item()} target")
    
    print("\n💡 预期效果:")
    print("  修复后,mAP分布应该变为: Small < Medium < Large (正常趋势)")
    print("  实际YOLO12n mAP可能从:")
    print("    Small: 13.30% → 实际small mAP (会降低,因为现在是真的small)")
    print("    Medium: 10.98% → 实际medium mAP (会提升,因为不再混入其他尺度)")
    print("    Large: 14.48% → 实际large mAP (会提升,原因同上)")
    print("=" * 80)

if __name__ == "__main__":
    analyze_bug()
    generate_fix()
    verify_fix()
    
    print("\n🚀 下一步操作:")
    print("1. 修改 ultralytics12/ultralytics/models/yolo/detect/val.py")
    print("2. 重新验证 YOLO12n 和 YOLO12x")
    print("3. 对比修复前后的mAP分布")
    print("4. 确认 Small < Medium < Large 的正常趋势")
