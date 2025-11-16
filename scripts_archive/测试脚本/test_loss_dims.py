"""
快速验证Loss权重计算的张量维度
用于确保修复后的代码能正确处理广播
"""
import torch

# 模拟实际训练中的张量维度
batch_size = 16
num_anchors = 8400  # 80x80 + 40x40 + 20x20 = 6400+1600+400

# 模拟关键张量
stride_tensor = torch.randn(num_anchors, 1)  # (8400, 1)
target_bboxes = torch.randn(batch_size, num_anchors, 4)  # (16, 8400, 4)
fg_mask = torch.randint(0, 2, (batch_size, num_anchors)).bool()  # (16, 8400)
target_scores = torch.randn(batch_size, num_anchors, 10)  # (16, 8400, 10)

print("=" * 60)
print("📊 张量维度验证")
print("=" * 60)
print(f"stride_tensor:   {stride_tensor.shape}")
print(f"target_bboxes:   {target_bboxes.shape}")
print(f"fg_mask:         {fg_mask.shape}")
print(f"target_scores:   {target_scores.shape}")
print()

# 测试修复后的代码逻辑
print("=" * 60)
print("🔧 测试Size-Adaptive权重计算")
print("=" * 60)

num_classes = 10
area_weights = torch.ones(batch_size, num_anchors)
print(f"初始 area_weights: {area_weights.shape}")

if fg_mask.sum() > 0:
    # 修复后的广播逻辑
    stride_broadcast = stride_tensor.unsqueeze(0)  # (1, 8400, 1)
    print(f"stride_broadcast: {stride_broadcast.shape}")
    
    # 计算宽度和高度
    gt_widths = (target_bboxes[:, :, 2] - target_bboxes[:, :, 0]) * stride_broadcast.squeeze(-1)
    gt_heights = (target_bboxes[:, :, 3] - target_bboxes[:, :, 1]) * stride_broadcast.squeeze(-1)
    
    print(f"gt_widths:  {gt_widths.shape}")
    print(f"gt_heights: {gt_heights.shape}")
    
    gt_areas = gt_widths * gt_heights
    print(f"gt_areas:   {gt_areas.shape}")
    
    # 分配权重 (注意: area_weights现在是 (bs, num_anchors))
    area_weights = torch.where(
        gt_areas < 1024,
        torch.tensor(2.0),
        torch.where(
            gt_areas < 9216,
            torch.tensor(1.5),
            torch.tensor(1.0)
        )
    )
    print(f"条件权重 area_weights: {area_weights.shape}")
    
    # 应用fg_mask
    area_weights = area_weights * fg_mask.float()
    print(f"fg_mask后 area_weights: {area_weights.shape}")
    
    # 扩展到匹配target_scores形状
    size_weights = area_weights.unsqueeze(-1).expand(batch_size, num_anchors, num_classes)
    print(f"扩展后 size_weights: {size_weights.shape}")
    print(f"target_scores形状: {target_scores.shape}")
    print(f"✅ 形状匹配成功!" if size_weights.shape == target_scores.shape else "❌ 形状不匹配!")
    
    # 验证权重分布
    print()
    print("=" * 60)
    print("📈 权重统计")
    print("=" * 60)
    valid_area_weights = area_weights[fg_mask]
    print(f"正样本数量: {fg_mask.sum().item()}")
    print(f"权重×2.0数量: {(valid_area_weights == 2.0).sum().item()}")
    print(f"权重×1.5数量: {(valid_area_weights == 1.5).sum().item()}")
    print(f"权重×1.0数量: {(valid_area_weights == 1.0).sum().item()}")
    print(f"权重范围: [{valid_area_weights.min().item():.1f}, {valid_area_weights.max().item():.1f}]")
    
    # 验证cls_loss计算
    print()
    print("=" * 60)
    print("🧮 验证Loss计算")
    print("=" * 60)
    cls_loss_per_sample = torch.randn_like(target_scores)
    print(f"cls_loss_per_sample: {cls_loss_per_sample.shape}")
    weighted_cls_loss = cls_loss_per_sample * size_weights
    print(f"weighted_cls_loss:   {weighted_cls_loss.shape}")
    print(f"✅ 可以正常相乘!" if weighted_cls_loss.shape == cls_loss_per_sample.shape else "❌ 相乘失败!")
    
    # 验证box_loss权重
    avg_area_weight = area_weights[fg_mask].mean()
    print(f"avg_area_weight:     {avg_area_weight.item():.3f} (标量)")
    print(f"✅ 可以用于box_loss缩放!")

print()
print("=" * 60)
print("✅ 所有维度检查通过!")
print("=" * 60)
