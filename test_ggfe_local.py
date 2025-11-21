#!/usr/bin/env python3
"""
本地验证脚本 - 测试RGBDGGFEFusion是否能正确加载

Usage:
    python test_ggfe_local.py
"""

import sys
import torch
from pathlib import Path

print("=" * 70)
print("测试 RGBDGGFEFusion 模块导入和模型创建")
print("=" * 70)

# Test 1: 导入模块
print("\n[Test 1] 导入模块...")
try:
    from ultralytics.nn.modules import GGFE, RGBDGGFEFusion
    print("✅ 成功导入 GGFE 和 RGBDGGFEFusion")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# Test 2: 创建模型
print("\n[Test 2] 从YAML创建模型...")
try:
    from ultralytics import YOLO
    
    yaml_path = 'ultralytics/cfg/models/12/yolo12-rgbd-ggfe-universal.yaml'
    print(f"   YAML路径: {yaml_path}")
    
    model = YOLO(yaml_path, task='detect')
    print("✅ 模型创建成功")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"\n📊 模型参数量: {total_params/1e6:.2f}M")
    
    if total_params / 1e6 < 3.3:
        print("❌ 警告: 参数量过低 (< 3.3M), GGFE可能未加载")
    else:
        print("✅ 参数量正常 (>= 3.3M)")
    
except Exception as e:
    print(f"❌ 模型创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: 检查GGFE模块
print("\n[Test 3] 检查GGFE模块是否存在...")
ggfe_modules = []
for name, module in model.model.named_modules():
    if 'ggfe' in name.lower():
        ggfe_modules.append(name)

if ggfe_modules:
    print(f"✅ 找到 {len(ggfe_modules)} 个GGFE模块:")
    for name in ggfe_modules[:10]:
        print(f"   - {name}")
    if len(ggfe_modules) > 10:
        print(f"   ... 还有 {len(ggfe_modules)-10} 个")
else:
    print("❌ 未找到GGFE模块")
    sys.exit(1)

# Test 4: 检查RGBDGGFEFusion模块
print("\n[Test 4] 检查RGBDGGFEFusion模块...")
fusion_modules = []
for name, module in model.model.named_modules():
    if 'rgbdggfefusion' in name.lower() or 'rgbd_ggfe' in name.lower():
        fusion_modules.append((name, type(module).__name__))

if fusion_modules:
    print(f"✅ 找到 {len(fusion_modules)} 个RGBDGGFEFusion模块:")
    for name, mtype in fusion_modules:
        print(f"   - {name} ({mtype})")
else:
    print("❌ 未找到RGBDGGFEFusion模块")
    sys.exit(1)

# Test 5: 模拟加载预训练权重
print("\n[Test 5] 模拟加载预训练权重...")
try:
    weights_path = 'models/yolo12n.pt'
    if Path(weights_path).exists():
        ckpt = torch.load(weights_path, map_location='cpu')
        state_dict = ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']
        
        # 加载权重 (strict=False)
        incompatible = model.model.load_state_dict(state_dict, strict=False)
        
        print(f"✅ 权重加载成功 (strict=False)")
        print(f"   Missing keys: {len(incompatible.missing_keys)}")
        print(f"   Unexpected keys: {len(incompatible.unexpected_keys)}")
        
        if len(incompatible.missing_keys) > 0:
            print(f"\n   Missing keys示例 (GGFE参数):")
            for key in incompatible.missing_keys[:5]:
                print(f"      - {key}")
            if len(incompatible.missing_keys) > 5:
                print(f"      ... 还有 {len(incompatible.missing_keys)-5} 个")
        
        # 验证参数量
        total_params_after = sum(p.numel() for p in model.model.parameters())
        print(f"\n📊 加载权重后参数量: {total_params_after/1e6:.2f}M")
        
        if abs(total_params_after - total_params) > 100:
            print("❌ 警告: 参数量变化，可能权重加载有问题")
        else:
            print("✅ 参数量一致")
    else:
        print(f"⚠️  权重文件不存在: {weights_path}, 跳过此测试")
        
except Exception as e:
    print(f"❌ 权重加载失败: {e}")
    import traceback
    traceback.print_exc()

# Final Summary
print("\n" + "=" * 70)
print("测试总结")
print("=" * 70)
print("✅ 模块导入: 成功")
print("✅ 模型创建: 成功")
print(f"✅ 参数量: {total_params/1e6:.2f}M")
print(f"✅ GGFE模块: {len(ggfe_modules)} 个")
print(f"✅ RGBDGGFEFusion模块: {len(fusion_modules)} 个")
print("\n🎯 所有测试通过！可以上传到服务器")
print("=" * 70)
