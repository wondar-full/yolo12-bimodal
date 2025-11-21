#!/usr/bin/env python3
"""
快速诊断脚本 - 验证GGFE是否正确加载

Usage:
    python check_ggfe_loaded.py runs/train/visdrone_ggfe_truly_fixed_100ep
"""

import sys
import torch
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import LOGGER


def check_ggfe_in_model(model_path):
    """检查模型是否包含GGFE模块"""
    print("=" * 70)
    print(f"Checking GGFE in: {model_path}")
    print("=" * 70)
    
    # 加载模型
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # 1. 检查参数量
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"\n📊 Total parameters: {total_params/1e6:.2f}M")
    
    if total_params / 1e6 < 3.3:
        print("❌ FAILED: Parameter count too low (< 3.3M)")
        print("   Expected: ~3.5M (baseline 3.0M + GGFE 0.5M)")
        print("   GGFE modules are NOT loaded!")
        ggfe_loaded = False
    else:
        print("✅ PASS: Parameter count looks good (>= 3.3M)")
        ggfe_loaded = True
    
    # 2. 检查GGFE模块存在
    print(f"\n🔍 Searching for GGFE modules...")
    ggfe_modules = []
    for name, module in model.model.named_modules():
        if 'ggfe' in name.lower():
            ggfe_modules.append(name)
    
    if ggfe_modules:
        print(f"✅ PASS: Found {len(ggfe_modules)} GGFE modules:")
        for name in ggfe_modules[:10]:
            print(f"   - {name}")
        if len(ggfe_modules) > 10:
            print(f"   ... and {len(ggfe_modules)-10} more")
        ggfe_loaded = True
    else:
        print("❌ FAILED: No GGFE modules found!")
        print("   The model is using standard YOLOv12 architecture")
        ggfe_loaded = False
    
    # 3. 检查RGBDGGFEFusion模块
    print(f"\n🔍 Searching for RGBDGGFEFusion modules...")
    fusion_modules = []
    for name, module in model.model.named_modules():
        if 'rgbdggfefusion' in name.lower() or 'rgbd_ggfe' in name.lower():
            fusion_modules.append(name)
    
    if fusion_modules:
        print(f"✅ PASS: Found {len(fusion_modules)} RGBDGGFEFusion modules:")
        for name in fusion_modules:
            print(f"   - {name}")
    else:
        print("❌ FAILED: No RGBDGGFEFusion modules found!")
    
    # 4. 打印模型摘要
    print(f"\n📋 Model Summary:")
    print(f"   - Model type: {type(model.model).__name__}")
    print(f"   - Number of layers: {len(list(model.model.modules()))}")
    
    # 5. 最终判决
    print("\n" + "=" * 70)
    if ggfe_loaded and ggfe_modules:
        print("✅ FINAL VERDICT: GGFE IS CORRECTLY LOADED!")
        print("   The model contains GGFE modules and has correct parameter count")
    else:
        print("❌ FINAL VERDICT: GGFE IS NOT LOADED!")
        print("   The model is using standard YOLOv12 architecture")
        print("   Training script needs to be fixed")
    print("=" * 70)
    
    return ggfe_loaded


def check_training_args(run_dir):
    """检查训练参数配置"""
    args_file = Path(run_dir) / "args.yaml"
    if not args_file.exists():
        print(f"\n⚠️  args.yaml not found in {run_dir}")
        return
    
    print(f"\n📄 Checking training arguments: {args_file}")
    
    import yaml
    with open(args_file, 'r') as f:
        args = yaml.safe_load(f)
    
    # 关键参数检查
    print(f"   - cfg: {args.get('cfg', 'N/A')}")
    print(f"   - model: {args.get('model', 'N/A')}")
    print(f"   - data: {args.get('data', 'N/A')}")
    
    if args.get('cfg') is None:
        print("   ❌ WARNING: cfg is null, GGFE config was NOT loaded!")
    else:
        print(f"   ✅ cfg is set to '{args.get('cfg')}'")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("Usage: python check_ggfe_loaded.py <path_to_run_dir_or_weights>")
        print("Example: python check_ggfe_loaded.py runs/train/visdrone_ggfe_truly_fixed_100ep")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    # 确定权重文件路径
    if input_path.is_dir():
        # 输入是训练目录
        run_dir = input_path
        weights_path = run_dir / "weights" / "best.pt"
        if not weights_path.exists():
            weights_path = run_dir / "weights" / "last.pt"
        if not weights_path.exists():
            print(f"❌ No weights found in {run_dir}")
            sys.exit(1)
        
        # 检查训练参数
        check_training_args(run_dir)
    else:
        # 输入是权重文件
        weights_path = input_path
    
    # 检查GGFE
    print(f"\n")
    success = check_ggfe_in_model(str(weights_path))
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
