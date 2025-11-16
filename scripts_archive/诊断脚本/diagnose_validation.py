"""
诊断脚本：检查 YOLO12x 验证失败的根本原因

问题现象：
  - YOLO12n (自己训练的 RGB-D): mAP = 34.96% ✅ 正常
  - YOLO12x (官方预训练 RGB-only): mAP = 5.26% ❌ 灾难性低

怀疑原因：
  1. YOLO12x 是 3 通道模型，但验证时输入了 4 通道 (RGB+D) 数据
  2. 深度图加载或融合逻辑有问题
  3. 验证脚本的数据配置有问题
"""

import torch
from pathlib import Path 

def check_model_input_channels(model_path):
    """检查模型期望的输入通道数"""
    print(f"\n{'='*80}")
    print(f"检查模型: {model_path}")
    print(f"{'='*80}\n")
    
    # 加载模型
    ckpt = torch.load(model_path, map_location='cpu')
    
    # 检查模型结构
    if 'model' in ckpt:
        model = ckpt['model']
        
        # 查找第一层卷积
        first_conv = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                first_conv = (name, module)
                break
        
        if first_conv:
            name, conv = first_conv
            print(f"第一层卷积: {name}")
            print(f"  输入通道: {conv.in_channels}")
            print(f"  输出通道: {conv.out_channels}")
            print(f"  卷积核尺寸: {conv.kernel_size}")
            print()
            
            if conv.in_channels == 3:
                print("❌ 警告: 这是一个 3 通道 (RGB-only) 模型!")
                print("   如果验证时输入 4 通道 (RGB+D) 数据，会导致维度不匹配！")
                print()
                print("解决方案:")
                print("  1. 验证 YOLO12x 时使用 RGB-only 数据 (不加载深度图)")
                print("  2. 或者重新训练 YOLO12x 以支持 RGB-D 输入")
            elif conv.in_channels == 4:
                print("✅ 正确: 这是一个 4 通道 (RGB-D) 模型")
            else:
                print(f"⚠️  未知: 输入通道数 = {conv.in_channels}")
    
    # 检查 YAML 配置
    if 'model' in ckpt and hasattr(ckpt['model'], 'yaml'):
        yaml_cfg = ckpt['model'].yaml
        print(f"\nYAML 配置:")
        print(f"  ch: {yaml_cfg.get('ch', 'N/A')}")
        print(f"  nc: {yaml_cfg.get('nc', 'N/A')}")


def check_validation_dataset_config():
    """检查验证数据集配置"""
    print(f"\n{'='*80}")
    print(f"检查验证数据集配置")
    print(f"{'='*80}\n")
    
    yaml_path = Path("data/visdrone-rgbd.yaml")
    if not yaml_path.exists():
        print(f"❌ 数据配置文件不存在: {yaml_path}")
        return
    
    import yaml
    with open(yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    print("数据集配置:")
    for key in ['train', 'val', 'train_depth', 'val_depth']:
        print(f"  {key}: {data_cfg.get(key, 'N/A')}")
    print()
    
    # 检查深度图路径是否存在
    if 'val_depth' in data_cfg:
        depth_path = Path(data_cfg['path']) / data_cfg['val_depth']
        if depth_path.exists():
            depth_files = list(depth_path.glob('*.png')) + list(depth_path.glob('*.jpg'))
            print(f"✅ 深度图路径存在: {depth_path}")
            print(f"   深度图数量: {len(depth_files)}")
        else:
            print(f"❌ 深度图路径不存在: {depth_path}")
    else:
        print("⚠️  配置文件中没有 'val_depth' 字段")
        print("   → 验证时不会加载深度图 (RGB-only)")


def diagnose_validation_failure():
    """综合诊断"""
    print("\n" + "="*80)
    print("Phase 3 验证失败诊断报告")
    print("="*80)
    
    # 问题总结
    print("\n📊 观察到的异常:")
    print("  1. YOLO12n (RGB-D, 自己训练): mAP = 34.96% ✅")
    print("  2. YOLO12x (RGB-only, 官方):  mAP = 5.26% ❌")
    print()
    
    # 检查模型
    yolo12n_path = "runs/train/phase3_channelc2f7/weights/best.pt"
    yolo12x_path = "models/yolo12x.pt"
    
    if Path(yolo12n_path).exists():
        check_model_input_channels(yolo12n_path)
    else:
        print(f"\n⚠️  YOLO12n 模型不存在: {yolo12n_path}")
    
    if Path(yolo12x_path).exists():
        check_model_input_channels(yolo12x_path)
    else:
        print(f"\n⚠️  YOLO12x 模型不存在: {yolo12x_path}")
    
    # 检查数据配置
    check_validation_dataset_config()
    
    # 结论
    print("\n" + "="*80)
    print("🔍 诊断结论")
    print("="*80)
    print()
    print("最可能的原因:")
    print("  YOLO12x 是 3 通道模型，但验证时数据集加载了 4 通道 (RGB+D) 数据")
    print()
    print("验证方法:")
    print("  1. 打开验证日志，查看是否有 'channel mismatch' 或 shape 错误")
    print("  2. 在 val_depth.sh 中添加 --verbose 查看详细信息")
    print()
    print("修复方案:")
    print("  方案 A: 为 RGB-only 模型创建单独的验证脚本 (不加载深度图)")
    print("  方案 B: 修改 dataset.py，根据模型通道数自动选择是否加载深度图")
    print("  方案 C: 重新训练 YOLO12x 以支持 RGB-D 输入")
    print()
    print("推荐方案: 方案 B (最灵活)")
    print()


if __name__ == "__main__":
    diagnose_validation_failure()
