"""
诊断 YOLO12x 模型是否正常

怀疑原因:
1. YOLO12x 模型文件可能损坏
2. 模型没有在 VisDrone 数据集上训练
3. 模型是其他任务的 (分割/分类/姿态估计)
"""

import torch
from pathlib import Path
from ultralytics import YOLO

def diagnose_yolo12x(model_path="models/yolo12x.pt"):
    """全面诊断 YOLO12x 模型"""
    
    print("=" * 80)
    print("🔍 YOLO12x 模型诊断")
    print("=" * 80)
    print()
    
    if not Path(model_path).exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    # 加载模型
    print(f"加载模型: {model_path}")
    ckpt = torch.load(model_path, map_location='cpu')
    
    # 1. 检查模型基本信息
    print("\n" + "=" * 80)
    print("1️⃣ 基本信息")
    print("=" * 80)
    
    for key in ['epoch', 'best_fitness', 'date']:
        if key in ckpt:
            print(f"  {key}: {ckpt[key]}")
    
    # 2. 检查模型任务类型
    print("\n" + "=" * 80)
    print("2️⃣ 任务类型")
    print("=" * 80)
    
    if 'model' in ckpt:
        model = ckpt['model']
        if hasattr(model, 'yaml'):
            task = model.yaml.get('task', 'detect')
            print(f"  Task: {task}")
            
            if task != 'detect':
                print(f"  ❌ 错误: 这不是一个检测模型，而是 {task} 模型！")
                print(f"     YOLO12x 应该是检测模型 (task='detect')")
                return
        
        # 3. 检查类别数
        print("\n" + "=" * 80)
        print("3️⃣ 类别信息")
        print("=" * 80)
        
        if hasattr(model, 'names'):
            names = model.names
            print(f"  类别数: {len(names)}")
            print(f"  类别名称:")
            for i, name in enumerate(names):
                print(f"    {i}: {name}")
            
            # 检查是否是 VisDrone 类别
            visdrone_classes = [
                'pedestrian', 'people', 'bicycle', 'car', 'van',
                'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
            ]
            
            if len(names) != 10:
                print(f"\n  ⚠️  警告: VisDrone 应该有 10 个类别，但模型有 {len(names)} 个")
                print(f"     可能原因: 模型在其他数据集上训练 (如 COCO-80类)")
            
            # 检查类别名称是否匹配
            if set(names) != set(visdrone_classes):
                print(f"\n  ❌ 错误: 类别名称不匹配 VisDrone!")
                print(f"     模型类别: {list(names)}")
                print(f"     VisDrone类别: {visdrone_classes}")
                print(f"\n  🔍 结论: **YOLO12x 不是在 VisDrone 上训练的模型！**")
                print(f"     可能是在 COCO 或其他数据集上预训练的通用模型")
                print(f"     在 VisDrone 验证集上性能极差 (mAP=5.26%) 是正常的！")
                return
    
    # 4. 检查模型架构
    print("\n" + "=" * 80)
    print("4️⃣ 模型架构")
    print("=" * 80)
    
    if 'model' in ckpt:
        model = ckpt['model']
        
        # 统计层数
        total_layers = 0
        conv_layers = 0
        for module in model.modules():
            total_layers += 1
            if isinstance(module, torch.nn.Conv2d):
                conv_layers += 1
        
        print(f"  总层数: {total_layers}")
        print(f"  卷积层数: {conv_layers}")
        
        # 检查第一层
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                print(f"\n  第一层卷积: {name}")
                print(f"    输入通道: {module.in_channels}")
                print(f"    输出通道: {module.out_channels}")
                print(f"    卷积核: {module.kernel_size}")
                
                if module.in_channels != 3:
                    print(f"    ⚠️  警告: 输入通道不是 3 (RGB)")
                break
    
    # 5. 检查训练数据集
    print("\n" + "=" * 80)
    print("5️⃣ 训练数据集")
    print("=" * 80)
    
    train_args = ckpt.get('train_args', {})
    if train_args:
        data_path = train_args.get('data', 'Unknown')
        print(f"  数据集路径: {data_path}")
        
        if 'coco' in str(data_path).lower():
            print(f"  ❌ 这是在 COCO 数据集上训练的模型！")
            print(f"     COCO 有 80 个类别，与 VisDrone (10类) 完全不同")
            print(f"     在 VisDrone 上验证性能极差 (5.26%) 是**预期行为**！")
        elif 'visdrone' not in str(data_path).lower():
            print(f"  ⚠️  警告: 数据集路径不包含 'visdrone'")
    else:
        print(f"  ⚠️  无法获取训练数据集信息")
    
    # 6. 使用 YOLO API 测试
    print("\n" + "=" * 80)
    print("6️⃣ API 测试")
    print("=" * 80)
    
    try:
        yolo_model = YOLO(model_path)
        print(f"  ✅ YOLO API 加载成功")
        print(f"  任务类型: {yolo_model.task}")
        print(f"  类别数: {len(yolo_model.names)}")
    except Exception as e:
        print(f"  ❌ YOLO API 加载失败: {e}")
    
    # 7. 总结
    print("\n" + "=" * 80)
    print("🎯 诊断总结")
    print("=" * 80)
    print()
    
    if 'model' in ckpt and hasattr(ckpt['model'], 'names'):
        names = list(ckpt['model'].names)
        visdrone_classes = [
            'pedestrian', 'people', 'bicycle', 'car', 'van',
            'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
        ]
        
        if len(names) == 80:
            print("🔍 结论: YOLO12x 是 **COCO 预训练模型** (80 类)")
            print()
            print("为什么 mAP 只有 5.26%?")
            print("  1. **类别不匹配**: COCO 80 类 vs VisDrone 10 类")
            print("  2. **域迁移问题**: COCO (地面视角) vs VisDrone (UAV 俯视)")
            print("  3. **目标尺度差异**: COCO (多为中大目标) vs VisDrone (68% 小目标)")
            print()
            print("为什么 Small > Medium > Large?")
            print("  1. **类别映射错误**: COCO 类别索引与 VisDrone 不对应")
            print("  2. **随机检测**: 模型在 VisDrone 类别上基本是随机预测")
            print("  3. **尺度判断混乱**: 由于类别错位，尺度统计失真")
            print()
            print("✅ 这不是代码 bug，而是**数据集不匹配**导致的!")
            print()
            print("解决方案:")
            print("  1. ❌ 不要在 VisDrone 上验证 COCO 预训练模型 (无意义)")
            print("  2. ✅ 使用在 VisDrone 上训练的模型 (如 YOLO12n-RGB-D)")
            print("  3. ✅ 或者微调 YOLO12x 在 VisDrone 数据集上")
        
        elif len(names) == 10 and set(names) == set(visdrone_classes):
            print("✅ 结论: YOLO12x 是 **VisDrone 训练模型**")
            print()
            print("但 mAP 只有 5.26%，可能原因:")
            print("  1. 训练未收敛")
            print("  2. 超参数设置不当")
            print("  3. 模型损坏")
            print("  4. 验证代码有 bug")
        
        else:
            print(f"⚠️  未知情况: 类别数 = {len(names)}")
            print(f"   类别名称: {names}")


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/yolo12x.pt"
    diagnose_yolo12x(model_path)
