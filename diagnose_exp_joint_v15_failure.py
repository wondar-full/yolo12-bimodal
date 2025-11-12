#!/usr/bin/env python3
"""
紧急诊断脚本：检查exp_joint_v15训练失败的根本原因

性能对比:
- exp_joint_v13 (yolo12s+yolo12n.pt): mAP 22.27%
- exp_joint_v15 (yolo12n+yolo12n.pt): mAP 19.51% (更差!)

可能原因:
1. UAVDT标签路径错误 (labels/rgb/ vs labels/)
2. UAVDT标签格式错误 (COCO未转YOLO)
3. RGB-D通道数错误 (3通道 vs 4通道)
4. 深度图加载失败

Author: yoloDepth Team
Date: 2025-11-06
"""

import os
import yaml
from pathlib import Path
import cv2
import numpy as np

def check_dataset_structure():
    """检查数据集目录结构"""
    print("=" * 80)
    print("1️⃣  检查数据集目录结构")
    print("=" * 80)
    print()
    
    # 读取数据集配置
    yaml_path = Path("data/visdrone_uavdt_joint.yaml")
    with open(yaml_path, encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    
    dataset_root = Path(data_cfg['path'])
    print(f"📂 数据集根目录: {dataset_root}")
    print()
    
    # 检查UAVDT目录
    uavdt_base = dataset_root / "UAVDT_YOLO/train"
    
    print("🔍 检查UAVDT目录结构:")
    print()
    
    # 检查图像目录
    rgb_dir = uavdt_base / "images/rgb"
    print(f"  RGB图像: {rgb_dir}")
    print(f"    存在: {'✅' if rgb_dir.exists() else '❌'}")
    if rgb_dir.exists():
        rgb_count = len(list(rgb_dir.glob("*.jpg")))
        print(f"    数量: {rgb_count}")
        if rgb_count > 0:
            sample = list(rgb_dir.glob("*.jpg"))[0]
            print(f"    示例: {sample.name}")
    print()
    
    # 检查深度图目录
    depth_dir = uavdt_base / "images/d"
    print(f"  深度图: {depth_dir}")
    print(f"    存在: {'✅' if depth_dir.exists() else '❌'}")
    if depth_dir.exists():
        depth_count = len(list(depth_dir.glob("*.png")))
        print(f"    数量: {depth_count}")
        if depth_count > 0:
            sample = list(depth_dir.glob("*.png"))[0]
            print(f"    示例: {sample.name}")
    print()
    
    # 检查标签目录 (关键!)
    print("  🎯 标签目录检查 (CRITICAL):")
    
    # 可能的标签路径
    label_paths = [
        uavdt_base / "labels",           # YOLO默认期望的路径
        uavdt_base / "labels/rgb",       # 用户实际的路径?
        uavdt_base / "labels/d",         # 深度标签?
    ]
    
    for label_path in label_paths:
        print(f"    {label_path}:")
        print(f"      存在: {'✅' if label_path.exists() else '❌'}")
        if label_path.exists():
            txt_count = len(list(label_path.glob("*.txt")))
            print(f"      数量: {txt_count}")
            if txt_count > 0:
                sample = list(label_path.glob("*.txt"))[0]
                print(f"      示例: {sample.name}")
                # 读取第一行检查格式
                with open(sample) as f:
                    first_line = f.readline().strip()
                    print(f"      格式: {first_line}")
                    parts = first_line.split()
                    if len(parts) == 5:
                        class_id, x, y, w, h = parts
                        print(f"        → YOLO格式 ✅ (class {class_id})")
                    else:
                        print(f"        → 未知格式 ❌ ({len(parts)}个字段)")
    print()
    
    return {
        'rgb_dir': rgb_dir,
        'depth_dir': depth_dir,
        'label_paths': label_paths
    }


def check_label_loading():
    """检查YOLO是否能正确加载标签"""
    print("=" * 80)
    print("2️⃣  检查YOLO标签加载逻辑")
    print("=" * 80)
    print()
    
    # YOLO默认的标签查找逻辑
    print("📝 YOLO默认标签路径映射:")
    print("  images/rgb/xxx.jpg → labels/rgb/xxx.txt ❌ (需要自定义)")
    print("  images/train/xxx.jpg → labels/train/xxx.txt ✅ (标准)")
    print()
    
    # 检查ultralytics/data/dataset.py中的逻辑
    dataset_py = Path("ultralytics/data/dataset.py")
    if dataset_py.exists():
        print("🔍 检查YOLORGBDDataset实现...")
        with open(dataset_py) as f:
            content = f.read()
            
            # 查找关键函数
            if "def img2label_paths" in content:
                print("  ✅ 找到img2label_paths函数")
            else:
                print("  ❌ 未找到img2label_paths函数 (可能有问题)")
            
            # 检查是否有RGB-D特定处理
            if "images/rgb" in content or "images/d" in content:
                print("  ✅ 找到RGB-D路径处理")
            else:
                print("  ⚠️  未找到RGB-D路径处理 (可能使用默认逻辑)")
    print()


def check_model_input_channels():
    """检查模型实际接收的通道数"""
    print("=" * 80)
    print("3️⃣  检查模型输入通道数")
    print("=" * 80)
    print()
    
    # 读取模型配置
    model_yaml = Path("ultralytics/cfg/models/12/yolo12n-rgbd-v1.yaml")
    if model_yaml.exists():
        with open(model_yaml, encoding='utf-8') as f:
            model_cfg = yaml.safe_load(f)
        
        # 检查RGBDStem配置
        backbone = model_cfg.get('backbone', [])
        if backbone and len(backbone) > 0:
            first_layer = backbone[0]
            print(f"📊 第一层配置: {first_layer}")
            if 'RGBDStem' in str(first_layer):
                args = first_layer[3]
                c1, c2 = args[0], args[1]
                print(f"  ✅ RGBDStem配置:")
                print(f"    输入通道 (c1): {c1} (应该是4)")
                print(f"    输出通道 (c2): {c2} (Nano应该是64)")
            else:
                print(f"  ❌ 第一层不是RGBDStem: {first_layer}")
    print()


def check_training_logs():
    """检查训练日志中的关键信息"""
    print("=" * 80)
    print("4️⃣  分析训练日志")
    print("=" * 80)
    print()
    
    # 读取results.csv
    results_csv = Path("runs/train/exp_joint_v15/results.csv")
    if results_csv.exists():
        import pandas as pd
        df = pd.read_csv(results_csv)
        
        print("📈 训练曲线分析:")
        print()
        
        # Epoch 10性能
        if len(df) >= 10:
            epoch10 = df.iloc[9]  # 0-indexed
            print(f"  Epoch 10:")
            print(f"    mAP@0.5: {epoch10['metrics/mAP50(B)']:.2%}")
            print(f"    Precision: {epoch10['metrics/precision(B)']:.2%}")
            print(f"    Recall: {epoch10['metrics/recall(B)']:.2%}")
        
        # 最终性能
        final = df.iloc[-1]
        print(f"\n  Epoch {int(final['epoch'])}:")
        print(f"    mAP@0.5: {final['metrics/mAP50(B)']:.2%}")
        print(f"    Precision: {final['metrics/precision(B)']:.2%}")
        print(f"    Recall: {final['metrics/recall(B)']:.2%}")
        
        # Loss收敛检查
        print(f"\n  Loss分析:")
        print(f"    Box Loss: {final['train/box_loss']:.4f} (训练) vs {final['val/box_loss']:.4f} (验证)")
        print(f"    Cls Loss: {final['train/cls_loss']:.4f} (训练) vs {final['val/cls_loss']:.4f} (验证)")
        
        # 过拟合检查
        val_train_ratio = final['val/box_loss'] / final['train/box_loss']
        print(f"\n  过拟合检查:")
        print(f"    Val/Train Loss比例: {val_train_ratio:.2f}")
        if val_train_ratio > 2.0:
            print(f"      ⚠️  严重过拟合!")
        elif val_train_ratio > 1.5:
            print(f"      ⚠️  轻微过拟合")
        else:
            print(f"      ✅ 正常")
        
        # 性能诊断
        print(f"\n  🔴 性能诊断:")
        final_map = final['metrics/mAP50(B)']
        if final_map < 0.15:
            print(f"    💥 性能极差 (mAP {final_map:.1%} < 15%)")
            print(f"    可能原因:")
            print(f"      1. 标签加载失败 (大部分图像没有标签)")
            print(f"      2. 标签格式错误 (class_id超出范围)")
            print(f"      3. RGB-D加载失败 (只用了RGB或只用了Depth)")
        elif final_map < 0.30:
            print(f"    ⚠️  性能很差 (mAP {final_map:.1%} < 30%)")
            print(f"    可能原因:")
            print(f"      1. 部分标签加载失败")
            print(f"      2. 数据增强过度")
            print(f"      3. 学习率不合适")
        else:
            print(f"    ✅ 性能合理 (mAP {final_map:.1%} ≥ 30%)")
    print()


def generate_recommendations():
    """生成修复建议"""
    print("=" * 80)
    print("5️⃣  修复建议")
    print("=" * 80)
    print()
    
    print("🔧 立即执行:")
    print()
    print("  1. 验证UAVDT标签路径:")
    print("     ```bash")
    print("     # 在服务器上执行")
    print("     ls /data2/user/2024/lzy/Datasets/UAVDT_YOLO/train/labels/*.txt | wc -l")
    print("     ls /data2/user/2024/lzy/Datasets/UAVDT_YOLO/train/labels/rgb/*.txt | wc -l")
    print("     ```")
    print("     期望: 其中一个有23,829个文件")
    print()
    
    print("  2. 检查标签格式:")
    print("     ```bash")
    print("     head /data2/user/2024/lzy/Datasets/UAVDT_YOLO/train/labels/rgb/*.txt | head -20")
    print("     ```")
    print("     期望: 每行格式为 'class_id x_center y_center width height'")
    print()
    
    print("  3. 测试数据加载:")
    print("     ```python")
    print("     from ultralytics.data import build_dataloader")
    print("     # 尝试加载一个batch")
    print("     # 检查batch['img'].shape是否是[B, 4, 640, 640]")
    print("     # 检查batch['bboxes']是否有足够的标注")
    print("     ```")
    print()
    
    print("  4. 如果标签在labels/rgb/，修改数据集配置:")
    print("     在visdrone_uavdt_joint.yaml中添加:")
    print("     ```yaml")
    print("     train_labels:")
    print("       - VisDrone2019-DET-YOLO/.../labels/rgb")
    print("       - UAVDT_YOLO/train/labels/rgb")
    print("     ```")
    print()


def main():
    """主函数"""
    print()
    print("🔍 exp_joint_v15 训练失败诊断")
    print()
    
    os.chdir(Path(__file__).parent)
    
    # 执行检查
    dirs = check_dataset_structure()
    check_label_loading()
    check_model_input_channels()
    check_training_logs()
    generate_recommendations()
    
    print("=" * 80)
    print("✅ 诊断完成！请根据上述建议修复问题")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
