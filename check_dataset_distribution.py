#!/usr/bin/env python3
"""
检查VisDrone数据集的目标尺度分布

用法:
    python check_dataset_distribution.py
"""

import torch
from pathlib import Path
import numpy as np

def check_visdrone_distribution():
    """统计VisDrone验证集的目标尺度分布"""
    
    # 数据集路径（根据你的实际路径修改）
    dataset_path = Path("/data2/user/2024/lzy/Datasets/VisDrone2019-DET-YOLO/VisDrone2YOLO/VisDrone2019-DET-val")
    
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        print("请在服务器上运行此脚本，或修改 dataset_path 变量")
        return
    
    labels_dir = dataset_path / "labels" / "rgb"
    
    if not labels_dir.exists():
        print(f"❌ Labels directory not found: {labels_dir}")
        return
    
    # COCO标准阈值
    small_thresh = 1024    # 32²
    medium_thresh = 9216   # 96²
    
    # 统计计数器
    small_count = 0
    medium_count = 0
    large_count = 0
    
    # VisDrone图像标准尺寸
    img_width, img_height = 1920, 1080
    
    # 遍历所有标签文件
    label_files = list(labels_dir.glob("*.txt"))
    print(f"📂 Found {len(label_files)} label files")
    
    for label_file in label_files:
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # YOLO format: class x_center y_center width height (normalized 0-1)
                    w_norm, h_norm = float(parts[3]), float(parts[4])
                    
                    # 转换为绝对像素
                    abs_w = w_norm * img_width
                    abs_h = h_norm * img_height
                    area = abs_w * abs_h
                    
                    # 分类统计
                    if area < small_thresh:
                        small_count += 1
                    elif area < medium_thresh:
                        medium_count += 1
                    else:
                        large_count += 1
    
    # 计算总数和百分比
    total = small_count + medium_count + large_count
    
    print(f"\n{'='*80}")
    print(f"{'VisDrone验证集目标尺度分布 (COCO标准)':^80}")
    print(f"{'='*80}\n")
    
    print(f"📊 总目标数: {total:,}\n")
    
    print(f"{'尺度范围':<20} {'数量':>10} {'百分比':>10} {'COCO标准':>15}")
    print(f"{'-'*80}")
    print(f"{'Small (<32²)':<20} {small_count:>10,} {small_count/total*100:>9.1f}% {f'< {small_thresh}':>15}")
    print(f"{'Medium (32²~96²)':<20} {medium_count:>10,} {medium_count/total*100:>9.1f}% {f'{small_thresh}-{medium_thresh}':>15}")
    print(f"{'Large (≥96²)':<20} {large_count:>10,} {large_count/total*100:>9.1f}% {f'≥ {medium_thresh}':>15}")
    print(f"{'='*80}\n")
    
    # 与RemDet/COCO对比
    print("📌 对比分析:\n")
    
    # COCO标准分布（通用数据集）
    coco_small_pct = 41.4
    coco_medium_pct = 34.5
    coco_large_pct = 24.1
    
    visdrone_small_pct = small_count / total * 100
    visdrone_medium_pct = medium_count / total * 100
    visdrone_large_pct = large_count / total * 100
    
    print(f"{'尺度':<15} {'VisDrone':>12} {'COCO标准':>12} {'差异':>12}")
    print(f"{'-'*60}")
    print(f"{'Small':<15} {visdrone_small_pct:>11.1f}% {coco_small_pct:>11.1f}% {visdrone_small_pct-coco_small_pct:>+11.1f}%")
    print(f"{'Medium':<15} {visdrone_medium_pct:>11.1f}% {coco_medium_pct:>11.1f}% {visdrone_medium_pct-coco_medium_pct:>+11.1f}%")
    print(f"{'Large':<15} {visdrone_large_pct:>11.1f}% {coco_large_pct:>11.1f}% {visdrone_large_pct-coco_large_pct:>+11.1f}%")
    print(f"{'='*60}\n")
    
    # 分析结论
    print("💡 结论:")
    if visdrone_medium_pct < 15:
        print(f"  ⚠️  Medium目标占比仅 {visdrone_medium_pct:.1f}% (远低于COCO的34.5%)")
        print(f"  ✅ Medium mAP={14.28:.2f}% 可能是正常现象（样本少+难度大）")
        print(f"  💡 建议: 优先改进Small mAP (占比{visdrone_small_pct:.1f}%)，对总体mAP影响更大")
    elif visdrone_medium_pct >= 15 and visdrone_medium_pct < 25:
        print(f"  ⏸️  Medium目标占比 {visdrone_medium_pct:.1f}% (低于COCO但尚可)")
        print(f"  ❌ Medium mAP={14.28:.2f}% 偏低，有改进空间")
        print(f"  💡 建议: Phase 3实施ChannelC2f改进中等尺度特征")
    else:
        print(f"  ❌ Medium目标占比 {visdrone_medium_pct:.1f}% (正常范围)")
        print(f"  🔴 Medium mAP={14.28:.2f}% 严重偏低！模型对Medium目标检测能力不足")
        print(f"  💡 建议: 立即优先Phase 3 + Phase 4改进Medium检测")
    
    print()

if __name__ == "__main__":
    check_visdrone_distribution()
