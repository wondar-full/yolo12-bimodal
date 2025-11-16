"""
UAVDT问题快速诊断脚本
====================

快速量化分析:
1. 超界框比例
2. 过大框比例
3. 过小框比例
4. 估算漏标情况
5. 与VisDrone对比

使用:
python quick_uavdt_diagnosis.py \
    --uavdt_root /data2/user/2024/lzy/Datasets/UAVDT_YOLO \
    --visdrone_root /data2/user/2024/lzy/Datasets/VisDrone
"""

import cv2
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import argparse


def diagnose_dataset(dataset_root, split='train'):
    """快速诊断数据集问题"""
    
    image_dir = Path(dataset_root) / split / 'images' / 'rgb'
    label_dir = Path(dataset_root) / split / 'labels' / 'rgb'
    
    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    
    # 统计
    stats = {
        'total_boxes': 0,
        'out_of_bound': 0,      # 超界框
        'too_small': 0,          # 过小(<5x5)
        'too_large': 0,          # 过大(>70%)
        'extreme_aspect': 0,     # 极端长宽比
        'boxes_per_image': [],   # 每张图的框数
    }
    
    print(f"分析 {len(image_files)} 张图像...")
    
    for img_file in tqdm(image_files, desc="诊断"):
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        
        img_h, img_w = img.shape[:2]
        
        label_file = label_dir / (img_file.stem + '.txt')
        if not label_file.exists():
            stats['boxes_per_image'].append(0)
            continue
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        box_count = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            try:
                x_c = float(parts[1])
                y_c = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                
                stats['total_boxes'] += 1
                box_count += 1
                
                # 检查超界
                x_min, y_min = x_c - w/2, y_c - h/2
                x_max, y_max = x_c + w/2, y_c + h/2
                if x_min < 0 or y_min < 0 or x_max > 1 or y_max > 1:
                    stats['out_of_bound'] += 1
                
                # 检查过小
                w_px, h_px = w * img_w, h * img_h
                if w_px < 5 or h_px < 5:
                    stats['too_small'] += 1
                
                # 检查过大
                if w > 0.7 or h > 0.7:
                    stats['too_large'] += 1
                
                # 检查长宽比
                if h_px > 0:
                    ar = w_px / h_px
                    if ar < 0.1 or ar > 10:
                        stats['extreme_aspect'] += 1
            
            except:
                continue
        
        stats['boxes_per_image'].append(box_count)
    
    return stats


def print_diagnosis(name, stats):
    """打印诊断结果"""
    print(f"\n{'='*80}")
    print(f"📊 {name} 数据集诊断")
    print(f"{'='*80}\n")
    
    total = stats['total_boxes']
    
    print(f"总框数:          {total:>10,}")
    print(f"超界框:          {stats['out_of_bound']:>10,} ({stats['out_of_bound']/total*100:>5.2f}%)")
    print(f"过小框(<5px):    {stats['too_small']:>10,} ({stats['too_small']/total*100:>5.2f}%)")
    print(f"过大框(>70%):    {stats['too_large']:>10,} ({stats['too_large']/total*100:>5.2f}%)")
    print(f"极端长宽比:      {stats['extreme_aspect']:>10,} ({stats['extreme_aspect']/total*100:>5.2f}%)")
    print()
    
    # 问题框总计
    total_problems = stats['out_of_bound'] + stats['too_small'] + \
                    stats['too_large'] + stats['extreme_aspect']
    print(f"🚨 问题框总计:   {total_problems:>10,} ({total_problems/total*100:>5.2f}%)")
    print()
    
    # 每张图的框数统计
    boxes_per_img = stats['boxes_per_image']
    if boxes_per_img:
        print(f"每张图平均框数: {np.mean(boxes_per_img):>10.2f}")
        print(f"中位数:          {np.median(boxes_per_img):>10.2f}")
        print(f"最小值:          {np.min(boxes_per_img):>10.0f}")
        print(f"最大值:          {np.max(boxes_per_img):>10.0f}")
        print(f"空图像数:        {sum(1 for x in boxes_per_img if x==0):>10,}")
    
    print(f"\n{'='*80}\n")


def compare_quality(uavdt_stats, visdrone_stats):
    """对比数据质量"""
    print(f"\n{'='*80}")
    print("🔍 质量对比: UAVDT vs VisDrone")
    print(f"{'='*80}\n")
    
    print(f"{'指标':<25} | {'UAVDT':>12} | {'VisDrone':>12} | {'差异':>10}")
    print("-" * 70)
    
    # 问题框比例
    uavdt_total = uavdt_stats['total_boxes']
    visdrone_total = visdrone_stats['total_boxes']
    
    metrics = [
        ('超界框比例(%)', 
         uavdt_stats['out_of_bound']/uavdt_total*100,
         visdrone_stats['out_of_bound']/visdrone_total*100),
        ('过小框比例(%)',
         uavdt_stats['too_small']/uavdt_total*100,
         visdrone_stats['too_small']/visdrone_total*100),
        ('过大框比例(%)',
         uavdt_stats['too_large']/uavdt_total*100,
         visdrone_stats['too_large']/visdrone_total*100),
        ('平均框数/图',
         np.mean(uavdt_stats['boxes_per_image']),
         np.mean(visdrone_stats['boxes_per_image'])),
    ]
    
    for name, uavdt_val, visdrone_val in metrics:
        diff = uavdt_val - visdrone_val
        print(f"{name:<25} | {uavdt_val:>12.2f} | {visdrone_val:>12.2f} | {diff:>+10.2f}")
    
    print(f"\n{'='*80}\n")
    
    # 结论
    print("💡 诊断结论:\n")
    
    uavdt_problem_rate = (uavdt_stats['out_of_bound'] + uavdt_stats['too_small'] + 
                         uavdt_stats['too_large']) / uavdt_total * 100
    visdrone_problem_rate = (visdrone_stats['out_of_bound'] + visdrone_stats['too_small'] + 
                            visdrone_stats['too_large']) / visdrone_total * 100
    
    if uavdt_problem_rate > 15:
        print(f"🚨 UAVDT问题框比例高达 {uavdt_problem_rate:.2f}% (vs VisDrone {visdrone_problem_rate:.2f}%)")
        print("   建议: 使用STRICT模式清洗数据")
    elif uavdt_problem_rate > 10:
        print(f"⚠️  UAVDT问题框比例 {uavdt_problem_rate:.2f}% (中等)")
        print("   建议: 使用MODERATE模式清洗数据")
    else:
        print(f"✅ UAVDT问题框比例 {uavdt_problem_rate:.2f}% (可接受)")
        print("   建议: 使用LOOSE模式清洗数据")
    
    print()
    
    # 漏标估计
    uavdt_avg = np.mean(uavdt_stats['boxes_per_image'])
    visdrone_avg = np.mean(visdrone_stats['boxes_per_image'])
    
    if uavdt_avg < visdrone_avg * 0.7:
        print(f"🔍 疑似漏标:")
        print(f"   UAVDT平均框数 ({uavdt_avg:.2f}) 远低于 VisDrone ({visdrone_avg:.2f})")
        print(f"   可能漏标率: ~{(1 - uavdt_avg/visdrone_avg)*100:.1f}%")
        print(f"   建议: 人工抽查可视化结果确认")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="UAVDT快速诊断")
    parser.add_argument('--uavdt_root', type=str, required=True)
    parser.add_argument('--visdrone_root', type=str, default=None)
    parser.add_argument('--split', type=str, default='train')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🔍 UAVDT快速诊断工具")
    print("="*80)
    
    # 诊断UAVDT
    uavdt_stats = diagnose_dataset(args.uavdt_root, args.split)
    print_diagnosis("UAVDT", uavdt_stats)
    
    # 诊断VisDrone (如果提供)
    if args.visdrone_root:
        visdrone_stats = diagnose_dataset(args.visdrone_root, args.split)
        print_diagnosis("VisDrone", visdrone_stats)
        compare_quality(uavdt_stats, visdrone_stats)
    
    print("\n✅ 诊断完成!")
    print("\n下一步建议:")
    print("1. 运行数据清洗工具: python uavdt_data_cleaner.py --mode moderate ...")
    print("2. 查看可视化结果确认漏标情况")
    print("3. 使用清洗后的数据重新训练")


if __name__ == "__main__":
    main()
