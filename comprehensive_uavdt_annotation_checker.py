"""
UAVDT标注质量综合检查工具
===========================

功能:
1. 基础统计分析 (bbox尺寸、类别分布、长宽比等)
2. 异常检测 (超界框、畸形框、极端尺寸、重叠度)
3. 可视化抽样 (随机绘制bbox验证标注正确性)
4. 与VisDrone对比 (找出数据集间的差异)

使用方法:
python comprehensive_uavdt_annotation_checker.py \
    --uavdt_root /data2/user/2024/lzy/Datasets/UAVDT_YOLO \
    --visdrone_root /data2/user/2024/lzy/Datasets/VisDrone \
    --output_dir ./uavdt_annotation_analysis \
    --num_visualize 100
"""

import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import json


class AnnotationChecker:
    """UAVDT标注质量检查器"""
    
    def __init__(self, dataset_root, dataset_name="UAVDT"):
        self.dataset_root = Path(dataset_root)
        self.dataset_name = dataset_name
        
        # 类别映射 (VisDrone 10类)
        self.class_names = {
            0: 'ignored',
            1: 'pedestrian',
            2: 'people',
            3: 'car',
            4: 'van',
            5: 'truck',
            6: 'tricycle',
            7: 'awning-tricycle',
            8: 'bus',
            9: 'motor'
        }
        
        # 统计信息
        self.stats = {
            'total_images': 0,
            'total_objects': 0,
            'category_count': Counter(),
            'bbox_areas': [],
            'bbox_widths': [],
            'bbox_heights': [],
            'aspect_ratios': [],
            'abnormal_boxes': [],  # 异常框
            'empty_images': [],    # 无标注图像
            'errors': [],
        }
    
    def check_split(self, split='train'):
        """检查某个split的标注质量"""
        print(f"\n{'='*80}")
        print(f"检查 {self.dataset_name} - {split} split")
        print(f"{'='*80}\n")
        
        # 路径
        image_dir = self.dataset_root / split / 'images' / 'rgb'
        label_dir = self.dataset_root / split / 'labels' / 'rgb'
        
        if not image_dir.exists():
            print(f"❌ 图像目录不存在: {image_dir}")
            return None
        
        if not label_dir.exists():
            print(f"❌ 标签目录不存在: {label_dir}")
            return None
        
        # 获取所有图像
        image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
        print(f"找到 {len(image_files)} 张图像")
        
        # 统计信息
        split_stats = {
            'total_images': len(image_files),
            'total_objects': 0,
            'category_count': Counter(),
            'bbox_areas': [],
            'bbox_widths': [],
            'bbox_heights': [],
            'aspect_ratios': [],
            'abnormal_boxes': [],
            'empty_images': [],
            'errors': [],
        }
        
        # 遍历每张图像
        for img_file in tqdm(image_files, desc=f"分析{split}集"):
            # 对应的标签文件
            label_file = label_dir / (img_file.stem + '.txt')
            
            # 读取图像尺寸
            img = cv2.imread(str(img_file))
            if img is None:
                split_stats['errors'].append(f"无法读取图像: {img_file}")
                continue
            
            img_h, img_w = img.shape[:2]
            
            # 检查标签文件
            if not label_file.exists():
                split_stats['empty_images'].append(str(img_file.name))
                continue
            
            # 读取标注
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                if not lines:
                    split_stats['empty_images'].append(str(img_file.name))
                    continue
                
                # 解析每个bbox
                for line_idx, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 5:
                        split_stats['errors'].append(
                            f"{img_file.name} 第{line_idx+1}行: 格式错误"
                        )
                        continue
                    
                    # 解析YOLO格式 (class x_center y_center width height)
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # 统计类别
                        split_stats['category_count'][class_id] += 1
                        split_stats['total_objects'] += 1
                        
                        # 计算实际像素尺寸
                        bbox_w_px = width * img_w
                        bbox_h_px = height * img_h
                        bbox_area = bbox_w_px * bbox_h_px
                        
                        # 记录尺寸信息
                        split_stats['bbox_widths'].append(bbox_w_px)
                        split_stats['bbox_heights'].append(bbox_h_px)
                        split_stats['bbox_areas'].append(bbox_area)
                        
                        # 长宽比
                        if bbox_h_px > 0:
                            aspect_ratio = bbox_w_px / bbox_h_px
                            split_stats['aspect_ratios'].append(aspect_ratio)
                        
                        # 异常检测
                        abnormal = self.detect_abnormal_bbox(
                            img_file.name, class_id, 
                            x_center, y_center, width, height,
                            img_w, img_h
                        )
                        if abnormal:
                            split_stats['abnormal_boxes'].append(abnormal)
                    
                    except ValueError as e:
                        split_stats['errors'].append(
                            f"{img_file.name} 第{line_idx+1}行: 解析错误 - {e}"
                        )
            
            except Exception as e:
                split_stats['errors'].append(f"{img_file.name}: {e}")
        
        return split_stats
    
    def detect_abnormal_bbox(self, img_name, class_id, 
                            x_center, y_center, width, height,
                            img_w, img_h):
        """检测异常bbox"""
        issues = []
        
        # 1. 超出图像边界
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2
        
        if x_min < 0 or y_min < 0 or x_max > 1 or y_max > 1:
            issues.append("超出边界")
        
        # 2. 尺寸异常 (过小或过大)
        bbox_w_px = width * img_w
        bbox_h_px = height * img_h
        
        if bbox_w_px < 3 or bbox_h_px < 3:
            issues.append(f"过小 ({bbox_w_px:.1f}x{bbox_h_px:.1f}px)")
        
        if width > 0.8 or height > 0.8:
            issues.append(f"过大 ({width:.2f}x{height:.2f})")
        
        # 3. 长宽比异常
        if bbox_h_px > 0:
            aspect_ratio = bbox_w_px / bbox_h_px
            if aspect_ratio > 10 or aspect_ratio < 0.1:
                issues.append(f"长宽比异常 ({aspect_ratio:.2f})")
        
        # 4. 中心点异常
        if x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1:
            issues.append("中心点超界")
        
        if issues:
            return {
                'image': img_name,
                'class': class_id,
                'bbox': [x_center, y_center, width, height],
                'issues': issues
            }
        
        return None
    
    def visualize_annotations(self, split='train', num_samples=50, output_dir='./vis'):
        """可视化标注结果"""
        print(f"\n{'='*80}")
        print(f"可视化 {self.dataset_name} - {split} split 的标注")
        print(f"{'='*80}\n")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 路径
        image_dir = self.dataset_root / split / 'images' / 'rgb'
        label_dir = self.dataset_root / split / 'labels' / 'rgb'
        
        # 随机选择图像
        image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
        if len(image_files) > num_samples:
            np.random.seed(42)
            image_files = np.random.choice(image_files, num_samples, replace=False)
        
        print(f"随机抽取 {len(image_files)} 张图像进行可视化...")
        
        # 颜色映射
        colors = {
            0: (128, 128, 128),  # ignored - 灰色
            1: (255, 0, 0),      # pedestrian - 红色
            2: (255, 165, 0),    # people - 橙色
            3: (0, 255, 0),      # car - 绿色
            4: (0, 255, 255),    # van - 青色
            5: (255, 255, 0),    # truck - 黄色
            6: (255, 0, 255),    # tricycle - 品红
            7: (128, 0, 255),    # awning-tricycle - 紫色
            8: (0, 0, 255),      # bus - 蓝色
            9: (255, 128, 0),    # motor - 橙红
        }
        
        for img_file in tqdm(image_files, desc="绘制标注"):
            # 读取图像
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            img_h, img_w = img.shape[:2]
            
            # 读取标签
            label_file = label_dir / (img_file.stem + '.txt')
            if not label_file.exists():
                continue
            
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # 绘制每个bbox
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # 转换为像素坐标
                    x_min = int((x_center - width / 2) * img_w)
                    y_min = int((y_center - height / 2) * img_h)
                    x_max = int((x_center + width / 2) * img_w)
                    y_max = int((y_center + height / 2) * img_h)
                    
                    # 绘制bbox
                    color = colors.get(class_id, (255, 255, 255))
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
                    
                    # 绘制类别标签
                    label_text = self.class_names.get(class_id, f"class_{class_id}")
                    cv2.putText(img, label_text, (x_min, y_min - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                except:
                    continue
            
            # 保存
            output_file = output_dir / f"{self.dataset_name}_{split}_{img_file.name}"
            cv2.imwrite(str(output_file), img)
        
        print(f"✅ 可视化完成! 保存到: {output_dir}")
        return output_dir
    
    def print_statistics(self, stats):
        """打印统计结果"""
        print(f"\n{'='*80}")
        print("📊 统计结果")
        print(f"{'='*80}\n")
        
        print(f"总图像数:     {stats['total_images']}")
        print(f"总对象数:     {stats['total_objects']}")
        print(f"空图像数:     {len(stats['empty_images'])}")
        print(f"异常框数:     {len(stats['abnormal_boxes'])}")
        print(f"错误数:       {len(stats['errors'])}")
        print()
        
        # 类别分布
        print("类别分布:")
        total_objs = sum(stats['category_count'].values())
        for class_id in sorted(stats['category_count'].keys()):
            count = stats['category_count'][class_id]
            percentage = count / total_objs * 100 if total_objs > 0 else 0
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            print(f"  {class_id} ({class_name:>20}): {count:>8} ({percentage:>5.2f}%)")
        print()
        
        # Bbox尺寸统计
        if stats['bbox_widths']:
            print("Bbox尺寸统计 (像素):")
            print(f"  宽度: min={np.min(stats['bbox_widths']):.1f}, "
                  f"mean={np.mean(stats['bbox_widths']):.1f}, "
                  f"median={np.median(stats['bbox_widths']):.1f}, "
                  f"max={np.max(stats['bbox_widths']):.1f}")
            print(f"  高度: min={np.min(stats['bbox_heights']):.1f}, "
                  f"mean={np.mean(stats['bbox_heights']):.1f}, "
                  f"median={np.median(stats['bbox_heights']):.1f}, "
                  f"max={np.max(stats['bbox_heights']):.1f}")
            print(f"  面积: min={np.min(stats['bbox_areas']):.1f}, "
                  f"mean={np.mean(stats['bbox_areas']):.1f}, "
                  f"median={np.median(stats['bbox_areas']):.1f}, "
                  f"max={np.max(stats['bbox_areas']):.1f}")
            print()
        
        # 长宽比
        if stats['aspect_ratios']:
            print("长宽比统计:")
            print(f"  min={np.min(stats['aspect_ratios']):.2f}, "
                  f"mean={np.mean(stats['aspect_ratios']):.2f}, "
                  f"median={np.median(stats['aspect_ratios']):.2f}, "
                  f"max={np.max(stats['aspect_ratios']):.2f}")
            print()
        
        # 异常框示例
        if stats['abnormal_boxes']:
            print(f"⚠️  发现 {len(stats['abnormal_boxes'])} 个异常框 (显示前10个):")
            for abnormal in stats['abnormal_boxes'][:10]:
                print(f"  - {abnormal['image']} | "
                      f"类别{abnormal['class']} | "
                      f"问题: {', '.join(abnormal['issues'])}")
            if len(stats['abnormal_boxes']) > 10:
                print(f"  ... 还有 {len(stats['abnormal_boxes']) - 10} 个异常框")
            print()
        
        # 错误示例
        if stats['errors']:
            print(f"❌ 发现 {len(stats['errors'])} 个错误 (显示前10个):")
            for error in stats['errors'][:10]:
                print(f"  - {error}")
            if len(stats['errors']) > 10:
                print(f"  ... 还有 {len(stats['errors']) - 10} 个错误")
            print()
    
    def plot_statistics(self, stats, output_file='stats.png'):
        """绘制统计图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.dataset_name} 标注质量分析', fontsize=16)
        
        # 1. 类别分布
        if stats['category_count']:
            ax = axes[0, 0]
            class_ids = sorted(stats['category_count'].keys())
            counts = [stats['category_count'][cid] for cid in class_ids]
            labels = [self.class_names.get(cid, f"c{cid}") for cid in class_ids]
            ax.bar(range(len(class_ids)), counts)
            ax.set_xticks(range(len(class_ids)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_title('类别分布')
            ax.set_ylabel('数量')
            ax.grid(axis='y', alpha=0.3)
        
        # 2. Bbox宽度分布
        if stats['bbox_widths']:
            ax = axes[0, 1]
            ax.hist(stats['bbox_widths'], bins=50, edgecolor='black', alpha=0.7)
            ax.set_title('Bbox宽度分布 (像素)')
            ax.set_xlabel('宽度')
            ax.set_ylabel('频次')
            ax.axvline(np.median(stats['bbox_widths']), color='r', 
                      linestyle='--', label=f'中位数={np.median(stats["bbox_widths"]):.1f}')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        # 3. Bbox高度分布
        if stats['bbox_heights']:
            ax = axes[0, 2]
            ax.hist(stats['bbox_heights'], bins=50, edgecolor='black', alpha=0.7)
            ax.set_title('Bbox高度分布 (像素)')
            ax.set_xlabel('高度')
            ax.set_ylabel('频次')
            ax.axvline(np.median(stats['bbox_heights']), color='r', 
                      linestyle='--', label=f'中位数={np.median(stats["bbox_heights"]):.1f}')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        # 4. Bbox面积分布 (对数坐标)
        if stats['bbox_areas']:
            ax = axes[1, 0]
            ax.hist(np.log10(stats['bbox_areas']), bins=50, edgecolor='black', alpha=0.7)
            ax.set_title('Bbox面积分布 (log10)')
            ax.set_xlabel('log10(面积)')
            ax.set_ylabel('频次')
            ax.grid(axis='y', alpha=0.3)
        
        # 5. 长宽比分布
        if stats['aspect_ratios']:
            ax = axes[1, 1]
            # 过滤极端值
            ar_filtered = [ar for ar in stats['aspect_ratios'] if 0.1 <= ar <= 10]
            ax.hist(ar_filtered, bins=50, edgecolor='black', alpha=0.7)
            ax.set_title('长宽比分布 (0.1-10)')
            ax.set_xlabel('长宽比')
            ax.set_ylabel('频次')
            ax.axvline(np.median(ar_filtered), color='r', 
                      linestyle='--', label=f'中位数={np.median(ar_filtered):.2f}')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        # 6. 小目标分析 (面积 < 32x32)
        if stats['bbox_areas']:
            ax = axes[1, 2]
            small_threshold = 32 * 32
            small_objs = [area for area in stats['bbox_areas'] if area < small_threshold]
            medium_objs = [area for area in stats['bbox_areas'] 
                          if small_threshold <= area < 96*96]
            large_objs = [area for area in stats['bbox_areas'] if area >= 96*96]
            
            sizes = ['Small\n(<32²)', 'Medium\n(32²-96²)', 'Large\n(≥96²)']
            counts = [len(small_objs), len(medium_objs), len(large_objs)]
            ax.bar(sizes, counts, color=['red', 'orange', 'green'], alpha=0.7)
            ax.set_title('目标尺寸分布')
            ax.set_ylabel('数量')
            for i, count in enumerate(counts):
                percentage = count / len(stats['bbox_areas']) * 100
                ax.text(i, count, f'{count}\n({percentage:.1f}%)', 
                       ha='center', va='bottom')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ 统计图表保存到: {output_file}")
        plt.close()


def compare_datasets(uavdt_stats, visdrone_stats, output_dir):
    """对比UAVDT和VisDrone数据集"""
    print(f"\n{'='*80}")
    print("📊 数据集对比: UAVDT vs VisDrone")
    print(f"{'='*80}\n")
    
    # 对比类别分布
    print("类别分布对比 (训练集):")
    print(f"{'类别':<20} | {'UAVDT':>12} | {'VisDrone':>12} | {'比例':>10}")
    print("-" * 60)
    
    all_classes = set(uavdt_stats['category_count'].keys()) | \
                  set(visdrone_stats['category_count'].keys())
    
    class_names = {
        0: 'ignored',
        1: 'pedestrian',
        2: 'people',
        3: 'car',
        4: 'van',
        5: 'truck',
        6: 'tricycle',
        7: 'awning-tricycle',
        8: 'bus',
        9: 'motor'
    }
    
    for class_id in sorted(all_classes):
        uavdt_count = uavdt_stats['category_count'].get(class_id, 0)
        visdrone_count = visdrone_stats['category_count'].get(class_id, 0)
        
        ratio = uavdt_count / visdrone_count if visdrone_count > 0 else float('inf')
        class_name = class_names.get(class_id, f"class_{class_id}")
        
        print(f"{class_name:<20} | {uavdt_count:>12,} | {visdrone_count:>12,} | {ratio:>10.2f}")
    
    print()
    
    # 对比bbox尺寸
    print("Bbox尺寸对比:")
    print(f"{'指标':<20} | {'UAVDT':>12} | {'VisDrone':>12}")
    print("-" * 50)
    
    metrics = [
        ('平均宽度', np.mean(uavdt_stats['bbox_widths']), np.mean(visdrone_stats['bbox_widths'])),
        ('平均高度', np.mean(uavdt_stats['bbox_heights']), np.mean(visdrone_stats['bbox_heights'])),
        ('平均面积', np.mean(uavdt_stats['bbox_areas']), np.mean(visdrone_stats['bbox_areas'])),
        ('中位数宽度', np.median(uavdt_stats['bbox_widths']), np.median(visdrone_stats['bbox_widths'])),
        ('中位数高度', np.median(uavdt_stats['bbox_heights']), np.median(visdrone_stats['bbox_heights'])),
    ]
    
    for metric_name, uavdt_val, visdrone_val in metrics:
        print(f"{metric_name:<20} | {uavdt_val:>12.1f} | {visdrone_val:>12.1f}")
    
    print()
    
    # 小目标比例对比
    uavdt_small = sum(1 for area in uavdt_stats['bbox_areas'] if area < 32*32)
    visdrone_small = sum(1 for area in visdrone_stats['bbox_areas'] if area < 32*32)
    uavdt_small_ratio = uavdt_small / len(uavdt_stats['bbox_areas']) * 100
    visdrone_small_ratio = visdrone_small / len(visdrone_stats['bbox_areas']) * 100
    
    print("小目标(<32x32)比例:")
    print(f"  UAVDT:    {uavdt_small:>8,} / {len(uavdt_stats['bbox_areas']):>8,} ({uavdt_small_ratio:.2f}%)")
    print(f"  VisDrone: {visdrone_small:>8,} / {len(visdrone_stats['bbox_areas']):>8,} ({visdrone_small_ratio:.2f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(description="UAVDT标注质量综合检查工具")
    parser.add_argument('--uavdt_root', type=str, required=True,
                       help='UAVDT数据集根目录')
    parser.add_argument('--visdrone_root', type=str, default=None,
                       help='VisDrone数据集根目录 (用于对比)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'],
                       help='要检查的splits (默认: train val)')
    parser.add_argument('--output_dir', type=str, default='./uavdt_annotation_analysis',
                       help='输出目录')
    parser.add_argument('--num_visualize', type=int, default=100,
                       help='可视化的图像数量')
    parser.add_argument('--skip_vis', action='store_true',
                       help='跳过可视化 (仅做统计分析)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("🔍 UAVDT标注质量综合检查工具")
    print("="*80)
    print(f"UAVDT根目录: {args.uavdt_root}")
    print(f"输出目录:     {args.output_dir}")
    print(f"检查splits:   {args.splits}")
    print("="*80)
    
    # 检查UAVDT
    uavdt_checker = AnnotationChecker(args.uavdt_root, "UAVDT")
    uavdt_results = {}
    
    for split in args.splits:
        stats = uavdt_checker.check_split(split)
        if stats:
            uavdt_results[split] = stats
            uavdt_checker.print_statistics(stats)
            
            # 绘制统计图
            plot_file = output_dir / f"uavdt_{split}_stats.png"
            uavdt_checker.plot_statistics(stats, plot_file)
            
            # 可视化
            if not args.skip_vis:
                vis_dir = output_dir / f"visualizations_{split}"
                uavdt_checker.visualize_annotations(split, args.num_visualize, vis_dir)
    
    # 如果提供了VisDrone路径,进行对比
    if args.visdrone_root and 'train' in uavdt_results:
        visdrone_checker = AnnotationChecker(args.visdrone_root, "VisDrone")
        visdrone_stats = visdrone_checker.check_split('train')
        
        if visdrone_stats:
            visdrone_checker.print_statistics(visdrone_stats)
            compare_datasets(uavdt_results['train'], visdrone_stats, output_dir)
    
    # 保存完整报告
    report_file = output_dir / "annotation_quality_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        # 转换numpy类型为Python原生类型
        serializable_results = {}
        for split, stats in uavdt_results.items():
            serializable_results[split] = {
                'total_images': stats['total_images'],
                'total_objects': stats['total_objects'],
                'category_count': dict(stats['category_count']),
                'num_abnormal_boxes': len(stats['abnormal_boxes']),
                'num_empty_images': len(stats['empty_images']),
                'num_errors': len(stats['errors']),
                'bbox_stats': {
                    'width_mean': float(np.mean(stats['bbox_widths'])) if stats['bbox_widths'] else 0,
                    'height_mean': float(np.mean(stats['bbox_heights'])) if stats['bbox_heights'] else 0,
                    'area_mean': float(np.mean(stats['bbox_areas'])) if stats['bbox_areas'] else 0,
                },
                'abnormal_boxes_sample': stats['abnormal_boxes'][:100],
            }
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 完整报告已保存到: {report_file}")
    print(f"\n🎉 检查完成! 所有结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
