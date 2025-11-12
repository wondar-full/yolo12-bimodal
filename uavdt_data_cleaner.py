"""
UAVDT数据清洗工具
=================

针对发现的问题:
1. 大量超出边界的框
2. 边界框过大 (粗糙标注)
3. 边界框过小 (<3x3px)
4. 长宽比异常

功能:
- 过滤异常bbox
- 生成清洗后的数据集副本
- 统计清洗前后对比
- 支持严格/宽松模式

使用方法:
python uavdt_data_cleaner.py \
    --dataset_root /data2/user/2024/lzy/Datasets/UAVDT_YOLO \
    --output_root /data2/user/2024/lzy/Datasets/UAVDT_YOLO_CLEANED \
    --mode strict \
    --min_size 5 \
    --max_size_ratio 0.7 \
    --backup
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import argparse
import json


class UAVDTCleaner:
    """UAVDT数据清洗器"""
    
    def __init__(self, mode='moderate'):
        """
        Args:
            mode: 清洗模式
                - 'strict': 严格模式 (去除更多可疑框)
                - 'moderate': 中等模式 (平衡质量和数量)
                - 'loose': 宽松模式 (仅去除明显错误)
        """
        self.mode = mode
        
        # 清洗参数 (根据模式调整)
        self.params = self._get_cleaning_params(mode)
        
        # 统计信息
        self.stats = {
            'total_images': 0,
            'total_boxes_before': 0,
            'total_boxes_after': 0,
            'removed_boxes': 0,
            'removed_by_reason': Counter(),
            'empty_images_before': 0,
            'empty_images_after': 0,
            'removed_images': [],
        }
    
    def _get_cleaning_params(self, mode):
        """获取清洗参数"""
        params_dict = {
            'strict': {
                'min_bbox_size_px': 8,        # 最小bbox尺寸 (像素)
                'max_width_ratio': 0.6,       # 最大宽度比例
                'max_height_ratio': 0.6,      # 最大高度比例
                'min_aspect_ratio': 0.15,     # 最小长宽比
                'max_aspect_ratio': 8.0,      # 最大长宽比
                'boundary_tolerance': 0.0,    # 边界容忍度 (0=不允许超界)
                'min_area_px': 64,            # 最小面积 (8x8)
            },
            'moderate': {
                'min_bbox_size_px': 5,
                'max_width_ratio': 0.7,
                'max_height_ratio': 0.7,
                'min_aspect_ratio': 0.1,
                'max_aspect_ratio': 10.0,
                'boundary_tolerance': 0.02,   # 允许2%超出
                'min_area_px': 25,            # 最小面积 (5x5)
            },
            'loose': {
                'min_bbox_size_px': 3,
                'max_width_ratio': 0.8,
                'max_height_ratio': 0.8,
                'min_aspect_ratio': 0.05,
                'max_aspect_ratio': 20.0,
                'boundary_tolerance': 0.05,   # 允许5%超出
                'min_area_px': 9,             # 最小面积 (3x3)
            }
        }
        return params_dict[mode]
    
    def is_valid_bbox(self, x_center, y_center, width, height, img_w, img_h):
        """
        判断bbox是否有效
        
        Returns:
            valid: bool
            reason: str (如果invalid)
        """
        # 计算像素尺寸
        bbox_w_px = width * img_w
        bbox_h_px = height * img_h
        bbox_area = bbox_w_px * bbox_h_px
        
        # 1. 检查尺寸过小
        if bbox_w_px < self.params['min_bbox_size_px'] or \
           bbox_h_px < self.params['min_bbox_size_px']:
            return False, f"过小({bbox_w_px:.1f}x{bbox_h_px:.1f})"
        
        if bbox_area < self.params['min_area_px']:
            return False, f"面积过小({bbox_area:.1f}px²)"
        
        # 2. 检查尺寸过大
        if width > self.params['max_width_ratio'] or \
           height > self.params['max_height_ratio']:
            return False, f"过大({width:.2f}x{height:.2f})"
        
        # 3. 检查长宽比
        if bbox_h_px > 0:
            aspect_ratio = bbox_w_px / bbox_h_px
            if aspect_ratio < self.params['min_aspect_ratio'] or \
               aspect_ratio > self.params['max_aspect_ratio']:
                return False, f"长宽比异常({aspect_ratio:.2f})"
        
        # 4. 检查边界
        tolerance = self.params['boundary_tolerance']
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2
        
        if x_min < -tolerance or y_min < -tolerance or \
           x_max > 1 + tolerance or y_max > 1 + tolerance:
            return False, "超出边界"
        
        # 5. 检查中心点
        if x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1:
            return False, "中心点超界"
        
        return True, None
    
    def clip_bbox(self, x_center, y_center, width, height):
        """将超出边界的bbox裁剪到[0, 1]范围内"""
        # 计算边界
        x_min = max(0, x_center - width / 2)
        y_min = max(0, y_center - height / 2)
        x_max = min(1, x_center + width / 2)
        y_max = min(1, y_center + height / 2)
        
        # 重新计算中心点和尺寸
        new_width = x_max - x_min
        new_height = y_max - y_min
        new_x_center = (x_min + x_max) / 2
        new_y_center = (y_min + y_max) / 2
        
        return new_x_center, new_y_center, new_width, new_height
    
    def clean_label_file(self, label_file, img_w, img_h, clip_boundary=False):
        """
        清洗单个标签文件
        
        Args:
            label_file: 标签文件路径
            img_w: 图像宽度
            img_h: 图像高度
            clip_boundary: 是否裁剪超界框 (而不是删除)
        
        Returns:
            cleaned_lines: 清洗后的标签行
            removed_count: 删除的框数量
            removed_reasons: 删除原因统计
        """
        if not label_file.exists():
            return [], 0, Counter()
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        cleaned_lines = []
        removed_count = 0
        removed_reasons = Counter()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 5:
                removed_count += 1
                removed_reasons['格式错误'] += 1
                continue
            
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # 尝试裁剪边界 (如果启用)
                if clip_boundary:
                    x_center, y_center, width, height = self.clip_bbox(
                        x_center, y_center, width, height
                    )
                
                # 验证bbox
                valid, reason = self.is_valid_bbox(
                    x_center, y_center, width, height, img_w, img_h
                )
                
                if valid:
                    # 保留
                    new_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    if len(parts) > 5:  # 保留额外字段
                        new_line += " " + " ".join(parts[5:])
                    cleaned_lines.append(new_line + "\n")
                else:
                    # 删除
                    removed_count += 1
                    removed_reasons[reason] += 1
            
            except ValueError:
                removed_count += 1
                removed_reasons['解析错误'] += 1
        
        return cleaned_lines, removed_count, removed_reasons
    
    def clean_split(self, dataset_root, output_root, split='train', 
                   clip_boundary=False, remove_empty=False):
        """
        清洗某个split的数据
        
        Args:
            dataset_root: 原始数据集根目录
            output_root: 输出数据集根目录
            split: train/val/test
            clip_boundary: 是否裁剪超界框
            remove_empty: 是否删除清洗后无标注的图像
        """
        print(f"\n{'='*80}")
        print(f"清洗 {split} split - 模式: {self.mode}")
        print(f"{'='*80}\n")
        
        dataset_root = Path(dataset_root)
        output_root = Path(output_root)
        
        # 输入路径
        image_dir = dataset_root / split / 'images' / 'rgb'
        label_dir = dataset_root / split / 'labels' / 'rgb'
        depth_dir = dataset_root / split / 'images' / 'depth'
        
        # 输出路径
        output_image_dir = output_root / split / 'images' / 'rgb'
        output_label_dir = output_root / split / 'labels' / 'rgb'
        output_depth_dir = output_root / split / 'images' / 'depth'
        
        # 创建输出目录
        output_image_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)
        if depth_dir.exists():
            output_depth_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图像
        image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
        print(f"找到 {len(image_files)} 张图像")
        
        split_stats = {
            'total_images': len(image_files),
            'total_boxes_before': 0,
            'total_boxes_after': 0,
            'removed_boxes': 0,
            'removed_by_reason': Counter(),
            'empty_images_before': 0,
            'empty_images_after': 0,
            'removed_images': [],
        }
        
        # 处理每张图像
        for img_file in tqdm(image_files, desc=f"清洗{split}集"):
            # 读取图像获取尺寸
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            img_h, img_w = img.shape[:2]
            
            # 清洗标签
            label_file = label_dir / (img_file.stem + '.txt')
            
            if not label_file.exists():
                split_stats['empty_images_before'] += 1
                if not remove_empty:
                    # 复制图像
                    shutil.copy(img_file, output_image_dir / img_file.name)
                    # 复制深度图
                    depth_file = depth_dir / img_file.name
                    if depth_file.exists():
                        shutil.copy(depth_file, output_depth_dir / img_file.name)
                else:
                    split_stats['removed_images'].append(str(img_file.name))
                continue
            
            # 清洗标签
            cleaned_lines, removed, reasons = self.clean_label_file(
                label_file, img_w, img_h, clip_boundary
            )
            
            # 统计
            original_count = removed + len(cleaned_lines)
            split_stats['total_boxes_before'] += original_count
            split_stats['total_boxes_after'] += len(cleaned_lines)
            split_stats['removed_boxes'] += removed
            split_stats['removed_by_reason'].update(reasons)
            
            if original_count == 0:
                split_stats['empty_images_before'] += 1
            
            # 处理清洗后的结果
            if len(cleaned_lines) == 0:
                split_stats['empty_images_after'] += 1
                if remove_empty:
                    split_stats['removed_images'].append(str(img_file.name))
                    continue
            
            # 保存清洗后的标签
            output_label_file = output_label_dir / label_file.name
            with open(output_label_file, 'w') as f:
                f.writelines(cleaned_lines)
            
            # 复制图像和深度图
            shutil.copy(img_file, output_image_dir / img_file.name)
            depth_file = depth_dir / img_file.name
            if depth_file.exists():
                shutil.copy(depth_file, output_depth_dir / img_file.name)
        
        # 打印统计
        self._print_split_stats(split_stats, split)
        
        return split_stats
    
    def _print_split_stats(self, stats, split):
        """打印清洗统计"""
        print(f"\n{'='*80}")
        print(f"📊 清洗统计 - {split} split")
        print(f"{'='*80}\n")
        
        print(f"总图像数:           {stats['total_images']}")
        print(f"移除图像数:         {len(stats['removed_images'])}")
        print(f"保留图像数:         {stats['total_images'] - len(stats['removed_images'])}")
        print()
        
        print(f"清洗前总框数:       {stats['total_boxes_before']}")
        print(f"清洗后总框数:       {stats['total_boxes_after']}")
        print(f"移除框数:           {stats['removed_boxes']}")
        print(f"保留率:             {stats['total_boxes_after']/stats['total_boxes_before']*100 if stats['total_boxes_before']>0 else 0:.2f}%")
        print()
        
        print(f"清洗前空图像:       {stats['empty_images_before']}")
        print(f"清洗后空图像:       {stats['empty_images_after']}")
        print(f"新增空图像:         {stats['empty_images_after'] - stats['empty_images_before']}")
        print()
        
        if stats['removed_by_reason']:
            print("移除原因分布:")
            total_removed = sum(stats['removed_by_reason'].values())
            for reason, count in stats['removed_by_reason'].most_common():
                percentage = count / total_removed * 100
                print(f"  {reason:<20}: {count:>8} ({percentage:>5.2f}%)")
        
        print(f"\n{'='*80}\n")
    
    def create_yaml_config(self, output_root, dataset_name="uavdt_cleaned"):
        """创建清洗后数据集的YAML配置"""
        yaml_content = f"""# UAVDT Cleaned Dataset Configuration
# 清洗模式: {self.mode}
# 清洗参数: {self.params}

path: {output_root}
train: train/images/rgb
val: val/images/rgb
test: test/images/rgb

train_depth: train/images/depth
val_depth: val/images/depth
test_depth: test/images/depth

names:
  0: ignored
  1: pedestrian
  2: people
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor

# 清洗说明
# - 移除超界框: boundary_tolerance={self.params['boundary_tolerance']}
# - 移除过小框: min_size={self.params['min_bbox_size_px']}px
# - 移除过大框: max_ratio={self.params['max_width_ratio']}
# - 移除畸形框: aspect_ratio=[{self.params['min_aspect_ratio']}, {self.params['max_aspect_ratio']}]
"""
        yaml_file = Path(output_root) / f"{dataset_name}.yaml"
        with open(yaml_file, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        print(f"✅ YAML配置已保存: {yaml_file}")
        return yaml_file


def main():
    parser = argparse.ArgumentParser(description="UAVDT数据清洗工具")
    parser.add_argument('--dataset_root', type=str, required=True,
                       help='原始UAVDT数据集根目录')
    parser.add_argument('--output_root', type=str, required=True,
                       help='清洗后数据集输出目录')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                       help='要清洗的splits')
    parser.add_argument('--mode', type=str, default='moderate',
                       choices=['strict', 'moderate', 'loose'],
                       help='清洗模式: strict(严格)/moderate(中等)/loose(宽松)')
    parser.add_argument('--clip_boundary', action='store_true',
                       help='裁剪超界框(而不是删除)')
    parser.add_argument('--remove_empty', action='store_true',
                       help='删除清洗后无标注的图像')
    parser.add_argument('--backup', action='store_true',
                       help='备份原始数据集')
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    
    # 确认操作
    print("="*80)
    print("🧹 UAVDT数据清洗工具")
    print("="*80)
    print(f"原始数据集: {dataset_root}")
    print(f"输出目录:   {output_root}")
    print(f"清洗模式:   {args.mode}")
    print(f"清洗splits: {args.splits}")
    print(f"裁剪超界框: {'是' if args.clip_boundary else '否'}")
    print(f"删除空图像: {'是' if args.remove_empty else '否'}")
    print("="*80)
    
    # 备份
    if args.backup:
        backup_dir = dataset_root.parent / f"{dataset_root.name}_backup"
        if not backup_dir.exists():
            print(f"\n📦 备份原始数据到: {backup_dir}")
            shutil.copytree(dataset_root, backup_dir)
        else:
            print(f"\n⚠️  备份目录已存在: {backup_dir}")
    
    # 创建清洗器
    cleaner = UAVDTCleaner(mode=args.mode)
    
    # 显示清洗参数
    print("\n清洗参数:")
    for key, value in cleaner.params.items():
        print(f"  {key}: {value}")
    print()
    
    confirm = input("确认开始清洗? (yes/no): ")
    if confirm.lower() not in ['yes', 'y']:
        print("❌ 操作已取消")
        return
    
    # 清洗每个split
    all_stats = {}
    for split in args.splits:
        stats = cleaner.clean_split(
            dataset_root, output_root, split,
            clip_boundary=args.clip_boundary,
            remove_empty=args.remove_empty
        )
        if stats:
            all_stats[split] = stats
    
    # 创建YAML配置
    yaml_file = cleaner.create_yaml_config(output_root, f"uavdt_{args.mode}")
    
    # 保存完整报告
    report_file = output_root / "cleaning_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        serializable_stats = {}
        for split, stats in all_stats.items():
            serializable_stats[split] = {
                'total_images': stats['total_images'],
                'removed_images': len(stats['removed_images']),
                'total_boxes_before': stats['total_boxes_before'],
                'total_boxes_after': stats['total_boxes_after'],
                'removed_boxes': stats['removed_boxes'],
                'retention_rate': stats['total_boxes_after']/stats['total_boxes_before']*100 if stats['total_boxes_before']>0 else 0,
                'removed_by_reason': dict(stats['removed_by_reason']),
                'empty_images_before': stats['empty_images_before'],
                'empty_images_after': stats['empty_images_after'],
            }
        
        report = {
            'cleaning_mode': args.mode,
            'cleaning_params': cleaner.params,
            'clip_boundary': args.clip_boundary,
            'remove_empty': args.remove_empty,
            'splits': serializable_stats,
        }
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 清洗报告已保存: {report_file}")
    
    # 总结
    print(f"\n{'='*80}")
    print("🎉 数据清洗完成!")
    print(f"{'='*80}\n")
    
    total_boxes_before = sum(s['total_boxes_before'] for s in all_stats.values())
    total_boxes_after = sum(s['total_boxes_after'] for s in all_stats.values())
    total_removed = sum(s['removed_boxes'] for s in all_stats.values())
    
    print(f"总处理框数:   {total_boxes_before:>10,}")
    print(f"保留框数:     {total_boxes_after:>10,}")
    print(f"移除框数:     {total_removed:>10,}")
    print(f"保留率:       {total_boxes_after/total_boxes_before*100:>10.2f}%")
    print()
    print(f"清洗后数据集: {output_root}")
    print(f"YAML配置:     {yaml_file}")
    print()
    print("下一步:")
    print("1. 检查清洗后的数据质量")
    print("2. 使用清洗后的YAML配置重新训练")
    print("3. 对比清洗前后的训练效果")


if __name__ == "__main__":
    main()
