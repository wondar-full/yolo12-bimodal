#!/usr/bin/env python3
"""
VisDrone数据集目标尺寸分布统计脚本
按照COCO标准和自定义阈值分析训练集/验证集的Small/Medium/Large目标分布
"""

import os
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm


class DatasetAnalyzer:
    """数据集目标尺寸分布分析器"""
    
    def __init__(self, data_yaml_path, img_size=640):
        """
        Args:
            data_yaml_path: 数据集YAML配置文件路径
            img_size: 训练/验证时的图像尺寸
        """
        self.img_size = img_size
        self.data_yaml_path = Path(data_yaml_path)
        
        # 加载数据集配置
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            self.data_config = yaml.safe_load(f)
        
        # 获取数据集根目录 (处理可能的列表格式)
        path_value = self.data_config.get('path', '.')
        if isinstance(path_value, list):
            # YAML中path可能是多行写法,取第一个非空值
            self.data_root = Path([p for p in path_value if p][0])
        else:
            self.data_root = Path(path_value)
        
        # COCO标准阈值
        self.coco_small_thresh = 32 * 32  # 1024
        self.coco_medium_thresh = 96 * 96  # 9216
        
        # 自定义阈值 (用于对比)
        self.custom_small_thresh = 32 * 32  # 1024
        self.custom_medium_thresh = 64 * 64  # 4096
        
        print(f"📁 数据集根目录: {self.data_root}")
        print(f"📐 训练图像尺寸: {img_size}×{img_size}")
        print(f"\n📏 COCO标准阈值:")
        print(f"  Small:  area < {self.coco_small_thresh} (<{int(np.sqrt(self.coco_small_thresh))}×{int(np.sqrt(self.coco_small_thresh))})")
        print(f"  Medium: {self.coco_small_thresh} ≤ area < {self.coco_medium_thresh} ({int(np.sqrt(self.coco_small_thresh))}×{int(np.sqrt(self.coco_small_thresh))} ~ {int(np.sqrt(self.coco_medium_thresh))}×{int(np.sqrt(self.coco_medium_thresh))})")
        print(f"  Large:  area ≥ {self.coco_medium_thresh} (≥{int(np.sqrt(self.coco_medium_thresh))}×{int(np.sqrt(self.coco_medium_thresh))})")
        print(f"\n📏 自定义阈值 (VisDrone优化):")
        print(f"  Small:  area < {self.custom_small_thresh} (<{int(np.sqrt(self.custom_small_thresh))}×{int(np.sqrt(self.custom_small_thresh))})")
        print(f"  Medium: {self.custom_small_thresh} ≤ area < {self.custom_medium_thresh} ({int(np.sqrt(self.custom_small_thresh))}×{int(np.sqrt(self.custom_small_thresh))} ~ {int(np.sqrt(self.custom_medium_thresh))}×{int(np.sqrt(self.custom_medium_thresh))})")
        print(f"  Large:  area ≥ {self.custom_medium_thresh} (≥{int(np.sqrt(self.custom_medium_thresh))}×{int(np.sqrt(self.custom_medium_thresh))})")
    
    def analyze_split(self, split='train'):
        """
        分析指定数据集划分 (train/val/test)
        
        Args:
            split: 'train' or 'val' or 'test'
            
        Returns:
            dict: 统计结果
        """
        print(f"\n{'='*80}")
        print(f"🔍 分析 {split.upper()} 数据集")
        print(f"{'='*80}")
        
        # 根据YAML配置获取路径
        if split == 'train':
            img_rel_path = self.data_config.get('train', 'images/train')
        elif split == 'val':
            img_rel_path = self.data_config.get('val', 'images/val')
        elif split == 'test':
            img_rel_path = self.data_config.get('test', 'images/test')
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # 推断标签路径
        # 例如: VisDrone2019-DET-train/images/rgb -> VisDrone2019-DET-train/labels
        # 通用规则: 替换 /images/xxx 为 /labels
        label_rel_path = img_rel_path
        if '/images/' in label_rel_path:
            # 找到/images/的位置,替换后面的部分
            parts = label_rel_path.split('/images/')
            label_rel_path = parts[0] + '/labels'
        elif '\\images\\' in label_rel_path:
            parts = label_rel_path.split('\\images\\')
            label_rel_path = parts[0] + '\\labels'
        else:
            # 如果路径中没有images,尝试直接替换
            label_rel_path = label_rel_path.replace('images', 'labels')
        
        # 构建完整路径
        img_dir = self.data_root / img_rel_path
        label_dir = self.data_root / label_rel_path
        
        print(f"📂 图像目录: {img_dir}")
        print(f"📂 标签目录: {label_dir}")
        
        if not label_dir.exists():
            print(f"❌ 标签目录不存在: {label_dir}")
            return None
        
        if not img_dir.exists():
            print(f"❌ 图像目录不存在: {img_dir}")
            return None
        
        # 收集所有标签文件
        label_files = list(label_dir.glob('*.txt'))
        print(f"📄 找到 {len(label_files)} 个标签文件")
        
        if len(label_files) == 0:
            print(f"⚠️  没有找到标签文件!")
            return None
        
        # 统计数据
        stats_coco = {
            'small': [],
            'medium': [],
            'large': [],
            'areas': [],
            'widths': [],
            'heights': []
        }
        
        stats_custom = {
            'small': [],
            'medium': [],
            'large': [],
        }
        
        total_objects = 0
        total_images = 0
        images_with_objects = 0
        
        # 遍历所有标签文件
        for label_file in tqdm(label_files, desc=f"处理{split}标签"):
            total_images += 1
            
            # 获取对应的图像文件
            img_file = img_dir / label_file.with_suffix('.jpg').name
            if not img_file.exists():
                img_file = img_dir / label_file.with_suffix('.png').name
            
            if not img_file.exists():
                continue
            
            # 读取图像尺寸
            from PIL import Image
            try:
                with Image.open(img_file) as img:
                    img_w, img_h = img.size
            except:
                # 如果读取失败,假设是640×640
                img_w, img_h = self.img_size, self.img_size
            
            # 读取标签
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) == 0:
                continue
            
            images_with_objects += 1
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                # YOLO格式: class x_center y_center width height (归一化)
                cls_id = int(parts[0])
                x_center, y_center, w_norm, h_norm = map(float, parts[1:5])
                
                # 转换为像素尺寸 (resize到训练尺寸)
                # 假设训练时会resize到img_size×img_size
                w_pixel = w_norm * self.img_size
                h_pixel = h_norm * self.img_size
                area = w_pixel * h_pixel
                
                total_objects += 1
                
                # 记录原始数据
                stats_coco['areas'].append(area)
                stats_coco['widths'].append(w_pixel)
                stats_coco['heights'].append(h_pixel)
                
                # COCO标准分类
                if area < self.coco_small_thresh:
                    stats_coco['small'].append(area)
                elif area < self.coco_medium_thresh:
                    stats_coco['medium'].append(area)
                else:
                    stats_coco['large'].append(area)
                
                # 自定义阈值分类
                if area < self.custom_small_thresh:
                    stats_custom['small'].append(area)
                elif area < self.custom_medium_thresh:
                    stats_custom['medium'].append(area)
                else:
                    stats_custom['large'].append(area)
        
        # 打印统计结果
        self._print_statistics(split, stats_coco, stats_custom, total_objects, 
                              total_images, images_with_objects)
        
        return {
            'stats_coco': stats_coco,
            'stats_custom': stats_custom,
            'total_objects': total_objects,
            'total_images': total_images,
            'images_with_objects': images_with_objects
        }
    
    def _print_statistics(self, split, stats_coco, stats_custom, total_objects, 
                         total_images, images_with_objects):
        """打印统计结果"""
        print(f"\n{'='*80}")
        print(f"📊 {split.upper()} 数据集统计结果")
        print(f"{'='*80}")
        
        print(f"\n📈 总体统计:")
        print(f"  总图像数: {total_images}")
        print(f"  有目标的图像: {images_with_objects} ({100*images_with_objects/total_images:.1f}%)")
        print(f"  总目标数: {total_objects}")
        print(f"  平均每张图像目标数: {total_objects/images_with_objects:.1f}")
        
        # 面积统计
        areas = np.array(stats_coco['areas'])
        print(f"\n📐 目标面积统计 (pixels²):")
        print(f"  最小: {areas.min():.1f}")
        print(f"  最大: {areas.max():.1f}")
        print(f"  平均: {areas.mean():.1f}")
        print(f"  中位数: {np.median(areas):.1f}")
        print(f"  25%分位: {np.percentile(areas, 25):.1f}")
        print(f"  75%分位: {np.percentile(areas, 75):.1f}")
        
        # 宽高统计
        widths = np.array(stats_coco['widths'])
        heights = np.array(stats_coco['heights'])
        print(f"\n📏 目标尺寸统计 (pixels):")
        print(f"  宽度: min={widths.min():.1f}, max={widths.max():.1f}, mean={widths.mean():.1f}")
        print(f"  高度: min={heights.min():.1f}, max={heights.max():.1f}, mean={heights.mean():.1f}")
        print(f"  长宽比: min={(widths/heights).min():.2f}, max={(widths/heights).max():.2f}, mean={(widths/heights).mean():.2f}")
        
        # COCO标准统计
        n_small_coco = len(stats_coco['small'])
        n_medium_coco = len(stats_coco['medium'])
        n_large_coco = len(stats_coco['large'])
        
        print(f"\n📊 COCO标准 (32²/96²) 尺寸分布:")
        print(f"  Small  (<32×32):   {n_small_coco:>6} ({100*n_small_coco/total_objects:>5.1f}%)")
        print(f"  Medium (32~96):    {n_medium_coco:>6} ({100*n_medium_coco/total_objects:>5.1f}%)")
        print(f"  Large  (≥96×96):   {n_large_coco:>6} ({100*n_large_coco/total_objects:>5.1f}%)")
        print(f"  总计:              {total_objects:>6} (100.0%)")
        
        # 自定义阈值统计
        n_small_custom = len(stats_custom['small'])
        n_medium_custom = len(stats_custom['medium'])
        n_large_custom = len(stats_custom['large'])
        
        print(f"\n📊 VisDrone优化 (32²/64²) 尺寸分布:")
        print(f"  Small  (<32×32):   {n_small_custom:>6} ({100*n_small_custom/total_objects:>5.1f}%)")
        print(f"  Medium (32~64):    {n_medium_custom:>6} ({100*n_medium_custom/total_objects:>5.1f}%)")
        print(f"  Large  (≥64×64):   {n_large_custom:>6} ({100*n_large_custom/total_objects:>5.1f}%)")
        print(f"  总计:              {total_objects:>6} (100.0%)")
        
        # 对比分析
        print(f"\n📈 阈值对比分析:")
        print(f"  Large目标数量:")
        print(f"    COCO标准 (96²): {n_large_coco:>6} ({100*n_large_coco/total_objects:>5.1f}%)")
        print(f"    VisDrone (64²): {n_large_custom:>6} ({100*n_large_custom/total_objects:>5.1f}%)")
        print(f"    增加倍数: {n_large_custom/n_large_coco if n_large_coco > 0 else 'N/A'}×")
        
        # 警告信息
        if n_large_coco < 100:
            print(f"\n⚠️  警告: COCO标准下Large目标仅{n_large_coco}个(<100), 统计不可靠!")
        if n_large_coco / total_objects < 0.01:
            print(f"⚠️  警告: Large目标占比{100*n_large_coco/total_objects:.1f}%(<1%), 建议调整阈值!")
    
    def plot_distribution(self, train_result, val_result, save_path='distribution_analysis.png'):
        """
        绘制数据分布对比图
        
        Args:
            train_result: 训练集统计结果
            val_result: 验证集统计结果
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('VisDrone Dataset Size Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 提取数据
        train_coco = train_result['stats_coco']
        train_custom = train_result['stats_custom']
        val_coco = val_result['stats_coco']
        val_custom = val_result['stats_custom']
        
        # 1. COCO标准 - 训练集
        ax = axes[0, 0]
        sizes_coco_train = [len(train_coco['small']), len(train_coco['medium']), len(train_coco['large'])]
        labels = ['Small\n(<32×32)', 'Medium\n(32~96)', 'Large\n(≥96×96)']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        ax.pie(sizes_coco_train, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Train Set - COCO Standard', fontsize=12, fontweight='bold')
        
        # 2. COCO标准 - 验证集
        ax = axes[0, 1]
        sizes_coco_val = [len(val_coco['small']), len(val_coco['medium']), len(val_coco['large'])]
        ax.pie(sizes_coco_val, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Val Set - COCO Standard', fontsize=12, fontweight='bold')
        
        # 3. COCO vs VisDrone对比
        ax = axes[0, 2]
        x = np.arange(3)
        width = 0.35
        train_coco_nums = [len(train_coco['small']), len(train_coco['medium']), len(train_coco['large'])]
        train_custom_nums = [len(train_custom['small']), len(train_custom['medium']), len(train_custom['large'])]
        ax.bar(x - width/2, train_coco_nums, width, label='COCO (96²)', color='#FF6B6B', alpha=0.8)
        ax.bar(x + width/2, train_custom_nums, width, label='VisDrone (64²)', color='#45B7D1', alpha=0.8)
        ax.set_xlabel('Object Size')
        ax.set_ylabel('Number of Objects')
        ax.set_title('Train Set - Threshold Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Small', 'Medium', 'Large'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 4. VisDrone优化 - 训练集
        ax = axes[1, 0]
        sizes_custom_train = [len(train_custom['small']), len(train_custom['medium']), len(train_custom['large'])]
        labels_custom = ['Small\n(<32×32)', 'Medium\n(32~64)', 'Large\n(≥64×64)']
        ax.pie(sizes_custom_train, labels=labels_custom, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Train Set - VisDrone Optimized', fontsize=12, fontweight='bold')
        
        # 5. VisDrone优化 - 验证集
        ax = axes[1, 1]
        sizes_custom_val = [len(val_custom['small']), len(val_custom['medium']), len(val_custom['large'])]
        ax.pie(sizes_custom_val, labels=labels_custom, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Val Set - VisDrone Optimized', fontsize=12, fontweight='bold')
        
        # 6. 面积分布直方图
        ax = axes[1, 2]
        all_areas_train = np.array(train_coco['areas'])
        all_areas_val = np.array(val_coco['areas'])
        bins = np.logspace(0, 5, 50)  # 对数刻度
        ax.hist(all_areas_train, bins=bins, alpha=0.5, label='Train', color='#FF6B6B', edgecolor='black')
        ax.hist(all_areas_val, bins=bins, alpha=0.5, label='Val', color='#45B7D1', edgecolor='black')
        ax.axvline(x=1024, color='red', linestyle='--', label='Small/Medium (32²)', linewidth=2)
        ax.axvline(x=4096, color='orange', linestyle='--', label='VisDrone (64²)', linewidth=2)
        ax.axvline(x=9216, color='green', linestyle='--', label='COCO (96²)', linewidth=2)
        ax.set_xscale('log')
        ax.set_xlabel('Object Area (pixels², log scale)')
        ax.set_ylabel('Frequency')
        ax.set_title('Area Distribution Histogram', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 分布图已保存到: {save_path}")
        plt.close()
    
    def generate_report(self, train_result, val_result, save_path='dataset_report.txt'):
        """
        生成详细的统计报告
        
        Args:
            train_result: 训练集统计结果
            val_result: 验证集统计结果
            save_path: 保存路径
        """
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("VisDrone Dataset Size Distribution Report\n")
            f.write("="*80 + "\n\n")
            
            # 训练集统计
            f.write("TRAINING SET\n")
            f.write("-"*80 + "\n")
            self._write_split_stats(f, train_result, 'train')
            
            f.write("\n" + "="*80 + "\n\n")
            
            # 验证集统计
            f.write("VALIDATION SET\n")
            f.write("-"*80 + "\n")
            self._write_split_stats(f, val_result, 'val')
            
            f.write("\n" + "="*80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*80 + "\n\n")
            
            # 分析Large目标占比
            train_large_coco = len(train_result['stats_coco']['large'])
            val_large_coco = len(val_result['stats_coco']['large'])
            train_total = train_result['total_objects']
            val_total = val_result['total_objects']
            
            train_large_pct = 100 * train_large_coco / train_total
            val_large_pct = 100 * val_large_coco / val_total
            
            if val_large_pct < 1.0:
                f.write(f"⚠️  CRITICAL: Val set has only {val_large_coco} large objects ({val_large_pct:.1f}%)\n")
                f.write(f"   This is statistically unreliable for evaluation!\n\n")
                f.write(f"💡 RECOMMENDATION: Use VisDrone optimized threshold (64²) instead\n")
                f.write(f"   - This will increase large objects to ~{len(val_result['stats_custom']['large'])} ")
                f.write(f"({100*len(val_result['stats_custom']['large'])/val_total:.1f}%)\n")
                f.write(f"   - More reliable for statistical evaluation\n")
                f.write(f"   - Better reflects UAV detection scenarios\n\n")
            
            if train_large_pct < 5.0:
                f.write(f"⚠️  WARNING: Train set has only {train_large_pct:.1f}% large objects\n")
                f.write(f"   Model may not learn large object detection well\n\n")
                f.write(f"💡 RECOMMENDATION: Consider data augmentation for large objects\n")
                f.write(f"   - Random crop & zoom\n")
                f.write(f"   - Copy-paste augmentation\n\n")
        
        print(f"\n📄 详细报告已保存到: {save_path}")
    
    def _write_split_stats(self, f, result, split):
        """写入单个数据集划分的统计信息"""
        stats_coco = result['stats_coco']
        stats_custom = result['stats_custom']
        total = result['total_objects']
        
        f.write(f"Total images: {result['total_images']}\n")
        f.write(f"Images with objects: {result['images_with_objects']}\n")
        f.write(f"Total objects: {total}\n\n")
        
        f.write("COCO Standard (32²/96²):\n")
        f.write(f"  Small  (<32×32):  {len(stats_coco['small']):>6} ({100*len(stats_coco['small'])/total:>5.1f}%)\n")
        f.write(f"  Medium (32~96):   {len(stats_coco['medium']):>6} ({100*len(stats_coco['medium'])/total:>5.1f}%)\n")
        f.write(f"  Large  (≥96×96):  {len(stats_coco['large']):>6} ({100*len(stats_coco['large'])/total:>5.1f}%)\n\n")
        
        f.write("VisDrone Optimized (32²/64²):\n")
        f.write(f"  Small  (<32×32):  {len(stats_custom['small']):>6} ({100*len(stats_custom['small'])/total:>5.1f}%)\n")
        f.write(f"  Medium (32~64):   {len(stats_custom['medium']):>6} ({100*len(stats_custom['medium'])/total:>5.1f}%)\n")
        f.write(f"  Large  (≥64×64):  {len(stats_custom['large']):>6} ({100*len(stats_custom['large'])/total:>5.1f}%)\n\n")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='分析VisDrone数据集的目标尺寸分布')
    parser.add_argument('--data', type=str, default='data/visdrone-rgbd.yaml',
                       help='数据集YAML配置文件路径')
    parser.add_argument('--img-size', type=int, default=640,
                       help='训练/验证图像尺寸')
    parser.add_argument('--save-dir', type=str, default='.',
                       help='结果保存目录')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = DatasetAnalyzer(args.data, args.img_size)
    
    # 分析训练集
    print("\n" + "="*80)
    print("开始分析数据集...")
    print("="*80)
    
    train_result = analyzer.analyze_split('train')
    val_result = analyzer.analyze_split('val')
    
    if train_result and val_result:
        # 绘制分布图
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        plot_path = save_dir / 'distribution_analysis.png'
        analyzer.plot_distribution(train_result, val_result, plot_path)
        
        # 生成报告
        report_path = save_dir / 'dataset_report.txt'
        analyzer.generate_report(train_result, val_result, report_path)
        
        print("\n" + "="*80)
        print("✅ 分析完成!")
        print("="*80)
        print(f"📊 可视化结果: {plot_path}")
        print(f"📄 详细报告: {report_path}")
        print("\n💡 建议:")
        
        val_large_coco = len(val_result['stats_coco']['large'])
        val_total = val_result['total_objects']
        val_large_pct = 100 * val_large_coco / val_total
        
        if val_large_pct < 1.0:
            print(f"⚠️  验证集Large目标仅{val_large_coco}个({val_large_pct:.1f}%), 统计不可靠!")
            print(f"💡 建议使用VisDrone优化阈值(64²), Large目标将增加到{len(val_result['stats_custom']['large'])}个")
            print(f"   ({100*len(val_result['stats_custom']['large'])/val_total:.1f}%)")
    else:
        print("\n❌ 数据集分析失败, 请检查路径配置!")


if __name__ == '__main__':
    main()
