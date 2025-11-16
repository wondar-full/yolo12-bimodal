"""
UAVDT数据集标注转换为YOLO格式 (带类别映射)

🚨 CRITICAL FIX: 正确映射UAVDT类别到VisDrone类别空间

UAVDT原始类别 (3类):
  1: car    # 小汽车
  2: truck  # 卡车
  3: bus    # 公交车

VisDrone类别空间 (10类):
  0: pedestrian
  1: people
  2: bicycle
  3: car        ← UAVDT的car应该映射到这里
  4: van
  5: truck      ← UAVDT的truck应该映射到这里
  6: tricycle
  7: awning-tricycle
  8: bus        ← UAVDT的bus应该映射到这里
  9: motor

类别映射表:
  UAVDT 1 (car)   → VisDrone 3 (car)
  UAVDT 2 (truck) → VisDrone 5 (truck)
  UAVDT 3 (bus)   → VisDrone 8 (bus)
"""

import os
from pathlib import Path
import argparse
from tqdm import tqdm
from collections import Counter


# ⚡ 关键修复: UAVDT → VisDrone 类别映射表
UAVDT_TO_VISDRONE = {
    1: 3,  # car → car
    2: 5,  # truck → truck
    3: 8,  # bus → bus
}

CATEGORY_NAMES = {
    1: "car",
    2: "truck",
    3: "bus",
}


def convert_bbox(size, box):
    """
    将边界框转换为YOLO格式
    
    Input: <bbox_left>, <bbox_top>, <bbox_width>, <bbox_height> (绝对像素值)
    Output: <x_center>, <y_center>, <width>, <height> (归一化到[0,1])
    """
    dw = 1. / size[0]  # width normalization factor
    dh = 1. / size[1]  # height normalization factor
    
    x_center = (box[0] + box[2] / 2.0) * dw
    y_center = (box[1] + box[3] / 2.0) * dh
    width = box[2] * dw
    height = box[3] * dh
    
    # 确保在[0, 1]范围内
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return (x_center, y_center, width, height)


def convert_uavdt_annotation(anno_file, img_size, output_file, stats):
    """
    转换单个UAVDT标注文件到YOLO格式
    
    Args:
        anno_file: UAVDT原始标注文件路径 (.txt)
        img_size: 图像尺寸 (width, height)
        output_file: 输出YOLO标注文件路径
        stats: 统计信息字典 (用于记录转换情况)
    """
    if not anno_file.exists():
        stats['missing_anno'] += 1
        return
    
    with open(anno_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    yolo_annotations = []
    
    for line_idx, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        parts = line.split(',')
        
        # UAVDT标注格式: bbox_left, bbox_top, bbox_width, bbox_height, score, category, ...
        if len(parts) < 6:
            stats['invalid_format'] += 1
            continue
        
        try:
            bbox_left = float(parts[0])
            bbox_top = float(parts[1])
            bbox_width = float(parts[2])
            bbox_height = float(parts[3])
            # score = float(parts[4])  # 通常为1.0,忽略
            uavdt_category = int(parts[5])  # UAVDT原始类别 (1/2/3)
        except (ValueError, IndexError) as e:
            stats['parse_error'] += 1
            continue
        
        # 过滤无效边界框
        if bbox_width <= 0 or bbox_height <= 0:
            stats['invalid_bbox'] += 1
            continue
        
        if bbox_left < 0 or bbox_top < 0:
            stats['invalid_bbox'] += 1
            continue
        
        # ⚡ 关键修复: 使用映射表转换类别
        if uavdt_category not in UAVDT_TO_VISDRONE:
            stats['unknown_category'] += 1
            stats['unknown_category_ids'].add(uavdt_category)
            continue
        
        visdrone_category = UAVDT_TO_VISDRONE[uavdt_category]
        
        # 记录类别统计
        stats['uavdt_categories'][uavdt_category] += 1
        stats['visdrone_categories'][visdrone_category] += 1
        
        # 转换为YOLO格式
        bbox = (bbox_left, bbox_top, bbox_width, bbox_height)
        yolo_bbox = convert_bbox(img_size, bbox)
        
        # 验证转换后的值
        if any(v < 0 or v > 1 for v in yolo_bbox):
            stats['out_of_bounds'] += 1
            continue
        
        yolo_annotations.append(
            f"{visdrone_category} {' '.join(f'{v:.6f}' for v in yolo_bbox)}\n"
        )
        stats['converted_objects'] += 1
    
    # 保存YOLO标注
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(yolo_annotations)
    
    if yolo_annotations:
        stats['converted_files'] += 1
    else:
        stats['empty_files'] += 1


def convert_dataset(uavdt_root, output_root, splits=['train', 'test']):
    """
    转换整个UAVDT数据集
    
    Args:
        uavdt_root: UAVDT数据集根目录
        output_root: 输出YOLO格式数据集根目录
        splits: 要转换的数据集划分 ['train', 'test']
    """
    uavdt_root = Path(uavdt_root)
    output_root = Path(output_root)
    
    print("=" * 80)
    print("UAVDT → YOLO 格式转换 (带VisDrone类别映射)")
    print("=" * 80)
    print(f"输入路径: {uavdt_root}")
    print(f"输出路径: {output_root}")
    print()
    print("类别映射表:")
    for uavdt_id, visdrone_id in UAVDT_TO_VISDRONE.items():
        print(f"  UAVDT {uavdt_id} ({CATEGORY_NAMES[uavdt_id]:<8}) → VisDrone {visdrone_id}")
    print("=" * 80)
    print()
    
    for split in splits:
        print(f"\n{'='*80}")
        print(f"处理 {split.upper()} 集")
        print(f"{'='*80}")
        
        # UAVDT目录结构:
        # UAVDT/
        #   M0101/
        #     Annotations/  ← 标注文件
        #     Imgs/         ← 图像文件
        #   M0102/
        #   ...
        
        # 初始化统计信息
        stats = {
            'total_sequences': 0,
            'total_files': 0,
            'converted_files': 0,
            'empty_files': 0,
            'missing_anno': 0,
            'missing_image': 0,
            'invalid_format': 0,
            'invalid_bbox': 0,
            'parse_error': 0,
            'unknown_category': 0,
            'out_of_bounds': 0,
            'converted_objects': 0,
            'uavdt_categories': Counter(),
            'visdrone_categories': Counter(),
            'unknown_category_ids': set(),
        }
        
        # 查找所有序列目录 (M0101, M0102, ...)
        sequence_dirs = sorted([d for d in uavdt_root.glob("M*") if d.is_dir()])
        
        if not sequence_dirs:
            print(f"❌ 错误: 在 {uavdt_root} 中没有找到序列目录 (M0101, M0102, ...)")
            print(f"请检查UAVDT数据集路径是否正确")
            continue
        
        stats['total_sequences'] = len(sequence_dirs)
        print(f"找到 {len(sequence_dirs)} 个序列目录")
        
        for seq_dir in tqdm(sequence_dirs, desc=f"转换{split}集"):
            anno_dir = seq_dir / "Annotations"
            img_dir = seq_dir / "Imgs"
            
            if not anno_dir.exists():
                print(f"⚠️  警告: 序列 {seq_dir.name} 没有Annotations目录")
                continue
            
            if not img_dir.exists():
                print(f"⚠️  警告: 序列 {seq_dir.name} 没有Imgs目录")
                continue
            
            # 获取所有标注文件
            anno_files = sorted(anno_dir.glob("*.txt"))
            
            for anno_file in anno_files:
                stats['total_files'] += 1
                
                # 查找对应的图像文件
                img_name = anno_file.stem + ".jpg"
                img_file = img_dir / img_name
                
                if not img_file.exists():
                    stats['missing_image'] += 1
                    continue
                
                # 获取图像尺寸
                try:
                    from PIL import Image
                    img = Image.open(img_file)
                    img_size = img.size  # (width, height)
                except Exception as e:
                    print(f"❌ 无法读取图像 {img_file}: {e}")
                    stats['missing_image'] += 1
                    continue
                
                # 转换标注
                output_label_dir = output_root / split / "labels" / "rgb" / seq_dir.name
                output_label_file = output_label_dir / anno_file.name
                
                convert_uavdt_annotation(anno_file, img_size, output_label_file, stats)
                
                # 复制图像到输出目录 (如果需要)
                output_img_dir = output_root / split / "images" / "rgb" / seq_dir.name
                output_img_dir.mkdir(parents=True, exist_ok=True)
                output_img_file = output_img_dir / img_name
                
                if not output_img_file.exists():
                    import shutil
                    shutil.copy2(img_file, output_img_file)
        
        # 打印统计信息
        print(f"\n{'='*80}")
        print(f"{split.upper()}集 转换统计")
        print(f"{'='*80}")
        print(f"序列目录数:       {stats['total_sequences']}")
        print(f"标注文件总数:     {stats['total_files']}")
        print(f"成功转换文件:     {stats['converted_files']}")
        print(f"空文件(无对象):   {stats['empty_files']}")
        print(f"成功转换对象:     {stats['converted_objects']}")
        print()
        print("错误统计:")
        print(f"  缺失标注文件:   {stats['missing_anno']}")
        print(f"  缺失图像文件:   {stats['missing_image']}")
        print(f"  标注格式错误:   {stats['invalid_format']}")
        print(f"  无效边界框:     {stats['invalid_bbox']}")
        print(f"  解析错误:       {stats['parse_error']}")
        print(f"  未知类别:       {stats['unknown_category']}")
        print(f"  坐标越界:       {stats['out_of_bounds']}")
        print()
        print("UAVDT原始类别分布:")
        for cat_id in sorted(stats['uavdt_categories'].keys()):
            count = stats['uavdt_categories'][cat_id]
            cat_name = CATEGORY_NAMES.get(cat_id, "unknown")
            print(f"  类别 {cat_id} ({cat_name:<8}): {count:>8} 个对象")
        print()
        print("转换后VisDrone类别分布:")
        for cat_id in sorted(stats['visdrone_categories'].keys()):
            count = stats['visdrone_categories'][cat_id]
            print(f"  类别 {cat_id}: {count:>8} 个对象")
        
        if stats['unknown_category_ids']:
            print()
            print(f"⚠️  发现未知类别ID: {sorted(stats['unknown_category_ids'])}")
            print("请检查UAVDT数据集或更新类别映射表")
        
        print(f"{'='*80}")
        print(f"✅ {split}集转换完成!")
        print(f"   标签保存在: {output_root / split / 'labels' / 'rgb'}")
        print(f"   图像保存在: {output_root / split / 'images' / 'rgb'}")
        print(f"{'='*80}\n")


def verify_conversion(output_root, splits=['train', 'test']):
    """
    验证转换结果
    """
    print("\n" + "="*80)
    print("验证转换结果")
    print("="*80)
    
    output_root = Path(output_root)
    
    for split in splits:
        label_dir = output_root / split / "labels" / "rgb"
        
        if not label_dir.exists():
            print(f"❌ {split}集标签目录不存在: {label_dir}")
            continue
        
        # 统计所有标签文件中的类别
        all_labels = list(label_dir.rglob("*.txt"))
        
        if not all_labels:
            print(f"❌ {split}集没有找到标签文件")
            continue
        
        category_counts = Counter()
        total_objects = 0
        
        for label_file in all_labels:
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cat_id = int(parts[0])
                        category_counts[cat_id] += 1
                        total_objects += 1
        
        print(f"\n{split.upper()}集验证结果:")
        print(f"  标签文件数: {len(all_labels)}")
        print(f"  总对象数: {total_objects}")
        print(f"  类别分布:")
        
        for cat_id in sorted(category_counts.keys()):
            count = category_counts[cat_id]
            percentage = (count / total_objects) * 100 if total_objects > 0 else 0
            print(f"    类别 {cat_id}: {count:>8} ({percentage:>5.2f}%)")
        
        # 检查是否包含预期的类别 (3, 5, 8)
        expected_categories = set(UAVDT_TO_VISDRONE.values())
        found_categories = set(category_counts.keys())
        
        if found_categories == expected_categories:
            print(f"  ✅ 类别检查通过! 找到预期的类别: {sorted(expected_categories)}")
        else:
            missing = expected_categories - found_categories
            unexpected = found_categories - expected_categories
            if missing:
                print(f"  ⚠️  缺失预期类别: {sorted(missing)}")
            if unexpected:
                print(f"  ⚠️  发现非预期类别: {sorted(unexpected)}")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="转换UAVDT数据集到YOLO格式 (带VisDrone类别映射)"
    )
    parser.add_argument(
        '--uavdt_root',
        type=str,
        required=True,
        help='UAVDT数据集根目录 (包含M0101, M0102等序列目录)'
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default='./UAVDT_YOLO',
        help='输出YOLO格式数据集根目录'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train'],
        help='要转换的数据集划分 (默认: train)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='转换后验证结果'
    )
    
    args = parser.parse_args()
    
    # 转换数据集
    convert_dataset(args.uavdt_root, args.output_root, args.splits)
    
    # 验证转换结果
    if args.verify:
        verify_conversion(args.output_root, args.splits)
    
    print("\n" + "="*80)
    print("✅ 所有转换任务完成!")
    print("="*80)
    print("\n下一步:")
    print("1. 验证类别分布是否正确 (应该看到类别 3, 5, 8)")
    print("2. 生成深度图 (如果需要)")
    print("3. 更新 data/visdrone_uavdt_joint.yaml 配置")
    print("4. 开始联合训练!")
    print()


if __name__ == "__main__":
    main()
