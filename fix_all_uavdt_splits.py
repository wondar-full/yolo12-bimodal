"""
UAVDT完整修复脚本 - 修复所有分割(train/val/test)
解决exp_joint_v16性能未提升的问题

使用方法:
    python fix_all_uavdt_splits.py --dataset_root /path/to/UAVDT_YOLO
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import shutil


def fix_labels_in_directory(label_dir, backup=True):
    """
    修复单个目录中的所有标签文件
    
    Args:
        label_dir: 标签目录路径
        backup: 是否备份
    
    Returns:
        stats: 统计信息字典
    """
    label_dir = Path(label_dir)
    
    if not label_dir.exists():
        print(f"  ⚠️  目录不存在,跳过: {label_dir}")
        return None
    
    # 查找标签文件
    label_files = list(label_dir.rglob("*.txt"))
    
    if not label_files:
        print(f"  ⚠️  没有找到标签文件,跳过: {label_dir}")
        return None
    
    print(f"  📁 找到 {len(label_files)} 个标签文件")
    
    # 备份
    if backup:
        backup_dir = label_dir.parent / f"{label_dir.name}_backup_{Path.cwd().stat().st_mtime}"
        if not backup_dir.exists():
            print(f"  📦 备份到: {backup_dir}")
            shutil.copytree(label_dir, backup_dir)
    
    # 统计信息
    stats = {
        'total_files': len(label_files),
        'modified_files': 0,
        'total_objects': 0,
        'modified_objects': 0,
        'category_before': Counter(),
        'category_after': Counter(),
        'errors': [],
    }
    
    # 处理每个文件
    for label_file in tqdm(label_files, desc="  修复标签", leave=False):
        try:
            # 读取
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            modified_lines = []
            file_modified = False
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    stats['errors'].append(f"{label_file}: 格式错误")
                    modified_lines.append(line + '\n')
                    continue
                
                try:
                    old_category = int(parts[0])
                    stats['category_before'][old_category] += 1
                    stats['total_objects'] += 1
                    
                    # 类别ID减1
                    new_category = old_category - 1
                    
                    # 验证范围
                    if new_category < 0 or new_category > 9:
                        stats['errors'].append(
                            f"{label_file}: ID超范围 {old_category}→{new_category}"
                        )
                        modified_lines.append(line + '\n')
                        continue
                    
                    # 构造新行
                    parts[0] = str(new_category)
                    new_line = ' '.join(parts) + '\n'
                    modified_lines.append(new_line)
                    
                    stats['category_after'][new_category] += 1
                    stats['modified_objects'] += 1
                    
                    if old_category != new_category:
                        file_modified = True
                
                except ValueError:
                    stats['errors'].append(f"{label_file}: 无法解析类别ID")
                    modified_lines.append(line + '\n')
            
            # 写回
            if file_modified:
                with open(label_file, 'w', encoding='utf-8') as f:
                    f.writelines(modified_lines)
                stats['modified_files'] += 1
        
        except Exception as e:
            stats['errors'].append(f"{label_file}: {e}")
    
    return stats


def delete_cache_files(dataset_root):
    """删除数据集中的所有缓存文件"""
    dataset_root = Path(dataset_root)
    cache_files = list(dataset_root.rglob("*.cache"))
    
    if not cache_files:
        print("\n  ℹ️  没有找到缓存文件")
        return 0
    
    print(f"\n  🗑️  找到 {len(cache_files)} 个缓存文件")
    
    deleted = 0
    for cache_file in cache_files:
        try:
            cache_file.unlink()
            deleted += 1
            print(f"    ✅ 删除: {cache_file.relative_to(dataset_root)}")
        except Exception as e:
            print(f"    ❌ 删除失败 {cache_file}: {e}")
    
    return deleted


def verify_labels(label_dir, expected_classes=[3, 5, 8]):
    """验证标签修复结果"""
    label_dir = Path(label_dir)
    
    if not label_dir.exists():
        return False, "目录不存在"
    
    # 统计类别分布
    class_counts = Counter()
    
    for label_file in label_dir.rglob("*.txt"):
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
                    except ValueError:
                        pass
    
    # 检查是否只包含预期类别
    actual_classes = set(class_counts.keys())
    expected_set = set(expected_classes)
    
    if actual_classes == expected_set:
        return True, class_counts
    else:
        return False, {
            'actual': actual_classes,
            'expected': expected_set,
            'counts': class_counts
        }


def print_stats(split_name, stats):
    """打印统计信息"""
    if stats is None:
        return
    
    print(f"\n  📊 {split_name} 修复统计:")
    print(f"    总文件数: {stats['total_files']}")
    print(f"    修改文件数: {stats['modified_files']}")
    print(f"    总对象数: {stats['total_objects']}")
    print(f"    修改对象数: {stats['modified_objects']}")
    
    print(f"\n    修复前类别分布:")
    for cat_id in sorted(stats['category_before'].keys()):
        count = stats['category_before'][cat_id]
        print(f"      类别 {cat_id}: {count:>8} 个")
    
    print(f"\n    修复后类别分布:")
    for cat_id in sorted(stats['category_after'].keys()):
        count = stats['category_after'][cat_id]
        print(f"      类别 {cat_id}: {count:>8} 个")
    
    if stats['errors']:
        print(f"\n    ⚠️  遇到 {len(stats['errors'])} 个错误 (显示前5个):")
        for error in stats['errors'][:5]:
            print(f"      - {error}")


def main():
    parser = argparse.ArgumentParser(
        description="修复UAVDT数据集所有分割的类别ID (4→3, 6→5, 9→8)"
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        required=True,
        help='UAVDT数据集根目录 (例如: /data2/.../UAVDT_YOLO)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='不备份原始文件 (谨慎使用!)'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='要修复的分割 (默认: train val test)'
    )
    parser.add_argument(
        '--no-delete-cache',
        action='store_true',
        help='不删除缓存文件'
    )
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    
    if not dataset_root.exists():
        print(f"❌ 错误: 数据集根目录不存在 {dataset_root}")
        return 1
    
    print("="*80)
    print("UAVDT数据集完整修复工具")
    print("="*80)
    print(f"数据集根目录: {dataset_root}")
    print(f"备份原始文件: {'否' if args.no_backup else '是'}")
    print(f"要修复的分割: {', '.join(args.splits)}")
    print("="*80)
    print()
    
    # 确认
    confirm = input("⚠️  此操作将修改标签文件,是否继续? (yes/no): ")
    if confirm.lower() not in ['yes', 'y']:
        print("❌ 操作已取消")
        return 0
    
    print()
    
    # 修复每个分割
    all_stats = {}
    
    for split in args.splits:
        print(f"{'='*80}")
        print(f"修复 {split.upper()} 集")
        print(f"{'='*80}")
        
        # RGB标签路径 (根据UAVDT目录结构)
        label_dir = dataset_root / split / 'labels' / 'rgb'
        
        if not label_dir.exists():
            # 尝试其他可能的路径
            alt_paths = [
                dataset_root / split / 'labels',
                dataset_root / split / 'rgb' / 'labels',
            ]
            
            for alt_path in alt_paths:
                if alt_path.exists():
                    label_dir = alt_path
                    break
        
        # 修复
        stats = fix_labels_in_directory(label_dir, backup=not args.no_backup)
        
        if stats:
            all_stats[split] = stats
            print_stats(split, stats)
            
            # 验证修复结果
            print(f"\n  🔍 验证修复结果...")
            success, result = verify_labels(label_dir)
            
            if success:
                print(f"  ✅ 验证通过! 类别分布:")
                for class_id, count in sorted(result.items()):
                    class_name = {3: 'car', 5: 'truck', 8: 'bus'}.get(class_id, 'unknown')
                    print(f"    类别 {class_id} ({class_name}): {count} 个")
            else:
                print(f"  ❌ 验证失败!")
                print(f"    预期类别: {result['expected']}")
                print(f"    实际类别: {result['actual']}")
                print(f"    分布: {result['counts']}")
        
        print()
    
    # 删除缓存
    if not args.no_delete_cache:
        print(f"{'='*80}")
        print("删除缓存文件")
        print(f"{'='*80}")
        
        deleted = delete_cache_files(dataset_root)
        print(f"  ✅ 共删除 {deleted} 个缓存文件")
        print()
    
    # 总结
    print("="*80)
    print("✅ 修复完成!")
    print("="*80)
    
    total_modified = sum(s['modified_objects'] for s in all_stats.values())
    print(f"\n总计修改了 {total_modified} 个对象标签")
    
    print("\n下一步:")
    print("  1. 验证所有分割的类别分布:")
    for split in args.splits:
        print(f"     cd {dataset_root}/{split}/labels/rgb")
        print(f"     find . -name '*.txt' -exec cat {{}} \\; | awk '{{print $1}}' | sort | uniq -c")
    
    print("\n  2. 删除VisDrone数据集的缓存 (如果使用联合训练):")
    print("     find /path/to/VisDrone -name '*.cache' -delete")
    
    print("\n  3. 重新训练 (禁用缓存):")
    print("     CUDA_VISIBLE_DEVICES=7 python train_depth.py \\")
    print("         --model ultralytics/cfg/models/12/yolo12n-rgbd-v1.yaml \\")
    print("         --weights yolo12n.pt \\")
    print("         --data data/visdrone_uavdt_joint.yaml \\")
    print("         --cache False \\")
    print("         --epochs 300 \\")
    print("         --batch 16 \\")
    print("         --name exp_joint_v17_all_splits_fixed")
    
    print("\n  4. 监控训练日志:")
    print("     tail -f runs/train/exp_joint_v17_all_splits_fixed/train.log | grep -i instance")
    
    print("\n  5. 预期结果:")
    print("     - 训练实例数: ~800k")
    print("     - Epoch 50 mAP: >30%")
    print("     - Epoch 150+ mAP: 40-45%")
    
    print("\n" + "="*80)
    
    return 0


if __name__ == "__main__":
    exit(main())
