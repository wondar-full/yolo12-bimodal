"""
UAVDT标签类别ID修复脚本 - 快速修正方案

当前问题:
  UAVDT标签的类别ID为 4, 6, 9
  应该是: 3, 5, 8

修复方案:
  批量修改所有标签文件,将类别ID减1
  4 → 3 (car)
  6 → 5 (truck)
  9 → 8 (bus)
"""

import os
from pathlib import Path
from tqdm import tqdm
from collections import Counter


def fix_uavdt_labels(label_dir, backup=True):
    """
    修复UAVDT标签的类别ID (全部减1)
    
    Args:
        label_dir: UAVDT标签目录
        backup: 是否备份原始文件
    """
    label_dir = Path(label_dir)
    
    if not label_dir.exists():
        print(f"❌ 错误: 目录不存在 {label_dir}")
        return
    
    # 查找所有标签文件
    label_files = list(label_dir.rglob("*.txt"))
    
    if not label_files:
        print(f"❌ 错误: 在 {label_dir} 中没有找到标签文件")
        return
    
    print(f"找到 {len(label_files)} 个标签文件")
    
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
    
    # 备份 (如果需要)
    if backup:
        backup_dir = label_dir.parent / f"{label_dir.name}_backup_before_fix"
        if not backup_dir.exists():
            print(f"📦 备份原始标签到: {backup_dir}")
            import shutil
            shutil.copytree(label_dir, backup_dir)
        else:
            print(f"⚠️  备份目录已存在: {backup_dir}")
    
    # 处理每个标签文件
    print("\n🔧 开始修复类别ID...")
    for label_file in tqdm(label_files, desc="修复进度"):
        try:
            # 读取原始标签
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
                    stats['errors'].append(f"{label_file}: 格式错误 - {line}")
                    modified_lines.append(line + '\n')
                    continue
                
                # 获取原始类别ID
                try:
                    old_category = int(parts[0])
                    stats['category_before'][old_category] += 1
                    stats['total_objects'] += 1
                    
                    # ⚡ 关键修复: 类别ID减1
                    new_category = old_category - 1
                    
                    # 验证新类别ID在有效范围内 (0-9)
                    if new_category < 0 or new_category > 9:
                        stats['errors'].append(
                            f"{label_file}: 类别ID超出范围 {old_category} → {new_category}"
                        )
                        modified_lines.append(line + '\n')
                        continue
                    
                    # 构造新的标签行
                    parts[0] = str(new_category)
                    new_line = ' '.join(parts) + '\n'
                    modified_lines.append(new_line)
                    
                    stats['category_after'][new_category] += 1
                    stats['modified_objects'] += 1
                    
                    if old_category != new_category:
                        file_modified = True
                
                except ValueError:
                    stats['errors'].append(f"{label_file}: 无法解析类别ID - {line}")
                    modified_lines.append(line + '\n')
            
            # 写回修改后的标签
            if file_modified:
                with open(label_file, 'w', encoding='utf-8') as f:
                    f.writelines(modified_lines)
                stats['modified_files'] += 1
        
        except Exception as e:
            stats['errors'].append(f"{label_file}: {e}")
    
    # 打印统计结果
    print("\n" + "="*80)
    print("✅ 修复完成!")
    print("="*80)
    print(f"总文件数:         {stats['total_files']}")
    print(f"修改文件数:       {stats['modified_files']}")
    print(f"总对象数:         {stats['total_objects']}")
    print(f"修改对象数:       {stats['modified_objects']}")
    print()
    print("修复前类别分布:")
    for cat_id in sorted(stats['category_before'].keys()):
        count = stats['category_before'][cat_id]
        print(f"  类别 {cat_id}: {count:>8} 个对象")
    print()
    print("修复后类别分布:")
    for cat_id in sorted(stats['category_after'].keys()):
        count = stats['category_after'][cat_id]
        print(f"  类别 {cat_id}: {count:>8} 个对象")
    
    if stats['errors']:
        print()
        print(f"⚠️  遇到 {len(stats['errors'])} 个错误:")
        for error in stats['errors'][:10]:  # 只显示前10个
            print(f"  - {error}")
        if len(stats['errors']) > 10:
            print(f"  ... 还有 {len(stats['errors']) - 10} 个错误")
    
    print("="*80)
    print()
    print("✅ 类别ID修复成功!")
    print()
    print("预期结果:")
    print("  类别 3: ~394633 个 (car)")
    print("  类别 5: ~17491 个 (truck)")
    print("  类别 8: ~10787 个 (bus)")
    print()
    print("验证命令:")
    print(f"  cd {label_dir}")
    print("  find . -name '*.txt' -exec cat {{}} \\; | awk '{print $1}' | sort | uniq -c")
    print()
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="修复UAVDT标签的类别ID (4→3, 6→5, 9→8)"
    )
    parser.add_argument(
        '--label_dir',
        type=str,
        required=True,
        help='UAVDT标签目录 (例如: /data2/.../UAVDT_YOLO/train/labels/rgb)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='不备份原始文件 (谨慎使用!)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("UAVDT标签类别ID修复工具")
    print("="*80)
    print(f"目标目录: {args.label_dir}")
    print(f"备份原始: {'否' if args.no_backup else '是'}")
    print()
    print("修复操作:")
    print("  类别 4 → 3 (car)")
    print("  类别 6 → 5 (truck)")
    print("  类别 9 → 8 (bus)")
    print("="*80)
    print()
    
    # 确认操作
    confirm = input("确认开始修复? (yes/no): ")
    if confirm.lower() not in ['yes', 'y']:
        print("❌ 操作已取消")
        return
    
    # 执行修复
    stats = fix_uavdt_labels(args.label_dir, backup=not args.no_backup)
    
    if stats:
        print("\n🎉 所有操作完成!")
        print("\n下一步:")
        print("1. 验证类别分布是否正确 (应该看到 3, 5, 8)")
        print("2. 开始重新训练 exp_joint_v16")
        print("3. 监控训练指标,确认修复有效")


if __name__ == "__main__":
    main()
