"""
检查VisDrone和UAVDT数据集的类别映射是否一致
Critical Issue: Class ID mismatch between datasets will cause catastrophic training failure!

问题症状:
- 联合训练后mAP大幅下降 (22.27% → 19.51%)
- 模型在验证集上表现异常 (应该提升但反而下降)

可能原因:
- VisDrone和UAVDT的类别ID编码不一致
- 例如: VisDrone的car=3, 但UAVDT的car=4
- 导致模型学习到错误的类别映射
"""

import os
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np


def load_yolo_labels(label_dir, max_samples=500):
    """
    加载YOLO格式标签文件并统计类别分布
    
    Args:
        label_dir: 标签目录路径
        max_samples: 最多读取多少个文件 (避免耗时过长)
    
    Returns:
        class_counts: {class_id: count} 字典
        class_examples: {class_id: [file_paths]} 字典 (用于后续检查)
    """
    label_dir = Path(label_dir)
    label_files = list(label_dir.glob("*.txt"))
    
    if not label_files:
        print(f"⚠️  警告: {label_dir} 中没有找到标签文件!")
        return {}, {}
    
    # 随机采样 (如果文件太多)
    if len(label_files) > max_samples:
        import random
        label_files = random.sample(label_files, max_samples)
    
    class_counts = Counter()
    class_examples = defaultdict(list)
    
    for label_file in label_files:
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:  # class_id x_center y_center width height
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    
                    # 记录示例文件 (每个类别最多记录5个)
                    if len(class_examples[class_id]) < 5:
                        class_examples[class_id].append(str(label_file))
        except Exception as e:
            print(f"❌ 读取文件失败 {label_file}: {e}")
    
    return dict(class_counts), dict(class_examples)


def analyze_class_distribution(dataset_name, label_dir):
    """
    分析数据集的类别分布
    """
    print(f"\n{'='*60}")
    print(f"分析数据集: {dataset_name}")
    print(f"标签目录: {label_dir}")
    print(f"{'='*60}")
    
    class_counts, class_examples = load_yolo_labels(label_dir, max_samples=1000)
    
    if not class_counts:
        print("❌ 无法加载标签数据!")
        return None, None
    
    # 按类别ID排序
    sorted_classes = sorted(class_counts.items())
    
    print("\n类别ID分布:")
    print(f"{'类别ID':<10} {'实例数':<15} {'占比':<10} {'示例文件'}")
    print("-" * 80)
    
    total_instances = sum(class_counts.values())
    for class_id, count in sorted_classes:
        percentage = (count / total_instances) * 100
        example_file = Path(class_examples[class_id][0]).name if class_examples[class_id] else "N/A"
        print(f"{class_id:<10} {count:<15} {percentage:>6.2f}%    {example_file}")
    
    print(f"\n总实例数: {total_instances}")
    print(f"类别范围: {min(class_counts.keys())} ~ {max(class_counts.keys())}")
    print(f"类别总数: {len(class_counts)}")
    
    return class_counts, class_examples


def compare_datasets(visdrone_counts, uavdt_counts):
    """
    对比两个数据集的类别分布
    """
    print(f"\n{'='*80}")
    print("🔍 类别映射对比分析")
    print(f"{'='*80}")
    
    if visdrone_counts is None or uavdt_counts is None:
        print("❌ 无法进行对比 (其中一个数据集加载失败)")
        return
    
    all_classes = sorted(set(visdrone_counts.keys()) | set(uavdt_counts.keys()))
    
    print(f"\n{'类别ID':<10} {'VisDrone实例数':<20} {'UAVDT实例数':<20} {'状态'}")
    print("-" * 80)
    
    mismatches = []
    
    for class_id in all_classes:
        vd_count = visdrone_counts.get(class_id, 0)
        ua_count = uavdt_counts.get(class_id, 0)
        
        # 判断是否有异常
        status = "✅ 正常"
        if vd_count == 0:
            status = "⚠️  VisDrone缺失"
            mismatches.append((class_id, "VisDrone缺失该类别"))
        elif ua_count == 0:
            status = "⚠️  UAVDT缺失"
            mismatches.append((class_id, "UAVDT缺失该类别"))
        
        print(f"{class_id:<10} {vd_count:<20} {ua_count:<20} {status}")
    
    # 检查类别数量是否一致
    print("\n" + "="*80)
    print("📊 统计摘要:")
    print(f"  VisDrone类别数: {len(visdrone_counts)}")
    print(f"  UAVDT类别数: {len(uavdt_counts)}")
    print(f"  共同类别数: {len(set(visdrone_counts.keys()) & set(uavdt_counts.keys()))}")
    print(f"  类别不匹配数: {len(mismatches)}")
    
    if mismatches:
        print("\n🚨 **严重问题**: 检测到类别不匹配!")
        print("这会导致联合训练失败的主要原因:")
        for class_id, reason in mismatches:
            print(f"  - 类别 {class_id}: {reason}")
        print("\n可能的解决方案:")
        print("  1. 检查UAVDT标签转换脚本是否正确映射了类别ID")
        print("  2. 验证VisDrone和UAVDT的类别定义是否一致")
        print("  3. 如果类别定义不同,需要创建类别映射表进行转换")
    else:
        print("\n✅ 类别映射检查通过!")
    
    return mismatches


def check_visdrone_yaml_mapping():
    """
    检查visdrone_uavdt_joint.yaml中的类别定义
    """
    print(f"\n{'='*80}")
    print("📄 检查YAML文件的类别定义")
    print(f"{'='*80}")
    
    yaml_path = Path("data/visdrone_uavdt_joint.yaml")
    if not yaml_path.exists():
        print(f"❌ YAML文件不存在: {yaml_path}")
        return None
    
    # 手动解析YAML中的类别定义
    with open(yaml_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    class_names = {}
    in_names_section = False
    
    for line in lines:
        if line.strip().startswith("names:"):
            in_names_section = True
            continue
        
        if in_names_section:
            # 匹配 "  0: pedestrian" 格式
            if line.strip() and not line.strip().startswith("#"):
                if ":" in line:
                    parts = line.strip().split(":", 1)
                    if len(parts) == 2:
                        try:
                            class_id = int(parts[0].strip())
                            class_name = parts[1].split("#")[0].strip()
                            class_names[class_id] = class_name
                        except ValueError:
                            pass
            
            # 遇到下一个顶级键则退出
            if line.startswith("# ") or (line.strip() and not line.startswith(" ")):
                if class_names:  # 已经读取到类别名称
                    break
    
    if class_names:
        print("\nYAML文件中定义的类别映射:")
        print(f"{'类别ID':<10} {'类别名称'}")
        print("-" * 40)
        for class_id in sorted(class_names.keys()):
            print(f"{class_id:<10} {class_names[class_id]}")
        print(f"\n总类别数: {len(class_names)}")
        return class_names
    else:
        print("❌ 无法从YAML文件中解析类别定义")
        return None


def suggest_fixes(visdrone_counts, uavdt_counts, yaml_classes):
    """
    根据分析结果提供修复建议
    """
    print(f"\n{'='*80}")
    print("💡 修复建议")
    print(f"{'='*80}")
    
    if visdrone_counts is None or uavdt_counts is None:
        print("❌ 数据不足,无法提供建议")
        return
    
    # 检查类别ID范围
    vd_min, vd_max = min(visdrone_counts.keys()), max(visdrone_counts.keys())
    ua_min, ua_max = min(uavdt_counts.keys()), max(uavdt_counts.keys())
    
    print(f"\n1. 类别ID范围检查:")
    print(f"   VisDrone: {vd_min} ~ {vd_max}")
    print(f"   UAVDT: {ua_min} ~ {ua_max}")
    
    if vd_min != ua_min or vd_max != ua_max:
        print("   🚨 **类别ID范围不一致!**")
        print("   → 这是导致训练失败的根本原因!")
    
    # 检查是否有类别缺失
    vd_classes = set(visdrone_counts.keys())
    ua_classes = set(uavdt_counts.keys())
    
    vd_only = vd_classes - ua_classes
    ua_only = ua_classes - vd_classes
    
    if vd_only:
        print(f"\n2. VisDrone独有的类别ID: {sorted(vd_only)}")
        print("   → UAVDT缺少这些类别,可能需要过滤或映射")
    
    if ua_only:
        print(f"\n3. UAVDT独有的类别ID: {sorted(ua_only)}")
        print("   → VisDrone缺少这些类别,可能需要过滤或映射")
    
    # 提供具体修复方案
    print(f"\n{'='*80}")
    print("🔧 推荐修复方案:")
    print(f"{'='*80}")
    
    print("\n方案A: 重新转换UAVDT标签 (推荐)")
    print("  如果UAVDT的原始标签类别与VisDrone不同,需要:")
    print("  1. 检查UAVDT原始类别定义 (可能是 [car, truck, bus] 三类)")
    print("  2. 修改 utils_convert_visdrone_to_yolo_Version2.py")
    print("  3. 添加类别映射表:")
    print("     UAVDT原始 → VisDrone标准")
    print("     例如: UAVDT的car(0) → VisDrone的car(3)")
    print("  4. 重新生成所有UAVDT标签文件")
    
    print("\n方案B: 只使用共同类别")
    print("  1. 在训练时过滤掉不匹配的类别")
    print("  2. 修改数据加载器,跳过未定义的类别ID")
    print("  3. 缺点: 可能损失部分数据")
    
    print("\n方案C: 单独训练后融合")
    print("  1. VisDrone和UAVDT分别训练各自的模型")
    print("  2. 使用知识蒸馏或模型融合技术")
    print("  3. 缺点: 更复杂,不如解决类别映射问题")
    
    print(f"\n{'='*80}")
    print("⚡ 立即行动:")
    print("  1. 运行此脚本的输出结果")
    print("  2. 检查UAVDT原始数据的类别定义")
    print("  3. 如果类别ID不匹配,优先使用【方案A】重新转换")
    print("  4. 转换后再次运行此脚本验证")
    print(f"{'='*80}\n")


def main():
    """
    主函数: 检查VisDrone和UAVDT的类别映射
    """
    print("\n" + "="*80)
    print("🔍 VisDrone + UAVDT 类别映射检查工具")
    print("="*80)
    print("目的: 诊断联合训练失败是否由类别ID不匹配导致")
    print("="*80 + "\n")
    
    # 定义数据集路径
    # 本地Windows路径 (用于本地测试)
    visdrone_local = r"F:\CV\Paper\yoloDepth\yoloDepth\data\VisDrone2019-DET-YOLO\VisDrone2YOLO\VisDrone2019-DET-train\labels\rgb"
    uavdt_local = r"F:\CV\Paper\yoloDepth\yoloDepth\data\UAVDT_YOLO\train\labels\rgb"
    
    # 服务器路径
    visdrone_server = "/data2/user/2024/lzy/Datasets/VisDrone2019-DET-YOLO/VisDrone2YOLO/VisDrone2019-DET-train/labels/rgb"
    uavdt_server = "/data2/user/2024/lzy/Datasets/UAVDT_YOLO/train/labels/rgb"
    
    # 自动检测运行环境
    if os.path.exists(visdrone_local):
        visdrone_path = visdrone_local
        uavdt_path = uavdt_local
        print("✅ 检测到本地环境 (Windows)")
    elif os.path.exists(visdrone_server):
        visdrone_path = visdrone_server
        uavdt_path = uavdt_server
        print("✅ 检测到服务器环境 (Linux)")
    else:
        print("❌ 错误: 无法找到数据集路径!")
        print("请修改脚本中的路径配置")
        return
    
    # 1. 分析VisDrone
    visdrone_counts, visdrone_examples = analyze_class_distribution(
        "VisDrone2019-DET", visdrone_path
    )
    
    # 2. 分析UAVDT
    uavdt_counts, uavdt_examples = analyze_class_distribution(
        "UAVDT", uavdt_path
    )
    
    # 3. 对比两个数据集
    mismatches = compare_datasets(visdrone_counts, uavdt_counts)
    
    # 4. 检查YAML文件定义
    yaml_classes = check_visdrone_yaml_mapping()
    
    # 5. 提供修复建议
    suggest_fixes(visdrone_counts, uavdt_counts, yaml_classes)
    
    # 6. 保存诊断结果
    output_file = "class_mapping_diagnosis.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("VisDrone + UAVDT 类别映射诊断报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"生成时间: {Path(__file__).stat().st_mtime}\n\n")
        
        f.write("VisDrone类别分布:\n")
        if visdrone_counts:
            for class_id in sorted(visdrone_counts.keys()):
                f.write(f"  类别{class_id}: {visdrone_counts[class_id]}个实例\n")
        
        f.write("\nUAVDT类别分布:\n")
        if uavdt_counts:
            for class_id in sorted(uavdt_counts.keys()):
                f.write(f"  类别{class_id}: {uavdt_counts[class_id]}个实例\n")
        
        if mismatches:
            f.write("\n类别不匹配问题:\n")
            for class_id, reason in mismatches:
                f.write(f"  - 类别{class_id}: {reason}\n")
    
    print(f"\n✅ 诊断结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
