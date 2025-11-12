"""
COCO2014 数据集分析脚本
分析COCO数据集结构，统计类别分布，找出与UAV任务相关的类别

Author: AI Assistant
Date: 2025-11-01
"""

import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm

def analyze_coco_dataset(json_path):
    """分析COCO数据集"""
    print(f"\n{'='*60}")
    print(f"分析 COCO 数据集: {json_path}")
    print(f"{'='*60}")
    
    print("加载 JSON 文件...")
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    # 1. 基本信息
    print(f"\n【基本信息】")
    print(f"图像数量: {len(coco_data['images']):,}")
    print(f"标注数量: {len(coco_data['annotations']):,}")
    print(f"类别数量: {len(coco_data['categories'])}")
    print(f"平均每张图像标注数: {len(coco_data['annotations'])/len(coco_data['images']):.2f}")
    
    # 2. 类别信息
    print(f"\n【COCO 80类别列表】")
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    for cat_id, cat_name in sorted(categories.items()):
        print(f"  {cat_id:2d}: {cat_name}")
    
    # 3. 统计每个类别的标注数
    print(f"\n【类别标注数量统计】")
    cat_counts = Counter()
    for ann in tqdm(coco_data['annotations'], desc="统计类别"):
        cat_counts[ann['category_id']] += 1
    
    print(f"\n{'类别ID':<8} {'类别名称':<20} {'标注数':<10} {'占比':<8}")
    print("-" * 60)
    total_ann = len(coco_data['annotations'])
    for cat_id, count in cat_counts.most_common():
        cat_name = categories[cat_id]
        percentage = count / total_ann * 100
        print(f"{cat_id:<8} {cat_name:<20} {count:<10,} {percentage:>6.2f}%")
    
    # 4. UAV相关类别识别
    print(f"\n【与 VisDrone/UAVDT 相关的类别】")
    
    # VisDrone 10类: ignored, pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor
    uav_related = {
        1: ('person', 'pedestrian'),           # person → pedestrian/people
        2: ('bicycle', 'bicycle'),             # bicycle → bicycle
        3: ('car', 'car'),                     # car → car
        4: ('motorcycle', 'motor'),            # motorcycle → motor
        6: ('bus', 'bus'),                     # bus → bus
        8: ('truck', 'truck/van'),             # truck → truck/van
    }
    
    print(f"\n{'COCO ID':<10} {'COCO名称':<15} {'对应VisDrone':<20} {'标注数':<10}")
    print("-" * 70)
    
    uav_total = 0
    for coco_id, (coco_name, vd_name) in uav_related.items():
        count = cat_counts[coco_id]
        uav_total += count
        print(f"{coco_id:<10} {coco_name:<15} {vd_name:<20} {count:<10,}")
    
    print(f"\n{'总计 UAV相关':<25} {uav_total:,} ({uav_total/total_ann*100:.1f}%)")
    print(f"{'其他类别':<25} {total_ann - uav_total:,} ({(total_ann-uav_total)/total_ann*100:.1f}%)")
    
    # 5. 目标尺寸分布
    print(f"\n【目标尺寸分布】")
    small_count = 0
    medium_count = 0
    large_count = 0
    
    for ann in tqdm(coco_data['annotations'], desc="统计尺寸"):
        area = ann['area']
        if area < 32 * 32:
            small_count += 1
        elif area < 96 * 96:
            medium_count += 1
        else:
            large_count += 1
    
    print(f"  Small (<32²):   {small_count:,} ({small_count/total_ann*100:.1f}%)")
    print(f"  Medium (32²-96²): {medium_count:,} ({medium_count/total_ann*100:.1f}%)")
    print(f"  Large (>96²):   {large_count:,} ({large_count/total_ann*100:.1f}%)")
    
    # 6. 对比 VisDrone/UAVDT
    print(f"\n【与 VisDrone/UAVDT 对比】")
    print(f"\n尺寸分布对比:")
    print(f"{'数据集':<12} {'Small':<10} {'Medium':<10} {'Large':<10}")
    print("-" * 50)
    print(f"{'VisDrone':<12} {'92.4%':<10} {'7.5%':<10} {'0.1%':<10}")
    print(f"{'UAVDT':<12} {'50.2%':<10} {'48.4%':<10} {'1.4%':<10}")
    print(f"{'COCO':<12} {small_count/total_ann*100:<10.1f} {medium_count/total_ann*100:<10.1f} {large_count/total_ann*100:<10.1f}")
    
    print(f"\n域差异分析:")
    print(f"  VisDrone/UAVDT: UAV俯视视角, 城市道路, 车辆/行人为主")
    print(f"  COCO:          地面平视视角, 室内外多样, 80类物体")
    print(f"  相似度:        ⚠️ 低 - 域差异较大!")
    
    return {
        'total_images': len(coco_data['images']),
        'total_annotations': len(coco_data['annotations']),
        'categories': categories,
        'cat_counts': cat_counts,
        'uav_related': uav_related,
        'uav_total': uav_total,
        'size_dist': {
            'small': small_count,
            'medium': medium_count,
            'large': large_count
        }
    }

def main():
    """主函数"""
    coco_root = Path(r'f:\CV\Paper\yoloDepth\yoloDepth\datasets\coco2014')
    
    # 检查数据集是否存在
    if not coco_root.exists():
        print(f"❌ 错误: COCO数据集不存在: {coco_root}")
        return
    
    # 分析训练集
    train_json = coco_root / 'annotations' / 'instances_train2014.json'
    if train_json.exists():
        train_stats = analyze_coco_dataset(train_json)
    else:
        print(f"⚠️ 警告: 训练集JSON不存在: {train_json}")
        train_stats = None
    
    # 总结与建议
    print(f"\n{'='*60}")
    print("分析结论与建议")
    print(f"{'='*60}")
    
    if train_stats:
        print(f"\n【COCO2014 训练集统计】")
        print(f"  总图像数: {train_stats['total_images']:,} 张")
        print(f"  总标注数: {train_stats['total_annotations']:,} 个")
        print(f"  UAV相关: {train_stats['uav_total']:,} 个 ({train_stats['uav_total']/train_stats['total_annotations']*100:.1f}%)")
        
        print(f"\n【使用建议】")
        print(f"\n方案1: 跳过COCO (推荐) ⭐")
        print(f"  - 先完成 VisDrone + UAVDT 联合训练")
        print(f"  - 如果性能达标 (>45% mAP), COCO不是必需的")
        print(f"  - 节省时间: 不需要生成深度图 (~40-50小时)")
        
        print(f"\n方案2: COCO预训练")
        print(f"  - 在COCO上预训练50 epochs (~30小时)")
        print(f"  - 然后在VisDrone+UAVDT上微调 (~40小时)")
        print(f"  - 总耗时: ~70小时")
        print(f"  - 风险: 域差异可能带来负迁移")
        
        print(f"\n方案3: 类别过滤 + 联合训练")
        print(f"  - 只保留6个UAV相关类别")
        print(f"  - 过滤后约 {train_stats['uav_total']:,} 个标注")
        print(f"  - 生成深度图 (~15-20小时)")
        print(f"  - 三数据集联合训练 (~50小时)")
        
        print(f"\n【我的推荐】")
        print(f"  1. ✅ 先运行 VisDrone + UAVDT 训练")
        print(f"  2. ✅ 评估性能 (目标: Overall > 45%)")
        print(f"  3. ⏸️  如果成功, 不需要COCO")
        print(f"  4. ⏸️  如果不够, 再尝试COCO预训练")
    
    print(f"\n{'='*60}")
    print("分析完成!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
