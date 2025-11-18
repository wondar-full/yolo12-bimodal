#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
COCO标准评估脚本 (整合版 - 方案A)
Step 1: 使用 model.val() 生成 predictions.json
Step 2: 使用 pycocotools 加载现有的 val.json 和 predictions.json 进行COCO标准评估

使用方法:
    # VisDrone 评估
    python val_coco_standard.py \
        --weights runs/train/visdrone_rgbd_n_300ep/weights/best.pt \
        --data data/visdrone-rgbd.yaml \
        --gt-json /data2/user/2024/lzy/Datasets/VisDrone2019-DET-COCO/annotations/VisDrone2019-DET_val_coco.json \
        --name visdrone_coco_eval
    
    # UAVDT 评估
    python val_coco_standard.py \
        --weights runs/train/uavdt_rgbd_n_300ep/weights/best.pt \
        --data data/uavdt-rgbd.yaml \
        --gt-json /data2/user/2024/lzy/Datasets/UAVDT/annotations/UAV-benchmark-M-Val.json \
        --name uavdt_coco_eval

输出指标 (完全对齐 RemDet):
    - AP@0.50:0.95 (IoU=0.50:0.95, area=all)
    - AP@0.50      (IoU=0.50, area=all) ← 主要对比指标
    - AP@0.75      (IoU=0.75, area=all)
    - AP_small     (IoU=0.50:0.95, area=small) ← UAV关键指标
    - AP_medium    (IoU=0.50:0.95, area=medium)
    - AP_large     (IoU=0.50:0.95, area=large)
    + AR (Average Recall) 系列指标
"""

import argparse
import json
from pathlib import Path
import sys

from ultralytics import YOLO


def evaluate_with_pycocotools(gt_json_path, pred_json_path):
    """
    使用 pycocotools 计算 COCO 标准指标
    
    Args:
        gt_json_path: Ground truth JSON 路径 (你已有的val.json)
        pred_json_path: Predictions JSON 路径 (model.val()生成的)
    
    Returns:
        metrics: 字典,包含所有 COCO 指标
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("❌ pycocotools not installed!")
        print("   Install with:")
        print("   - Linux: pip install pycocotools")
        print("   - Windows: pip install pycocotools-windows")
        return None
    
    print(f"\n📂 Loading Ground Truth: {gt_json_path}")
    coco_gt = COCO(gt_json_path)
    
    print(f"📂 Loading Predictions: {pred_json_path}")
    coco_pred = coco_gt.loadRes(pred_json_path)
    
    # 创建评估器
    print("\n🔍 Running COCO evaluation...")
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    
    # 运行评估
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # 提取指标
    metrics = {
        "AP@0.50:0.95": coco_eval.stats[0],  # AP at IoU=0.50:0.95
        "AP@0.50": coco_eval.stats[1],       # AP at IoU=0.50
        "AP@0.75": coco_eval.stats[2],       # AP at IoU=0.75
        "AP_small": coco_eval.stats[3],      # AP for small objects
        "AP_medium": coco_eval.stats[4],     # AP for medium objects
        "AP_large": coco_eval.stats[5],      # AP for large objects
        "AR@0.50:0.95 (max=1)": coco_eval.stats[6],
        "AR@0.50:0.95 (max=10)": coco_eval.stats[7],
        "AR@0.50:0.95 (max=100)": coco_eval.stats[8],
        "AR_small": coco_eval.stats[9],
        "AR_medium": coco_eval.stats[10],
        "AR_large": coco_eval.stats[11]
    }
    
    return metrics


def print_remdet_comparison(metrics, dataset_name):
    """
    打印与 RemDet 的对比表格
    
    Args:
        metrics: COCO评估指标字典
        dataset_name: 'VisDrone' 或 'UAVDT'
    """
    print("\n" + "="*80)
    print(f"📊 {dataset_name} Results - RemDet Comparison")
    print("="*80)
    
    # RemDet baseline (根据数据集选择)
    if dataset_name == 'VisDrone':
        remdet_baselines = {
            'RemDet-Tiny': {'AP@0.50:0.95': 20.1, 'AP@0.50': 33.5, 'AP@0.75': 20.9, 'AP_small': 9.6, 'AP_medium': 30.4, 'AP_large': 49.7},
            'RemDet-S':    {'AP@0.50:0.95': 26.3, 'AP@0.50': 42.3, 'AP@0.75': 27.8, 'AP_small': 14.5, 'AP_medium': 39.1, 'AP_large': 55.8},
            'RemDet-M':    {'AP@0.50:0.95': 28.0, 'AP@0.50': 45.0, 'AP@0.75': 29.6, 'AP_small': 16.2, 'AP_medium': 41.5, 'AP_large': 57.3},
            'RemDet-L':    {'AP@0.50:0.95': 29.5, 'AP@0.50': 47.4, 'AP@0.75': 30.9, 'AP_small': 18.5, 'AP_medium': 43.5, 'AP_large': 58.1},
            'RemDet-X':    {'AP@0.50:0.95': 29.9, 'AP@0.50': 48.3, 'AP@0.75': 31.0, 'AP_small': 19.5, 'AP_medium': 44.1, 'AP_large': 58.6}
        }
        primary_baseline = 'RemDet-X'
    else:  # UAVDT
        remdet_baselines = {
            'RemDet-L': {'AP@0.50:0.95': 20.6, 'AP@0.50': 34.5, 'AP@0.75': 20.5, 'AP_small': 12.6, 'AP_medium': 29.0, 'AP_large': 46.8}
        }
        primary_baseline = 'RemDet-L'
    
    # 打印主要指标对比
    print("\n🎯 Main Metrics (vs {})".format(primary_baseline))
    print("-"*80)
    print(f"{'Metric':<20} {'YoloDepth':<15} {primary_baseline:<15} {'Δ':<15}")
    print("-"*80)
    
    main_metrics = ['AP@0.50:0.95', 'AP@0.50', 'AP@0.75', 'AP_small', 'AP_medium', 'AP_large']
    baseline = remdet_baselines[primary_baseline]
    
    for metric in main_metrics:
        our_val = metrics[metric] * 100  # 转为百分比
        baseline_val = baseline[metric]
        delta = our_val - baseline_val
        delta_str = f"{delta:+.1f}%" if delta != 0 else "0.0%"
        
        # 颜色标记 (仅在终端显示时有效)
        if delta > 0:
            delta_str = f"✅ {delta_str}"
        elif delta < -2:
            delta_str = f"❌ {delta_str}"
        else:
            delta_str = f"➖ {delta_str}"
        
        print(f"{metric:<20} {our_val:>6.1f}%{'':<8} {baseline_val:>6.1f}%{'':<8} {delta_str}")
    
    # 打印完整基线对比
    if len(remdet_baselines) > 1:
        print("\n📋 Full RemDet Baseline Comparison (AP@0.50)")
        print("-"*80)
        print(f"{'Model':<20} {'AP@0.50':<15} {'vs Ours':<15}")
        print("-"*80)
        for model, values in remdet_baselines.items():
            our_val = metrics['AP@0.50'] * 100
            baseline_val = values['AP@0.50']
            delta = our_val - baseline_val
            delta_str = f"{delta:+.1f}%"
            print(f"{model:<20} {baseline_val:>6.1f}%{'':<8} {delta_str}")
    
    # 打印次要指标
    print("\n📈 Additional Metrics")
    print("-"*80)
    print(f"{'Metric':<30} {'Value':<15}")
    print("-"*80)
    for metric in ['AR@0.50:0.95 (max=100)', 'AR_small', 'AR_medium', 'AR_large']:
        val = metrics[metric] * 100
        print(f"{metric:<30} {val:>6.1f}%")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='COCO标准评估 (整合版)')
    parser.add_argument('--weights', type=str, required=True, help='模型权重路径')
    parser.add_argument('--data', type=str, required=True, help='数据集YAML配置文件')
    parser.add_argument('--gt-json', type=str, required=True, help='Ground Truth COCO JSON路径')
    parser.add_argument('--name', type=str, default='coco_eval', help='实验名称')
    parser.add_argument('--imgsz', type=int, default=640, help='图片尺寸')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='0', help='CUDA设备')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'], help='数据集分割')
    parser.add_argument('--save-json', action='store_true', help='保存predictions.json供后续检查')
    
    args = parser.parse_args()
    
    # 创建输出目录
    save_dir = Path('runs') / 'val' / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("🚀 COCO标准评估 (两步法)")
    print("="*80)
    print(f"📁 Weights:     {args.weights}")
    print(f"📁 Data YAML:   {args.data}")
    print(f"📁 GT JSON:     {args.gt_json}")
    print(f"📁 Save dir:    {save_dir}")
    print(f"🖼️  Image size:  {args.imgsz}")
    print(f"🔢 Batch size:  {args.batch}")
    print(f"🎮 Device:      {args.device}")
    print("="*80 + "\n")
    
    # 检查GT JSON是否存在
    if not Path(args.gt_json).exists():
        print(f"❌ Error: Ground Truth JSON not found: {args.gt_json}")
        print("\n可用的GT JSON路径:")
        print("  VisDrone: /data2/user/2024/lzy/Datasets/VisDrone2019-DET-COCO/annotations/VisDrone2019-DET_val_coco.json")
        print("  UAVDT:    /data2/user/2024/lzy/Datasets/UAVDT/annotations/UAV-benchmark-M-Val.json")
        sys.exit(1)
    
    # =================================================================
    # Step 1: 运行 YOLO 验证生成 predictions.json
    # =================================================================
    print("="*80)
    print("📝 Step 1/2: Running YOLO Validation to Generate predictions.json")
    print("="*80)
    
    model = YOLO(args.weights)
    
    # 运行验证 (save_json=True 会自动生成 predictions.json)
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
        save_json=True,  # 关键: 生成COCO格式的predictions.json
        project=str(save_dir.parent),
        name=args.name,
        verbose=True
    )
    
    print("   ✅ YOLO validation completed")
    
    # 查找生成的 predictions.json
    # Ultralytics 默认保存在 runs/val/{name}/predictions.json
    pred_json_path = save_dir / 'predictions.json'
    
    if not pred_json_path.exists():
        print(f"❌ Error: predictions.json not found at {pred_json_path}")
        print("   Please check if save_json=True worked correctly")
        sys.exit(1)
    
    print(f"   📂 predictions.json saved to: {pred_json_path}")
    
    # =================================================================
    # Step 2: 使用 pycocotools 进行 COCO 标准评估
    # =================================================================
    print("\n" + "="*80)
    print("📊 Step 2/2: Evaluating with pycocotools")
    print("="*80)
    
    metrics = evaluate_with_pycocotools(args.gt_json, str(pred_json_path))
    
    if metrics is None:
        print("❌ Evaluation failed (pycocotools not available)")
        sys.exit(1)
    
    # 确定数据集名称
    if 'visdrone' in args.data.lower() or 'visdrone' in args.gt_json.lower():
        dataset_name = 'VisDrone'
    elif 'uavdt' in args.data.lower() or 'uavdt' in args.gt_json.lower():
        dataset_name = 'UAVDT'
    else:
        dataset_name = 'Unknown'
    
    # 打印与RemDet的对比
    print_remdet_comparison(metrics, dataset_name)
    
    # 保存指标到文件
    metrics_file = save_dir / 'coco_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 COCO metrics saved to {metrics_file}")
    
    # 生成简洁的结果报告
    report_file = save_dir / 'evaluation_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"{dataset_name} COCO Evaluation Report\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {args.weights}\n")
        f.write(f"Data:  {args.data}\n")
        f.write(f"GT:    {args.gt_json}\n\n")
        
        f.write("Main Metrics:\n")
        f.write("-"*80 + "\n")
        for metric in ['AP@0.50:0.95', 'AP@0.50', 'AP@0.75', 'AP_small', 'AP_medium', 'AP_large']:
            val = metrics[metric] * 100
            f.write(f"{metric:<20} {val:>6.2f}%\n")
        
        f.write("\nAdditional Metrics:\n")
        f.write("-"*80 + "\n")
        for metric in ['AR@0.50:0.95 (max=100)', 'AR_small', 'AR_medium', 'AR_large']:
            val = metrics[metric] * 100
            f.write(f"{metric:<30} {val:>6.2f}%\n")
    
    print(f"📄 Evaluation report saved to {report_file}")
    
    # 可选: 保存JSON供手动检查
    if args.save_json:
        print(f"\n📦 JSON files for manual inspection:")
        print(f"   GT:   {args.gt_json}")
        print(f"   Pred: {pred_json_path}")
    
    print("\n✅ Evaluation completed!")
    print("="*80)
    
    # 返回主要指标用于后续分析
    return {
        'AP@0.50:0.95': metrics['AP@0.50:0.95'] * 100,
        'AP@0.50': metrics['AP@0.50'] * 100,
        'AP_small': metrics['AP_small'] * 100
    }


if __name__ == '__main__':
    main()
