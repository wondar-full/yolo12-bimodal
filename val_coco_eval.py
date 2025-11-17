#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
COCO风格评估脚本 (方案A - 分开训练)
使用 pycocotools 生成完整的 COCO 标准指标,对齐 RemDet 论文 Table 1 & Table 2

使用方法:
    # VisDrone 评估
    python val_coco_eval.py \
        --weights runs/train/visdrone_rgbd_n_300ep/weights/best.pt \
        --data data/visdrone-rgbd.yaml \
        --name visdrone_coco_eval
    
    # UAVDT 评估
    python val_coco_eval.py \
        --weights runs/train/uavdt_rgbd_n_300ep/weights/best.pt \
        --data data/uavdt-rgbd.yaml \
        --name uavdt_coco_eval

输出指标 (完全对齐 RemDet):
    - AP@0.50:0.95 (IoU=0.50:0.95, area=all)
    - AP@0.50      (IoU=0.50, area=all)
    - AP@0.75      (IoU=0.75, area=all)
    - AP_small     (IoU=0.50:0.95, area=small)
    - AP_medium    (IoU=0.50:0.95, area=medium)
    - AP_large     (IoU=0.50:0.95, area=large)
    + AR (Average Recall) 系列指标
"""

import argparse
import json
import os
from pathlib import Path
import yaml
import numpy as np
import torch
from tqdm import tqdm

from ultralytics import YOLO


def create_coco_annotations(data_yaml, split='val'):
    """
    从 YOLO 标注创建 COCO 格式的 ground truth JSON
    
    Args:
        data_yaml: 数据集配置文件路径
        split: 'val' 或 'test'
    
    Returns:
        coco_gt: COCO格式的字典 (可直接保存为JSON)
    """
    # 读取配置
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    dataset_root = Path(data['path'])
    img_dir = dataset_root / data[split]
    label_dir = dataset_root / 'labels' / split
    
    nc = data['nc']
    names = data['names']
    
    # COCO格式字典
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # 添加类别信息
    for i in range(nc):
        coco_gt["categories"].append({
            "id": i + 1,  # COCO类别ID从1开始
            "name": names[i],
            "supercategory": "object"
        })
    
    # 遍历图片
    img_files = sorted(Path(img_dir).glob('*.jpg')) + sorted(Path(img_dir).glob('*.png'))
    ann_id = 1  # 标注ID计数器
    
    for img_id, img_path in enumerate(tqdm(img_files, desc=f"Creating COCO GT for {split}"), 1):
        # 读取图片尺寸
        from PIL import Image
        img = Image.open(img_path)
        width, height = img.size
        
        # 添加图片信息
        coco_gt["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": width,
            "height": height
        })
        
        # 读取对应的标注
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                cls_id = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:5])
                
                # YOLO格式 → COCO格式 (归一化坐标 → 像素坐标)
                x_min = (x_center - w / 2) * width
                y_min = (y_center - h / 2) * height
                bbox_w = w * width
                bbox_h = h * height
                area = bbox_w * bbox_h
                
                coco_gt["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id + 1,  # COCO类别ID从1开始
                    "bbox": [x_min, y_min, bbox_w, bbox_h],  # [x, y, width, height]
                    "area": area,
                    "iscrowd": 0
                })
                ann_id += 1
    
    return coco_gt


def yolo_results_to_coco(results, img_id_map):
    """
    将 YOLO 检测结果转为 COCO 格式的预测 JSON
    
    Args:
        results: YOLO model.val() 返回的结果
        img_id_map: {img_filename: coco_image_id} 映射
    
    Returns:
        coco_pred: COCO格式的预测列表
    """
    coco_pred = []
    
    for result in results:
        img_name = Path(result.path).name
        img_id = img_id_map.get(img_name)
        if img_id is None:
            continue
        
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # 转换为COCO格式 [x, y, width, height]
            bbox = [
                float(x1),
                float(y1),
                float(x2 - x1),
                float(y2 - y1)
            ]
            
            coco_pred.append({
                "image_id": img_id,
                "category_id": cls_id + 1,  # COCO类别ID从1开始
                "bbox": bbox,
                "score": conf
            })
    
    return coco_pred


def evaluate_with_pycocotools(gt_json_path, pred_json_path):
    """
    使用 pycocotools 计算 COCO 标准指标
    
    Args:
        gt_json_path: Ground truth JSON 路径
        pred_json_path: Predictions JSON 路径
    
    Returns:
        metrics: 字典,包含所有 COCO 指标
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("❌ pycocotools not installed!")
        print("   Install with: pip install pycocotools")
        return None
    
    # 加载GT和预测
    coco_gt = COCO(gt_json_path)
    coco_pred = coco_gt.loadRes(pred_json_path)
    
    # 创建评估器
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
    parser = argparse.ArgumentParser(description='COCO-style evaluation for yoloDepth')
    parser.add_argument('--weights', type=str, required=True, help='Model weights path')
    parser.add_argument('--data', type=str, required=True, help='Dataset YAML path')
    parser.add_argument('--name', type=str, default='coco_eval', help='Experiment name')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='0', help='CUDA device')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'], help='Dataset split')
    parser.add_argument('--save-json', action='store_true', help='Save COCO JSONs for manual inspection')
    
    args = parser.parse_args()
    
    # 创建输出目录
    save_dir = Path('runs') / 'val' / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("🚀 Starting COCO-style Evaluation")
    print("="*80)
    print(f"📁 Weights:  {args.weights}")
    print(f"📁 Data:     {args.data}")
    print(f"📁 Save dir: {save_dir}")
    print(f"🖼️  Image size: {args.imgsz}")
    print(f"🔢 Batch size: {args.batch}")
    print(f"🎮 Device:   {args.device}")
    print("="*80 + "\n")
    
    # 1. 创建 COCO Ground Truth JSON
    print("📝 Step 1/4: Creating COCO Ground Truth JSON...")
    gt_json_path = save_dir / f"gt_{args.split}.json"
    
    if not gt_json_path.exists():
        coco_gt = create_coco_annotations(args.data, split=args.split)
        with open(gt_json_path, 'w') as f:
            json.dump(coco_gt, f)
        print(f"   ✅ Saved to {gt_json_path}")
        print(f"   📊 {len(coco_gt['images'])} images, {len(coco_gt['annotations'])} annotations")
    else:
        print(f"   ♻️  Using existing GT JSON: {gt_json_path}")
        with open(gt_json_path, 'r') as f:
            coco_gt = json.load(f)
        print(f"   📊 {len(coco_gt['images'])} images, {len(coco_gt['annotations'])} annotations")
    
    # 创建图片ID映射
    img_id_map = {img['file_name']: img['id'] for img in coco_gt['images']}
    
    # 2. 运行 YOLO 验证生成预测
    print("\n🔍 Step 2/4: Running YOLO Validation...")
    model = YOLO(args.weights)
    
    # 使用 model.val() 并保存结果
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
        save_json=False,  # 我们手动生成COCO格式JSON
        verbose=False
    )
    
    print("   ✅ Validation completed")
    
    # 3. 转换预测为 COCO 格式
    print("\n📝 Step 3/4: Converting predictions to COCO format...")
    
    # 重新运行一次获取详细结果 (model.val()不返回详细boxes)
    pred_results = model.predict(
        source=Path(args.data).parent / 'data' / args.split / 'images' / 'rgb',
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
        stream=True
    )
    
    coco_pred = yolo_results_to_coco(pred_results, img_id_map)
    
    pred_json_path = save_dir / f"pred_{args.split}.json"
    with open(pred_json_path, 'w') as f:
        json.dump(coco_pred, f)
    
    print(f"   ✅ Saved to {pred_json_path}")
    print(f"   📊 {len(coco_pred)} predictions")
    
    # 4. 使用 pycocotools 评估
    print("\n📊 Step 4/4: Evaluating with pycocotools...")
    metrics = evaluate_with_pycocotools(str(gt_json_path), str(pred_json_path))
    
    if metrics is None:
        print("❌ Evaluation failed (pycocotools not available)")
        return
    
    # 确定数据集名称
    dataset_name = 'VisDrone' if 'visdrone' in args.data.lower() else 'UAVDT'
    
    # 打印与RemDet的对比
    print_remdet_comparison(metrics, dataset_name)
    
    # 保存指标到文件
    metrics_file = save_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Metrics saved to {metrics_file}")
    
    # 可选: 保存JSON供手动检查
    if args.save_json:
        print(f"\n📦 COCO JSON files saved:")
        print(f"   GT:   {gt_json_path}")
        print(f"   Pred: {pred_json_path}")
    
    print("\n✅ Evaluation completed!")
    print("="*80)


if __name__ == '__main__':
    main()
