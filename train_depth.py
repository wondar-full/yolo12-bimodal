#!/usr/bin/env python3
# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
YOLOv12-RGBD Training Script for yoloDepth Project

This script trains the dual-modal YOLO12 model on RGB-D data with configurations
aligned to RemDet (AAAI2025) for fair comparison on UAV object detection tasks.

Created: 2025-10-26
Author: Generated for yoloDepth project
Target: Exceed RemDet performance on VisDrone dataset

Usage:
    Basic training:
        python train_depth.py --data visdrone-rgbd.yaml --epochs 300
    
    Multi-GPU training:
        python train_depth.py --data visdrone-rgbd.yaml --device 0,1,2,3
    
    Resume from checkpoint:
        python train_depth.py --resume runs/train/exp1/weights/last.pt
    
    Custom hyperparameters:
        python train_depth.py --data visdrone-rgbd.yaml --batch 32 --lr0 0.02
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to Python path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv12-RGBD root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Import Ultralytics
try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER, colorstr
except ImportError:
    raise ImportError(
        "Ultralytics package not found. Install with: pip install ultralytics>=8.3.155"
    )


def parse_args():
    """
    Parse command-line arguments for training configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments with training parameters.
    """
    parser = argparse.ArgumentParser(description="Train YOLOv12-RGBD on RGB-D datasets")
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="ultralytics/cfg/models/12/yolo12s-rgbd-v1.yaml",
        help="Model configuration YAML file path"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="/data2/user/2024/lzy/yolo12-bimodal/models/yolo12s.pt",
        help="Pretrained weights path (optional). Use yolo12s.pt for RGB branch initialization"
    )
    
    # Dataset configuration
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Dataset YAML file path (must include train_depth and val_depth)"
    )
    
    # Training hyperparameters (RemDet-aligned)
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs (RemDet uses 300)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (RemDet uses 16)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (RemDet uses 640×640)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="2",
        help="CUDA device(s) to use, e.g. '0' or '0,1,2,3' for multi-GPU"
    )
    
    # Optimizer settings
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        choices=["SGD", "Adam", "AdamW"],
        help="Optimizer type (RemDet uses SGD)"
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate (RemDet uses 0.01)"
    )
    parser.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help="Final learning rate factor (lr_final = lr0 * lrf)"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.937,
        help="SGD momentum (RemDet uses 0.937)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0005,
        help="Weight decay for optimizer (RemDet uses 0.0005)"
    )
    
    # Data augmentation (RemDet-aligned)
    parser.add_argument(
        "--mosaic",
        type=float,
        default=1.0,
        help="Mosaic augmentation probability (RemDet uses 1.0 = 100%)"
    )
    parser.add_argument(
        "--mixup",
        type=float,
        default=0.15,
        help="MixUp augmentation probability (RemDet uses 0.15 = 15%)"
    )
    parser.add_argument(
        "--hsv_h",
        type=float,
        default=0.015,
        help="HSV-Hue augmentation range"
    )
    parser.add_argument(
        "--hsv_s",
        type=float,
        default=0.7,
        help="HSV-Saturation augmentation range"
    )
    parser.add_argument(
        "--hsv_v",
        type=float,
        default=0.4,
        help="HSV-Value augmentation range"
    )
    parser.add_argument(
        "--degrees",
        type=float,
        default=0.0,
        help="Rotation augmentation range (degrees). RemDet disables rotation"
    )
    parser.add_argument(
        "--translate",
        type=float,
        default=0.1,
        help="Translation augmentation range (fraction of image size)"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.9,
        help="Scaling augmentation range (0.9 means ±10%)"
    )
    parser.add_argument(
        "--fliplr",
        type=float,
        default=0.5,
        help="Horizontal flip probability (RemDet uses 0.5 = 50%)"
    )
    parser.add_argument(
        "--flipud",
        type=float,
        default=0.0,
        help="Vertical flip probability (RemDet disables vertical flip)"
    )
    
    # Training settings
    parser.add_argument(
        "--warmup_epochs",
        type=float,
        default=3.0,
        help="Warmup epochs (RemDet uses 3)"
    )
    parser.add_argument(
        "--close_mosaic",
        type=int,
        default=10,
        help="Disable mosaic in last N epochs (stabilize training)"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Use Automatic Mixed Precision (AMP) for faster training"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (epochs without improvement)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="",
        choices=["", "ram", "disk"],
        help="Cache images: '' (no cache), 'ram' (in RAM), 'disk' (on disk)"
    )
    
    # Validation settings
    parser.add_argument(
        "--val",
        action="store_true",
        default=True,
        help="Validate after each epoch"
    )
    parser.add_argument(
        "--save_period",
        type=int,
        default=10,
        help="Save checkpoint every N epochs (-1 to disable)"
    )
    
    # Logging and output
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Project folder for saving training results"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="exp",
        help="Experiment name (results saved in project/name)"
    )
    parser.add_argument(
        "--exist_ok",
        action="store_true",
        help="Overwrite existing project/name folder"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Resume training from checkpoint path (e.g., runs/train/exp1/weights/last.pt)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print verbose training logs"
    )
    
    # Advanced settings
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic mode (may reduce performance)"
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        default=True,
        help="Save training plots (loss curves, metrics, etc.)"
    )
    
    return parser.parse_args()


def validate_args(args):
    """
    Validate and preprocess arguments.
    
    Args:
        args: Parsed command-line arguments.
        
    Raises:
        FileNotFoundError: If required files don't exist.
        ValueError: If argument values are invalid.
    """
    # Check model config exists
    if not Path(args.model).exists():
        raise FileNotFoundError(
            f"Model config not found: {args.model}\n"
            f"Available configs in ultralytics/cfg/models/12/:\n"
            f"  - yolo12s-rgbd-v1.yaml (RGB-D dual-modal)\n"
            f"  - yolo12.yaml (RGB-only baseline)"
        )
    
    # Check data config exists
    if not Path(args.data).exists():
        raise FileNotFoundError(
            f"Dataset config not found: {args.data}\n"
            f"Expected format:\n"
            f"  path: /path/to/dataset\n"
            f"  train: images/train\n"
            f"  val: images/val\n"
            f"  train_depth: depths/train  # Required for RGB-D\n"
            f"  val_depth: depths/val      # Required for RGB-D\n"
            f"  nc: 10\n"
            f"  names: ['class1', 'class2', ...]"
        )
    
    # Check weights exist if specified
    if args.weights and not Path(args.weights).exists():
        raise FileNotFoundError(
            f"Pretrained weights not found: {args.weights}\n"
            f"Available options:\n"
            f"  - Leave empty for random initialization\n"
            f"  - Use yolo12s.pt for RGB branch pretrained weights\n"
            f"  - Use previous checkpoint for resuming training"
        )
    
    # Validate device format
    if not all(c.isdigit() or c == ',' for c in args.device):
        raise ValueError(
            f"Invalid device format: {args.device}\n"
            f"Expected: '0' (single GPU) or '0,1,2,3' (multi-GPU)"
        )
    
    # Warn about incompatible settings
    if args.cache == "ram" and args.batch > 16:
        LOGGER.warning(
            f"Large batch size ({args.batch}) with RAM caching may cause OOM. "
            f"Consider reducing batch size or using cache='disk'"
        )
    
    LOGGER.info(f"{colorstr('Configuration validated:')} All checks passed ✅")


def print_training_summary(args):
    """
    Print training configuration summary before starting.
    
    Args:
        args: Parsed command-line arguments.
    """
    summary = f"""
{colorstr('bold', 'blue', '='*70)}
{colorstr('bold', 'blue', 'YOLOv12-RGBD Training Configuration')}
{colorstr('bold', 'blue', '='*70)}

{colorstr('bold', 'Model Settings:')}
  Model Config:     {args.model}
  Pretrained:       {args.weights if args.weights else 'None (random init)'}
  Input Size:       {args.imgsz}×{args.imgsz}
  Batch Size:       {args.batch}
  Device:           {args.device}

{colorstr('bold', 'Dataset Settings:')}
  Data Config:      {args.data}
  Epochs:           {args.epochs}
  Warmup Epochs:    {args.warmup_epochs}
  Close Mosaic:     Last {args.close_mosaic} epochs

{colorstr('bold', 'Optimizer Settings:')} {colorstr('green', '(RemDet-Aligned)')}
  Optimizer:        {args.optimizer}
  Learning Rate:    {args.lr0} → {args.lr0 * args.lrf} (cosine decay)
  Momentum:         {args.momentum}
  Weight Decay:     {args.weight_decay}

{colorstr('bold', 'Augmentation Settings:')} {colorstr('green', '(RemDet-Aligned)')}
  Mosaic:           {args.mosaic * 100:.0f}%
  MixUp:            {args.mixup * 100:.0f}%
  HSV-H:            {args.hsv_h}
  HSV-S:            {args.hsv_s}
  HSV-V:            {args.hsv_v}
  Translation:      ±{args.translate * 100:.0f}%
  Scale:            {args.scale}
  FlipLR:           {args.fliplr * 100:.0f}%

{colorstr('bold', 'Training Settings:')}
  AMP:              {'Enabled' if args.amp else 'Disabled'}
  Workers:          {args.workers}
  Cache:            {args.cache if args.cache else 'Disabled'}
  Patience:         {args.patience} epochs
  Save Period:      Every {args.save_period} epochs

{colorstr('bold', 'Output Settings:')}
  Project:          {args.project}
  Name:             {args.name}
  Resume:           {args.resume if args.resume else 'No'}

{colorstr('bold', 'blue', '='*70)}
"""
    print(summary)


def main():
    """
    Main training function.
    
    This function:
    1. Parses command-line arguments
    2. Validates configuration
    3. Initializes YOLO model
    4. Starts training with RemDet-aligned hyperparameters
    5. Saves results and generates plots
    """
    # Parse arguments
    args = parse_args()
    
    # Validate arguments
    validate_args(args)
    
    # Print training summary
    print_training_summary(args)
    
    # Initialize model
    LOGGER.info(f"{colorstr('bold', 'Initializing model...')}")
    if args.resume:
        # Resume from checkpoint
        model = YOLO(args.resume)
        LOGGER.info(f"{colorstr('green', f'Resumed from checkpoint: {args.resume}')}")
    elif args.weights:
        # Load pretrained weights
        model = YOLO(args.model)
        model.load(args.weights)
        LOGGER.info(f"{colorstr('green', f'Loaded pretrained weights: {args.weights}')}")
    else:
        # Random initialization
        model = YOLO(args.model)
        LOGGER.info(f"{colorstr('yellow', 'Random initialization (no pretrained weights)')}")
    
    # Start training
    LOGGER.info(f"{colorstr('bold', 'magenta', 'Starting training...')}")
    
    try:
        results = model.train(
            # Data and model
            data=args.data,
            
            # Training settings
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            
            # Optimizer
            optimizer=args.optimizer,
            lr0=args.lr0,
            lrf=args.lrf,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            
            # Augmentation
            mosaic=args.mosaic,
            mixup=args.mixup,
            hsv_h=args.hsv_h,
            hsv_s=args.hsv_s,
            hsv_v=args.hsv_v,
            degrees=args.degrees,
            translate=args.translate,
            scale=args.scale,
            fliplr=args.fliplr,
            flipud=args.flipud,
            close_mosaic=args.close_mosaic,
            
            # Performance
            amp=args.amp,
            workers=args.workers,
            cache=args.cache if args.cache else False,
            
            # Validation
            val=args.val,
            patience=args.patience,
            save_period=args.save_period,
            
            # Logging
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
            verbose=args.verbose,
            plots=args.plots,
            seed=args.seed,
            deterministic=args.deterministic,
        )
        
        # Training completed
        LOGGER.info(f"{colorstr('bold', 'green', '✅ Training completed successfully!')}")
        LOGGER.info(f"{colorstr('Results saved to:')} {results.save_dir}")
        
        # Print final metrics
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            LOGGER.info(f"\n{colorstr('bold', 'Final Metrics:')}")
            LOGGER.info(f"  mAP@0.5:      {metrics.get('metrics/mAP50(B)', 0):.1f}%")
            LOGGER.info(f"  mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 0):.1f}%")
            
            # Compare with RemDet if metrics available
            remdet_map50 = 45.2  # RemDet-X benchmark
            remdet_map_small = 21.3
            current_map50 = metrics.get('metrics/mAP50(B)', 0)
            
            if current_map50 > 0:
                improvement = current_map50 - remdet_map50
                if improvement > 0:
                    LOGGER.info(f"\n{colorstr('bold', 'green', f'🎉 Exceeded RemDet by {improvement:.1f} points!')}")
                else:
                    LOGGER.info(f"\n{colorstr('yellow', f'Gap to RemDet: {abs(improvement):.1f} points (target: exceed 45.2%)')}")
        
        return results
        
    except Exception as e:
        LOGGER.error(f"{colorstr('bold', 'red', f'❌ Training failed: {str(e)}')}")
        raise


if __name__ == "__main__":
    main()


# 📚 八股知识扩展: 训练脚本设计
"""
1. 为什么要单独写train_depth.py而不直接用yolo train?
   答: (1) RemDet对齐: 需要精确控制超参数(mosaic=1.0, mixup=0.15等)
       (2) RGB-D特定: 需要YOLORGBDDataset,不是标准YOLO数据集
       (3) 实验管理: 方便记录配置、对比基线、生成论文表格
       (4) 扩展性: 未来可添加SOLR loss、自定义callbacks等

2. 为什么warmup_epochs=3这么少?
   答: (1) RemDet论文设置: 300 epochs中前3个epoch做warmup
       (2) 避免过长warmup: 学习率长期过低会浪费训练时间
       (3) SGD优化器: 相比Adam,SGD对warmup不那么敏感
       (4) 实验验证: YOLOv8系列测试表明3 epoch足够

3. close_mosaic=10的作用?
   答: (1) Mosaic增强在后期可能引入噪声(拼接导致不自然)
       (2) 最后10个epoch关闭,让模型学习真实分布
       (3) 稳定收敛: 类似学习率decay,逐步降低增强强度
       (4) RemDet也采用类似策略(虽未明确写)

4. 如何判断训练成功?
   答: (1) Loss收敛: box_loss, cls_loss, dfl_loss均下降
       (2) mAP上升: 特别关注mAP_small(小目标性能)
       (3) 无NaN/Inf: 检查RGBDStem的gate_mean(应在0.3-0.7)
       (4) 速度合理: 4090应在50-60 FPS(若<30需优化)

5. 训练失败常见原因?
   答: (1) 数据路径错误: RGB和Depth不匹配
       (2) 显存不足: batch_size过大或cache='ram'
       (3) 梯度爆炸: AMP + 深度特征可能导致(需要梯度裁剪)
       (4) 学习率过高: 对于pretrained模型,lr0=0.01可能太大

思考题:
Q1: 如果只有8GB显存,如何调整配置?
A1: (1) batch=8 (减半)
    (2) cache=False (禁用缓存)
    (3) workers=4 (减少dataloader并行)
    (4) imgsz=512 (降低分辨率,不推荐)

Q2: 如何加速训练?
A2: (1) cache='ram' (显存足够时)
    (2) 多GPU: device='0,1,2,3'
    (3) AMP=True (已默认开启)
    (4) workers=16 (增加dataloader线程)
    (5) 预计算depth缓存(避免重复预处理)

Q3: 如何验证RGB-D是否生效?
A3: (1) 打印model.model[0],应看到RGBDStem
    (2) 监控gate_mean(应>0, 说明depth有贡献)
    (3) 对比RGB-only baseline(应有+2-5% mAP提升)
    (4) 可视化depth预处理结果
"""
