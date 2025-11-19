#!/usr/bin/env python3
"""
YOLOv12-RGBD Training Script with SOLR Loss (Simplified Version)

Based on train_uav_joint.py pattern - no custom Trainer needed!

Usage:
    python train_depth_solr_v2.py --data visdrone-rgbd.yaml --cfg n --epochs 300
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import LOGGER

# Import SOLR (will be integrated via callback)
try:
    from ultralytics.utils.solr_loss import SOLRDetectionLoss
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.torch_utils import unwrap_model
    SOLR_AVAILABLE = True
except ImportError:
    LOGGER.warning("SOLR not available, will use standard loss")
    SOLR_AVAILABLE = False


def integrate_solr_loss(trainer):
    """Callback to integrate SOLR loss at training start."""
    if not SOLR_AVAILABLE:
        return
    
    model = unwrap_model(trainer.model)
    if not hasattr(model, 'model') or not hasattr(model.model[-1], 'no'):
        return
    
    LOGGER.info("\n🔧 Integrating SOLR loss...")
    
    # Get SOLR params from model.custom_args (set in main)
    custom_args = getattr(trainer.model, 'custom_args', None)
    small_weight = getattr(custom_args, 'small_weight', 2.5) if custom_args else 2.5
    medium_weight = getattr(custom_args, 'medium_weight', 2.0) if custom_args else 2.0
    large_weight = getattr(custom_args, 'large_weight', 1.0) if custom_args else 1.0
    small_thresh = getattr(custom_args, 'small_thresh', 32) if custom_args else 32
    large_thresh = getattr(custom_args, 'large_thresh', 96) if custom_args else 96
    
    # Create SOLR loss
    base_loss = v8DetectionLoss(model)
    model.criterion = SOLRDetectionLoss(
        base_loss=base_loss,
        small_weight=small_weight,
        medium_weight=medium_weight,
        large_weight=large_weight,
        small_thresh=small_thresh,
        large_thresh=large_thresh,
        image_size=trainer.args.imgsz
    )
    
    LOGGER.info(f"✅ SOLR loss integrated: small={small_weight}x, medium={medium_weight}x, large={large_weight}x")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train YOLOv12-RGBD with SOLR loss")
    
    # Model and data
    parser.add_argument("--model", type=str, default="ultralytics/cfg/models/12/yolo12-rgbd-v2.1-universal.yaml")
    parser.add_argument("--cfg", type=str, default="n", help="Model size (n/s/m/l/x)")
    parser.add_argument("--data", type=str, required=True, help="Dataset YAML")
    parser.add_argument("--weights", type=str, default="", help="Pretrained weights")
    
    # Training
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    
    # SOLR parameters
    parser.add_argument("--small_weight", type=float, default=2.5)
    parser.add_argument("--medium_weight", type=float, default=2.0)
    parser.add_argument("--large_weight", type=float, default=1.0)
    parser.add_argument("--small_thresh", type=int, default=32)
    parser.add_argument("--large_thresh", type=int, default=96)
    
    # Optimizer (RemDet-aligned)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    
    # Augmentation (RemDet-aligned)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.15)
    parser.add_argument("--close_mosaic", type=int, default=10)
    
    # Experiment
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="visdrone_solr")
    parser.add_argument("--exist_ok", action="store_true")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save_period", type=int, default=50)
    parser.add_argument("--patience", type=int, default=100)
    
    # Misc
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--cache", action="store_true", default=False)
    
    return parser.parse_args()


def main():
    """Main training function."""
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    args = parse_args()
    
    # Print configuration
    LOGGER.info("=" * 70)
    LOGGER.info("YOLOv12-RGBD Training with SOLR Loss (v2)")
    LOGGER.info("=" * 70)
    LOGGER.info(f"Model: {args.model}")
    LOGGER.info(f"Data: {args.data}")
    LOGGER.info(f"Weights: {args.weights if args.weights else 'None (from scratch)'}")
    LOGGER.info(f"Epochs: {args.epochs}, Batch: {args.batch}, Device: {args.device}")
    LOGGER.info(f"SOLR: small={args.small_weight}x, medium={args.medium_weight}x, large={args.large_weight}x")
    LOGGER.info("=" * 70)
    
    # Initialize model
    if args.weights:
        model = YOLO(args.weights)
        LOGGER.info(f"Loaded pretrained weights from {args.weights}")
    else:
        model = YOLO(args.model, task='detect')
        if args.cfg:
            model.model_name = f"yolo12{args.cfg}"
            LOGGER.info(f"Training YOLO12-{args.cfg.upper()} from scratch")
    
    # Store SOLR params as model attribute (for callback to access)
    class CustomArgs:
        pass
    model.custom_args = CustomArgs()
    model.custom_args.small_weight = args.small_weight
    model.custom_args.medium_weight = args.medium_weight
    model.custom_args.large_weight = args.large_weight
    model.custom_args.small_thresh = args.small_thresh
    model.custom_args.large_thresh = args.large_thresh
    
    # Register SOLR integration callback
    if SOLR_AVAILABLE:
        model.add_callback('on_train_start', integrate_solr_loss)
    
    # Start training
    results = model.train(
        # Data
        data=args.data,
        
        # Training schedule
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        
        # Optimizer (RemDet-aligned)
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        
        # Augmentation (RemDet-aligned)
        mosaic=args.mosaic,
        mixup=args.mixup,
        close_mosaic=args.close_mosaic,
        
        # Hardware
        device=args.device,
        workers=args.workers,
        amp=args.amp,
        cache=args.cache,
        
        # Experiment
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        resume=args.resume,
        save_period=args.save_period,
        patience=args.patience,
    )
    
    # Print results
    LOGGER.info("=" * 70)
    LOGGER.info("Training completed!")
    LOGGER.info("=" * 70)


if __name__ == '__main__':
    main()
