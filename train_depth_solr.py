#!/usr/bin/env python3
# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
YOLOv12-RGBD Training Script with SOLR Loss

This script trains the dual-modal YOLO12 model with SOLR (Small Object Loss Reweighting)
to improve performance on small and medium objects in UAV detection datasets.

Key Features:
    - SOLR loss for balanced training across object sizes
    - RGB-D dual-modal input support
    - RemDet-aligned hyperparameters
    - Automatic SOLR weight tuning

Target: Close the gap with RemDet on VisDrone (especially AP_m: medium objects)

Usage:
    Quick test (10 epochs, nano model):
        python train_depth_solr.py --data visdrone-rgbd.yaml --cfg n --epochs 10
    
    Full training (300 epochs, small model):
        python train_depth_solr.py --data visdrone-rgbd.yaml --cfg s --epochs 300
    
    Medium model with custom SOLR weights:
        python train_depth_solr.py --data visdrone-rgbd.yaml --cfg m \\
            --small_weight 2.5 --medium_weight 2.5 --large_weight 1.0
    
    Multi-size training (compare n/s/m/l/x):
        python train_depth_solr.py --data visdrone-rgbd.yaml --cfg n --name solr_n_300ep
        python train_depth_solr.py --data visdrone-rgbd.yaml --cfg s --name solr_s_300ep
        python train_depth_solr.py --data visdrone-rgbd.yaml --cfg m --name solr_m_300ep --batch 8
        python train_depth_solr.py --data visdrone-rgbd.yaml --cfg l --name solr_l_300ep --batch 4
        python train_depth_solr.py --data visdrone-rgbd.yaml --cfg x --name solr_x_300ep --batch 2
    
    Multi-GPU training (large model):
        python train_depth_solr.py --data visdrone-rgbd.yaml --cfg l --device 0,1,2,3

Created: 2025-11-19
Author: Generated for yolo12-bimodal project
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to Python path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolo12-bimodal root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Import Ultralytics
try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER, colorstr
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.solr_loss import SOLRDetectionLoss
    from ultralytics.utils.loss import v8DetectionLoss
except ImportError as e:
    raise ImportError(
        f"Import error: {e}\n"
        "Make sure ultralytics package is installed and solr_loss.py is in ultralytics/utils/"
    )


class SOLRTrainer(DetectionTrainer):
    """
    Custom trainer that integrates SOLR loss with YOLOv12 detection.
    
    This trainer extends the standard DetectionTrainer to use SOLR (Small Object
    Loss Reweighting) instead of the standard v8DetectionLoss. The integration
    is seamless - we simply replace the loss function while keeping all other
    training logic unchanged.
    
    Attributes:
        solr_weights: Dict containing SOLR weight parameters
    """
    
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        """
        Initialize SOLR trainer.
        
        Args:
            cfg: Configuration dict or path to YAML file (can be None when loading pretrained weights)
            overrides: Dict of hyperparameter overrides (can be None)
            _callbacks: Optional callbacks for training events
        """
        # CRITICAL FIX: Only initialize overrides, keep cfg as-is
        # cfg=None triggers Ultralytics to load default config (correct behavior)
        # cfg={} triggers strict validation mode (incorrect, causes SyntaxError)
        if overrides is None:
            overrides = {}
        
        # Extract SOLR parameters from overrides before calling super().__init__
        # Use pop() to remove them so parent class doesn't receive unknown params
        self.solr_weights = {
            'small_weight': overrides.pop('small_weight', 2.5),
            'medium_weight': overrides.pop('medium_weight', 2.0),
            'large_weight': overrides.pop('large_weight', 1.0),
            'small_thresh': overrides.pop('small_thresh', 32),
            'large_thresh': overrides.pop('large_thresh', 96),
        }
        
        # Pass cfg as-is (None or path), let parent handle it correctly
        super().__init__(cfg, overrides, _callbacks)
    
    def set_model_attributes(self):
        """
        Set model attributes and replace loss function with SOLR loss.
        
        This method is called during model initialization. We override it to:
        1. Call the parent method to set up the model normally
        2. Replace the standard v8DetectionLoss with SOLRDetectionLoss
        """
        # Call parent method to set up model
        super().set_model_attributes()
        
        # Replace loss function with SOLR loss
        if hasattr(self.model, 'model') and hasattr(self.model.model[-1], 'no'):
            LOGGER.info(f"\n{colorstr('SOLR:')} Integrating SOLR loss...")
            
            # Create base detection loss
            base_loss = v8DetectionLoss(self.model)
            
            # Wrap with SOLR
            self.model.criterion = SOLRDetectionLoss(
                base_loss=base_loss,
                small_weight=self.solr_weights.get('small_weight', 2.5),
                medium_weight=self.solr_weights.get('medium_weight', 2.0),
                large_weight=self.solr_weights.get('large_weight', 1.0),
                small_thresh=self.solr_weights.get('small_thresh', 32),
                large_thresh=self.solr_weights.get('large_thresh', 96),
                image_size=self.args.imgsz
            )
            
            LOGGER.info(f"{colorstr('SOLR:')} ✅ SOLR loss integrated successfully!")


def parse_args():
    """
    Parse command-line arguments for SOLR training.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train YOLOv12-RGBD with SOLR loss on RGB-D datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="ultralytics/cfg/models/12/yolo12-rgbd-v2.1-universal.yaml",
        help="Model configuration YAML file path (universal config recommended)"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="n",
        help="Model size (n/s/m/l/x), used with universal config. "
             "n=nano(~3M), s=small(~11M), m=medium(~22M), l=large(~44M), x=xlarge(~66M)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Pretrained weights path (optional, e.g., yolo12n.pt)"
    )
    
    # Dataset configuration
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Dataset YAML file (must include train_depth and val_depth for RGB-D)"
    )
    
    # Training hyperparameters (RemDet-aligned defaults)
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (width=height)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device(s), e.g. '0' or '0,1,2,3'"
    )
    
    # SOLR-specific parameters ⭐
    parser.add_argument(
        "--small_weight",
        type=float,
        default=2.5,
        help="SOLR loss weight for small objects (<small_thresh px). "
             "Increase to 3.0 if AP_s is still low."
    )
    parser.add_argument(
        "--medium_weight",
        type=float,
        default=2.0,
        help="SOLR loss weight for medium objects (small_thresh to large_thresh px). "
             "This is the KEY parameter for closing RemDet gap. Try 2.5 if AP_m improvement is insufficient."
    )
    parser.add_argument(
        "--large_weight",
        type=float,
        default=1.0,
        help="SOLR loss weight for large objects (>large_thresh px). "
             "Keep at 1.0 unless AP_l drops significantly."
    )
    parser.add_argument(
        "--small_thresh",
        type=int,
        default=32,
        help="Threshold (pixels) separating small from medium objects"
    )
    parser.add_argument(
        "--large_thresh",
        type=int,
        default=96,
        help="Threshold (pixels) separating medium from large objects"
    )
    
    # Optimizer settings (RemDet-aligned)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        choices=["SGD", "Adam", "AdamW"],
        help="Optimizer type"
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate"
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
        help="SGD momentum"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0005,
        help="Weight decay"
    )
    
    # Data augmentation (RemDet-aligned)
    parser.add_argument(
        "--mosaic",
        type=float,
        default=1.0,
        help="Mosaic augmentation probability (1.0 = always on)"
    )
    parser.add_argument(
        "--mixup",
        type=float,
        default=0.15,
        help="MixUp augmentation probability"
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
    
    # Training settings
    parser.add_argument(
        "--warmup_epochs",
        type=float,
        default=3.0,
        help="Warmup epochs"
    )
    parser.add_argument(
        "--close_mosaic",
        type=int,
        default=10,
        help="Disable mosaic in last N epochs"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Use Automatic Mixed Precision"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (epochs)"
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
        help="Cache images: '' (no cache), 'ram', or 'disk'"
    )
    
    # Output settings
    parser.add_argument(
        "--name",
        type=str,
        default="visdrone_solr",
        help="Experiment name (results saved to runs/train/<name>)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Project directory"
    )
    parser.add_argument(
        "--exist_ok",
        action="store_true",
        help="Overwrite existing project/name if exists"
    )
    parser.add_argument(
        "--save_period",
        type=int,
        default=10,
        help="Save checkpoint every N epochs (-1 to disable)"
    )
    
    # Validation settings
    parser.add_argument(
        "--val",
        action="store_true",
        default=True,
        help="Validate after each epoch"
    )
    
    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Resume training from checkpoint (e.g., runs/train/exp1/weights/last.pt)"
    )
    
    return parser.parse_args()


def main():
    """
    Main training function with SOLR loss.
    
    Workflow:
        1. Parse command-line arguments
        2. Print configuration summary
        3. Create SOLR trainer
        4. Start training with SOLR loss
        5. Print final results
    """
    args = parse_args()
    
    # Print training configuration
    print(f"\n{'='*70}")
    print(f"YOLOv12-RGBD Training with SOLR Loss")
    print(f"{'='*70}")
    print(f"\n📦 Model Configuration:")
    print(f"   Model:   {args.model}")
    print(f"   Weights: {args.weights if args.weights else 'None (training from scratch)'}")
    print(f"\n📊 Dataset:")
    print(f"   Data:    {args.data}")
    print(f"\n🎯 Training Settings:")
    print(f"   Epochs:  {args.epochs}")
    print(f"   Batch:   {args.batch}")
    print(f"   ImgSize: {args.imgsz}")
    print(f"   Device:  {args.device}")
    print(f"\n⭐ SOLR Parameters:")
    print(f"   Small weight:  {args.small_weight}x  (objects < {args.small_thresh}px)")
    print(f"   Medium weight: {args.medium_weight}x  (objects {args.small_thresh}-{args.large_thresh}px) ← KEY for RemDet gap")
    print(f"   Large weight:  {args.large_weight}x   (objects > {args.large_thresh}px)")
    print(f"\n🔧 Optimizer:")
    print(f"   Type:          {args.optimizer}")
    print(f"   Learning rate: {args.lr0} → {args.lr0 * args.lrf}")
    print(f"   Momentum:      {args.momentum}")
    print(f"   Weight decay:  {args.weight_decay}")
    print(f"\n🎨 Data Augmentation:")
    print(f"   Mosaic: {args.mosaic}")
    print(f"   MixUp:  {args.mixup}")
    print(f"\n💾 Output:")
    print(f"   Project: {args.project}")
    print(f"   Name:    {args.name}")
    print(f"{'='*70}\n")
    
    # Prepare training arguments
    train_args = {
        # Model
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device,
        
        # SOLR parameters (will be extracted in SOLRTrainer.__init__)
        'small_weight': args.small_weight,
        'medium_weight': args.medium_weight,
        'large_weight': args.large_weight,
        'small_thresh': args.small_thresh,
        'large_thresh': args.large_thresh,
        
        # Optimizer
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        
        # Augmentation
        'mosaic': args.mosaic,
        'mixup': args.mixup,
        'hsv_h': args.hsv_h,
        'hsv_s': args.hsv_s,
        'hsv_v': args.hsv_v,
        'degrees': 0.0,      # RemDet disables rotation
        'translate': 0.1,
        'scale': 0.9,
        'fliplr': 0.5,
        'flipud': 0.0,       # RemDet disables vertical flip
        
        # Training
        'warmup_epochs': args.warmup_epochs,
        'close_mosaic': args.close_mosaic,
        'amp': args.amp,
        'patience': args.patience,
        'workers': args.workers,
        'cache': args.cache,
        
        # Output
        'name': args.name,
        'project': args.project,
        'exist_ok': args.exist_ok,
        'save_period': args.save_period,
        
        # Validation
        'val': args.val,
        
        # Resume
        'resume': args.resume if args.resume else False,
    }
    
    # Create model
    if args.resume:
        # Resume from checkpoint
        LOGGER.info(f"Resuming training from {args.resume}")
        model = YOLO(args.resume)
    else:
        # Start new training
        if args.weights:
            # Load pretrained weights
            LOGGER.info(f"Loading pretrained weights from {args.weights}")
            model = YOLO(args.weights)
        else:
            # Train from scratch with universal config
            LOGGER.info(f"Training from scratch with model config: {args.model}")
            model = YOLO(args.model, task='detect')
            
            # Set model size if --cfg is specified
            if args.cfg:
                model.model_name = f"yolo12{args.cfg}"
                LOGGER.info(f"Using model size: YOLO12-{args.cfg.upper()} (with SOLR loss)")
                
                # Print expected model stats
                size_info = {
                    'n': '~3M params, ~8G FLOPs (对标RemDet-Tiny)',
                    's': '~11M params, ~46G FLOPs (对标RemDet-S)',
                    'm': '~22M params, ~92G FLOPs (对标RemDet-M)',
                    'l': '~44M params, ~184G FLOPs (对标RemDet-L)',
                    'x': '~66M params, ~276G FLOPs (对标RemDet-X)',
                }
                if args.cfg in size_info:
                    LOGGER.info(f"Expected model size: {size_info[args.cfg]}")
    
    # Train with SOLR loss
    LOGGER.info("Starting training with SOLR loss...")
    results = model.train(
        **train_args,
        trainer=SOLRTrainer  # Use custom SOLR trainer
    )
    
    # Print final results
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"\nResults saved to: {results.save_dir if hasattr(results, 'save_dir') else args.project + '/' + args.name}")
    print(f"\nNext steps:")
    print(f"  1. Check training curves: results.png")
    print(f"  2. Validate on test set: python val_coco_eval.py --weights runs/train/{args.name}/weights/best.pt")
    print(f"  3. Compare with RemDet benchmarks")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
