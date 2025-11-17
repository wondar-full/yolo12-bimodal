#!/usr/bin/env python3
"""
=====================================================================
YOLO12-S RGB-D Joint Training Script (VisDrone + UAVDT)
=====================================================================
Created: 2025-11-16
Purpose: Train RGB-D model on VisDrone+UAVDT to align with RemDet paper
Goal: Surpass RemDet-X 45.2% mAP@0.5 with dual-modal fusion

RemDet Training Protocol (from paper):
- Datasets: VisDrone (6,471) + UAVDT (23,258) = 29,729 images
- Method: From scratch (NO COCO pretraining)
- Epochs: 300
- Optimizer: SGD (lr=0.01, momentum=0.937, weight_decay=0.0005)
- Data augmentation: Mosaic=1.0, MixUp=0.15
- Batch size: 128 (total across all GPUs)
- Evaluation: VisDrone val set only

Usage:
    # Full 300-epoch training (single GPU)
    python train_uav_joint.py --device 0 --batch 16
    
    # Multi-GPU training (8 GPUs, batch=16 per GPU = 128 total)
    python -m torch.distributed.run --nproc_per_node 8 train_uav_joint.py --batch 16
    
    # Quick test (10 epochs)
    python train_uav_joint.py --device 0 --batch 8 --epochs 10 --name test_joint
=====================================================================
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Iterable

from ultralytics.models import YOLO
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model

# =====================================================================
# Training Configuration Constants (RemDet-Aligned)
# =====================================================================

# Positive sample assignment warmup
TAL_TOPK_WARMUP_EPOCHS = 20  # First 20 epochs use relaxed topk
TAL_TOPK_HIGH = 12  # Relaxed topk value during warmup
TAL_TOPK_BASE = 10  # Default topk after warmup

# Loss gain scheduling (focus on localization after warmup)
BOX_GAIN_RAMP_START = 40  # Start increasing box loss weight at epoch 40
BOX_GAIN_RAMP_EPOCHS = 40  # Ramp up over 40 epochs
BOX_GAIN_TARGET = 8.5  # Target box loss gain
DFL_GAIN_TARGET = 1.7  # Target DFL loss gain

# Monitoring
MONITOR_FILENAME = "rgbd_joint_monitor.csv"  # Training metrics log

# =====================================================================
# Utility Functions
# =====================================================================


def _as_float(value) -> float:
    """Convert tensor-like objects to float on CPU without breaking gradients."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean of valid values or NaN if empty."""
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def _safe_extrema(values: Iterable[float]) -> tuple[float, float]:
    """Return the min and max of valid values or (NaN, NaN) if empty."""
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return float("nan"), float("nan")
    return min(vals), max(vals)


def _get_detection_loss(trainer):
    """Get the detection loss module from trainer."""
    model = unwrap_model(trainer.model)
    loss_module = getattr(model, "criterion", None)
    if loss_module is not None:
        return loss_module
    if hasattr(model, "init_criterion"):
        loss_module = model.init_criterion()
        setattr(model, "criterion", loss_module)
        return loss_module
    return None


def _collect_gate_stats(trainer) -> tuple[list[float], list[float]]:
    """Collect gating statistics from RGBDStem/RGBDMidFusion modules."""
    model = unwrap_model(trainer.model)
    means, stds = [], []
    for module in model.modules():
        gate_mean = getattr(module, "last_gate_mean", None)
        gate_std = getattr(module, "last_gate_std", None)
        if gate_mean is not None and gate_std is not None:
            means.append(_as_float(gate_mean))
            stds.append(_as_float(gate_std))
    return means, stds


def _collect_geometry_stats(trainer) -> tuple[list[float], list[float]]:
    """Collect geometry prior statistics from depth modules."""
    model = unwrap_model(trainer.model)
    qualities, edges = [], []
    for module in model.modules():
        geo_q = getattr(module, "last_geo_quality", None)
        geo_e = getattr(module, "last_geo_edge", None)
        if geo_q is not None and geo_e is not None:
            qualities.append(_as_float(geo_q))
            edges.append(_as_float(geo_e))
    return qualities, edges


# =====================================================================
# Training Schedule Callbacks (RemDet-Aligned)
# =====================================================================


def schedule_assignment(trainer) -> None:
    """
    Adjust Task-Aligned Assigner topk during training.
    
    Strategy:
    - Epochs 0-19: topk=12 (relaxed, more positive samples for RGB-D warmup)
    - Epochs 20+: topk=10 (default, stricter assignment)
    
    Rationale: Dual-modal fusion needs more positive samples initially
    to learn effective gating weights.
    """
    loss_module = _get_detection_loss(trainer)
    if loss_module is None or not hasattr(loss_module, "assigner"):
        return
    
    # Store base topk on first call
    if not hasattr(schedule_assignment, "_base_topk"):
        schedule_assignment._base_topk = getattr(loss_module.assigner, "topk", TAL_TOPK_BASE)
    
    # Get custom args from model
    custom_args = getattr(trainer.model, "custom_args", None)
    warmup_epochs = getattr(custom_args, "tal_topk_warmup", TAL_TOPK_WARMUP_EPOCHS) if custom_args else TAL_TOPK_WARMUP_EPOCHS
    high_topk = getattr(custom_args, "tal_topk_high", TAL_TOPK_HIGH) if custom_args else TAL_TOPK_HIGH
    
    # Use high topk during warmup, then revert to base
    target_topk = high_topk if trainer.epoch < warmup_epochs else schedule_assignment._base_topk
    
    if loss_module.assigner.topk != target_topk:
        loss_module.assigner.topk = target_topk
        LOGGER.debug(f"[RGB-D Joint] Set TAL topk to {target_topk} at epoch {trainer.epoch}")


def adjust_loss_gains(trainer) -> None:
    """
    Gradually increase box and DFL loss gains after initial training.
    
    Strategy:
    - Epochs 0-39: Use default gains (box=7.5, dfl=1.5)
    - Epochs 40-79: Linear ramp to target gains (box=8.5, dfl=1.7)
    - Epochs 80+: Keep target gains
    
    Rationale: Early training focuses on feature learning (classification),
    later training refines localization (box regression).
    """
    loss_module = _get_detection_loss(trainer)
    if loss_module is None or not hasattr(loss_module, "hyp"):
        return
    
    # Store base gains on first call
    if not hasattr(adjust_loss_gains, "_base_box"):
        adjust_loss_gains._base_box = float(loss_module.hyp.box)
        adjust_loss_gains._base_dfl = float(loss_module.hyp.dfl)
    
    # Get custom args from model
    custom_args = getattr(trainer.model, "custom_args", None)
    ramp_start = getattr(custom_args, "box_gain_ramp_start", BOX_GAIN_RAMP_START) if custom_args else BOX_GAIN_RAMP_START
    ramp_epochs = max(1, getattr(custom_args, "box_gain_ramp_epochs", BOX_GAIN_RAMP_EPOCHS) if custom_args else BOX_GAIN_RAMP_EPOCHS)
    target_box = getattr(custom_args, "box_gain_target", BOX_GAIN_TARGET) if custom_args else BOX_GAIN_TARGET
    target_dfl = getattr(custom_args, "dfl_gain_target", DFL_GAIN_TARGET) if custom_args else DFL_GAIN_TARGET
    
    if trainer.epoch < ramp_start:
        # Before ramp: use base gains
        loss_module.hyp.box = adjust_loss_gains._base_box
        loss_module.hyp.dfl = adjust_loss_gains._base_dfl
        return
    
    # During ramp: linear interpolation
    progress = min(1.0, (trainer.epoch - ramp_start + 1) / ramp_epochs)
    loss_module.hyp.box = adjust_loss_gains._base_box + (target_box - adjust_loss_gains._base_box) * progress
    loss_module.hyp.dfl = adjust_loss_gains._base_dfl + (target_dfl - adjust_loss_gains._base_dfl) * progress


# =====================================================================
# Monitoring Callback (Export Metrics to CSV)
# =====================================================================


def log_rgbd_monitor(trainer) -> None:
    """
    Log comprehensive training metrics to CSV for offline analysis.
    
    Metrics logged:
    - Loss components (box, cls, dfl)
    - Assignment statistics (fg_ratio, topk)
    - RGB-D fusion stats (gate_mean, gate_std)
    - Geometry prior stats (quality, edge)
    - SOLR stats (if enabled)
    - Validation metrics (mAP@0.5, mAP_small, etc.)
    """
    if RANK not in {-1, 0}:  # Only log on main process
        return
    
    loss_module = _get_detection_loss(trainer)
    if loss_module is None:
        return
    
    # Collect RGB-D fusion statistics
    gate_means, gate_stds = _collect_gate_stats(trainer)
    geo_qualities, geo_edges = _collect_geometry_stats(trainer)
    geo_quality_min, geo_quality_max = _safe_extrema(geo_qualities)
    geo_edge_min, geo_edge_max = _safe_extrema(geo_edges)
    
    # Extract loss components
    loss_items = getattr(loss_module, "last_loss_items", (float("nan"),) * 3)
    box_loss, cls_loss, dfl_loss = (float(x) for x in loss_items)
    loss_ratio = box_loss / max(cls_loss, 1e-6) if not math.isnan(box_loss) and not math.isnan(cls_loss) else float("nan")
    
    # SOLR statistics (if enabled)
    solr_bins = getattr(loss_module, "last_solr_bins", (float("nan"),) * 3)
    if isinstance(solr_bins, (tuple, list)) and len(solr_bins) == 3:
        solr_small_count, solr_medium_count, solr_large_count = solr_bins
    else:
        solr_small_count = solr_medium_count = solr_large_count = float("nan")
    solr_avg_weight = _as_float(getattr(loss_module, "last_solr_avg_weight", float("nan")))
    solr_weighted_sum = _as_float(getattr(loss_module, "last_solr_weighted_sum", float("nan")))
    target_scores_sum = _as_float(getattr(loss_module, "last_target_scores_sum", float("nan")))
    
    # Prepare metrics row
    monitor_dir = Path(trainer.save_dir) / "metrics"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    monitor_path = monitor_dir / MONITOR_FILENAME
    
    row = {
        "epoch": trainer.epoch + 1,
        "fg_ratio": getattr(loss_module, "last_fg_ratio", float("nan")),
        "fg_count": getattr(loss_module, "last_fg_count", 0),
        "anchor_count": getattr(loss_module, "last_anchor_count", 0),
        "tal_topk": getattr(getattr(loss_module, "assigner", None), "topk", float("nan")),
        "box_gain": getattr(loss_module.hyp, "box", float("nan")),
        "dfl_gain": getattr(loss_module.hyp, "dfl", float("nan")),
        "loss_box": box_loss,
        "loss_cls": cls_loss,
        "loss_dfl": dfl_loss,
        "loss_box_cls_ratio": loss_ratio,
        "solr_enabled": bool(getattr(loss_module, "solr_enabled", False)),
        "solr_small_thresh": _as_float(getattr(loss_module, "solr_small", float("nan"))),
        "solr_medium_thresh": _as_float(getattr(loss_module, "solr_medium", float("nan"))),
        "solr_small_gain": _as_float(getattr(loss_module, "solr_small_gain", float("nan"))),
        "solr_medium_gain": _as_float(getattr(loss_module, "solr_medium_gain", float("nan"))),
        "solr_large_gain": _as_float(getattr(loss_module, "solr_large_gain", float("nan"))),
        "solr_small_count": solr_small_count,
        "solr_medium_count": solr_medium_count,
        "solr_large_count": solr_large_count,
        "solr_avg_weight": solr_avg_weight,
        "solr_weighted_scores_sum": solr_weighted_sum,
        "target_scores_sum": target_scores_sum,
        "gate_mean": _safe_mean(gate_means),
        "gate_std": _safe_mean(gate_stds),
        "gate_min": min(gate_means) if gate_means else float("nan"),
        "gate_max": max(gate_means) if gate_means else float("nan"),
        "geo_quality_mean": _safe_mean(geo_qualities),
        "geo_quality_min": geo_quality_min,
        "geo_quality_max": geo_quality_max,
        "geo_edge_mean": _safe_mean(geo_edges),
        "geo_edge_min": geo_edge_min,
        "geo_edge_max": geo_edge_max,
        "metrics/mAP95(B)": trainer.metrics.get("metrics/mAP95(B)", float("nan")) if trainer.metrics else float("nan"),
        "metrics/mAP95(S)": trainer.metrics.get("metrics/mAP95(S)", float("nan")) if trainer.metrics else float("nan"),
        "metrics/mAP95(M)": trainer.metrics.get("metrics/mAP95(M)", float("nan")) if trainer.metrics else float("nan"),
        "metrics/mAP95(L)": trainer.metrics.get("metrics/mAP95(L)", float("nan")) if trainer.metrics else float("nan"),
        "metrics/mAP50(B)": trainer.metrics.get("metrics/mAP50(B)", float("nan")) if trainer.metrics else float("nan"),
    }
    
    # Write to CSV (append mode, create header if new file)
    write_header = not monitor_path.exists()
    with monitor_path.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# =====================================================================
# Main Training Function
# =====================================================================


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="YOLO12-S RGB-D Joint Training (VisDrone + UAVDT)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and data
    parser.add_argument(
        "--model",
        type=str,
        default="/data2/user/2024/lzy/yolo12-bimodal/ultralytics/cfg/models/12/yolo12s-rgbd-v1.yaml",
        help="Path to model config YAML"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="",
        help="Model size (n/s/m/l/x), only used with universal config"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/uav-joint-rgbd.yaml",
        help="Path to dataset config YAML"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="",
        help="Path to pretrained weights (empty = from scratch)"
    )
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=300, help="Total training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device (e.g., '0' or '0,1,2,3')")
    parser.add_argument("--workers", type=int, default=8, help="Number of data loader workers")
    
    # Optimizer (RemDet-aligned)
    parser.add_argument("--optimizer", type=str, default="SGD", help="Optimizer type")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.937, help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Warmup epochs")
    parser.add_argument("--cos_lr", action="store_true", default=True, help="Use cosine LR scheduler")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final LR factor (for cosine)")
    
    # Data augmentation (RemDet-aligned)
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation probability")
    parser.add_argument("--mixup", type=float, default=0.15, help="MixUp augmentation probability")
    parser.add_argument("--close_mosaic", type=int, default=10, help="Disable mosaic in last N epochs")
    
    # Mixed precision and efficiency
    parser.add_argument("--amp", action="store_true", default=True, help="Use AMP (automatic mixed precision)")
    parser.add_argument("--cache", action="store_true", default=False, help="Cache images in RAM")
    
    # Experiment management
    parser.add_argument("--project", type=str, default="runs/train", help="Project directory")
    parser.add_argument("--name", type=str, default="rgbd_v2.1_joint_300ep", help="Experiment name")
    parser.add_argument("--exist_ok", action="store_true", help="Allow overwriting existing experiment")
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint path")
    parser.add_argument("--save_period", type=int, default=50, help="Save checkpoint every N epochs")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience")
    
    # RGB-D specific
    parser.add_argument("--solr", action="store_true", default=False, help="Enable SOLR (size-oriented loss reweighting)")
    
    # Advanced training schedules
    parser.add_argument("--tal_topk_warmup", type=int, default=TAL_TOPK_WARMUP_EPOCHS, help="TAL topk warmup epochs")
    parser.add_argument("--tal_topk_high", type=int, default=TAL_TOPK_HIGH, help="TAL topk during warmup")
    parser.add_argument("--box_gain_ramp_start", type=int, default=BOX_GAIN_RAMP_START, help="Epoch to start box gain ramp")
    parser.add_argument("--box_gain_ramp_epochs", type=int, default=BOX_GAIN_RAMP_EPOCHS, help="Box gain ramp duration")
    parser.add_argument("--box_gain_target", type=float, default=BOX_GAIN_TARGET, help="Target box loss gain")
    parser.add_argument("--dfl_gain_target", type=float, default=DFL_GAIN_TARGET, help="Target DFL loss gain")
    
    return parser.parse_args()


def main():
    """Main training entry point."""
    # Avoid OpenMP initialization error (common in some Conda environments)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    # Parse arguments
    args = parse_args()
    
    # Log configuration
    LOGGER.info("=" * 70)
    LOGGER.info("YOLO12-S RGB-D Joint Training (VisDrone + UAVDT)")
    LOGGER.info("=" * 70)
    LOGGER.info(f"Model: {args.model}")
    LOGGER.info(f"Data: {args.data}")
    LOGGER.info(f"Pretrained: {args.pretrained if args.pretrained else 'None (from scratch)'}")
    LOGGER.info(f"Epochs: {args.epochs}, Batch: {args.batch}, Image size: {args.imgsz}")
    LOGGER.info(f"Optimizer: {args.optimizer}, LR: {args.lr0}, Momentum: {args.momentum}")
    LOGGER.info(f"Device: {args.device}, Workers: {args.workers}")
    LOGGER.info(f"Mosaic: {args.mosaic}, MixUp: {args.mixup}")
    LOGGER.info(f"SOLR: {args.solr}")
    LOGGER.info("=" * 70)
    
    # Check if model file exists
    model_path = Path(args.model)
    if not model_path.exists():
        LOGGER.error(f"Model config file not found: {args.model}")
        LOGGER.error("Please check the path and try again.")
        sys.exit(1)
    
    # Check if data file exists
    data_path = Path(args.data)
    if not data_path.exists():
        LOGGER.error(f"Dataset config file not found: {args.data}")
        LOGGER.error("Please check the path and try again.")
        sys.exit(1)
    
    # Initialize model
    if args.pretrained:
        # Load from pretrained weights
        model = YOLO(args.pretrained)
        LOGGER.info(f"Loaded pretrained weights from {args.pretrained}")
    else:
        # Load from config (from scratch)
        if args.cfg:
            # Use cfg parameter to specify model size (n/s/m/l/x)
            model = YOLO(args.model, task='detect')
            model.model_name = f"yolo12{args.cfg}"
            LOGGER.info(f"Training YOLO12-{args.cfg.upper()} from scratch (no pretrained weights)")
        else:
            model = YOLO(args.model)
            LOGGER.info("Training from scratch (no pretrained weights)")
    
    # Register training callbacks
    model.add_callback('on_train_epoch_start', schedule_assignment)
    model.add_callback('on_train_epoch_start', adjust_loss_gains)
    model.add_callback('on_fit_epoch_end', log_rgbd_monitor)
    
    # Calculate nominal batch size for gradient accumulation
    # RemDet uses total batch=128 across 8 GPUs (16 per GPU)
    # We use nbs=128 to maintain equivalent gradient accumulation
    nbs = 128  # Nominal batch size for weight decay scaling
    
    # Store custom args as model attributes (for callbacks to access)
    model.custom_args = args
    
    # Start training
    results = model.train(
        # Data configuration
        data=args.data,
        
        # Training schedule
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        nbs=nbs,  # Nominal batch size for gradient accumulation
        
        # Optimizer configuration (RemDet-aligned)
        optimizer=args.optimizer,
        lr0=args.lr0,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        cos_lr=args.cos_lr,
        lrf=args.lrf,
        
        # Data augmentation (RemDet-aligned)
        mosaic=args.mosaic,
        mixup=args.mixup,
        close_mosaic=args.close_mosaic,
        
        # Hardware and efficiency
        device=args.device,
        workers=args.workers,
        cache=args.cache,
        amp=args.amp,
        
        # Experiment management
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        resume=args.resume,
        save_period=args.save_period,
        patience=args.patience,
    )
    
    # Log final results
    LOGGER.info("=" * 70)
    LOGGER.info("Training completed!")
    LOGGER.info("=" * 70)
    if results and hasattr(results, 'results_dict'):
        LOGGER.info(f"Final metrics: {results.results_dict}")
    LOGGER.info(f"Results saved to: {model.trainer.save_dir}")
    LOGGER.info(f"Monitor CSV: {model.trainer.save_dir}/metrics/{MONITOR_FILENAME}")
    LOGGER.info("=" * 70)


# =====================================================================
# 📚 八股知识点: 多数据集联合训练
# =====================================================================
#
# Q1: VisDrone和UAVDT数据量不平衡(6,471 vs 23,258),如何采样？
# A: Ultralytics默认按数据集大小比例采样:
#    - UAVDT采样概率: 23,258 / (6,471 + 23,258) = 78.2%
#    - VisDrone采样概率: 6,471 / 29,729 = 21.8%
#    
#    可选策略:
#    - 均衡采样: 修改dataset.py,每个数据集50%概率
#    - 加权采样: 在YAML中设置dataset_weights=[2.0, 1.0]
#    
#    RemDet论文未说明,推测用默认比例采样(最简单)
#
# Q2: 为什么nbs=128而不是直接设batch=128？
# A: Ultralytics的梯度累积策略:
#    - batch: 实际batch size per GPU (受显存限制)
#    - nbs: 名义batch size (用于weight decay缩放)
#    - accumulate = nbs / batch = 128 / 16 = 8 steps
#    
#    RemDet: 8 GPUs × 16 batch = 128 total
#    单GPU模拟: 1 GPU × 16 batch × 8 accumulate = 128 equivalent
#    
#    为什么重要? weight_decay需要根据effective batch调整
#
# Q3: 为什么from scratch而不是ImageNet预训练？
# A: UAV场景的特点:
#    (1) 俯视视角 vs ImageNet的平视视角 - domain gap大
#    (2) 小目标居多 vs ImageNet的大目标 - 尺度分布不同
#    (3) RemDet实验证明: from scratch效果更好
#    
#    ImageNet预训练的陷阱:
#    - 低层特征(边缘/纹理)可以迁移
#    - 高层特征(语义)不适配 → 反而限制学习
#    
#    From scratch的优势:
#    - 自由学习UAV特定特征(俯视角度、小目标特征等)
#    - 不受ImageNet数据分布偏见影响
#
# Q4: cos_lr=True + lrf=0.01的作用？
# A: 余弦退火学习率调度器:
#    - 起始LR: lr0 = 0.01
#    - 结束LR: lr0 * lrf = 0.01 * 0.01 = 0.0001
#    - 曲线形状: 余弦衰减(先快后慢)
#    
#    RemDet用"Flat-Cosine":
#    - 前期保持高LR(快速收敛)
#    - 后期平滑衰减(精细调优)
#    - lrf=0.01模拟"平坦"效果(不会降到太低)
#
# Q5: 为什么mosaic=1.0而mixup=0.15？
# A: 数据增强的权衡:
#    - Mosaic=1.0: 每个batch都用mosaic(4张图拼接)
#      好处: 增加小目标密度,模拟遮挡
#      代价: 训练速度降低~20%
#    
#    - MixUp=0.15: 15%概率混合两张图
#      好处: 提升泛化能力,减少过拟合
#      代价: 可能模糊边界框(UAV场景不利)
#    
#    RemDet经验: Mosaic重要(UAV小目标),MixUp适度即可
#
# Q6: close_mosaic=10的目的？
# A: 训练末期关闭Mosaic增强:
#    - 前290 epochs: mosaic=1.0 (强增强)
#    - 后10 epochs: mosaic=0.0 (原图训练)
#    
#    原因: Mosaic会"破坏"图像原始分布
#    - 模型在拼接图上学习,但测试是原图
#    - 末期用原图fine-tune → 减少train-test gap
#    
#    类似ImageNet训练的"减弱增强"策略
#
# Q7: patience=100是否太大？
# A: Early stopping的耐心:
#    - patience=100: 连续100 epochs无mAP提升才停止
#    - RemDet训练300 epochs,未提及early stopping
#    
#    为什么设这么大?
#    (1) 大数据集(29K图)需要更长收敛
#    (2) 学习率余弦衰减 → 后期LR很低,提升缓慢
#    (3) 避免误判: 可能在第250 epoch突然提升
#    
#    实际建议: 300 epoch固定训练,不用early stopping
# =====================================================================


if __name__ == '__main__':
    main()
