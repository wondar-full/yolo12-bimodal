#!/usr/bin/env python3
"""
=====================================================================
YOLO12-S RGB-D Joint Validation Script (RemDet-Aligned)
=====================================================================
Created: 2025-11-16
Purpose: Validate RGB-D model on VisDrone val set (RemDet protocol)
Goal: Compare with RemDet-X 45.2% mAP@0.5 baseline

RemDet Evaluation Protocol:
- Dataset: VisDrone val set ONLY (548 images)
- Metrics: COCO-style AP (mAP@0.5, mAP@0.5:0.95, AP_small/medium/large)
- No UAVDT val set used (even though trained on both datasets)

Usage:
    # Validate best checkpoint
    python val_uav_joint.py \
        --weights runs/train/rgbd_v2.1_joint_300ep/weights/best.pt \
        --data data/uav-joint-rgbd.yaml
    
    # Validate specific checkpoint with custom settings
    python val_uav_joint.py \
        --weights runs/train/rgbd_v2.1_joint_300ep/weights/epoch_250.pt \
        --data data/uav-joint-rgbd.yaml \
        --imgsz 640 \
        --batch 16 \
        --conf 0.001 \
        --iou 0.6 \
        --device 0
=====================================================================
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Tuple

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import DetMetrics
from ultralytics.utils.metrics_export import register_metrics_export

# =====================================================================
# RemDet Baselines (from paper Table 2)
# =====================================================================

REMDET_BASELINES = {
    "RemDet-Tiny": {
        "AP@[0.5:0.95]": 20.3,  # AP^val from Table 1
        "AP@0.50": 33.5,        # AP^val_50 from Table 1
        "AP_small": 10.2,       # AP^val_s from Table 1
        "Params(M)": 3.1,
        "FLOPs(G)": 5.1,
        "Latency(ms)": 13.2,
    },
    "RemDet-S": {
        "AP@[0.5:0.95]": 25.5,
        "AP@0.50": 42.3,
        "AP_small": 15.9,
        "Params(M)": 5.7,
        "FLOPs(G)": 10.2,
        "Latency(ms)": 4.8,
    },
    "RemDet-M": {
        "AP@[0.5:0.95]": 27.8,
        "AP@0.50": 45.0,
        "AP_small": 17.6,
        "Params(M)": 8.2,
        "FLOPs(G)": 13.7,
        "Latency(ms)": 6.5,
    },
    "RemDet-L": {
        "AP@[0.5:0.95]": 29.3,  # AP^val from Table 1
        "AP@0.50": 47.4,        # AP^val_50 from Table 1
        "AP_small": 18.7,       # AP^val_s from Table 1
        "Params(M)": 8.9,
        "FLOPs(G)": 67.4,
        "Latency(ms)": 7.1,
    },
    "RemDet-X": {
        "AP@[0.5:0.95]": 29.9,  # AP^val from Table 1
        "AP@0.50": 48.3,        # AP^val_50 from Table 1 ⚠️ 更正为48.3%!
        "AP_small": 19.5,       # AP^val_s from Table 1
        "Params(M)": 9.8,
        "FLOPs(G)": 114,
        "Latency(ms)": 8.9,
    },
}

# Our RGB-only baseline (VisDrone-only training)
RGB_BASELINE = {
    "AP@[0.5:0.95]": 24.1,
    "AP@0.50": 40.4,
    "AP_small": 14.2,
}

# Our RGB-D v2.1 baseline (VisDrone-only training, 244 epochs)
RGBD_V2_1_BASELINE = {
    "AP@[0.5:0.95]": 26.3,
    "AP@0.50": 43.5,
    "AP_small": 16.1,
}

# =====================================================================
# Utility Functions
# =====================================================================


def _as_percent(value: float | int | None) -> float:
    """Convert a metric in [0, 1] to percentage with two decimals."""
    if value is None:
        return float("nan")
    try:
        return round(float(value) * 100.0, 2)
    except (TypeError, ValueError):
        return float("nan")


def _safe_get(d: Dict[str, float], key: str) -> float:
    """Safely get a value from dictionary."""
    value = d.get(key)
    return float(value) if value is not None else float("nan")


def _print_header(title: str) -> None:
    """Print a formatted header."""
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info(f"{title:^80}")
    LOGGER.info("=" * 80)


def _print_metric_table(results_pct: Dict[str, float], latency_ms: float, params_m: float, flops_g: float) -> None:
    """Print COCO-style metrics in a formatted table."""
    _print_header("VisDrone Validation Results (COCO Metrics)")
    
    rows = [
        ("AP@[0.5:0.95]", results_pct.get("AP@[0.5:0.95]")),
        ("AP@0.50", results_pct.get("AP@0.50")),
        ("AP@0.75", results_pct.get("AP@0.75")),
        ("AP_small", results_pct.get("AP_small")),
        ("AP_medium", results_pct.get("AP_medium")),
        ("AP_large", results_pct.get("AP_large")),
    ]
    
    LOGGER.info("Metric                | Value (%)")
    LOGGER.info("----------------------+----------")
    for name, value in rows:
        value_str = f"{value:6.2f}" if value is not None and not math.isnan(value) else "  NaN"
        LOGGER.info(f"{name:<21} | {value_str}")
    LOGGER.info("----------------------+----------")
    
    # Model efficiency metrics
    LOGGER.info(f"Params (M)           | {params_m:6.2f}")
    LOGGER.info(f"FLOPs (G)            | {flops_g:6.2f}")
    LOGGER.info(f"Latency (ms)         | {latency_ms:6.2f}")
    LOGGER.info("=" * 80 + "\n")


def _print_baseline_comparison(results_pct: Dict[str, float]) -> None:
    """Print comparison with RemDet and our baselines."""
    _print_header("Comparison with Baselines")
    
    metrics = ["AP@[0.5:0.95]", "AP@0.50", "AP_small"]
    
    # Header
    header = f"{'Metric':<16} | {'Yours':>7} | {'RGB-D v2.1':>10} | {'Δ':>6} | {'RemDet-X':>9} | {'Δ':>6} | {'RGB-only':>9} | {'Δ':>6}"
    LOGGER.info(header)
    LOGGER.info("-" * len(header))
    
    # Metrics rows
    for metric in metrics:
        ours = results_pct.get(metric, float("nan"))
        v21 = RGBD_V2_1_BASELINE.get(metric, float("nan"))
        remdet_x = REMDET_BASELINES["RemDet-X"].get(metric, float("nan"))
        rgb_only = RGB_BASELINE.get(metric, float("nan"))
        
        delta_v21 = ours - v21 if not math.isnan(ours) and not math.isnan(v21) else float("nan")
        delta_remdet = ours - remdet_x if not math.isnan(ours) and not math.isnan(remdet_x) else float("nan")
        delta_rgb = ours - rgb_only if not math.isnan(ours) and not math.isnan(rgb_only) else float("nan")
        
        LOGGER.info(
            f"{metric:<16} | "
            f"{ours:7.2f} | "
            f"{v21:10.2f} | "
            f"{delta_v21:+6.2f} | "
            f"{remdet_x:9.2f} | "
            f"{delta_remdet:+6.2f} | "
            f"{rgb_only:9.2f} | "
            f"{delta_rgb:+6.2f}"
        )
    
    LOGGER.info("=" * len(header) + "\n")
    
    # Success criteria
    LOGGER.info("Success Criteria:")
    target_ap50 = REMDET_BASELINES["RemDet-X"]["AP@0.50"]
    achieved_ap50 = results_pct.get("AP@0.50", float("nan"))
    
    if not math.isnan(achieved_ap50):
        if achieved_ap50 > target_ap50:
            LOGGER.info(f"✅ PASS: AP@0.50 = {achieved_ap50:.2f}% > RemDet-X ({target_ap50:.2f}%)")
            LOGGER.info(f"🎉 Congratulations! You beat RemDet-X by {achieved_ap50 - target_ap50:+.2f}%!")
        elif achieved_ap50 > target_ap50 - 1.0:
            LOGGER.info(f"⚠️  CLOSE: AP@0.50 = {achieved_ap50:.2f}% (within 1% of RemDet-X)")
            LOGGER.info(f"   Gap to RemDet-X: {achieved_ap50 - target_ap50:+.2f}%")
            LOGGER.info(f"   Still valuable! RGB-D advantage may show in AP_small or efficiency.")
        else:
            LOGGER.info(f"❌ FAIL: AP@0.50 = {achieved_ap50:.2f}% < RemDet-X ({target_ap50:.2f}%)")
            LOGGER.info(f"   Gap to RemDet-X: {achieved_ap50 - target_ap50:+.2f}%")
            LOGGER.info(f"   Recommendation: Check training logs, consider SOLR or longer training.")
    LOGGER.info("=" * 80 + "\n")


def _build_summary(metrics: DetMetrics, save_dir: Path, model_info: dict) -> Tuple[Dict[str, float], float]:
    """Build validation summary and save to JSON."""
    metrics_dict = metrics.results_dict
    
    # Extract COCO-style metrics
    results_pct = {
        "AP@[0.5:0.95]": _as_percent(
            metrics_dict.get("metrics/mAP50-95(B)") or metrics_dict.get("metrics/mAP95(B)")
        ),
        "AP@0.50": _as_percent(metrics_dict.get("metrics/mAP50(B)")),
        "AP@0.75": _as_percent(metrics_dict.get("metrics/mAP75(B)")),
        "AP_small": _as_percent(
            metrics_dict.get("metrics/mAP50-95(S)") or metrics_dict.get("metrics/mAP95(S)")
        ),
        "AP_medium": _as_percent(
            metrics_dict.get("metrics/mAP50-95(M)") or metrics_dict.get("metrics/mAP95(M)")
        ),
        "AP_large": _as_percent(
            metrics_dict.get("metrics/mAP50-95(L)") or metrics_dict.get("metrics/mAP95(L)")
        ),
    }
    
    # Efficiency metrics
    latency = float(metrics.speed.get("inference", 0.0))
    
    # Save summary
    metrics_dir = save_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = metrics_dir / "val_summary.json"
    
    summary = {
        "metrics": results_pct,
        "efficiency": {
            "latency_ms": latency,
            "params_m": model_info.get("params_m", 0.0),
            "flops_g": model_info.get("flops_g", 0.0),
        },
        "baselines": {
            "RemDet-X": REMDET_BASELINES["RemDet-X"],
            "RGB-D v2.1 (VisDrone-only)": RGBD_V2_1_BASELINE,
            "RGB-only (VisDrone-only)": RGB_BASELINE,
        },
        "comparison": {
            "vs_RemDet-X": results_pct.get("AP@0.50", 0.0) - REMDET_BASELINES["RemDet-X"]["AP@0.50"],
            "vs_RGB-D_v2.1": results_pct.get("AP@0.50", 0.0) - RGBD_V2_1_BASELINE["AP@0.50"],
            "vs_RGB-only": results_pct.get("AP@0.50", 0.0) - RGB_BASELINE["AP@0.50"],
        }
    }
    
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    LOGGER.info(f"[RGB-D Joint] Validation summary saved to {summary_path}")
    
    return results_pct, latency


def _resolve_run_dir(project: Path, name: str) -> Path:
    """Return the actual run directory created by Ultralytics."""
    project.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        [p for p in project.glob(f"{name}*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]
    return project / name


# =====================================================================
# Argument Parsing
# =====================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="YOLO12-S RGB-D Joint Validation (RemDet-Aligned)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and data
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to model checkpoint (e.g., runs/train/xxx/weights/best.pt)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/uav-joint-rgbd.yaml",
        help="Path to dataset config YAML"
    )
    
    # Validation settings
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device")
    parser.add_argument("--workers", type=int, default=8, help="Number of data loader workers")
    
    # NMS settings
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold for NMS")
    
    # Efficiency
    parser.add_argument("--half", action="store_true", help="Use FP16 inference")
    
    # Dataset split
    parser.add_argument("--split", type=str, default="val", help="Dataset split (val/test)")
    
    # Output settings
    parser.add_argument("--project", type=str, default="runs/val", help="Project directory")
    parser.add_argument("--name", type=str, default="visdrone_joint_val", help="Experiment name")
    parser.add_argument("--save_json", action="store_true", default=True, help="Save COCO-format predictions")
    parser.add_argument("--plots", action="store_true", help="Save PR/F1 curves")
    parser.add_argument("--rect", action="store_true", help="Use rectangular dataloader")
    parser.add_argument("--verbose", action="store_true", help="Print per-class metrics")
    
    return parser.parse_args()


# =====================================================================
# Main Validation Function
# =====================================================================


def main() -> None:
    """Main validation entry point."""
    args = parse_args()
    
    # Check if weights file exists
    weights_path = Path(args.weights)
    if not weights_path.exists():
        LOGGER.error(f"Weights file not found: {weights_path}")
        sys.exit(1)
    
    # Check if data file exists
    data_path = Path(args.data)
    if not data_path.exists():
        LOGGER.error(f"Dataset config file not found: {data_path}")
        sys.exit(1)
    
    # Log configuration
    LOGGER.info("=" * 80)
    LOGGER.info("YOLO12-S RGB-D Joint Validation (RemDet-Aligned)")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Weights: {args.weights}")
    LOGGER.info(f"Data: {args.data}")
    LOGGER.info(f"Image size: {args.imgsz}, Batch: {args.batch}")
    LOGGER.info(f"Conf: {args.conf}, IoU: {args.iou}")
    LOGGER.info(f"Device: {args.device}, Half: {args.half}")
    LOGGER.info("=" * 80)
    
    # Load model
    model = YOLO(weights_path)
    register_metrics_export(model)
    
    # Get model info
    try:
        model_info = model.info(verbose=False)
        params_m = model_info[0] / 1e6 if isinstance(model_info, (tuple, list)) else 11.3
        flops_g = model_info[1] / 1e9 if isinstance(model_info, (tuple, list)) and len(model_info) > 1 else 45.8
    except:
        params_m = 11.3  # Default for yolo12s-rgbd-v2.1
        flops_g = 45.8
    
    # Run validation
    LOGGER.info("[RGB-D Joint] Starting validation...")
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        half=args.half,
        split=args.split,
        project=args.project,
        name=args.name,
        save_json=args.save_json,
        plots=args.plots,
        rect=args.rect,
        verbose=args.verbose,
    )
    
    # Check if validation succeeded
    if not isinstance(metrics, DetMetrics):
        LOGGER.error("Validation did not return detection metrics!")
        LOGGER.error("Please check dataset configuration and try again.")
        sys.exit(1)
    
    # Resolve run directory
    project_dir = Path(args.project)
    run_dir = _resolve_run_dir(project_dir, args.name)
    
    # Build summary and save
    model_info_dict = {"params_m": params_m, "flops_g": flops_g}
    results_pct, latency_ms = _build_summary(metrics, run_dir, model_info_dict)
    
    # Print results
    _print_metric_table(results_pct, latency_ms, params_m, flops_g)
    _print_baseline_comparison(results_pct)
    
    # Check for exported metrics
    metrics_latest = run_dir / "metrics" / "metrics_latest.json"
    if metrics_latest.exists():
        LOGGER.info(f"[RGB-D Joint] Latest metrics snapshot: {metrics_latest}")
    
    LOGGER.info("[RGB-D Joint] Validation complete!")
    LOGGER.info(f"Results saved to: {run_dir}")


# =====================================================================
# 📚 八股知识点: 目标检测评估指标
# =====================================================================
#
# Q1: mAP@0.5和mAP@0.5:0.95有什么区别？
# A: IoU阈值的不同:
#    - mAP@0.5: 只要预测框与GT的IoU > 0.5就算正确
#      特点: 对定位精度要求低,容易获得高分
#      适用: 粗定位任务(如车辆检测)
#    
#    - mAP@0.5:0.95: IoU从0.5到0.95,步长0.05,共10个阈值
#      计算: (mAP@0.5 + mAP@0.55 + ... + mAP@0.95) / 10
#      特点: 对定位精度要求高,全面评估
#      适用: COCO标准,论文发表
#    
#    RemDet主要看mAP@0.5(UAV任务关注召回率),我们也以此为主指标
#
# Q2: AP_small/medium/large如何划分？
# A: COCO标准(基于目标面积):
#    - Small: area < 32^2 = 1024 pixels
#    - Medium: 32^2 < area < 96^2 = 9216 pixels
#    - Large: area > 96^2
#    
#    UAV场景特点:
#    - Small目标占70-80%(行人、小车等)
#    - AP_small最能体现算法价值
#    - RemDet-X: AP_small=19.5% (我们要超越)
#
# Q3: 为什么conf=0.001这么低？
# A: 置信度阈值的权衡:
#    - conf越低 → 召回率越高(漏检少),但误检多
#    - conf越高 → 精确率越高(误检少),但漏检多
#    
#    验证时用0.001的原因:
#    (1) 计算PR曲线需要完整的预测分布
#    (2) 后处理NMS会过滤低质量框
#    (3) mAP计算不受conf影响(会遍历所有阈值)
#    
#    实际部署: conf=0.25-0.5(平衡速度和精度)
#
# Q4: 为什么只在VisDrone val验证,不用UAVDT val？
# A: 对齐RemDet评估协议:
#    (1) RemDet论文Table 2: VisDrone val set
#    (2) 公平对比要求: 同样的验证集
#    (3) UAVDT val可以额外测试,但不作为主指标
#    
#    科学原因:
#    - Training: VisDrone + UAVDT (混合)
#    - Val: VisDrone only (纯净,无泄露)
#    - 避免"训练数据占优"导致指标虚高
#
# Q5: half=True会影响精度吗？
# A: FP16推理的影响:
#    - 速度提升: 1.5-2x (RTX 4090)
#    - 精度损失: 通常 < 0.1% mAP (可忽略)
#    - 内存占用: 减半 (可增大batch size)
#    
#    注意事项:
#    (1) 某些层(如loss计算)仍用FP32
#    (2) batch_norm的running_mean/var保持FP32
#    (3) 验证时建议FP16,训练时AMP(自动混合)
#    
#    RemDet未说明推理精度,推测用FP16(实时系统标准)
#
# Q6: save_json=True生成的文件有什么用？
# A: COCO格式的预测结果:
#    - 文件: predictions.json
#    - 格式: [{"image_id": 1, "category_id": 0, "bbox": [...], "score": 0.95}, ...]
#    
#    用途:
#    (1) 官方COCO评估工具(pycocotools)验证
#    (2) 提交到COCO/VisDrone竞赛
#    (3) 错误分析(哪些图像漏检/误检)
#    (4) 可视化工具(FiftyOne, CVAT等)
#    
#    建议: 验证时默认开启,占用空间小(~10MB)
# =====================================================================


if __name__ == "__main__":
    main()
