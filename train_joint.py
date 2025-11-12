#!/usr/bin/env python3
"""
VisDrone + UAVDT 联合训练脚本

目标: 通过多数据集联合训练超越RemDet论文性能
- VisDrone: 6,471张 (小目标丰富)
- UAVDT: 23,829张 (大目标丰富)
- 总计: 30,300张训练图像

预期性能:
  Overall mAP: 45-47% (vs RemDet 38.9%, +15%)
  Small mAP:   35-38% (vs RemDet 12.7%, +180%)
  Medium mAP:  48-50% (vs RemDet 33.0%, +50%)
  Large mAP:   42-45% (vs RemDet 44.5%, 持平)

使用方法:
  # 服务器训练 (推荐)
  CUDA_VISIBLE_DEVICES=7 nohup python train_joint.py > train_joint.log 2>&1 &
  
  # 监控进度
  tail -f train_joint.log
  
  # 查看TensorBoard
  tensorboard --logdir runs/train/exp_joint_v1

作者: yoloDepth Team
日期: 2025-11-02
版本: v1.0
"""

from ultralytics import YOLO
from pathlib import Path
import torch
import sys

def main():
    """VisDrone+UAVDT联合训练主函数"""
    
    print("=" * 80)
    print("VisDrone + UAVDT 联合数据集训练")
    print("=" * 80)
    print()
    
    # ========================================================================
    # 配置参数
    # ========================================================================
    
    # 模型配置
    MODEL_YAML = "ultralytics/cfg/models/12/yolo12n-rgbd.yaml"  # RGB-D双模态
    PRETRAINED_WEIGHTS = "yolo12n.pt"  # ImageNet预训练权重
    
    # 数据配置
    DATA_YAML = "data/visdrone_uavdt_joint.yaml"  # 联合数据集配置
    
    # 训练参数
    EPOCHS = 300
    BATCH_SIZE = 16
    IMGSZ = 640
    DEVICE = 0  # GPU 0
    WORKERS = 8
    
    # 优化器设置
    OPTIMIZER = "AdamW"
    LR0 = 0.001       # 初始学习率
    LRF = 0.01        # 最终学习率倍数 (lr0 * lrf)
    MOMENTUM = 0.937
    WEIGHT_DECAY = 0.0005
    WARMUP_EPOCHS = 3
    CLOSE_MOSAIC = 10  # 最后N个epoch关闭mosaic
    
    # 数据增强 (UAV场景优化)
    HSV_H = 0.015      # 色调增强
    HSV_S = 0.7        # 饱和度增强
    HSV_V = 0.4        # 亮度增强
    DEGREES = 0.0      # 旋转 (UAV场景禁用)
    TRANSLATE = 0.1    # 平移
    SCALE = 0.5        # 缩放
    FLIPUD = 0.0       # 上下翻转 (UAV场景禁用)
    FLIPLR = 0.5       # 左右翻转
    MOSAIC = 1.0       # Mosaic增强
    MIXUP = 0.0        # Mixup (禁用)
    
    # 实验设置
    PROJECT = "runs/train"
    NAME = "exp_joint_v1"
    SAVE_PERIOD = 50   # 每50个epoch保存一次
    PATIENCE = 100     # 早停耐心值
    
    # ========================================================================
    # 环境检查
    # ========================================================================
    
    print("🔍 环境检查...")
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA不可用! 请检查GPU驱动和PyTorch安装")
        sys.exit(1)
    print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
    print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 检查模型配置
    model_path = Path(MODEL_YAML)
    if not model_path.exists():
        print(f"❌ 模型配置不存在: {MODEL_YAML}")
        sys.exit(1)
    print(f"✅ 模型配置: {MODEL_YAML}")
    
    # 检查数据配置
    data_path = Path(DATA_YAML)
    if not data_path.exists():
        print(f"❌ 数据配置不存在: {DATA_YAML}")
        sys.exit(1)
    print(f"✅ 数据配置: {DATA_YAML}")
    
    # 检查预训练权重
    weights_path = Path(PRETRAINED_WEIGHTS)
    if not weights_path.exists():
        print(f"⚠️  预训练权重不存在: {PRETRAINED_WEIGHTS}")
        print(f"   将从零开始训练")
        PRETRAINED_WEIGHTS = MODEL_YAML  # 从模型配置开始
    else:
        print(f"✅ 预训练权重: {PRETRAINED_WEIGHTS}")
    
    print()
    
    # ========================================================================
    # 加载模型
    # ========================================================================
    
    print("🔧 加载模型...")
    model = YOLO(PRETRAINED_WEIGHTS)
    print("✅ 模型加载成功")
    print()
    
    # ========================================================================
    # 训练配置总结
    # ========================================================================
    
    print("📋 训练配置:")
    print(f"  模型架构:      {MODEL_YAML}")
    print(f"  预训练权重:    {PRETRAINED_WEIGHTS}")
    print(f"  数据集:        {DATA_YAML}")
    print(f"  训练数据量:    30,300张 (VisDrone 6,471 + UAVDT 23,829)")
    print(f"  验证数据量:    548张 (仅VisDrone,对齐RemDet评估)")
    print(f"  Epoch数:       {EPOCHS}")
    print(f"  Batch Size:    {BATCH_SIZE}")
    print(f"  图像尺寸:      {IMGSZ}")
    print(f"  设备:          GPU {DEVICE}")
    print(f"  优化器:        {OPTIMIZER}")
    print(f"  学习率:        {LR0} → {LR0 * LRF}")
    print(f"  保存路径:      {PROJECT}/{NAME}")
    print()
    
    # ========================================================================
    # 性能目标
    # ========================================================================
    
    print("🎯 性能目标:")
    print()
    print("  RemDet-Tiny基线 (AAAI2025):")
    print("    Overall mAP@0.5:  38.9%")
    print("    Small mAP:        12.7%")
    print("    Medium mAP:       33.0%")
    print("    Large mAP:        44.5%")
    print()
    print("  我们的目标 (VisDrone+UAVDT):")
    print("    Overall mAP@0.5:  45-47% (+15%)")
    print("    Small mAP:        35-38% (+180%)")
    print("    Medium mAP:       48-50% (+50%)")
    print("    Large mAP:        42-45% (持平)")
    print()
    print("  成功标准:")
    print("    ✅ 最低:  Overall ≥45%, Small ≥35%, Medium ≥48%")
    print("    ✅ 目标:  Overall ≥46%, Small ≥36%, Medium ≥49%")
    print("    ✅ 优秀:  Overall ≥47%, Small ≥38%, Medium ≥50%")
    print()
    
    # ========================================================================
    # 数据集优势分析
    # ========================================================================
    
    print("💪 联合数据集优势:")
    print()
    print("  VisDrone贡献:")
    print("    - Small目标丰富: 92.4% (碾压性优势)")
    print("    - 类别全面: 10类 (pedestrian, car, van, truck, bus...)")
    print("    - 场景多样: 城市、乡村、高速公路")
    print()
    print("  UAVDT贡献:")
    print("    - Large目标丰富: 5,900个 (vs VisDrone仅443个,13倍提升!)")
    print("    - Medium目标丰富: 48.4% (vs VisDrone 7.5%, 6倍提升!)")
    print("    - 数据量大: 23,829张 (扩大训练规模)")
    print()
    print("  互补性:")
    print("    VisDrone (Small) + UAVDT (Medium/Large) = 全尺度覆盖!")
    print()
    
    # ========================================================================
    # 开始训练
    # ========================================================================
    
    print("🚀 开始训练...")
    print("=" * 80)
    print()
    
    results = model.train(
        # 数据配置
        data=DATA_YAML,
        
        # 训练时长
        epochs=EPOCHS,
        
        # 批处理设置
        batch=BATCH_SIZE,
        imgsz=IMGSZ,
        
        # 设备和工作线程
        device=DEVICE,
        workers=WORKERS,
        
        # 优化器
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        close_mosaic=CLOSE_MOSAIC,
        
        # 数据增强
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        flipud=FLIPUD,
        fliplr=FLIPLR,
        mosaic=MOSAIC,
        mixup=MIXUP,
        
        # 保存设置
        project=PROJECT,
        name=NAME,
        exist_ok=False,
        save_period=SAVE_PERIOD,
        patience=PATIENCE,
        
        # 日志设置
        verbose=True,
        plots=True,
        save=True,
        
        # 其他设置
        amp=True,        # 自动混合精度
        deterministic=False,
        val=True,        # 每个epoch后验证
    )
    
    print()
    print("=" * 80)
    print("✅ 训练完成!")
    print("=" * 80)
    print()
    
    # ========================================================================
    # 训练后总结
    # ========================================================================
    
    best_weights = Path(PROJECT) / NAME / "weights" / "best.pt"
    last_weights = Path(PROJECT) / NAME / "weights" / "last.pt"
    
    print("📊 训练结果:")
    print(f"  最佳权重:  {best_weights}")
    print(f"  最终权重:  {last_weights}")
    print()
    
    print("🔍 下一步操作:")
    print()
    print("  1. 验证性能:")
    print(f"     python val_depth.py \\")
    print(f"       --weights {best_weights} \\")
    print(f"       --data {DATA_YAML} \\")
    print(f"       --batch 16")
    print()
    print("  2. 对比RemDet:")
    print("     - 查看 runs/train/exp_joint_v1/results.csv")
    print("     - 重点关注: mAP@0.5, mAP_small, mAP_medium, mAP_large")
    print()
    print("  3. 判断结果:")
    print("     - Overall ≥45%: ✅ 成功超越RemDet!")
    print("     - Small ≥35%:   ✅ 碾压RemDet (12.7% → 35%+)")
    print("     - Medium ≥48%:  ✅ 大幅超越RemDet (33.0% → 48%+)")
    print("     - Large ≥40%:   ✅ 接近RemDet水平")
    print()
    print("  4. 如果性能不理想:")
    print("     - 检查训练日志中的loss曲线")
    print("     - 查看是否过拟合 (train mAP >> val mAP)")
    print("     - 考虑调整数据增强或学习率")
    print("     - 尝试加权采样策略平衡VisDrone和UAVDT")
    print()
    
    print("=" * 80)
    print("🎉 祝您训练顺利,超越RemDet!")
    print("=" * 80)


if __name__ == "__main__":
    main()
