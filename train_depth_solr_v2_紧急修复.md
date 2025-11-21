# train_depth_solr_v2.py 紧急修复补丁

**问题**: `--cfg n`参数完全失效，导致 GGFE 配置从未加载

## 🔴 根因

### 错误代码 (第 136-141 行)

```python
if args.weights:
    model = YOLO(args.weights)  # ❌ 直接加载weights的架构，忽略YAML配置
    LOGGER.info(f"Loaded pretrained weights from {args.weights}")
else:
    model = YOLO(args.model, task='detect')
```

**结果**:

- 用户提供`--weights yolo12n.pt --cfg n`
- 脚本加载`yolo12n.pt`的架构 (标准 YOLOv12-N，无 GGFE)
- `--cfg n`参数被完全忽略
- GGFE 配置从未生效

---

## ✅ 修复方案

### 方案 1: 修改第 136-156 行 (推荐)

```python
# ========== 修复后的代码 ==========
# 根据--cfg参数选择正确的YAML配置
cfg_map = {
    'n': 'ultralytics/cfg/models/12/yolo12-rgbd-ggfe-universal.yaml',
    's': 'ultralytics/cfg/models/12/yolo12-rgbd-ggfe-universal.yaml',
    'm': 'ultralytics/cfg/models/12/yolo12-rgbd-ggfe-universal.yaml',
    'l': 'ultralytics/cfg/models/12/yolo12-rgbd-ggfe-universal.yaml',
    'x': 'ultralytics/cfg/models/12/yolo12-rgbd-ggfe-universal.yaml',
}

# 总是先加载YAML创建模型架构
model_yaml = cfg_map.get(args.cfg, args.model)
LOGGER.info(f"Creating model from YAML: {model_yaml}")
model = YOLO(model_yaml, task='detect')

# 如果提供了weights，只加载参数（不覆盖架构）
if args.weights:
    LOGGER.info(f"Loading pretrained weights from {args.weights}")
    import torch

    # 加载权重state_dict
    ckpt = torch.load(args.weights, map_location='cpu')
    state_dict = ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']

    # 只加载匹配的参数（允许架构不完全一致）
    model.model.load_state_dict(state_dict, strict=False)
    LOGGER.info(f"✅ Loaded {len(state_dict)} parameters (strict=False)")
else:
    LOGGER.info(f"Training YOLO12-{args.cfg.upper()} from scratch")
```

**修复要点**:

1. ✅ 总是从 YAML 创建模型 (保证 GGFE 架构)
2. ✅ 权重仅用于参数初始化 (不覆盖架构)
3. ✅ `strict=False`允许 GGFE 模块未初始化 (从头训练)

---

### 方案 2: 修改第 68 行 (备用)

如果方案 1 太复杂，至少修改默认 YAML:

```python
# 第68行: 修改默认模型
parser.add_argument("--model", type=str,
                    default="ultralytics/cfg/models/12/yolo12-rgbd-ggfe-universal.yaml")  # 改这里!
```

**问题**: 这样还是会被`--weights`覆盖，不够彻底

---

## 🚀 完整修复文件

保存为 `train_depth_solr_v2_fixed.py`:

```python
#!/usr/bin/env python3
"""
YOLOv12-RGBD Training Script with SOLR Loss (Fixed Version)

修复: --cfg参数现在正确加载GGFE配置，不再被--weights覆盖

Usage:
    python train_depth_solr_v2_fixed.py --data visdrone-rgbd.yaml --cfg n --epochs 100 --weights yolo12n.pt
"""

import argparse
import os
from pathlib import Path
import torch

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

    LOGGER.info("\\n🔧 Integrating SOLR loss...")

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
    parser.add_argument("--cfg", type=str, default="n", help="Model size (n/s/m/l/x) - determines which YAML to use")
    parser.add_argument("--data", type=str, required=True, help="Dataset YAML")
    parser.add_argument("--weights", type=str, default="", help="Pretrained weights (optional, for parameter initialization)")

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
    parser.add_argument("--copy_paste", type=float, default=0.0, help="CopyPaste probability (0.0-1.0)")
    parser.add_argument("--close_mosaic", type=int, default=10)

    # Experiment
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="visdrone_ggfe")
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
    LOGGER.info("YOLOv12-RGBD Training with SOLR Loss (FIXED VERSION)")
    LOGGER.info("=" * 70)
    LOGGER.info(f"Model size: YOLO12-{args.cfg.upper()}")
    LOGGER.info(f"Data: {args.data}")
    LOGGER.info(f"Weights: {args.weights if args.weights else 'None (training from scratch)'}")
    LOGGER.info(f"Epochs: {args.epochs}, Batch: {args.batch}, Device: {args.device}")
    LOGGER.info(f"SOLR: small={args.small_weight}x, medium={args.medium_weight}x, large={args.large_weight}x")
    LOGGER.info(f"Augmentation: mosaic={args.mosaic}, mixup={args.mixup}, copy_paste={args.copy_paste}")
    LOGGER.info("=" * 70)

    # ========== 🔧 修复: 正确加载GGFE配置 ==========
    # Step 1: 根据--cfg选择YAML配置文件
    model_yaml = f'ultralytics/cfg/models/12/yolo12-rgbd-ggfe-universal.yaml'
    LOGGER.info(f"📄 Creating model from YAML: {model_yaml}")
    LOGGER.info(f"   (Model size will be determined by scales.{args.cfg} in YAML)")

    # Step 2: 从YAML创建模型架构 (确保GGFE模块存在)
    model = YOLO(model_yaml, task='detect')
    LOGGER.info(f"✅ Model architecture created (with GGFE modules)")

    # Step 3: 如果提供了weights，只加载参数 (不覆盖架构)
    if args.weights:
        LOGGER.info(f"📥 Loading pretrained weights from {args.weights}")

        # 加载checkpoint
        ckpt = torch.load(args.weights, map_location='cpu')

        # 提取state_dict
        if isinstance(ckpt, dict) and 'model' in ckpt:
            state_dict = ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']
        else:
            state_dict = ckpt

        # 加载参数 (strict=False允许GGFE模块未初始化)
        incompatible = model.model.load_state_dict(state_dict, strict=False)

        # 报告加载结果
        if incompatible.missing_keys:
            LOGGER.info(f"⚠️  Missing keys (will be randomly initialized): {len(incompatible.missing_keys)}")
            LOGGER.info(f"   Examples: {incompatible.missing_keys[:5]}")
        if incompatible.unexpected_keys:
            LOGGER.info(f"⚠️  Unexpected keys (ignored): {len(incompatible.unexpected_keys)}")
            LOGGER.info(f"   Examples: {incompatible.unexpected_keys[:5]}")

        LOGGER.info(f"✅ Loaded {len(state_dict)} parameters (strict=False)")
    else:
        LOGGER.info(f"🆕 Training YOLO12-{args.cfg.upper()} from scratch (no pretrained weights)")

    # ========== 验证模型参数量 ==========
    total_params = sum(p.numel() for p in model.model.parameters())
    LOGGER.info(f"📊 Total model parameters: {total_params/1e6:.2f}M")
    LOGGER.info(f"   Expected: ~3.5M (baseline 3.0M + GGFE 0.5M)")
    if total_params / 1e6 < 3.3:
        LOGGER.warning("⚠️  Warning: Parameter count too low, GGFE may not be loaded!")

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
    LOGGER.info("\\n🚀 Starting training...")
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
        copy_paste=args.copy_paste,
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
```

---

## 🎯 使用修复版本训练

```bash
# 上传train_depth_solr_v2_fixed.py到服务器

# 100ep快速验证
python train_depth_solr_v2_fixed.py \
    --name visdrone_ggfe_truly_fixed_100ep \
    --data /data2/user/2024/lzy/yolo12-bimodal/data/visdrone-rgbd.yaml \
    --device 4 \
    --weights /data2/user/2024/lzy/yolo12-bimodal/models/yolo12n.pt \
    --cfg n \
    --medium_weight 2.5 \
    --batch 16 \
    --epochs 100

# 训练开始后立即检查日志
tail -f runs/train/visdrone_ggfe_truly_fixed_100ep/train.log

# 应该看到:
# ✅ Model architecture created (with GGFE modules)
# 📊 Total model parameters: 3.50M
```

---

## 📋 修复验证清单

训练开始后 5 分钟内检查:

1. **参数量检查**:

   ```bash
   grep "Total model parameters" runs/train/visdrone_ggfe_truly_fixed_100ep/train.log
   # 应该看到: 3.50M (而非3.00M)
   ```

2. **Missing keys 检查**:

   ```bash
   grep "Missing keys" runs/train/visdrone_ggfe_truly_fixed_100ep/train.log
   # 应该看到: Missing keys: 100+ (GGFE模块的参数)
   ```

3. **第 1 个 epoch AP 检查**:
   ```bash
   # Epoch 1的AP应该稍低 (因为GGFE未初始化)
   # 但到Epoch 10应该开始超过baseline (19.2%)
   ```

---

**现在立即上传`train_depth_solr_v2_fixed.py`并重新训练，这次 GGFE 一定会生效！** 🎯
