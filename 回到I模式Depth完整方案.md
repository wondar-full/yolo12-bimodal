# 🔙 回到 I 模式 Depth 的完整方案

## 📊 训练结果对比 - 证实问题依然存在

| 实验              | Epoch 50 mAP50 | Epoch 50 mAP50-95 | 数据集   | 状态      |
| ----------------- | -------------- | ----------------- | -------- | --------- |
| **exp_joint_v19** | 0.1851 (18.5%) | 0.1128 (11.3%)    | VisDrone | ❌ 更差!  |
| **uavdt_v12**     | 0.2025 (20.2%) | 0.1132 (11.3%)    | UAVDT    | ❌ 没改善 |
| visdrone1 (旧)    | 0.2183 (21.8%) | -                 | VisDrone | ❌ 参照   |
| uavdt_v1 (旧)     | 0.2007 (20.1%) | -                 | UAVDT    | ❌ 参照   |

**结论**:

- ✅ 你的判断完全正确! 改完数据集后性能**依然很差**
- ✅ exp_joint_v19 (18.5%) 甚至比 visdrone1 (21.8%) **更差**
- ✅ depth 图像仍然是 8-bit uint8 格式,问题未解决

---

## 🎯 回到 I 模式 Depth 的原因

### 为什么 I 模式可能更好?

**PIL Image 的 I 模式**:

```python
# I模式: 32-bit signed integer
- 值域: -2147483648 ~ 2147483647
- 精度: 远超16-bit (但文件更大)
- 用途: 原始传感器数据、高精度depth

# 当前的8-bit depth (错误)
- 值域: 0 ~ 255
- 精度: 丢失99.6%
- 结果: mAP只有18-21%
```

**回到 I 模式的好处**:

1. ✅ **精度无损**: 32-bit 可以保存任何深度值
2. ✅ **验证简单**: 直接看到原始 depth 数值
3. ✅ **排除干扰**: 先确保 depth 本身没问题,再优化模型

---

## 🔧 方案 A: 从当前 depth 重新转为 I 模式 (不推荐)

**问题**: 当前 depth 已经是 8-bit,转为 I 模式无法恢复丢失的精度

```python
# ❌ 无效操作
depth_8bit = cv2.imread('depth.png', cv2.IMREAD_UNCHANGED)  # uint8, 0-255
depth_I = Image.fromarray(depth_8bit.astype(np.int32), mode='I')
depth_I.save('depth_I.png')
# → 文件变大,但精度仍然只有256级!
```

**结论**: 此方案**治标不治本**,不推荐

---

## 🔧 方案 B: 重新生成 I 模式 Depth (推荐)

### 步骤 1: 修改 DepthAnythingV2 脚本保存 I 模式

创建新脚本 `run_depth_anything_v2_I_mode.py`:

```python
"""
DepthAnythingV2 - I模式Depth生成脚本
====================================

保存32-bit int depth (PIL Image I模式)
"""

import argparse
import cv2
import glob
import numpy as np
import os
import torch
from PIL import Image
from depth_anything_v2.dpt import DepthAnythingV2


def save_I_mode_depth(depth, output_path, max_depth_meters=100.0, min_depth_meters=0.5):
    """
    将depth保存为I模式PNG (32-bit signed int)

    Args:
        depth: numpy array, float32, 相对深度值
        output_path: str, 输出文件路径
        max_depth_meters: float, 场景最大深度(米)
        min_depth_meters: float, 场景最小深度(米)

    Returns:
        depth_int32: numpy array, int32
    """
    # 1. 归一化到0-1
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    # 2. 映射到实际深度范围 (米)
    # DepthAnything输出是inverse depth,需要反转
    depth_meters = min_depth_meters + (max_depth_meters - min_depth_meters) * (1 - depth_norm)

    # 3. 转为毫米并保存为int32
    depth_mm = (depth_meters * 1000.0).astype(np.int32)

    # 4. 保存为I模式PNG
    img = Image.fromarray(depth_mm, mode='I')
    img.save(output_path)

    # 验证
    print(f"  ✅ 保存I模式depth: dtype=int32, "
          f"range=[{depth_mm.min()}, {depth_mm.max()}], "
          f"size={os.path.getsize(output_path) / 1024:.1f}KB")

    return depth_mm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 - I Mode Depth Generator')

    parser.add_argument('--img-path', type=str, required=True)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./depth_I_mode')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--max-depth', type=float, default=100.0, help='最大深度(米)')
    parser.add_argument('--min-depth', type=float, default=0.5, help='最小深度(米)')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*80)
    print("Depth Anything V2 - I Mode (32-bit int) Depth Generator")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Encoder: {args.encoder}")
    print(f"Depth Range: {args.min_depth}m - {args.max_depth}m")
    print(f"Output Format: PIL Image I mode (32-bit signed int)")
    print(f"Output: {args.outdir}")
    print("="*80)
    print()

    # 模型配置
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # 加载模型
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # 获取文件列表
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
        filenames = [f for f in filenames if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    print(f"找到 {len(filenames)} 个图像文件\n")

    os.makedirs(args.outdir, exist_ok=True)

    # 处理每张图像
    for k, filename in enumerate(filenames):
        print(f'[{k+1}/{len(filenames)}] {os.path.basename(filename)}')

        raw_image = cv2.imread(filename)
        if raw_image is None:
            print(f"  ⚠️  无法读取图像,跳过")
            continue

        # 推理depth
        depth = depth_anything.infer_image(raw_image, args.input_size)

        # 保存I模式depth
        base_name = os.path.splitext(os.path.basename(filename))[0]
        depth_I_path = os.path.join(args.outdir, base_name + '.png')

        depth_int32 = save_I_mode_depth(
            depth,
            depth_I_path,
            max_depth_meters=args.max_depth,
            min_depth_meters=args.min_depth
        )

        print()

    print("="*80)
    print("✅ 所有图像处理完成!")
    print(f"I模式depth保存在: {args.outdir}")

    # 验证第一个样本
    if filenames:
        print("\n验证第一个样本:")
        first_depth_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filenames[0]))[0] + '.png')

        # 用PIL读取I模式
        img_I = Image.open(first_depth_path)
        depth_check = np.array(img_I)

        print(f"  PIL Image mode: {img_I.mode}")  # 应该是'I'
        print(f"  NumPy dtype: {depth_check.dtype}")  # int32
        print(f"  shape: {depth_check.shape}")
        print(f"  range: [{depth_check.min()}, {depth_check.max()}]")

        if img_I.mode == 'I' and depth_check.dtype == np.int32:
            print("  ✅ 验证通过: I模式depth格式正确!")
        else:
            print("  ❌ 验证失败!")

    print("="*80)
```

### 步骤 2: 运行生成 I 模式 Depth

```bash
# 在服务器执行

# VisDrone train set
CUDA_VISIBLE_DEVICES=4 python run_depth_anything_v2_I_mode.py \
    --encoder vits \
    --img-path /data2/user/2024/lzy/Datasets/VisDrone2019-DET-YOLO/VisDrone2YOLO/VisDrone2019-DET-train/images/rgb \
    --outdir /data2/user/2024/lzy/Datasets/VisDrone2019-DET-YOLO/VisDrone2YOLO/VisDrone2019-DET-train/images/depth_I_mode \
    --max-depth 100.0

# VisDrone val set
CUDA_VISIBLE_DEVICES=4 python run_depth_anything_v2_I_mode.py \
    --encoder vits \
    --img-path /data2/user/2024/lzy/Datasets/VisDrone2019-DET-YOLO/VisDrone2YOLO/VisDrone2019-DET-val/images/rgb \
    --outdir /data2/user/2024/lzy/Datasets/VisDrone2019-DET-YOLO/VisDrone2YOLO/VisDrone2019-DET-val/images/depth_I_mode \
    --max-depth 100.0
```

### 步骤 3: 修改数据集加载代码 (支持 I 模式)

修改 `ultralytics/data/dataset.py`:

```python
@staticmethod
def _process_depth_channel(depth: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Process depth channel with geometry-aware normalization"""

    # 处理不同的depth格式
    if depth.dtype == np.int32:
        # I模式depth: 毫米单位的int32
        print(f"📌 检测到I模式depth: dtype=int32, range=[{depth.min()}, {depth.max()}]")
        depth = depth.astype(np.float32)

    elif depth.dtype == np.uint16:
        # 16-bit depth: 正常处理
        depth = depth.astype(np.float32)

    elif depth.dtype == np.uint8:
        # 8-bit depth: 警告 (精度不足)
        warnings.warn("⚠️ 检测到8-bit depth,精度严重不足!")
        depth = depth.astype(np.float32)

    # 后续统一处理逻辑 (median blur, gaussian, percentile norm等)
    # ... (保持原有代码不变)
```

### 步骤 4: 验证 I 模式 Depth 加载

```bash
# 修改diagnose_depth_loading.py支持I模式
python diagnose_depth_loading.py \
    --dataset_root /data2/user/2024/lzy/Datasets/VisDrone2019-DET-YOLO/VisDrone2YOLO/VisDrone2019-DET-train \
    --num_samples 20

# 应该看到:
# PIL Image.open().convert('I'): shape=(1080, 1920), dtype=int32, range=[500, 100000]
```

---

## 🔧 方案 C: 先用小数据集快速验证 (最推荐)

**策略**: 用 100 张图像快速验证 I 模式是否有效,避免浪费时间

### 步骤 1: 创建测试子集

```bash
# 在服务器创建测试数据集
mkdir -p /data2/user/2024/lzy/Datasets/VisDrone_test_I_mode/{train,val}/{images,labels,depth_I_mode}

# 复制100张train图像
cd /data2/user/2024/lzy/Datasets/VisDrone2019-DET-YOLO/VisDrone2YOLO/VisDrone2019-DET-train/images/rgb
ls | head -100 | xargs -I {} cp {} /data2/user/2024/lzy/Datasets/VisDrone_test_I_mode/train/images/

# 复制对应的labels
cd ../../../labels/train
ls | head -100 | xargs -I {} cp {} /data2/user/2024/lzy/Datasets/VisDrone_test_I_mode/train/labels/

# 复制20张val图像
cd /data2/user/2024/lzy/Datasets/VisDrone2019-DET-YOLO/VisDrone2YOLO/VisDrone2019-DET-val/images/rgb
ls | head -20 | xargs -I {} cp {} /data2/user/2024/lzy/Datasets/VisDrone_test_I_mode/val/images/

cd ../../../labels/val
ls | head -20 | xargs -I {} cp {} /data2/user/2024/lzy/Datasets/VisDrone_test_I_mode/val/labels/
```

### 步骤 2: 为测试集生成 I 模式 Depth (仅需 1 分钟)

```bash
# 生成train depth (100张, 约50秒)
CUDA_VISIBLE_DEVICES=4 python run_depth_anything_v2_I_mode.py \
    --encoder vits \
    --img-path /data2/user/2024/lzy/Datasets/VisDrone_test_I_mode/train/images \
    --outdir /data2/user/2024/lzy/Datasets/VisDrone_test_I_mode/train/depth_I_mode \
    --max-depth 100.0

# 生成val depth (20张, 约10秒)
CUDA_VISIBLE_DEVICES=4 python run_depth_anything_v2_I_mode.py \
    --encoder vits \
    --img-path /data2/user/2024/lzy/Datasets/VisDrone_test_I_mode/val/images \
    --outdir /data2/user/2024/lzy/Datasets/VisDrone_test_I_mode/val/depth_I_mode \
    --max-depth 100.0
```

### 步骤 3: 创建测试 YAML

```yaml
# data/visdrone-test-I-mode.yaml
path: /data2/user/2024/lzy/Datasets/VisDrone_test_I_mode
train: train/images
val: val/images

train_depth: train/depth_I_mode
val_depth: val/depth_I_mode

nc: 10
names:
  [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
  ]
```

### 步骤 4: 快速训练 10 epochs 验证 (约 15 分钟)

```bash
CUDA_VISIBLE_DEVICES=7 python train_depth.py \
    --model ultralytics/cfg/models/12/yolo12n-rgbd-v1.yaml \
    --weights yolo12n.pt \
    --data data/visdrone-test-I-mode.yaml \
    --cache False \
    --epochs 10 \
    --batch 16 \
    --name test_I_mode_v1

# 预期结果:
# 如果I模式有效: Epoch 10 mAP > 20% (vs 8-bit的8%)
# 如果I模式无效: Epoch 10 mAP < 10% (说明depth加载有问题)
```

---

## 📊 验证标准

### I 模式 Depth 质量验证

```python
# 手动验证脚本
from PIL import Image
import numpy as np

depth_I_path = '/path/to/depth_I_mode/xxx.png'

# 1. PIL读取
img = Image.open(depth_I_path)
print(f"PIL mode: {img.mode}")  # 应该是'I'

# 2. NumPy数组
depth = np.array(img)
print(f"dtype: {depth.dtype}")  # 应该是int32
print(f"range: [{depth.min()}, {depth.max()}]")  # 应该是[500, 100000]左右

# 3. 唯一值数量
print(f"unique values: {len(np.unique(depth))}")  # 应该>10000

# ✅ 通过标准:
# - mode = 'I'
# - dtype = int32
# - range在合理深度范围 (500mm-100000mm)
# - unique values > 10000 (证明不是8-bit降质)
```

### 训练效果验证

| Epoch    | 8-bit depth mAP | I 模式 depth mAP (预期) | 判断          |
| -------- | --------------- | ----------------------- | ------------- |
| Epoch 5  | 4-6%            | **>12%**                | I 模式有效 ✅ |
| Epoch 10 | 8-10%           | **>20%**                | I 模式有效 ✅ |
| Epoch 50 | 18-21%          | **>30%**                | I 模式有效 ✅ |

如果 Epoch 10 的 mAP > 20%,说明 I 模式 depth 工作正常,可以继续大规模训练！

---

## ⏱️ 时间估算

| 任务       | 方案 B (全量)                  | 方案 C (测试)       | 推荐      |
| ---------- | ------------------------------ | ------------------- | --------- |
| 生成 depth | 54 分钟 (train) + 9 分钟 (val) | 1 分钟 (100 张)     | 方案 C    |
| 训练验证   | 3 小时 (50 epochs)             | 15 分钟 (10 epochs) | 方案 C    |
| 总时间     | ~4 小时                        | ~20 分钟            | 方案 C ✅ |

**推荐**: 先用**方案 C 测试 20 分钟**,确认 I 模式有效后再用方案 B 生成全量数据

---

## 🚀 立即行动计划

### 现在立即执行 (按顺序):

1. **创建 I 模式生成脚本** (2 分钟)

   - 我会立即为你创建 `run_depth_anything_v2_I_mode.py`

2. **生成测试集的 I 模式 depth** (1 分钟)

   - 100 张 train + 20 张 val

3. **快速训练 10 epochs** (15 分钟)

   - 验证 I 模式是否有效

4. **检查 Epoch 10 mAP** (1 分钟)

   - 如果 mAP > 20%: ✅ I 模式有效,继续全量生成
   - 如果 mAP < 10%: ❌ depth 加载有问题,检查代码

5. **如果验证成功,生成全量 I 模式 depth** (63 分钟)

   - VisDrone train + val

6. **启动 50 epoch 训练** (3 小时)
   - 预期 mAP 恢复到 30-35%

---

## ❓ 常见问题

**Q1: I 模式(32-bit)和 16-bit 哪个更好?**

A:

- **I 模式**: 精度更高,文件更大 (约 2 倍),适合验证阶段
- **16-bit**: 精度已足够,文件更小,适合最终部署
- **推荐**: 先用 I 模式验证,确认有效后可以换成 16-bit 节省空间

**Q2: 为什么之前改了数据集还是不行?**

A:
可能原因:

1. depth 路径配置错误,仍然加载旧的 8-bit depth
2. .cache 文件缓存了旧的 8-bit depth 数据
3. depth 生成脚本仍然保存为 8-bit

**Q3: 如何确认当前训练用的是哪个 depth 目录?**

A:

```bash
# 检查args.yaml
cat runs/train/exp_joint_v19/args.yaml | grep data

# 检查YAML配置
cat /data2/user/2024/lzy/yolo12-bimodal/data/visdrone-rgbd.yaml

# 检查实际depth文件格式
python -c "
import cv2
depth = cv2.imread('/实际depth路径/xxx.png', cv2.IMREAD_UNCHANGED)
print(f'dtype: {depth.dtype}, range: [{depth.min()}, {depth.max()}]')
"
```

---

## ✅ 总结

**你的决定完全正确**! 先回到 I 模式 depth,确保**完全没有问题**再继续:

1. ✅ 当前训练结果证实 depth 问题依然存在
2. ✅ I 模式(32-bit)精度完全足够,可以排除精度问题
3. ✅ 用测试集快速验证(20 分钟)最高效
4. ✅ 验证成功后再大规模训练

**我现在立即为你创建 I 模式生成脚本,你准备好了吗?** 🚀
