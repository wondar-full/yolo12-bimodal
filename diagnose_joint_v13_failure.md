# exp_joint_v13 训练失败根因分析

## 🔴 **严重问题：mAP 仅 22.27% (目标 45%，差距-22.73%)**

### 📊 **训练配置分析**

从 `runs/train/exp_joint_v13/args.yaml` 发现的问题：

| 配置项         | 值                     | 问题                          |
| -------------- | ---------------------- | ----------------------------- |
| **model**      | `yolo12s-rgbd-v1.yaml` | ⚠️ 使用 Small 模型            |
| **pretrained** | `yolo12n.pt`           | ❌ **使用 Nano 权重**         |
| **不匹配**     | -                      | **🔥 架构和权重尺寸不匹配！** |

### 🎯 **根本原因分析**

#### **原因 1: 模型-权重不匹配 (最可能)**

```yaml
# 配置中的矛盾
model: ultralytics/cfg/models/12/yolo12s-rgbd-v1.yaml # Small架构
pretrained: yolo12n.pt # Nano权重

# 参数量对比
yolo12n: ~3M parameters
yolo12s: ~11M parameters # 3.7倍大小
```

**后果**:

- 权重加载时会**报错或跳过不匹配的层**
- 大部分层使用**随机初始化**（未预训练）
- 导致训练从几乎零开始，300 epochs 远远不够
- 解释了为什么 loss 收敛但性能极低

#### **原因 2: labels/rgb/ vs labels/ 路径问题**

UAVDT 的标签在 `labels/rgb/` 目录，但 YOLO 可能期望在 `labels/` 目录（与 images/rgb/对应）。

**验证方法**:

```python
# 在YOLORGBDDataset.get_img_files()中
# 如果标签路径是 /images/rgb/xxx.jpg
# 那么期望标签在 /labels/rgb/xxx.txt (✅ 正确)
# 而不是 /labels/xxx.txt (❌ 错误)
```

根据你的确认，UAVDT 的标签确实在`labels/rgb/`，**如果 YOLO 数据加载器实现正确，这应该没问题**。

但需要确认：**ultralytics/data/dataset.py 的 YOLORGBDDataset 是否正确处理了 images/rgb → labels/rgb 的映射？**

#### **原因 3: 深度图路径问题**

YAML 配置中：

```yaml
train:
  - VisDrone2019-DET-YOLO/.../images/rgb
  - UAVDT_YOLO/train/images/rgb

train_depth:
  - VisDrone2019-DET-YOLO/.../images/d
  - UAVDT_YOLO/train/images/d
```

**验证**: 深度图是否真的被加载？还是训练时只用了 RGB（3 通道）？

### 🔍 **诊断步骤**

#### **Step 1: 检查权重加载情况（服务器）**

```bash
# 在服务器上查看训练日志的开头部分
head -100 /data2/user/2024/lzy/yolo12-bimodal/runs/train/exp_joint_v13/train.log | grep -E "weight|load|Transferred|WARNING|ERROR"
```

**期望看到**:

```
Transferred 123/456 items from yolo12n.pt  # ⚠️ 应该看到transferred数量很少
WARNING: Some weights not loaded  # ⚠️ 或者看到警告
```

#### **Step 2: 检查数据加载器是否正确加载深度图**

在训练脚本中添加 debug 输出：

```python
# train_joint.py 或 train_depth.py
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/12/yolo12s-rgbd-v1.yaml')

# 在model.train()之前添加：
print(f"Model input channels: {model.model.yaml.get('ch', 3)}")
# 应该输出: Model input channels: 4

# 加载数据集
from ultralytics.data.dataset import YOLORGBDDataset
dataset = YOLORGBDDataset(...)
img, depth = dataset[0]['img'][:3], dataset[0]['img'][3:]
print(f"RGB shape: {img.shape}, Depth shape: {depth.shape}")
# 应该输出: RGB shape: (3, 640, 640), Depth shape: (1, 640, 640)
```

#### **Step 3: 检查 UAVDT 标签是否被正确加载**

```bash
# 服务器上检查训练时实际加载的标签
# 如果训练日志有详细输出
grep "labels loaded" /data2/user/2024/lzy/yolo12-bimodal/runs/train/exp_joint_v13/train.log

# 或者查看数据集统计
grep "instances" /data2/user/2024/lzy/yolo12-bimodal/runs/train/exp_joint_v13/train.log | head -10
```

**期望看到**:

```
Train: ... images, ... instances  # 应该有30,300张图像
Val: 548 images, ... instances  # 验证集只有VisDrone
```

### ✅ **修复方案**

#### **方案 A: 修复模型-权重匹配（推荐）**

```bash
# 重新训练，使用匹配的模型和权重
CUDA_VISIBLE_DEVICES=7 python train_depth.py \
    --data data/visdrone_uavdt_joint.yaml \
    --model ultralytics/cfg/models/12/yolo12n-rgbd-v1.yaml \  # ✅ 使用Nano架构
    --weights yolo12n.pt \  # ✅ 使用Nano权重
    --epochs 300 \
    --batch 16 \
    --name exp_joint_v14_fixed \
    --device 7
```

**或者使用 Small 的权重**:

```bash
CUDA_VISIBLE_DEVICES=7 python train_depth.py \
    --data data/visdrone_uavdt_joint.yaml \
    --model ultralytics/cfg/models/12/yolo12s-rgbd-v1.yaml \  # ✅ Small架构
    --weights yolo12s.pt \  # ✅ Small权重
    --epochs 300 \
    --batch 16 \
    --name exp_joint_v14_s \
    --device 7
```

#### **方案 B: 从头训练（不使用预训练权重）**

如果 RGB-D 架构和 RGB-only 预训练权重不兼容，可以从头训练：

```bash
CUDA_VISIBLE_DEVICES=7 python train_depth.py \
    --data data/visdrone_uavdt_joint.yaml \
    --model ultralytics/cfg/models/12/yolo12n-rgbd-v1.yaml \
    --epochs 500 \  # ⚠️ 需要更多epochs（从头训练）
    --batch 16 \
    --name exp_joint_v14_scratch \
    --device 7
    # 不指定--weights，从随机初始化开始
```

#### **方案 C: 验证数据加载（先测试）**

在重新训练前，先验证数据加载是否正确：

```bash
# 在服务器上运行
cd /data2/user/2024/lzy/yolo12-bimodal
python -c "
from ultralytics.data import build_dataloader
from ultralytics.data.dataset import YOLORGBDDataset
import yaml

# 加载配置
with open('data/visdrone_uavdt_joint.yaml') as f:
    data = yaml.safe_load(f)

print(f'Dataset config: {data}')
print(f'Channels: {data.get(\"channels\", 3)}')  # 应该是4

# 尝试加载一个batch
# 如果成功，说明数据加载器没问题
print('Testing data loading...')
# dataloader = build_dataloader(...)  # 根据实际API调整
print('✅ Data loading successful!')
"
```

### 📝 **下一步行动**

**立即执行**:

1. ✅ **确认数据集完整性** - 已完成（23,829 张全部对齐）

2. ⚠️ **检查训练日志** - 需要在服务器上执行：

   ```bash
   head -100 /data2/user/2024/lzy/yolo12-bimodal/runs/train/exp_joint_v13/train.log
   ```

   查找权重加载警告

3. 🔥 **修复并重新训练** - 使用匹配的模型和权重：

   ```bash
   # 选择方案A（推荐yolo12n匹配）
   CUDA_VISIBLE_DEVICES=7 python train_depth.py \
       --data data/visdrone_uavdt_joint.yaml \
       --model ultralytics/cfg/models/12/yolo12n-rgbd-v1.yaml \
       --weights yolo12n.pt \
       --epochs 300 \
       --batch 16 \
       --name exp_joint_v14_fixed \
       --device 7
   ```

4. 📊 **监控新训练** - 关键指标：
   - Epoch 10: mAP 应该 >15% (vs 当前~10%)
   - Epoch 50: mAP 应该 >30% (vs 当前~22%)
   - Epoch 150: mAP 应该 >40% (目标达成)

### 🎯 **预期效果**

修复模型-权重匹配后：

- ✅ 权重正确加载，训练从良好的初始化开始
- ✅ mAP 在早期就应该看到明显提升
- ✅ 最终 mAP 应该达到 40-45%（接近或超越目标）

如果修复后性能仍然不佳，再考虑其他原因（数据加载、标签路径等）。

---

**优先级**:

1. 🔥🔥🔥 **修复模型-权重匹配** ← **最可能的原因，立即修复**
2. ⚠️ 验证数据加载是否正确（深度图、标签）
3. ⏸️ 其他优化（loss 权重、数据增强等）等基础问题解决后再考虑
