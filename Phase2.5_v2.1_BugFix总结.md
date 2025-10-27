# Phase 2.5 v2.1 - 分尺度 mAP 为 0 的 Bug 修复总结

**日期**: 2025/01/XX  
**问题**: size-wise mAP (small/medium/large) 全部返回 0  
**严重性**: 🔴 Critical - 阻碍 VisDrone 评估的核心功能

---

## 问题诊断

### 症状

运行`python val_visdrone.py --model best.pt`后:

```
mAP@0.5: 43.51% ✅ 正常
mAP@0.75: 27.20% ✅ 正常
Small (<32×32): 0.00% ❌ 异常
Medium (32~64): 0.00% ❌ 异常
Large (>64×64): 0.00% ❌ 异常
```

### 根本原因分析

通过代码审查发现**3 个环节的断链**:

#### 1️⃣ Dataset 没有计算 target_areas

**文件**: `ultralytics/data/dataset.py` - `YOLODataset.update_labels_info()`

```python
# ❌ 原代码: 没有计算目标面积
def update_labels_info(self, label: dict) -> dict:
    bboxes = label.pop("bboxes")
    # ... 处理bboxes/segments/keypoints
    label["instances"] = Instances(...)
    return label  # ← 缺少 target_areas
```

**问题**: label 字典中根本没有`target_areas`字段

---

#### 2️⃣ Validator 没有使用 DetMetricsVisDrone

**文件**: `ultralytics/models/yolo/detect/val.py` - `DetectionValidator.__init__()`

```python
# ❌ 原代码: 写死使用DetMetrics (不支持分尺度)
def __init__(self, ...):
    super().__init__(...)
    self.metrics = DetMetrics()  # ← 不支持small/medium/large
```

**问题**: 即使 dataset 提供了 target_areas,标准的 DetMetrics 也不会处理

---

#### 3️⃣ Validator 没有传递 target_areas 到 metrics

**文件**: `ultralytics/models/yolo/detect/val.py` - `update_metrics()`

```python
# ❌ 原代码: 即使pbatch有target_areas,也没传给metrics
self.metrics.update_stats({
    "tp": ...,
    "target_cls": cls,
    "conf": ...,
    # ← 缺少 "target_areas": pbatch["target_areas"]
})
```

**问题**: 数据流到 validator 就断了,metrics 收不到 target_areas

---

## 完整修复方案

### 修复 1: Dataset 计算 target_areas

**文件**: `ultralytics/data/dataset.py`  
**修改**: `YOLODataset.update_labels_info()` (Line ~275-300)

```python
def update_labels_info(self, label: dict) -> dict:
    bboxes = label.pop("bboxes")
    segments = label.pop("segments", [])
    keypoints = label.pop("keypoints", None)
    bbox_format = label.pop("bbox_format")
    normalized = label.pop("normalized")

    # 🆕 计算目标面积 (for VisDrone size-wise metrics)
    if len(bboxes) > 0:
        if bbox_format == "xyxy":
            widths = bboxes[:, 2] - bboxes[:, 0]
            heights = bboxes[:, 3] - bboxes[:, 1]
        elif bbox_format == "xywh":
            widths = bboxes[:, 2]
            heights = bboxes[:, 3]
        else:
            widths = heights = np.zeros(len(bboxes))

        # 如果是归一化坐标,需要乘以图像尺寸
        if normalized:
            img_h, img_w = label.get("ori_shape", (640, 640))[:2]
            widths = widths * img_w
            heights = heights * img_h

        target_areas = (widths * heights).astype(np.float32)
    else:
        target_areas = np.array([], dtype=np.float32)

    label["target_areas"] = target_areas  # 🆕 添加到label字典

    # ... 后续处理
    label["instances"] = Instances(...)
    return label
```

**关键点**:

- 支持`xyxy`和`xywh`两种 bbox 格式
- 处理归一化/非归一化坐标
- 空 bbox 时返回空数组(避免崩溃)

---

### 修复 2: Augment 转换 target_areas 为 tensor

**文件**: `ultralytics/data/augment.py`  
**修改**: `Format.__call__()` (Line ~2205)

```python
labels["img"] = self._format_img(img)
labels["cls"] = torch.from_numpy(cls) if nl else torch.zeros(nl, 1)
labels["bboxes"] = torch.from_numpy(instances.bboxes) if nl else torch.zeros((nl, 4))

# 🆕 处理target_areas (for VisDrone size-wise metrics)
if "target_areas" in labels:
    target_areas = labels["target_areas"]
    if isinstance(target_areas, np.ndarray):
        labels["target_areas"] = torch.from_numpy(target_areas) if nl else torch.zeros(nl)
```

**关键点**:

- 将 numpy 数组转为 tensor (与 cls/bboxes 一致)
- 兼容旧代码(没有 target_areas 的场景)

---

### 修复 3: Collate 函数处理 target_areas

**文件**: `ultralytics/data/dataset.py`  
**修改**: `YOLODataset.collate_fn()` (Line ~335)

```python
@staticmethod
def collate_fn(batch: list[dict]) -> dict:
    # ... 处理其他字段

    # 🆕 target_areas 需要concat (与bboxes/cls一样)
    if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb", "target_areas"}:
        value = torch.cat(value, 0)

    # ...
```

**关键点**:

- target_areas 与 bboxes/cls 一样,需要跨 batch 拼接
- 使用`torch.cat`而不是`torch.stack`

---

### 修复 4: Validator 使用 DetMetricsVisDrone

**文件**: `ultralytics/models/yolo/detect/val.py`  
**修改 1**: 导入 DetMetricsVisDrone (Line ~17)

```python
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.metrics_visdrone import DetMetricsVisDrone  # 🆕 添加
```

**修改 2**: `__init__()` (Line ~63)

```python
def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
    super().__init__(...)
    # ... 其他初始化

    # 🆕 根据args.visdrone_mode决定使用哪个metrics类
    visdrone_mode = getattr(self.args, 'visdrone_mode', False)
    if visdrone_mode:
        LOGGER.info(f"Using DetMetricsVisDrone with visdrone_mode={visdrone_mode}")
        small_thresh = getattr(self.args, 'small_thresh', 1024)    # 默认32x32
        medium_thresh = getattr(self.args, 'medium_thresh', 4096)  # 默认64x64
        self.metrics = DetMetricsVisDrone(
            visdrone_mode=visdrone_mode,
            small_thresh=small_thresh,
            medium_thresh=medium_thresh,
        )
    else:
        LOGGER.info("Using standard DetMetrics")
        self.metrics = DetMetrics()
```

**关键点**:

- 使用`getattr`避免旧代码报错(没有 visdrone_mode 参数)
- 默认阈值: small<1024, medium=1024~4096, large>4096
- 向下兼容:非 VisDrone 任务仍用 DetMetrics

---

### 修复 5: Validator 传递 target_areas

**文件**: `ultralytics/models/yolo/detect/val.py`  
**修改 1**: `_prepare_batch()` (Line ~165)

```python
def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
    idx = batch["batch_idx"] == si
    cls = batch["cls"][idx].squeeze(-1)
    bbox = batch["bboxes"][idx]
    # ...

    # 🆕 提取target_areas (如果存在)
    target_areas = batch.get("target_areas", None)
    if target_areas is not None and len(idx) > 0:
        target_areas = target_areas[idx]  # 过滤当前batch的areas

    result = {
        "cls": cls,
        "bboxes": bbox,
        # ...
    }

    # 🆕 只在target_areas存在时添加(避免普通YOLO任务报错)
    if target_areas is not None:
        result["target_areas"] = target_areas

    return result
```

**修改 2**: `update_metrics()` (Line ~205)

```python
def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
    for si, pred in enumerate(preds):
        pbatch = self._prepare_batch(si, batch)
        predn = self._prepare_pred(pred)
        cls = pbatch["cls"].cpu().numpy()
        no_pred = predn["cls"].shape[0] == 0

        # 🆕 构建stats字典,包含target_areas(如果存在)
        stats_dict = {
            **self._process_batch(predn, pbatch),
            "target_cls": cls,
            "target_img": np.unique(cls),
            "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
            "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
        }

        # 🆕 如果pbatch有target_areas,添加到stats(for VisDrone size-wise metrics)
        if "target_areas" in pbatch:
            target_areas = pbatch["target_areas"]
            # 确保转换为numpy数组
            if isinstance(target_areas, torch.Tensor):
                target_areas = target_areas.cpu().numpy()
            stats_dict["target_areas"] = target_areas

        self.metrics.update_stats(stats_dict)
```

**关键点**:

- 使用`batch.get("target_areas", None)`避免 KeyError
- 根据 batch_idx 过滤对应的 areas
- tensor→numpy 转换(metrics 期望 numpy)

---

### 修复 6: val_visdrone.py 传递参数

**文件**: `val_visdrone.py`  
**修改**: `validate_visdrone()` (Line ~355)

```python
val_args = dict(
    data=args.data,
    batch=args.batch,
    # ... 其他参数

    # 🆕 添加VisDrone特定参数
    visdrone_mode=True,  # 启用VisDrone分尺度评估
    small_thresh=args.small_thresh,
    medium_thresh=args.medium_thresh,
)
```

---

## 数据流全貌

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Dataset: YOLODataset.update_labels_info()               │
│    计算: target_areas = (w * h).astype(np.float32)         │
│    输出: label["target_areas"] = np.array([...])           │
└────────────────────┬───────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Augment: Format.__call__()                              │
│    转换: torch.from_numpy(target_areas)                    │
│    输出: label["target_areas"] = torch.tensor([...])       │
└────────────────────┬───────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Collate: YOLODataset.collate_fn()                       │
│    拼接: torch.cat([areas1, areas2, ...], dim=0)           │
│    输出: batch["target_areas"] = torch.tensor([全部areas]) │
└────────────────────┬───────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Validator: DetectionValidator._prepare_batch()         │
│    过滤: target_areas[batch_idx == si]                    │
│    输出: pbatch["target_areas"] = torch.tensor([单图areas])│
└────────────────────┬───────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Validator: update_metrics()                             │
│    添加: stats_dict["target_areas"] = areas.cpu().numpy() │
│    输出: self.metrics.update_stats(stats_dict)             │
└────────────────────┬───────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Metrics: DetMetricsVisDrone.update_stats()             │
│    检查: if 'target_areas' in stat:                       │
│    分类: small_mask = areas < 1024                        │
│          medium_mask = 1024 <= areas < 4096               │
│          large_mask = areas >= 4096                       │
│    存储: self.stats_by_size['small']['tp'].append(...)    │
└────────────────────┬───────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. Metrics: DetMetricsVisDrone.process()                   │
│    计算: ap_per_class(stats_by_size['small'])             │
│    输出: self.box_small.map50 = XX.XX%                    │
│          self.box_medium.map50 = XX.XX%                   │
│          self.box_large.map50 = XX.XX%                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 测试验证

### 本地测试命令

```bash
# 进入项目目录
cd f:\CV\Paper\yoloDepth\yoloDepth

# 运行验证 (使用v2.1模型或任何best.pt)
python val_visdrone.py --model runs/train/rgbd_v2.1_full/weights/best.pt

# 期望输出:
# ╔════════════════════════════════════════════════════════════════╗
# ║             RemDet Comparison Report                          ║
# ╠════════════════════════════════════════════════════════════════╣
# ║ 📊 Accuracy Metrics:                                          ║
# ║   mAP@0.5      43.51%       45.2%        -1.69%   -3.7%    ❌ ║
# ║   mAP@0.75     27.20%       28.5%        -1.30%   -4.6%    ❌ ║
# ║                                                                ║
# ║ 📐 By Object Size:                                            ║
# ║   Small        15.20%       21.3%        -6.10%   -28.6%   ❌ ← 不再是0!
# ║   Medium       35.80%       N/A          N/A                  ║ ← 不再是0!
# ║   Large        52.30%       N/A          N/A                  ║ ← 不再是0!
# ║                                                                ║
# ║ ⚡ Efficiency Metrics:                                        ║
# ║   Latency(ms)  11.20        12.8         -1.60ms  -12.5%   ✅ ║
# ║   FLOPs(G)     48.30        52.4         -4.10G   -7.8%    ✅ ║
# ║   Params(M)    9.60         16.3         -6.70M   -41.1%   ✅ ║
# ╚════════════════════════════════════════════════════════════════╝
```

### 验证标准

| 指标                                   | 预期范围    | 说明                                      |
| -------------------------------------- | ----------- | ----------------------------------------- |
| **mAP_small**                          | 15% ~ 18%   | UAV 小目标主战场,低于 15%说明模型能力不足 |
| **mAP_medium**                         | 35% ~ 40%   | 中等目标,应高于 small                     |
| **mAP_large**                          | 50% ~ 55%   | 大目标,应最高                             |
| **mAP_small < mAP_medium < mAP_large** | ✅ 必须满足 | 尺度递增规律                              |

---

## 修改文件清单

| 文件                                    | 修改类型    | 关键修改点                                        |
| --------------------------------------- | ----------- | ------------------------------------------------- |
| `ultralytics/data/dataset.py`           | 🔧 逻辑增强 | update_labels_info() + collate_fn()               |
| `ultralytics/data/augment.py`           | 🔧 逻辑增强 | Format.**call**()                                 |
| `ultralytics/models/yolo/detect/val.py` | 🔧 架构修改 | **init**() + \_prepare_batch() + update_metrics() |
| `val_visdrone.py`                       | 🔧 参数增加 | validate_visdrone()                               |
| `ultralytics/utils/metrics_visdrone.py` | ✅ 无需修改 | 逻辑已正确,等待数据输入                           |

---

## 后续行动

### ✅ 立即验证 (本地)

```bash
python val_visdrone.py --model runs/train/rgbd_v2.1_full/weights/best.pt
```

### ✅ 服务器测试 (如果本地通过)

```bash
# 上传修改后的文件到服务器
scp ultralytics/data/dataset.py user@server:/path/to/yoloDepth/ultralytics/data/
scp ultralytics/data/augment.py user@server:/path/to/yoloDepth/ultralytics/data/
scp ultralytics/models/yolo/detect/val.py user@server:/path/to/yoloDepth/ultralytics/models/yolo/detect/
scp val_visdrone.py user@server:/path/to/yoloDepth/

# 服务器运行
python val_visdrone.py --model runs/train/rgbd_v2.1_full/weights/best.pt
```

### ✅ 对比 RGB-only vs RGB-D (如果修复成功)

```bash
# RGB-D模型
python val_visdrone.py --model runs/train/rgbd_v2.1_full/weights/best.pt

# RGB-only模型 (baseline)
python val_visdrone.py --model runs/train/rgb_only/weights/best.pt --data data/visdrone.yaml

# 对比mAP_small改进幅度
```

---

## 易错点警告 ⚠️

### 1. 归一化坐标的面积计算

```python
# ❌ 错误: 直接用归一化坐标计算
areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
# 结果: area < 1 (因为归一化到0~1), 全部被分类为small

# ✅ 正确: 先反归一化
if normalized:
    widths = widths * img_w
    heights = heights * img_h
areas = widths * heights  # 像素面积
```

### 2. Collate 时的拼接方式

```python
# ❌ 错误: 使用stack (会增加维度)
if k == "target_areas":
    value = torch.stack(value, 0)  # shape: [batch, num_boxes]

# ✅ 正确: 使用cat (与bboxes/cls一致)
if k in {"bboxes", "cls", "target_areas"}:
    value = torch.cat(value, 0)  # shape: [total_boxes]
```

### 3. Validator 中的 batch_idx 过滤

```python
# ❌ 错误: 忘记过滤areas
target_areas = batch["target_areas"]  # 全batch的areas

# ✅ 正确: 根据batch_idx过滤
idx = batch["batch_idx"] == si
target_areas = batch["target_areas"][idx]  # 当前图片的areas
```

---

## 八股知识点关联

本次修复涉及的核心概念:

- **[024] Validator 与 Metrics 的协作机制** (待添加到八股.md)
- **[025] YOLO 数据流: Dataset → Augment → Collate → Validator** (待添加)
- **[026] Tensor 拼接: stack vs cat 的区别** (待添加)

---

## 成功标志

当看到以下输出时,bug 已完全修复:

```
Small (<32×32):   15.20%  (vs RemDet 21.3%, Gap: -28.6%)
Medium (32~64):   35.80%  (vs RemDet N/A)
Large (>64×64):   52.30%  (vs RemDet N/A)
```

**预期改进**: RGB-D 的 mAP_small 应比 RGB-only **提升 2-3%** (如果深度信息有效)

---

**文档版本**: v2.1  
**最后更新**: 2025/01/XX  
**修复人**: AI Copilot
