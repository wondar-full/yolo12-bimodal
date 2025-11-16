# 🐛 尺度分类 Bug 完整修复方案

## 问题诊断

### Bug 根源

在`yoloDepth`项目中，存在**两处独立的尺度分类 Bug**：

#### Bug 1: dataset.py 中使用错误的图像尺寸

**位置**: `ultralytics/data/dataset.py` line ~292

**错误代码**:

```python
# 如果是归一化坐标,需要乘以图像尺寸才能得到像素面积
if normalized:
    img_h, img_w = label.get("ori_shape", label.get("resized_shape", (640, 640)))[:2]  # ❌ BUG!
    widths = widths * img_w
    heights = heights * img_h

target_areas = (widths * heights).astype(np.float32)
```

**问题分析**:

1. 使用了`ori_shape`（原始图像尺寸，例如 2000×1500）
2. 但 bbox 在数据增强后已经是**相对于 640×640 的归一化坐标**
3. 计算出的`target_areas`严重偏大（例如：32×32 的小目标被错误计算为 100×75 = 7500 像素 ²）

#### Bug 2: val.py 中预测框使用归一化面积

**位置**: `ultralytics/models/yolo/detect/val.py` line ~361-365

**错误代码**:

```python
# Pred框尺寸分类 (根据预测框自己的面积)
pred_widths = preds["bboxes"][:, 2] - preds["bboxes"][:, 0]  # ❌ 归一化坐标!
pred_heights = preds["bboxes"][:, 3] - preds["bboxes"][:, 1]  # ❌ 归一化坐标!
pred_areas = pred_widths * pred_heights  # ❌ 归一化面积 (0-1)
pred_small_mask = pred_areas < small_thresh  # ❌ 比较 0.01 < 1024 → True
```

**问题分析**:

1. 预测框 bbox 是归一化坐标（0-1 范围）
2. 计算出的面积是归一化面积（例如 0.0025, 0.01, 0.04）
3. 但阈值是像素面积（1024, 9216）
4. **几乎所有预测框都被判定为 small**（因为 0.04 < 1024）

## 🔧 完整修复方案

### 方案 1: 修复 dataset.py（推荐 ✅）

这是**最根本的修复**，直接使用正确的图像尺寸：

```python
# ultralytics/data/dataset.py line ~290-296

# ❌ 原始代码
if normalized:
    img_h, img_w = label.get("ori_shape", label.get("resized_shape", (640, 640)))[:2]
    widths = widths * img_w
    heights = heights * img_h

target_areas = (widths * heights).astype(np.float32)

# ✅ 修复代码
if normalized:
    # 🔧 Bug Fix: 使用resize后的图像尺寸,而非原始尺寸
    # 在验证时,bbox归一化是相对于resize后的尺寸(通常640×640)
    img_h, img_w = label.get("resized_shape", (640, 640))[:2]

    # 如果resized_shape不存在,尝试从img获取
    if "img" in label and label["img"] is not None:
        img_h, img_w = label["img"].shape[:2]

    widths = widths * img_w
    heights = heights * img_h

target_areas = (widths * heights).astype(np.float32)
```

**优点**:

- 一处修复，GT 和 Pred 统一使用正确的面积
- 符合数据流的语义（resize 后的 bbox → resize 后的尺寸）

### 方案 2: 修复 val.py 中的预测框分类

```python
# ultralytics/models/yolo/detect/val.py line ~361-370

# ❌ 原始代码
pred_widths = preds["bboxes"][:, 2] - preds["bboxes"][:, 0]
pred_heights = preds["bboxes"][:, 3] - preds["bboxes"][:, 1]
pred_areas = pred_widths * pred_heights
pred_small_mask = pred_areas < small_thresh
pred_medium_mask = (pred_areas >= small_thresh) & (pred_areas < medium_thresh)
pred_large_mask = pred_areas >= medium_thresh

# ✅ 修复代码
# 🔧 Bug Fix: 获取图像尺寸,将归一化面积转换为像素面积
img_shape = batch["img"].shape  # [B, C, H, W]
img_h, img_w = img_shape[2], img_shape[3]  # 通常是640×640

# 计算预测框的像素级宽高和面积
pred_widths = (preds["bboxes"][:, 2] - preds["bboxes"][:, 0]) * img_w  # 转换为像素
pred_heights = (preds["bboxes"][:, 3] - preds["bboxes"][:, 1]) * img_h  # 转换为像素
pred_areas = pred_widths * pred_heights  # 像素面积

# 尺度判断 (现在可以正确比较)
pred_small_mask = pred_areas < small_thresh  # 1024 pixels²
pred_medium_mask = (pred_areas >= small_thresh) & (pred_areas < medium_thresh)  # 1024~9216
pred_large_mask = pred_areas >= medium_thresh  # >=9216
```

## 📋 实施步骤

### Step 1: 修改 dataset.py

**文件**: `yoloDepth/ultralytics/data/dataset.py`
**行号**: ~290-296

修改后的完整代码：

```python
if len(bboxes) > 0:
    # 计算bbox宽高
    if bbox_format == "xyxy":
        widths = bboxes[:, 2] - bboxes[:, 0]
        heights = bboxes[:, 3] - bboxes[:, 1]
    elif bbox_format == "xywh":
        widths = bboxes[:, 2]
        heights = bboxes[:, 3]
    else:
        widths = np.zeros(len(bboxes))
        heights = np.zeros(len(bboxes))

    # 🔧 Bug Fix: 使用正确的图像尺寸
    if normalized:
        # 优先使用resized_shape (验证时bbox是相对于resize后的尺寸)
        img_h, img_w = label.get("resized_shape", (640, 640))[:2]

        # 如果resized_shape不存在,尝试从img获取实际尺寸
        if "img" in label and label["img"] is not None:
            img_h, img_w = label["img"].shape[:2]

        widths = widths * img_w
        heights = heights * img_h

    target_areas = (widths * heights).astype(np.float32)
else:
    target_areas = np.array([], dtype=np.float32)

label["target_areas"] = target_areas
```

### Step 2: 修改 val.py

**文件**: `yoloDepth/ultralytics/models/yolo/detect/val.py`
**行号**: ~361-370

修改后的代码：

```python
# Pred框尺寸分类
# 🔧 Bug Fix: 将归一化坐标转换为像素坐标
img_shape = batch["img"].shape  # [B, C, H, W]
img_h, img_w = img_shape[2], img_shape[3]

pred_widths = (preds["bboxes"][:, 2] - preds["bboxes"][:, 0]) * img_w
pred_heights = (preds["bboxes"][:, 3] - preds["bboxes"][:, 1]) * img_h
pred_areas = pred_widths * pred_heights  # 像素面积

pred_small_mask = pred_areas < small_thresh
pred_medium_mask = (pred_areas >= small_thresh) & (pred_areas < medium_thresh)
pred_large_mask = pred_areas >= medium_thresh
```

### Step 3: 上传到服务器

```powershell
# 在本地PowerShell执行
cd f:\CV\Paper\yoloDepth\yoloDepth

# 上传修复后的dataset.py
scp ultralytics/data/dataset.py ubuntu@10.16.62.111:/data2/user/2024/lzy/yolo12-bimodal/ultralytics/data/

# 上传修复后的val.py
scp ultralytics/models/yolo/detect/val.py ubuntu@10.16.62.111:/data2/user/2024/lzy/yolo12-bimodal/ultralytics/models/yolo/detect/
```

### Step 4: 重新验证

```bash
# 在服务器上执行
cd /data2/user/2024/lzy/yolo12-bimodal
conda activate lzy-yolo12

# 验证YOLO12n (Phase 3)
sh val_depth.sh

# 预期结果:
# Small mAP < Medium mAP < Large mAP (正常分布)
# 实际性能指标将更准确
```

## 📊 预期效果

### 修复前 (错误的尺度分类)

```
Small objects  - mAP50: 13.30%  ❌ (错误地包含了大量medium/large目标)
Medium objects - mAP50: 10.98%  ❌ (错误分类)
Large objects  - mAP50: 14.48%  ❌ (错误分类)
趋势: Small ≈ Large > Medium (混乱)
```

### 修复后 (正确的尺度分类)

```
Small objects  - mAP50: ~10%    ✅ (真实的小目标性能)
Medium objects - mAP50: ~25%    ✅ (中等目标更容易检测)
Large objects  - mAP50: ~40%    ✅ (大目标最容易)
趋势: Small < Medium < Large (正常分布!)
```

### 关键指标变化

| 指标        | 修复前 | 修复后(预期) | 说明                         |
| ----------- | ------ | ------------ | ---------------------------- |
| Small mAP   | 13.30% | ~10%         | 会降低(真实 small 更难)      |
| Medium mAP  | 10.98% | ~25%         | 会提升(不再混入错误分类)     |
| Large mAP   | 14.48% | ~40%         | 会大幅提升(大目标本来就容易) |
| Overall mAP | 34.96% | ~35%         | 总体 mAP 基本不变            |

## 🎯 八股知识点总结

### 归一化坐标的陷阱

**问题**: 为什么归一化坐标要乘以正确的图像尺寸?

**答案**:

```python
# YOLO数据流
原始图像 (2000×1500)
  ↓ resize
resize图像 (640×640)
  ↓ normalize
归一化bbox (0-1范围, 相对于640×640)
  ↓ 计算面积
需要乘以 640×640, 而非 2000×1500!

# 错误示例
bbox_norm = [0.1, 0.1, 0.15, 0.15]  # 相对于640×640
width_pixel_wrong = 0.05 * 2000 = 100 pixels  # ❌ 错误!
width_pixel_right = 0.05 * 640 = 32 pixels    # ✅ 正确!

area_wrong = 100 * 75 = 7500 pixels²  # ❌ 被错误判定为large
area_right = 32 * 32 = 1024 pixels²   # ✅ 正确判定为small
```

### 尺度分类的标准

**COCO 标准**:

- Small: area < 32² = 1024 pixels²
- Medium: 1024 ≤ area < 96² = 9216 pixels²
- Large: area ≥ 9216 pixels²

**注意**: 这些阈值是**像素面积**，不是归一化面积！

## ✅ 验证清单

- [ ] 修改 `ultralytics/data/dataset.py` (line ~292)
- [ ] 修改 `ultralytics/models/yolo/detect/val.py` (line ~361-370)
- [ ] 上传到服务器
- [ ] 重新验证 YOLO12n
- [ ] 确认 Small < Medium < Large
- [ ] 对比修复前后的 mAP 分布
- [ ] 更新改进记录.md

## 🚨 常见错误

1. **只修复了 val.py，忘记修复 dataset.py** → GT 面积仍然错误
2. **使用了 ori_shape 而非 resized_shape** → 面积计算错误
3. **忘记转换预测框面积** → GT 正确但 Pred 错误
4. **混淆了像素面积和归一化面积** → 阈值判断错误
