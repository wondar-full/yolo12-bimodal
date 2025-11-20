# 八股知识点 #46: CopyPaste 数据增强详解

> **创建时间**: 2025-11-20  
> **适用场景**: 目标检测、实例分割，尤其是小目标检测  
> **本项目应用**: 提升 VisDrone 的 AP_s 和 AP_m

---

## 一、CopyPaste 是什么？

### 1.1 定义

**CopyPaste (复制-粘贴增强)** 是一种专门针对目标检测和实例分割的数据增强方法，由 Google 在 2020 年的论文中提出：

**论文**: _Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation_  
**核心思想**: 从一张图像中复制目标实例，粘贴到另一张图像中，增加训练样本的多样性。

### 1.2 工作原理

```
原始流程（无CopyPaste）：
Image_A → Model → Predictions_A
Image_B → Model → Predictions_B

使用CopyPaste：
Image_A + Image_B → [复制A中的目标，粘贴到B] → Image_B' → Model → Predictions_B'
```

**具体步骤**：

1. **选择源图像**: 从训练集中随机选择一张图像 A
2. **提取目标**: 从图像 A 中随机选择若干个目标实例（带 mask 和 bbox）
3. **选择目标图像**: 选择另一张图像 B
4. **粘贴**: 将提取的目标粘贴到图像 B 的随机位置
5. **更新标注**: 更新图像 B 的标注框和类别信息

**示例**（伪代码）：

```python
def copy_paste_augmentation(image_A, image_B, labels_A, labels_B, p=0.5):
    """
    CopyPaste数据增强

    Args:
        image_A: 源图像 (H, W, 3)
        image_B: 目标图像 (H, W, 3)
        labels_A: 源图像的标注 [(class, x, y, w, h), ...]
        labels_B: 目标图像的标注 [(class, x, y, w, h), ...]
        p: 概率（0.0-1.0）

    Returns:
        augmented_image_B: 增强后的图像
        augmented_labels_B: 增强后的标注
    """
    if random.random() > p:
        return image_B, labels_B  # 不应用CopyPaste

    # 1. 从image_A中随机选择要复制的目标（例如50%的目标）
    num_objects_to_copy = int(len(labels_A) * 0.5)
    objects_to_copy = random.sample(labels_A, num_objects_to_copy)

    # 2. 对每个要复制的目标
    for obj in objects_to_copy:
        # 2.1 提取目标区域（使用mask或bbox）
        x, y, w, h = obj['bbox']
        obj_region = image_A[y:y+h, x:x+w]  # 裁剪目标区域

        # 2.2 在image_B中找到合适的粘贴位置（避免与现有目标重叠）
        paste_x, paste_y = find_valid_position(image_B, labels_B, w, h)

        # 2.3 粘贴到image_B
        image_B[paste_y:paste_y+h, paste_x:paste_x+w] = obj_region

        # 2.4 更新标注
        new_label = (obj['class'], paste_x, paste_y, w, h)
        labels_B.append(new_label)

    return image_B, labels_B
```

---

## 二、为什么 CopyPaste 对小目标有效？

### 2.1 增加小目标的出现频率

**问题**: 在 VisDrone 等 UAV 数据集中，虽然小目标占比 50%，但每张图像中小目标数量有限。

**CopyPaste 的解决方案**:

- 从多张图像中收集小目标
- 粘贴到单张图像中，增加单张图像的目标密度
- 模型可以在更密集的场景中学习

**示例**:

```
原始图像A: 包含3个小目标
原始图像B: 包含2个小目标

CopyPaste后:
图像B': 包含2个原有小目标 + 1个从A复制的小目标 = 3个小目标
```

### 2.2 增加小目标的多样性

**问题**: 小目标通常在固定背景下出现（如天空中的无人机）。

**CopyPaste 的解决方案**:

- 将小目标粘贴到不同背景
- 增加目标与背景的组合
- 提升模型的泛化能力

**示例**:

```
原始: 无人机总是在蓝天背景
CopyPaste后: 无人机可能出现在建筑物、树木、道路等不同背景
```

### 2.3 缓解类别不平衡

**问题**: 某些小目标类别（如"person"）数量远少于其他类别。

**CopyPaste 的解决方案**:

- 可以选择性地复制少数类别的目标
- 平衡类别分布

---

## 三、在 Ultralytics/YOLOv12 中的使用

### 3.1 参数说明

**主要参数**:

- **`copy_paste`**: 概率值（0.0-1.0）

  - `0.0`: 禁用 CopyPaste（默认）
  - `0.1`: 10%的训练图像会应用 CopyPaste
  - `0.5`: 50%的训练图像会应用 CopyPaste
  - `1.0`: 所有训练图像都应用 CopyPaste

- **`copy_paste_mode`**: 粘贴策略
  - `"flip"`: 翻转粘贴（默认）- 复制的目标会被随机翻转
  - `"mixup"`: 混合粘贴 - 使用 MixUp 的方式混合

**在代码中的使用**:

**方式 1: 命令行参数**（推荐）

```bash
python train_depth_solr_v2.py \
    --cfg n \
    --data visdrone-rgbd.yaml \
    --copy_paste 0.1 \
    --epochs 100
```

**方式 2: 在 Python 代码中**

```python
from ultralytics import YOLO

model = YOLO("yolo12n.yaml")
model.train(
    data="visdrone-rgbd.yaml",
    epochs=100,
    copy_paste=0.1,  # 10%概率应用CopyPaste
    copy_paste_mode="flip",
)
```

**方式 3: 修改配置文件**

```yaml
# ultralytics/cfg/default.yaml
copy_paste: 0.1 # 原来是0.0
copy_paste_mode: flip
```

### 3.2 推荐设置

**针对 VisDrone 数据集**:

```python
copy_paste = 0.1  # 保守设置，避免过度增强
```

**原因**:

- VisDrone 已经有 6,471 张训练图像，数据量足够
- 过高的 copy_paste 概率可能导致**过拟合增强策略**
- 0.1 的概率在 COCO 等大型数据集上已验证有效

**如果数据量较小**（如只有 1000 张图像）:

```python
copy_paste = 0.3  # 可以适当提高
```

---

## 四、CopyPaste vs 其他增强方法

### 4.1 与 Mosaic 对比

| 特性           | Mosaic             | CopyPaste            |
| -------------- | ------------------ | -------------------- |
| **原理**       | 拼接 4 张图像      | 复制目标到另一张图像 |
| **目标密度**   | 增加 4 倍          | 适度增加             |
| **背景多样性** | 低（拼接边界明显） | 高（自然融合）       |
| **小目标效果** | 中等               | **更好**             |
| **适用场景**   | 通用目标检测       | 小目标检测           |

**结论**: Mosaic 和 CopyPaste 可以**同时使用**，互补优势。

### 4.2 与 MixUp 对比

| 特性           | MixUp              | CopyPaste          |
| -------------- | ------------------ | ------------------ |
| **原理**       | 像素级混合两张图像 | 对象级复制粘贴     |
| **标注处理**   | 软标签（加权平均） | 硬标签（直接叠加） |
| **目标边界**   | 模糊               | 清晰               |
| **小目标效果** | 一般               | **更好**           |

**结论**: CopyPaste 更适合小目标，因为保留了清晰的边界。

---

## 五、在本项目中的应用

### 5.1 为什么需要 CopyPaste？

**当前问题**:

- YOLO12-N+SOLR 的 AP_s=9.95%（vs RemDet-Tiny 12.7%，差 2.75%）
- YOLO12-L+SOLR 的 AP_s=18.77%（vs RemDet-L 18.7%，已持平）

**RemDet 使用了 CopyPaste**:

- 论文中明确提到使用了 CopyPaste 增强
- 这是 RemDet 在小目标上表现好的原因之一

**我们当前的增强策略**:

```python
mosaic = 1.0     # ✅ 已使用
mixup = 0.15     # ✅ 已使用
copy_paste = 0.0 # ❌ 未使用（默认禁用）
```

**改进后**:

```python
mosaic = 1.0
mixup = 0.15
copy_paste = 0.1 # ✅ 新增
```

### 5.2 预期效果

**保守估计**（基于 COCO 和 VisDrone 的实验）:

- **AP_s 提升**: +0.5~1.0%
- **AP_m 提升**: +0.5~1.0%
- **AP_l 影响**: ±0~0.3%（基本不变）

**具体到 YOLO12-N**:

```
原始: AP_s=9.95%, AP_m=29.61%
加入CopyPaste后: AP_s≈10.5-11%, AP_m≈30-31%
```

### 5.3 实验计划

**Phase 1: 单独测试 CopyPaste 效果**

```bash
# 基线（无CopyPaste）
python train_depth_solr_v2.py --cfg n --epochs 100 --name baseline

# 加入CopyPaste
python train_depth_solr_v2.py --cfg n --copy_paste 0.1 --epochs 100 --name copypaste_0.1
```

**Phase 2: 与 medium_weight 调参结合**

```bash
# 最优配置
python train_depth_solr_v2.py --cfg n \
    --medium_weight 2.5 \
    --copy_paste 0.1 \
    --epochs 100 --name solr_optimized
```

**Phase 3: 完整训练 300 epochs**

```bash
python train_depth_solr_v2.py --cfg n \
    --medium_weight 2.5 \
    --copy_paste 0.1 \
    --epochs 300 --name final_300ep
```

---

## 六、面试八股

### Q1: 什么是 CopyPaste 增强？

**标准答案**:
CopyPaste 是一种目标检测/实例分割的数据增强方法，通过从一张图像中复制目标实例（连同标注），粘贴到另一张图像中，增加训练样本的多样性。它由 Google 在 2020 年提出，在 COCO 等大型数据集上验证有效，尤其对小目标检测有显著提升。

**追问 1: 为什么对小目标有效？**

1. **增加出现频率**: 从多张图像收集小目标，粘贴到单张图像，增加单张图像的目标密度
2. **增加背景多样性**: 小目标可以出现在不同背景下，提升泛化能力
3. **缓解类别不平衡**: 可以选择性复制少数类别的目标

**追问 2: CopyPaste 和 Mosaic 有什么区别？**

- **Mosaic**: 拼接 4 张图像，增加目标密度 4 倍，但拼接边界明显
- **CopyPaste**: 复制单个目标，自然融合，更适合小目标
- **结论**: 两者可以同时使用，互补优势

### Q2: 如何在 YOLOv8/YOLOv12 中使用 CopyPaste？

**标准答案**:

```python
# 方式1: 命令行
python train.py --copy_paste 0.1

# 方式2: 代码
model.train(data="dataset.yaml", copy_paste=0.1)

# 参数说明:
# copy_paste=0.0 → 禁用（默认）
# copy_paste=0.1 → 10%图像应用CopyPaste
# copy_paste=1.0 → 所有图像应用CopyPaste
```

**追问: 如何选择 copy_paste 的值？**

- **大数据集**（>5000 张）: 0.1~0.2（保守）
- **中数据集**（1000-5000 张）: 0.2~0.3
- **小数据集**（<1000 张）: 0.3~0.5（激进）
- **经验**: 过高的概率可能导致过拟合增强策略

### Q3: CopyPaste 的潜在问题？

**标准答案**:

1. **不自然的组合**: 复制的目标可能与背景不协调（如雪地上出现沙漠车辆）
2. **遮挡问题**: 粘贴的目标可能遮挡原有目标，需要处理遮挡关系
3. **计算开销**: 增加训练时间（需要额外的复制粘贴操作）

**解决方案**:

1. **智能选择粘贴位置**: 避免与现有目标重叠
2. **背景一致性检查**: 只在相似背景间复制粘贴
3. **控制应用概率**: 不要过度使用（推荐 0.1-0.3）

---

## 七、总结

### 核心要点

1. ✅ **CopyPaste 是专门针对小目标的有效增强**
2. ✅ **RemDet 使用了 CopyPaste，这是我们需要对齐的**
3. ✅ **推荐设置**: `copy_paste=0.1`（保守但有效）
4. ✅ **可以与 Mosaic、MixUp 同时使用**
5. ✅ **预期提升**: AP_s 和 AP_m 各提升 0.5-1%

### 立即行动

```bash
# 测试CopyPaste效果
python train_depth_solr_v2.py --cfg n \
    --data visdrone-rgbd.yaml \
    --copy_paste 0.1 \
    --epochs 100 \
    --name test_copypaste

# 结合medium_weight调参
python train_depth_solr_v2.py --cfg n \
    --data visdrone-rgbd.yaml \
    --medium_weight 2.5 \
    --copy_paste 0.1 \
    --epochs 100 \
    --name solr_mw2.5_cp0.1
```

### 参考资料

1. **原始论文**: _Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation_ (CVPR 2021)
2. **Ultralytics 文档**: https://docs.ultralytics.com/usage/cfg/#augmentation
3. **RemDet 论文**: AAAI 2025（使用了 CopyPaste）

---

**文档结束** - 现在您可以正确使用`--copy_paste`参数了！
