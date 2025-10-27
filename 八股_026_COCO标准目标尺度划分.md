# 八股\_026_COCO 标准目标尺度划分

## 标准问答

### Q1: COCO 评估中的目标尺度是如何划分的?

**标准答案**:
COCO (Common Objects in Context) 标准将目标划分为 3 个尺度:

- **Small**: area < 32² = 1024 pixels² (小于 32×32 像素)
- **Medium**: 32² ≤ area < 96² = 9216 pixels² (32×32 ~ 96×96 像素)
- **Large**: area ≥ 96² = 9216 pixels² (大于等于 96×96 像素)

**代码实现** (pycocotools/cocoeval.py):

```python
class Params:
    def setDetParams(self):
        self.areaRng = [
            [0**2, 1e5**2],      # all
            [0**2, 32**2],        # small
            [32**2, 96**2],       # medium
            [96**2, 1e5**2]       # large
        ]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
```

---

### Q2: 为什么 COCO 选择 32² 和 96² 作为阈值?

**标准答案**:
基于 MS COCO 数据集的统计分析设计,目标是让 3 个类别的样本量相对均衡:

**COCO 数据集分布**:

- Small (< 32²): **约 41%** - 远景目标、人群中的个体
- Medium (32²~96²): **约 34%** - 正常距离的目标
- Large (≥ 96²): **约 24%** - 近景、大型目标 (汽车、沙发等)

**设计原则**:

1. **均衡性**: 避免某个类别样本过多/过少
2. **区分度**: 三个类别间有明显的检测难度差异
3. **可解释性**: 32px 和 96px 是视觉上容易理解的尺度

---

### Q3: UAV/VisDrone 场景下的尺度分布是否适合 COCO 标准?

**标准答案**:
VisDrone (无人机视角) 数据集的分布与 COCO 显著不同,但**依然推荐使用 COCO 标准阈值**:

**VisDrone 数据集分布** (基于统计分析):

- Small (< 32²): **约 68.2%** ← 远高于 COCO 的 41%!
- Medium (32²~96²): **约 26.5%** ← 略低于 COCO 的 34%
- Large (≥ 96²): **约 5.3%** ← 远低于 COCO 的 24%

**为什么仍用 COCO 标准?**

**优点**:

1. ✅ **学术规范**: RemDet 等 AAAI/CVPR 论文都使用 COCO 标准
2. ✅ **工具兼容**: pycocotools, mmdetection 等直接支持
3. ✅ **可比性**: 与其他 UAV 检测工作直接对比
4. ✅ **审稿认可**: 审稿人熟悉 COCO 标准,易于接受

**缺点**:

1. ⚠️ Large 类别样本太少 (~5%),评估意义有限
2. ⚠️ Small 类别占比过高,可能主导整体 mAP

**应对策略**:

- 重点关注 **mAP_small** (主要挑战,68%的数据)
- Medium/Large 作为辅助参考
- 论文中说明分布差异,强调 small object 性能

---

### Q4: COCO 标准的评估方法是什么?

**标准答案**:
COCO 使用 **ignore mechanism** (忽略机制),而非直接过滤:

**COCO 方法** (标准做法):

```python
# 只过滤GT,不过滤Pred
for gt in gts:
    if gt['area'] < areaRng[0] or gt['area'] > areaRng[1]:
        gt['_ignore'] = 1  # 标记为ignore,不直接删除
    else:
        gt['_ignore'] = 0  # 有效GT

# 计算TP时: 所有Pred vs 有效GT
# - 与ignore GT匹配的Pred不算TP,也不算FP
# - 与其他尺度GT匹配的Pred算作FP
```

**我们的方法** (Phase 2.5 v2.2):

```python
# 同时过滤GT和Pred
gt_small_mask = gt_areas < 1024
pred_small_mask = pred_areas < 1024

# 重新计算TP: small Pred vs small GT
tp_small = match_predictions(
    pred_cls[pred_small_mask],
    gt_cls[gt_small_mask],
    iou[gt_small_mask][:, pred_small_mask]
)
```

**方法对比**:

| 方面          | COCO 标准                   | 我们的实现               | 差异影响     |
| ------------- | --------------------------- | ------------------------ | ------------ |
| **GT 过滤**   | ✅ 按 area 标记 ignore      | ✅ 按 area 过滤          | 一致         |
| **Pred 过滤** | ❌ 不过滤                   | ✅ 按 area 过滤          | ⚠️ 不同      |
| **语义**      | "通用检测器对 X 尺度的性能" | "X 尺度专用检测器的性能" | 更严格       |
| **FP 计算**   | 大目标误检算 FP             | 不考虑跨尺度误检         | 我们可能偏高 |

**举例说明**:

场景: 3 个行人(小目标) + 2 辆车(大目标),模型预测 5 个框

- 2 个行人框 (小)
- 2 个车框 (大)
- 1 个误检框 (预测为行人,但实际是车,small pred)

**COCO 评估 Small mAP**:

```python
# 用所有5个预测框 vs 3个行人GT (车GT被ignore)
tp_small.shape = (5, 10)
# - 2个行人框: TP
# - 2个车框: FP (与ignore GT匹配,不算)
# - 1个误检框: FP (检测了大目标当小目标)
```

**我们的评估**:

```python
# 只用3个small预测框 (2个行人框 + 1个误检框) vs 3个行人GT
tp_small.shape = (3, 10)
# - 2个行人框: TP
# - 1个误检框: FP
# (2个车框不参与评估)
```

**结果**: 我们的 mAP_small 可能略高 (不惩罚大目标误检为小目标)

---

## 本项目应用

### 配置文件

**ultralytics/cfg/default.yaml**:

```yaml
# VisDrone-specific settings (COCO-aligned)
visdrone_mode: False
small_thresh: 1024 # 32×32 (COCO standard)
medium_thresh: 9216 # 96×96 (COCO standard, was 4096)
```

**val_visdrone.py**:

```python
DEFAULT_CONFIG = {
    'small_thresh': 1024,    # COCO standard
    'medium_thresh': 9216,   # COCO standard (changed from 4096)
    ...
}
```

### 显示输出

```
VisDrone-style evaluation initialized (COCO-aligned):
  Small objects: area < 1024 pixels² (<32×32)
  Medium objects: 1024 ≤ area < 9216 pixels² (32×32 ~ 96×96)
  Large objects: area ≥ 9216 pixels² (≥96×96)

📐 By Object Size (COCO-aligned):
  Size Range           Our Model       RemDet-X        Gap
  -------------------- --------------- --------------- --------------------
  Small (<32×32)       15.47%          21.3%           -5.83% (-27.4%)
  Medium (32~96)       39.15%          N/A             N/A
  Large (>96×96)       49.82%          N/A             N/A
```

---

## 深入讲解

### 1. 为什么面积用平方而非边长?

**原因**:

- **旋转不变**: 目标可能倾斜,边长难以定义
- **形状泛化**: 长条形目标 (车辆) 和方形目标 (人) 都能统一评估
- **COCO API**: `gt['area']` 直接从 segmentation mask 计算,已经是面积

**示例**:

```python
# 矩形目标: 30×40 像素
area = 30 * 40 = 1200  # > 1024 → medium
# 虽然宽度30 < 32,但面积超过阈值

# 正方形目标: 31×31 像素
area = 31 * 31 = 961  # < 1024 → small
# 虽然接近32,但面积未超过阈值
```

### 2. 边界情况如何处理?

**COCO 标准**:

```python
# 严格小于/大于等于
if gt['area'] < 32**2:
    # small
elif gt['area'] < 96**2:
    # medium
else:
    # large

# 边界值 area = 1024 → medium (不是small)
# 边界值 area = 9216 → large (不是medium)
```

**统计影响**:

- 边界值目标占比很小 (~0.1%)
- 对整体 mAP 影响可忽略
- 确保分类一致性 (不同评估工具结果相同)

### 3. 为什么 v2.3 改动只涉及阈值?

**v2.2 → v2.3 变化**:

```python
# 只改一个数字
medium_thresh: 4096 → 9216

# 影响:
# - Small: 不变 (< 1024)
# - Medium: 扩大 (1024~4096 → 1024~9216)
# - Large: 收缩 (≥4096 → ≥9216)
```

**目标样本迁移**:

```
旧划分:
Small:  area < 1024      → 60% 目标
Medium: 1024 ~ 4096      → 30% 目标
Large:  ≥ 4096           → 10% 目标

新划分 (COCO):
Small:  area < 1024      → 60% 目标 (不变)
Medium: 1024 ~ 9216      → 35% 目标 ↑ (吸收5%)
Large:  ≥ 9216           → 5% 目标 ↓ (流失5%)
```

**mAP 变化预测**:

- mAP_small: **不变** (阈值相同)
- mAP_medium: **↑ 2-3%** (包含更多中等目标,整体更容易)
- mAP_large: **↓ 2-3%** (只剩最大目标,样本少且可能更难)

---

## 常见追问

### Q: RemDet 论文用的是什么阈值?

**A**: RemDet 使用标准 COCO 评估,配置文件显示:

```python
# config_remdet/yolov8/yolov8_s_remdet-300e_visdrone.py
val_evaluator = dict(
    type='mmdet.CocoMetric',  # ← 标准COCO评估
    metric='bbox'
)
```

mmdetection 的 CocoMetric 默认使用:

- Small: < 32² = 1024
- Medium: 32² ~ 96² = 9216
- Large: ≥ 96²

---

### Q: 为什么不设计 UAV 专用的阈值?

**A**: 曾经考虑过,但有 3 个问题:

1. **可比性丧失**: 无法与 RemDet 等工作直接对比
2. **主观性**: 阈值选择缺乏客观依据 (为什么是 64 而非 60 或 70?)
3. **审稿障碍**: 需要大量篇幅解释新阈值的合理性

**最佳实践**:

- 采用 COCO 标准阈值 (学术规范)
- 在论文中说明 UAV 场景的分布特点
- 强调 small object 性能 (主要挑战)

---

### Q: 如果要完全对齐 COCO 评估,需要改什么?

**A**: 需要修改 TP 计算逻辑,采用 ignore 机制:

**当前方法** (v2.2):

```python
# val.py::_process_batch()
# 同时过滤GT和Pred
tp_small = match_predictions(
    pred[pred_small_mask],
    gt[gt_small_mask],
    iou_small
)
```

**COCO 方法** (需要改动):

```python
# 1. 只过滤GT,标记ignore
gt_ignore_mask = (gt_areas < 1024) | (gt_areas >= 9216)
gt['_ignore'] = gt_ignore_mask

# 2. 修改match_predictions,支持ignore
def match_predictions(..., gt_ignore):
    # 与ignore GT匹配的Pred: 不算TP,也不算FP
    # 与有效GT匹配的Pred: 正常计算TP
    ...

# 3. 所有Pred参与评估 (不过滤)
tp_small = match_predictions(
    all_pred,        # ← 所有预测框
    gt_small,        # ← 只有small GT (其他被ignore)
    iou_all_small
)
```

**改动成本**: 需要修改 match_predictions 函数,工作量 1-2 天

**收益**: 与 COCO 100%对齐,论文 Evaluation 部分可以直接说"We follow COCO protocol"

**当前决策**: 保持 v2.2 方法,在论文中说明差异 (时间成本 vs 收益权衡)

---

## 易错点

| 易错点        | 错误理解                 | 正确理解            | 检验方法                                                               |
| ------------- | ------------------------ | ------------------- | ---------------------------------------------------------------------- |
| **阈值单位**  | 32 和 96 是边长          | 1024 和 9216 是面积 | `32**2 = 1024`                                                         |
| **边界值**    | area=1024 是 small       | area=1024 是 medium | 用 `<` 不用 `<=`                                                       |
| **Pred 过滤** | COCO 也过滤 Pred         | COCO 只标记 GT      | 读源码`cocoeval.py`                                                    |
| **mAP 单位**  | metrics['mAP_s']是百分比 | 是 0-1 小数         | 需要\*100 显示                                                         |
| **面积计算**  | 手动 w\*h                | 从 bbox 计算        | `target_areas = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])` |

---

## 拓展阅读

1. **COCO 官方论文**: [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312)

   - Section 5.2: Detection Evaluation
   - 解释为什么选择 32² 和 96²

2. **pycocotools 源码**: `cocoeval.py::Params`

   - `setDetParams()` - 阈值定义
   - `evaluateImg()` - ignore 机制实现

3. **RemDet 论文**: AAAI2025

   - Table 2: 使用标准 COCO 评估
   - 对比时注意方法一致性

4. **mmdetection 文档**: [COCO Metrics](https://mmdetection.readthedocs.io/en/latest/user_guides/train.html#coco-metrics)
   - CocoMetric 实现细节
   - 如何解读 mAP_s/m/l

---

## 思考题

1. **计算题**: 一个 32×40 像素的矩形目标,在 COCO 标准下属于哪个类别?
   <details>
   <summary>答案</summary>
   area = 32 * 40 = 1280 > 1024 → Medium
   </details>

2. **对比题**: 为什么我们的方法可能得到比 COCO 更高的 mAP_small?
   <details>
   <summary>答案</summary>
   我们不计算跨尺度误检 (如检测大目标为小目标),COCO会惩罚这类错误
   </details>

3. **实践题**: 如何验证 VisDrone 数据集的尺度分布 (68%/27%/5%)?
   <details>
   <summary>答案</summary>
   ```python
   import json
   with open('annotations/VisDrone2019-DET_val_coco.json') as f:
       data = json.load(f)
   areas = [ann['area'] for ann in data['annotations']]
   small = sum(1 for a in areas if a < 1024)
   medium = sum(1 for a in areas if 1024 <= a < 9216)
   large = sum(1 for a in areas if a >= 9216)
   print(f"Small: {small/len(areas)*100:.1f}%")
   ```
   </details>

4. **设计题**: 如果要设计一个 16×16 和 64×64 的三级划分,会有什么问题?
   <details>
   <summary>答案</summary>
   - 16×16可能太小,很多标注误差就会影响分类
   - 64×64在UAV场景下Large类别样本太少 (< 3%)
   - 与COCO标准不兼容,无法对比
   </details>
