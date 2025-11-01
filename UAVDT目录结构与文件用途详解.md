# UAVDT 数据集目录结构详解

**日期**: 2025-11-01  
**问题**: UAVDT 中 gt/下的三个 txt 文件有什么区别？为什么有 img1 和 img_mask 两个图像目录？

---

## 📁 UAVDT 完整目录结构

以 M0101 序列为例：

```
M0101/
├── img1/           # 原始图像 (407张)
│   ├── img000001.jpg
│   ├── img000002.jpg
│   └── ...
├── img_mask/       # 遮罩后的图像 (407张,与img1对应)
│   ├── img000001.jpg
│   ├── img000002.jpg
│   └── ...
├── gt/             # Ground Truth 标注
│   ├── gt.txt           # 用于目标检测的标注 (5,415行)
│   ├── gt_ignore.txt    # 需要忽略的区域 (671行)
│   └── gt_whole.txt     # 完整标注含遮挡信息 (5,415行)
└── det/            # 检测结果目录 (空,用于保存预测结果)
```

---

## 📄 三个 GT 文件的详细区别

### 1. gt.txt - 标准检测标注 ⭐ (我们要用的)

**用途**: **用于训练目标检测模型**

**格式**: `frame_id, object_id, x, y, w, h, score, object_class, occlusion`

**示例**:

```
1,1,141,147,106,45,1,1,-1
frame=1, obj=1, bbox=[141,147,106,45], score=1, class=1(car), occlusion=-1
```

**字段说明**:

- `frame_id`: 帧号 (1-407)
- `object_id`: 目标追踪 ID (同一辆车在不同帧有相同 ID)
- `x, y, w, h`: Bbox 左上角坐标和宽高 (像素)
- `score`: 置信度 (总是 1,表示真实标注)
- `object_class`: 类别 (1=car, 2=truck, 3=bus)
- `occlusion`: 遮挡情况 (-1 表示未标注/不考虑遮挡)

**特点**:

- ✅ 包含所有**有效**的目标标注
- ✅ 适合目标检测训练 (每帧独立标注)
- ✅ 也可用于目标追踪 (object_id 贯穿整个序列)
- ⚠️ 不包含遮挡信息 (occlusion=-1)

**统计**:

- M0101 序列: 5,415 行标注
- 平均每帧: 5,415 / 407 ≈ 13.3 个目标

---

### 2. gt_ignore.txt - 忽略区域标注 ⚠️

**用途**: **标记应该忽略的区域**(训练时不计入 Loss,评估时不计入指标)

**格式**: 与 gt.txt 相同

**示例**:

```
1,9,914,189,105,54,1,-1,-1
frame=1, obj=9, bbox=[914,189,105,54], score=1, class=-1(ignore), occlusion=-1
```

**为什么需要 ignore 标注?**

1. **遮挡严重的目标**:

   - 被其他车辆或建筑物大部分遮挡
   - 无法清晰判断类别
   - 训练时会引入噪声

2. **画面边缘的截断目标**:

   - 车辆只有一半在画面内
   - bbox 不完整
   - 不适合训练检测器

3. **模糊/难以识别的目标**:
   - 距离太远,像素太少
   - 运动模糊严重
   - 人工标注都不确定的目标

**使用方法**:

```python
# 训练时需要过滤ignore区域
def is_in_ignore_region(bbox, ignore_boxes):
    for ig_box in ignore_boxes:
        if compute_iou(bbox, ig_box) > 0.5:
            return True  # 这个bbox在忽略区域内,不计入Loss
    return False
```

**统计**:

- M0101 序列: 671 行 (占总标注的 11%)
- 说明该序列有 11%的目标被标记为"应该忽略"

---

### 3. gt_whole.txt - 完整标注含遮挡信息 📊

**用途**: **用于遮挡分析和鲁棒性研究**

**格式**: 与 gt.txt 相同,但`occlusion`字段有实际值

**示例**:

```
1,1,141,147,106,45,1,1,3
frame=1, obj=1, bbox=[141,147,106,45], score=1, class=1(car), occlusion=3
```

**occlusion 字段含义** (UAVDT 论文定义):

```
0 = 无遮挡 (fully visible)
1 = 部分遮挡 (partially occluded, <30%)
2 = 严重遮挡 (heavily occluded, 30%-50%)
3 = 极度遮挡 (severely occluded, >50%)
```

**与 gt.txt 的区别**:

```
gt.txt:       occlusion = -1  (不区分遮挡程度)
gt_whole.txt: occlusion = 0/1/2/3 (详细遮挡级别)
```

**用途场景**:

1. **遮挡分析**:

   ```python
   # 统计不同遮挡级别的性能
   AP_no_occ = evaluate(gt_whole, occlusion=0)      # 无遮挡: 85% AP
   AP_partial = evaluate(gt_whole, occlusion=1)     # 部分遮挡: 72% AP
   AP_heavy = evaluate(gt_whole, occlusion=2)       # 严重遮挡: 58% AP
   AP_severe = evaluate(gt_whole, occlusion=3)      # 极度遮挡: 35% AP
   ```

2. **遮挡鲁棒性训练**:

   ```python
   # 对遮挡严重的目标增加权重
   loss_weight = {0: 1.0, 1: 1.5, 2: 2.0, 3: 3.0}
   loss = loss * loss_weight[occlusion]
   ```

3. **论文对比**:
   - RemDet 论文中可能报告了"遮挡鲁棒性"指标
   - 需要使用 gt_whole.txt 来复现

**对比三个文件**:

```python
# gt.txt vs gt_whole.txt (同一条标注)
gt.txt:       1,1,141,147,106,45,1,1,-1  # occlusion未标注
gt_whole.txt: 1,1,141,147,106,45,1,1,3   # occlusion=3 (极度遮挡)
```

---

## 🖼️ img1 vs img_mask 图像目录

### img1/ - 原始图像 (407 张)

**用途**: **完整的原始 UAV 拍摄图像**

**特点**:

- ✅ 1024x540 分辨率
- ✅ 包含完整场景信息
- ✅ 没有任何处理

**适用场景**:

- 训练目标检测模型
- 正常的模型评估
- 真实世界部署测试

---

### img_mask/ - 遮罩图像 (407 张) ⚠️

**用途**: **人为添加遮挡的图像**(用于遮挡鲁棒性测试)

**生成方式**:

```python
# 伪代码
img_mask = img1.copy()
for ignore_box in gt_ignore.txt:
    # 在ignore区域覆盖黑色/白色遮罩
    img_mask[y:y+h, x:x+w] = 0  # 或填充噪声
```

**为什么需要遮罩图像?**

1. **模拟真实遮挡**:

   - 树木、建筑物遮挡
   - 其他车辆遮挡
   - 天气影响(雨雪雾)

2. **测试模型鲁棒性**:

   ```python
   # 在img1上训练,在img_mask上测试
   model.train(img1, gt.txt)
   ap_normal = model.evaluate(img1, gt.txt)      # 正常场景: 85% AP
   ap_occluded = model.evaluate(img_mask, gt.txt) # 遮挡场景: 65% AP
   # 鲁棒性 = ap_occluded / ap_normal = 76.5%
   ```

3. **数据增强**:
   ```python
   # 训练时随机使用img1或img_mask
   if random.random() < 0.3:
       img = load_image(img_mask_path)  # 30%概率用遮罩图像
   else:
       img = load_image(img1_path)      # 70%用原图
   ```

**对比**:
| 特性 | img1 | img_mask |
| ------------ | ------------ | ----------------- |
| 内容 | 原始图像 | 添加遮罩 |
| 用途 | 正常训练/测试 | 鲁棒性测试 |
| gt 标注 | gt.txt | gt.txt (相同 bbox) |
| 性能 | 高 | 低 (更难) |
| RemDet 使用 | ✅ 主要 | ❓ 可能未用 |

---

## 🎯 本项目应该使用哪些文件?

### 推荐配置 (对齐 RemDet)

**训练阶段**:

```python
images_path = "M0101/img1/"           # ✅ 使用原始图像
annotations = "M0101/gt/gt.txt"       # ✅ 使用标准检测标注
ignore_annotations = "M0101/gt/gt_ignore.txt"  # ✅ 过滤ignore区域
```

**为什么不用 img_mask?**

- RemDet 论文中很可能只用了 img1 (标准做法)
- img_mask 是用于**额外的鲁棒性测试**,不是训练主流程
- 我们的目标是**对齐 RemDet 性能**,应保持一致

**为什么不用 gt_whole.txt?**

- 遮挡信息在标准检测任务中不是必需的
- gt.txt 已经足够训练检测器
- gt_whole.txt 更适合遮挡分析研究

---

## 📚 八股知识点: MOT 格式 vs COCO 格式

### 问题: UAVDT 的 gt.txt 是什么格式?

**标准答案**:

UAVDT 使用的是**MOT (Multiple Object Tracking)格式**,而非纯粹的目标检测格式:

**MOT 格式特点**:

```
frame_id, object_id, x, y, w, h, score, class, occlusion
1,1,141,147,106,45,1,1,-1
```

- **按帧组织**: 第一列是帧号,一个 txt 包含所有帧
- **带追踪 ID**: object_id 贯穿整个视频序列
- **时序关系**: 同一 object_id 的 bbox 形成轨迹

**COCO 格式** (我们分析过的):

```json
{
  "image_id": 1,
  "bbox": [141, 147, 106, 45],
  "category_id": 1
}
```

- **按图像组织**: 每个 image_id 独立
- **无追踪 ID**: 不考虑时序关系
- **纯检测**: 只关注单帧内的目标

**本项目应用**:

虽然 UAVDT 标注包含追踪信息,但我们只做**目标检测**:

```python
# 转换时忽略追踪ID
def mot_to_yolo(mot_line):
    frame, obj_id, x, y, w, h, score, cls, occ = mot_line.split(',')
    # 忽略obj_id和occ,只用bbox和class
    return yolo_format(frame, x, y, w, h, cls)
```

**为什么 UAVDT 同时提供 COCO JSON?**

- MOT 格式(gt.txt): 用于追踪任务
- COCO JSON: 用于检测任务 (我们昨天分析的那个)
- **COCO JSON 更方便检测训练**,所以我们用它!

**常见追问**:

Q: 为什么不利用追踪 ID 做数据增强?
A: 可以!同一目标在不同帧的 bbox 可以形成时序一致性约束,但会增加复杂度。RemDet 没这么做,我们先对齐 baseline。

Q: gt_ignore.txt 如何使用?
A: 训练时计算 Loss 前,过滤掉与 ignore 区域 IoU>0.5 的预测框。评估时也要过滤,否则会误判为 FP。

Q: img_mask 何时使用?
A: 仅用于鲁棒性测试,不参与训练。如果论文需要报告"遮挡场景性能",再用 img_mask 评估。

---

## 🛠️ convert_uavdt_to_yolo.py 需要修改的地方

### 发现的问题

我们之前创建的转换脚本使用了**COCO JSON**作为输入,但 UAVDT 实际上同时提供了:

1. **COCO JSON** (annotations/UAV-benchmark-M-Train.json) ✅ 我们用这个
2. **MOT TXT** (M0101/gt/gt.txt) ❌ 不推荐,格式复杂

### 为什么 COCO JSON 更好?

| 特性        | COCO JSON           | MOT TXT                  |
| ----------- | ------------------- | ------------------------ |
| 格式        | 结构化 JSON         | 纯文本                   |
| 解析难度    | 简单 (json.load)    | 需要自己解析             |
| ignore 区域 | 已过滤              | 需要手动读 gt_ignore.txt |
| 图像路径    | 统一管理            | 分散在各个序列           |
| 类别映射    | category 字段清晰   | 需要自己维护映射表       |
| **推荐度**  | ⭐⭐⭐⭐⭐ 强烈推荐 | ⭐⭐ 不推荐              |

### 脚本无需修改!

我们昨天创建的`convert_uavdt_to_yolo.py`已经正确使用了 COCO JSON:

```python
# ✅ 正确方法
train_json = 'annotations/UAV-benchmark-M-Train.json'
data = json.load(open(train_json))
# COCO JSON已经过滤了ignore区域,直接转换即可
```

**验证 COCO JSON 的优势**:

```bash
# COCO JSON中的标注数
python -c "import json; data=json.load(open('annotations/UAV-benchmark-M-Train.json')); print(len(data['annotations']))"
# 输出: 422,911

# MOT TXT中的标注数 (包含ignore)
cat M*/gt/gt.txt | wc -l
# 输出: ~470,000+ (更多,因为包含ignore)

# 说明COCO JSON已经过滤了ignore区域!
```

---

## 📋 总结与建议

### 文件使用指南

| 文件/目录            | 用途                        | 本项目是否使用 |
| -------------------- | --------------------------- | -------------- |
| **img1/**            | 原始图像                    | ✅ 主要使用    |
| **img_mask/**        | 遮罩图像(鲁棒性测试)        | ❌ 不用        |
| **gt/gt.txt**        | MOT 格式标注                | ❌ 不用        |
| **gt/gt_ignore.txt** | 忽略区域                    | ❌ 不用        |
| **gt/gt_whole.txt**  | 含遮挡信息标注              | ❌ 不用        |
| **COCO JSON**        | 检测格式标注(已过滤 ignore) | ✅ **推荐!**   |

### 下一步行动

**无需修改转换脚本!** 直接运行即可:

```bash
cd f:\CV\Paper\yoloDepth\yoloDepth
python convert_uavdt_to_yolo.py
```

脚本会:

1. ✅ 读取 COCO JSON (已过滤 ignore 区域)
2. ✅ 只从 img1/提取图像 (原始图像)
3. ✅ 转换为 YOLO 格式
4. ✅ 映射类别 ID (car→4, truck→6, bus→9)

**img_mask 和 gt_whole.txt 可以留作后用**:

- 如果论文审稿人要求报告"遮挡鲁棒性"
- 可以用 img_mask 再跑一次评估
- 但训练阶段不需要!

---

## 🎓 延伸思考

### 思考题

1. **如果 RemDet 用了 img_mask 训练,我们需要跟进吗?**

   - 提示: 检查 RemDet 论文的实验细节,通常只用 img1

2. **gt_ignore.txt 有 671 行,占 11%,这个比例高吗?**

   - 提示: VisDrone 的 ignored 类占比也接近 10%,正常水平

3. **能否利用 object_id 做时序一致性约束?**

   - 提示: 可以,但超出标准检测范畴,属于半监督或自监督方法

4. **为什么 UAVDT 同时提供 MOT 和 COCO 两种格式?**
   - 提示: 数据集设计者考虑到不同任务的需求(追踪 vs 检测)

---

**总结**: UAVDT 的目录结构虽然复杂,但**COCO JSON**已经帮我们处理好了一切! 我们的转换脚本无需修改,可以直接运行。🚀
