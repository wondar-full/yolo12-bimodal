# UAVDT 数据集使用总结 - 下一步行动清单

**日期**: 2025-10-31  
**任务**: 将 UAVDT 与 VisDrone 联合训练,复现 RemDet 多数据集策略

---

## ✅ 已完成的工作

### 1. 数据集分析 (analyze_uavdt.py)

**UAVDT 数据集关键信息**:

```
位置: yoloDepth/datasets/UAVDT/
格式: COCO JSON (annotations/UAV-benchmark-M-Train.json, 74.5MB)
结构: images/UAV-benchmark-M/{M0101, M0201, ...}/ (30个视频序列)

训练集: 23,829张图像, 422,911个标注 (平均17.75个/张)
类别: 3类 - car(0) 394,633个, truck(1) 17,491个, bus(2) 10,787个

尺寸分布 (关键!):
  Small (<32²):  212,423 (50.2%) ← 比VisDrone少
  Medium (32²-96²): 204,588 (48.4%) ← 几乎一半!
  Large (>96²):   5,900 (1.4%)   ← VisDrone的13倍!
```

**互补性分析**:

| 数据集   | Small | Medium | Large | 总计  | 用途      |
| -------- | ----- | ------ | ----- | ----- | --------- |
| VisDrone | 92.4% | 7.5%   | 0.1%  | ~400K | 主数据集  |
| UAVDT    | 50.2% | 48.4%  | 1.4%  | 422K  | 补充 M/L  |
| 联合     | 71.3% | 27.9%  | 0.8%  | ~822K | **完美!** |

**结论**: UAVDT 不是用来提升 Small 的(VisDrone 已够),而是**补充 Medium 和 Large 样本**!

---

### 2. 转换脚本准备

已创建以下脚本,**随时可以运行**:

#### convert_uavdt_to_yolo.py

- **功能**: COCO JSON → YOLO TXT 格式转换
- **输入**: UAVDT/annotations/UAV-benchmark-M-Train.json
- **输出**: UAVDT_YOLO/train/{images, labels}/
- **类别映射**: car(0)→4, truck(1)→6, bus(2)→9 (对齐 VisDrone)
- **图像重组**: M0101/img1/xxx.jpg → M0101_xxx.jpg
- **预计耗时**: 5-10 分钟

#### generate_depths_uavdt.py

- **功能**: 为 UAVDT 生成 RGB-D 深度图
- **模型**: ZoeDepth (torch.hub)
- **输入**: UAVDT_YOLO/train/images/ (23,829 张)
- **输出**: UAVDT_YOLO/train/depths/ (灰度 PNG)
- **预计耗时**: 4-6 小时 (GPU 加速,~1 秒/张)
- **存储需求**: ~20GB

---

### 3. 文档完善

#### UAVDT 数据集分析与使用方案.md

- **第 1 部分**: 数据集结构完整分析
- **第 2 部分**: RemDet 论文中的使用方式推测
- **第 3 部分**: COCO vs YOLO 格式八股讲解
- **第 4 部分**: 数据预处理流程 (转换+深度生成)
- **第 5 部分**: 多数据集联合训练配置
- **第 6 部分**: 预期性能提升分析

#### 八股.md 新增知识点

- **[知识点 12]**: COCO 格式 vs YOLO 格式标注转换
  - 转换公式、归一化原理、常见易错点
- **[知识点 13]**: 多数据集联合训练策略
  - 加权采样、灾难性遗忘、域对齐

---

## 🚀 下一步行动 (按顺序执行)

### Step 1: 转换 UAVDT 标注 ⏱️ 10 分钟

```bash
cd f:\CV\Paper\yoloDepth\yoloDepth
python convert_uavdt_to_yolo.py
```

**验证**:

```bash
# 检查输出目录
dir UAVDT_YOLO\train\images  # 应该有23,829个.jpg
dir UAVDT_YOLO\train\labels  # 应该有23,829个.txt

# 随机查看一个标注文件
type UAVDT_YOLO\train\labels\M0101_img000001.txt
# 格式: class_id center_x center_y width height (归一化)
# 示例: 4 0.5234 0.3567 0.0456 0.0234
```

---

### Step 2: 生成 UAVDT 深度图 ⏱️ 4-6 小时

**重要**: 这一步需要在**服务器**上运行 (需要 GPU 和 ZoeDepth 环境)

```bash
# 1. 上传脚本到服务器
scp generate_depths_uavdt.py user@server:/path/to/yoloDepth/

# 2. 在服务器上运行 (建议用tmux/screen后台运行)
cd /path/to/yoloDepth
CUDA_VISIBLE_DEVICES=7 python generate_depths_uavdt.py

# 3. 监控进度 (另一个终端)
watch -n 60 'ls UAVDT_YOLO/train/depths/*.png | wc -l'
# 应该每小时增加~3600张
```

**验证**:

```bash
# 检查深度图数量
ls UAVDT_YOLO/train/depths/*.png | wc -l  # 应该=23,829

# 随机查看几张深度图质量
ls UAVDT_YOLO/train/depths/ | head -5
# 用图像查看器打开,应该看到灰度深度图
```

---

### Step 3: 创建联合数据集配置 ⏱️ 5 分钟

#### 方案 A: 简单拼接 (先试这个)

创建 `data/visdrone_uavdt_joint.yaml`:

```yaml
# VisDrone + UAVDT 联合数据集 (简单拼接版本)

path: f:/CV/Paper/yoloDepth/yoloDepth/datasets

# 训练集: 直接拼接两个数据集
train:
  - VisDrone/images/train # 6,471张
  - UAVDT_YOLO/train/images # 23,829张
  # 总计: 30,300张 (vs RemDet的30K+)

# 验证集: 只用VisDrone (对齐RemDet评估)
val: VisDrone/images/val # 548张

# 深度图路径
train_depth:
  - VisDrone/depths/train
  - UAVDT_YOLO/train/depths

val_depth: VisDrone/depths/val

# 类别数 (使用VisDrone的10类)
nc: 10

# 类别名称
names:
  0: ignored
  1: pedestrian
  2: people
  3: bicycle
  4: car # ← UAVDT的car映射到这里
  5: van
  6: truck # ← UAVDT的truck映射到这里
  7: tricycle
  8: awning-tricycle
  9: bus # ← UAVDT的bus映射到这里
  10: motor
# 注意: UAVDT只提供car/truck/bus三类
# 其他7类只有VisDrone提供
```

**优点**: 实现简单,Ultralytics 原生支持
**缺点**: UAVDT 占 80%,VisDrone 仅 20% (可能不平衡)

---

### Step 4: 启动联合训练 ⏱️ 30-40 小时

```bash
# 在服务器上运行
cd /path/to/yoloDepth
CUDA_VISIBLE_DEVICES=7 python train_depth.py \
    --data data/visdrone_uavdt_joint.yaml \
    --epochs 300 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --project runs/train \
    --name exp_joint_visdrone_uavdt_v1 \
    --weights yolo12n.pt \
    --save_period 50 \
    --patience 100 \
    --optimizer AdamW \
    --lr0 0.001 \
    --cos_lr
```

**关键参数说明**:

- `--weights yolo12n.pt`: 从 ImageNet 预训练开始 (不是 VisDrone best.pt!)
- `--epochs 300`: 足够长的训练时间
- `--batch 16`: 根据 RTX 4090 显存调整
- `--save_period 50`: 每 50 epochs 保存一次,方便中途评估

**监控指标** (每天检查):

```bash
# 查看训练日志
tail -f runs/train/exp_joint_visdrone_uavdt_v1/train.log

# 关键指标
# Epoch 50:  Small mAP应该>25%, Large mAP应该>30%
# Epoch 100: Small mAP应该>30%, Large mAP应该>38%
# Epoch 200: Small mAP应该>35%, Large mAP应该>42%
# Epoch 300: 目标达成! Small 35-38%, Large 42-45%
```

---

### Step 5: 验证性能提升 ⏱️ 30 分钟

训练完成后:

```bash
# 在VisDrone验证集上完整评估
python val_depth.py \
    --data data/visdrone_uavdt_joint.yaml \
    --weights runs/train/exp_joint_visdrone_uavdt_v1/weights/best.pt \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --split val \
    --save-json
```

**期望结果对比**:

| 指标        | Baseline (VisDrone only) | 联合训练目标 | RemDet-Tiny | 胜率  |
| ----------- | ------------------------ | ------------ | ----------- | ----- |
| Overall mAP | 41.0%                    | **45-47%**   | 38.9%       | +15%  |
| Small mAP   | 30.94%                   | **35-38%**   | 12.7%       | +180% |
| Medium mAP  | 46.24%                   | **48-50%**   | 33.0%       | +50%  |
| Large mAP   | 36.70%                   | **42-45%**   | 44.5%       | -2%   |
| Params      | 2.70M                    | 2.70M        | ~2.5M       | 持平  |
| FLOPs       | 11.95G                   | 11.95G       | ~12G        | 持平  |

**判断标准**:

- ✅ **成功**: Overall > 45%, Small > 35%, Medium > 48%
- ⚠️ **部分成功**: Overall > 43%, 但 Small 或 Large 未达标 → 考虑加权采样
- ❌ **失败**: Overall < 42% → 检查数据加载、类别映射是否正确

---

## 🔧 可能的问题与解决方案

### 问题 1: 转换时提示"图像不存在"

**原因**: COCO JSON 中的 file_name 可能是`img_mask`或`img1`目录
**解决**: `convert_uavdt_to_yolo.py`已处理,尝试两个路径

### 问题 2: ZoeDepth 加载失败

**原因**: 缺少依赖或网络问题
**解决**:

```bash
pip install timm
# 或手动下载模型权重
wget https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_N.pt
```

### 问题 3: 深度生成速度太慢 (<0.5 秒/张)

**原因**: 没有使用 GPU 或 GPU 利用率低
**解决**:

```bash
# 检查CUDA
python -c "import torch; print(torch.cuda.is_available())"
# 检查GPU负载
nvidia-smi
```

### 问题 4: 训练时 Loss 出现 NaN

**原因**: 可能是类别映射错误或 bbox 超界
**解决**:

```bash
# 检查转换后的标注
python -c "
with open('UAVDT_YOLO/train/labels/M0101_img000001.txt') as f:
    for line in f:
        parts = line.split()
        assert 4 <= int(parts[0]) <= 9  # 类别应该是4,6,9
        assert all(0 <= float(x) <= 1 for x in parts[1:])  # 坐标应该在[0,1]
"
```

### 问题 5: Large mAP 没有提升

**可能原因**:

1. UAVDT 的 Large 样本采样不足 (被 Small 主导)
2. 类别映射错误 (UAVDT 的车辆没有对应到 VisDrone)
3. 训练不充分 (300 epochs 可能不够)

**解决**:

1. 检查训练日志中 UAVDT 图像的采样比例
2. 随机抽查几张 UAVDT 图像的预测结果
3. 如果确认是采样问题,切换到**方案 B: 加权采样**

---

## 📊 预期性能提升分析

### 为什么联合训练会提升性能?

**1. Large 目标样本增加 13 倍**:

```
VisDrone Large: 443个 (0.1%)
UAVDT Large: 5,900个 (1.4%)
联合后: 6,343个 (0.8%) → 13倍提升!
```

**2. Medium 目标样本增加近 4 倍**:

```
VisDrone Medium: ~30,000个 (7.5%)
UAVDT Medium: 204,588个 (48.4%)
联合后: ~234,588个 (28.5%) → 3.7倍提升!
```

**3. 数据多样性提升**:

- VisDrone: 城市/郊区 UAV 视角,10 类目标
- UAVDT: 高速公路/城市道路,3 类车辆
- 联合: 更丰富的场景和尺度变化

**4. 对齐 RemDet 的训练条件**:

- RemDet 使用: VisDrone + UAVDT + (COCO?)
- 我们使用: VisDrone + UAVDT
- 数据量对齐: 30K vs 30K+

### RemDet vs 我们的优势分析

| 维度        | RemDet-Tiny         | 我们的 YOLO12n-RGBD      | 优势方   |
| ----------- | ------------------- | ------------------------ | -------- |
| 输入模态    | RGB 单模态          | **RGB-D 双模态**         | **我们** |
| Small mAP   | 12.7%               | **30.94%** (已碾压+143%) | **我们** |
| Medium mAP  | 33.0%               | **46.24%** (已领先+40%)  | **我们** |
| Large mAP   | 44.5%               | 36.70% (落后 8%)         | RemDet   |
| Overall mAP | 38.9%               | ~41% → **45-47%** (目标) | **我们** |
| 参数量      | ~2.5M               | 2.70M (持平)             | 持平     |
| FLOPs       | ~12G                | 11.95G (略优)            | **我们** |
| 训练数据    | VisDrone+UAVDT+COCO | VisDrone+UAVDT (对齐)    | 持平     |

**结论**:

- Small/Medium 已经领先,联合训练后将进一步扩大优势
- Large 追平后(42-45%),Overall 将全面超越 RemDet-Tiny
- 双模态融合是核心优势,RemDet 无法复现!

---

## 📝 检查清单 (执行前确认)

在开始之前,请确认:

- [x] ✅ UAVDT 数据集存在: `yoloDepth/datasets/UAVDT/`
- [x] ✅ 数据集分析完成: 23,829 张, COCO 格式, 3 类, 尺寸互补
- [x] ✅ 转换脚本准备好: `convert_uavdt_to_yolo.py`
- [x] ✅ 深度生成脚本准备好: `generate_depths_uavdt.py`
- [ ] ⏳ 服务器 ZoeDepth 环境就绪 (需要确认)
- [ ] ⏳ 服务器存储空间充足 (~20GB for depths)
- [ ] ⏳ VisDrone 深度图已生成 (train + val)
- [ ] ⏳ train_depth.py 支持多路径输入 (需要验证)

---

## 🎯 成功标准

### 最低目标 (Must Have)

- Overall mAP > 43% (+5% vs baseline)
- Small mAP > 33% (+7% vs current)
- Large mAP > 40% (+9% vs current)

### 理想目标 (Should Have)

- Overall mAP > 45% (+10%)
- Small mAP > 35% (+13%)
- Medium mAP > 48% (+4%)
- Large mAP > 42% (+14%)

### 超越目标 (Nice to Have)

- Overall mAP > 47% (超越 RemDet-Tiny 8 个点!)
- Small mAP > 38% (超越 RemDet-Tiny 25 个点!)
- 所有指标全面领先 RemDet-Tiny

---

## 📞 需要你确认的信息

1. **ZoeDepth 环境**:

   - 服务器上是否已安装 ZoeDepth?
   - 如果没有,是否可以安装? (`pip install timm`)

2. **存储空间**:

   - 服务器 `/data/` 目录剩余空间是多少?
   - UAVDT 深度图需要约 20GB

3. **训练时间**:

   - 300 epochs 预计 30-40 小时,可以接受吗?
   - 是否需要先用 50 epochs 快速验证?

4. **数据路径**:
   - 服务器上 VisDrone 数据集路径是?
   - 转换后的 UAVDT_YOLO 应该放在哪里?

**请告诉我这些信息,然后我们就可以开始执行了!** 🚀
