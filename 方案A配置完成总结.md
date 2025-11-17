# 方案 A 配置完成总结

## 📅 完成时间

2025 年 (当前会话)

## 🎯 核心决策

### 问题 1: 训练方式选择

**答案**: **方案 A - 分开训练**

**理由**:

1. RemDet 论文 Table 1 (VisDrone) 和 Table 2 (UAVDT) 模型规格不同
   - Table 1: Tiny/S/M/L/X (5 个模型)
   - Table 2: 仅 L (1 个模型)
2. 分开训练能够清晰对齐两个表格的评估结果
3. 避免类别数不一致带来的复杂性 (VisDrone 10 类 vs UAVDT 3 类)

### 问题 2: 评估标准

**答案**: **COCO 风格评估 (pycocotools)**

**关键指标** (对齐 RemDet):

- AP@0.50:0.95 (IoU=0.50:0.95, area=all)
- AP@0.50 (IoU=0.50, area=all) ← **主要对比指标**
- AP@0.75 (IoU=0.75, area=all)
- AP_small (IoU=0.50:0.95, area=small) ← **UAV 关键指标**
- AP_medium (IoU=0.50:0.95, area=medium)
- AP_large (IoU=0.50:0.95, area=large)

### 问题 3: 数据集处理

- **VisDrone**: 保持 10 类,无需预处理
- **UAVDT**: 只保留 car/truck/bus 三类,重映射 ID 为 0/1/2

---

## 📁 已创建/修改的文件

### 数据集配置文件

#### 1. `data/visdrone-rgbd.yaml` ✅

- **状态**: 已更新为方案 A 配置
- **路径**: `/data2/user/2024/lzy/Datasets/VisDrone2019-DET-YOLO/VisDrone2YOLO`
- **类别数**: 10 (pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor)
- **训练集**: 6,471 张
- **验证集**: 548 张
- **RemDet 基线**:
  - RemDet-X: AP@0.5=48.3%, AP_s=19.5%
  - **目标**: AP@0.5=50-51%, AP_s=21-22%

#### 2. `data/uavdt-rgbd.yaml` ✅

- **状态**: 新创建
- **路径**: `/data2/user/2024/lzy/Datasets/UAVDT_YOLO`
- **类别数**: 3 (car, truck, bus)
- **类别映射**: car→0, truck→1, bus→2 (对应 VisDrone 的 3/5/8)
- **训练集**: 23,258 张
- **验证集**: 15,069 张
- **RemDet 基线**:
  - RemDet-L: AP@0.5=34.5%, AP_s=12.6%
  - **目标**: AP@0.5=36-37%, AP_s=14-15%

### 训练与验证脚本

#### 3. `train_uav_joint.py` (已存在,可复用)

- **状态**: 可直接用于方案 A,只需指定不同的--data 参数
- **功能**:
  - ✅ RGB-D 双模态数据加载
  - ✅ RemDet 对齐的超参数 (SGD, lr=0.01, mosaic=1.0, mixup=0.15)
  - ✅ 多模型尺寸支持 (n/s/m/l/x)
  - ✅ 自定义 callbacks (TAL topk warmup, loss gain scheduling)

#### 4. `val_coco_eval.py` ✅

- **状态**: 新创建 (核心评估脚本)
- **功能**:
  1. 从 YOLO 标注生成 COCO 格式 GT JSON
  2. 运行 YOLO 验证并转换预测为 COCO 格式
  3. 使用 pycocotools 计算所有 COCO 指标
  4. 自动对比 RemDet baseline (Table 1 或 Table 2)
  5. 生成详细的性能报告
- **输出**:
  - 完整的 6 个核心指标
  - 与 RemDet 的差异 (Δ 值)
  - COCO JSON 文件 (可选,供手动检查)

### 辅助工具脚本

#### 5. `filter_uavdt_labels.py` ✅

- **状态**: 新创建
- **功能**:
  - 过滤 UAVDT 标注,只保留 car/truck/bus
  - 重新映射类别 ID 为 0/1/2
  - 复制对应的 RGB 图片
  - 统计类别分布和数据完整性
- **使用**:
  ```bash
  python filter_uavdt_labels.py \
      --src /data2/.../UAVDT_YOLO_raw \
      --dst /data2/.../UAVDT_YOLO \
      --verbose
  ```

#### 6. `diagnose_dataset.py` ✅

- **状态**: 新创建
- **功能**:
  - 检查 RGB/深度图/标注文件的数量和对应关系
  - 验证图片尺寸一致性
  - 检查标注格式正确性 (类别 ID 范围、坐标范围)
  - 分析深度图质量 (值范围、零像素比例)
  - 统计类别分布和平衡性
- **使用**:
  ```bash
  python diagnose_dataset.py --data data/visdrone-rgbd.yaml --split val
  python diagnose_dataset.py --data data/uavdt-rgbd.yaml --split val
  ```

### 文档

#### 7. `方案A_分开训练指南.md` ✅

- **状态**: 新创建 (完整部署指南)
- **内容**:
  - 方案概述与依据
  - 训练目标 (对齐 RemDet Table 1 & 2)
  - 完整训练流程 (4 个 Phase)
    - Phase 1: 环境准备
    - Phase 2: 快速测试 (10 epochs)
    - Phase 3: 完整训练 (300 epochs)
    - Phase 4: COCO 评估与对比
  - 常见问题与解决方案
  - 实验记录模板
  - 八股知识点: 分开训练 vs 联合训练

#### 8. `是否联合训练的分析与决策.md` (之前创建)

- **状态**: 已存在
- **内容**: 详细的方案 A vs 联合训练对比分析

---

## 🔧 技术架构

### 数据流

```
原始数据
  ├─ RGB图片 (jpg/png)
  ├─ 深度图 (png, 需要生成)
  └─ YOLO标注 (txt)
       ↓
[diagnose_dataset.py] ← 验证数据完整性
       ↓
[train_uav_joint.py]
  ├─ YOLORGBDDataset 加载RGB+Depth
  ├─ RGBDStem 早期融合 (4通道输入)
  ├─ RGBDMidFusion 中期融合 (特征级)
  └─ DetectionModel (YOLO12架构)
       ↓
训练 (300 epochs)
  ├─ Optimizer: SGD (lr=0.01, momentum=0.937)
  ├─ Augmentation: Mosaic (1.0) + MixUp (0.15)
  └─ Loss: box_loss + cls_loss + dfl_loss
       ↓
最佳模型 (best.pt)
       ↓
[val_coco_eval.py]
  ├─ 生成COCO GT JSON
  ├─ 生成预测 JSON
  ├─ pycocotools.COCOeval
  └─ 输出6个核心指标
       ↓
结果对比
  ├─ VisDrone vs RemDet Table 1
  └─ UAVDT vs RemDet Table 2
```

### 融合策略 (当前实现)

```
RGB Input (3通道)         Depth Input (1通道)
      ↓                        ↓
  [Conv3x3]               [Conv3x3]
      ↓                        ↓
  RGB特征                  Depth特征
      └──────── Concat ──────┘
                 ↓
          融合特征 (64通道)
                 ↓
         [RGBDStem输出]
                 ↓
          Backbone继续处理
```

---

## 📊 预期性能目标

### VisDrone Benchmark

| Model                  | AP@0.5:0.95  | AP@0.5       | AP@0.75      | AP_s         | AP_m         | AP_l         |
| ---------------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| RemDet-X (baseline)    | 29.9         | 48.3         | 31.0         | 19.5         | 44.1         | 58.6         |
| **yoloDepth-n (目标)** | **31-32**    | **50-51**    | **32-33**    | **21-22**    | **46-47**    | **60-61**    |
| **预期提升**           | **+1.1~2.1** | **+1.7~2.7** | **+1.0~2.0** | **+1.5~2.5** | **+1.9~2.9** | **+1.4~2.4** |

**核心关注**:

- **AP@0.5**: 主要对比指标,目标超越 RemDet-X 2 个百分点
- **AP_small**: 小目标检测,UAV 场景的核心挑战

### UAVDT Benchmark

| Model                  | AP@0.5:0.95  | AP@0.5       | AP@0.75      | AP_s         | AP_m         | AP_l         |
| ---------------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| RemDet-L (baseline)    | 20.6         | 34.5         | 20.5         | 12.6         | 29.0         | 46.8         |
| **yoloDepth-n (目标)** | **22-23**    | **36-37**    | **22-23**    | **14-15**    | **31-32**    | **48-49**    |
| **预期提升**           | **+1.4~2.4** | **+1.5~2.5** | **+1.5~2.5** | **+1.4~2.4** | **+2.0~3.0** | **+1.2~2.2** |

**核心关注**:

- **AP@0.5**: 目标超越 RemDet-L 2 个百分点
- **AP_small**: UAVDT 小目标更难,提升 1.5-2.5 个点即为显著进步

---

## ⚙️ 训练命令速查

### Phase 2: 快速测试 (10 epochs)

```bash
# VisDrone 测试
python train_uav_joint.py \
    --data data/visdrone-rgbd.yaml \
    --cfg n --batch 16 --epochs 10 \
    --name visdrone_rgbd_n_10ep_test

# UAVDT 测试
python train_uav_joint.py \
    --data data/uavdt-rgbd.yaml \
    --cfg n --batch 16 --epochs 10 \
    --name uavdt_rgbd_n_10ep_test
```

### Phase 3: 完整训练 (300 epochs)

```bash
# VisDrone 完整训练
python train_uav_joint.py \
    --data data/visdrone-rgbd.yaml \
    --cfg n --batch 16 --epochs 300 \
    --imgsz 640 --device 0 --workers 8 \
    --name visdrone_rgbd_n_300ep \
    --cache ram \
    --optimizer SGD --lr0 0.01 --momentum 0.937 --weight_decay 0.0005 \
    --mosaic 1.0 --mixup 0.15 --close_mosaic 10

# UAVDT 完整训练
python train_uav_joint.py \
    --data data/uavdt-rgbd.yaml \
    --cfg n --batch 16 --epochs 300 \
    --imgsz 640 --device 0 --workers 8 \
    --name uavdt_rgbd_n_300ep \
    --cache ram \
    --optimizer SGD --lr0 0.01 --momentum 0.937 --weight_decay 0.0005 \
    --mosaic 1.0 --mixup 0.15 --close_mosaic 10
```

### Phase 4: COCO 评估

```bash
# VisDrone COCO 评估
python val_coco_eval.py \
    --weights runs/train/visdrone_rgbd_n_300ep/weights/best.pt \
    --data data/visdrone-rgbd.yaml \
    --name visdrone_coco_eval_final \
    --batch 16 --device 0 --save-json

# UAVDT COCO 评估
python val_coco_eval.py \
    --weights runs/train/uavdt_rgbd_n_300ep/weights/best.pt \
    --data data/uavdt-rgbd.yaml \
    --name uavdt_coco_eval_final \
    --batch 16 --device 0 --save-json
```

---

## 🚨 关键注意事项

### 1. UAVDT 标注预处理 (必须!)

**问题**: UAVDT 原始数据包含 5 类 (car/truck/bus/group/person)
**解决**: 运行 `filter_uavdt_labels.py` 过滤为 3 类

```bash
python filter_uavdt_labels.py \
    --src /data2/user/2024/lzy/Datasets/UAVDT_YOLO_raw \
    --dst /data2/user/2024/lzy/Datasets/UAVDT_YOLO \
    --verbose
```

### 2. 深度图生成

**要求**: 所有 RGB 图片必须有对应的深度图
**工具**: `run_depth_anything_v2_I_mode.py`

```bash
# VisDrone
python run_depth_anything_v2_I_mode.py \
    --images /data2/.../VisDrone2YOLO/train/images/rgb \
    --output /data2/.../VisDrone2YOLO/train/images/d \
    --batch 16 --device 0

# UAVDT (同理)
```

### 3. pycocotools 安装

**Windows**: `pip install pycocotools-windows`
**Linux**: `pip install pycocotools`

### 4. 显存管理

- **batch=16**: 需要 ~16-20GB 显存 (RTX 4090)
- **显存不足**: 使用 `--batch 8 --accumulate 2` 或 `--cache False`

### 5. 训练时间估计

- **VisDrone (6,471 张, 300 epochs)**: ~18-20 小时 (RTX 4090, cache=ram)
- **UAVDT (23,258 张, 300 epochs)**: ~60-70 小时 (RTX 4090, cache=ram)

---

## 📝 接下来的工作

### 立即执行 (本地)

1. ✅ **已完成**: 创建所有必要的配置文件和脚本
2. ⏳ **待执行**: 在本地验证所有脚本的语法正确性
   ```bash
   python -m py_compile filter_uavdt_labels.py
   python -m py_compile diagnose_dataset.py
   python -m py_compile val_coco_eval.py
   ```

### 远程服务器操作

#### Step 1: 上传代码 (本地 → 服务器)

```bash
# 将 yoloDepth/ 整个目录上传到服务器
scp -r yoloDepth/ user@server:/path/to/project/
```

#### Step 2: 预处理 UAVDT (服务器)

```bash
cd /path/to/project/yoloDepth
python filter_uavdt_labels.py \
    --src /data2/user/2024/lzy/Datasets/UAVDT_YOLO_raw \
    --dst /data2/user/2024/lzy/Datasets/UAVDT_YOLO \
    --verbose
```

#### Step 3: 生成深度图 (服务器)

```bash
# VisDrone train
python run_depth_anything_v2_I_mode.py \
    --images /data2/.../VisDrone2YOLO/train/images/rgb \
    --output /data2/.../VisDrone2YOLO/train/images/d \
    --batch 16 --device 0

# VisDrone val
python run_depth_anything_v2_I_mode.py \
    --images /data2/.../VisDrone2YOLO/val/images/rgb \
    --output /data2/.../VisDrone2YOLO/val/images/d \
    --batch 16 --device 0

# UAVDT train
python run_depth_anything_v2_I_mode.py \
    --images /data2/.../UAVDT_YOLO/train/images/rgb \
    --output /data2/.../UAVDT_YOLO/train/images/d \
    --batch 16 --device 0

# UAVDT val
python run_depth_anything_v2_I_mode.py \
    --images /data2/.../UAVDT_YOLO/val/images/rgb \
    --output /data2/.../UAVDT_YOLO/val/images/d \
    --batch 16 --device 0
```

#### Step 4: 数据完整性检查 (服务器)

```bash
# 检查 VisDrone
python diagnose_dataset.py --data data/visdrone-rgbd.yaml --split train
python diagnose_dataset.py --data data/visdrone-rgbd.yaml --split val

# 检查 UAVDT
python diagnose_dataset.py --data data/uavdt-rgbd.yaml --split train
python diagnose_dataset.py --data data/uavdt-rgbd.yaml --split val
```

#### Step 5: 快速测试 (服务器)

```bash
# VisDrone 10 epochs 测试
python train_uav_joint.py \
    --data data/visdrone-rgbd.yaml \
    --cfg n --batch 16 --epochs 10 \
    --name visdrone_rgbd_n_10ep_test

# UAVDT 10 epochs 测试
python train_uav_joint.py \
    --data data/uavdt-rgbd.yaml \
    --cfg n --batch 16 --epochs 10 \
    --name uavdt_rgbd_n_10ep_test
```

#### Step 6: 验证 COCO 评估 (服务器)

```bash
# 测试 pycocotools
python val_coco_eval.py \
    --weights runs/train/visdrone_rgbd_n_10ep_test/weights/best.pt \
    --data data/visdrone-rgbd.yaml \
    --name test_coco_eval
```

#### Step 7: 完整训练 (服务器)

```bash
# 启动 VisDrone 300 epochs (后台运行)
nohup python train_uav_joint.py \
    --data data/visdrone-rgbd.yaml \
    --cfg n --batch 16 --epochs 300 \
    --name visdrone_rgbd_n_300ep \
    --cache ram \
    > logs/visdrone_train.log 2>&1 &

# 启动 UAVDT 300 epochs (后台运行)
nohup python train_uav_joint.py \
    --data data/uavdt-rgbd.yaml \
    --cfg n --batch 16 --epochs 300 \
    --name uavdt_rgbd_n_300ep \
    --cache ram \
    > logs/uavdt_train.log 2>&1 &
```

#### Step 8: 最终评估与对比 (服务器)

```bash
# VisDrone 最终评估
python val_coco_eval.py \
    --weights runs/train/visdrone_rgbd_n_300ep/weights/best.pt \
    --data data/visdrone-rgbd.yaml \
    --name visdrone_final_eval \
    --save-json

# UAVDT 最终评估
python val_coco_eval.py \
    --weights runs/train/uavdt_rgbd_n_300ep/weights/best.pt \
    --data data/uavdt-rgbd.yaml \
    --name uavdt_final_eval \
    --save-json
```

---

## 🎯 成功标准

### 短期目标 (1-2 周)

- [ ] UAVDT 标注预处理完成
- [ ] 所有深度图生成完毕
- [ ] 数据完整性检查通过
- [ ] 10 epochs 快速测试成功
- [ ] pycocotools 评估正常运行

### 中期目标 (1 个月)

- [ ] VisDrone 300 epochs 训练完成
- [ ] UAVDT 300 epochs 训练完成
- [ ] VisDrone AP@0.5 ≥ 50% (vs RemDet-X 48.3%)
- [ ] UAVDT AP@0.5 ≥ 36% (vs RemDet-L 34.5%)

### 长期目标 (2-3 个月)

- [ ] 所有指标超越 RemDet baseline
- [ ] 小目标检测性能提升显著 (AP_small +2 个点)
- [ ] 完成消融实验 (RGB-only vs RGB-D)
- [ ] 撰写论文投稿

---

## 📚 相关文档索引

1. **方案 A\_分开训练指南.md** - 完整训练流程和常见问题
2. **是否联合训练的分析与决策.md** - 方案对比分析
3. **八股.md** - 知识点汇总 (持续更新)
4. **改进记录.md** - 实验日志 (待更新)
5. **data/visdrone-rgbd.yaml** - VisDrone 配置
6. **data/uavdt-rgbd.yaml** - UAVDT 配置
7. **val_coco_eval.py** - COCO 评估脚本
8. **filter_uavdt_labels.py** - UAVDT 预处理工具
9. **diagnose_dataset.py** - 数据诊断工具

---

## ✅ 当前会话完成清单

- ✅ 精简 `data/visdrone-rgbd.yaml` 为方案 A 专用配置
- ✅ 创建 `data/uavdt-rgbd.yaml` (3 类配置)
- ✅ 创建 `val_coco_eval.py` (完整 COCO 评估)
- ✅ 创建 `filter_uavdt_labels.py` (UAVDT 预处理)
- ✅ 创建 `diagnose_dataset.py` (数据诊断)
- ✅ 创建 `方案A_分开训练指南.md` (完整部署文档)
- ✅ 创建 TODO 列表 (12 项任务)
- ✅ 创建本总结文档

**总计新增/修改文件**: 8 个

---

## 🚀 现在可以开始训练了!

所有必要的配置文件、脚本和文档已经准备完毕。请按照 `方案A_分开训练指南.md` 中的流程,逐步执行以下操作:

1. **本地验证**: 检查所有新创建脚本的语法
2. **上传代码**: 将 yoloDepth/ 上传到服务器
3. **预处理数据**: 运行 filter_uavdt_labels.py
4. **生成深度**: 运行 run_depth_anything_v2_I_mode.py
5. **诊断检查**: 运行 diagnose_dataset.py
6. **快速测试**: 10 epochs 验证流程
7. **完整训练**: 300 epochs 获取最终结果
8. **COCO 评估**: 对比 RemDet baseline

**祝训练顺利! 期待超越 RemDet 的好成绩!** 🎉
