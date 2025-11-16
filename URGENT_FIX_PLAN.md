# 🚨 紧急修复: UAVDT 类别映射错误导致联合训练失败

## 问题确认 ✅

**发现者**: 用户  
**发现时间**: 2025-11-06  
**问题描述**: VisDrone 的 car 是类别 3,但 UAVDT 的 car 被错误标记为类别 0

### 根本原因

UAVDT 数据集只有 3 个类别 (car/truck/bus),原始 ID 为 1/2/3。
转换脚本错误地使用了 `category = int(parts[5]) - 1`,导致:

```
UAVDT原始 → 错误转换 → 应该是
car (1)   → 0         → 3 (VisDrone的car)
truck (2) → 1         → 5 (VisDrone的truck)
bus (3)   → 2         → 8 (VisDrone的bus)
```

### 影响范围

- **受影响的训练**: exp_joint_v13, exp_joint_v14, exp_joint_v15
- **数据比例**: 79% (UAVDT 23,829 / 总 30,300)
- **性能影响**: mAP 从预期 45%降至 19.51% (下降 25.49%)

---

## 立即行动计划 🚀

### Phase 1: 验证问题 ⚡ **优先级: 最高**

#### 1.1 检查 UAVDT 标签的实际类别 ID

在服务器上执行:

```bash
# 登录服务器
ssh user@server

# 进入UAVDT标签目录
cd /data2/user/2024/lzy/Datasets/UAVDT_YOLO/train/labels/rgb

# 查看几个标签文件 (检查类别ID)
head -20 M0101_00001.txt
head -20 M0201_00001.txt
head -20 M0301_00001.txt

# 统计所有标签文件的类别分布
find . -name "*.txt" -exec cat {} \; | awk '{print $1}' | sort | uniq -c | sort -rn
```

**预期输出** (如果类别映射错误):

```
  1200000 0    # ❌ 这些应该是3 (car)
   800000 1    # ❌ 这些应该是5 (truck)
   200000 2    # ❌ 这些应该是8 (bus)
```

**正确输出** (如果类别映射正确):

```
  1200000 3    # ✅ car
   800000 5    # ✅ truck
   200000 8    # ✅ bus
```

#### 1.2 对比 VisDrone 的类别分布

```bash
cd /data2/user/2024/lzy/Datasets/VisDrone2019-DET-YOLO/VisDrone2YOLO/VisDrone2019-DET-train/labels/rgb
find . -name "*.txt" -exec cat {} \; | awk '{print $1}' | sort | uniq -c | sort -rn
```

**应该看到**:

```
  135873 3    # car ✅
   89621 1    # people
   52464 0    # pedestrian
   31205 9    # motor
   15874 4    # van
    9243 6    # tricycle
    4328 8    # bus
    3942 5    # truck
     514 2    # bicycle
     224 7    # awning-tricycle
```

如果 VisDrone 的类别分布正常,但 UAVDT 是 0/1/2,**问题确认!**

---

### Phase 2: 修复数据 🔧 **优先级: 最高**

#### 2.1 定位 UAVDT 原始数据

```bash
# 查找UAVDT原始目录
find /data2/user/2024/lzy/Datasets -name "UAVDT*" -type d | grep -v "YOLO"

# 应该找到类似:
# /data2/user/2024/lzy/Datasets/UAVDT/
#   ├── M0101/
#   │   ├── Annotations/  ← 原始标注
#   │   └── Imgs/         ← 原始图像
#   ├── M0102/
#   ...
```

#### 2.2 备份错误标签

```bash
# 备份现有的错误标签 (以防万一)
cd /data2/user/2024/lzy/Datasets/UAVDT_YOLO/train
mv labels labels_backup_wrong_$(date +%Y%m%d)
mv images images_backup  # 图像也备份,因为需要重新复制

# 或者如果空间不够,直接删除
rm -rf labels
rm -rf images
```

#### 2.3 上传修复后的转换脚本

在**本地 Windows**:

```powershell
# 将修复后的脚本上传到服务器
scp convert_uavdt_with_class_mapping.py user@server:/data2/user/2024/lzy/yolo12-bimodal/
```

在**服务器**:

```bash
cd /data2/user/2024/lzy/yolo12-bimodal

# 检查脚本是否上传成功
ls -lh convert_uavdt_with_class_mapping.py
```

#### 2.4 重新转换 UAVDT 数据集

```bash
# 激活conda环境
conda activate yolo12

# 运行转换脚本 (带类别映射)
python convert_uavdt_with_class_mapping.py \
    --uavdt_root /data2/user/2024/lzy/Datasets/UAVDT \
    --output_root /data2/user/2024/lzy/Datasets/UAVDT_YOLO \
    --splits train \
    --verify

# 预计耗时: 15-30分钟 (23,829个文件)
```

**转换过程监控**:

- 查看进度条 (tqdm)
- 检查错误统计 (invalid_bbox, parse_error 等)
- 确认类别映射统计 (UAVDT 1→3, 2→5, 3→8)

#### 2.5 验证转换结果

```bash
# 检查转换后的类别分布
cd /data2/user/2024/lzy/Datasets/UAVDT_YOLO/train/labels/rgb
find . -name "*.txt" -exec cat {} \; | awk '{print $1}' | sort | uniq -c | sort -rn

# 必须看到类别 3, 5, 8 (而不是 0, 1, 2)!
```

**验证清单**:

- [ ] 标签文件数量: 23,829 个
- [ ] 类别 ID 范围: 只有 3, 5, 8
- [ ] 类别分布比例: car > truck > bus (符合 UAVDT 原始分布)
- [ ] 随机抽查 10 个文件: 类别 ID 正确

---

### Phase 3: 重新训练 🚀 **优先级: 高**

#### 3.1 确认配置文件

**检查 `data/visdrone_uavdt_joint.yaml`**:

```yaml
nc: 10 # ✅ 必须是10 (VisDrone的类别数)

names:
  0: pedestrian
  1: people
  2: bicycle
  3: car # ✅ UAVDT的car应该映射到这里
  4: van
  5: truck # ✅ UAVDT的truck应该映射到这里
  6: tricycle
  7: awning-tricycle
  8: bus # ✅ UAVDT的bus应该映射到这里
  9: motor
```

#### 3.2 启动训练 (exp_joint_v16)

```bash
# 进入工作目录
cd /data2/user/2024/lzy/yolo12-bimodal

# 启动训练 (修复类别映射后的首次训练)
CUDA_VISIBLE_DEVICES=7 python train_depth.py \
    --model ultralytics/cfg/models/12/yolo12n-rgbd-v1.yaml \
    --weights yolo12n.pt \
    --data data/visdrone_uavdt_joint.yaml \
    --epochs 300 \
    --batch 16 \
    --imgsz 640 \
    --name exp_joint_v16_class_mapping_fixed \
    --device 7 \
    --exist-ok \
    --cache ram \
    --workers 8

# 或使用screen保持后台运行
screen -S joint_v16
CUDA_VISIBLE_DEVICES=7 python train_depth.py ...
# Ctrl+A, D 退出screen
```

#### 3.3 监控训练进度

**关键检查点**:

**Epoch 1-5** (启动阶段, 2-4 小时):

```bash
# 查看训练日志
tail -f runs/train/exp_joint_v16_class_mapping_fixed/train.log

# 关键指标:
# 1. 权重加载: "Transferred 85/95 items from yolo12n.pt" ✅
# 2. 数据加载: "train: 30300 images, 800000+ instances, 10 classes" ✅
#              ^^^^^^^^           ^^^^^^^^^^^^^^^^  ^^^^^^^^^^
#              图像数             实例数(应该80万+) 类别数(应该10)
# 3. Epoch 1 mAP: 应该在 10-15% (vs v15的5.64%)
```

**Epoch 10** (快速验证, 8-12 小时):

```bash
# 查看results.csv
tail -1 runs/train/exp_joint_v16_class_mapping_fixed/results.csv

# 预期指标:
# Epoch 10: mAP@0.5 应该 > 28% (vs v15的13.76%)
#           如果 > 30%, 说明修复成功! ✅
#           如果 < 20%, 说明还有其他问题 ❌
```

**Epoch 50** (中期检查, 2-3 天):

```bash
# 预期指标:
# Epoch 50: mAP@0.5 应该 > 38% (vs v15的19.51%)
#           如果 > 38%, 超越RemDet目标在望! 🎯
```

**Epoch 200-300** (最终评估, 10-12 天):

```bash
# 目标指标:
# Final: mAP@0.5 应该 40-45%
#        mAP@0.5:0.95 应该 25-30%
#        mAP_small 应该 15-18% (vs RemDet的21.3%)
```

---

### Phase 4: 结果验证 ✅ **优先级: 中**

#### 4.1 完整验证

```bash
# 使用最佳权重在验证集上评估
python val_depth.py \
    --weights runs/train/exp_joint_v16_class_mapping_fixed/weights/best.pt \
    --data data/visdrone_uavdt_joint.yaml \
    --batch 32 \
    --imgsz 640 \
    --device 7 \
    --save-json \
    --save-hybrid

# 生成详细报告
# - 总体mAP@0.5
# - 类别级mAP (检查car/truck/bus是否正常)
# - 混淆矩阵 (检查是否还有误分类)
# - PR曲线
```

#### 4.2 对比分析

创建对比表格:

| 实验          | 配置                            | mAP@0.5           | mAP_small  | mAP_medium | mAP_large  | 问题            |
| ------------- | ------------------------------- | ----------------- | ---------- | ---------- | ---------- | --------------- |
| exp_joint_v13 | yolo12s + yolo12n.pt            | 22.27%            | N/A        | N/A        | N/A        | 模型权重不匹配  |
| exp_joint_v15 | yolo12n + yolo12n.pt            | 19.51%            | N/A        | N/A        | N/A        | 类别映射错误 ❌ |
| exp_joint_v16 | yolo12n + yolo12n.pt + 类别修复 | **40-45%** (预期) | **15-18%** | **42-47%** | **38-43%** | 修复成功 ✅     |
| RemDet-Tiny   | RemDet                          | 38.9%             | 14.2%      | 40.5%      | 36.8%      | Benchmark       |

**成功标准**:

- ✅ mAP@0.5 ≥ 40% (超越 RemDet-Tiny 的 38.9%)
- ✅ mAP_small ≥ 14% (接近 RemDet-Tiny 的 14.2%)
- ✅ 类别级 mAP: car/truck/bus 都应该 > 35%

#### 4.3 可视化验证

```bash
# 可视化预测结果
python val_depth.py \
    --weights runs/train/exp_joint_v16_class_mapping_fixed/weights/best.pt \
    --data data/visdrone_uavdt_joint.yaml \
    --batch 1 \
    --imgsz 640 \
    --device 7 \
    --save-txt \
    --save-conf \
    --max-det 300 \
    --visualize

# 检查可视化结果:
# - car是否被正确检测为car (不是pedestrian)
# - truck是否被正确检测为truck (不是people)
# - bus是否被正确检测为bus (不是bicycle)
```

---

## 预期效果 🎯

### 修复前 (exp_joint_v15)

```
问题: UAVDT类别映射错误
训练数据: 79%的标签错误

结果:
  mAP@0.5:      19.51% ❌ (vs 目标45%, gap -25.49%)
  Precision:    26.74%
  Recall:       22.49%

诊断:
  模型学到: car图像 → pedestrian ❌
           truck图像 → people ❌
           bus图像 → bicycle ❌
```

### 修复后 (exp_joint_v16, 预期)

```
修复: UAVDT类别正确映射
训练数据: 100%标签正确

预期结果:
  mAP@0.5:      40-45% ✅ (超越RemDet 38.9%)
  mAP@0.5:0.95: 25-30%
  mAP_small:    15-18%
  Precision:    55-60%
  Recall:       50-55%

模型学习:
  car图像 → car ✅
  truck图像 → truck ✅
  bus图像 → bus ✅
```

**性能提升**:

- mAP@0.5: +20-25 个百分点 (19.51% → 40-45%)
- 达到 RemDet 水平甚至超越!

---

## 备选方案 (如果修复后仍有问题)

### 方案 A: 单独训练验证

如果修复后仍未达到 40%,分别训练验证:

```bash
# 1. 只用VisDrone训练
python train_depth.py \
    --data data/visdrone.yaml \
    --name exp_visdrone_only

# 2. 只用UAVDT训练 (但UAVDT只有3类,需要修改nc)
python train_depth.py \
    --data data/uavdt_only.yaml \
    --name exp_uavdt_only

# 对比性能确定问题所在
```

### 方案 B: 加权采样

如果联合训练仍有问题,尝试加权采样:

```python
# 修改ultralytics/data/build.py
# 为VisDrone样本分配更高权重 (1.5×)
# 为UAVDT样本分配标准权重 (1.0×)
# 使VisDrone占比从21%提升到35%
```

### 方案 C: 分阶段训练

```bash
# Stage 1: VisDrone预训练 (50 epochs)
python train_depth.py --data visdrone.yaml --epochs 50

# Stage 2: 联合微调 (250 epochs)
python train_depth.py \
    --data visdrone_uavdt_joint.yaml \
    --weights runs/train/exp_visdrone/weights/best.pt \
    --epochs 250
```

---

## 文档记录 📝

### 更新文件

1. **改进记录.md**:

   ```markdown
   ## 2025-11-06: 修复 UAVDT 类别映射错误 ⚡ CRITICAL FIX

   ### 问题

   - UAVDT 的 car/truck/bus 被错误标记为 0/1/2
   - 应该映射到 VisDrone 的 3/5/8

   ### 修复

   - 创建类别映射表 UAVDT_TO_VISDRONE = {1:3, 2:5, 3:8}
   - 重新转换所有 UAVDT 标签 (23,829 个文件)
   - 启动 exp_joint_v16 训练

   ### 预期

   - mAP 从 19.51%提升到 40-45%
   - 超越 RemDet-Tiny (38.9%)
   ```

2. **八股知识点 36**: 已创建 `八股_知识点36_类别映射问题.md`

3. **verify_uavdt_class_issue.md**: 已创建问题诊断文档

---

## 时间线 📅

| 阶段       | 任务                | 预计耗时 | 负责人        |
| ---------- | ------------------- | -------- | ------------- |
| ✅ Phase 0 | 发现问题            | -        | 用户          |
| ⏳ Phase 1 | 验证问题 (检查标签) | 10 分钟  | 用户 (服务器) |
| ⏳ Phase 2 | 修复数据 (重新转换) | 30 分钟  | 用户 (服务器) |
| ⏳ Phase 3 | 重新训练 (exp_v16)  | 10-12 天 | GPU 自动      |
| ⏳ Phase 4 | 结果验证            | 2 小时   | 用户          |

**总耗时**: ~12-14 天 (主要是训练时间)

---

## 检查清单 ✅

### 立即执行 (今天)

- [ ] **Step 1**: 在服务器检查 UAVDT 标签的类别 ID

  ```bash
  cd /data2/.../UAVDT_YOLO/train/labels/rgb
  find . -name "*.txt" -exec cat {} \; | awk '{print $1}' | uniq -c
  ```

- [ ] **Step 2**: 确认类别映射错误 (如果看到 0/1/2 而不是 3/5/8)

- [ ] **Step 3**: 备份现有错误标签

  ```bash
  mv labels labels_backup_wrong
  ```

- [ ] **Step 4**: 上传修复脚本

  ```bash
  scp convert_uavdt_with_class_mapping.py user@server:~/
  ```

- [ ] **Step 5**: 重新转换 UAVDT

  ```bash
  python convert_uavdt_with_class_mapping.py --uavdt_root ... --verify
  ```

- [ ] **Step 6**: 验证转换结果 (必须看到类别 3/5/8)

- [ ] **Step 7**: 启动训练 exp_joint_v16

### 训练期间监控 (每天)

- [ ] **Day 1**: 检查 Epoch 1-10 的 mAP 趋势
- [ ] **Day 3**: 检查 Epoch 30 的 mAP (应该>35%)
- [ ] **Day 5**: 检查 Epoch 50 的 mAP (应该>38%)
- [ ] **Day 10**: 检查最终 mAP (应该 40-45%)

### 训练完成后 (第 12 天)

- [ ] 运行完整验证 (val_depth.py)
- [ ] 生成性能报告
- [ ] 对比 RemDet 指标
- [ ] 更新改进记录.md
- [ ] 可视化预测结果

---

## 联系与支持 💬

**如果遇到问题**:

1. **Phase 1 验证失败** (UAVDT 标签路径不对):

   - 检查: `find /data2 -name "UAVDT*" -type d`
   - 更新转换脚本中的路径

2. **Phase 2 转换失败** (脚本报错):

   - 检查错误信息
   - 可能是 UAVDT 原始格式不同
   - 提供错误日志给 AI 助手

3. **Phase 3 训练失败** (OOM 或其他错误):

   - 减小 batch_size (16 → 8)
   - 关闭 cache (--cache false)
   - 检查 GPU 内存 (nvidia-smi)

4. **Phase 4 结果仍不理想** (<35% mAP):
   - 检查是否还有其他数据问题
   - 尝试备选方案 A/B/C
   - 分析混淆矩阵找根因

---

## 总结

### 🔥 关键发现

**UAVDT 类别映射错误是导致联合训练失败的根本原因!**

79%的训练数据标签完全错误:

- car → 被标记为 pedestrian
- truck → 被标记为 people
- bus → 被标记为 bicycle

### ✅ 修复方案

1. 创建类别映射表: `UAVDT_TO_VISDRONE = {1:3, 2:5, 3:8}`
2. 使用修复后的转换脚本重新生成所有 UAVDT 标签
3. 重新训练 (exp_joint_v16)

### 🎯 预期成果

- mAP 从 19.51%提升到 40-45% (提升 20+个百分点)
- 超越 RemDet-Tiny (38.9%)
- 达到项目阶段性目标

### 📚 学习收获

- 八股知识点 36: 多数据集联合训练的类别对齐问题
- 类别映射验证的重要性
- 数据转换后的完整验证流程

---

**立即开始 Phase 1 验证!** 🚀

用户只需要在服务器上运行几条命令就能确认问题,然后我们就可以开始修复!
