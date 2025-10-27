# Phase 2.5 验证指标对齐 RemDet - 执行指南 (v2.0)

> **完成时间**: 2025/10/27 16:30 (v2.0 更新)  
> **状态**: ✅ 代码已完善(新增 mAP75, Latency, FLOPs),待服务器验证  
> **优先级**: 🔴 CRITICAL (阻塞 Phase 3)  
> **更新内容**: 补充 RemDet 论文完整指标,简化命令行参数

---

## 📋 v2.0 更新内容

### ✅ 新增指标

1. **mAP@0.75**: 评估定位精度(RemDet 论文 Table 2 关键指标)
2. **Latency (ms)**: 推理延迟(含 warmup + 100 次平均)
3. **FLOPs (G)**: 理论计算量(thop 库测量)
4. **Params (M)**: 参数量(模型存储大小)

### ✅ 简化命令行

**之前**: 需要传递 10+个参数 (--data, --batch, --conf, --iou, --max-det, ...)

```bash
# ❌ 太繁琐
python val_visdrone.py \
    --model runs/train/rgbd_v2.1_full/weights/best.pt \
    --data data/visdrone-rgbd.yaml \
    --batch 16 \
    --conf 0.001 \
    --iou 0.45 \
    --max-det 300 \
    --small-thresh 1024 \
    --medium-thresh 4096 \
    --visdrone-mode \
    --plots \
    --name v2.1_eval
```

**现在**: 仅需 --model (其他全部使用 DEFAULT_CONFIG)

```bash
# ✅ 极简,所有配置在代码里
python val_visdrone.py --model runs/train/rgbd_v2.1_full/weights/best.pt
```

### ✅ 完整 RemDet 对比报告

---

## 📋 快速总结

### 核心问题

"yolo 自带的指标还没有完全对其 remdet" - YOLO 默认 COCO 评估,缺少 VisDrone 特定的分尺度 mAP

### 解决方案

1. ✅ 创建`metrics_visdrone.py` - VisDrone 专用评估类
2. ✅ 创建`val_visdrone.py` - RemDet 对齐的验证脚本
3. ⏳ 重新评估 v2.1 和 RGB-only,获得分尺度指标
4. ⏳ 确认 v2.1 的真实 mAP_small (预期 15-18%)

### 预期影响

- **整体mAP@0.5**: 应保持 43.51%±0.5% (核心参数已对齐)
- **新增 mAP_small**: 预期 15-18% (vs RemDet 21.3%, gap -3~-6%)
- **指导 Phase 3**: 如果 mAP_small<15%,优先实现 SOLR Loss

---

## 🚀 立即行动 (今天必须完成)

### Step 1: 修改数据加载器添加面积信息

**问题**: 当前`dataset.py`不返回`target_areas`,导致无法分尺度统计

**修改文件**: `ultralytics/data/dataset.py`

**位置**: `YOLORGBDDataset`类的`get_label_info`方法

**修改内容**:

```python
# 原代码 (约Line 450):
def get_label_info(self, index):
    """返回标签信息."""
    label = self.labels[index]
    # ... existing code ...
    return {
        'bboxes': bboxes,  # [N, 4] xyxy格式
        'cls': cls,        # [N,] 类别索引
        # ... other fields ...
    }

# 修改后:
def get_label_info(self, index):
    """返回标签信息."""
    label = self.labels[index]
    # ... existing code ...

    # 新增: 计算目标面积 (xyxy格式)
    if len(bboxes) > 0:
        w = bboxes[:, 2] - bboxes[:, 0]  # width
        h = bboxes[:, 3] - bboxes[:, 1]  # height
        areas = w * h  # [N,] 面积数组
    else:
        areas = np.array([])

    return {
        'bboxes': bboxes,
        'cls': cls,
        'areas': areas,  # ← 新增字段
        # ... other fields ...
    }
```

**验证方法**:

```python
# 测试代码
from ultralytics.data.dataset import YOLORGBDDataset

dataset = YOLORGBDDataset('data/visdrone-rgbd.yaml', split='val')
sample = dataset[0]
print('Areas:', sample['areas'])  # 应该输出 [1234.5, 567.8, ...] 面积数组
```

### Step 2: 在服务器上运行验证

**命令 1**: 评估 v2.1 RGB-D 模型

```bash
cd /path/to/yoloDepth

python val_visdrone.py \
    --model runs/train/rgbd_v2.1_full/weights/best.pt \
    --data data/visdrone-rgbd.yaml \
    --batch 16 \
    --conf 0.001 \
    --iou 0.45 \
    --max-det 300 \
    --small-thresh 1024 \
    --medium-thresh 4096 \
    --visdrone-mode \
    --plots \
    --name v2.1_remdet_aligned \
    --device 0
```

**预期输出**:

```
runs/val/v2.1_remdet_aligned/
├── results.csv                # mAP@0.5=43.51%, mAP@0.5:0.95=26.49%
├── results_by_size.csv        # mAP_small, mAP_medium, mAP_large
├── remdet_comparison.txt      # vs RemDet-X详细对比
├── PR_curve.png               # 全局PR曲线
├── Small-PR_curve.png         # 小目标PR曲线
├── Medium-PR_curve.png        # 中目标PR曲线
├── Large-PR_curve.png         # 大目标PR曲线
└── confusion_matrix.png
```

**命令 2**: 评估 RGB-only baseline

```bash
python val_visdrone.py \
    --model runs/train/rgb_only/weights/best.pt \
    --data data/visdrone.yaml \
    --batch 16 \
    --conf 0.001 \
    --iou 0.45 \
    --max-det 300 \
    --visdrone-mode \
    --plots \
    --name rgb_remdet_aligned \
    --device 0
```

### Step 3: 对比结果与更新文档

**查看 RemDet 对比报告**:

```bash
cat runs/val/v2.1_remdet_aligned/remdet_comparison.txt
cat runs/val/rgb_remdet_aligned/remdet_comparison.txt
```

**关键验证点**:

1. ✅ mAP@0.5保持 43.51%±0.5% (整体性能不变)
2. ✅ v2.1 mAP_small > RGB-only mAP_small (深度提升小目标)
3. ✅ Gap to RemDet 确认 (指导 Phase 3 优先级)

**更新文档** (如果验证成功):

```bash
# 1. 更新v2.1_performance_analysis.md
#    添加分尺度mAP表格

# 2. 更新改进记录.md
#    在"## 2025/10/27 15:00"条目下添加"实际效果"

# 3. 生成对比图表 (可选)
python plot_size_comparison.py  # 生成Small/Medium/Large mAP对比柱状图
```

---

## 📊 预期结果分析

### Scenario A: mAP_small = 15-18% (理想)

```
✅ 表现: 优于预期
✅ 解读: 深度信息有效提升小目标检测
✅ 行动: 继续Phase 3 (ChannelC2f),预期47% overall mAP

对比RemDet:
- Overall gap: -1.69% (43.51% vs 45.2%)
- Small gap: -3~-6% (15-18% vs 21.3%)
```

### Scenario B: mAP_small = 12-15% (中等)

```
⚠️  表现: 符合预期下限
⚠️  解读: 小目标提升有限,需专项优化
⚠️  行动: 优先Phase 4 (SOLR Loss),目标+3-5% mAP_small

对比RemDet:
- Overall gap: -1.69%
- Small gap: -6~-9% (12-15% vs 21.3%) ← CRITICAL
```

### Scenario C: mAP_small < 12% (低于预期)

```
❌ 表现: 低于预期
❌ 解读: 深度信息未能有效用于小目标
❌ 行动:
   1. 检查深度图质量 (小目标区域)
   2. 检查RGBDMidFusion的attention权重 (是否过低)
   3. 优先实现SOLR Loss + 调整融合权重
```

---

## 🔍 问题诊断清单

### 如果整体mAP@0.5下降>0.5%

```
可能原因:
1. ❌ 数据加载错误 (RGB-D未正确对齐)
2. ❌ 评估参数错误 (conf, iou, max_det)
3. ❌ 模型加载错误 (加载了错误的权重)

诊断方法:
python -c "
from ultralytics import YOLO
model = YOLO('runs/train/rgbd_v2.1_full/weights/best.pt')
print(model.model.model[0])  # 应该是RGBDStem
"
```

### 如果 mAP_small 无输出

```
可能原因:
1. ❌ areas字段缺失 (dataset未修改)
2. ❌ visdrone_mode未启用
3. ❌ stats_by_size为空 (无小目标数据)

诊断方法:
python -c "
from ultralytics.data.dataset import YOLORGBDDataset
ds = YOLORGBDDataset('data/visdrone-rgbd.yaml', split='val')
sample = ds[0]
assert 'areas' in sample, 'Missing areas field!'
print('Areas OK:', sample['areas'][:5])
"
```

### 如果 PR 曲线图缺失

```
可能原因:
1. ❌ --plots参数未传递
2. ❌ save_dir权限问题
3. ❌ matplotlib库未安装

诊断方法:
ls runs/val/v2.1_remdet_aligned/*.png
# 应该看到8个PNG文件 (4个PR + 4个其他)
```

---

## 📝 下一步计划 (Phase 3 准备)

### 如果验证成功 (mAP_small ≥ 15%)

```
Phase 3: ChannelC2f实现
目标: mAP@0.5 45-46% (RemDet-X为45.2%)
时间: 2-3天

具体任务:
1. 实现ChannelC2f模块 (ultralytics/nn/modules/block.py)
2. 创建v3.0 YAML配置 (替换Layer 2/4的C3k2)
3. 10-epoch快速测试
4. 100-epoch完整训练
5. 对比v2.1性能 (预期+1.5-1.8%)
```

### 如果验证显示 mAP_small < 15%

```
Phase 4优先: SOLR Loss实现
目标: mAP_small 18-20% (+3-5%)
时间: 1-2天

具体任务:
1. 实现SOLR loss (ultralytics/utils/loss.py)
2. 修改训练脚本添加SOLR权重
3. 重新训练v2.1 + SOLR
4. 对比小目标性能提升
5. 再决定是否实现ChannelC2f
```

---

## 🎯 成功标准

### 必须达成 (CRITICAL)

- [x] metrics_visdrone.py 创建完成
- [x] val_visdrone.py 创建完成
- [ ] 成功运行验证 (无错误)
- [ ] 获得分尺度 mAP (small/medium/large)
- [ ] 确认整体mAP@0.5在 43-44%范围

### 应该达成 (HIGH)

- [ ] mAP_small ≥ 15% (优于 RGB-only)
- [ ] 生成 8 个 PR 曲线图
- [ ] remdet_comparison.txt 输出正常
- [ ] 更新 v2.1_performance_analysis.md

### 可选达成 (MEDIUM)

- [ ] 绘制分尺度 mAP 对比图
- [ ] 分析每个类别的 small/medium/large mAP
- [ ] 提取 RGBDMidFusion 的 attention 统计

---

## 📚 相关文档

- **改进记录**: `改进记录.md` → "2025/10/27 15:00 — Phase 2.5"
- **八股知识**: `八股.md` → 新增知识点 #017-#021
- **性能分析**: `v2.1_performance_analysis.md` (待更新)
- **验证脚本**: `val_visdrone.py` (新建)
- **评估类**: `ultralytics/utils/metrics_visdrone.py` (新建)

---

## 🤝 需要帮助?

### 常见问题

**Q1: dataset.py 修改后报错 "KeyError: 'areas'"**

```python
# 检查修改是否正确
grep -n "areas" ultralytics/data/dataset.py
# 应该在get_label_info方法中看到areas计算
```

**Q2: val_visdrone.py 报 "ModuleNotFoundError: No module named 'ultralytics.utils.metrics_visdrone'"**

```bash
# 检查文件是否在正确位置
ls ultralytics/utils/metrics_visdrone.py
# 如果不存在,从yoloDepth复制到服务器
```

**Q3: 验证速度太慢**

```bash
# 增大batch size (如果显存足够)
--batch 32  # vs 默认16

# 减少plots生成
--plots false  # 跳过PR曲线绘制

# 使用FP16推理
--half
```

### 联系我

如果遇到以上未覆盖的问题,请提供:

1. 完整错误日志 (stderr 输出)
2. 运行的完整命令
3. 环境信息 (Python 版本, CUDA 版本, ultralytics 版本)

---

## ✅ Checklist (执行前检查)

- [ ] 已修改`dataset.py`添加`areas`字段
- [ ] 已将`metrics_visdrone.py`复制到服务器
- [ ] 已将`val_visdrone.py`复制到服务器
- [ ] 已确认模型权重路径正确
- [ ] 已确认数据集路径正确 (`data/visdrone-rgbd.yaml`)
- [ ] 已分配足够显存 (至少 12GB for batch=16)
- [ ] 已准备好记录结果到文档

---

**最后提醒**: Phase 2.5 是 Phase 3 的前置依赖,必须先确认验证指标对齐,才能开始 ChannelC2f 实现。预计总耗时 2-4 小时 (修改代码 30 分钟 + 验证运行 1-2 小时 + 分析结果 30 分钟)。

🚀 **现在开始执行吧！**
