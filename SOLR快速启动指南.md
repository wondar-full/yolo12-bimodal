# 🚀 SOLR 快速启动指南 - yolo12-bimodal

> **当前项目**: yolo12-bimodal  
> **新增功能**: SOLR (Small Object Loss Reweighting)  
> **目标**: 提升中等目标检测性能,缩小与 RemDet 的差距

---

## ✅ 文件清单

### 已创建文件 (3 个)

| 文件                             | 大小    | 用途               |
| -------------------------------- | ------- | ------------------ |
| `ultralytics/utils/solr_loss.py` | ~600 行 | SOLR 核心实现      |
| `train_depth_solr.py`            | ~500 行 | SOLR 训练脚本      |
| `代码审查报告_SOLR集成.md`       | ~400 行 | 代码审查与问题分析 |

### 文件位置

```
yolo12-bimodal/
├── ultralytics/
│   └── utils/
│       └── solr_loss.py          ← 新增 ✅
├── train_depth_solr.py            ← 新增 ✅
└── 代码审查报告_SOLR集成.md       ← 新增 ✅
```

---

## ⚡ 立即开始 (3 步)

### 第 1 步: 测试 SOLR 模块 (本地, 5 分钟)

```bash
# 在Windows本地测试
cd f:\CV\Paper\yoloDepth\yolo12-bimodal

# 运行单元测试
python -m ultralytics.utils.solr_loss
```

**预期输出**:

```
============================================================
SOLR (Small Object Loss Reweighting) Initialized
============================================================
Size Thresholds:
  Small objects:  < 32px
  Medium objects: 32-96px
  Large objects:  > 96px

Loss Weights:
  Small:  2.5x  ← High priority
  Medium: 2.0x  ← Target RemDet gap
  Large:  1.0x   ← Baseline

Input size: 640×640
============================================================

Testing SOLR Loss Module
============================================================

Test Results:
  Target 1 (small):  weight = 2.50 (expected 2.5)
  Target 2 (medium): weight = 2.00 (expected 2.0)
  Target 3 (large):  weight = 1.00 (expected 1.0)

Statistics:
  Small:  1 (33.3%)
  Medium: 1 (33.3%)
  Large:  1 (33.3%)
  Avg weight: 2.00
  Avg size: 106.7px

============================================================
✅ All tests passed!
============================================================
```

**如果测试通过** → 进入第 2 步  
**如果测试失败** → 检查文件路径,确保`solr_loss.py`在`ultralytics/utils/`目录

---

### 第 2 步: 上传到服务器 (5 分钟)

```bash
# 方式A: scp上传
cd f:\CV\Paper\yoloDepth\yolo12-bimodal

# 上传SOLR模块
scp ultralytics\utils\solr_loss.py user@server:/path/to/yolo12-bimodal/ultralytics/utils/

# 上传训练脚本
scp train_depth_solr.py user@server:/path/to/yolo12-bimodal/

# 方式B: Git提交推送 (推荐)
git add ultralytics/utils/solr_loss.py
git add train_depth_solr.py
git add 代码审查报告_SOLR集成.md
git commit -m "Add SOLR loss for improved medium object detection"
git push

# 服务器端拉取
ssh user@server
cd /path/to/yolo12-bimodal
git pull
```

---

### 第 3 步: 服务器端快速测试 (30-60 分钟)

```bash
# SSH登录服务器
ssh user@server
cd /path/to/yolo12-bimodal

# 测试SOLR导入
python -c "from ultralytics.utils.solr_loss import SOLRLoss; print('✅ SOLR模块导入成功')"

# 启动10 epochs快速测试
python train_depth_solr.py \
    --data data/visdrone-rgbd.yaml \
    --epochs 10 \
    --batch 16 \
    --device 0 \
    --name test_solr_10ep
```

**监控训练**:

```bash
# 实时查看日志
tail -f runs/train/test_solr_10ep/train.log

# 应该看到:
# ✅ SOLR权重已成功集成到损失函数
# ✅ 深度图已加载 ✓
# ✅ Epoch 1/10: ...
```

---

## 📊 完整训练 (测试通过后)

### 启动 300 epochs 训练

```bash
# 使用默认SOLR权重 (small=2.5, medium=2.0, large=1.0)
python train_depth_solr.py \
    --data data/visdrone-rgbd.yaml \
    --epochs 300 \
    --batch 16 \
    --device 0 \
    --optimizer SGD \
    --lr0 0.01 \
    --mosaic 1.0 \
    --mixup 0.15 \
    --name visdrone_n_solr_300ep

# 后台运行 (避免SSH断开)
nohup python train_depth_solr.py \
    --data data/visdrone-rgbd.yaml \
    --epochs 300 \
    --batch 16 \
    --device 0 \
    --name visdrone_n_solr_300ep \
    > train_solr.log 2>&1 &

# 监控
tail -f train_solr.log
```

---

### 自定义 SOLR 权重

```bash
# 如果AP_m提升不足,增加medium_weight到2.5
python train_depth_solr.py \
    --data data/visdrone-rgbd.yaml \
    --epochs 300 \
    --batch 16 \
    --small_weight 2.5 \
    --medium_weight 2.5 \
    --large_weight 1.0 \
    --name visdrone_n_solr_m25

# 如果AP_s太低,增加small_weight到3.0
python train_depth_solr.py \
    --data data/visdrone-rgbd.yaml \
    --epochs 300 \
    --batch 16 \
    --small_weight 3.0 \
    --medium_weight 2.0 \
    --large_weight 1.0 \
    --name visdrone_n_solr_s30
```

---

## 🔍 调试检查点

### 检查点 1: 深度图归一化

```python
# 在train_depth_solr.py中添加 (line ~400,在训练开始前)
# 临时调试代码
class DebugSOLRTrainer(SOLRTrainer):
    def _do_train(self, world_size=1):
        # 在第一个batch打印数据范围
        for i, batch in enumerate(self.train_loader):
            if i == 0:
                print(f"\n{'='*60}")
                print("DEBUG: First batch data ranges")
                print(f"{'='*60}")
                print(f"RGB range: [{batch['img'][:, :3].min():.3f}, {batch['img'][:, :3].max():.3f}]")
                if batch['img'].shape[1] > 3:
                    print(f"Depth range: [{batch['img'][:, 3].min():.3f}, {batch['img'][:, 3].max():.3f}]")
                print(f"bboxes shape: {batch['bboxes'].shape}")
                print(f"bboxes range: [{batch['bboxes'].min():.3f}, {batch['bboxes'].max():.3f}]")
                print(f"{'='*60}\n")
                break

        # 继续正常训练
        super()._do_train(world_size)

# 使用: model.train(..., trainer=DebugSOLRTrainer)
```

**预期输出**:

```
============================================================
DEBUG: First batch data ranges
============================================================
RGB range: [0.000, 1.000]       ← 应该在[0, 1]
Depth range: [0.000, 1.000]     ← 应该在[0, 1],与RGB一致
bboxes shape: torch.Size([N, 4])
bboxes range: [0.000, 1.000]    ← 应该在[0, 1],归一化坐标
============================================================
```

**如果 Depth range 不在[0, 1]**:

- 检查`ultralytics/data/dataset.py`或`augment.py`
- 确认深度图加载时是否除以 255 或 65535

---

### 检查点 2: SOLR 权重应用

```bash
# 查看训练日志,确认SOLR已集成
grep "SOLR" runs/train/test_solr_10ep/train.log

# 应该看到:
# SOLR: Integrating SOLR loss...
# ============================================================
# SOLR (Small Object Loss Reweighting) Initialized
# ============================================================
# SOLR: ✅ SOLR loss integrated successfully!
```

---

### 检查点 3: 性能提升验证

```bash
# 训练完成后,对比baseline
# Baseline (无SOLR): runs/train/visdrone_baseline/results.txt
# SOLR:              runs/train/visdrone_n_solr_300ep/results.txt

# 查看mAP
tail -10 runs/train/visdrone_n_solr_300ep/results.txt

# 或使用COCO评估
python val_coco_eval.py \
    --weights runs/train/visdrone_n_solr_300ep/weights/best.pt \
    --data data/visdrone-rgbd.yaml
```

**预期提升**:

- AP@0.50: +2.5~3.5% (从 32.57% → 35~36%)
- AP_m: +2~3% (从 28.86% → 31~32%)
- AP_s: +1~1.5% (从 9.61% → 10.5~11%)

---

## 🚨 常见问题

### Q1: ImportError: No module named 'solr_loss'

**原因**: `solr_loss.py`未上传或路径错误

**解决**:

```bash
# 检查文件存在
ls ultralytics/utils/solr_loss.py

# 重新上传
scp ultralytics/utils/solr_loss.py user@server:/path/to/yolo12-bimodal/ultralytics/utils/
```

---

### Q2: CUDA out of memory

**原因**: batch size 太大

**解决**:

```bash
# 减小batch size
python train_depth_solr.py --data ... --batch 8

# 或使用梯度累积
python train_depth_solr.py --data ... --batch 8 --accumulate 2  # 等效batch=16
```

---

### Q3: mAP 没有提升

**可能原因**:

1. SOLR 权重不够大 (尝试 medium_weight=2.5)
2. 训练不够充分 (等待更多 epochs)
3. 数据问题 (检查深度图归一化)

**解决**:

```bash
# 增加中等目标权重
python train_depth_solr.py --data ... --medium_weight 2.5

# 或查看SOLR统计
# 添加调试代码打印每个batch的目标尺寸分布
```

---

## 📚 参数调优指南

### SOLR 权重选择

| 数据集特点          | small_weight | medium_weight | large_weight | 说明           |
| ------------------- | ------------ | ------------- | ------------ | -------------- |
| **VisDrone (默认)** | 2.5          | 2.0           | 1.0          | 平衡所有尺寸   |
| **小目标很多**      | 3.0          | 2.0           | 1.0          | 强调小目标     |
| **中目标是瓶颈**    | 2.5          | 2.5           | 1.0          | 重点优化中目标 |
| **保守策略**        | 2.0          | 1.5           | 1.0          | 避免过拟合     |

### 尺寸阈值调整

```bash
# 如果你的数据集小目标定义不同
# 例如: 小目标<24px, 大目标>64px
python train_depth_solr.py \
    --data ... \
    --small_thresh 24 \
    --large_thresh 64
```

---

## ✅ 成功检查清单

### 集成测试

- [ ] `python -m ultralytics.utils.solr_loss`通过
- [ ] 无 ImportError

### 上传确认

- [ ] `solr_loss.py`在`ultralytics/utils/`
- [ ] `train_depth_solr.py`在项目根目录
- [ ] Git 已提交推送

### 快速测试 (10 epochs)

- [ ] 训练正常启动
- [ ] 日志显示"SOLR 权重已成功集成"
- [ ] 深度图正常加载
- [ ] Loss 正常下降
- [ ] 无 CUDA 错误

### 完整训练 (300 epochs)

- [ ] 训练正常进行
- [ ] 定期检查 mAP (每 50 epochs)
- [ ] AP_m 有提升趋势

---

## 🎯 预期结果

### VisDrone 验证集 (300 epochs)

| 指标          | Baseline   | +SOLR        | 提升            | RemDet    | 差距          |
| ------------- | ---------- | ------------ | --------------- | --------- | ------------- |
| AP@0.50       | 32.57%     | **35~36%**   | **+2.5~3.5%**   | 37.1%     | **-1~2%**     |
| AP@[0.5:0.95] | 18.75%     | **20.5~21%** | **+1.75~2.25%** | 21.8%     | **-0.3~1.3%** |
| AP_s          | 9.61%      | **10.5~11%** | **+1~1.5%**     | 10.7%     | ✅ **接近**   |
| **AP_m**      | **28.86%** | **31~32%**   | **+2~3%**       | **33.0%** | **-1~2%**     |
| AP_l          | 43.29%     | **43.5~44%** | **+0.2~0.7%**   | 44.5%     | **-0.5~1%**   |

**关键**: SOLR 主要提升**中等目标**(AP_m),这是 RemDet 的主要优势区域!

---

## 📞 需要帮助?

如果遇到问题,提供:

1. 错误信息 (完整 traceback)
2. 运行命令
3. 训练日志 (最后 100 行)

```bash
# 获取错误日志
tail -100 runs/train/test_solr_10ep/train.log
```

---

**所有文件已准备就绪!** 🎉

**下一步**:

1. 本地测试 SOLR 模块 (`python -m ultralytics.utils.solr_loss`)
2. 上传到服务器
3. 运行 10 epochs 快速测试
4. 启动 300 epochs 完整训练

**祝训练顺利,早日超越 RemDet!** 🚀
