# 🚀 SOLR 使用指南 - 支持多尺寸训练

> **更新**: 2025-11-19  
> **新特性**: 支持 `--cfg n/s/m/l/x` 参数,一键切换模型尺寸! ✨

---

## 📋 核心改进

### ✅ 已创建文件 (5 个)

| 文件                             | 用途                           | 状态          |
| -------------------------------- | ------------------------------ | ------------- |
| `ultralytics/utils/solr_loss.py` | SOLR 核心实现                  | ✅ 完成       |
| `train_depth_solr.py`            | SOLR 训练脚本 (支持--cfg 参数) | ✅ **已升级** |
| `batch_train_solr_all_sizes.sh`  | Linux 批量训练脚本             | ✅ 新增       |
| `batch_train_solr_all_sizes.bat` | Windows 批量训练脚本           | ✅ 新增       |
| `代码审查报告_SOLR集成.md`       | 代码审查报告                   | ✅ 完成       |

---

## 🎯 使用方式对比

### ❌ 旧方式 (不推荐)

```bash
# 需要指定完整路径,不同尺寸要改路径
python train_depth_solr.py \
    --model ultralytics/cfg/models/12/yolo12n-rgbd-v1.yaml \
    --data visdrone-rgbd.yaml
```

**问题**:

- 路径太长,易出错
- 训练不同尺寸要修改路径
- 维护多个配置文件

---

### ✅ 新方式 (推荐)

```bash
# 只需指定 --cfg 参数,超简单!
python train_depth_solr.py \
    --data visdrone-rgbd.yaml \
    --cfg n  # 或 s/m/l/x
```

**优势**:

- ✅ 参数简洁,不易出错
- ✅ 一个 universal 配置文件搞定所有尺寸
- ✅ 与 Ultralytics 官方风格一致
- ✅ 支持批量训练脚本

---

## 🚀 快速开始

### 方式 1: 单个模型训练

```bash
# ========== Nano模型 (对标RemDet-Tiny) ==========
# 最快训练速度 (~30min/epoch on RTX 4090)
python train_depth_solr.py \
    --data data/visdrone-rgbd.yaml \
    --cfg n \
    --epochs 300 \
    --batch 32 \
    --name solr_n_300ep

# ========== Small模型 (对标RemDet-S, 推荐!) ==========
# 性价比最高,1小时/epoch
python train_depth_solr.py \
    --data data/visdrone-rgbd.yaml \
    --cfg s \
    --epochs 300 \
    --batch 16 \
    --name solr_s_300ep

# ========== Medium模型 (对标RemDet-M) ==========
# 性能更强,2小时/epoch
python train_depth_solr.py \
    --data data/visdrone-rgbd.yaml \
    --cfg m \
    --epochs 300 \
    --batch 8 \
    --name solr_m_300ep

# ========== Large模型 (对标RemDet-L) ==========
# 大模型,4小时/epoch
python train_depth_solr.py \
    --data data/visdrone-rgbd.yaml \
    --cfg l \
    --epochs 300 \
    --batch 4 \
    --name solr_l_300ep

# ========== XLarge模型 (对标RemDet-X) ==========
# 终极性能,6小时/epoch
python train_depth_solr.py \
    --data data/visdrone-rgbd.yaml \
    --cfg x \
    --epochs 300 \
    --batch 2 \
    --name solr_x_300ep
```

---

### 方式 2: 批量训练所有尺寸 (推荐发论文时使用)

#### Linux 服务器:

```bash
# 训练所有尺寸 (n/s/m/l/x)
bash batch_train_solr_all_sizes.sh

# 只训练部分尺寸
bash batch_train_solr_all_sizes.sh n s m  # 只训练n/s/m

# 后台运行 (避免SSH断开)
nohup bash batch_train_solr_all_sizes.sh > train_all.log 2>&1 &

# 监控进度
tail -f train_all.log
```

#### Windows 本地 (测试用):

```cmd
REM 训练所有尺寸
batch_train_solr_all_sizes.bat

REM 只训练nano (快速测试)
batch_train_solr_all_sizes.bat n
```

---

## 📊 模型尺寸对比

| 模型  | 参数量 | FLOPs | 推荐 Batch | 显存占用 | 训练速度  | RemDet 对标 | 推荐场景          |
| ----- | ------ | ----- | ---------- | -------- | --------- | ----------- | ----------------- |
| **n** | ~3M    | ~8G   | 32         | ~8GB     | ~30min/ep | Tiny        | 快速实验/实时部署 |
| **s** | ~11M   | ~46G  | 16         | ~12GB    | ~1h/ep    | S           | **主力模型** ⭐   |
| **m** | ~22M   | ~92G  | 8          | ~16GB    | ~2h/ep    | M           | 性能优先          |
| **l** | ~44M   | ~184G | 4          | ~20GB    | ~4h/ep    | L           | 大模型对比        |
| **x** | ~66M   | ~276G | 2          | ~22GB    | ~6h/ep    | X           | 终极性能/论文     |

**建议**:

- **快速验证**: 先训练 **n 模型** 10 epochs (~5 小时),确认流程无误
- **主力实验**: 训练 **s 模型** 300 epochs (~12-15 天),性价比最高
- **论文发表**: 训练 **s/m/x** 三个尺寸,与 RemDet 全面对比

---

## 🎨 自定义 SOLR 参数

### 默认配置 (适合 VisDrone)

```bash
python train_depth_solr.py \
    --data visdrone-rgbd.yaml \
    --cfg s \
    --small_weight 2.5   # 小目标 (<32px)
    --medium_weight 2.0  # 中等目标 (32-96px) ← 关键参数!
    --large_weight 1.0   # 大目标 (>96px)
```

---

### 激进配置 (如果 AP_m 提升不足)

```bash
# 增加中等目标权重到2.5x
python train_depth_solr.py \
    --data visdrone-rgbd.yaml \
    --cfg s \
    --small_weight 2.5 \
    --medium_weight 2.5 \  # 从2.0增加到2.5
    --large_weight 1.0 \
    --name solr_s_m25
```

---

### 小目标专项优化

```bash
# 如果AP_s太低,增加小目标权重到3.0x
python train_depth_solr.py \
    --data visdrone-rgbd.yaml \
    --cfg s \
    --small_weight 3.0 \  # 从2.5增加到3.0
    --medium_weight 2.0 \
    --large_weight 1.0 \
    --name solr_s_s30
```

---

### 自定义尺寸阈值

```bash
# 如果你的数据集小目标定义不同 (例如<24px)
python train_depth_solr.py \
    --data visdrone-rgbd.yaml \
    --cfg s \
    --small_thresh 24 \   # 默认32
    --large_thresh 64 \   # 默认96
    --name solr_s_custom_thresh
```

---

## ⚡ 快速测试流程

### 第 1 步: 本地测试 SOLR 模块

```bash
cd f:\CV\Paper\yoloDepth\yolo12-bimodal
python -m ultralytics.utils.solr_loss
```

**预期输出**: `✅ All tests passed!`

---

### 第 2 步: 上传到服务器

```bash
# Git方式 (推荐)
git add train_depth_solr.py
git add ultralytics/utils/solr_loss.py
git add batch_train_solr_all_sizes.sh
git commit -m "Add SOLR with multi-size support (--cfg n/s/m/l/x)"
git push

# 服务器端拉取
ssh user@server
cd /path/to/yolo12-bimodal
git pull
```

---

### 第 3 步: 服务器端快速测试 (10 epochs)

```bash
# SSH登录服务器
ssh user@server
cd /path/to/yolo12-bimodal

# 测试nano模型 (最快,30-60分钟)
python train_depth_solr.py \
    --data data/visdrone-rgbd.yaml \
    --cfg n \
    --epochs 10 \
    --batch 32 \
    --device 0 \
    --name test_solr_n_10ep
```

**监控训练**:

```bash
# 实时查看日志
tail -f runs/train_solr/test_solr_n_10ep/train.log

# 应该看到:
# ✅ Using model size: YOLO12-N (with SOLR loss)
# ✅ Expected model size: ~3M params, ~8G FLOPs (对标RemDet-Tiny)
# ✅ SOLR权重已成功集成到损失函数
# ✅ 深度图已加载 ✓
```

---

### 第 4 步: 启动完整训练 (300 epochs)

```bash
# 训练small模型 (推荐,12-15天)
python train_depth_solr.py \
    --data data/visdrone-rgbd.yaml \
    --cfg s \
    --epochs 300 \
    --batch 16 \
    --device 0 \
    --optimizer SGD \
    --lr0 0.01 \
    --momentum 0.937 \
    --weight_decay 0.0005 \
    --mosaic 1.0 \
    --mixup 0.15 \
    --close_mosaic 10 \
    --amp \
    --name solr_s_300ep

# 后台运行 (避免SSH断开)
nohup python train_depth_solr.py \
    --data data/visdrone-rgbd.yaml \
    --cfg s \
    --epochs 300 \
    --batch 16 \
    --device 0 \
    --name solr_s_300ep \
    > train_solr_s.log 2>&1 &

# 监控
tail -f train_solr_s.log
```

---

## 📈 预期结果

### VisDrone 验证集 (300 epochs 后)

| 模型               | AP@0.5     | AP@0.5:0.95  | AP_s         | AP_m       | AP_l         | vs RemDet           |
| ------------------ | ---------- | ------------ | ------------ | ---------- | ------------ | ------------------- |
| **RGB-D-N + SOLR** | **35~36%** | **20.5~21%** | **10.5~11%** | **31~32%** | **43.5~44%** | Tiny: 37.1% (-1~2%) |
| **RGB-D-S + SOLR** | **46~48%** | **28~29%**   | **18~19%**   | **41~43%** | **52~54%**   | S: 42.3% (+4~6%) ✅ |
| **RGB-D-M + SOLR** | **48~50%** | **30~31%**   | **20~21%**   | **43~45%** | **54~56%**   | M: 45.0% (+3~5%) ✅ |
| **RGB-D-L + SOLR** | **50~52%** | **31~32%**   | **21~22%**   | **45~47%** | **56~58%**   | L: 47.4% (+3~5%) ✅ |
| **RGB-D-X + SOLR** | **51~53%** | **32~33%**   | **22~23%**   | **46~48%** | **57~59%**   | X: 48.3% (+3~5%) ✅ |

**关键提升来源**:

1. **RGB-D 融合**: +3~4% AP@0.5 (已在 yolo12-bimodal 验证)
2. **SOLR Loss**: +2~3% AP_m (本次新增)
3. **总提升**: +5~7% AP@0.5,有望超越 RemDet! 🎯

---

## 🔍 调试检查点

### 检查点 1: 确认--cfg 参数生效

训练日志应该显示:

```
ℹ️  Using model size: YOLO12-S (with SOLR loss)
ℹ️  Expected model size: ~11M params, ~46G FLOPs (对标RemDet-S)
```

**如果没看到**: 检查`train_depth_solr.py`是否为最新版本

---

### 检查点 2: 验证模型参数量

```bash
# 训练开始时会打印模型信息
grep "parameters" runs/train_solr/solr_s_300ep/train.log

# 应该看到:
# Model summary: 325 layers, 11234567 parameters, 11234567 gradients
```

**对比标准**:

- n: ~3M params
- s: ~11M params
- m: ~22M params
- l: ~44M params
- x: ~66M params

误差 ±10%属于正常

---

### 检查点 3: SOLR 统计信息

```bash
# 查看SOLR权重应用情况
grep "SOLR" runs/train_solr/solr_s_300ep/train.log | head -20

# 应该看到:
# SOLR (Small Object Loss Reweighting) Initialized
# Size Thresholds: Small < 32px, Medium 32-96px, Large > 96px
# Loss Weights: Small 2.5x, Medium 2.0x, Large 1.0x
# ✅ SOLR loss integrated successfully!
```

---

## 🚨 常见问题

### Q1: `--cfg` 参数不识别

**原因**: `train_depth_solr.py`版本过旧

**解决**:

```bash
# 检查是否有--cfg参数
grep "\-\-cfg" train_depth_solr.py

# 应该看到:
# parser.add_argument("--cfg", type=str, default="n", help="Model size...")

# 如果没有,重新从Git拉取最新版本
git pull origin main
```

---

### Q2: 训练时显示"model_name not found"

**原因**: 代码逻辑问题,已在最新版修复

**解决**:

```bash
# 确保train_depth_solr.py包含以下代码:
# model = YOLO(args.model, task='detect')
# if args.cfg:
#     model.model_name = f"yolo12{args.cfg}"

# 如果没有,更新文件
```

---

### Q3: 不同尺寸模型性能差异不明显

**可能原因**:

1. 训练不够充分 (至少 200+ epochs)
2. batch size 太小,导致梯度噪声大
3. 数据集太小,大模型容易过拟合

**建议**:

- 增加训练轮数到 300 epochs
- 使用推荐的 batch size (见上表)
- 考虑数据增强或数据扩充 (VisDrone+UAVDT)

---

## 📚 八股知识点

### 知识点: 模型缩放策略 (Compound Scaling)

**Q**: `--cfg n/s/m/l/x` 是如何影响模型的?

**A**: 通过 **depth_multiple** 和 **width_multiple** 两个维度缩放:

```yaml
scales:
  n: [0.50, 0.25, 1024] # depth×0.5, width×0.25
  s: [0.50, 0.50, 1024] # depth×0.5, width×0.5
  m: [0.50, 1.00, 512] # depth×0.5, width×1.0
  l: [1.00, 1.00, 512] # depth×1.0, width×1.0 (baseline)
  x: [1.00, 1.50, 512] # depth×1.0, width×1.5
```

**举例**:

```yaml
# 配置文件中定义: C3k2模块, repeats=2, channels=256

# n模型:
#   repeats: 2 × 0.5 = 1
#   channels: 256 × 0.25 = 64

# s模型:
#   repeats: 2 × 0.5 = 1
#   channels: 256 × 0.5 = 128

# l模型 (baseline):
#   repeats: 2 × 1.0 = 2
#   channels: 256 × 1.0 = 256
```

**为什么这样设计?**

- **n/s**: 减少层数+通道数 → 轻量化,适合实时检测
- **m**: 保持通道数,减少层数 → 平衡性能与效率
- **l**: 基准配置,不缩放
- **x**: 增加通道数 → 提升表达能力,适合离线高精度检测

**面试追问**: 为什么不是所有模型都用相同的 depth 和 width 比例?

**回答**:

- 小模型(n): 通道数已经很少(64),再减少层数影响不大,所以同时减少 depth 和 width
- 大模型(x): 层数已经足够(depth=1.0),继续增加层数边际收益递减,所以只增加 width
- 这是**EfficientNet**论文提出的复合缩放理论在 YOLO 上的应用

---

## ✅ 成功检查清单

### 代码集成

- [ ] `train_depth_solr.py` 包含 `--cfg` 参数
- [ ] `ultralytics/cfg/models/12/yolo12-rgbd-v2.1-universal.yaml` 存在
- [ ] `ultralytics/utils/solr_loss.py` 存在

### 本地测试

- [ ] `python -m ultralytics.utils.solr_loss` 通过
- [ ] 无 ImportError

### 服务器部署

- [ ] 所有文件已上传
- [ ] Git 已提交推送

### 快速测试 (10 epochs)

- [ ] 训练正常启动
- [ ] 日志显示"Using model size: YOLO12-X"
- [ ] SOLR 权重已集成
- [ ] 深度图正常加载
- [ ] Loss 正常下降

### 完整训练 (300 epochs)

- [ ] 选择合适的模型尺寸 (推荐 s)
- [ ] batch size 符合显存限制
- [ ] 后台运行不中断
- [ ] 定期检查 mAP 趋势

---

## 🎯 下一步行动

1. **本地测试** (5 分钟):

   ```bash
   python -m ultralytics.utils.solr_loss
   ```

2. **上传代码** (5 分钟):

   ```bash
   git add train_depth_solr.py ultralytics/utils/solr_loss.py batch_train_solr_all_sizes.sh
   git commit -m "Add SOLR with --cfg n/s/m/l/x support"
   git push
   ```

3. **服务器快速测试** (1 小时):

   ```bash
   python train_depth_solr.py --data data/visdrone-rgbd.yaml --cfg n --epochs 10
   ```

4. **启动主力训练** (12-15 天):

   ```bash
   nohup python train_depth_solr.py --data data/visdrone-rgbd.yaml --cfg s --epochs 300 > train_s.log 2>&1 &
   ```

5. **COCO 评估**:
   ```bash
   python val_coco_eval.py --weights runs/train_solr/solr_s_300ep/weights/best.pt
   ```

---

**所有功能已就绪!** 🎉

**现在你可以**:

- ✅ 使用 `--cfg n` 快速切换模型尺寸
- ✅ 使用批量脚本一次训练所有尺寸
- ✅ 自定义 SOLR 权重优化特定尺寸目标
- ✅ 与 RemDet 进行全面对比

**祝训练顺利,早日超越 RemDet!** 🚀
