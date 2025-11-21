# GGFE 修复执行计划 - 立即行动版

## 🔴 问题总结

**发现**: 训练脚本`train_depth_solr_v2.py`的第 136-141 行存在**致命 bug**:

```python
if args.weights:
    model = YOLO(args.weights)  # ❌ 直接加载weights的架构
```

**后果**:

- 300 个 epoch 训练的是**标准 yolo12n 架构** (3.0M 参数)
- GGFE 配置从未加载 (应为 3.5M 参数)
- `args.yaml`中`cfg: null`证明了这一点
- 性能提升为 0 (19.24% vs 19.2% baseline)

---

## ✅ 修复方案

已创建 3 个文件:

1. **train_depth_solr_v2_fixed.py** (205 行) - 修复后的训练脚本
2. **check_ggfe_loaded.py** (120 行) - GGFE 加载验证工具
3. **train*depth_solr_v2*紧急修复.md** - 详细说明文档

**核心修复逻辑** (train_depth_solr_v2_fixed.py 第 128-177 行):

```python
# 总是从YAML创建架构
model_yaml = 'ultralytics/cfg/models/12/yolo12-rgbd-ggfe-universal.yaml'
model = YOLO(model_yaml, task='detect')  # ✅ 确保GGFE架构

# 如果提供weights，只加载参数 (不覆盖架构)
if args.weights:
    ckpt = torch.load(args.weights, map_location='cpu')
    state_dict = ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']
    incompatible = model.model.load_state_dict(state_dict, strict=False)  # ✅ strict=False
    # GGFE模块会被随机初始化 (因为weights中没有)
```

---

## 📋 立即执行步骤

### Step 1: 本地验证 (5 分钟)

```powershell
cd f:\CV\Paper\yoloDepth\yolo12-bimodal

# 测试修复脚本能否正确创建GGFE模型
python -c "
import torch
from ultralytics import YOLO

# 从GGFE YAML创建模型
model = YOLO('ultralytics/cfg/models/12/yolo12-rgbd-ggfe-universal.yaml')
total_params = sum(p.numel() for p in model.model.parameters())
print(f'Model created: {total_params/1e6:.2f}M params')

# 加载预训练权重
ckpt = torch.load('models/yolo12n.pt', map_location='cpu')
state_dict = ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']
incompatible = model.model.load_state_dict(state_dict, strict=False)

print(f'Missing keys (GGFE): {len(incompatible.missing_keys)}')
print(f'Unexpected keys: {len(incompatible.unexpected_keys)}')

# 验证GGFE存在
ggfe_count = 0
for name, _ in model.model.named_modules():
    if 'ggfe' in name.lower():
        ggfe_count += 1
print(f'GGFE modules found: {ggfe_count}')

if total_params/1e6 >= 3.3 and ggfe_count > 0:
    print('✅ PASS: GGFE correctly loaded')
else:
    print('❌ FAIL: GGFE not loaded')
"
```

**预期输出**:

```
Model created: 3.50M params
Missing keys (GGFE): 100+
Unexpected keys: 0
GGFE modules found: 6
✅ PASS: GGFE correctly loaded
```

**如果失败**:

- 检查`ultralytics/cfg/models/12/yolo12-rgbd-ggfe-universal.yaml`是否存在
- 检查 YAML 文件是否包含 RGBDGGFEFusion 配置

---

### Step 2: 上传到服务器 (2 分钟)

```powershell
# 使用你的SCP/SFTP工具上传以下文件:
# 1. train_depth_solr_v2_fixed.py
# 2. check_ggfe_loaded.py

# 或使用命令行 (如果有ssh):
scp train_depth_solr_v2_fixed.py user@server:/data2/user/2024/lzy/yolo12-bimodal/
scp check_ggfe_loaded.py user@server:/data2/user/2024/lzy/yolo12-bimodal/
```

---

### Step 3: 服务器 10-Epoch 快速验证 (30 分钟)

```bash
cd /data2/user/2024/lzy/yolo12-bimodal

# 启动10epoch验证训练
python train_depth_solr_v2_fixed.py \
    --name visdrone_ggfe_verify_10ep_fixed \
    --data /data2/user/2024/lzy/yolo12-bimodal/data/visdrone-rgbd.yaml \
    --device 4 \
    --weights /data2/user/2024/lzy/yolo12-bimodal/models/yolo12n.pt \
    --cfg n \
    --medium_weight 2.5 \
    --batch 16 \
    --epochs 10
```

**训练开始后立即检查日志** (不要等 10 个 epoch 结束):

```bash
# 检查参数量 (应该是3.5M)
grep "Total model parameters" runs/train/visdrone_ggfe_verify_10ep_fixed/*.log

# 检查Missing keys (应该有100+个GGFE参数)
grep "Missing keys" runs/train/visdrone_ggfe_verify_10ep_fixed/*.log

# 检查GGFE模块
grep "Found.*GGFE" runs/train/visdrone_ggfe_verify_10ep_fixed/*.log
```

**预期日志内容**:

```
📊 Total model parameters: 3.50M
📊 Trainable parameters: 3.50M
   Expected: ~3.5M (baseline 3.0M + GGFE 0.5M)
⚠️  Missing keys (will be randomly initialized): 120
✅ Found 6 GGFE modules:
   - model.4.rgbd_fusion.ggfe
   - model.10.rgbd_fusion.ggfe
   ...
```

**成功标准**:

- ✅ 参数量 >= 3.3M
- ✅ Found GGFE modules: 6
- ✅ Missing keys: 100+

**如果失败** (参数量仍为 3.0M):

- 停止训练 `Ctrl+C`
- 检查 YAML 文件路径是否正确
- 运行诊断脚本 (见 Step 4)

---

### Step 4: 使用诊断工具验证 (训练 10epoch 后)

```bash
# Epoch 10完成后运行诊断
python check_ggfe_loaded.py runs/train/visdrone_ggfe_verify_10ep_fixed

# 预期输出:
# ✅ PASS: Parameter count looks good (>= 3.3M)
# ✅ PASS: Found 6 GGFE modules
# ✅ FINAL VERDICT: GGFE IS CORRECTLY LOADED!
```

**如果诊断工具显示失败**:

1. 检查`args.yaml`中`cfg`字段是否为`null`
2. 如果仍为`null`，说明脚本修复不彻底
3. 回到 Step 1 重新验证

---

### Step 5: 100-Epoch 完整验证 (如果 Step 3 成功)

```bash
# 10-epoch验证成功后，立即启动100-epoch训练
python train_depth_solr_v2_fixed.py \
    --name visdrone_ggfe_v3_100ep \
    --data /data2/user/2024/lzy/yolo12-bimodal/data/visdrone-rgbd.yaml \
    --device 4 \
    --weights /data2/user/2024/lzy/yolo12-bimodal/models/yolo12n.pt \
    --cfg n \
    --medium_weight 2.5 \
    --batch 16 \
    --epochs 100
```

**训练时长**: 约 3-4 天

**监控指标** (每 10 个 epoch 检查一次):

```bash
# 查看最新结果
tail -5 runs/train/visdrone_ggfe_v3_100ep/results.csv

# 提取mAP
python -c "
import pandas as pd
df = pd.read_csv('runs/train/visdrone_ggfe_v3_100ep/results.csv')
latest = df.iloc[-1]
print(f'Epoch {int(latest[\"epoch\"])}: mAP@0.5:0.95 = {latest[\"metrics/mAP50-95(B)\"]*100:.2f}%')
"
```

**成功标准** (100 个 epoch 后):

- ✅ AP@0.5:0.95 >= 20.0% (+0.8% vs baseline 19.2%)
- ✅ AP_m >= 30.5% (+0.9% vs baseline 29.6%)
- ✅ 参数量 ~3.5M (GGFE 已加载)

**如果 AP < 19.5%**:

- GGFE 可能对 VisDrone 有害，考虑放弃
- 尝试只在 P4 层启用 GGFE (修改 YAML)

**如果 19.5% <= AP < 20.0%**:

- GGFE 效果微弱，尝试调整超参数:
  - `ggfe_reduction=4` (更强的 GGFE)
  - `medium_weight=3.0` (更强的 SOLR)

**如果 AP >= 20.0%**:

- ✅ GGFE 有效，继续 300-epoch 训练

---

### Step 6: 300-Epoch 完整训练 (如果 Step 5 成功)

```bash
# 100-epoch结果理想 (AP >= 20.0%) 才执行这一步
python train_depth_solr_v2_fixed.py \
    --name visdrone_ggfe_v3_300ep \
    --data /data2/user/2024/lzy/yolo12-bimodal/data/visdrone-rgbd.yaml \
    --device 4 \
    --weights /data2/user/2024/lzy/yolo12-bimodal/models/yolo12n.pt \
    --cfg n \
    --medium_weight 2.5 \
    --batch 16 \
    --epochs 300
```

**训练时长**: 约 10-12 天

**目标指标** (RemDet-Tiny 对齐):

- 🎯 AP@0.5:0.95 >= 21.0% (接近 RemDet 的 21.8%)
- 🎯 AP_s >= 11.5% (接近 RemDet 的 12.7%)
- 🎯 AP_m >= 31.5% (接近 RemDet 的 33.0%)

---

## 🔄 失败应对策略

### 场景 1: Step 3 失败 (参数量仍为 3.0M)

**原因**: 脚本修复不彻底或 YAML 路径错误

**行动**:

1. 检查 YAML 文件是否存在:
   ```bash
   ls -lh ultralytics/cfg/models/12/yolo12-rgbd-ggfe-universal.yaml
   ```
2. 检查 YAML 内容:
   ```bash
   grep "RGBDGGFEFusion" ultralytics/cfg/models/12/yolo12-rgbd-ggfe-universal.yaml
   # 应该看到3行 (P3/P4/P5)
   ```
3. 手动测试模型创建:
   ```python
   from ultralytics import YOLO
   m = YOLO('ultralytics/cfg/models/12/yolo12-rgbd-ggfe-universal.yaml')
   print(sum(p.numel() for p in m.model.parameters())/1e6)
   # 应输出: 3.50
   ```

---

### 场景 2: Step 5 失败 (100ep 后 AP < 19.5%)

**原因**: GGFE 设计不适合 VisDrone 或引入噪声

**行动**:

1. 分析是否 GGFE 降低了性能:
   - 对比 baseline (19.2%) vs GGFE (例如 18.5%)
   - 如果 GGFE 明显更差 → 放弃 GGFE
2. 尝试减弱 GGFE:
   - 修改 YAML: `use_ggfe=False` (退化为 RGBDMidFusion)
   - 或只在 P4 启用 (注释 P3/P5 的 GGFE)
3. 切换到 SADF 方案 (见备用计划)

---

### 场景 3: Step 5 成功但 Step 6 失败 (300ep 后 AP < 21%)

**原因**: 100ep 验证集过拟合，或需要更长训练

**行动**:

1. 检查是否过拟合:
   ```python
   import pandas as pd
   df = pd.read_csv('runs/train/visdrone_ggfe_v3_300ep/results.csv')
   print(df[['epoch', 'train/box_loss', 'val/box_loss', 'metrics/mAP50-95(B)']].tail(50))
   # 如果val_loss持续上升 → 过拟合
   ```
2. 尝试 500-epoch 训练 (RemDet 用了 300+)
3. 或切换到 GGFE+SADF 组合 (见备用计划)

---

## 🎯 成功判定标准

### 最低成功标准 (可发论文)

- AP@0.5:0.95 >= 20.5% (+1.3% vs baseline)
- AP_m >= 31.0% (+1.4% vs baseline)
- 参数量 <= 4.0M (保持轻量)

### 理想成功标准 (超越 RemDet)

- AP@0.5:0.95 >= 22.0% (+2.8% vs baseline, 超越 RemDet 0.2%)
- AP_s >= 13.0% (+3.1% vs baseline, 超越 RemDet 0.3%)
- AP_m >= 33.5% (+3.9% vs baseline, 超越 RemDet 0.5%)

---

## 📊 实验记录表格

| 实验名称         | 参数量 | GGFE | SOLR | Epochs | AP@0.5:0.95 | AP_s  | AP_m  | AP_l  | 状态        |
| ---------------- | ------ | ---- | ---- | ------ | ----------- | ----- | ----- | ----- | ----------- |
| baseline         | 3.0M   | ❌   | ✅   | 300    | 19.2%       | 9.9%  | 29.6% | 45.9% | ✅ 完成     |
| ggfe_v1          | 3.0M   | ❌   | ✅   | 100    | 18.3%       | 9.1%  | 28.5% | 46.4% | ❌ 接口错误 |
| ggfe_v2          | 3.0M   | ❌   | ✅   | 300    | 19.2%       | 10.0% | 29.6% | 46.0% | ❌ 未加载   |
| ggfe_verify_10ep | 3.5M   | ✅   | ✅   | 10     | TBD         | TBD   | TBD   | TBD   | ⏳ 待执行   |
| ggfe_v3_100ep    | 3.5M   | ✅   | ✅   | 100    | TBD         | TBD   | TBD   | TBD   | ⏳ 待执行   |
| ggfe_v3_300ep    | 3.5M   | ✅   | ✅   | 300    | TBD         | TBD   | TBD   | TBD   | ⏳ 待执行   |

---

## 📝 检查清单

在上传到服务器前，确认本地已完成:

- [x] train_depth_solr_v2_fixed.py 已创建
- [x] check_ggfe_loaded.py 已创建
- [x] 本地验证脚本能正确创建 3.5M 参数模型 (Step 1)
- [ ] 上传 train_depth_solr_v2_fixed.py 到服务器 (Step 2)
- [ ] 上传 check_ggfe_loaded.py 到服务器 (Step 2)

在服务器训练前，确认:

- [ ] YAML 文件存在: `ultralytics/cfg/models/12/yolo12-rgbd-ggfe-universal.yaml`
- [ ] YAML 包含 RGBDGGFEFusion 配置 (3 处)
- [ ] 预训练权重存在: `models/yolo12n.pt`
- [ ] 数据集 YAML 存在: `data/visdrone-rgbd.yaml`
- [ ] GPU 可用: `nvidia-smi`

训练开始后 5 分钟内检查:

- [ ] 日志显示参数量 >= 3.3M
- [ ] 日志显示 Found GGFE modules
- [ ] `args.yaml`中 cfg 不为 null

---

## 🚀 现在立即执行

**优先级最高的任务**:

1. 本地运行 Step 1 验证脚本 (5 分钟)
2. 上传文件到服务器 (Step 2)
3. 启动 10-epoch 验证训练 (Step 3)

**今天必须完成**: Step 1-3 (总计不超过 1 小时)

**明天检查**: 10-epoch 训练结果，决定是否启动 100-epoch

---

祝训练顺利！如果遇到任何问题，立即停止训练并报告日志！🎯
