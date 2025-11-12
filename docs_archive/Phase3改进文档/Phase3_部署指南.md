# Phase 3 部署指南 - ChannelC2f 多尺度训练

## 📋 总览

**目标**: 训练 5 个尺度的 ChannelC2f 模型 (n, s, m, l, x) 以对标 RemDet 论文

**核心改进**: 在 P4 层 (16x16 下采样) 添加 Channel Attention，专门提升 Medium 目标检测性能

**预期成果**:

- Medium mAP: 14.28% → 20%+ (最低目标 +5.7%)
- Overall mAP: 44.03% → 46%+ (最低目标 +1.97%)
- 对标 RemDet-{Tiny, S, M, L, X} 五个尺度

---

## 🚀 快速部署 (3 步)

### Step 1: 本地上传文件 (Windows PowerShell)

```powershell
# 切换到项目目录
cd f:\CV\Paper\yoloDepth\yoloDepth

# 上传核心文件 (3 files)
scp ultralytics/nn/modules/block.py ubuntu@10.16.62.111:/data2/user/2024/lzy/yolo12-bimodal/ultralytics/nn/modules/
scp ultralytics/nn/modules/__init__.py ubuntu@10.16.62.111:/data2/user/2024/lzy/yolo12-bimodal/ultralytics/nn/modules/
scp ultralytics/nn/tasks.py ubuntu@10.16.62.111:/data2/user/2024/lzy/yolo12-bimodal/ultralytics/nn/

# 上传多尺度配置 (5 files)
scp ultralytics/cfg/models/12/yolo12n-rgbd-channelc2f.yaml ubuntu@10.16.62.111:/data2/user/2024/lzy/yolo12-bimodal/ultralytics/cfg/models/12/
scp ultralytics/cfg/models/12/yolo12s-rgbd-channelc2f.yaml ubuntu@10.16.62.111:/data2/user/2024/lzy/yolo12-bimodal/ultralytics/cfg/models/12/
scp ultralytics/cfg/models/12/yolo12m-rgbd-channelc2f.yaml ubuntu@10.16.62.111:/data2/user/2024/lzy/yolo12-bimodal/ultralytics/cfg/models/12/
scp ultralytics/cfg/models/12/yolo12l-rgbd-channelc2f.yaml ubuntu@10.16.62.111:/data2/user/2024/lzy/yolo12-bimodal/ultralytics/cfg/models/12/
scp ultralytics/cfg/models/12/yolo12x-rgbd-channelc2f.yaml ubuntu@10.16.62.111:/data2/user/2024/lzy/yolo12-bimodal/ultralytics/cfg/models/12/

# 上传训练脚本 (3 files)
scp train_phase3.py ubuntu@10.16.62.111:/data2/user/2024/lzy/yolo12-bimodal/
scp verify_phase3.py ubuntu@10.16.62.111:/data2/user/2024/lzy/yolo12-bimodal/
scp test_phase3.py ubuntu@10.16.62.111:/data2/user/2024/lzy/yolo12-bimodal/

# 上传批处理脚本 (2 files, 可选)
scp train_all_scales.sh ubuntu@10.16.62.111:/data2/user/2024/lzy/yolo12-bimodal/
scp validate_all_phase3.sh ubuntu@10.16.62.111:/data2/user/2024/lzy/yolo12-bimodal/
```

**上传完成提示**:

```powershell
Write-Host "✅ All files uploaded!" -ForegroundColor Green
```

---

### Step 2: 服务器验证 (Linux Terminal)

```bash
# SSH 登录
ssh ubuntu@10.16.62.111

# 切换目录并激活环境
cd /data2/user/2024/lzy/yolo12-bimodal
conda activate lzy-yolo12

# 运行验证脚本 (8 checks)
python verify_phase3.py
```

**预期输出**:

```
✅ Check 1/8: block.py exists
✅ Check 2/8: ChannelAttention class complete
✅ Check 3/8: ChannelC2f class complete
✅ Check 4/8: block.py __all__ exports
✅ Check 5/8: modules/__init__.py exports
✅ Check 6/8: tasks.py imports
✅ Check 7/8: YAML config exists
✅ Check 8/8: Python import test

================================================================================
✅ All 8 checks passed! Phase 3 deployment verified.
================================================================================
```

**运行模型构建测试**:

```bash
python test_phase3.py
```

**预期输出**:

```
1️⃣ Building model from YAML...
✅ Model built successfully

2️⃣ Testing forward pass...
✅ Forward pass successful

3️⃣ Checking parameter count...
✅ Parameters: 9,518,124 (~9.52M, +1.4% vs Phase 1)

4️⃣ Verifying ChannelAttention integration...
✅ ChannelAttention found in model.model.6.ca

5️⃣ Comparing with Phase 1 baseline...
✅ Phase 3 adds channel attention to P4 layer

================================================================================
✅ All tests passed! Model ready for training.
================================================================================
```

---

### Step 3: 开始训练

#### 选项 A: 批量训练所有尺度 (推荐)

```bash
# 添加执行权限
chmod +x train_all_scales.sh validate_all_phase3.sh

# 使用 tmux 启动训练 (可在后台运行)
tmux new -s phase3_training
./train_all_scales.sh

# 分离会话: Ctrl+B, D
# 重新连接: tmux attach -t phase3_training
```

#### 选项 B: 单独训练某个尺度

```bash
# 例如: 训练 YOLO12n (最快，2天)
CUDA_VISIBLE_DEVICES=6 nohup python train_phase3.py \
    --model ultralytics/cfg/models/12/yolo12n-rgbd-channelc2f.yaml \
    --name phase3_channelc2f_n \
    > logs/phase3_n.log 2>&1 &

# 监控训练日志
tail -f logs/phase3_n.log

# 检查 mAP 进度
grep "mAP50-95" logs/phase3_n.log
```

---

## 📊 训练时间表

| 模型    | 参数量 | Batch Size | LR0    | 训练时间 | 对标 RemDet | 优先级              |
| ------- | ------ | ---------- | ------ | -------- | ----------- | ------------------- |
| YOLO12n | ~2.5M  | 32         | 0.001  | ~2 天    | RemDet-Tiny | **High** (快速验证) |
| YOLO12s | ~9.5M  | 16         | 0.001  | ~3 天    | RemDet-S    | **High** (主要对比) |
| YOLO12m | ~20M   | 8          | 0.0008 | ~5 天    | RemDet-M    | Medium              |
| YOLO12l | ~40M   | 4          | 0.0005 | ~7 天    | RemDet-L    | Medium              |
| YOLO12x | ~60M   | 4          | 0.0005 | ~10 天   | RemDet-X    | Low (可选)          |

**总训练时间**: ~27 天 (GPU: RTX 4090)

**建议策略**:

1. **优先训练 YOLO12n** (2 天) - 快速验证 ChannelC2f 是否有效
2. 如果 YOLO12n 达标 (Medium mAP ≥18%) → 继续训练其他尺度
3. 如果 YOLO12n 未达标 (Medium mAP <18%) → 分析原因，调整方案后重新训练

---

## 🎯 成功标准

### Minimum (Phase 3 有效)

- **Medium mAP**: ≥18.0% (baseline: 14.28%, +3.7%)
- **Overall mAP**: ≥45.0% (baseline: 44.03%, +0.97%)
- **Medium Recall**: ≥18.0% (baseline: 11.7%, +6.3%)

### Target (论文可发表)

- **Medium mAP**: ≥20.0% (baseline: 14.28%, +5.7%)
- **Overall mAP**: ≥46.0% (baseline: 44.03%, +1.97%)
- **Medium Recall**: ≥20.0% (baseline: 11.7%, +8.3%)

### Excellent (超越 RemDet)

- **Medium mAP**: ≥23.0% (baseline: 14.28%, +8.7%)
- **Overall mAP**: ≥47.0% (baseline: 44.03%, +2.97%)
- **Medium Recall**: ≥25.0% (baseline: 11.7%, +13.3%)

---

## 🔧 验证结果 (训练完成后)

```bash
# 验证单个模型
CUDA_VISIBLE_DEVICES=6 python val_depeth.py \
    --model runs/train/phase3_channelc2f_n/weights/best.pt \
    --data data/visdrone-rgbd.yaml

# 批量验证所有尺度
./validate_all_phase3.sh
```

**关键指标检查**:

```bash
# 查看 Medium mAP (最关键!)
grep "Medium.*mAP" results/phase3_validation/phase3_val_n/results.txt

# 查看 Overall mAP
grep "all.*mAP" results/phase3_validation/phase3_val_n/results.txt

# 查看 Recall
grep "Recall" results/phase3_validation/phase3_val_n/results.txt
```

---

## 📂 文件结构说明

```
yoloDepth/
├── ultralytics/
│   ├── nn/
│   │   ├── modules/
│   │   │   ├── block.py              # ✅ ChannelAttention + ChannelC2f 实现
│   │   │   └── __init__.py           # ✅ 导出 ChannelC2f
│   │   └── tasks.py                  # ✅ 注册 base_modules + repeat_modules
│   └── cfg/
│       └── models/
│           └── 12/
│               ├── yolo12n-rgbd-channelc2f.yaml  # ✅ Nano 配置
│               ├── yolo12s-rgbd-channelc2f.yaml  # ✅ Small 配置
│               ├── yolo12m-rgbd-channelc2f.yaml  # ✅ Medium 配置
│               ├── yolo12l-rgbd-channelc2f.yaml  # ✅ Large 配置
│               └── yolo12x-rgbd-channelc2f.yaml  # ✅ XLarge 配置
├── train_phase3.py                   # ✅ 训练脚本 (支持预训练加载)
├── verify_phase3.py                  # ✅ 部署验证脚本 (8 checks)
├── test_phase3.py                    # ✅ 模型构建测试
├── train_all_scales.sh               # ✅ 批量训练所有尺度
└── validate_all_phase3.sh            # ✅ 批量验证所有尺度
```

---

## 📋 常见问题排查

### 问题 1: 验证失败

**症状**: `verify_phase3.py` 某些 check 失败

**解决方案**:

```bash
# 检查文件是否完整上传
ls -lh ultralytics/nn/modules/block.py
ls -lh ultralytics/cfg/models/12/yolo12*-rgbd-channelc2f.yaml

# 重新上传缺失文件
# (在本地 PowerShell 重新执行 scp 命令)

# 重新运行验证
python verify_phase3.py
```

---

### 问题 2: 模型构建失败

**症状**: `test_phase3.py` 报错

**解决方案**:

```bash
# 检查 CUDA 是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 检查依赖版本
pip show ultralytics torch

# 查看详细错误日志
python test_phase3.py 2>&1 | tee test_debug.log
```

---

### 问题 3: 训练启动失败

**症状**: 训练命令执行后立即退出

**解决方案**:

```bash
# 检查数据集路径
ls data/visdrone-rgbd.yaml
cat data/visdrone-rgbd.yaml

# 检查 GPU 可用性
nvidia-smi

# 查看训练日志
cat logs/phase3_n.log

# 减小 batch size (如果显存不足)
# 修改 train_all_scales.sh 中的 BATCH_SIZE
```

---

### 问题 4: 训练中断

**症状**: 训练过程中突然停止

**解决方案**:

```bash
# 检查 GPU 显存占用
watch -n 1 nvidia-smi

# 恢复训练 (从断点继续)
CUDA_VISIBLE_DEVICES=6 python train_phase3.py \
    --model runs/train/phase3_channelc2f_n/weights/last.pt \
    --resume

# 或调整超参数后重新训练
# - 降低 batch_size
# - 使用梯度累积 (accumulate=2)
```

---

### 问题 5: Medium mAP 没有提升

**症状**: 训练完成后 Medium mAP 仍然低于 18%

**分析步骤**:

```bash
# 1. 检查 ChannelAttention 是否生效
python test_phase3.py  # 确认 model.model.6.ca 存在

# 2. 对比 Phase 1 baseline
python compare_phases.py --baseline phase1_test7 --current phase3_channelc2f_n

# 3. 分析失败案例
# 查看 validation 输出的预测图 (runs/val/phase3_val_n/)
```

**可能的改进方向**:

1. **调整 reduction 参数** (默认 16):
   - 尝试 `reduction=8` (更强的通道注意力)
   - 尝试 `reduction=32` (更轻量的通道注意力)
2. **增加 Layer 6 的 repeats**:
   - 修改 YAML: `[-1, 6, ChannelC2f, [512, True, 1, 0.5, 16]]` (从 4 改为 6)
3. **尝试不同的融合位置**:
   - 在 Layer 8 (P5/32) 也添加 ChannelC2f
   - 或在 Neck 部分添加 ChannelAttention

---

## 📊 监控训练进度

### 选项 A: 查看日志文件

```bash
# 实时监控
tail -f logs/phase3_n.log

# 定期检查 mAP
grep "mAP50-95" logs/phase3_n.log

# 查看 epoch 进度
grep "Epoch" logs/phase3_n.log | tail -n 10
```

### 选项 B: TensorBoard (如果启用)

```bash
# 服务器端启动
tensorboard --logdir runs/train --port 6006

# 本地浏览器访问 (需要端口转发)
# 在本地终端执行:
ssh -L 6006:localhost:6006 ubuntu@10.16.62.111

# 浏览器打开: http://localhost:6006
```

### 选项 C: 定期验证

```bash
# 每隔几个 epoch 运行一次验证
CUDA_VISIBLE_DEVICES=6 python val_depeth.py \
    --model runs/train/phase3_channelc2f_n/weights/last.pt \
    --data data/visdrone-rgbd.yaml
```

---

## 🎓 八股知识点

### 1. Channel Attention (SE Block) 原理

**标准定义**:

- **Squeeze**: 全局平均池化 `[B,C,H,W] → [B,C,1,1]`
- **Excitation**: 两层全连接 (bottleneck + expansion) + Sigmoid
- **Reweight**: 逐通道乘法 `y = x * σ(FC2(ReLU(FC1(GAP(x)))))`

**本项目应用**:

```python
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),  # Bottleneck
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),  # Expansion
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)  # [B,C,H,W] → [B,C,1,1]
        y = self.fc(y)        # Learn channel importance [0-1]
        return x * y.expand_as(x)  # Reweight channels
```

**关键参数 - reduction**:

- `reduction=16`: 标准 SE block (SENet 论文默认值)
- `reduction=8`: 更强的注意力 (更多参数，更强表达能力)
- `reduction=32`: 更轻量 (更少参数，更快速度)

**为什么有效?**

- 学习通道间依赖关系 (channel-wise recalibration)
- 自适应地强化重要通道，抑制冗余通道
- 对小目标和中等目标特别有效 (需要更强的特征判别能力)

---

### 2. parse_model() 中的 base_modules 和 repeat_modules

**base_modules** (frozenset):

- **作用**: 自动插入 `c1` (输入通道) 和 `c2` (输出通道) 参数
- **原理**:
  ```python
  if m in base_modules:
      c1, c2 = ch[f], args[0]  # 从前一层获取输入通道，从 YAML 获取输出通道
      args = [c1, c2, *args[1:]]  # 插入到 args 开头
  ```
- **示例**:
  ```yaml
  # YAML: [-1, 1, Conv, [512, 3, 2]]
  # parse_model() 自动转换为:
  # Conv(c1=256, c2=512, k=3, s=2)
  #      ^^^^  从前一层推断
  ```

**repeat_modules** (frozenset):

- **作用**: 自动插入 `n` (重复次数) 参数
- **原理**:
  ```python
  if m in repeat_modules:
      n = max(round(n * depth), 1)  # 根据 depth multiplier 缩放
      args.insert(2, n)  # 插入到第 3 个位置 (after c1, c2)
  ```
- **示例**:
  ```yaml
  # YAML: [-1, 4, ChannelC2f, [512, True, 1, 0.5, 16]]
  #            ^
  #            repeats field
  # parse_model() 自动转换为:
  # ChannelC2f(c1=512, c2=512, n=4, shortcut=True, g=1, e=0.5, reduction=16)
  #                            ^^^  从 repeats field 插入
  ```

**为什么需要注册?**

- 如果不注册 → YAML 参数直接传递给构造函数 → 参数错位 → TypeError
- 如果注册 → parse_model() 自动处理参数插入 → 参数对齐正确

**常见错误**:

```python
# ❌ 忘记注册 ChannelC2f
base_modules = frozenset({Conv, ...})  # 没有 ChannelC2f
# YAML: [-1, 4, ChannelC2f, [512, 4, True, 1, 0.5, 16]]
# 实际调用: ChannelC2f(512, 4, True, 1, 0.5, 16)
# 期望调用: ChannelC2f(c1=512, c2=512, n=4, shortcut=True, g=1, e=0.5, reduction=16)
# 结果: TypeError

# ✅ 正确注册
base_modules = frozenset({..., ChannelC2f})
repeat_modules = frozenset({..., ChannelC2f})
# YAML: [-1, 4, ChannelC2f, [512, True, 1, 0.5, 16]]
# 实际调用: ChannelC2f(c1=512, c2=512, n=4, shortcut=True, g=1, e=0.5, reduction=16)
# 结果: ✅ 正确构建
```

---

### 3. P4 层 (16x16 下采样) 为什么对 Medium 目标重要?

**Feature Pyramid 结构**:

```
P3 (8x8):    对应小目标 (0-32²)     [80x80 feature map]
P4 (16x16):  对应中等目标 (32²-96²)  [40x40 feature map] ⭐
P5 (32x32):  对应大目标 (96²+)      [20x20 feature map]
```

**VisDrone Dataset 中的 Medium 目标**:

- **定义**: 32² ≤ bbox_area < 96² (1024 到 9216 像素)
- **典型尺寸**: 汽车 (~40x60), 行人 (~20x50), 卡车 (~50x80)
- **检测难点**:
  - 特征表达不足 (P4 层 40x40 feature map 上只有几个像素)
  - 容易被背景干扰 (UAV 视角下背景复杂)
  - 与 Small/Large 目标混淆 (边界模糊)

**为什么 Phase 1 的 P4 层不够?**

```python
# Phase 1 (baseline):
- [-1, 1, Conv, [512, 3, 2]]       # 5-P4/16 (下采样)
- [-1, 4, A2C2f, [512, True, 1]]   # 6-P4/16 (特征提取)
#        ^^^^^^
#        A2C2f: 标准 C2f + 残差连接，但没有通道注意力
```

**Phase 3 的改进**:

```python
# Phase 3 (ChannelC2f):
- [-1, 1, Conv, [512, 3, 2]]                     # 5-P4/16 (下采样)
- [-1, 4, ChannelC2f, [512, True, 1, 0.5, 16]]   # 6-P4/16 (特征提取 + 通道注意力)
#        ^^^^^^^^^^^
#        ChannelC2f = A2C2f + ChannelAttention
#                     自适应强化重要通道 (如边缘、纹理、形状)
```

**预期效果**:

- ChannelAttention 学习到 Medium 目标的判别性通道 (如车辆边缘、行人轮廓)
- 抑制背景干扰通道 (如道路纹理、树木)
- 提升 Medium mAP: 14.28% → 20%+ (至少 +5.7%)

---

## 🔄 下一步计划

### Phase 3 成功后 (Medium mAP ≥20%)

- **Phase 4**: SOLR Loss (Spatial-aware Object Localization Refinement)
- **Phase 5**: 多尺度测试与对比 RemDet
- **Phase 6**: 论文撰写与投稿

### Phase 3 失败 (Medium mAP <18%)

- **方案 A**: 调整 ChannelAttention 参数 (reduction, 位置)
- **方案 B**: 尝试其他注意力机制 (CBAM, ECA, Coordinate Attention)
- **方案 C**: 增强 P4 层容量 (更多 repeats, 更宽通道)

---

## ✅ 部署检查清单

- [ ] 本地上传 13 个文件 (核心 3 + 配置 5 + 脚本 5)
- [ ] 服务器验证通过 (`python verify_phase3.py`)
- [ ] 模型构建测试通过 (`python test_phase3.py`)
- [ ] 添加脚本执行权限 (`chmod +x train_all_scales.sh`)
- [ ] 启动 YOLO12n 训练 (优先级最高)
- [ ] 监控训练日志 (`tail -f logs/phase3_n.log`)
- [ ] 定期检查 mAP 进度 (每 10 epoch)
- [ ] 训练完成后验证结果 (`python val_depeth.py`)
- [ ] 对比 Phase 1 baseline (Medium mAP 是否提升)
- [ ] 决定是否继续训练其他尺度

---

**准备就绪！现在可以开始上传文件并启动训练了。** 🚀

**推荐顺序**:

1. 先上传所有文件
2. 运行验证和测试
3. 启动 YOLO12n (2 天，最快验证)
4. 如果 YOLO12n 成功 → 启动 YOLO12s (3 天，主要对比)
5. 其他尺度根据需要训练

**有任何问题随时问我！Good luck!** 💪
