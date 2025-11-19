# 🎯 SOLR 快速参考卡

> **一页纸速查表** - 最常用的命令和参数

---

## 📋 基础命令

### 测试 SOLR 模块 (5 分钟)

```bash
python -m ultralytics.utils.solr_loss
```

**预期**: `✅ All tests passed!`

---

### 快速测试训练 (1 小时)

```bash
python train_depth_solr.py \
    --data data/visdrone-rgbd.yaml \
    --cfg n \
    --epochs 10 \
    --batch 32 \
    --name test_10ep
```

---

### 完整训练 (12-15 天)

```bash
nohup python train_depth_solr.py \
    --data data/visdrone-rgbd.yaml \
    --cfg s \
    --epochs 300 \
    --batch 16 \
    --name solr_s_300ep \
    > train_s.log 2>&1 &
```

---

### 批量训练所有尺寸

```bash
# Linux
bash batch_train_solr_all_sizes.sh

# Windows
batch_train_solr_all_sizes.bat
```

---

## 🎚️ 模型尺寸参数

| `--cfg` | 参数量 | FLOPs | batch | 显存  | 时间/epoch |
| ------- | ------ | ----- | ----- | ----- | ---------- |
| `n`     | ~3M    | ~8G   | 32    | ~8GB  | ~30min     |
| `s` ⭐  | ~11M   | ~46G  | 16    | ~12GB | ~1h        |
| `m`     | ~22M   | ~92G  | 8     | ~16GB | ~2h        |
| `l`     | ~44M   | ~184G | 4     | ~20GB | ~4h        |
| `x`     | ~66M   | ~276G | 2     | ~22GB | ~6h        |

---

## ⚙️ SOLR 参数

### 默认配置 (适合 VisDrone)

```bash
--small_weight 2.5    # 小目标 (<32px)
--medium_weight 2.0   # 中等目标 (32-96px) ← 关键!
--large_weight 1.0    # 大目标 (>96px)
```

---

### 激进配置 (AP_m 不足时)

```bash
--medium_weight 2.5  # 增加到2.5x
```

---

### 小目标专项优化

```bash
--small_weight 3.0   # 增加到3.0x
```

---

## 📊 预期结果 (VisDrone)

| 模型 | AP@0.5 | AP_m   | RemDet | 提升  |
| ---- | ------ | ------ | ------ | ----- |
| s    | 46~48% | 41~43% | 42.3%  | +4~6% |
| m    | 48~50% | 43~45% | 45.0%  | +3~5% |
| x    | 51~53% | 46~48% | 48.3%  | +3~5% |

---

## 🔍 常用监控命令

### 查看训练日志

```bash
tail -f runs/train_solr/solr_s_300ep/train.log
```

---

### 查看最新 mAP

```bash
tail -20 runs/train_solr/solr_s_300ep/results.txt
```

---

### 检查 GPU 占用

```bash
nvidia-smi
watch -n 1 nvidia-smi  # 实时监控
```

---

### 查看训练进度

```bash
grep "Epoch" runs/train_solr/solr_s_300ep/train.log | tail -5
```

---

## 🚨 故障排查

| 问题           | 原因              | 解决               |
| -------------- | ----------------- | ------------------ |
| `--cfg` 不识别 | 版本过旧          | `git pull`         |
| CUDA OOM       | batch 太大        | 减小`--batch`      |
| Import Error   | 文件缺失          | 检查`solr_loss.py` |
| 参数量不对     | model_name 未设置 | 检查代码版本       |

---

## 📁 关键文件路径

```
yolo12-bimodal/
├── train_depth_solr.py                      ← 训练脚本
├── ultralytics/utils/solr_loss.py           ← SOLR核心
├── ultralytics/cfg/models/12/
│   └── yolo12-rgbd-v2.1-universal.yaml      ← 模型配置
├── batch_train_solr_all_sizes.sh            ← 批量训练(Linux)
├── batch_train_solr_all_sizes.bat           ← 批量训练(Windows)
└── SOLR多尺寸训练指南.md                    ← 完整文档
```

---

## 🎯 推荐训练流程

1. **本地测试** (5 分钟):

   ```bash
   python -m ultralytics.utils.solr_loss
   ```

2. **上传代码** (5 分钟):

   ```bash
   git add . && git commit -m "Add SOLR" && git push
   ```

3. **服务器快速测试** (1 小时):

   ```bash
   python train_depth_solr.py --cfg n --epochs 10
   ```

4. **启动主力训练** (12-15 天):

   ```bash
   nohup python train_depth_solr.py --cfg s --epochs 300 > train_s.log 2>&1 &
   ```

5. **COCO 评估**:
   ```bash
   python val_coco_eval.py --weights runs/train_solr/solr_s_300ep/weights/best.pt
   ```

---

## 💡 快速技巧

### 后台运行不中断

```bash
nohup python train_depth_solr.py ... > train.log 2>&1 &
```

---

### 实时监控多个指标

```bash
# 新开终端1: 监控日志
tail -f train.log

# 新开终端2: 监控GPU
watch -n 1 nvidia-smi

# 新开终端3: 监控mAP
watch -n 60 "tail -5 runs/train_solr/solr_s_300ep/results.txt"
```

---

### 批量训练特定尺寸

```bash
# 只训练s和m
bash batch_train_solr_all_sizes.sh s m
```

---

### 自定义实验名称

```bash
python train_depth_solr.py \
    --cfg s \
    --name solr_s_m25_300ep \  # 自定义名称
    --medium_weight 2.5         # 加入配置信息
```

---

## 📞 获取帮助

```bash
# 查看所有参数
python train_depth_solr.py --help

# 查看SOLR参数
python train_depth_solr.py --help | grep -A 2 "SOLR"
```

---

**打印此页,贴在显示器旁边!** 📌
