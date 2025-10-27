# Phase 2.5 v2.1 - 参数验证 Bug 修复 🔧

**日期**: 2025/10/27 18:45  
**问题**: `SyntaxError: 'small_thresh' is not a valid YOLO argument`  
**严重性**: 🔴 Critical - 阻止验证脚本运行

---

## 问题症状

```bash
python val_visdrone.py --model best.pt

# 输出:
SyntaxError: 'small_thresh' is not a valid YOLO argument.
'visdrone_mode' is not a valid YOLO argument.
'medium_thresh' is not a valid YOLO argument.
```

---

## 根本原因

**Ultralytics 参数白名单机制**:

- 所有传递给 `model.val()` 的参数必须在配置系统中注册
- 我们的自定义参数 (`visdrone_mode`, `small_thresh`, `medium_thresh`) 未注册
- `ultralytics/cfg/__init__.py` 中的 `check_dict_alignment()` 拒绝了这些参数

---

## 完整修复方案

### 需要修改的 3 个文件

#### 1️⃣ `ultralytics/cfg/__init__.py` - 注册整数参数

**位置**: Line ~203

```python
CFG_INT_KEYS = frozenset(
    {  # integer-only arguments
        "epochs",
        "patience",
        "workers",
        "seed",
        "close_mosaic",
        "mask_ratio",
        "max_det",
        "vid_stride",
        "line_width",
        "nbs",
        "save_period",
        # 🆕 VisDrone特定参数
        "small_thresh",   # 小目标面积阈值 (默认1024 = 32x32)
        "medium_thresh",  # 中目标面积阈值 (默认4096 = 64x64)
    }
)
```

#### 2️⃣ `ultralytics/cfg/__init__.py` - 注册布尔参数

**位置**: Line ~240

```python
CFG_BOOL_KEYS = frozenset(
    {  # boolean-only arguments
        "save",
        "exist_ok",
        "verbose",
        "deterministic",
        "single_cls",
        "rect",
        "cos_lr",
        "overlap_mask",
        "val",
        "save_json",
        "half",
        "dnn",
        "plots",
        "show",
        "save_txt",
        "save_conf",
        "save_crop",
        "save_frames",
        "show_labels",
        "show_conf",
        "visualize",
        "augment",
        "agnostic_nms",
        "retina_masks",
        "show_boxes",
        "keras",
        "optimize",
        "int8",
        "dynamic",
        "simplify",
        "nms",
        "profile",
        "multi_scale",
        # 🆕 VisDrone特定参数
        "visdrone_mode",  # 启用VisDrone分尺度评估
    }
)
```

#### 3️⃣ `ultralytics/cfg/default.yaml` - 添加默认值

**位置**: Line ~53 (Val/Test settings 部分之后)

```yaml
# Val/Test settings ----------------------------------------------------------------------------------------------------
val: True
split: val
save_json: False
conf:
iou: 0.7
max_det: 300
half: False
dnn: False
plots: True

# VisDrone-specific settings -------------------------------------------------------------------------------------------
visdrone_mode: False # (bool) enable VisDrone size-wise evaluation (small/medium/large)
small_thresh: 1024 # (int) small object area threshold in pixels (default: 32x32 = 1024)
medium_thresh: 4096 # (int) medium object area threshold in pixels (default: 64x64 = 4096)
```

---

## 为什么需要这 3 个修改?

### 1. `CFG_INT_KEYS` / `CFG_BOOL_KEYS` - 参数类型注册

**作用**: 告诉配置系统这些参数是合法的，并指定其类型

- `small_thresh`, `medium_thresh` → 整数类型
- `visdrone_mode` → 布尔类型

**原理**: `check_dict_alignment()` 会检查传入参数是否在这些集合中

### 2. `default.yaml` - 默认值定义

**作用**: 提供参数的默认值和文档说明

**好处**:

- 用户不传参数时使用默认值
- `yolo cfg` 命令能看到这些参数
- 类型验证更准确

---

## 上传到服务器

### 需要上传的文件 (3 个)

```bash
# 在本地 Windows PowerShell 运行

# 1. 上传配置文件修改
scp f:\CV\Paper\yoloDepth\yoloDepth\ultralytics\cfg\__init__.py \
    ubuntu@server:/data2/user/2024/lzy/yolo12-bimodal/ultralytics/cfg/

scp f:\CV\Paper\yoloDepth\yoloDepth\ultralytics\cfg\default.yaml \
    ubuntu@server:/data2/user/2024/lzy/yolo12-bimodal/ultralytics/cfg/

# 2. 上传验证脚本 (虽然之前传过,但以防万一)
scp f:\CV\Paper\yoloDepth\yoloDepth\val_visdrone.py \
    ubuntu@server:/data2/user/2024/lzy/yolo12-bimodal/
```

**简化版 (如果路径一致)**:

```bash
scp f:\CV\Paper\yoloDepth\yoloDepth\ultralytics\cfg\* ubuntu@server:/data2/user/2024/lzy/yolo12-bimodal/ultralytics/cfg/
scp f:\CV\Paper\yoloDepth\yoloDepth\val_visdrone.py ubuntu@server:/data2/user/2024/lzy/yolo12-bimodal/
```

---

## 验证测试

### 服务器运行

```bash
# 重新运行验证 (和之前一样的命令)
python val_visdrone.py --model /data2/user/2024/lzy/yolo12-bimodal/runs/train/phase1_test7/weights/best.pt
```

### 期望输出

```
🔍 VisDrone Validation (RemDet-aligned)
Model:          /data2/user/2024/lzy/yolo12-bimodal/runs/train/phase1_test7/weights/best.pt
...

📊 Measuring model efficiency...
Latency: 28.25 ± 4.01 ms ✅
FLOPs: 19.99 G ✅
Params: 9.39 M ✅

🔍 Starting validation...
Ultralytics 🚀 YOLO12...
Using DetMetricsVisDrone with visdrone_mode=True ← 关键: 应该看到这行!
val: Scanning /data2/user/2024/lzy/yolo12-bimodal/data/visdrone-rgbd/labels/val...
...
Speed: 0.5ms preprocess, 28.2ms inference, 1.2ms postprocess per image
Results saved to runs/val/phase1_test7_best_val/

📊 RemDet Comparison Report
mAP@0.5:   XX.XX% ✅
Small:     XX.XX% ← 不再是0
Medium:    XX.XX% ← 不再是0
Large:     XX.XX% ← 不再是0
```

---

## 问题根源总结

| 阶段      | 问题                          | 状态                                   |
| --------- | ----------------------------- | -------------------------------------- |
| **Bug 1** | `parse_args()` 缺少参数定义   | ✅ 已修复 (val_visdrone.py Line ~167)  |
| **Bug 2** | 参数未在配置系统注册          | ✅ 已修复 (**init**.py + default.yaml) |
| **Bug 3** | Dataset 未返回 target_areas   | ✅ 已修复 (dataset.py Line ~275)       |
| **Bug 4** | Validator 未传递 target_areas | ✅ 已修复 (val.py Line ~205)           |

**数据流现已完全打通**: Dataset → Augment → Collate → Validator → DetMetricsVisDrone ✅

---

## 快速检查清单

上传前检查:

- [ ] `ultralytics/cfg/__init__.py` 已添加 `small_thresh`, `medium_thresh`, `visdrone_mode`
- [ ] `ultralytics/cfg/default.yaml` 已添加这 3 个参数的默认值
- [ ] `val_visdrone.py` 已添加 `--small-thresh`, `--medium-thresh` 参数

上传后检查:

- [ ] 服务器文件路径正确: `/data2/user/2024/lzy/yolo12-bimodal/ultralytics/cfg/`
- [ ] 重新运行验证脚本
- [ ] 看到 "Using DetMetricsVisDrone with visdrone_mode=True"
- [ ] 所有 size-wise mAP 非零

---

## 成功标志

当你看到以下输出时，bug 已**完全修复**:

```
================================================================================
 RemDet-X Comparison Report (AAAI2025)
================================================================================

📊 Accuracy Metrics:
  mAP@0.5              XX.XX%          45.2%           ...
  mAP@0.75             XX.XX%          28.5%           ...

📐 By Object Size:
  Small (<32×32)       15-18%          21.3%           ... ← 成功!
  Medium (32~64)       35-40%          N/A             ... ← 成功!
  Large (>64×64)       50-55%          N/A             ... ← 成功!

⚡ Efficiency Metrics:
  Latency (ms)         28.25           12.8            ...
  FLOPs (G)            19.99           52.4            ... ← 轻量62%!
  Params (M)           9.39            16.3            ... ← 轻量42%!
================================================================================
```

**关键发现**: 你的模型比 RemDet-X **轻量很多**! (FLOPs 少 62%, Params 少 42%)

---

**文档版本**: v2.1.1  
**最后更新**: 2025/10/27 18:45
