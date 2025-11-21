# 八股知识点 #53: PyTorch 模型加载的三种方式与 strict 参数

## 📚 标准例子

### 场景 1: 完全加载 (strict=True, 默认)

```python
import torch
from ultralytics import YOLO

# 创建模型
model = YOLO('yolo12n.pt')

# 加载另一个相同架构的权重
state_dict = torch.load('another_yolo12n.pt')['model']
model.model.load_state_dict(state_dict, strict=True)  # ✅ 默认行为
```

**特点**:

- 要求权重文件和模型架构**完全一致**
- 键名必须完全匹配 (包括层数、参数名)
- 如果有缺失或多余的键，抛出异常

**适用场景**:

- 恢复训练 (resume)
- 加载完全相同架构的 checkpoint
- 严格的模型验证

---

### 场景 2: 部分加载 (strict=False)

```python
# 创建新架构 (比原模型多了GGFE模块)
model = YOLO('yolo12-rgbd-ggfe-universal.yaml')

# 加载旧架构的权重 (没有GGFE参数)
state_dict = torch.load('yolo12n.pt')['model']
incompatible = model.model.load_state_dict(state_dict, strict=False)  # ✅ 允许不匹配

# 检查不匹配的键
print(f"Missing keys: {len(incompatible.missing_keys)}")  # GGFE的参数
print(f"Unexpected keys: {len(incompatible.unexpected_keys)}")  # 旧模型多余的参数
```

**特点**:

- 只加载**匹配的键**，忽略不匹配的
- `missing_keys`: 模型有但权重没有 → 保持随机初始化
- `unexpected_keys`: 权重有但模型没有 → 直接忽略
- **不抛出异常**，返回不匹配信息

**适用场景** (本项目的核心):

- 在预训练基础上添加新模块 (GGFE, SADF 等)
- 迁移学习 (backbone 相同，head 不同)
- 模型架构渐进式改进

---

### 场景 3: 直接加载权重文件 (YOLO 特有)

```python
# Ultralytics的便捷方法
model = YOLO('yolo12n.pt')  # ❌ 加载权重 + 架构

# 等价于:
ckpt = torch.load('yolo12n.pt')
model_config = ckpt['model'].yaml  # 权重文件内嵌的YAML配置
model = YOLO(model_config)  # 使用内嵌配置创建模型
model.model.load_state_dict(ckpt['model'].state_dict())  # 加载权重
```

**特点**:

- 权重文件`.pt`包含模型架构信息 (YAML)
- `YOLO(weights_path)`会使用权重的架构
- **忽略命令行传入的 YAML 配置** (这是我们的 bug 根源!)

**陷阱**:

```python
# ❌ 错误: 期望加载GGFE配置，但实际加载yolo12n.pt的配置
model = YOLO('yolo12n.pt')
# model的架构来自yolo12n.pt内嵌的YAML (没有GGFE!)

# ✅ 正确: 先创建GGFE架构，再加载参数
model = YOLO('yolo12-rgbd-ggfe-universal.yaml')  # 创建GGFE架构
state_dict = torch.load('yolo12n.pt')['model']
model.model.load_state_dict(state_dict, strict=False)  # 只加载匹配的参数
```

---

## 🔧 本项目应用

### 问题代码 (train_depth_solr_v2.py 第 136-141 行)

```python
if args.weights:
    model = YOLO(args.weights)  # ❌ 场景3: 直接加载yolo12n.pt
    # 结果: 使用yolo12n.pt的架构 (没有GGFE)
else:
    model = YOLO(args.model)  # ✅ 场景1: 使用YAML创建架构
```

**Bug 分析**:

1. 用户总是提供`--weights yolo12n.pt`
2. 进入`if args.weights`分支
3. `YOLO(args.weights)`加载 yolo12n.pt 的**架构** + 权重
4. GGFE YAML 配置被完全忽略
5. 训练 300 个 epoch，但架构始终是标准 yolo12n (3.0M 参数)

---

### 修复代码 (train_depth_solr_v2_fixed.py 第 128-177 行)

```python
# ✅ Step 1: 总是从YAML创建架构
model_yaml = 'ultralytics/cfg/models/12/yolo12-rgbd-ggfe-universal.yaml'
model = YOLO(model_yaml, task='detect')  # 场景1: 创建GGFE架构

# ✅ Step 2: 如果提供weights，使用场景2 (部分加载)
if args.weights:
    ckpt = torch.load(args.weights, map_location='cpu')
    state_dict = ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']

    # 场景2: strict=False, 只加载匹配的参数
    incompatible = model.model.load_state_dict(state_dict, strict=False)

    # GGFE模块的参数在missing_keys中 (随机初始化)
    print(f"Missing keys (GGFE): {len(incompatible.missing_keys)}")
```

**修复效果**:

1. 模型架构来自 GGFE YAML (3.5M 参数)
2. Backbone 参数来自 yolo12n.pt (预训练)
3. GGFE 参数随机初始化 (从头训练)
4. 两者完美结合!

---

## 🎓 深入讲解

### Q1: 为什么 strict=False 不会报错，但模型能正常训练?

**A1**:

- PyTorch 允许**部分参数未初始化**
- `missing_keys`中的参数会保持`__init__`中的初始化 (随机/零初始化)
- 训练时梯度正常反向传播，未加载的参数从头学习

**示例**:

```python
class Model(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 64, 3)  # ← 随机初始化
        self.ggfe = GGFE(64)              # ← 随机初始化

    def forward(self, x):
        x = self.conv1(x)
        x = self.ggfe(x)
        return x

# 加载预训练权重 (只有conv1)
state_dict = {'conv1.weight': ..., 'conv1.bias': ...}
model.load_state_dict(state_dict, strict=False)

# 结果:
# conv1: 来自预训练 (ImageNet特征)
# ggfe:  随机初始化 (需要从头训练)
```

---

### Q2: missing_keys 有 100+个，会不会影响训练效果?

**A2**: **不会**，这是正常现象!

**原因**:

- 每个 GGFE 模块有~20 个参数 (Conv, BN, Attention 等)
- 3 个融合点 (P3/P4/P5) × 20 参数/模块 = 60 个 missing keys
- 加上 GeometryPriorGenerator 的参数 → 100+个 missing keys

**优势**:

- Backbone (conv, C2f 等) 来自预训练 → **收敛快**
- GGFE 从头学习 → **适应深度图特性**
- 比完全随机初始化好得多!

**对比**:

```
| 初始化方式      | Backbone | GGFE | 训练难度 | 最终精度 |
|----------------|----------|------|---------|---------|
| 完全随机        | 随机     | 随机 | 高      | 低      |
| 预训练+strict=True | 预训练 | ❌无法加载 | N/A | N/A |
| 预训练+strict=False | 预训练 | 随机 | 中 ✅ | 高 ✅ |
```

---

### Q3: unexpected_keys 是什么，需要担心吗?

**A3**: 通常**不需要**担心

**场景**:

- 旧模型有额外的层 (如分类头)
- 新模型删除了某些模块
- 权重文件包含优化器状态 (非模型参数)

**示例**:

```python
# 旧模型 (分类任务)
old_model = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.Linear(64, 1000)  # ImageNet 1000类
)

# 新模型 (检测任务)
new_model = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.Linear(64, 80)  # COCO 80类
)

# 加载
state_dict = old_model.state_dict()
new_model.load_state_dict(state_dict, strict=False)
# unexpected_keys: ['1.weight', '1.bias']  ← 旧的Linear层
# missing_keys: ['1.weight', '1.bias']     ← 新的Linear层
```

**本项目**:

- yolo12n.pt → yolo12-ggfe: 通常**没有 unexpected_keys**
- 因为 GGFE 是**添加**模块，而非替换

---

## 💡 常见追问

### Q4: 如果 missing_keys 太多 (>50%), 预训练还有用吗?

**A**: 有用，但效果递减

**经验法则**:

- Missing < 10%: 预训练效果显著 (收敛快, 精度高)
- Missing 10-30%: 预训练仍有帮助 (中等收敛速度)
- Missing 30-50%: 预训练效果微弱 (轻微加速)
- Missing > 50%: 预训练几乎无用 (考虑从头训练)

**本项目**:

- GGFE 参数: ~0.5M
- 总参数: 3.5M
- Missing 率: 0.5/3.5 = 14% ✅ 预训练仍有效!

---

### Q5: strict=False 会不会加载错误的参数?

**A**: **不会**，PyTorch 按**键名精确匹配**

**安全机制**:

```python
# 模型
model.conv1.weight  # shape: [64, 3, 3, 3]
model.conv2.weight  # shape: [128, 64, 3, 3]

# 权重文件
state_dict = {
    'conv1.weight': torch.randn(64, 3, 3, 3),   # ✅ 匹配
    'conv2.weight': torch.randn(128, 64, 3, 3), # ✅ 匹配
    'conv3.weight': torch.randn(256, 128, 3, 3) # ❌ 模型没有conv3
}

# 加载 (strict=False)
model.load_state_dict(state_dict, strict=False)
# 结果: conv1, conv2加载成功; conv3被忽略
```

**键名必须完全一致**:

- `model.4.rgbd_fusion.conv.weight` ✅ 匹配
- `model.4.rgbd_fusion.ggfe.conv.weight` ❌ 不在 state_dict 中 (missing)

---

## ⚠️ 易错点

### 易错点 1: 混淆"加载架构"和"加载参数"

```python
# ❌ 错误理解: 以为只加载参数
model = YOLO('yolo12n.pt')
# 实际: 同时加载架构 + 参数

# ✅ 正确做法: 分离架构和参数
model = YOLO('custom.yaml')  # 架构来自YAML
state_dict = torch.load('yolo12n.pt')['model']
model.model.load_state_dict(state_dict, strict=False)  # 只加载参数
```

---

### 易错点 2: 忘记检查 missing_keys

```python
# ❌ 危险: 不检查不匹配
model.load_state_dict(state_dict, strict=False)
# 可能所有参数都missing，但不报错!

# ✅ 安全: 检查加载结果
incompatible = model.load_state_dict(state_dict, strict=False)
if len(incompatible.missing_keys) > len(state_dict) * 0.5:
    print("⚠️  Warning: 超过50%的参数未加载!")
```

---

### 易错点 3: state_dict 格式不统一

```python
# Ultralytics权重文件结构
ckpt = torch.load('yolo12n.pt')
# ckpt = {
#     'model': DetectionModel(...),  # ← 模型对象
#     'optimizer': ...,
#     'epoch': 300,
#     ...
# }

# ❌ 错误: 直接传整个ckpt
model.load_state_dict(ckpt, strict=False)  # TypeError!

# ✅ 正确: 提取state_dict
if isinstance(ckpt, dict) and 'model' in ckpt:
    state_dict = ckpt['model'].state_dict()  # 模型对象 → 字典
else:
    state_dict = ckpt  # 已经是字典
model.load_state_dict(state_dict, strict=False)
```

---

## 📖 拓展阅读

### 官方文档

- [torch.nn.Module.load_state_dict](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict)
- [Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

### 相关八股

- 知识点 #14: PyTorch checkpoint 保存与恢复
- 知识点 #52: 接口设计陷阱 (本次 bug 的前序)

### 迁移学习经典论文

- [How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792)
- 发现: 浅层特征 (边缘, 纹理) 更通用，深层特征 (语义) 更任务特定

---

## 🧪 思考题

### 题目 1: 参数初始化顺序

```python
# 创建模型 (默认随机初始化)
model = YOLO('yolo12-ggfe.yaml')

# 1. 先加载预训练权重 (strict=False)
state_dict = torch.load('yolo12n.pt')['model']
model.load_state_dict(state_dict, strict=False)

# 2. 然后手动初始化GGFE
for name, module in model.named_modules():
    if 'ggfe' in name.lower():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)

# 问题: 这个顺序正确吗? 为什么?
```

<details>
<summary>答案</summary>

**正确**!

原因:

1. `load_state_dict(strict=False)`只覆盖**匹配的键**
2. GGFE 参数在 missing_keys 中，保持随机初始化
3. 步骤 2 手动重新初始化 GGFE (覆盖步骤 1 的随机值)
4. 最终: Backbone=预训练, GGFE=Kaiming 初始化

**最佳实践**: 在创建模型时直接指定初始化方法 (在`__init__`中)

</details>

---

### 题目 2: 诊断参数未加载

```python
# 加载后发现精度很低
model.load_state_dict(state_dict, strict=False)

# 如何快速诊断是否大量参数未加载?
# 提示: 不要手动数missing_keys
```

<details>
<summary>答案</summary>

```python
incompatible = model.load_state_dict(state_dict, strict=False)

# 方法1: 统计missing参数量
total_params = sum(p.numel() for p in model.parameters())
missing_params = sum(
    model.state_dict()[k].numel()
    for k in incompatible.missing_keys
)
missing_ratio = missing_params / total_params
print(f"Missing: {missing_ratio*100:.1f}%")

# 方法2: 对比state_dict大小
loaded_keys = len(state_dict) - len(incompatible.unexpected_keys)
model_keys = len(model.state_dict())
print(f"Loaded: {loaded_keys}/{model_keys} keys")

# 方法3: 检查特定模块
for name, param in model.named_parameters():
    if name not in state_dict:
        print(f"Not loaded: {name} ({param.numel()} params)")
```

</details>

---

### 题目 3: strict=False 的安全性

```python
# 场景: 不小心加载了错误的权重文件
model = YOLO('yolo12n.yaml')  # 检测模型
state_dict = torch.load('yolo12-cls.pt')['model']  # 分类模型的权重
model.load_state_dict(state_dict, strict=False)

# 问题: 会发生什么? 模型能正常训练吗?
```

<details>
<summary>答案</summary>

**取决于架构重叠度**:

1. **Backbone 完全相同** (常见):

   - Backbone 参数加载成功
   - Head 参数全部 missing (随机初始化)
   - ✅ 能正常训练，但 Head 需要从头学习

2. **Backbone 部分相同**:

   - 匹配的层加载成功
   - 不匹配的层随机初始化
   - ✅ 能训练，但预训练效果打折扣

3. **完全不同**:
   - 所有参数 missing
   - ⚠️ 等价于随机初始化，预训练无用

**安全检查**:

```python
incompatible = model.load_state_dict(state_dict, strict=False)
loaded_ratio = 1 - len(incompatible.missing_keys) / len(model.state_dict())
if loaded_ratio < 0.3:
    raise ValueError(f"只加载了{loaded_ratio*100:.1f}%的参数，可能权重文件不匹配!")
```

</details>

---

## 📝 总结

**核心要点**:

1. `YOLO(weights_path)` = 加载架构 + 参数 (危险!)
2. `strict=False` = 只加载匹配的键 (安全且灵活)
3. 修改架构时必须分离: 先创建新架构，再加载旧参数
4. missing_keys < 50%时，预训练仍有效

**本项目教训**:

- ❌ 错误: `model = YOLO(args.weights)` → 忽略 GGFE 配置
- ✅ 正确: `model = YOLO(yaml); model.load_state_dict(state_dict, strict=False)`

**适用场景**:

- 添加新模块 (GGFE, SADF, Attention 等)
- 迁移学习 (换 head, 换 backbone)
- 渐进式架构改进 (逐步添加模块)

**记住**: PyTorch 的灵活性是双刃剑，strict=False 很强大，但必须理解其行为! 🎯
