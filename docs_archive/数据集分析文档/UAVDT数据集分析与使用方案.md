# UAVDT 数据集分析与多模态训练方案

**日期**: 2025-10-31  
**数据集位置**: `yoloDepth\datasets\UAVDT`  
**目标**: 将 UAVDT 与 VisDrone 联合训练,复现 RemDet 的多数据集策略

---

## 📊 UAVDT 数据集结构分析

### 1. 基本信息

```
数据集完整路径: yoloDepth/datasets/UAVDT/
├── images/
│   └── UAV-benchmark-M/
│       ├── M0101/  # 序列1
│       ├── M0201/  # 序列2
│       ├── ...
│       └── M1401/  # 序列30
└── annotations/
    ├── UAV-benchmark-M-Train.json  (74.5MB, COCO格式)
    └── UAV-benchmark-M-Val.json
```

**关键数据**:

- ✅ 训练图像: **23,829 张** (vs 之前估算的 23,258,更多!)
- ✅ 训练序列: **30 个视频序列**
- ✅ 总标注数: **422,911 个** (平均每张图 17.75 个目标!)
- ✅ 标注格式: **COCO JSON** (与 VisDrone 的 YOLO 格式不同)

### 2. 类别信息

```python
Categories (3类):
  - ID: 0, Name: car      → 394,633个标注 (93.3%)
  - ID: 1, Name: truck    →  17,491个标注 (4.1%)
  - ID: 2, Name: bus      →  10,787个标注 (2.6%)
```

**与 VisDrone 的类别对应**:

```python
# VisDrone有10个类别(ID 0-9)
UAVDT → VisDrone 映射:
  car (0)   → car (4)        # VisDrone的car是ID 4
  truck (1) → truck (6)      # VisDrone的truck是ID 6
  bus (2)   → bus (9)        # VisDrone的bus是ID 9
```

**重要**: UAVDT 只有车辆类,没有行人/自行车等,这是**领域互补**的关键!

### 3. 目标尺寸分布 (重点!)

```
Small (<32²=1024像素²):   212,423个 (50.2%)  ← 比VisDrone少
Medium (32²-96²):         204,588个 (48.4%)  ← 几乎一半!
Large (>96²=9216像素²):     5,900个 (1.4%)   ← VisDrone的13倍!
```

**关键发现**:

| 数据集   | Small | Medium | Large  | 总计     |
| -------- | ----- | ------ | ------ | -------- |
| VisDrone | 92.4% | 7.5%   | 0.1%   | ~400K    |
| UAVDT    | 50.2% | 48.4%  | 1.4%   | 422,911  |
| 互补性   | 低    | **高** | **高** | **完美** |

**结论**: UAVDT 不是用来提升 Small 的(VisDrone 已经够了),而是**补充 Medium 和 Large 样本**!

---

## 📚 八股知识点: COCO 格式 vs YOLO 格式

### 问题: 什么是 COCO JSON 格式?

**标准答案**:
COCO (Common Objects in Context) JSON 是一种常用的目标检测标注格式,包含 4 个主要字段:

```json
{
  "images": [
    {
      "id": 0,
      "file_name": "M1306/img_mask/img000001.jpg",
      "width": 1024,
      "height": 540
    }
  ],
  "annotations": [
    {
      "id": 0,
      "image_id": 0,
      "category_id": 0,
      "bbox": [829, 179, 45, 20],  # [x_min, y_min, width, height] 绝对像素坐标
      "area": 900,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 0, "name": "car"}
  ]
}
```

**YOLO 格式** (VisDrone 使用的):

```
# 每个图像一个txt文件,每行一个目标
class_id center_x center_y width height  # 归一化到[0,1]
4 0.5234 0.3567 0.0456 0.0234
```

**本项目应用**:

- VisDrone: 已经是 YOLO 格式,可以直接用
- UAVDT: COCO 格式,需要转换 → `convert_uavdt_to_yolo.py`

**转换核心代码**:

```python
# COCO bbox [x, y, w, h] → YOLO [cx, cy, w, h] (归一化)
x_min, y_min, bbox_w, bbox_h = coco_bbox
img_w, img_h = image_width, image_height

center_x = (x_min + bbox_w / 2) / img_w
center_y = (y_min + bbox_h / 2) / img_h
norm_w = bbox_w / img_w
norm_h = bbox_h / img_h

yolo_format = f"{class_id} {center_x} {center_y} {norm_w} {norm_h}"
```

**常见追问**: 为什么 YOLO 用归一化坐标?

- **答**: 归一化后坐标与图像尺寸无关,模型可以处理任意大小的图像
- 训练时 resize 到 640x640,推理时可以用其他尺寸
- 简化数据增强(缩放/裁剪)的坐标变换

**易错点**:

- ❌ 忘记归一化 → bbox 坐标>1,训练失败
- ❌ COCO 的 bbox 是[x,y,w,h],不是[x1,y1,x2,y2]
- ❌ 类别 ID 没有对应 VisDrone → 类别错乱

---

## 🔍 RemDet 论文中的 UAVDT 使用方式

### RemDet 的多数据集策略

根据论文和我们的分析:

**RemDet 使用的数据集**:

1. **VisDrone-DET** (6,471 train) - 主数据集,评估基准
2. **UAVDT** (23,829 train) - 补充 Medium/Large 样本
3. **COCO** (可能用于预训练或辅助训练)

**训练策略** (推测,论文未明确说明):

```
方案推测: 联合训练 (Joint Training)

VisDrone : UAVDT = 1.0 : 0.5 采样权重
    ↓
每个epoch:
  - 60% batch来自VisDrone
  - 40% batch来自UAVDT
    ↓
验证/测试只用VisDrone (对齐benchmark)
```

**为什么这样设计?**

1. **VisDrone 是主任务** - 评估在 VisDrone 上,需要更多采样
2. **UAVDT 是辅助** - 只补充 Medium/Large,不应主导训练
3. **域相似性高** - 都是 UAV 视角,联合训练不会冲突
4. **类别互补** - UAVDT 只有车辆,不影响 VisDrone 的行人/自行车检测

### RemDet 的性能提升分析

| 指标        | RemDet-Tiny | 我们的 baseline (VisDrone only) | 联合训练预期 |
| ----------- | ----------- | ------------------------------- | ------------ |
| Overall mAP | 38.9%       | ~41%                            | **45-47%**   |
| Small mAP   | 12.7%       | **30.94%** (已碾压)             | **35-38%**   |
| Medium mAP  | 33.0%       | 46.24% (已超越)                 | **48-50%**   |
| Large mAP   | 44.5%       | 36.70% (落后)                   | **42-45%**   |

**关键洞察**:

- 我们 Small 已经领先 RemDet-Tiny **143%** (+18.2 个点)
- Large 落后是数据问题 (443 vs 5,900 样本)
- 加入 UAVDT 后,Large 性能将大幅提升
- 有机会在**所有指标上**全面超越 RemDet-Tiny!

---

## 🛠️ UAVDT 数据预处理流程

### Phase 1: COCO JSON → YOLO TXT 转换

#### 1.1 创建转换脚本

```python
# convert_uavdt_to_yolo.py
import json
import os
from pathlib import Path
from tqdm import tqdm

# 类别映射: UAVDT → VisDrone
CATEGORY_MAP = {
    0: 4,  # car → car (VisDrone ID 4)
    1: 6,  # truck → truck (VisDrone ID 6)
    2: 9   # bus → bus (VisDrone ID 9)
}

def convert_coco_to_yolo(json_path, images_root, output_root):
    """
    将UAVDT的COCO JSON转换为YOLO格式

    Args:
        json_path: COCO JSON文件路径
        images_root: 图像根目录 (UAV-benchmark-M/)
        output_root: 输出根目录
    """
    print(f"加载 {json_path}...")
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    images = {img['id']: img for img in coco_data['images']}
    annotations = coco_data['annotations']

    # 创建输出目录
    labels_dir = Path(output_root) / 'labels'
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 按图像ID分组标注
    img_annotations = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)

    print(f"转换 {len(images)} 张图像的标注...")
    for img_id, img_info in tqdm(images.items()):
        # 获取图像信息
        img_w = img_info['width']
        img_h = img_info['height']
        file_name = img_info['file_name']  # "M1306/img_mask/img000001.jpg"

        # 提取序列名和图像名
        # "M1306/img_mask/img000001.jpg" → "M1306_img000001"
        parts = file_name.split('/')
        seq_name = parts[0]  # "M1306"
        img_name = parts[-1].replace('.jpg', '')  # "img000001"

        # 输出标注文件路径
        label_file = labels_dir / f"{seq_name}_{img_name}.txt"

        # 转换该图像的所有标注
        yolo_lines = []
        if img_id in img_annotations:
            for ann in img_annotations[img_id]:
                # COCO bbox: [x_min, y_min, width, height]
                x_min, y_min, bbox_w, bbox_h = ann['bbox']

                # 转换为YOLO格式: [center_x, center_y, width, height] (归一化)
                center_x = (x_min + bbox_w / 2) / img_w
                center_y = (y_min + bbox_h / 2) / img_h
                norm_w = bbox_w / img_w
                norm_h = bbox_h / img_h

                # 映射类别ID
                coco_cat_id = ann['category_id']
                yolo_cat_id = CATEGORY_MAP[coco_cat_id]

                # YOLO格式: class_id cx cy w h
                yolo_line = f"{yolo_cat_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n"
                yolo_lines.append(yolo_line)

        # 写入文件 (即使没有标注也创建空文件)
        with open(label_file, 'w') as f:
            f.writelines(yolo_lines)

    print(f"✅ 转换完成! 标注文件保存到: {labels_dir}")

def organize_images(images_root, output_root):
    """
    重组图像目录结构: M0101/img1/xxx.jpg → images/M0101_xxx.jpg
    """
    images_dir = Path(output_root) / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    sequences = sorted(Path(images_root).glob('M*'))
    print(f"重组 {len(sequences)} 个序列的图像...")

    for seq_dir in tqdm(sequences):
        seq_name = seq_dir.name  # "M0101"
        img1_dir = seq_dir / 'img1'

        if not img1_dir.exists():
            print(f"⚠️ 序列 {seq_name} 没有img1目录,跳过")
            continue

        # 复制所有图像
        for img_path in img1_dir.glob('*.jpg'):
            img_name = img_path.stem  # "img000001"
            new_name = f"{seq_name}_{img_name}.jpg"
            new_path = images_dir / new_name

            # 创建软链接(节省空间) 或 复制文件
            if not new_path.exists():
                new_path.symlink_to(img_path.absolute())  # Windows可能需要管理员权限
                # 或使用: shutil.copy(img_path, new_path)

    print(f"✅ 图像重组完成! 保存到: {images_dir}")

if __name__ == '__main__':
    # 路径配置
    uavdt_root = r'f:\CV\Paper\yoloDepth\yoloDepth\datasets\UAVDT'
    output_root = r'f:\CV\Paper\yoloDepth\yoloDepth\datasets\UAVDT_YOLO'

    # 转换训练集
    print("\n" + "="*60)
    print("转换 UAVDT Train 标注")
    print("="*60)
    convert_coco_to_yolo(
        json_path=f'{uavdt_root}/annotations/UAV-benchmark-M-Train.json',
        images_root=f'{uavdt_root}/images/UAV-benchmark-M',
        output_root=f'{output_root}/train'
    )

    # 重组训练集图像
    organize_images(
        images_root=f'{uavdt_root}/images/UAV-benchmark-M',
        output_root=f'{output_root}/train'
    )

    # 转换验证集
    print("\n" + "="*60)
    print("转换 UAVDT Val 标注")
    print("="*60)
    convert_coco_to_yolo(
        json_path=f'{uavdt_root}/annotations/UAV-benchmark-M-Val.json',
        images_root=f'{uavdt_root}/images/UAV-benchmark-M',
        output_root=f'{output_root}/val'
    )

    # 重组验证集图像
    organize_images(
        images_root=f'{uavdt_root}/images/UAV-benchmark-M',
        output_root=f'{output_root}/val'
    )

    print("\n" + "="*60)
    print("✅ UAVDT 数据集转换完成!")
    print("="*60)
    print(f"输出目录: {output_root}")
    print("目录结构:")
    print("  train/")
    print("    images/  (23,829张)")
    print("    labels/  (23,829个txt)")
    print("  val/")
    print("    images/")
    print("    labels/")
```

#### 1.2 运行转换

```bash
cd f:\CV\Paper\yoloDepth\yoloDepth
python convert_uavdt_to_yolo.py
```

**预计耗时**: 5-10 分钟 (主要是 JSON 解析和文件创建)

---

### Phase 2: 生成 RGB-D 深度图

#### 2.1 使用 ZoeDepth 批量生成

```python
# generate_depths_uavdt.py
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np

# 加载 ZoeDepth 模型
print("加载 ZoeDepth 模型...")
model = torch.hub.load('isl-org/ZoeDepth', 'ZoeD_N', pretrained=True)
model.eval()
model.cuda()

def generate_depth(image_path, output_path):
    """为单张图像生成深度图"""
    # 读取RGB图像
    rgb = Image.open(image_path).convert('RGB')

    # 生成深度图
    with torch.no_grad():
        depth = model.infer_pil(rgb)

    # 归一化到0-255
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255
    depth_uint8 = depth_normalized.astype(np.uint8)

    # 保存为灰度PNG
    depth_img = Image.fromarray(depth_uint8, mode='L')
    depth_img.save(output_path)

def batch_generate_depths(images_dir, output_dir):
    """批量生成深度图"""
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有图像
    image_files = sorted(images_dir.glob('*.jpg'))
    print(f"找到 {len(image_files)} 张图像")

    # 批量处理
    for img_path in tqdm(image_files, desc="生成深度图"):
        depth_path = output_dir / img_path.name.replace('.jpg', '.png')

        if depth_path.exists():
            continue  # 跳过已存在的

        try:
            generate_depth(img_path, depth_path)
        except Exception as e:
            print(f"⚠️ {img_path.name} 生成失败: {e}")

if __name__ == '__main__':
    uavdt_yolo = r'f:\CV\Paper\yoloDepth\yoloDepth\datasets\UAVDT_YOLO'

    # 训练集
    print("\n" + "="*60)
    print("生成 UAVDT Train 深度图")
    print("="*60)
    batch_generate_depths(
        images_dir=f'{uavdt_yolo}/train/images',
        output_dir=f'{uavdt_yolo}/train/depths'
    )

    # 验证集
    print("\n" + "="*60)
    print("生成 UAVDT Val 深度图")
    print("="*60)
    batch_generate_depths(
        images_dir=f'{uavdt_yolo}/val/images',
        output_dir=f'{uavdt_yolo}/val/depths'
    )

    print("\n✅ 深度图生成完成!")
```

#### 2.2 运行深度生成

```bash
cd f:\CV\Paper\yoloDepth\yoloDepth
python generate_depths_uavdt.py
```

**预计耗时**: 4-6 小时 (23,829 张图像 × ~1 秒/张)

---

## 🔧 多数据集联合训练配置

### 方案 1: 简单拼接 (推荐先试)

创建 `data/visdrone_uavdt_joint.yaml`:

```yaml
# VisDrone + UAVDT 联合数据集配置

path: f:/CV/Paper/yoloDepth/yoloDepth/datasets # 数据集根目录

# 训练集: 拼接两个数据集
train:
  - VisDrone/images/train # 6,471张
  - UAVDT_YOLO/train/images # 23,829张
  # 总计: 30,300张

# 验证集: 只用VisDrone (对齐RemDet评估)
val: VisDrone/images/val # 548张

# 深度图路径
train_depth:
  - VisDrone/depths/train
  - UAVDT_YOLO/train/depths

val_depth: VisDrone/depths/val

# 类别数 (使用VisDrone的10类)
nc: 10

# 类别名称
names:
  0: ignored
  1: pedestrian
  2: people
  3: bicycle
  4: car
  5: van
  6: truck
  7: tricycle
  8: awning-tricycle
  9: bus
  10: motor
# 注意: UAVDT只有car(4), truck(6), bus(9)三类
# 其他类别只有VisDrone提供
```

**优点**: 实现简单,Ultralytics 原生支持多路径
**缺点**: 无法控制采样权重 (UAVDT 占 80%,VisDrone 仅 20%)

### 方案 2: 加权采样 (更优,需修改代码)

创建 `data/visdrone_uavdt_weighted.yaml`:

```yaml
path: f:/CV/Paper/yoloDepth/yoloDepth/datasets

# 数据集列表 (带权重)
datasets:
  - name: visdrone
    train: VisDrone/images/train
    val: VisDrone/images/val
    train_depth: VisDrone/depths/train
    val_depth: VisDrone/depths/val
    weight: 1.0 # 100%采样率

  - name: uavdt
    train: UAVDT_YOLO/train/images
    train_depth: UAVDT_YOLO/train/depths
    weight: 0.5 # 50%采样率 (相对VisDrone)

# 验证只用VisDrone
val_dataset: visdrone

nc: 10
names:
  [
    ignored,
    pedestrian,
    people,
    bicycle,
    car,
    van,
    truck,
    tricycle,
    awning-tricycle,
    bus,
    motor,
  ]
```

**采样策略**:

```python
# 每个epoch的batch分布 (伪代码)
visdrone_samples = 6471 * 1.0 = 6471
uavdt_samples = 23829 * 0.5 = 11915
total_samples = 18386

每个batch (16张图):
  - VisDrone: 6张 (35%)
  - UAVDT: 10张 (65%)
```

**实现**: 需要修改 `ultralytics/data/dataset.py` 添加 `WeightedMultiDataset` 类

---

## 📝 TODO 更新

基于 UAVDT 数据集的实际情况,更新待办事项:

### ✅ 已完成

- [x] 确认 UAVDT 数据集存在 (yoloDepth/datasets/UAVDT)
- [x] 分析 UAVDT 结构 (COCO 格式, 23,829 张, 3 类, 422K 标注)
- [x] 理解尺寸分布互补性 (UAVDT 提供 48% Medium, 1.4% Large)

### ⏳ 待执行 (优先级 1)

1. **转换 UAVDT 标注** (预计 10 分钟)
   - 运行 `convert_uavdt_to_yolo.py`
   - 输出: UAVDT_YOLO/train/{images,labels}/
2. **生成 UAVDT 深度图** (预计 4-6 小时)
   - 运行 `generate_depths_uavdt.py`
   - 输出: UAVDT_YOLO/train/depths/
3. **创建联合数据集配置**
   - 方案 1: `visdrone_uavdt_joint.yaml` (简单拼接)
   - 方案 2: `visdrone_uavdt_weighted.yaml` (加权采样)
4. **启动联合训练**
   ```bash
   CUDA_VISIBLE_DEVICES=7 python train_depth.py \
       --data data/visdrone_uavdt_joint.yaml \
       --epochs 300 \
       --batch 16 \
       --imgsz 640 \
       --device 0 \
       --project runs/train \
       --name exp_joint_visdrone_uavdt_v1 \
       --weights yolo12n.pt \
       --save_period 50
   ```

### 🎯 预期结果

| 指标        | Baseline (VisDrone only) | 联合训练目标 | RemDet-Tiny | 胜率  |
| ----------- | ------------------------ | ------------ | ----------- | ----- |
| Overall mAP | 41%                      | **45-47%**   | 38.9%       | +15%  |
| Small mAP   | 30.94%                   | **35-38%**   | 12.7%       | +180% |
| Medium mAP  | 46.24%                   | **48-50%**   | 33.0%       | +50%  |
| Large mAP   | 36.70%                   | **42-45%**   | 44.5%       | -2%   |

**关键目标**: 在 Small 和 Medium 上全面碾压 RemDet, Large 追平即可!

---

## 🚀 下一步行动

**请确认以下信息,然后我们开始执行**:

1. ✅ UAVDT 数据集路径确认: `yoloDepth\datasets\UAVDT`
2. ❓ 服务器上是否已安装 ZoeDepth? (需要用来生成深度图)
3. ❓ 是否有足够的存储空间? (深度图需要~20GB)
4. ❓ 训练环境的 GPU 显存? (batch size 可能需要调整)

**我现在立即创建转换脚本,你确认后就可以运行!** 🎯
