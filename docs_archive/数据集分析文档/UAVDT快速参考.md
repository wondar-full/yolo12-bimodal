# UAVDT 目录结构快速参考

## 📁 目录结构 (以 M0101 为例)

```
M0101/
├── img1/              ← ✅ 原始图像 (407张) - 训练用这个!
├── img_mask/          ← ❌ 遮罩图像 (鲁棒性测试用,训练不用)
├── gt/
│   ├── gt.txt         ← ❌ MOT格式标注 (不推荐)
│   ├── gt_ignore.txt  ← ❌ 忽略区域 (COCO JSON已过滤)
│   └── gt_whole.txt   ← ❌ 含遮挡信息 (研究用,训练不需要)
└── det/               ← 空目录 (保存预测结果用)
```

## ⚡ 快速决策

**我们应该使用**:

- ✅ **images**: `img1/` (原始图像)
- ✅ **annotations**: `annotations/UAV-benchmark-M-Train.json` (COCO 格式)
- ✅ **转换脚本**: `convert_uavdt_to_yolo.py` (无需修改!)

**不要使用**:

- ❌ `img_mask/` - 遮罩图像
- ❌ `gt/gt.txt` - MOT 格式
- ❌ `gt/gt_ignore.txt` - 忽略区域

## 🎯 三个 gt 文件的区别

| 文件          | 用途                 | occlusion 字段 | 行数  | 是否使用     |
| ------------- | -------------------- | -------------- | ----- | ------------ |
| gt.txt        | 标准检测标注         | -1 (未标注)    | 5,415 | ❌           |
| gt_ignore.txt | 应该忽略的区域       | -1             | 671   | ❌           |
| gt_whole.txt  | 含遮挡级别标注       | 0/1/2/3        | 5,415 | ❌           |
| **COCO JSON** | **检测标注(已过滤)** | -              | 422K  | ✅ **推荐!** |

**为什么用 COCO JSON?**

1. ✅ 已经过滤了 ignore 区域 (422K vs 470K+)
2. ✅ JSON 结构化,解析简单
3. ✅ 图像路径统一管理
4. ✅ 类别映射清晰

## 🖼️ img1 vs img_mask

| 特性      | img1          | img_mask             |
| --------- | ------------- | -------------------- |
| 内容      | 原始图像      | 人为添加遮罩         |
| 用途      | 训练/正常测试 | 遮挡鲁棒性测试       |
| RemDet 用 | ✅ 主要       | ❓ 可能未用          |
| 我们用    | ✅ 训练       | ❌ 暂不用 (后期可选) |

## 📝 MOT 格式说明

**格式**: `frame_id, object_id, x, y, w, h, score, class, occlusion`

**示例**:

```
1,1,141,147,106,45,1,1,-1
↓
帧1, 目标1, bbox=[141,147,106,45], 置信度1, 类别1(car), 遮挡-1
```

**为什么不用 MOT 格式?**

- MOT 用于追踪任务,我们只做检测
- 需要手动处理 ignore 区域
- COCO JSON 更方便

## ✅ 结论

**转换脚本无需修改!** 直接运行:

```bash
cd f:\CV\Paper\yoloDepth\yoloDepth
python convert_uavdt_to_yolo.py
```

脚本会:

1. ✅ 读取 COCO JSON (已过滤 ignore)
2. ✅ 只从 img1/提取图像
3. ✅ 转换为 YOLO 格式
4. ✅ 映射类别 (car→4, truck→6, bus→9)

**img_mask 和 gt_whole.txt 留作后用**:

- 论文需要报告"遮挡鲁棒性"时再用
- 训练阶段不需要!
