#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断predictions.json和GT JSON的image_id匹配问题
"""
import json
from pathlib import Path

# 读取predictions.json
pred_path = Path("runs/val/visdrone_coco_eval_n5/predictions.json")
with open(pred_path, 'r') as f:
    pred_data = json.load(f)

# 提取所有唯一的image_id
unique_pred_ids = set()
for pred in pred_data:
    unique_pred_ids.add(pred['image_id'])

print(f"📊 predictions.json stats:")
print(f"   Total detections: {len(pred_data)}")
print(f"   Unique image_ids: {len(unique_pred_ids)}")
print(f"\n🔍 Sample prediction image_ids:")
for i, img_id in enumerate(list(unique_pred_ids)[:10]):
    print(f"   [{i}] {img_id}")

# 告诉用户如何在远程服务器检查GT JSON
print(f"\n" + "="*80)
print("📝 请在远程服务器执行以下命令,检查GT JSON:")
print("="*80)
print("""
cd /data2/user/2024/lzy/yolo12-bimodal/yoloDepth
python << 'EOF'
import json
from pathlib import Path

gt_path = '/data2/user/2024/lzy/Datasets/VisDrone2019-DET-COCO/annotations/VisDrone2019-DET_val_coco.json'
with open(gt_path, 'r') as f:
    gt_data = json.load(f)

print(f"GT JSON stats:")
print(f"   Total images: {len(gt_data['images'])}")

print("\\nSample GT file_names:")
for i, img in enumerate(gt_data['images'][:10]):
    print(f"   id={img['id']}, file_name={img['file_name']}")

# 检查是否包含 _d_ 标记
has_d_marker = sum(1 for img in gt_data['images'] if '_d_' in img['file_name'])
print(f"\\nFile names with '_d_' marker: {has_d_marker}/{len(gt_data['images'])}")

# 尝试匹配第一个预测
pred_samples = ['0000256_02173_d_0000030.jpg', '0000249_02468_d_0000008.jpg', '0000364_01765_d_0000782.jpg']
print("\\nMatching test:")
for pred_name in pred_samples:
    # 直接匹配
    direct = [img for img in gt_data['images'] if img['file_name'] == pred_name]
    if direct:
        print(f"   ✅ {pred_name} → id={direct[0]['id']}")
    else:
        # 尝试不带 _d_ 的匹配
        no_d_name = pred_name.replace('_d_', '_')
        indirect = [img for img in gt_data['images'] if img['file_name'] == no_d_name]
        if indirect:
            print(f"   🔄 {pred_name} → (remove _d_) → {no_d_name} → id={indirect[0]['id']}")
        else:
            print(f"   ❌ {pred_name} NOT FOUND (even without _d_)")
EOF
""")

print("\n💡 根据输出结果,我们可以判断:")
print("   1. GT JSON中的file_name是否包含 '_d_' 标记")
print("   2. 如果不包含,需要修改Step 2的匹配逻辑")
print("   3. 具体的修改策略(去除_d_或其他)")
