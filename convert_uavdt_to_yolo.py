"""
UAVDT数据集转换脚本: COCO JSON → YOLO TXT
将UAVDT的COCO格式标注转换为YOLO格式,并重组图像目录结构

Author: AI Assistant
Date: 2025-10-31
"""

import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# 类别映射: UAVDT (0-2) → VisDrone (0-9)
CATEGORY_MAP = {
    0: 4,  # car → car (VisDrone ID 4)
    1: 6,  # truck → truck (VisDrone ID 6)
    2: 9   # bus → bus (VisDrone ID 9)
}

def convert_coco_to_yolo(json_path, images_root, output_root, split='train'):
    """
    将UAVDT的COCO JSON转换为YOLO格式
    
    Args:
        json_path: COCO JSON文件路径
        images_root: 图像根目录 (UAV-benchmark-M/)
        output_root: 输出根目录
        split: 'train' 或 'val'
    """
    print(f"\n{'='*60}")
    print(f"转换 UAVDT {split.upper()} 数据集")
    print(f"{'='*60}")
    
    print(f"[1/4] 加载 {json_path}...")
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    images = {img['id']: img for img in coco_data['images']}
    annotations = coco_data['annotations']
    
    print(f"      - 图像数: {len(images)}")
    print(f"      - 标注数: {len(annotations)}")
    print(f"      - 类别数: {len(coco_data['categories'])}")
    
    # 创建输出目录
    output_path = Path(output_root) / split
    labels_dir = output_path / 'labels/rgb'
    images_dir = output_path / 'images/rgb'
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # 按图像ID分组标注
    print(f"[2/4] 按图像分组标注...")
    img_annotations = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)
    
    # 转换标注并复制图像
    print(f"[3/4] 转换标注格式...")
    converted_count = 0
    empty_count = 0
    
    for img_id, img_info in tqdm(images.items(), desc="      转换进度"):
        # 获取图像信息
        img_w = img_info['width']
        img_h = img_info['height']
        file_name = img_info['file_name']  # "M1306/img_mask/img000001.jpg" 或 "M1306/img1/img000001.jpg"
        
        # 提取序列名和图像名
        parts = file_name.split('/')
        seq_name = parts[0]  # "M1306"
        img_name = parts[-1].replace('.jpg', '')  # "img000001"
        
        # 新文件名: M1306_img000001
        new_base_name = f"{seq_name}_{img_name}"
        
        # 源图像路径 (尝试img1和img_mask两个可能的目录)
        src_img_path1 = Path(images_root) / seq_name / 'img1' / parts[-1]
        src_img_path2 = Path(images_root) / seq_name / 'img_mask' / parts[-1]
        
        if src_img_path1.exists():
            src_img_path = src_img_path1
        elif src_img_path2.exists():
            src_img_path = src_img_path2
        else:
            print(f"⚠️ 图像不存在: {seq_name}/{parts[-1]}")
            continue
        
        # 目标路径
        dst_img_path = images_dir / f"{new_base_name}.jpg"
        label_file = labels_dir / f"{new_base_name}.txt"
        
        # 复制图像 (如果还没复制过)
        if not dst_img_path.exists():
            shutil.copy(src_img_path, dst_img_path)
        
        # 转换该图像的所有标注
        yolo_lines = []
        if img_id in img_annotations:
            for ann in img_annotations[img_id]:
                # COCO bbox: [x_min, y_min, width, height]
                x_min, y_min, bbox_w, bbox_h = ann['bbox']
                
                # 过滤无效bbox
                if bbox_w <= 0 or bbox_h <= 0:
                    continue
                
                # 转换为YOLO格式: [center_x, center_y, width, height] (归一化)
                center_x = (x_min + bbox_w / 2) / img_w
                center_y = (y_min + bbox_h / 2) / img_h
                norm_w = bbox_w / img_w
                norm_h = bbox_h / img_h
                
                # 确保坐标在[0,1]范围内
                center_x = max(0, min(1, center_x))
                center_y = max(0, min(1, center_y))
                norm_w = max(0, min(1, norm_w))
                norm_h = max(0, min(1, norm_h))
                
                # 映射类别ID
                coco_cat_id = ann['category_id']
                yolo_cat_id = CATEGORY_MAP[coco_cat_id]
                
                # YOLO格式: class_id cx cy w h
                yolo_line = f"{yolo_cat_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n"
                yolo_lines.append(yolo_line)
        
        # 写入标注文件
        with open(label_file, 'w') as f:
            f.writelines(yolo_lines)
        
        if len(yolo_lines) > 0:
            converted_count += 1
        else:
            empty_count += 1
    
    print(f"[4/4] 转换完成!")
    print(f"      ✅ 有效图像: {converted_count}")
    print(f"      ⚠️  空标注: {empty_count}")
    print(f"      📁 图像目录: {images_dir}")
    print(f"      📁 标注目录: {labels_dir}")
    
    return converted_count

def main():
    """主函数"""
    # 路径配置
    uavdt_root = Path(r'/data2/user/2024/lzy/Datasets/UAVDT')
    output_root = Path(r'/data2/user/2024/lzy/Datasets/UAVDT_YOLO')
    images_root = uavdt_root / 'images' / 'UAV-benchmark-M'
    
    print("\n" + "="*60)
    print("UAVDT 数据集转换工具")
    print("="*60)
    print(f"源目录: {uavdt_root}")
    print(f"输出目录: {output_root}")
    
    # 检查源目录
    if not uavdt_root.exists():
        print(f"❌ 错误: UAVDT数据集不存在: {uavdt_root}")
        return
    
    if not images_root.exists():
        print(f"❌ 错误: 图像目录不存在: {images_root}")
        return
    
    # 转换训练集
    train_json = uavdt_root / 'annotations' / 'UAV-benchmark-M-Train.json'
    if train_json.exists():
        train_count = convert_coco_to_yolo(
            json_path=train_json,
            images_root=images_root,
            output_root=output_root,
            split='train'
        )
    else:
        print(f"⚠️ 训练集JSON不存在: {train_json}")
        train_count = 0
    
    # 转换验证集
    val_json = uavdt_root / 'annotations' / 'UAV-benchmark-M-Val.json'
    if val_json.exists():
        val_count = convert_coco_to_yolo(
            json_path=val_json,
            images_root=images_root,
            output_root=output_root,
            split='val'
        )
    else:
        print(f"⚠️ 验证集JSON不存在: {val_json}")
        val_count = 0
    
    # 总结
    print("\n" + "="*60)
    print("✅ UAVDT 数据集转换完成!")
    print("="*60)
    print(f"训练集: {train_count} 张图像")
    print(f"验证集: {val_count} 张图像")
    print(f"\n输出目录结构:")
    print(f"{output_root}/")
    print(f"├── train/")
    print(f"│   ├── images/  (*.jpg)")
    print(f"│   └── labels/  (*.txt)")
    print(f"└── val/")
    print(f"    ├── images/")
    print(f"    └── labels/")
    
    print(f"\n下一步:")
    print(f"1. 检查输出目录: {output_root}")
    print(f"2. 生成深度图: python generate_depths_uavdt.py")
    print(f"3. 创建联合数据集配置: data/visdrone_uavdt_joint.yaml")

if __name__ == '__main__':
    main()
