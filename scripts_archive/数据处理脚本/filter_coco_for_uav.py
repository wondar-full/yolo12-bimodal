"""
过滤COCO数据集,只保留UAV相关类别
将COCO的80类过滤为6类(person/bicycle/car/motorcycle/bus/truck)
并转换为YOLO格式,映射到VisDrone类别ID

Author: AI Assistant
Date: 2025-11-01
"""

import json
import shutil
from pathlib import Path
from tqdm import tqdm

# COCO类别 → VisDrone类别映射
COCO_TO_VISDRONE = {
    1: 1,   # person → pedestrian (VisDrone ID 1)
    2: 3,   # bicycle → bicycle (VisDrone ID 3)
    3: 4,   # car → car (VisDrone ID 4)
    4: 10,  # motorcycle → motor (VisDrone ID 10)
    6: 9,   # bus → bus (VisDrone ID 9)
    8: 6,   # truck → truck (VisDrone ID 6)
}

# 可选: 也映射到van
# 7: 5,  # train → 跳过
# 可以考虑将truck同时映射到van

def filter_coco_annotations(json_path, output_path, keep_categories):
    """
    过滤COCO JSON,只保留指定类别
    
    Args:
        json_path: 原始COCO JSON路径
        output_path: 过滤后的JSON路径
        keep_categories: 要保留的类别ID列表
    """
    print(f"加载 {json_path}...")
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    # 1. 过滤categories
    print("过滤类别...")
    filtered_categories = [
        cat for cat in coco_data['categories'] 
        if cat['id'] in keep_categories
    ]
    
    # 2. 过滤annotations
    print("过滤标注...")
    filtered_annotations = []
    filtered_image_ids = set()
    
    for ann in tqdm(coco_data['annotations'], desc="处理标注"):
        if ann['category_id'] in keep_categories:
            filtered_annotations.append(ann)
            filtered_image_ids.add(ann['image_id'])
    
    # 3. 过滤images (只保留有相关标注的图像)
    print("过滤图像...")
    filtered_images = [
        img for img in coco_data['images']
        if img['id'] in filtered_image_ids
    ]
    
    # 4. 保存过滤后的JSON
    filtered_data = {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': filtered_categories,
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', [])
    }
    
    print(f"保存到 {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(filtered_data, f)
    
    print(f"✅ 过滤完成!")
    print(f"   原始图像: {len(coco_data['images']):,} → 过滤后: {len(filtered_images):,}")
    print(f"   原始标注: {len(coco_data['annotations']):,} → 过滤后: {len(filtered_annotations):,}")
    
    return filtered_data

def convert_coco_to_yolo(coco_data, images_dir, output_dir):
    """
    将COCO JSON转换为YOLO格式
    
    Args:
        coco_data: COCO JSON数据
        images_dir: COCO图像目录
        output_dir: 输出YOLO目录
    """
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    
    labels_dir = output_dir / 'labels'
    images_out_dir = output_dir / 'images'
    
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_out_dir.mkdir(parents=True, exist_ok=True)
    
    # 建立 image_id → image_info 映射
    images = {img['id']: img for img in coco_data['images']}
    
    # 按图像ID分组标注
    print("按图像分组标注...")
    img_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)
    
    print(f"转换为YOLO格式...")
    converted_count = 0
    
    for img_id, img_info in tqdm(images.items(), desc="转换进度"):
        file_name = img_info['file_name']
        img_w = img_info['width']
        img_h = img_info['height']
        
        # 源图像路径
        src_img_path = images_dir / file_name
        if not src_img_path.exists():
            continue
        
        # 目标路径
        base_name = Path(file_name).stem
        dst_img_path = images_out_dir / f"{base_name}.jpg"
        label_file = labels_dir / f"{base_name}.txt"
        
        # 创建软链接或复制图像
        if not dst_img_path.exists():
            try:
                dst_img_path.symlink_to(src_img_path.absolute())
            except:
                shutil.copy(src_img_path, dst_img_path)
        
        # 转换标注
        yolo_lines = []
        if img_id in img_annotations:
            for ann in img_annotations[img_id]:
                # COCO bbox: [x_min, y_min, width, height]
                x_min, y_min, bbox_w, bbox_h = ann['bbox']
                
                # 过滤无效bbox
                if bbox_w <= 0 or bbox_h <= 0:
                    continue
                
                # 转换为YOLO格式
                center_x = (x_min + bbox_w / 2) / img_w
                center_y = (y_min + bbox_h / 2) / img_h
                norm_w = bbox_w / img_w
                norm_h = bbox_h / img_h
                
                # 裁剪到[0,1]
                center_x = max(0, min(1, center_x))
                center_y = max(0, min(1, center_y))
                norm_w = max(0, min(1, norm_w))
                norm_h = max(0, min(1, norm_h))
                
                # 映射类别ID
                coco_cat_id = ann['category_id']
                yolo_cat_id = COCO_TO_VISDRONE[coco_cat_id]
                
                # YOLO格式
                yolo_line = f"{yolo_cat_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n"
                yolo_lines.append(yolo_line)
        
        # 写入标注文件
        if len(yolo_lines) > 0:
            with open(label_file, 'w') as f:
                f.writelines(yolo_lines)
            converted_count += 1
    
    print(f"✅ 转换完成!")
    print(f"   转换图像: {converted_count:,}")
    print(f"   输出目录: {output_dir}")

def main():
    """主函数"""
    coco_root = Path(r'f:\CV\Paper\yoloDepth\yoloDepth\datasets\coco2014')
    output_root = Path(r'f:\CV\Paper\yoloDepth\yoloDepth\datasets\COCO_UAV_YOLO')
    
    print("\n" + "="*60)
    print("COCO 数据集过滤与转换工具")
    print("="*60)
    print(f"源目录: {coco_root}")
    print(f"输出目录: {output_root}")
    
    # 检查源目录
    if not coco_root.exists():
        print(f"❌ 错误: COCO数据集不存在: {coco_root}")
        return
    
    # 要保留的类别
    keep_categories = list(COCO_TO_VISDRONE.keys())
    print(f"\n保留类别: {keep_categories}")
    print("类别映射:")
    coco_names = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 6: 'bus', 8: 'truck'}
    vd_names = {1: 'pedestrian', 3: 'bicycle', 4: 'car', 10: 'motor', 9: 'bus', 6: 'truck'}
    for coco_id, vd_id in COCO_TO_VISDRONE.items():
        print(f"  COCO {coco_id:2d} ({coco_names[coco_id]:<12}) → VisDrone {vd_id:2d} ({vd_names[vd_id]})")
    
    # 处理训练集
    train_json = coco_root / 'annotations' / 'instances_train2014.json'
    if train_json.exists():
        print("\n" + "="*60)
        print("处理训练集")
        print("="*60)
        
        # 1. 过滤JSON
        filtered_json = output_root / 'annotations' / 'instances_train_uav.json'
        filtered_json.parent.mkdir(parents=True, exist_ok=True)
        
        filtered_data = filter_coco_annotations(
            json_path=train_json,
            output_path=filtered_json,
            keep_categories=keep_categories
        )
        
        # 2. 转换为YOLO格式
        convert_coco_to_yolo(
            coco_data=filtered_data,
            images_dir=coco_root / 'train2014',
            output_dir=output_root / 'train'
        )
    else:
        print(f"⚠️ 训练集JSON不存在: {train_json}")
    
    # 总结
    print("\n" + "="*60)
    print("✅ COCO 过滤与转换完成!")
    print("="*60)
    print(f"输出目录:")
    print(f"  {output_root}/")
    print(f"  ├── train/")
    print(f"  │   ├── images/  (过滤后的图像)")
    print(f"  │   └── labels/  (YOLO格式标注)")
    print(f"  └── annotations/")
    print(f"      └── instances_train_uav.json  (过滤后的COCO JSON)")
    
    print(f"\n下一步:")
    print(f"1. 生成深度图: python generate_depths_coco.py")
    print(f"2. 创建三数据集配置: data/visdrone_uavdt_coco_joint.yaml")
    print(f"3. 启动联合训练")
    
    print(f"\n⚠️  注意:")
    print(f"   - COCO域差异较大,建议先不用")
    print(f"   - 优先完成 VisDrone + UAVDT 训练")
    print(f"   - 如果性能达标, COCO不是必需的")

if __name__ == '__main__':
    main()
