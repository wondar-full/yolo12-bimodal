"""
验证VisDrone+UAVDT联合数据集配置
检查路径、文件数量、RGB-Depth配对是否正确

运行: python verify_joint_dataset.py
"""

import sys
from pathlib import Path
from ultralytics import YOLO
from ultralytics.data import YOLORGBDDataset
from ultralytics.utils import yaml_load
import numpy as np

def verify_yaml_config(yaml_path):
    """验证YAML配置文件"""
    print("\n" + "="*60)
    print("Step 1: 验证YAML配置")
    print("="*60)
    
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        print(f"❌ YAML文件不存在: {yaml_path}")
        return False
    
    data = yaml_load(yaml_path)
    
    # 检查必需字段
    required_fields = ['path', 'train', 'val', 'train_depth', 'val_depth', 'nc', 'names']
    for field in required_fields:
        if field not in data:
            print(f"❌ 缺少必需字段: {field}")
            return False
        print(f"✅ {field}: {data[field] if field not in ['names'] else '...'}")
    
    # 检查路径格式
    root = Path(data['path'])
    print(f"\n数据集根目录: {root}")
    
    if not root.exists():
        print(f"❌ 根目录不存在: {root}")
        return False
    
    # 检查训练集路径
    train_paths = data['train'] if isinstance(data['train'], list) else [data['train']]
    train_depth_paths = data['train_depth'] if isinstance(data['train_depth'], list) else [data['train_depth']]
    
    print(f"\n训练集数量: {len(train_paths)}")
    for i, (rgb_path, depth_path) in enumerate(zip(train_paths, train_depth_paths)):
        full_rgb = root / rgb_path
        full_depth = root / depth_path
        
        print(f"\n数据集 {i+1}:")
        print(f"  RGB:   {rgb_path}")
        print(f"         存在: {'✅' if full_rgb.exists() else '❌'}")
        
        print(f"  Depth: {depth_path}")
        print(f"         存在: {'✅' if full_depth.exists() else '❌'}")
        
        if full_rgb.exists():
            rgb_files = list(full_rgb.glob('*.jpg')) + list(full_rgb.glob('*.png'))
            print(f"         图像数: {len(rgb_files)}")
        
        if full_depth.exists():
            depth_files = list(full_depth.glob('*.png')) + list(full_depth.glob('*.jpg'))
            print(f"         深度图数: {len(depth_files)}")
    
    return True

def verify_dataset_loading(yaml_path):
    """验证数据集加载"""
    print("\n" + "="*60)
    print("Step 2: 验证数据集加载")
    print("="*60)
    
    try:
        # 加载训练集
        dataset = YOLORGBDDataset(
            img_path='train',
            data=yaml_path,
            augment=False,
            batch_size=1
        )
        
        print(f"✅ 数据集加载成功!")
        print(f"   总图像数: {len(dataset.im_files)}")
        
        # 统计各数据集数量
        visdrone_count = sum(1 for p in dataset.im_files if 'VisDrone' in p)
        uavdt_count = sum(1 for p in dataset.im_files if 'UAVDT' in p)
        
        print(f"   VisDrone: {visdrone_count}")
        print(f"   UAVDT: {uavdt_count}")
        print(f"   期望: VisDrone ~6,471, UAVDT ~23,829")
        
        # 检查深度图
        if dataset.depth_files:
            print(f"\n✅ 深度图已启用")
            print(f"   深度图数: {len(dataset.depth_files)}")
            
            # 检查配对
            if len(dataset.im_files) == len(dataset.depth_files):
                print(f"✅ RGB-Depth完全配对")
            else:
                print(f"⚠️ 配对不完整: {len(dataset.im_files)} RGB, {len(dataset.depth_files)} Depth")
        else:
            print(f"❌ 深度图未加载! 检查路径是否正确")
            return False
        
        return dataset
    
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def verify_sample_loading(dataset):
    """验证样本加载"""
    print("\n" + "="*60)
    print("Step 3: 验证样本加载")
    print("="*60)
    
    try:
        # 加载第一张VisDrone图像
        visdrone_idx = next(i for i, p in enumerate(dataset.im_files) if 'VisDrone' in p)
        print(f"\n测试VisDrone样本 (index {visdrone_idx}):")
        print(f"  RGB路径: {dataset.im_files[visdrone_idx]}")
        print(f"  Depth路径: {dataset.depth_files[visdrone_idx] if dataset.depth_files else 'None'}")
        
        img, _, _ = dataset.load_image(visdrone_idx)
        print(f"  图像形状: {img.shape}")
        
        if img.shape[2] == 4:
            print(f"  ✅ RGB-D加载成功 (4通道)")
            print(f"     RGB范围: {img[:,:,:3].min():.2f} - {img[:,:,:3].max():.2f}")
            print(f"     Depth范围: {img[:,:,3].min():.2f} - {img[:,:,3].max():.2f}")
        else:
            print(f"  ❌ 只有RGB通道 ({img.shape[2]}通道)")
            return False
        
        # 加载第一张UAVDT图像
        uavdt_idx = next((i for i, p in enumerate(dataset.im_files) if 'UAVDT' in p), None)
        if uavdt_idx:
            print(f"\n测试UAVDT样本 (index {uavdt_idx}):")
            print(f"  RGB路径: {dataset.im_files[uavdt_idx]}")
            print(f"  Depth路径: {dataset.depth_files[uavdt_idx] if dataset.depth_files else 'None'}")
            
            img, _, _ = dataset.load_image(uavdt_idx)
            print(f"  图像形状: {img.shape}")
            
            if img.shape[2] == 4:
                print(f"  ✅ RGB-D加载成功 (4通道)")
            else:
                print(f"  ❌ 只有RGB通道 ({img.shape[2]}通道)")
                return False
        
        return True
    
    except Exception as e:
        print(f"❌ 样本加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_labels(dataset):
    """验证标签加载"""
    print("\n" + "="*60)
    print("Step 4: 验证标签")
    print("="*60)
    
    try:
        # 检查几个样本的标签
        sample_indices = [0, len(dataset)//2, -1]
        
        for idx in sample_indices:
            if idx < 0:
                idx = len(dataset) + idx
            
            img_path = dataset.im_files[idx]
            label = dataset.get_labels()[idx] if hasattr(dataset, 'get_labels') else None
            
            dataset_name = "VisDrone" if "VisDrone" in img_path else "UAVDT"
            print(f"\n样本 {idx} ({dataset_name}):")
            print(f"  路径: {Path(img_path).name}")
            
            if label is not None and 'cls' in label:
                classes = label['cls'].astype(int)
                print(f"  标签数: {len(classes)}")
                print(f"  类别: {np.unique(classes).tolist()}")
            else:
                print(f"  ⚠️ 无标签或标签格式异常")
        
        return True
    
    except Exception as e:
        print(f"❌ 标签验证失败: {e}")
        return False

def main():
    yaml_path = 'data/visdrone_uavdt_joint.yaml'
    
    print("\n" + "="*60)
    print("VisDrone+UAVDT联合数据集配置验证")
    print("="*60)
    
    # Step 1: 验证YAML
    if not verify_yaml_config(yaml_path):
        print("\n❌ YAML配置验证失败!")
        return False
    
    # Step 2: 加载数据集
    dataset = verify_dataset_loading(yaml_path)
    if dataset is None:
        print("\n❌ 数据集加载失败!")
        return False
    
    # Step 3: 验证样本加载
    if not verify_sample_loading(dataset):
        print("\n❌ 样本加载验证失败!")
        return False
    
    # Step 4: 验证标签
    if not verify_labels(dataset):
        print("\n❌ 标签验证失败!")
        return False
    
    # 总结
    print("\n" + "="*60)
    print("✅ 全部验证通过!")
    print("="*60)
    print(f"总图像数: {len(dataset.im_files)}")
    print(f"深度图数: {len(dataset.depth_files) if dataset.depth_files else 0}")
    print(f"\n可以开始训练:")
    print(f"  CUDA_VISIBLE_DEVICES=7 python train_depth.py \\")
    print(f"      --data {yaml_path} \\")
    print(f"      --epochs 300 \\")
    print(f"      --batch 16 \\")
    print(f"      --name exp_joint_v1 \\")
    print(f"      --weights yolo12n.pt")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
