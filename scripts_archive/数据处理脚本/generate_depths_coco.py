"""
为COCO数据集生成深度图
注意: 仅在确定需要COCO参与训练时才运行此脚本!

Author: AI Assistant
Date: 2025-11-01
"""

import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
import sys

def load_zoedepth_model():
    """加载ZoeDepth模型"""
    print("正在加载 ZoeDepth 模型...")
    try:
        model = torch.hub.load('isl-org/ZoeDepth', 'ZoeD_N', pretrained=True)
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
            print(f"✅ 模型加载成功! 使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("✅ 模型加载成功! 使用CPU (速度较慢)")
        
        return model
    
    except Exception as e:
        print(f"❌ ZoeDepth加载失败: {e}")
        print("\n请先安装ZoeDepth:")
        print("  pip install timm")
        sys.exit(1)

def generate_depth(model, image_path, output_path, device='cuda'):
    """为单张图像生成深度图"""
    try:
        rgb = Image.open(image_path).convert('RGB')
        
        with torch.no_grad():
            depth = model.infer_pil(rgb)
        
        # 归一化到0-255
        depth_min = depth.min()
        depth_max = depth.max()
        
        if depth_max - depth_min > 0:
            depth_normalized = (depth - depth_min) / (depth_max - depth_min) * 255
        else:
            depth_normalized = np.zeros_like(depth)
        
        depth_uint8 = depth_normalized.astype(np.uint8)
        
        # 保存为灰度PNG
        depth_img = Image.fromarray(depth_uint8, mode='L')
        depth_img.save(output_path)
        
        return True
    
    except Exception as e:
        print(f"⚠️ 生成失败: {image_path.name} - {e}")
        return False

def batch_generate_depths(model, images_dir, output_dir, device='cuda'):
    """批量生成深度图"""
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像
    image_files = sorted(images_dir.glob('*.jpg'))
    print(f"找到 {len(image_files)} 张图像")
    
    if len(image_files) == 0:
        print(f"⚠️ 警告: {images_dir} 中没有找到图像文件!")
        return 0
    
    # 统计
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    # 批量处理
    for img_path in tqdm(image_files, desc="生成深度图"):
        depth_path = output_dir / img_path.name.replace('.jpg', '.png')
        
        # 跳过已存在的
        if depth_path.exists():
            skip_count += 1
            continue
        
        # 生成深度图
        if generate_depth(model, img_path, depth_path, device):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n统计:")
    print(f"  ✅ 成功: {success_count}")
    print(f"  ⏭️  跳过: {skip_count}")
    print(f"  ❌ 失败: {fail_count}")
    
    return success_count

def main():
    """主函数"""
    
    print("\n" + "="*60)
    print("⚠️  COCO 深度图生成工具 - 重要提示!")
    print("="*60)
    print("\n在运行此脚本之前,请确认:")
    print("1. ✅ 已经完成 VisDrone + UAVDT 训练")
    print("2. ✅ 性能不够,确定需要COCO辅助")
    print("3. ✅ 已经运行 filter_coco_for_uav.py 过滤COCO")
    print("4. ✅ 服务器有足够存储空间 (~30-40GB)")
    print("5. ✅ 愿意等待 10-15 小时生成深度图")
    
    print("\n如果以上条件不满足,建议:")
    print("❌ 不要运行此脚本!")
    print("✅ 先完成 VisDrone + UAVDT 训练")
    print("✅ 评估性能后再决定是否需要COCO")
    
    # 确认
    response = input("\n确定要继续吗? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("❌ 已取消。建议先完成 VisDrone + UAVDT 训练。")
        return
    
    # 路径配置
    coco_uav_yolo = Path(r'f:\CV\Paper\yoloDepth\yoloDepth\datasets\COCO_UAV_YOLO')
    
    print("\n" + "="*60)
    print("COCO 深度图生成")
    print("="*60)
    print(f"数据集路径: {coco_uav_yolo}")
    
    # 检查目录
    if not coco_uav_yolo.exists():
        print(f"❌ 错误: COCO_UAV_YOLO目录不存在: {coco_uav_yolo}")
        print("请先运行: python filter_coco_for_uav.py")
        return
    
    # 检查训练集图像
    train_images = coco_uav_yolo / 'train' / 'images'
    if not train_images.exists() or len(list(train_images.glob('*.jpg'))) == 0:
        print(f"❌ 错误: 训练集图像不存在: {train_images}")
        print("请先运行: python filter_coco_for_uav.py")
        return
    
    # 加载模型
    model = load_zoedepth_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 生成训练集深度图
    print("\n" + "="*60)
    print("生成 COCO Train 深度图")
    print("="*60)
    print("⚠️  预计耗时: 10-15 小时 (根据过滤后的图像数量)")
    
    train_count = batch_generate_depths(
        model=model,
        images_dir=train_images,
        output_dir=coco_uav_yolo / 'train' / 'depths',
        device=device
    )
    
    # 总结
    print("\n" + "="*60)
    print("✅ 深度图生成完成!")
    print("="*60)
    print(f"训练集: {train_count} 张")
    print(f"\n输出目录:")
    print(f"{coco_uav_yolo}/")
    print(f"├── train/")
    print(f"│   ├── images/  (RGB)")
    print(f"│   ├── labels/  (YOLO)")
    print(f"│   └── depths/  (Depth) ← 新生成")
    
    print(f"\n下一步:")
    print(f"1. 创建三数据集配置: data/visdrone_uavdt_coco_joint.yaml")
    print(f"   train:")
    print(f"     - VisDrone/images/train")
    print(f"     - UAVDT_YOLO/train/images")
    print(f"     - COCO_UAV_YOLO/train/images")
    print(f"\n2. 启动三数据集联合训练")
    print(f"   python train_depth.py \\")
    print(f"       --data visdrone_uavdt_coco_joint.yaml \\")
    print(f"       --epochs 300 \\")
    print(f"       --name exp_joint_with_coco_v1")
    
    print(f"\n⚠️  重要提示:")
    print(f"   - COCO域差异大,可能带来负迁移")
    print(f"   - 建议对比 with/without COCO 的性能")
    print(f"   - 如果with COCO性能更差,说明不需要COCO")

if __name__ == '__main__':
    main()
