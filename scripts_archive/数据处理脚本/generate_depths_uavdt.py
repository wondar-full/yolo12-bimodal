"""
UAVDT数据集深度图生成脚本
使用ZoeDepth模型为UAVDT的RGB图像生成对应的深度图

Author: AI Assistant
Date: 2025-10-31
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
        # 尝试加载ZoeDepth
        model = torch.hub.load('isl-org/ZoeDepth', 'ZoeD_N', pretrained=True)
        model.eval()
        
        # 尝试使用GPU
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
        print("  或参考: https://github.com/isl-org/ZoeDepth")
        sys.exit(1)

def generate_depth(model, image_path, output_path, device='cuda'):
    """
    为单张图像生成深度图
    
    Args:
        model: ZoeDepth模型
        image_path: 输入RGB图像路径
        output_path: 输出深度图路径
        device: 'cuda' 或 'cpu'
    """
    try:
        # 读取RGB图像
        rgb = Image.open(image_path).convert('RGB')
        
        # 生成深度图
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
    """
    批量生成深度图
    
    Args:
        model: ZoeDepth模型
        images_dir: 输入图像目录
        output_dir: 输出深度图目录
        device: 'cuda' 或 'cpu'
    """
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
    uavdt_yolo = Path(r'f:\CV\Paper\yoloDepth\yoloDepth\datasets\UAVDT_YOLO')
    
    print("\n" + "="*60)
    print("UAVDT 深度图生成工具")
    print("="*60)
    print(f"数据集路径: {uavdt_yolo}")
    
    # 检查目录
    if not uavdt_yolo.exists():
        print(f"❌ 错误: UAVDT_YOLO目录不存在: {uavdt_yolo}")
        print("请先运行: python convert_uavdt_to_yolo.py")
        return
    
    # 检查训练集图像
    train_images = uavdt_yolo / 'train' / 'images'
    if not train_images.exists() or len(list(train_images.glob('*.jpg'))) == 0:
        print(f"❌ 错误: 训练集图像不存在: {train_images}")
        print("请先运行: python convert_uavdt_to_yolo.py")
        return
    
    # 加载模型
    model = load_zoedepth_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 生成训练集深度图
    print("\n" + "="*60)
    print("生成 UAVDT Train 深度图")
    print("="*60)
    train_count = batch_generate_depths(
        model=model,
        images_dir=train_images,
        output_dir=uavdt_yolo / 'train' / 'depths',
        device=device
    )
    
    # 生成验证集深度图
    val_images = uavdt_yolo / 'val' / 'images'
    if val_images.exists() and len(list(val_images.glob('*.jpg'))) > 0:
        print("\n" + "="*60)
        print("生成 UAVDT Val 深度图")
        print("="*60)
        val_count = batch_generate_depths(
            model=model,
            images_dir=val_images,
            output_dir=uavdt_yolo / 'val' / 'depths',
            device=device
        )
    else:
        print("\n⚠️ 验证集图像不存在,跳过")
        val_count = 0
    
    # 总结
    print("\n" + "="*60)
    print("✅ 深度图生成完成!")
    print("="*60)
    print(f"训练集: {train_count} 张")
    print(f"验证集: {val_count} 张")
    print(f"\n输出目录:")
    print(f"{uavdt_yolo}/")
    print(f"├── train/")
    print(f"│   ├── images/  (RGB)")
    print(f"│   ├── labels/  (YOLO)")
    print(f"│   └── depths/  (Depth) ← 新生成")
    print(f"└── val/")
    print(f"    └── depths/  (Depth) ← 新生成")
    
    print(f"\n下一步:")
    print(f"1. 验证深度图质量: 打开几张depths/*.png查看")
    print(f"2. 创建联合数据集配置: data/visdrone_uavdt_joint.yaml")
    print(f"3. 启动联合训练: python train_depth.py --data visdrone_uavdt_joint.yaml")

if __name__ == '__main__':
    main()
