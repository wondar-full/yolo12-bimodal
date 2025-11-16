"""
DepthAnythingV2 - I模式Depth生成脚本
====================================

保存32-bit int depth (PIL Image I模式)

使用方法:
CUDA_VISIBLE_DEVICES=4 python run_depth_anything_v2_I_mode.py \
    --encoder vits \
    --img-path /data2/user/2024/lzy/Datasets/VisDrone2019-DET-YOLO/VisDrone2YOLO/VisDrone2019-DET-train/images/rgb \
    --outdir /data2/user/2024/lzy/Datasets/VisDrone2019-DET-YOLO/VisDrone2YOLO/VisDrone2019-DET-train/images/depth_I_mode \
    --max-depth 100.0
"""

import argparse
import cv2
import glob
import numpy as np
import os
import torch
from PIL import Image

from depth_anything_v2.dpt import DepthAnythingV2


def save_I_mode_depth(depth, output_path, max_depth_meters=100.0, min_depth_meters=0.5):
    """
    将depth保存为I模式PNG (32-bit signed int)
    
    Args:
        depth: numpy array, float32, 相对深度值 (DepthAnything输出)
        output_path: str, 输出文件路径
        max_depth_meters: float, 场景最大深度(米)
        min_depth_meters: float, 场景最小深度(米)
    
    Returns:
        depth_int32: numpy array, int32
    """
    # 1. 归一化到0-1
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    
    # 2. 映射到实际深度范围 (米)
    # DepthAnything输出是inverse depth,需要反转
    # depth_norm=1.0 → 近处 → min_depth
    # depth_norm=0.0 → 远处 → max_depth
    depth_meters = min_depth_meters + (max_depth_meters - min_depth_meters) * (1 - depth_norm)
    
    # 3. 转为毫米并保存为int32
    # 毫米单位提供更高精度: 0.5m-100m → 500-100000
    depth_mm = (depth_meters * 1000.0).astype(np.int32)
    
    # 4. 保存为I模式PNG
    img = Image.fromarray(depth_mm, mode='I')
    img.save(output_path)
    
    # 验证
    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"  ✅ I模式depth: dtype=int32, "
          f"range=[{depth_mm.min()}, {depth_mm.max()}], "
          f"size={file_size_kb:.1f}KB")
    
    return depth_mm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 - I Mode Depth Generator')
    
    parser.add_argument('--img-path', type=str, required=True,
                       help='输入RGB图像路径或目录')
    parser.add_argument('--input-size', type=int, default=518,
                       help='模型输入尺寸')
    parser.add_argument('--outdir', type=str, default='./depth_I_mode',
                       help='输出目录')
    
    parser.add_argument('--encoder', type=str, default='vits', 
                       choices=['vits', 'vitb', 'vitl', 'vitg'],
                       help='编码器类型: vits(最快), vitb, vitl, vitg(最准)')
    
    parser.add_argument('--max-depth', type=float, default=100.0,
                       help='场景最大深度(米), UAV场景推荐100')
    parser.add_argument('--min-depth', type=float, default=0.5,
                       help='场景最小深度(米), UAV场景推荐0.5')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    print("="*80)
    print("Depth Anything V2 - I Mode (32-bit int) Depth Generator")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Encoder: {args.encoder}")
    print(f"Depth Range: {args.min_depth}m - {args.max_depth}m")
    print(f"Output Format: PIL Image I mode (32-bit signed int, 毫米单位)")
    print(f"Output: {args.outdir}")
    print("="*80)
    print()
    
    # 模型配置
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # 加载模型
    print("加载模型...")
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    checkpoint_path = f'checkpoints/depth_anything_v2_{args.encoder}.pth'
    depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    print(f"✅ 模型加载完成: {checkpoint_path}\n")
    
    # 获取文件列表
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
        filenames = [f for f in filenames if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    print(f"找到 {len(filenames)} 个图像文件\n")
    
    if len(filenames) == 0:
        print("❌ 未找到任何图像文件!")
        print(f"   检查路径: {args.img_path}")
        exit(1)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # 处理每张图像
    import time
    start_time = time.time()
    
    for k, filename in enumerate(filenames):
        print(f'[{k+1}/{len(filenames)}] {os.path.basename(filename)}')
        
        # 读取RGB图像
        raw_image = cv2.imread(filename)
        if raw_image is None:
            print(f"  ⚠️  无法读取图像,跳过")
            continue
        
        # 推理depth
        depth = depth_anything.infer_image(raw_image, args.input_size)
        
        # 保存I模式depth
        base_name = os.path.splitext(os.path.basename(filename))[0]
        depth_I_path = os.path.join(args.outdir, base_name + '.png')
        
        depth_int32 = save_I_mode_depth(
            depth, 
            depth_I_path,
            max_depth_meters=args.max_depth,
            min_depth_meters=args.min_depth
        )
        
        # 进度统计
        if (k + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (k + 1)
            remaining = avg_time * (len(filenames) - k - 1)
            print(f"  ⏱️  已处理 {k+1}/{len(filenames)}, "
                  f"平均 {avg_time:.2f}秒/张, "
                  f"预计剩余 {remaining/60:.1f}分钟\n")
        else:
            print()
    
    total_time = time.time() - start_time
    
    print("="*80)
    print("✅ 所有图像处理完成!")
    print(f"总耗时: {total_time/60:.1f}分钟")
    print(f"平均速度: {total_time/len(filenames):.2f}秒/张")
    print(f"I模式depth保存在: {args.outdir}")
    print("="*80)
    
    # 验证第一个样本
    if filenames:
        print("\n验证第一个样本:")
        first_depth_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filenames[0]))[0] + '.png')
        
        # 用PIL读取I模式
        img_I = Image.open(first_depth_path)
        depth_check = np.array(img_I)
        
        print(f"  文件路径: {first_depth_path}")
        print(f"  PIL Image mode: {img_I.mode}")  # 应该是'I'
        print(f"  NumPy dtype: {depth_check.dtype}")  # int32
        print(f"  shape: {depth_check.shape}")
        print(f"  range: [{depth_check.min()}, {depth_check.max()}]")
        print(f"  unique values: {len(np.unique(depth_check))}")
        
        if img_I.mode == 'I' and depth_check.dtype == np.int32 and depth_check.max() > 1000:
            print("  ✅ 验证通过: I模式depth格式正确!")
            print(f"  ✅ 深度范围合理: {depth_check.min()/1000:.1f}m - {depth_check.max()/1000:.1f}m")
            print(f"  ✅ 精度足够: {len(np.unique(depth_check))} 个唯一值 (远超8-bit的256)")
        else:
            print("  ❌ 验证失败: depth格式不正确!")
    
    print("="*80)
    print("\n下一步:")
    print("1. 验证depth质量:")
    print(f"   python diagnose_depth_loading.py --dataset_root /path/to/dataset --num_samples 20")
    print("\n2. 更新数据集YAML配置,指向新的depth_I_mode目录")
    print("\n3. 删除旧的.cache文件:")
    print("   find /data2/user/2024/lzy/Datasets -name '*.cache' -delete")
    print("\n4. 启动训练验证效果")
    print("="*80)
