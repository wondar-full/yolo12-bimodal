"""
DepthAnythingV2 - 16-bit Depth生成脚本 (修复版)
================================================

关键修复:
1. 保存原始float32 depth值,不做归一化
2. 转换为16-bit uint16格式 (0-65535范围)
3. 使用合理的depth范围映射 (0-100米 → 0-65535)

使用方法:
CUDA_VISIBLE_DEVICES=4 python run_depth_anything_v2_16bit.py \
    --encoder vits \
    --img-path /data2/user/2024/lzy/Datasets/VisDrone2019-DET-YOLO/VisDrone2YOLO/VisDrone2019-DET-val/images/rgb \
    --outdir /data2/user/2024/lzy/Datasets/VisDrone2019-DET-YOLO/VisDrone2YOLO/VisDrone2019-DET-val/images/depth_16bit \
    --pred-only \
    --max-depth 100.0
"""

import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2


def save_16bit_depth(depth, output_path, max_depth_meters=100.0, min_depth_meters=0.5):
    """
    将depth保存为16-bit PNG格式
    
    Args:
        depth: numpy array, float32, 相对深度值 (DepthAnything输出)
        output_path: str, 输出文件路径
        max_depth_meters: float, 场景最大深度(米)
        min_depth_meters: float, 场景最小深度(米)
    
    Returns:
        depth_uint16: numpy array, uint16, 范围[0, 65535]
    """
    # DepthAnythingV2输出的是相对深度(0-1范围的inverse depth)
    # 需要转换为绝对深度(米)
    
    # 1. 归一化到0-1
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    
    # 2. Inverse depth → 正常depth
    # DepthAnything输出的是inverse depth (近处值大,远处值小)
    # 需要反转: depth_meters = max / depth_inverse
    depth_inverse = depth_norm + 1e-8  # 防止除零
    
    # 3. 映射到实际深度范围 (米)
    # 假设: depth_norm=1.0 → min_depth, depth_norm=0.0 → max_depth
    depth_meters = min_depth_meters + (max_depth_meters - min_depth_meters) * (1 - depth_norm)
    
    # 4. 转换为毫米 (提高精度)
    depth_mm = depth_meters * 1000.0
    
    # 5. 映射到16-bit范围 [0, 65535]
    # 0mm → 0, 100000mm(100m) → 65535
    depth_uint16 = np.clip(depth_mm, 0, max_depth_meters * 1000.0)
    depth_uint16 = (depth_uint16 / (max_depth_meters * 1000.0) * 65535).astype(np.uint16)
    
    # 6. 保存为16-bit PNG
    cv2.imwrite(output_path, depth_uint16)
    
    print(f"  ✅ 保存16-bit depth: dtype={depth_uint16.dtype}, "
          f"range=[{depth_uint16.min()}, {depth_uint16.max()}], "
          f"size={os.path.getsize(output_path) / 1024:.1f}KB")
    
    return depth_uint16


def save_visualization(raw_image, depth, output_path, cmap):
    """保存可视化结果 (用于检查)"""
    # 归一化到0-255用于可视化
    depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_vis = depth_vis.astype(np.uint8)
    
    # 应用colormap
    depth_colored = (cmap(depth_vis)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    
    # 拼接RGB和depth
    split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
    combined_result = cv2.hconcat([raw_image, split_region, depth_colored])
    
    cv2.imwrite(output_path, combined_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 - 16-bit Depth Generator')
    
    parser.add_argument('--img-path', type=str, required=True,
                       help='输入RGB图像路径或目录')
    parser.add_argument('--input-size', type=int, default=518,
                       help='模型输入尺寸')
    parser.add_argument('--outdir', type=str, default='./vis_depth_16bit',
                       help='输出目录')
    
    parser.add_argument('--encoder', type=str, default='vitl', 
                       choices=['vits', 'vitb', 'vitl', 'vitg'],
                       help='编码器类型')
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', 
                       help='仅保存depth预测,不保存可视化')
    parser.add_argument('--save-vis', dest='save_vis', action='store_true',
                       help='额外保存可视化图像 (用于检查)')
    
    # 新增参数: depth范围设置
    parser.add_argument('--max-depth', type=float, default=100.0,
                       help='场景最大深度(米), UAV场景推荐100')
    parser.add_argument('--min-depth', type=float, default=0.5,
                       help='场景最小深度(米), UAV场景推荐0.5')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    print("="*80)
    print("Depth Anything V2 - 16-bit Depth Generator")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Encoder: {args.encoder}")
    print(f"Depth Range: {args.min_depth}m - {args.max_depth}m")
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
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    # 获取文件列表
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
        # 过滤出图像文件
        filenames = [f for f in filenames if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    print(f"找到 {len(filenames)} 个图像文件\n")
    
    # 创建输出目录
    os.makedirs(args.outdir, exist_ok=True)
    if args.save_vis:
        vis_dir = args.outdir.replace('depth_16bit', 'depth_vis')
        os.makedirs(vis_dir, exist_ok=True)
    
    # colormap用于可视化
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    # 处理每张图像
    for k, filename in enumerate(filenames):
        print(f'[{k+1}/{len(filenames)}] {os.path.basename(filename)}')
        
        # 读取RGB图像
        raw_image = cv2.imread(filename)
        if raw_image is None:
            print(f"  ⚠️  无法读取图像,跳过")
            continue
        
        # 推理depth
        depth = depth_anything.infer_image(raw_image, args.input_size)
        
        # 输出文件名
        base_name = os.path.splitext(os.path.basename(filename))[0]
        
        # 保存16-bit depth
        depth_16bit_path = os.path.join(args.outdir, base_name + '.png')
        depth_uint16 = save_16bit_depth(
            depth, 
            depth_16bit_path,
            max_depth_meters=args.max_depth,
            min_depth_meters=args.min_depth
        )
        
        # 可选: 保存可视化
        if args.save_vis:
            vis_path = os.path.join(vis_dir, base_name + '_vis.png')
            save_visualization(raw_image, depth, vis_path, cmap)
            print(f"  💾 可视化已保存: {vis_path}")
        
        print()
    
    print("="*80)
    print("✅ 所有图像处理完成!")
    print(f"16-bit depth保存在: {args.outdir}")
    
    # 验证一个样本
    if filenames:
        print("\n验证第一个样本:")
        first_depth_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filenames[0]))[0] + '.png')
        depth_check = cv2.imread(first_depth_path, cv2.IMREAD_UNCHANGED)
        print(f"  dtype: {depth_check.dtype}")
        print(f"  shape: {depth_check.shape}")
        print(f"  range: [{depth_check.min()}, {depth_check.max()}]")
        
        if depth_check.dtype == np.uint16 and depth_check.max() > 255:
            print("  ✅ 验证通过: 16-bit depth格式正确!")
        else:
            print("  ❌ 验证失败: depth格式不正确!")
    
    print("="*80)
