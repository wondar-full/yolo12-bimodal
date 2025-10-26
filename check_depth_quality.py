#!/usr/bin/env python3
"""
Depth Map Quality Checker
检查生成的深度图质量是否符合要求

Usage:
    python check_depth_quality.py --depth_dir ../testDepth
"""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt


def check_depth_map(depth_path):
    """
    检查单个深度图的质量
    
    Returns:
        dict: 质量统计信息
    """
    # 读取深度图
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    
    if depth is None:
        return {"error": "Failed to load image"}
    
    # 基本信息
    stats = {
        "filename": depth_path.name,
        "shape": depth.shape,
        "dtype": str(depth.dtype),
        "channels": len(depth.shape),
    }
    
    # 如果是多通道，转换为单通道
    if len(depth.shape) == 3:
        if depth.shape[2] == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        else:
            depth = depth[:, :, 0]
        stats["warning"] = "Multi-channel depth converted to single channel"
    
    # 数值范围
    stats["min_value"] = float(depth.min())
    stats["max_value"] = float(depth.max())
    stats["mean_value"] = float(depth.mean())
    stats["std_value"] = float(depth.std())
    
    # 动态范围
    dynamic_range = stats["max_value"] - stats["min_value"]
    stats["dynamic_range"] = float(dynamic_range)
    
    # 零值比例（可能是无效深度）
    zero_ratio = (depth == 0).sum() / depth.size
    stats["zero_ratio"] = float(zero_ratio)
    
    # NaN/Inf检查
    if depth.dtype in [np.float32, np.float64]:
        stats["has_nan"] = bool(np.isnan(depth).any())
        stats["has_inf"] = bool(np.isinf(depth).any())
    else:
        stats["has_nan"] = False
        stats["has_inf"] = False
    
    # 质量评估
    issues = []
    if dynamic_range < 10:
        issues.append(f"❌ Very low dynamic range ({dynamic_range:.1f})")
    elif dynamic_range < 50:
        issues.append(f"⚠️  Low dynamic range ({dynamic_range:.1f})")
    else:
        issues.append(f"✅ Good dynamic range ({dynamic_range:.1f})")
    
    if zero_ratio > 0.5:
        issues.append(f"❌ Too many zeros ({zero_ratio*100:.1f}%)")
    elif zero_ratio > 0.1:
        issues.append(f"⚠️  Many zeros ({zero_ratio*100:.1f}%)")
    else:
        issues.append(f"✅ Few zeros ({zero_ratio*100:.1f}%)")
    
    if stats.get("has_nan") or stats.get("has_inf"):
        issues.append("❌ Contains NaN/Inf values")
    else:
        issues.append("✅ No NaN/Inf")
    
    stats["issues"] = issues
    
    return stats


def visualize_depth(depth_path, save_dir=None):
    """
    可视化深度图
    """
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    
    if depth is None:
        print(f"❌ Failed to load: {depth_path}")
        return
    
    # 多通道转单通道
    if len(depth.shape) == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY) if depth.shape[2] == 3 else depth[:, :, 0]
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Depth Quality Analysis: {depth_path.name}", fontsize=14, fontweight='bold')
    
    # 1. 原始深度图
    im0 = axes[0, 0].imshow(depth, cmap='viridis')
    axes[0, 0].set_title('Raw Depth Map')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # 2. 归一化深度图
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    im1 = axes[0, 1].imshow(depth_norm, cmap='jet')
    axes[0, 1].set_title('Normalized Depth (MINMAX)')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # 3. 直方图
    axes[1, 0].hist(depth.ravel(), bins=100, color='blue', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Depth Value Distribution')
    axes[1, 0].set_xlabel('Depth Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 统计信息
    stats_text = f"""
    Shape: {depth.shape}
    Dtype: {depth.dtype}
    
    Min: {depth.min():.2f}
    Max: {depth.max():.2f}
    Mean: {depth.mean():.2f}
    Std: {depth.std():.2f}
    
    Dynamic Range: {depth.max() - depth.min():.2f}
    Zero Ratio: {(depth == 0).sum() / depth.size * 100:.2f}%
    
    Median: {np.median(depth):.2f}
    25th percentile: {np.percentile(depth, 25):.2f}
    75th percentile: {np.percentile(depth, 75):.2f}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                     verticalalignment='center', transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # 保存或显示
    if save_dir:
        save_path = Path(save_dir) / f"{depth_path.stem}_analysis.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Check depth map quality")
    parser.add_argument("--depth_dir", type=str, required=True, help="Directory containing depth maps")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--save_vis", type=str, default=None, help="Directory to save visualizations")
    parser.add_argument("--max_files", type=int, default=10, help="Max number of files to check")
    
    args = parser.parse_args()
    
    # 查找深度图
    depth_dir = Path(args.depth_dir)
    if not depth_dir.exists():
        print(f"❌ Directory not found: {depth_dir}")
        sys.exit(1)
    
    depth_files = list(depth_dir.glob("*.png")) + list(depth_dir.glob("*.jpg"))
    
    if not depth_files:
        print(f"❌ No PNG/JPG files found in: {depth_dir}")
        sys.exit(1)
    
    print(f"Found {len(depth_files)} depth maps in {depth_dir}")
    print("=" * 80)
    
    # 检查前N个文件
    all_stats = []
    for i, depth_path in enumerate(depth_files[:args.max_files]):
        print(f"\n[{i+1}/{min(len(depth_files), args.max_files)}] Checking: {depth_path.name}")
        
        stats = check_depth_map(depth_path)
        all_stats.append(stats)
        
        if "error" in stats:
            print(f"  ❌ Error: {stats['error']}")
            continue
        
        print(f"  Shape: {stats['shape']}, Dtype: {stats['dtype']}")
        print(f"  Range: [{stats['min_value']:.1f}, {stats['max_value']:.1f}]")
        print(f"  Mean: {stats['mean_value']:.2f}, Std: {stats['std_value']:.2f}")
        
        for issue in stats['issues']:
            print(f"  {issue}")
        
        # 可视化
        if args.visualize:
            visualize_depth(depth_path, save_dir=args.save_vis)
    
    # 汇总统计
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    
    valid_stats = [s for s in all_stats if "error" not in s]
    
    if valid_stats:
        print(f"Valid depth maps: {len(valid_stats)}/{len(all_stats)}")
        
        # 平均动态范围
        avg_dynamic_range = np.mean([s['dynamic_range'] for s in valid_stats])
        print(f"Average dynamic range: {avg_dynamic_range:.2f}")
        
        # 平均零值比例
        avg_zero_ratio = np.mean([s['zero_ratio'] for s in valid_stats])
        print(f"Average zero ratio: {avg_zero_ratio*100:.2f}%")
        
        # 数据类型分布
        dtypes = [s['dtype'] for s in valid_stats]
        unique_dtypes = set(dtypes)
        print(f"Data types: {unique_dtypes}")
        
        # 质量评估
        print("\nQuality Assessment:")
        
        good_range = sum(1 for s in valid_stats if s['dynamic_range'] >= 50)
        print(f"  Good dynamic range (≥50): {good_range}/{len(valid_stats)}")
        
        low_zeros = sum(1 for s in valid_stats if s['zero_ratio'] < 0.1)
        print(f"  Low zero ratio (<10%): {low_zeros}/{len(valid_stats)}")
        
        # 总体建议
        print("\n" + "=" * 80)
        print("Recommendations:")
        print("=" * 80)
        
        if avg_dynamic_range < 50:
            print("⚠️  ISSUE: Low average dynamic range")
            print("   → Check depth estimation model settings")
            print("   → Ensure input images have good contrast")
            print("   → Consider using 16-bit PNG instead of 8-bit")
        else:
            print("✅ Dynamic range is good")
        
        if avg_zero_ratio > 0.1:
            print("⚠️  ISSUE: Many zero values in depth maps")
            print("   → Zero may indicate invalid/unknown depth")
            print("   → YOLORGBDDataset will handle this, but affects quality")
            print("   → Consider post-processing to fill holes")
        else:
            print("✅ Zero ratio is acceptable")
        
        # 检查8-bit vs 16-bit
        if all(s['dtype'] == 'uint8' for s in valid_stats):
            print("\n⚠️  NOTE: All depth maps are 8-bit (uint8)")
            print("   → 8-bit depth: 0-255 range (256 discrete levels)")
            print("   → 16-bit depth: 0-65535 range (65536 levels)")
            print("   → For better precision, regenerate with 16-bit PNG")
            print("   → Command example:")
            print("     cv2.imwrite('depth.png', depth.astype(np.uint16))")
        elif all(s['dtype'] == 'uint16' for s in valid_stats):
            print("\n✅ Using 16-bit depth maps (excellent precision)")
        
    else:
        print("❌ No valid depth maps found")


if __name__ == "__main__":
    main()
