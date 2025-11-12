"""
诊断Depth图像加载问题
=====================

检查点:
1. imread对不同格式depth图的处理
2. 验证depth图是否被正确加载和预处理
3. 对比I模式和L模式的差异
4. 检查depth通道是否全零

使用:
python diagnose_depth_loading.py \
    --dataset_root /data2/user/2024/lzy/Datasets/VisDrone \
    --num_samples 10
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from PIL import Image
import matplotlib.pyplot as plt


def test_imread_modes():
    """测试imread对不同深度图格式的处理"""
    print("="*80)
    print("测试imread函数对不同depth图像格式的处理")
    print("="*80)
    
    # 模拟imread函数
    def imread(filename, flags=cv2.IMREAD_COLOR):
        file_bytes = np.fromfile(filename, np.uint8)
        if filename.endswith((".tiff", ".tif")):
            success, frames = cv2.imdecodemulti(file_bytes, cv2.IMREAD_UNCHANGED)
            if success:
                return frames[0] if len(frames) == 1 and frames[0].ndim == 3 else np.stack(frames, axis=2)
            return None
        else:
            im = cv2.imdecode(file_bytes, flags)
            return im[..., None] if im is not None and im.ndim == 2 else im
    
    # 测试不同的flags
    test_cases = [
        (cv2.IMREAD_COLOR, "IMREAD_COLOR (默认)"),
        (cv2.IMREAD_GRAYSCALE, "IMREAD_GRAYSCALE"),
        (cv2.IMREAD_UNCHANGED, "IMREAD_UNCHANGED"),
    ]
    
    print("\n📌 关键发现:")
    print("  imread默认使用cv2.IMREAD_COLOR,对depth图像(灰度)的行为:")
    print("  - IMREAD_COLOR: 将灰度图转为3通道BGR(重复通道)")
    print("  - IMREAD_GRAYSCALE: 保持单通道灰度")
    print("  - IMREAD_UNCHANGED: 保持原始位深度(16-bit等)")
    print()
    
    return test_cases


def diagnose_dataset_depth(dataset_root, split='train', num_samples=10):
    """诊断数据集中depth图的加载情况"""
    
    print("\n" + "="*80)
    print(f"诊断数据集: {dataset_root}/{split}")
    print("="*80 + "\n")
    
    dataset_root = Path(dataset_root)
    
    # 路径
    image_dir = dataset_root / split / 'images'
    depth_dir = dataset_root / split / 'images' / 'depth'
    
    if not depth_dir.exists():
        print(f"❌ Depth目录不存在: {depth_dir}")
        return
    
    # 获取depth文件
    depth_files = sorted(list(depth_dir.glob("*.jpg")) + 
                        list(depth_dir.glob("*.png")) + 
                        list(depth_dir.glob("*.tif")))
    
    if not depth_files:
        print(f"❌ 未找到depth图像文件")
        return
    
    print(f"找到 {len(depth_files)} 个depth文件")
    print(f"文件格式: {[f.suffix for f in depth_files[:5]]}")
    print()
    
    # 抽样检查
    import random
    random.seed(42)
    samples = random.sample(depth_files, min(num_samples, len(depth_files)))
    
    # 统计信息
    stats = {
        'total': 0,
        'all_zero': 0,
        'has_value': 0,
        'channels': [],
        'dtypes': [],
        'value_ranges': [],
    }
    
    print(f"抽样检查 {len(samples)} 个depth文件...")
    print()
    
    for i, depth_file in enumerate(samples, 1):
        print(f"[{i}/{len(samples)}] {depth_file.name}")
        
        # 方法1: cv2.imread (默认IMREAD_COLOR)
        depth_cv2_color = cv2.imread(str(depth_file), cv2.IMREAD_COLOR)
        
        # 方法2: cv2.imread (IMREAD_UNCHANGED)
        depth_cv2_unchanged = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
        
        # 方法3: PIL Image.open + convert("L")
        depth_pil_l = np.array(Image.open(depth_file).convert("L"))
        
        # 方法4: PIL Image.open + convert("I")
        try:
            depth_pil_i = np.array(Image.open(depth_file).convert("I"))
        except:
            depth_pil_i = None
        
        # 方法5: ultralytics的imread (默认IMREAD_COLOR)
        from ultralytics.utils.patches import imread
        depth_ultra = imread(str(depth_file))
        
        print(f"  cv2.imread(IMREAD_COLOR):   shape={depth_cv2_color.shape if depth_cv2_color is not None else None}, "
              f"dtype={depth_cv2_color.dtype if depth_cv2_color is not None else None}, "
              f"range=[{depth_cv2_color.min():.1f}, {depth_cv2_color.max():.1f}]" if depth_cv2_color is not None else "None")
        
        print(f"  cv2.imread(IMREAD_UNCHANGED): shape={depth_cv2_unchanged.shape if depth_cv2_unchanged is not None else None}, "
              f"dtype={depth_cv2_unchanged.dtype if depth_cv2_unchanged is not None else None}, "
              f"range=[{depth_cv2_unchanged.min():.1f}, {depth_cv2_unchanged.max():.1f}]" if depth_cv2_unchanged is not None else "None")
        
        print(f"  PIL Image.open().convert('L'): shape={depth_pil_l.shape}, "
              f"dtype={depth_pil_l.dtype}, "
              f"range=[{depth_pil_l.min():.1f}, {depth_pil_l.max():.1f}]")
        
        if depth_pil_i is not None:
            print(f"  PIL Image.open().convert('I'): shape={depth_pil_i.shape}, "
                  f"dtype={depth_pil_i.dtype}, "
                  f"range=[{depth_pil_i.min():.1f}, {depth_pil_i.max():.1f}]")
        
        print(f"  ultralytics imread():        shape={depth_ultra.shape if depth_ultra is not None else None}, "
              f"dtype={depth_ultra.dtype if depth_ultra is not None else None}, "
              f"range=[{depth_ultra.min():.1f}, {depth_ultra.max():.1f}]" if depth_ultra is not None else "None")
        
        # 检查是否全零
        if depth_ultra is not None:
            is_zero = (depth_ultra == 0).all()
            has_value = (depth_ultra > 0).any()
            
            print(f"  ⚠️  全零: {is_zero}, 有非零值: {has_value}")
            
            if is_zero:
                stats['all_zero'] += 1
            elif has_value:
                stats['has_value'] += 1
                stats['value_ranges'].append((depth_ultra.min(), depth_ultra.max()))
            
            stats['channels'].append(depth_ultra.shape[2] if depth_ultra.ndim == 3 else 1)
            stats['dtypes'].append(str(depth_ultra.dtype))
        
        stats['total'] += 1
        print()
    
    # 汇总统计
    print("="*80)
    print("📊 统计汇总")
    print("="*80)
    print(f"检查文件数:     {stats['total']}")
    print(f"全零depth:      {stats['all_zero']} ({stats['all_zero']/stats['total']*100:.1f}%)")
    print(f"有效depth:      {stats['has_value']} ({stats['has_value']/stats['total']*100:.1f}%)")
    
    if stats['channels']:
        from collections import Counter
        print(f"通道数分布:     {dict(Counter(stats['channels']))}")
        print(f"数据类型分布:   {dict(Counter(stats['dtypes']))}")
    
    if stats['value_ranges']:
        min_vals = [r[0] for r in stats['value_ranges']]
        max_vals = [r[1] for r in stats['value_ranges']]
        print(f"值域范围:")
        print(f"  最小值: [{np.min(min_vals):.1f}, {np.max(min_vals):.1f}]")
        print(f"  最大值: [{np.min(max_vals):.1f}, {np.max(max_vals):.1f}]")
    
    print()
    
    # 诊断结论
    print("="*80)
    print("💡 诊断结论")
    print("="*80)
    
    if stats['all_zero'] / stats['total'] > 0.5:
        print("🚨 严重问题: 超过50%的depth图像全零!")
        print("   可能原因:")
        print("   1. depth图像格式不对(需要16-bit PNG,但保存成了8-bit)")
        print("   2. depth图像路径错误,加载了错误的文件")
        print("   3. depth图像生成过程有问题")
    elif stats['all_zero'] > 0:
        print(f"⚠️  警告: {stats['all_zero']}/{stats['total']} 个depth图像全零")
        print("   建议: 检查这些文件是否损坏或生成失败")
    else:
        print("✅ Depth图像加载正常,未发现全零问题")
    
    print()
    
    # 关键发现
    if stats['channels'] and max(stats['channels']) == 3:
        print("🔍 关键发现: imread使用IMREAD_COLOR将灰度depth转为3通道BGR!")
        print("   问题: 这会导致depth数据被错误地复制成3个相同通道")
        print("   解决: imread应使用cv2.IMREAD_UNCHANGED保持原始位深度")
        print()
        print("   示例:")
        print("   ❌ 错误: imread(depth_path)  # 默认IMREAD_COLOR")
        print("   ✅ 正确: imread(depth_path, cv2.IMREAD_UNCHANGED)")
    
    print("="*80)
    print()


def check_dataloader_output(dataset_root, split='train'):
    """检查DataLoader输出的实际数据"""
    print("\n" + "="*80)
    print("检查DataLoader实际输出")
    print("="*80 + "\n")
    
    # 尝试导入数据集类
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        
        from ultralytics.data import build_dataloader
        from ultralytics.cfg import get_cfg
        from ultralytics import YOLO
        
        # 加载配置
        cfg = get_cfg()
        cfg.data = f"{dataset_root}/visdrone-rgbd.yaml"  # 假设yaml文件
        cfg.batch = 1
        cfg.workers = 0
        
        print("尝试创建DataLoader...")
        # 这里需要实际的yaml配置文件才能运行
        # 仅作为示例代码
        
    except Exception as e:
        print(f"⚠️  无法创建DataLoader: {e}")
        print("   这需要完整的训练环境和配置文件")


def main():
    parser = argparse.ArgumentParser(description="诊断Depth图像加载")
    parser.add_argument('--dataset_root', type=str, required=True,
                       help='数据集根目录')
    parser.add_argument('--split', type=str, default='train',
                       help='要检查的split')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='抽样检查的数量')
    
    args = parser.parse_args()
    
    # 测试imread模式
    test_imread_modes()
    
    # 诊断数据集
    diagnose_dataset_depth(args.dataset_root, args.split, args.num_samples)
    
    print("\n下一步:")
    print("1. 如果发现imread使用错误的flags,修改dataset.py中的加载代码")
    print("2. 将 imread(depth_path) 改为 imread(depth_path, cv2.IMREAD_UNCHANGED)")
    print("3. 重新训练验证修复效果")


if __name__ == "__main__":
    main()
