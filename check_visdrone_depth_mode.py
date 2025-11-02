"""
快速检查VisDrone深度图模式 (L vs I)
运行此脚本确定VisDrone使用的是哪种模式，以便UAVDT保持一致
"""

from PIL import Image
import numpy as np
from pathlib import Path

def check_depth_mode(depth_path):
    """检查单张深度图的模式和属性"""
    try:
        img = Image.open(depth_path)
        arr = np.array(img)
        
        print(f"\n{'='*60}")
        print(f"文件: {depth_path.name}")
        print(f"{'='*60}")
        print(f"PIL模式:    {img.mode}")
        print(f"图像尺寸:   {img.size} (width x height)")
        print(f"NumPy类型:  {arr.dtype}")
        print(f"数值范围:   {arr.min()} - {arr.max()}")
        print(f"平均值:     {arr.mean():.2f}")
        print(f"文件大小:   {depth_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # 判断模式
        if img.mode == 'L':
            print(f"\n✅ 确认: 使用 **L模式** (8-bit 灰度)")
            print(f"   - 每个像素: 1 byte")
            print(f"   - 数值范围: 0-255")
            print(f"   - 理论大小: {img.size[0] * img.size[1] / 1024 / 1024:.2f} MB (未压缩)")
            return 'L'
        
        elif img.mode == 'I':
            print(f"\n✅ 确认: 使用 **I模式** (32-bit 整数)")
            print(f"   - 每个像素: 4 bytes")
            print(f"   - 数值范围: -{2**31} ~ {2**31-1}")
            print(f"   - 理论大小: {img.size[0] * img.size[1] * 4 / 1024 / 1024:.2f} MB (未压缩)")
            return 'I'
        
        else:
            print(f"\n⚠️ 警告: 使用 **{img.mode}模式** (非标准深度图格式)")
            return img.mode
    
    except Exception as e:
        print(f"❌ 错误: 无法读取 {depth_path.name} - {e}")
        return None

def main():
    print("\n" + "="*60)
    print("VisDrone 深度图模式检查工具")
    print("="*60)
    
    # ⚠️ 如果自动检测失败,请手动指定路径:
    # 取消下面一行的注释,并填写你的VisDrone深度图路径
    # manual_path = Path(r'你的VisDrone深度图路径')
    manual_path = None
    
    if manual_path and manual_path.exists():
        depth_dir = manual_path
        print(f"✅ 使用手动指定路径")
    else:
        # VisDrone深度图目录 - 尝试多个可能的位置
        possible_bases = [
            Path(r'/data2/user/2024/lzy/Datasets/VisDrone2019-DET-YOLO/VisDrone2YOLO'),
            Path(r'f:\CV\Paper\yoloDepth\datasets\VisDrone'),
            Path(r'f:\CV\datasets\VisDrone'),
            Path(r'.\datasets\VisDrone'),
        ]
        
        possible_paths = []
        for base in possible_bases:
            if base.exists():
                possible_paths.extend([
                    base / 'VisDrone2019-DET-train''/images/d',
                    base / 'train' / 'depths',
                    base / 'VisDrone2019-DET-train' / 'depths',
                ])
        
        depth_dir = None
        for path in possible_paths:
            if path.exists() and list(path.glob('*.png')):
                depth_dir = path
                break
        
        if not depth_dir:
            print(f"\n❌ 错误: 找不到VisDrone深度图目录!")
            print(f"\n尝试过的路径:")
            for path in possible_paths[:6]:  # 只显示前6个
                print(f"  - {path}")
            print(f"  ...")
            print(f"\n📝 解决方案:")
            print(f"1. 找到你的VisDrone深度图目录 (包含.png文件)")
            print(f"2. 修改此脚本第75行:")
            print(f"   manual_path = Path(r'你的实际路径')")
            print(f"3. 重新运行脚本")
            print(f"\n或者直接告诉我路径,我来修改!")
            return
    
    print(f"深度图目录: {depth_dir}")
    
    # 获取所有深度图
    depth_files = sorted(depth_dir.glob('*.png'))
    
    if len(depth_files) == 0:
        print(f"❌ 错误: {depth_dir} 中没有找到深度图!")
        return
    
    print(f"找到 {len(depth_files)} 张深度图\n")
    
    # 检查前3张深度图
    modes = []
    for i, depth_path in enumerate(depth_files[:3]):
        mode = check_depth_mode(depth_path)
        if mode:
            modes.append(mode)
    
    # 总结
    print(f"\n" + "="*60)
    print("总结与建议")
    print("="*60)
    
    if len(set(modes)) == 1:
        mode = modes[0]
        print(f"✅ VisDrone深度图统一使用: **{mode}模式**")
        
        if mode == 'L':
            print(f"\n📝 UAVDT深度图生成建议:")
            print(f"   ✅ generate_depths_uavdt.py **无需修改**")
            print(f"   ✅ 当前代码已使用L模式 (第70行)")
            print(f"   ✅ 数值范围: 0-255")
            
        elif mode == 'I':
            print(f"\n📝 UAVDT深度图生成建议:")
            print(f"   ⚠️ generate_depths_uavdt.py **需要修改**")
            print(f"   ⚠️ 第70行: mode='L' → mode='I'")
            print(f"   ⚠️ 数值范围: 0-255 → 0-65535")
            print(f"\n修改代码:")
            print(f"   # 当前 (L模式)")
            print(f"   depth_uint8 = depth_normalized.astype(np.uint8)")
            print(f"   depth_img = Image.fromarray(depth_uint8, mode='L')")
            print(f"")
            print(f"   # 修改为 (I模式)")
            print(f"   depth_int32 = (depth_normalized * 65535 / 255).astype(np.int32)")
            print(f"   depth_img = Image.fromarray(depth_int32, mode='I')")
    
    else:
        print(f"⚠️ 警告: VisDrone深度图使用了多种模式: {set(modes)}")
        print(f"   建议重新生成VisDrone深度图以保持一致性")
    
    # 存储估算
    if modes and modes[0] in ['L', 'I']:
        mode = modes[0]
        sample_img = Image.open(depth_files[0])
        single_size_mb = sample_img.size[0] * sample_img.size[1] * (1 if mode == 'L' else 4) / 1024 / 1024
        
        print(f"\n📊 UAVDT深度图存储估算 (23,829张):")
        print(f"   模式: {mode}")
        print(f"   单张大小: ~{single_size_mb:.2f} MB")
        print(f"   总大小: ~{single_size_mb * 23829 / 1024:.1f} GB")
        
        if mode == 'L':
            print(f"   ✅ 存储友好 (8-bit)")
        else:
            print(f"   ⚠️ 存储占用大 (32-bit, 是L模式的4倍)")
    
    print(f"\n下一步:")
    print(f"1. 如果需要修改代码,我会立即为你修改")
    print(f"2. 然后运行: python generate_depths_uavdt.py")
    print(f"3. 开始生成UAVDT深度图 (4-6小时)")

if __name__ == '__main__':
    main()
