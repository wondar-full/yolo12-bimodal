"""
将3通道灰度图转换为单通道L模式
如果VisDrone深度图是3通道RGB格式（但三个通道值相同），转为单通道L模式
"""

from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm

def convert_rgb_to_grayscale(input_dir, output_dir=None, inplace=False):
    """
    将RGB灰度图转为L模式灰度图
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录 (如果为None且inplace=True，则覆盖原文件)
        inplace: 是否覆盖原文件
    """
    input_dir = Path(input_dir)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    elif not inplace:
        output_dir = input_dir.parent / f"{input_dir.name}_L"
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = input_dir
    
    image_files = sorted(input_dir.glob('*.png'))
    print(f"找到 {len(image_files)} 张图像")
    
    converted = 0
    skipped = 0
    
    for img_path in tqdm(image_files, desc="转换为L模式"):
        try:
            img = Image.open(img_path)
            arr = np.array(img)
            
            # 检查是否需要转换
            if img.mode == 'L':
                skipped += 1
                if not inplace and output_dir != input_dir:
                    # 如果不是原地替换，复制文件
                    img.save(output_dir / img_path.name)
                continue
            
            # 如果是RGB模式，检查三个通道是否相同
            if len(arr.shape) == 3 and arr.shape[2] == 3:
                # 检查是否是灰度图（三个通道值相同）
                if np.allclose(arr[:, :, 0], arr[:, :, 1]) and np.allclose(arr[:, :, 1], arr[:, :, 2]):
                    # 取第一个通道
                    gray = arr[:, :, 0]
                    gray_img = Image.fromarray(gray, mode='L')
                    gray_img.save(output_dir / img_path.name)
                    converted += 1
                else:
                    print(f"⚠️ {img_path.name} 不是灰度图（三通道值不同），跳过")
                    skipped += 1
            else:
                print(f"⚠️ {img_path.name} 格式异常: {img.mode}, shape={arr.shape}")
                skipped += 1
        
        except Exception as e:
            print(f"❌ {img_path.name} 转换失败: {e}")
            skipped += 1
    
    print(f"\n统计:")
    print(f"  ✅ 转换: {converted}")
    print(f"  ⏭️  跳过: {skipped}")
    print(f"\n输出目录: {output_dir}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='将RGB灰度图转换为L模式')
    parser.add_argument('--input', type=str, required=True, help='输入目录')
    parser.add_argument('--output', type=str, help='输出目录 (默认: input_L)')
    parser.add_argument('--inplace', action='store_true', help='覆盖原文件 (谨慎使用!)')
    
    args = parser.parse_args()
    
    if args.inplace:
        print("⚠️ 警告: 将覆盖原文件!")
        confirm = input("确认继续? (yes/no): ")
        if confirm.lower() != 'yes':
            print("已取消")
            return
    
    convert_rgb_to_grayscale(args.input, args.output, args.inplace)

if __name__ == '__main__':
    main()
