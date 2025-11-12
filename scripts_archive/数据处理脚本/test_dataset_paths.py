#!/usr/bin/env python3
"""
测试数据集路径是否正确配置
快速验证YAML配置和实际文件结构是否匹配
"""

import yaml
from pathlib import Path

def test_dataset_paths(yaml_path):
    """测试数据集路径配置"""
    
    print("="*80)
    print("🔍 测试数据集路径配置")
    print("="*80)
    
    # 加载YAML
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 获取根路径
    path_value = config.get('path', '.')
    if isinstance(path_value, list):
        data_root = Path([p for p in path_value if p][0])
    else:
        data_root = Path(path_value)
    
    print(f"\n📂 数据集根目录: {data_root}")
    print(f"   存在: {'✅' if data_root.exists() else '❌'}")
    
    # 测试各个划分
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*80}")
        print(f"📊 {split.upper()} 划分")
        print(f"{'='*80}")
        
        # 图像路径
        img_key = split
        if img_key in config:
            img_rel_path = config[img_key]
            img_dir = data_root / img_rel_path
            print(f"\n📷 RGB图像:")
            print(f"   相对路径: {img_rel_path}")
            print(f"   完整路径: {img_dir}")
            print(f"   存在: {'✅' if img_dir.exists() else '❌'}")
            
            if img_dir.exists():
                # 统计图像数量
                jpg_files = list(img_dir.glob('*.jpg'))
                png_files = list(img_dir.glob('*.png'))
                total_imgs = len(jpg_files) + len(png_files)
                print(f"   图像数量: {total_imgs} (.jpg: {len(jpg_files)}, .png: {len(png_files)})")
                
                # 显示前3个文件
                if jpg_files:
                    print(f"   示例文件: {jpg_files[0].name}")
        else:
            print(f"\n📷 RGB图像: ❌ YAML中未配置'{split}'键")
        
        # 深度图路径
        depth_key = f'{split}_depth'
        if depth_key in config:
            depth_rel_path = config[depth_key]
            depth_dir = data_root / depth_rel_path
            print(f"\n🌊 深度图:")
            print(f"   相对路径: {depth_rel_path}")
            print(f"   完整路径: {depth_dir}")
            print(f"   存在: {'✅' if depth_dir.exists() else '❌'}")
            
            if depth_dir.exists():
                # 统计深度图数量
                png_files = list(depth_dir.glob('*.png'))
                jpg_files = list(depth_dir.glob('*.jpg'))
                total_depths = len(png_files) + len(jpg_files)
                print(f"   深度图数量: {total_depths} (.png: {len(png_files)}, .jpg: {len(jpg_files)})")
                
                # 显示前3个文件
                if png_files:
                    print(f"   示例文件: {png_files[0].name}")
        else:
            print(f"\n🌊 深度图: ⚠️  YAML中未配置'{depth_key}'键")
        
        # 标签路径 (推断)
        if img_key in config:
            img_rel_path = config[img_key]
            
            # 推断标签路径
            label_rel_path = img_rel_path
            if '/images/' in label_rel_path:
                parts = label_rel_path.split('/images/')
                label_rel_path = parts[0] + '/labels'
            elif '\\images\\' in label_rel_path:
                parts = label_rel_path.split('\\images\\')
                label_rel_path = parts[0] + '\\labels'
            else:
                label_rel_path = label_rel_path.replace('images', 'labels')
            
            label_dir = data_root / label_rel_path
            print(f"\n🏷️  标签:")
            print(f"   推断路径: {label_rel_path}")
            print(f"   完整路径: {label_dir}")
            print(f"   存在: {'✅' if label_dir.exists() else '❌'}")
            
            if label_dir.exists():
                # 统计标签数量
                txt_files = list(label_dir.glob('*.txt'))
                print(f"   标签数量: {len(txt_files)}")
                
                # 显示前3个文件
                if txt_files:
                    print(f"   示例文件: {txt_files[0].name}")
                    
                    # 读取第一个标签文件检查格式
                    with open(txt_files[0], 'r') as f:
                        lines = f.readlines()
                    if lines:
                        print(f"   标签格式示例: {lines[0].strip()}")
                        print(f"   该文件目标数: {len(lines)}")
    
    print(f"\n{'='*80}")
    print("✅ 路径测试完成!")
    print("="*80)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='测试数据集路径配置')
    parser.add_argument('--data', type=str, default='data/visdrone-rgbd.yaml',
                       help='数据集YAML配置文件路径')
    
    args = parser.parse_args()
    
    test_dataset_paths(args.data)
