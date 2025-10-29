#!/usr/bin/env python3
"""
诊断脚本:检查为什么验证时没有输出 Small/Medium/Large 的 mAP
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics.utils.metrics_visdrone import DetMetricsVisDrone
import numpy as np


def test_metrics_initialization():
    """测试 Metrics 初始化"""
    print("=" * 70)
    print("测试 1: DetMetricsVisDrone 初始化")
    print("=" * 70)
    
    # 创建 metrics 对象 (DetMetricsVisDrone 不接受 save_dir/plot 参数)
    metrics = DetMetricsVisDrone(visdrone_mode=True)
    
    print(f"✅ visdrone_mode: {metrics.visdrone_mode}")
    print(f"✅ small_area_thresh: {metrics.small_area_thresh}")
    print(f"✅ medium_area_thresh: {metrics.medium_area_thresh}")
    print(f"✅ stats_by_size keys: {list(metrics.stats_by_size.keys())}")
    
    for size_key in ['small', 'medium', 'large']:
        print(f"   - stats_by_size['{size_key}']: {list(metrics.stats_by_size[size_key].keys())}")
    
    print("\n✅ 测试 1 通过: 初始化正常\n")
    return metrics


def test_update_stats(metrics):
    """测试 update_stats 方法"""
    print("=" * 70)
    print("测试 2: update_stats() 方法")
    print("=" * 70)
    
    # 模拟一个 batch 的统计数据
    stat = {
        # 全局统计
        "tp": np.random.rand(10, 10) > 0.5,  # [N_pred, N_iou_thresh]
        "conf": np.random.rand(10),
        "pred_cls": np.random.randint(0, 10, size=10),
        "target_cls": np.array([0, 1, 2, 3, 4]),
        "target_img": np.array([0, 1, 2, 3, 4]),
        
        # 分尺度统计 (Phase 2.5 v2.2 格式)
        "tp_small": np.random.rand(3, 10) > 0.5,
        "conf_small": np.random.rand(3),
        "pred_cls_small": np.random.randint(0, 10, size=3),
        "target_cls_small": np.array([0, 1]),
        
        "tp_medium": np.random.rand(5, 10) > 0.5,
        "conf_medium": np.random.rand(5),
        "pred_cls_medium": np.random.randint(0, 10, size=5),
        "target_cls_medium": np.array([2, 3, 4]),
        
        "tp_large": np.random.rand(2, 10) > 0.5,
        "conf_large": np.random.rand(2),
        "pred_cls_large": np.random.randint(0, 10, size=2),
        "target_cls_large": np.array([5]),
    }
    
    metrics.update_stats(stat)
    
    # 检查 stats_by_size 是否被填充
    print(f"✅ stats_by_size['small']['tp'] 长度: {len(metrics.stats_by_size['small']['tp'])}")
    print(f"✅ stats_by_size['medium']['tp'] 长度: {len(metrics.stats_by_size['medium']['tp'])}")
    print(f"✅ stats_by_size['large']['tp'] 长度: {len(metrics.stats_by_size['large']['tp'])}")
    
    if all(len(metrics.stats_by_size[s]['tp']) > 0 for s in ['small', 'medium', 'large']):
        print("\n✅ 测试 2 通过: update_stats 正确填充 stats_by_size\n")
    else:
        print("\n❌ 测试 2 失败: stats_by_size 未被填充!\n")
        return False
    
    return True


def test_process_method(metrics):
    """测试 process() 方法"""
    print("=" * 70)
    print("测试 3: process() 方法输出")
    print("=" * 70)
    
    # 设置类别名称
    metrics.names = {i: f"class_{i}" for i in range(10)}
    
    # 调用 process (应该打印分尺度 mAP)
    print("\n🔍 调用 metrics.process()...\n")
    stats = metrics.process()
    
    print(f"\n✅ process() 返回: {type(stats)}")
    print(f"✅ box_small.map50: {metrics.box_small.map50:.4f}")
    print(f"✅ box_medium.map50: {metrics.box_medium.map50:.4f}")
    print(f"✅ box_large.map50: {metrics.box_large.map50:.4f}")
    
    print("\n✅ 测试 3 完成\n")
    return stats


def test_visdrone_mode_flag():
    """测试 visdrone_mode 标志传递"""
    print("=" * 70)
    print("测试 4: visdrone_mode 标志传递")
    print("=" * 70)
    
    # 模拟从 val.py 创建 metrics 的方式
    from ultralytics import YOLO
    
    print("\n🔍 检查 args.visdrone_mode 是否传递到 metrics...\n")
    
    # 检查默认 args
    from ultralytics.cfg import get_cfg
    args = get_cfg()
    
    print(f"✅ 默认 args 是否有 visdrone_mode: {hasattr(args, 'visdrone_mode')}")
    if hasattr(args, 'visdrone_mode'):
        print(f"   - visdrone_mode = {args.visdrone_mode}")
    else:
        print(f"   - ⚠️ 没有 visdrone_mode 属性,需要在 val.py 中显式设置!")
    
    print("\n✅ 测试 4 完成\n")


def main():
    """主测试流程"""
    print("\n" + "="*70)
    print("🔍 诊断: 为什么验证时没有输出 Small/Medium/Large mAP")
    print("="*70 + "\n")
    
    # 测试 1: 初始化
    metrics = test_metrics_initialization()
    
    # 测试 2: update_stats
    if not test_update_stats(metrics):
        print("❌ update_stats 测试失败,后续测试终止")
        return
    
    # 测试 3: process
    test_process_method(metrics)
    
    # 测试 4: visdrone_mode 标志
    test_visdrone_mode_flag()
    
    print("\n" + "="*70)
    print("✅ 诊断完成!")
    print("="*70)
    print("\n💡 下一步:")
    print("   1. 确认 val.py 中正确设置 visdrone_mode=True")
    print("   2. 确认 _process_batch 返回 tp_small/medium/large")
    print("   3. 在服务器上运行此脚本: python diagnose_metrics_output.py")
    print()


if __name__ == "__main__":
    main()
