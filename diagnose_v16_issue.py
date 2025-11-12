"""
exp_joint_v16 诊断脚本
数据集修复后性能仍然很低的原因分析
"""

import os
from pathlib import Path
from collections import Counter
import yaml

def check_uavdt_labels_fixed():
    """检查UAVDT标签是否真的修复了"""
    print("="*80)
    print("📋 检查1: UAVDT标签修复验证")
    print("="*80)
    
    # 本地无法检查服务器数据，需要在服务器运行
    print("⚠️  需要在服务器上运行以下命令验证:")
    print()
    print("cd /data2/user/2024/lzy/Datasets/UAVDT_YOLO/train/labels/rgb")
    print("find . -name '*.txt' -exec cat {} \\; | awk '{print $1}' | sort | uniq -c")
    print()
    print("✅ 应该看到: 394633 3, 17491 5, 10787 8")
    print("❌ 如果仍然是: 394633 4, 17491 6, 10787 9 → 修复脚本未执行或未生效")
    print()


def check_dataset_yaml():
    """检查数据集YAML配置"""
    print("="*80)
    print("📋 检查2: 数据集配置验证")
    print("="*80)
    
    # 本地查看配置
    yaml_path = Path("ultralytics/data/visdrone_uavdt_joint.yaml")
    
    if not yaml_path.exists():
        print(f"❌ 配置文件不存在: {yaml_path}")
        print("⚠️  需要在服务器查看:")
        print("  cat /data2/user/2024/lzy/yolo12-bimodal/data/visdrone_uavdt_joint.yaml")
    else:
        print(f"✅ 配置文件: {yaml_path}")
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        print(f"Names: {config.get('names', 'NOT FOUND')}")
        print(f"NC: {config.get('nc', 'NOT FOUND')}")
    print()


def analyze_performance_gap():
    """分析性能差距"""
    print("="*80)
    print("📊 性能对比分析")
    print("="*80)
    
    print("实验结果对比:")
    print("  exp_joint_v15 (标签错误):  mAP@0.5 = 19.51% (最后epoch)")
    print("  exp_joint_v16 (标签修复?): mAP@0.5 = 20.82% (最后epoch)")
    print("  提升幅度: +1.31 百分点")
    print()
    print("🤔 分析:")
    print("  1. 预期提升: +20~25个百分点 (从19.51% → 40-45%)")
    print("  2. 实际提升: +1.31个百分点 (从19.51% → 20.82%)")
    print("  3. 结论: **数据集修复可能没有生效!**")
    print()
    print("可能原因:")
    print("  ❌ 标签文件未真正修改 (脚本未执行/失败)")
    print("  ❌ 训练时仍在读取旧的缓存数据")
    print("  ❌ 数据集路径配置错误,读取了未修复的副本")
    print("  ❌ 只修复了train集,val集未修复")
    print()


def check_class_distribution():
    """检查类别分布"""
    print("="*80)
    print("📋 检查3: 训练日志中的类别分布")
    print("="*80)
    
    print("⚠️  需要在服务器检查训练日志:")
    print()
    print("grep -i 'instance' /data2/user/2024/lzy/yolo12-bimodal/runs/train/exp_joint_v16/train.log | head -20")
    print()
    print("✅ 应该看到: ~800k instances (VisDrone 200k + UAVDT 422k + 其他)")
    print("❌ 如果看到: ~200k instances → UAVDT数据未正确加载")
    print()


def check_validation_set():
    """检查验证集"""
    print("="*80)
    print("📋 检查4: 验证集标签")
    print("="*80)
    
    print("⚠️  关键问题: 是否只修复了train集,而忘记修复val/test集?")
    print()
    print("在服务器运行:")
    print()
    print("# 检查验证集标签")
    print("cd /data2/user/2024/lzy/Datasets/UAVDT_YOLO/val/labels/rgb")
    print("find . -name '*.txt' -exec cat {} \\; | awk '{print $1}' | sort | uniq -c")
    print()
    print("# 检查测试集标签 (如果有)")
    print("cd /data2/user/2024/lzy/Datasets/UAVDT_YOLO/test/labels/rgb")
    print("find . -name '*.txt' -exec cat {} \\; | awk '{print $1}' | sort | uniq -c")
    print()
    print("💡 重要:")
    print("  如果val集仍然是4,6,9 → 模型训练正确但验证错误!")
    print("  这会导致训练mAP看起来正常,但验证mAP很低!")
    print()


def check_cache_issue():
    """检查缓存问题"""
    print("="*80)
    print("📋 检查5: 数据加载缓存")
    print("="*80)
    
    print("YOLO可能缓存了旧的标签数据!")
    print()
    print("解决方案:")
    print("1. 删除缓存文件:")
    print("   rm -f /data2/user/2024/lzy/Datasets/UAVDT_YOLO/train/*.cache")
    print("   rm -f /data2/user/2024/lzy/Datasets/UAVDT_YOLO/val/*.cache")
    print()
    print("2. 强制重新扫描:")
    print("   在训练脚本中添加 --cache False")
    print()


def root_cause_checklist():
    """根本原因检查清单"""
    print("="*80)
    print("🔍 根本原因排查清单")
    print("="*80)
    
    checklist = [
        ("1. 训练集标签已修复", "cd /path/to/train/labels; grep命令检查", "3,5,8"),
        ("2. 验证集标签已修复", "cd /path/to/val/labels; grep命令检查", "3,5,8"),
        ("3. 测试集标签已修复", "cd /path/to/test/labels; grep命令检查", "3,5,8"),
        ("4. 缓存文件已删除", "rm *.cache", "无.cache文件"),
        ("5. 数据集路径正确", "检查visdrone_uavdt_joint.yaml", "指向正确目录"),
        ("6. 训练实例数正确", "查看train.log", "~800k instances"),
        ("7. 无其他数据副本", "检查是否有旧数据集", "只有一份数据"),
    ]
    
    print()
    for task, command, expected in checklist:
        print(f"☐ {task}")
        print(f"   命令: {command}")
        print(f"   预期: {expected}")
        print()


def suggest_next_steps():
    """建议下一步操作"""
    print("="*80)
    print("🚀 建议的诊断步骤")
    print("="*80)
    
    steps = [
        {
            "step": "步骤1: 验证训练集标签",
            "action": "在服务器运行类别统计命令",
            "expected": "看到 3,5,8",
            "if_fail": "重新运行 fix_uavdt_category_ids.py"
        },
        {
            "step": "步骤2: 验证验证集标签",
            "action": "统计val集的类别分布",
            "expected": "看到 3,5,8",
            "if_fail": "对val集也运行修复脚本"
        },
        {
            "step": "步骤3: 清除缓存",
            "action": "删除所有.cache文件",
            "expected": "缓存被清除",
            "if_fail": "手动检查是否有遗漏"
        },
        {
            "step": "步骤4: 重新训练",
            "action": "使用--cache False重新训练",
            "expected": "mAP提升到40%+",
            "if_fail": "检查数据加载逻辑"
        },
    ]
    
    for i, step_info in enumerate(steps, 1):
        print(f"\n{i}. {step_info['step']}")
        print(f"   操作: {step_info['action']}")
        print(f"   预期: {step_info['expected']}")
        print(f"   失败处理: {step_info['if_fail']}")
    
    print()
    print("="*80)


def create_verification_script():
    """生成服务器验证脚本"""
    print("="*80)
    print("📝 生成服务器验证脚本")
    print("="*80)
    
    script = """#!/bin/bash
# UAVDT数据集验证脚本
# 在服务器上运行此脚本以诊断问题

echo "========================================="
echo "UAVDT数据集完整性验证"
echo "========================================="

DATASET_ROOT="/data2/user/2024/lzy/Datasets/UAVDT_YOLO"

# 1. 检查训练集
echo ""
echo "1. 训练集标签分布:"
cd $DATASET_ROOT/train/labels/rgb
echo "  统计类别ID:"
find . -name "*.txt" -exec cat {} \\; | awk '{print $1}' | sort | uniq -c
echo ""
echo "  ✅ 应该看到: 394633 3, 17491 5, 10787 8"
echo "  ❌ 如果是: 394633 4, 17491 6, 10787 9 → 未修复"

# 2. 检查验证集
echo ""
echo "2. 验证集标签分布:"
cd $DATASET_ROOT/val/labels/rgb
echo "  统计类别ID:"
find . -name "*.txt" -exec cat {} \\; | awk '{print $1}' | sort | uniq -c
echo ""
echo "  ✅ 应该也是: X 3, X 5, X 8"
echo "  ❌ 如果是: X 4, X 6, X 9 → 验证集未修复 (关键问题!)"

# 3. 检查缓存文件
echo ""
echo "3. 检查缓存文件:"
cd $DATASET_ROOT
find . -name "*.cache" -type f
echo "  如果有.cache文件 → 可能使用了旧数据"
echo "  建议删除: find . -name '*.cache' -delete"

# 4. 检查训练日志
echo ""
echo "4. 检查最近训练的实例数:"
TRAIN_LOG="/data2/user/2024/lzy/yolo12-bimodal/runs/train/exp_joint_v16/train.log"
if [ -f "$TRAIN_LOG" ]; then
    grep -i "instance" $TRAIN_LOG | head -5
    echo "  ✅ 应该看到: ~800000 instances"
    echo "  ❌ 如果只有: ~200000 instances → UAVDT未正确加载"
else
    echo "  ⚠️  训练日志不存在: $TRAIN_LOG"
fi

echo ""
echo "========================================="
echo "验证完成!"
echo "========================================="
"""
    
    output_path = Path("verify_uavdt_dataset.sh")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(script)
    
    print(f"✅ 已生成验证脚本: {output_path}")
    print()
    print("使用方法:")
    print(f"  1. 上传到服务器: scp {output_path} user@server:~/")
    print("  2. 添加执行权限: chmod +x verify_uavdt_dataset.sh")
    print("  3. 运行: ./verify_uavdt_dataset.sh")
    print()


def main():
    """主函数"""
    print("\n" + "="*80)
    print("🔬 exp_joint_v16 性能未提升诊断报告")
    print("="*80)
    print()
    
    print("问题描述:")
    print("  修复UAVDT类别ID后 (4→3, 6→5, 9→8)")
    print("  性能仅从 19.51% 提升到 20.82%")
    print("  远低于预期的 40-45%")
    print()
    
    # 运行各项检查
    check_uavdt_labels_fixed()
    check_dataset_yaml()
    analyze_performance_gap()
    check_class_distribution()
    check_validation_set()
    check_cache_issue()
    root_cause_checklist()
    suggest_next_steps()
    create_verification_script()
    
    print("="*80)
    print("📌 最可能的原因 (优先级排序)")
    print("="*80)
    print()
    print("🥇 原因1: 验证集标签未修复")
    print("   - 训练集修复了,但val集仍然是4,6,9")
    print("   - 导致训练正常,但验证mAP很低")
    print("   - 验证方法: 检查val/labels/rgb的类别分布")
    print()
    print("🥈 原因2: 数据缓存问题")
    print("   - YOLO缓存了修复前的标签")
    print("   - 训练时读取的是旧缓存")
    print("   - 验证方法: 删除.cache文件重新训练")
    print()
    print("🥉 原因3: 训练集未真正修复")
    print("   - fix脚本未执行或执行失败")
    print("   - 标签仍然是4,6,9")
    print("   - 验证方法: 检查train/labels/rgb的类别分布")
    print()
    print("="*80)
    print()
    print("🎯 立即行动:")
    print("  1. 运行 verify_uavdt_dataset.sh (已生成)")
    print("  2. 根据输出结果确定具体原因")
    print("  3. 修复后重新训练 exp_joint_v17")
    print()


if __name__ == "__main__":
    main()
