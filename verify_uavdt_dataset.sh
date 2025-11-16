#!/bin/bash
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
find . -name "*.txt" -exec cat {} \; | awk '{print $1}' | sort | uniq -c
echo ""
echo "  ✅ 应该看到: 394633 3, 17491 5, 10787 8"
echo "  ❌ 如果是: 394633 4, 17491 6, 10787 9 → 未修复"

# 2. 检查验证集
echo ""
echo "2. 验证集标签分布:"
cd $DATASET_ROOT/val/labels/rgb
echo "  统计类别ID:"
find . -name "*.txt" -exec cat {} \; | awk '{print $1}' | sort | uniq -c
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
