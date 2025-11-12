#!/bin/bash
# UAVDT数据集正确验证脚本
# 考虑labels目录也分rgb/和d/子目录

echo "========== UAVDT数据集验证（正确版本）=========="
echo ""

# 检查RGB图像
RGB_COUNT=$(find /data2/user/2024/lzy/Datasets/UAVDT_YOLO/train/images/rgb -name "*.jpg" 2>/dev/null | wc -l)
echo "✅ RGB图像数量: $RGB_COUNT"

# 检查深度图
DEPTH_COUNT=$(find /data2/user/2024/lzy/Datasets/UAVDT_YOLO/train/images/d -name "*.png" 2>/dev/null | wc -l)
echo "✅ 深度图数量: $DEPTH_COUNT"

# 检查标签（正确路径：labels/rgb/）
LABEL_RGB_COUNT=$(find /data2/user/2024/lzy/Datasets/UAVDT_YOLO/train/labels/rgb -name "*.txt" 2>/dev/null | wc -l)
echo "✅ 标签文件数量 (labels/rgb/): $LABEL_RGB_COUNT"

# 检查标签（如果有labels/d/）
LABEL_D_COUNT=$(find /data2/user/2024/lzy/Datasets/UAVDT_YOLO/train/labels/d -name "*.txt" 2>/dev/null | wc -l)
if [ $LABEL_D_COUNT -gt 0 ]; then
    echo "✅ 标签文件数量 (labels/d/): $LABEL_D_COUNT"
fi

echo ""
echo "========== 数据对齐检查 =========="

# RGB-标签对齐检查
if [ $RGB_COUNT -eq $LABEL_RGB_COUNT ]; then
    echo "✅ RGB-标签对齐: 完美对齐 ($RGB_COUNT = $LABEL_RGB_COUNT)"
else
    echo "❌ RGB-标签不对齐: RGB=$RGB_COUNT, 标签=$LABEL_RGB_COUNT"
fi

# RGB-深度图对齐检查
if [ $RGB_COUNT -eq $DEPTH_COUNT ]; then
    echo "✅ RGB-深度图对齐: 完美对齐 ($RGB_COUNT = $DEPTH_COUNT)"
else
    echo "⚠️  RGB-深度图不对齐: RGB=$RGB_COUNT, 深度=$DEPTH_COUNT"
fi

echo ""
echo "========== 验证结论 =========="
if [ $RGB_COUNT -eq 23829 ] && [ $LABEL_RGB_COUNT -eq 23829 ] && [ $DEPTH_COUNT -eq 23829 ]; then
    echo "🎉🎉🎉 UAVDT数据集完整！所有数据完美对齐！"
    echo "✅ 可以立即开始联合训练！"
else
    echo "⚠️  数据集不完整，详情："
    echo "   期望: 23829张"
    echo "   实际: RGB=$RGB_COUNT, 深度=$DEPTH_COUNT, 标签=$LABEL_RGB_COUNT"
fi

echo ""
echo "========== 目录结构示例 =========="
echo "UAVDT_YOLO/train/"
echo "├── images/"
echo "│   ├── rgb/         ($RGB_COUNT 张 .jpg)"
echo "│   └── d/           ($DEPTH_COUNT 张 .png)"
echo "└── labels/"
echo "    ├── rgb/         ($LABEL_RGB_COUNT 张 .txt)"
if [ $LABEL_D_COUNT -gt 0 ]; then
    echo "    └── d/           ($LABEL_D_COUNT 张 .txt)"
fi
echo ""
