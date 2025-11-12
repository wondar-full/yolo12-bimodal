#!/bin/bash

# 基础验证 (查看全局分布)
echo ""
echo "🔍 Step 1: 基础验证 - 查看全局Small/Medium/Large分布"
echo "----------------------------------------"
CUDA_VISIBLE_DEVICES=7 sh val_depth.sh

echo ""
echo "📋 验证完成! 请查看上方的 '📊 Global Size Distribution' 输出"
echo ""
echo "关键问题:"
echo "1. Large目标GT占比是否<5%? (如果是,说明阈值可能需要调整)"
echo "2. Large目标Pred占比是否与GT接近? (如果差异大,说明模型学习有问题)"
echo "3. Large Recall=0.316是因为样本少还是模型差?"
echo ""

read -p "按Enter键继续诊断实验,或Ctrl+C退出......" 

# 诊断实验1: NMS调优
echo ""
echo "🔍 Step 2: NMS调优实验"
echo "----------------------------------------"
echo "测试IoU=0.3 (降低阈值,减少误删)"

# 备份原始val_depth.sh
cp val_depth.sh val_depth.sh.bak

# 修改NMS IoU阈值
sed -i 's/--iou [0-9.]\+/--iou 0.3/g' val_depth.sh

CUDA_VISIBLE_DEVICES=7 sh val_depth.sh

echo ""
echo "📋 NMS=0.3结果:"
echo "如果Large Recall提升 → NMS误删问题"
echo "如果Large Recall不变 → 模型检测能力问题"
echo ""

read -p "按Enter键继续下一个实验..."

# 诊断实验2: max_det调优
echo ""
echo "🔍 Step 3: max_det调优实验"
echo "----------------------------------------"
echo "测试max_det=500 (增加检测数量上限)"

# 恢复原始配置
cp val_depth.sh.bak val_depth.sh

# 修改max_det
sed -i 's/--max-det [0-9]\+/--max-det 500/g' val_depth.sh

CUDA_VISIBLE_DEVICES=7 sh val_depth.sh

echo ""
echo "📋 max_det=500结果:"
echo "如果Large Recall提升 → 检测数量不足"
echo "如果Large Recall不变 → 其他原因"
echo ""

# 恢复原始配置
cp val_depth.sh.bak val_depth.sh
rm val_depth.sh.bak

echo ""
echo "=================================="
echo "✅ 诊断实验完成!"
echo "=================================="
echo ""
echo "📊 总结建议:"
echo "1. 查看全局分布统计,确认Large目标数量"
echo "2. 对比NMS/max_det实验结果,定位问题"
echo "3. 如果是模型问题 → 需要Loss加权/训练策略改进"
echo "4. 如果是后处理问题 → 调整NMS/max_det参数"
echo ""
echo "📁 所有结果保存在 runs/val/ 目录"
echo "下一步: 根据诊断结果制定改进方案"
