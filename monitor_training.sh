#!/bin/bash
# 训练监控脚本
# 用途: 定期检查训练进度和关键指标

TRAIN_DIR="runs/train/exp_loss_weighted_v1"
LOG_FILE="${TRAIN_DIR}/training.log"
RESULTS_CSV="${TRAIN_DIR}/results.csv"

echo "========================================"
echo "📊 Loss权重改进训练监控"
echo "========================================"
echo ""

# 检查训练是否在运行
if pgrep -f "train_depth.py" > /dev/null; then
    echo "✅ 训练进程运行中"
else
    echo "⚠️  未检测到训练进程"
fi
echo ""

# 检查日志文件
if [ -f "${LOG_FILE}" ]; then
    echo "📄 训练日志: ${LOG_FILE}"
    echo ""
    
    # 提取最新epoch信息
    echo "📈 最新训练进度:"
    tail -n 30 "${LOG_FILE}" | grep "Epoch" | tail -n 5
    echo ""
    
    # 提取Loss信息
    echo "📉 最新Loss值:"
    tail -n 100 "${LOG_FILE}" | grep "box_loss\|cls_loss\|dfl_loss" | tail -n 3
    echo ""
else
    echo "❌ 日志文件不存在: ${LOG_FILE}"
fi

# 检查results.csv
if [ -f "${RESULTS_CSV}" ]; then
    echo "📊 性能指标统计 (最近5个epoch):"
    echo "   Epoch | mAP@0.5 | Precision | Recall | box_loss | cls_loss"
    echo "   ------|---------|-----------|--------|----------|----------"
    tail -n 5 "${RESULTS_CSV}" | awk -F',' '{printf "   %5s | %7s | %9s | %6s | %8s | %8s\n", $1, $8, $9, $10, $5, $6}'
    echo ""
    
    # 对比Baseline (假设Baseline mAP@0.5 = 41%)
    LATEST_MAP=$(tail -n 1 "${RESULTS_CSV}" | awk -F',' '{print $8}')
    echo "🎯 性能对比:"
    echo "   Baseline mAP@0.5: ~41%"
    echo "   当前 mAP@0.5: ${LATEST_MAP}"
    echo ""
else
    echo "⚠️  结果文件不存在,训练可能刚开始"
    echo ""
fi

# GPU使用情况
echo "🖥️  GPU状态:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | grep "^7"
echo ""

# 磁盘空间
echo "💾 磁盘空间:"
df -h /data2 | tail -n 1 | awk '{printf "   已用: %s / 总计: %s (%s)\n", $3, $2, $5}'
echo ""

echo "========================================"
echo "💡 提示"
echo "========================================"
echo "1. 训练预计15-20小时完成"
echo "2. 每50个epoch自动保存checkpoint"
echo "3. 最佳模型保存在: ${TRAIN_DIR}/weights/best.pt"
echo "4. 实时查看日志: tail -f ${LOG_FILE}"
echo "5. 查看TensorBoard: tensorboard --logdir ${TRAIN_DIR}"
echo ""
