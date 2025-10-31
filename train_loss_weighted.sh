#!/bin/bash
# Loss权重改进训练脚本
# 用途: 验证尺寸自适应Loss加权(Small×2.0, Medium×1.5, Large×1.0)的效果

echo "========================================"
echo "🚀 Loss权重改进训练启动"
echo "========================================"
echo ""

# 训练参数
DATA_YAML="data/visdrone-rgbd.yaml"
EPOCHS=300
BATCH_SIZE=8
IMG_SIZE=640
DEVICE=2
PROJECT="runs/train"
NAME="exp_loss_weighted_v1"
WEIGHTS="yolo12n.pt"

echo "📋 训练配置:"
echo "   数据集: ${DATA_YAML}"
echo "   轮次: ${EPOCHS} epochs"
echo "   批次大小: ${BATCH_SIZE}"
echo "   图像尺寸: ${IMG_SIZE}×${IMG_SIZE}"
echo "   GPU设备: ${DEVICE}"
echo "   保存路径: ${PROJECT}/${NAME}"
echo "   预训练权重: ${WEIGHTS}"
echo ""

echo "🎯 改进内容:"
echo "   ✅ Loss权重按GT尺寸调整"
echo "      - Small目标 (<32×32):  Loss权重 ×2.0"
echo "      - Medium目标 (32~96):   Loss权重 ×1.5"
echo "      - Large目标 (≥96×96):   Loss权重 ×1.0"
echo ""

echo "📊 预期效果:"
echo "   - Small mAP: 30.94% → 33-34% (+2-3%)"
echo "   - Overall mAP: ~41% → 42-43%"
echo ""

echo "⏱️  预计训练时间: 15-20小时 (RTX 4090)"
echo ""

read -p "确认开始训练? (按Enter继续, Ctrl+C取消) " confirm

echo ""
echo "========================================"
echo "🏃 开始训练..."
echo "========================================"
echo ""

# 创建输出目录(如果不存在)
mkdir -p ${PROJECT}/${NAME}

# 启动训练
CUDA_VISIBLE_DEVICES=${DEVICE} python train_depth.py \
    --data ${DATA_YAML} \
    --epochs ${EPOCHS} \
    --batch ${BATCH_SIZE} \
    --imgsz ${IMG_SIZE} \
    --device 0 \
    --project ${PROJECT} \
    --name ${NAME} \
    --save_period 50 \
    --patience 100 \
    --workers 8 \
    2>&1 | tee ${PROJECT}/${NAME}/training.log

echo ""
echo "========================================"
echo "✅ 训练完成!"
echo "========================================"
echo ""

echo "📁 结果保存在:"
echo "   权重: ${PROJECT}/${NAME}/weights/best.pt"
echo "   日志: ${PROJECT}/${NAME}/training.log"
echo "   曲线: ${PROJECT}/${NAME}/results.png"
echo ""

echo "📊 下一步验证:"
echo "   python val_depth.py --weights ${PROJECT}/${NAME}/weights/best.pt"
echo ""
