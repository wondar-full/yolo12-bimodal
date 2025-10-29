#!/bin/bash
# Validation script for RGB-only models (e.g., official YOLO12x pretrained)
# 验证 RGB-only 模型 (如官方预训练的 YOLO12x)s

MODEL=${1:-"models/yolo12x.pt"}
DATA=${2:-"data/visdrone-rgb-only.yaml"}  # 🆕 使用 RGB-only 配置
BATCH=${3:-16}
IMGSZ=${4:-640}
DEVICE=${5:-0}
CONF=${6:-0.001}
IOU=${7:-0.45}
MAX_DET=${8:-300}

echo ""
echo "🔍 VisDrone Validation (RGB-only)"
echo "Model:          $MODEL"
echo "Data:           $DATA"
echo "Batch size:     $BATCH"
echo "Image size:     $IMGSZ"
echo "Device:         $DEVICE"
echo "Confidence:     $CONF"
echo "NMS IoU:        $IOU"
echo "Max detections: $MAX_DET"
echo ""

# Check if model exists
if [ ! -f "$MODEL" ]; then
    echo "❌ Model not found: $MODEL"
    exit 1
fi

echo "Loading model from $MODEL..."
echo ""

# Run validation (RGB-only, no depth)
CUDA_VISIBLE_DEVICES=$DEVICE python val_depeth.py \
    --model "$MODEL" \
    --data "$DATA" \
    --batch "$BATCH" \
    --imgsz "$IMGSZ" \
    --conf-thres "$CONF" \
    --iou-thres "$IOU" \
    --max-det "$MAX_DET" \
    --visdrone-mode \
    --device "$DEVICE"

echo ""
echo "✅ Validation Complete!"
