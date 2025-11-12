#!/bin/bash
# Phase 3: Batch Training Script for All Model Scales
# 训练所有尺度的 ChannelC2f 模型以对标 RemDet

# ================================================================================================
# Configuration
# ================================================================================================

CUDA_DEVICE=6  # GPU device ID
BASE_DIR="/data2/user/2024/lzy/yolo12-bimodal"
DATA_CFG="data/visdrone-rgbd.yaml"
EPOCHS=150
BATCH_SIZE=16  # 根据GPU显存调整

# Pretrained weights (optional, set to empty string for training from scratch)
PRETRAINED_N="yolo12n.pt"  # or "runs/train/phase1_n/weights/best.pt"
PRETRAINED_S="yolo12s.pt"  # or "runs/train/phase1_s/weights/best.pt"
PRETRAINED_M="yolo12m.pt"
PRETRAINED_L="yolo12l.pt"
PRETRAINED_X="yolo12x.pt"

# ================================================================================================
# Model Scales Configuration
# ================================================================================================

# Format: "scale:model_config:pretrained:batch_size:lr0"
MODELS=(
    "n:ultralytics/cfg/models/12/yolo12n-rgbd-channelc2f.yaml:${PRETRAINED_N}:32:0.001"
    "s:ultralytics/cfg/models/12/yolo12s-rgbd-channelc2f.yaml:${PRETRAINED_S}:16:0.001"
    "m:ultralytics/cfg/models/12/yolo12m-rgbd-channelc2f.yaml:${PRETRAINED_M}:8:0.0008"
    "l:ultralytics/cfg/models/12/yolo12l-rgbd-channelc2f.yaml:${PRETRAINED_L}:4:0.0005"
    "x:ultralytics/cfg/models/12/yolo12x-rgbd-channelc2f.yaml:${PRETRAINED_X}:4:0.0005"
)

# ================================================================================================
# Functions
# ================================================================================================

train_model() {
    local scale=$1
    local model_cfg=$2
    local pretrained=$3
    local batch=$4
    local lr=$5
    
    echo "================================================================================"
    echo "Training YOLO12-${scale^^} ChannelC2f (Phase 3)"
    echo "================================================================================"
    echo "Model: $model_cfg"
    echo "Pretrained: ${pretrained:-None (from scratch)}"
    echo "Batch Size: $batch"
    echo "Learning Rate: $lr"
    echo "Epochs: $EPOCHS"
    echo ""
    
    # Training command
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train_phase3.py \
        --model "$model_cfg" \
        --data "$DATA_CFG" \
        --epochs $EPOCHS \
        --batch $batch \
        --lr0 $lr \
        --name "phase3_channelc2f_${scale}" \
        --pretrained "$pretrained" \
        --project "runs/train" \
        --exist-ok
    
    if [ $? -eq 0 ]; then
        echo "✅ YOLO12-${scale^^} training completed successfully!"
    else
        echo "❌ YOLO12-${scale^^} training failed!"
        return 1
    fi
    
    echo ""
}

# ================================================================================================
# Main Training Loop
# ================================================================================================

echo "================================================================================"
echo "Phase 3: Batch Training - All Model Scales"
echo "================================================================================"
echo "Objective: Train n, s, m, l, x models to benchmark against RemDet"
echo ""
echo "Models to train:"
for model_spec in "${MODELS[@]}"; do
    IFS=':' read -ra PARTS <<< "$model_spec"
    echo "  - YOLO12-${PARTS[0]^^} (batch=${PARTS[3]}, lr0=${PARTS[4]})"
done
echo ""
echo "Press Enter to start training (or Ctrl+C to cancel)..."
read

# Train each model sequentially
SUCCESS_COUNT=0
TOTAL_COUNT=${#MODELS[@]}

for model_spec in "${MODELS[@]}"; do
    IFS=':' read -ra PARTS <<< "$model_spec"
    scale="${PARTS[0]}"
    model_cfg="${PARTS[1]}"
    pretrained="${PARTS[2]}"
    batch="${PARTS[3]}"
    lr="${PARTS[4]}"
    
    echo ""
    echo "================================================================================"
    echo "[$((SUCCESS_COUNT + 1))/$TOTAL_COUNT] Starting: YOLO12-${scale^^}"
    echo "================================================================================"
    echo ""
    
    train_model "$scale" "$model_cfg" "$pretrained" "$batch" "$lr"
    
    if [ $? -eq 0 ]; then
        ((SUCCESS_COUNT++))
        echo "✅ Progress: $SUCCESS_COUNT/$TOTAL_COUNT models completed"
    else
        echo "❌ Training failed for YOLO12-${scale^^}"
        echo "   Continue with next model? (y/n)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Training aborted."
            exit 1
        fi
    fi
    
    echo ""
    sleep 5  # Brief pause between models
done

# ================================================================================================
# Summary
# ================================================================================================

echo ""
echo "================================================================================"
echo "Training Summary"
echo "================================================================================"
echo "Completed: $SUCCESS_COUNT/$TOTAL_COUNT models"
echo ""

if [ $SUCCESS_COUNT -eq $TOTAL_COUNT ]; then
    echo "✅ All models trained successfully!"
    echo ""
    echo "Results locations:"
    for model_spec in "${MODELS[@]}"; do
        IFS=':' read -ra PARTS <<< "$model_spec"
        scale="${PARTS[0]}"
        echo "  - YOLO12-${scale^^}: runs/train/phase3_channelc2f_${scale}/weights/best.pt"
    done
    echo ""
    echo "Next steps:"
    echo "  1. Validate all models:"
    echo "     ./validate_all_phase3.sh"
    echo ""
    echo "  2. Compare with RemDet:"
    echo "     python compare_with_remdet.py"
    echo ""
    echo "  3. Generate paper tables:"
    echo "     python generate_tables.py --phase 3"
else
    echo "⚠️  Some models failed to train. Please check logs above."
fi

echo "================================================================================"
