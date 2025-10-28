#!/bin/bash
# Phase 3: Validation Script for All Model Scales
# 验证所有尺度的 ChannelC2f 模型

# ================================================================================================
# Configuration
# ================================================================================================

CUDA_DEVICE=6
DATA_CFG="data/visdrone-rgbd.yaml"
BATCH_SIZE=1  # Validation batch size
RESULTS_DIR="results/phase3_validation"

# ================================================================================================
# Model Paths
# ================================================================================================

declare -A MODEL_PATHS=(
    ["n"]="runs/train/phase3_channelc2f_n/weights/best.pt"
    ["s"]="runs/train/phase3_channelc2f_s/weights/best.pt"
    ["m"]="runs/train/phase3_channelc2f_m/weights/best.pt"
    ["l"]="runs/train/phase3_channelc2f_l/weights/best.pt"
    ["x"]="runs/train/phase3_channelc2f_x/weights/best.pt"
)

# ================================================================================================
# Functions
# ================================================================================================

validate_model() {
    local scale=$1
    local model_path=$2
    
    echo "================================================================================"
    echo "Validating YOLO12-${scale^^} ChannelC2f"
    echo "================================================================================"
    echo "Model: $model_path"
    echo ""
    
    if [ ! -f "$model_path" ]; then
        echo "❌ Model not found: $model_path"
        return 1
    fi
    
    # Run validation
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python val_depeth.py \
        --model "$model_path" \
        --data "$DATA_CFG" \
        --batch $BATCH_SIZE \
        --name "phase3_val_${scale}" \
        --project "$RESULTS_DIR" \
        --exist-ok
    
    if [ $? -eq 0 ]; then
        echo "✅ YOLO12-${scale^^} validation completed successfully!"
        return 0
    else
        echo "❌ YOLO12-${scale^^} validation failed!"
        return 1
    fi
}

extract_metrics() {
    local scale=$1
    local log_file="${RESULTS_DIR}/phase3_val_${scale}/results.txt"
    
    if [ -f "$log_file" ]; then
        echo "Results for YOLO12-${scale^^}:"
        grep -E "(mAP50|mAP50-95|Precision|Recall|Small|Medium|Large)" "$log_file" || echo "  (metrics not found in log)"
    else
        echo "Log file not found: $log_file"
    fi
}

# ================================================================================================
# Main Validation Loop
# ================================================================================================

echo "================================================================================"
echo "Phase 3: Batch Validation - All Model Scales"
echo "================================================================================"
echo ""

mkdir -p "$RESULTS_DIR"

# Validate each model
SUCCESS_COUNT=0
TOTAL_COUNT=${#MODEL_PATHS[@]}

for scale in n s m l x; do
    model_path="${MODEL_PATHS[$scale]}"
    
    echo ""
    echo "[$((SUCCESS_COUNT + 1))/$TOTAL_COUNT] Validating: YOLO12-${scale^^}"
    echo ""
    
    validate_model "$scale" "$model_path"
    
    if [ $? -eq 0 ]; then
        ((SUCCESS_COUNT++))
    fi
    
    echo ""
    sleep 2
done

# ================================================================================================
# Results Summary
# ================================================================================================

echo ""
echo "================================================================================"
echo "Validation Summary"
echo "================================================================================"
echo "Completed: $SUCCESS_COUNT/$TOTAL_COUNT models"
echo ""

if [ $SUCCESS_COUNT -gt 0 ]; then
    echo "Detailed Results:"
    echo "--------------------------------------------------------------------------------"
    for scale in n s m l x; do
        echo ""
        extract_metrics "$scale"
    done
    echo "--------------------------------------------------------------------------------"
fi

echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Next steps:"
echo "  1. Analyze medium object performance:"
echo "     python analyze_medium_performance.py --phase 3"
echo ""
echo "  2. Compare with Phase 1 baseline:"
echo "     python compare_phases.py --baseline phase1 --current phase3"
echo ""
echo "  3. Generate RemDet comparison table:"
echo "     python compare_with_remdet.py --results $RESULTS_DIR"
echo ""
echo "================================================================================"
