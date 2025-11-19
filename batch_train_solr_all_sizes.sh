#!/bin/bash
# 批量训练所有尺寸的YOLO12-RGBD模型 (with SOLR loss)
# 用途: 对比不同模型尺寸 (n/s/m/l/x) 与RemDet的性能差异
# 
# 使用方法:
#   bash batch_train_solr_all_sizes.sh
#
# 或分别运行:
#   bash batch_train_solr_all_sizes.sh n  # 只训练nano
#   bash batch_train_solr_all_sizes.sh s  # 只训练small
#   bash batch_train_solr_all_sizes.sh m  # 只训练medium

set -e  # 遇到错误立即退出

# ================================================================================================
# 配置参数
# ================================================================================================
DATA_YAML="data/visdrone-rgbd.yaml"  # 数据集配置文件
EPOCHS=300                           # 训练轮数
DEVICE="0"                           # GPU设备号 (多卡用 "0,1,2,3")
PROJECT="runs/train_solr"            # 输出目录
BASE_BATCH=16                        # 基准batch size (会根据模型大小自动调整)

# SOLR权重配置
SMALL_WEIGHT=2.5    # 小目标权重
MEDIUM_WEIGHT=2.0   # 中等目标权重 (关键参数,针对RemDet差距)
LARGE_WEIGHT=1.0    # 大目标权重

# ================================================================================================
# 模型配置 (size: batch_size)
# ================================================================================================
declare -A MODEL_CONFIGS=(
    ["n"]="32"   # nano:   ~3M params,  ~8G FLOPs  → batch=32
    ["s"]="16"   # small:  ~11M params, ~46G FLOPs → batch=16
    ["m"]="8"    # medium: ~22M params, ~92G FLOPs → batch=8
    ["l"]="4"    # large:  ~44M params, ~184G FLOPs → batch=4
    ["x"]="2"    # xlarge: ~66M params, ~276G FLOPs → batch=2
)

# RemDet对标表 (用于对比)
declare -A REMDET_TARGETS=(
    ["n"]="RemDet-Tiny (AP@0.5: 37.1%, AP_m: 33.0%)"
    ["s"]="RemDet-S (AP@0.5: 42.3%, AP_m: 38.5%)"
    ["m"]="RemDet-M (AP@0.5: 45.0%, AP_m: 41.2%)"
    ["l"]="RemDet-L (AP@0.5: 47.4%, AP_m: 43.6%)"
    ["x"]="RemDet-X (AP@0.5: 48.3%, AP_m: 44.8%)"
)

# ================================================================================================
# 辅助函数
# ================================================================================================
print_header() {
    echo ""
    echo "================================================================================================"
    echo "$1"
    echo "================================================================================================"
    echo ""
}

print_info() {
    echo "ℹ️  $1"
}

print_success() {
    echo "✅ $1"
}

print_error() {
    echo "❌ $1"
}

# ================================================================================================
# 训练函数
# ================================================================================================
train_model() {
    local size=$1
    local batch=${MODEL_CONFIGS[$size]}
    local name="solr_${size}_300ep"
    local target=${REMDET_TARGETS[$size]}
    
    print_header "Training YOLO12-RGBD-${size^^} with SOLR"
    
    print_info "Configuration:"
    print_info "  Model size:    ${size} (batch=${batch})"
    print_info "  Target:        ${target}"
    print_info "  SOLR weights:  small=${SMALL_WEIGHT}x, medium=${MEDIUM_WEIGHT}x, large=${LARGE_WEIGHT}x"
    print_info "  Epochs:        ${EPOCHS}"
    print_info "  Device:        ${DEVICE}"
    print_info "  Output:        ${PROJECT}/${name}"
    echo ""
    
    # 开始训练
    print_info "Starting training at $(date '+%Y-%m-%d %H:%M:%S')..."
    
    python train_depth_solr.py \
        --data "${DATA_YAML}" \
        --cfg "${size}" \
        --epochs ${EPOCHS} \
        --batch ${batch} \
        --device "${DEVICE}" \
        --small_weight ${SMALL_WEIGHT} \
        --medium_weight ${MEDIUM_WEIGHT} \
        --large_weight ${LARGE_WEIGHT} \
        --optimizer SGD \
        --lr0 0.01 \
        --momentum 0.937 \
        --weight_decay 0.0005 \
        --mosaic 1.0 \
        --mixup 0.15 \
        --close_mosaic 10 \
        --amp \
        --project "${PROJECT}" \
        --name "${name}" \
        --exist_ok
    
    # 检查训练结果
    if [ $? -eq 0 ]; then
        print_success "Training completed successfully!"
        print_info "Results saved to: ${PROJECT}/${name}"
        print_info "Finished at $(date '+%Y-%m-%d %H:%M:%S')"
        
        # 显示最佳mAP (如果results.txt存在)
        local results_file="${PROJECT}/${name}/results.txt"
        if [ -f "${results_file}" ]; then
            local best_map=$(tail -1 "${results_file}" | awk '{print $7}')  # mAP@0.5
            local best_map50_95=$(tail -1 "${results_file}" | awk '{print $8}')  # mAP@0.5:0.95
            print_success "Best mAP@0.5:     ${best_map}"
            print_success "Best mAP@0.5:0.95: ${best_map50_95}"
        fi
    else
        print_error "Training failed for size ${size}!"
        exit 1
    fi
    
    echo ""
}

# ================================================================================================
# 主程序
# ================================================================================================
main() {
    print_header "🚀 YOLO12-RGBD Multi-Size Training with SOLR"
    
    # 检查数据集文件
    if [ ! -f "${DATA_YAML}" ]; then
        print_error "Dataset config not found: ${DATA_YAML}"
        print_info "Please check the path and try again."
        exit 1
    fi
    print_success "Dataset config found: ${DATA_YAML}"
    
    # 确定要训练的模型尺寸
    if [ $# -eq 0 ]; then
        # 未指定参数,训练所有尺寸
        SIZES_TO_TRAIN=("n" "s" "m" "l" "x")
        print_info "No size specified, will train all sizes: n, s, m, l, x"
        print_info "Estimated total time: ~14-16 hours (on RTX 4090)"
        echo ""
        read -p "Continue? [y/N] " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Training cancelled."
            exit 0
        fi
    else
        # 只训练指定的尺寸
        SIZES_TO_TRAIN=("$@")
        print_info "Will train sizes: ${SIZES_TO_TRAIN[@]}"
    fi
    
    # 开始批量训练
    START_TIME=$(date +%s)
    
    for size in "${SIZES_TO_TRAIN[@]}"; do
        # 验证尺寸参数
        if [[ ! " n s m l x " =~ " ${size} " ]]; then
            print_error "Invalid model size: ${size} (must be n/s/m/l/x)"
            continue
        fi
        
        # 训练当前尺寸
        train_model "${size}"
        
        # 训练间隔 (避免GPU过热)
        if [ "${size}" != "${SIZES_TO_TRAIN[-1]}" ]; then
            print_info "Cooling down for 60 seconds before next training..."
            sleep 60
        fi
    done
    
    # 计算总训练时间
    END_TIME=$(date +%s)
    TOTAL_TIME=$((END_TIME - START_TIME))
    HOURS=$((TOTAL_TIME / 3600))
    MINUTES=$(((TOTAL_TIME % 3600) / 60))
    
    print_header "🎉 All Training Completed!"
    print_success "Total time: ${HOURS}h ${MINUTES}m"
    print_info "Results directory: ${PROJECT}/"
    
    # 生成结果对比表
    print_header "📊 Results Summary"
    echo ""
    printf "%-8s %-12s %-12s %-12s %-40s\n" "Model" "mAP@0.5" "mAP@0.5:0.95" "Best Epoch" "Target (RemDet)"
    echo "--------------------------------------------------------------------------------------------"
    
    for size in "${SIZES_TO_TRAIN[@]}"; do
        local name="solr_${size}_300ep"
        local results_file="${PROJECT}/${name}/results.txt"
        
        if [ -f "${results_file}" ]; then
            local best_map=$(tail -1 "${results_file}" | awk '{print $7}')
            local best_map50_95=$(tail -1 "${results_file}" | awk '{print $8}')
            local best_epoch=$(tail -1 "${results_file}" | awk '{print $1}')
            local target=${REMDET_TARGETS[$size]}
            
            printf "%-8s %-12s %-12s %-12s %-40s\n" \
                "${size^^}" "${best_map}" "${best_map50_95}" "${best_epoch}" "${target}"
        else
            printf "%-8s %-12s %-12s %-12s %-40s\n" \
                "${size^^}" "N/A" "N/A" "N/A" "${REMDET_TARGETS[$size]}"
        fi
    done
    
    echo ""
    print_info "Next steps:"
    print_info "  1. Run COCO evaluation: python val_coco_eval.py --weights ${PROJECT}/solr_s_300ep/weights/best.pt"
    print_info "  2. Compare with RemDet benchmarks"
    print_info "  3. Analyze which size achieves best performance/efficiency trade-off"
    echo ""
}

# 运行主程序
main "$@"
