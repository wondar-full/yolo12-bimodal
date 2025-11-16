#!/bin/bash
# =====================================================================
# 批量训练脚本: YOLO12 n/s/m/l/x 所有尺寸
# =====================================================================
# 用途: 依次训练5个不同尺寸的RGB-D模型,对标RemDet全系列
# 预计总耗时: 2-3周 (单卡RTX 4090)
# =====================================================================

# 数据集配置
DATA_CONFIG="data/uav-joint-rgbd.yaml"
MODEL_CONFIG="ultralytics/cfg/models/12/yolo12-rgbd-v2.1-universal.yaml"
DEVICE="0"
EPOCHS=300

# 训练根目录
PROJECT_DIR="runs/train"

# =====================================================================
# 训练配置 (根据显存调整batch size)
# =====================================================================
declare -A BATCH_SIZES
BATCH_SIZES[n]=32  # YOLOv12-N: 小模型,大batch
BATCH_SIZES[s]=16  # YOLOv12-S: 标准配置
BATCH_SIZES[m]=8   # YOLOv12-M: 中等batch
BATCH_SIZES[l]=4   # YOLOv12-L: 小batch
BATCH_SIZES[x]=2   # YOLOv12-X: 最小batch (24GB显存极限)

# =====================================================================
# 函数: 训练单个模型
# =====================================================================
train_model() {
    local size=$1
    local batch=${BATCH_SIZES[$size]}
    local name="rgbd_v2.1_${size}_joint_300ep"
    
    echo ""
    echo "======================================================================"
    echo "  开始训练: YOLO12-${size^^} (batch=${batch})"
    echo "======================================================================"
    echo "  预计耗时: $(get_estimated_time $size)"
    echo "  输出目录: ${PROJECT_DIR}/${name}"
    echo "======================================================================"
    echo ""
    
    python train_uav_joint.py \
        --model ${MODEL_CONFIG} \
        --cfg ${size} \
        --data ${DATA_CONFIG} \
        --epochs ${EPOCHS} \
        --batch ${batch} \
        --imgsz 640 \
        --device ${DEVICE} \
        --workers 8 \
        --project ${PROJECT_DIR} \
        --name ${name} \
        --save_period 50 \
        --patience 100 \
        --exist_ok
    
    # 检查训练是否成功
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ YOLO12-${size^^} 训练完成!"
        echo "   最佳权重: ${PROJECT_DIR}/${name}/weights/best.pt"
        echo ""
        
        # 立即验证
        echo "开始验证 YOLO12-${size^^}..."
        python val_uav_joint.py \
            --weights ${PROJECT_DIR}/${name}/weights/best.pt \
            --data ${DATA_CONFIG} \
            --device ${DEVICE} \
            --batch 16 \
            --project runs/val \
            --name visdrone_${size}_val
        
        echo ""
        echo "✅ YOLO12-${size^^} 验证完成!"
        echo "   结果摘要: runs/val/visdrone_${size}_val/metrics/val_summary.json"
        echo ""
    else
        echo ""
        echo "❌ YOLO12-${size^^} 训练失败!"
        echo "   请检查日志: ${PROJECT_DIR}/${name}/"
        echo ""
        exit 1
    fi
}

# =====================================================================
# 函数: 预估训练时间
# =====================================================================
get_estimated_time() {
    local size=$1
    case $size in
        n) echo "1-2天" ;;
        s) echo "2-3天" ;;
        m) echo "3-4天" ;;
        l) echo "4-5天" ;;
        x) echo "5-6天" ;;
    esac
}

# =====================================================================
# 主流程
# =====================================================================
main() {
    echo "======================================================================"
    echo "  YOLO12 RGB-D 全系列训练 (n/s/m/l/x)"
    echo "======================================================================"
    echo "  数据集: VisDrone + UAVDT (29,729 训练图像)"
    echo "  Epochs: ${EPOCHS}"
    echo "  设备: GPU ${DEVICE}"
    echo "  预计总耗时: 2-3周"
    echo "======================================================================"
    echo ""
    
    # 确认是否继续
    read -p "确认开始训练? (y/n): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "已取消训练"
        exit 0
    fi
    
    # 依次训练5个模型
    # 建议顺序: s → m → x → n → l (按重要性排序)
    
    echo ""
    echo "======================================================================"
    echo "  训练顺序: s → m → x → n → l"
    echo "  原因: s是主力模型,优先验证; m/x用于论文对比; n用于部署; l可选"
    echo "======================================================================"
    echo ""
    
    # 1. YOLOv12-S (主力模型,对标RemDet-S)
    train_model "s"
    
    # 2. YOLOv12-M (性价比模型,对标RemDet-M)
    train_model "m"
    
    # 3. YOLOv12-X (终极性能,对标RemDet-X)
    train_model "x"
    
    # 4. YOLOv12-N (轻量级,对标RemDet-Tiny)
    train_model "n"
    
    # 5. YOLOv12-L (可选,对标RemDet-L)
    read -p "是否训练 YOLO12-L? (y/n): " train_l
    if [ "$train_l" == "y" ] || [ "$train_l" == "Y" ]; then
        train_model "l"
    else
        echo "跳过 YOLO12-L 训练"
    fi
    
    # 训练完成总结
    echo ""
    echo "======================================================================"
    echo "  🎉 所有模型训练完成!"
    echo "======================================================================"
    echo ""
    echo "结果对比:"
    echo "  YOLO12-N vs RemDet-Tiny: runs/val/visdrone_n_val/metrics/val_summary.json"
    echo "  YOLO12-S vs RemDet-S:    runs/val/visdrone_s_val/metrics/val_summary.json"
    echo "  YOLO12-M vs RemDet-M:    runs/val/visdrone_m_val/metrics/val_summary.json"
    echo "  YOLO12-X vs RemDet-X:    runs/val/visdrone_x_val/metrics/val_summary.json"
    echo ""
    echo "下一步:"
    echo "  1. 查看所有val_summary.json,对比RemDet基线"
    echo "  2. 生成性能对比表格 (用于论文)"
    echo "  3. 分析哪个尺寸的RGB-D优势最明显"
    echo "======================================================================"
}

# 运行主流程
main

# =====================================================================
# 使用说明
# =====================================================================
# 1. 给脚本添加执行权限:
#    chmod +x batch_train_all_sizes.sh
#
# 2. 运行脚本:
#    nohup ./batch_train_all_sizes.sh > batch_train.log 2>&1 &
#
# 3. 查看进度:
#    tail -f batch_train.log
#
# 4. 如果中断后恢复,可以注释掉已完成的模型,只训练剩余的
