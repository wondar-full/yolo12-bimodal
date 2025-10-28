"""
Phase 3 批量训练 - 上传检查清单
Upload Checklist for Server Deployment

使用说明：
  1. 逐项检查并上传文件
  2. 在服务器上运行验证脚本
  3. 确认所有检查通过后开始训练
"""

# ================================================================================================
# 📋 文件上传清单 (12 个核心文件 + 2 个脚本)
# ================================================================================================

UPLOAD_CHECKLIST = {
    "核心实现文件 (4)": [
        {
            "local": "ultralytics/nn/modules/block.py",
            "remote": "/data2/user/2024/lzy/yolo12-bimodal/ultralytics/nn/modules/block.py",
            "description": "ChannelAttention + ChannelC2f 实现",
            "size": "~1000 lines",
            "critical": True,
        },
        {
            "local": "ultralytics/nn/modules/__init__.py",
            "remote": "/data2/user/2024/lzy/yolo12-bimodal/ultralytics/nn/modules/__init__.py",
            "description": "模块导出 (ChannelC2f)",
            "size": "~100 lines",
            "critical": True,
        },
        {
            "local": "ultralytics/nn/tasks.py",
            "remote": "/data2/user/2024/lzy/yolo12-bimodal/ultralytics/nn/tasks.py",
            "description": "模型解析逻辑 (base_modules + repeat_modules)",
            "size": "~3000 lines",
            "critical": True,
        },
    ],
    
    "多尺度配置文件 (5)": [
        {
            "local": "ultralytics/cfg/models/12/yolo12n-rgbd-channelc2f.yaml",
            "remote": "/data2/user/2024/lzy/yolo12-bimodal/ultralytics/cfg/models/12/yolo12n-rgbd-channelc2f.yaml",
            "description": "Nano 模型 (对标 RemDet-Tiny, ~2.5M params)",
            "size": "~200 lines",
            "critical": True,
        },
        {
            "local": "ultralytics/cfg/models/12/yolo12s-rgbd-channelc2f.yaml",
            "remote": "/data2/user/2024/lzy/yolo12-bimodal/ultralytics/cfg/models/12/yolo12s-rgbd-channelc2f.yaml",
            "description": "Small 模型 (对标 RemDet-S, ~9.5M params)",
            "size": "~200 lines",
            "critical": True,
        },
        {
            "local": "ultralytics/cfg/models/12/yolo12m-rgbd-channelc2f.yaml",
            "remote": "/data2/user/2024/lzy/yolo12-bimodal/ultralytics/cfg/models/12/yolo12m-rgbd-channelc2f.yaml",
            "description": "Medium 模型 (对标 RemDet-M, ~20M params)",
            "size": "~200 lines",
            "critical": True,
        },
        {
            "local": "ultralytics/cfg/models/12/yolo12l-rgbd-channelc2f.yaml",
            "remote": "/data2/user/2024/lzy/yolo12-bimodal/ultralytics/cfg/models/12/yolo12l-rgbd-channelc2f.yaml",
            "description": "Large 模型 (对标 RemDet-L, ~40M params)",
            "size": "~200 lines",
            "critical": True,
        },
        {
            "local": "ultralytics/cfg/models/12/yolo12x-rgbd-channelc2f.yaml",
            "remote": "/data2/user/2024/lzy/yolo12-bimodal/ultralytics/cfg/models/12/yolo12x-rgbd-channelc2f.yaml",
            "description": "XLarge 模型 (对标 RemDet-X, ~60M params)",
            "size": "~200 lines",
            "critical": True,
        },
    ],
    
    "训练与验证脚本 (3)": [
        {
            "local": "train_phase3.py",
            "remote": "/data2/user/2024/lzy/yolo12-bimodal/train_phase3.py",
            "description": "训练脚本 (含预训练加载)",
            "size": "~300 lines",
            "critical": True,
        },
        {
            "local": "verify_phase3.py",
            "remote": "/data2/user/2024/lzy/yolo12-bimodal/verify_phase3.py",
            "description": "部署验证脚本 (8 checks)",
            "size": "~200 lines",
            "critical": True,
        },
        {
            "local": "test_phase3.py",
            "remote": "/data2/user/2024/lzy/yolo12-bimodal/test_phase3.py",
            "description": "模型构建测试",
            "size": "~150 lines",
            "critical": True,
        },
    ],
    
    "批处理脚本 (2, 可选)": [
        {
            "local": "train_all_scales.sh",
            "remote": "/data2/user/2024/lzy/yolo12-bimodal/train_all_scales.sh",
            "description": "批量训练所有尺度",
            "size": "~200 lines",
            "critical": False,
        },
        {
            "local": "validate_all_phase3.sh",
            "remote": "/data2/user/2024/lzy/yolo12-bimodal/validate_all_phase3.sh",
            "description": "批量验证所有尺度",
            "size": "~150 lines",
            "critical": False,
        },
    ],
}

# ================================================================================================
# 📤 PowerShell 上传命令 (复制粘贴到本地终端)
# ================================================================================================

UPLOAD_COMMANDS = """
# 切换到项目目录
cd f:\\CV\\Paper\\yoloDepth\\yoloDepth

# ================================================================================================
# 1. 核心实现文件 (4 files)
# ================================================================================================

scp ultralytics/nn/modules/block.py ubuntu@10.16.62.111:/data2/user/2024/lzy/yolo12-bimodal/ultralytics/nn/modules/

scp ultralytics/nn/modules/__init__.py ubuntu@10.16.62.111:/data2/user/2024/lzy/yolo12-bimodal/ultralytics/nn/modules/

scp ultralytics/nn/tasks.py ubuntu@10.16.62.111:/data2/user/2024/lzy/yolo12-bimodal/ultralytics/nn/

# ================================================================================================
# 2. 多尺度配置文件 (5 files)
# ================================================================================================

scp ultralytics/cfg/models/12/yolo12n-rgbd-channelc2f.yaml ubuntu@10.16.62.111:/data2/user/2024/lzy/yolo12-bimodal/ultralytics/cfg/models/12/

scp ultralytics/cfg/models/12/yolo12s-rgbd-channelc2f.yaml ubuntu@10.16.62.111:/data2/user/2024/lzy/yolo12-bimodal/ultralytics/cfg/models/12/

scp ultralytics/cfg/models/12/yolo12m-rgbd-channelc2f.yaml ubuntu@10.16.62.111:/data2/user/2024/lzy/yolo12-bimodal/ultralytics/cfg/models/12/

scp ultralytics/cfg/models/12/yolo12l-rgbd-channelc2f.yaml ubuntu@10.16.62.111:/data2/user/2024/lzy/yolo12-bimodal/ultralytics/cfg/models/12/

scp ultralytics/cfg/models/12/yolo12x-rgbd-channelc2f.yaml ubuntu@10.16.62.111:/data2/user/2024/lzy/yolo12-bimodal/ultralytics/cfg/models/12/

# ================================================================================================
# 3. 训练与验证脚本 (3 files)
# ================================================================================================

scp train_phase3.py ubuntu@10.16.62.111:/data2/user/2024/lzy/yolo12-bimodal/

scp verify_phase3.py ubuntu@10.16.62.111:/data2/user/2024/lzy/yolo12-bimodal/

scp test_phase3.py ubuntu@10.16.62.111:/data2/user/2024/lzy/yolo12-bimodal/

# ================================================================================================
# 4. 批处理脚本 (2 files, 可选)
# ================================================================================================

scp train_all_scales.sh ubuntu@10.16.62.111:/data2/user/2024/lzy/yolo12-bimodal/

scp validate_all_phase3.sh ubuntu@10.16.62.111:/data2/user/2024/lzy/yolo12-bimodal/

# ================================================================================================
# 上传完成提示
# ================================================================================================

Write-Host "✅ All files uploaded!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps on server:" -ForegroundColor Yellow
Write-Host "  1. ssh ubuntu@10.16.62.111"
Write-Host "  2. cd /data2/user/2024/lzy/yolo12-bimodal"
Write-Host "  3. conda activate lzy-yolo12"
Write-Host "  4. python verify_phase3.py"
Write-Host "  5. python test_phase3.py"
Write-Host "  6. chmod +x train_all_scales.sh validate_all_phase3.sh"
Write-Host "  7. ./train_all_scales.sh"
"""

# ================================================================================================
# 🔍 服务器验证命令
# ================================================================================================

SERVER_VERIFICATION = """
# SSH 登录服务器
ssh ubuntu@10.16.62.111

# 切换到项目目录
cd /data2/user/2024/lzy/yolo12-bimodal

# 激活环境
conda activate lzy-yolo12

# ================================================================================================
# Step 1: 运行验证脚本 (8 checks)
# ================================================================================================

python verify_phase3.py

# 预期输出:
# ✅ Check 1/8: block.py exists
# ✅ Check 2/8: ChannelAttention class complete
# ✅ Check 3/8: ChannelC2f class complete
# ✅ Check 4/8: block.py __all__ exports
# ✅ Check 5/8: modules/__init__.py exports
# ✅ Check 6/8: tasks.py imports
# ✅ Check 7/8: YAML config exists
# ✅ Check 8/8: Python import test
# 
# ================================================================================
# ✅ All 8 checks passed! Phase 3 deployment verified.
# ================================================================================

# ================================================================================================
# Step 2: 运行模型构建测试
# ================================================================================================

python test_phase3.py

# 预期输出:
# ================================================================================
# Phase 3 ChannelC2f Model Construction Test
# ================================================================================
# 
# 1️⃣ Building model from YAML...
# ✅ Model built successfully
# 
# 2️⃣ Testing forward pass...
# ✅ Forward pass successful
# 
# 3️⃣ Checking parameter count...
# ✅ Parameters: 9,518,124 (~9.52M, +1.4% vs Phase 1)
# 
# 4️⃣ Verifying ChannelAttention integration...
# ✅ ChannelAttention found in model.model.6.ca
# 
# 5️⃣ Comparing with Phase 1 baseline...
# ✅ Phase 3 adds channel attention to P4 layer
# 
# ================================================================================
# ✅ All tests passed! Model ready for training.
# ================================================================================

# ================================================================================================
# Step 3: 准备训练脚本
# ================================================================================================

# 添加执行权限
chmod +x train_all_scales.sh validate_all_phase3.sh

# 检查预训练权重 (可选)
ls -lh yolo12*.pt
# 如果没有，可以从本地上传或从官方下载

# ================================================================================================
# Step 4: 开始训练
# ================================================================================================

# 选项 A: 批量训练所有尺度 (推荐)
./train_all_scales.sh

# 选项 B: 单独训练某个尺度 (例如 YOLO12n)
CUDA_VISIBLE_DEVICES=6 nohup python train_phase3.py \\
    --model ultralytics/cfg/models/12/yolo12n-rgbd-channelc2f.yaml \\
    --name phase3_n > logs/phase3_n.log 2>&1 &

# 监控训练日志
tail -f logs/phase3_n.log

# 或使用 tmux (推荐)
tmux new -s phase3_training
./train_all_scales.sh
# Ctrl+B, D 分离会话

# 重新连接
tmux attach -t phase3_training
"""

# ================================================================================================
# 📊 预期训练时间表
# ================================================================================================

TRAINING_SCHEDULE = {
    "yolo12n": {
        "params": "~2.5M",
        "days": 2,
        "batch_size": 32,
        "remdet_target": "RemDet-Tiny",
        "priority": "High (快速验证方案)",
    },
    "yolo12s": {
        "params": "~9.5M",
        "days": 3,
        "batch_size": 16,
        "remdet_target": "RemDet-S",
        "priority": "High (主要对比)",
    },
    "yolo12m": {
        "params": "~20M",
        "days": 5,
        "batch_size": 8,
        "remdet_target": "RemDet-M",
        "priority": "Medium",
    },
    "yolo12l": {
        "params": "~40M",
        "days": 7,
        "batch_size": 4,
        "remdet_target": "RemDet-L",
        "priority": "Medium",
    },
    "yolo12x": {
        "params": "~60M",
        "days": 10,
        "batch_size": 4,
        "remdet_target": "RemDet-X",
        "priority": "Low (可选)",
    },
}

# ================================================================================================
# 🎯 成功标准
# ================================================================================================

SUCCESS_CRITERIA = {
    "Minimum (Phase 3 有效)": {
        "medium_mAP": "≥18.0%",
        "overall_mAP": "≥45.0%",
        "improvement": "+3.7% (Medium), +0.97% (Overall)",
    },
    "Target (论文可发表)": {
        "medium_mAP": "≥20.0%",
        "overall_mAP": "≥46.0%",
        "improvement": "+5.7% (Medium), +1.97% (Overall)",
    },
    "Excellent (超越 RemDet)": {
        "medium_mAP": "≥23.0%",
        "overall_mAP": "≥47.0%",
        "improvement": "+8.7% (Medium), +2.97% (Overall)",
    },
}

# ================================================================================================
# 📋 常见问题排查
# ================================================================================================

TROUBLESHOOTING = {
    "验证失败": [
        "检查文件是否完整上传: ls -lh ultralytics/nn/modules/block.py",
        "检查 YAML 文件路径: ls ultralytics/cfg/models/12/yolo12*-rgbd-channelc2f.yaml",
        "重新运行验证: python verify_phase3.py",
    ],
    
    "模型构建失败": [
        "检查 CUDA 是否可用: python -c 'import torch; print(torch.cuda.is_available())'",
        "检查依赖版本: pip show ultralytics torch",
        "查看详细错误: python test_phase3.py 2>&1 | tee test_debug.log",
    ],
    
    "训练启动失败": [
        "检查数据集路径: ls data/visdrone-rgbd.yaml",
        "检查 GPU 可用性: nvidia-smi",
        "查看训练日志: cat logs/phase3_n.log",
        "减小 batch size: 修改 train_all_scales.sh 中的 BATCH_SIZE",
    ],
    
    "训练中断": [
        "检查 GPU 显存: watch -n 1 nvidia-smi",
        "恢复训练: model.train(resume=True)",
        "调整超参数: 降低 batch_size 或使用梯度累积",
    ],
    
    "Medium mAP 没提升": [
        "检查 ChannelAttention 是否生效: 查看模型架构输出",
        "对比 Phase 1 baseline: python compare_phases.py",
        "分析失败案例: 查看 validation 输出的预测图",
        "考虑调整 reduction 参数: 默认16，可尝试8或32",
    ],
}

# ================================================================================================
# 🚀 快速开始指南
# ================================================================================================

QUICK_START = """
# ================================================================================================
# Phase 3 快速开始 - 3 步部署
# ================================================================================================

# Step 1: 本地上传 (Windows PowerShell)
cd f:\\CV\\Paper\\yoloDepth\\yoloDepth
# 运行上述 UPLOAD_COMMANDS 中的 scp 命令

# Step 2: 服务器验证 (Linux Terminal)
ssh ubuntu@10.16.62.111
cd /data2/user/2024/lzy/yolo12-bimodal
conda activate lzy-yolo12
python verify_phase3.py && python test_phase3.py

# Step 3: 开始训练
chmod +x train_all_scales.sh
tmux new -s phase3
./train_all_scales.sh

# 分离会话: Ctrl+B, D
# 重新连接: tmux attach -t phase3

# ================================================================================================
# 监控训练进度
# ================================================================================================

# 选项 A: 查看日志
tail -f logs/phase3_n.log

# 选项 B: TensorBoard (如果启用)
tensorboard --logdir runs/train --port 6006
# 本地浏览器: ssh -L 6006:localhost:6006 ubuntu@10.16.62.111

# 选项 C: 定期检查 mAP
grep "mAP50-95" logs/phase3_n.log

# ================================================================================================
# 预期时间线
# ================================================================================================

# Day 0-2:   YOLO12n 训练 (最快验证方案是否有效)
# Day 2-5:   YOLO12s 训练 (主要对比模型)
# Day 5-10:  YOLO12m 训练
# Day 10-17: YOLO12l 训练
# Day 17-27: YOLO12x 训练

# 如果 YOLO12n 结果不理想 (Medium mAP <18%), 可提前调整策略:
#   - 调整 ChannelAttention reduction (8, 16, 32)
#   - 增加 Layer 6 的 repeats (4 → 6)
#   - 尝试不同的融合位置

# ================================================================================================
# 成功指标检查 (以 YOLO12n 为例)
# ================================================================================================

# 训练完成后运行验证
CUDA_VISIBLE_DEVICES=6 python val_depeth.py \\
    --model runs/train/phase3_channelc2f_n/weights/best.pt

# 检查关键指标:
# 1. Medium mAP:    目标 ≥18% (baseline: 14.28%)
# 2. Medium Recall: 目标 ≥20% (baseline: 11.7%)
# 3. Overall mAP:   目标 ≥45% (baseline: 44.03%)

# 如果达到目标 → 继续训练其他尺度
# 如果未达到   → 分析原因，调整方案
"""

# ================================================================================================
# 打印使用说明
# ================================================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Phase 3 部署检查清单与快速开始指南")
    print("=" * 80)
    print()
    
    print("📋 文件上传清单:")
    print("-" * 80)
    total_files = sum(len(files) for files in UPLOAD_CHECKLIST.values())
    print(f"总计: {total_files} 个文件")
    for category, files in UPLOAD_CHECKLIST.items():
        print(f"\n{category}:")
        for file in files:
            status = "🔴 Critical" if file["critical"] else "🟡 Optional"
            print(f"  {status} {file['local']}")
            print(f"          → {file['description']}")
    print()
    
    print("=" * 80)
    print("📤 PowerShell 上传命令 (复制到本地终端)")
    print("=" * 80)
    print(UPLOAD_COMMANDS)
    
    print("=" * 80)
    print("🔍 服务器验证命令")
    print("=" * 80)
    print(SERVER_VERIFICATION)
    
    print("=" * 80)
    print("📊 训练时间表")
    print("=" * 80)
    for scale, info in TRAINING_SCHEDULE.items():
        print(f"\n{scale.upper()}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    print()
    print("=" * 80)
    print("🎯 成功标准")
    print("=" * 80)
    for level, criteria in SUCCESS_CRITERIA.items():
        print(f"\n{level}:")
        for key, value in criteria.items():
            print(f"  {key}: {value}")
    
    print()
    print("=" * 80)
    print("🚀 快速开始")
    print("=" * 80)
    print(QUICK_START)
    
    print()
    print("=" * 80)
    print("📋 常见问题排查")
    print("=" * 80)
    for issue, solutions in TROUBLESHOOTING.items():
        print(f"\n{issue}:")
        for i, solution in enumerate(solutions, 1):
            print(f"  {i}. {solution}")
    
    print()
    print("=" * 80)
    print("✅ 部署准备完成！")
    print("=" * 80)
    print()
    print("下一步: 复制上述 PowerShell 命令上传文件到服务器")
    print()
