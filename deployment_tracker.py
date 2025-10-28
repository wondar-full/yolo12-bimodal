"""
Phase 3 部署可视化流程与状态追踪

运行方式:
    python deployment_tracker.py

功能:
    1. 显示部署进度
    2. 检查文件上传状态
    3. 生成训练计划表
    4. 追踪训练进度
"""

import os
from pathlib import Path
from datetime import datetime, timedelta

# ================================================================================================
# 配置
# ================================================================================================

LOCAL_ROOT = Path("f:/CV/Paper/yoloDepth/yoloDepth")
SERVER_ROOT = "/data2/user/2024/lzy/yolo12-bimodal"

# 需要上传的文件列表
FILES_TO_UPLOAD = {
    "核心实现": [
        "ultralytics/nn/modules/block.py",
        "ultralytics/nn/modules/__init__.py",
        "ultralytics/nn/tasks.py",
    ],
    "多尺度配置": [
        "ultralytics/cfg/models/12/yolo12n-rgbd-channelc2f.yaml",
        "ultralytics/cfg/models/12/yolo12s-rgbd-channelc2f.yaml",
        "ultralytics/cfg/models/12/yolo12m-rgbd-channelc2f.yaml",
        "ultralytics/cfg/models/12/yolo12l-rgbd-channelc2f.yaml",
        "ultralytics/cfg/models/12/yolo12x-rgbd-channelc2f.yaml",
    ],
    "训练脚本": [
        "train_phase3.py",
        "verify_phase3.py",
        "test_phase3.py",
    ],
    "批处理脚本": [
        "train_all_scales.sh",
        "validate_all_phase3.sh",
    ],
}

# 训练计划
TRAINING_PLAN = {
    "n": {
        "name": "YOLO12n",
        "params": "2.5M",
        "batch_size": 32,
        "lr0": 0.001,
        "days": 2,
        "priority": "🔴 High",
        "remdet": "RemDet-Tiny",
    },
    "s": {
        "name": "YOLO12s",
        "params": "9.5M",
        "batch_size": 16,
        "lr0": 0.001,
        "days": 3,
        "priority": "🔴 High",
        "remdet": "RemDet-S",
    },
    "m": {
        "name": "YOLO12m",
        "params": "20M",
        "batch_size": 8,
        "lr0": 0.0008,
        "days": 5,
        "priority": "🟡 Medium",
        "remdet": "RemDet-M",
    },
    "l": {
        "name": "YOLO12l",
        "params": "40M",
        "batch_size": 4,
        "lr0": 0.0005,
        "days": 7,
        "priority": "🟡 Medium",
        "remdet": "RemDet-L",
    },
    "x": {
        "name": "YOLO12x",
        "params": "60M",
        "batch_size": 4,
        "lr0": 0.0005,
        "days": 10,
        "priority": "🟢 Low",
        "remdet": "RemDet-X",
    },
}

# ================================================================================================
# 可视化函数
# ================================================================================================

def print_header(title):
    """打印标题"""
    print()
    print("=" * 100)
    print(f"  {title}")
    print("=" * 100)
    print()


def print_section(title):
    """打印子标题"""
    print()
    print("-" * 100)
    print(f"  {title}")
    print("-" * 100)


def check_local_files():
    """检查本地文件是否存在"""
    print_header("Phase 3 部署状态检查")
    
    total_files = 0
    missing_files = 0
    
    for category, files in FILES_TO_UPLOAD.items():
        print_section(f"{category} ({len(files)} 个文件)")
        
        for file in files:
            total_files += 1
            file_path = LOCAL_ROOT / file
            
            if file_path.exists():
                size = file_path.stat().st_size
                size_kb = size / 1024
                print(f"  ✅ {file:<70} ({size_kb:>7.1f} KB)")
            else:
                missing_files += 1
                print(f"  ❌ {file:<70} (NOT FOUND)")
    
    print()
    print(f"总计: {total_files} 个文件")
    print(f"就绪: {total_files - missing_files} 个")
    print(f"缺失: {missing_files} 个")
    
    if missing_files > 0:
        print()
        print("⚠️  警告: 部分文件缺失，请先生成或检查路径！")
        return False
    else:
        print()
        print("✅ 所有文件准备完毕，可以开始上传！")
        return True


def print_upload_commands():
    """打印上传命令"""
    print_header("文件上传命令 (PowerShell)")
    
    print("# 切换到项目目录")
    print(f"cd {LOCAL_ROOT}")
    print()
    
    for category, files in FILES_TO_UPLOAD.items():
        print(f"# {category}")
        for file in files:
            # 使用正斜杠 (PowerShell 支持)
            local_file = file.replace("\\", "/")
            remote_file = f"{SERVER_ROOT}/{file}"
            print(f"scp {local_file} ubuntu@10.16.62.111:{remote_file}")
        print()


def print_training_plan():
    """打印训练计划表"""
    print_header("多尺度训练计划")
    
    print(f"{'模型':<12} {'参数量':<10} {'Batch':<8} {'LR0':<10} {'天数':<8} {'优先级':<15} {'对标 RemDet':<15}")
    print("-" * 100)
    
    total_days = 0
    for scale, info in TRAINING_PLAN.items():
        print(
            f"{info['name']:<12} "
            f"{info['params']:<10} "
            f"{info['batch_size']:<8} "
            f"{info['lr0']:<10} "
            f"{info['days']:<8} "
            f"{info['priority']:<15} "
            f"{info['remdet']:<15}"
        )
        total_days += info["days"]
    
    print("-" * 100)
    print(f"总训练时间: ~{total_days} 天")
    print()
    print("📅 预计时间线 (假设今天开始):")
    
    start_date = datetime.now()
    current_date = start_date
    
    for scale, info in TRAINING_PLAN.items():
        end_date = current_date + timedelta(days=info["days"])
        print(
            f"  {info['name']}: "
            f"{current_date.strftime('%m/%d')} - {end_date.strftime('%m/%d')} "
            f"({info['days']} 天) "
            f"{info['priority']}"
        )
        current_date = end_date
    
    print()
    print(f"🏁 预计完成时间: {current_date.strftime('%Y-%m-%d')}")


def print_success_criteria():
    """打印成功标准"""
    print_header("Phase 3 成功标准")
    
    criteria = {
        "Minimum (方案有效)": {
            "Medium mAP": "≥18.0% (baseline: 14.28%, +3.7%)",
            "Overall mAP": "≥45.0% (baseline: 44.03%, +0.97%)",
            "Medium Recall": "≥18.0% (baseline: 11.7%, +6.3%)",
        },
        "Target (论文可发表)": {
            "Medium mAP": "≥20.0% (baseline: 14.28%, +5.7%)",
            "Overall mAP": "≥46.0% (baseline: 44.03%, +1.97%)",
            "Medium Recall": "≥20.0% (baseline: 11.7%, +8.3%)",
        },
        "Excellent (超越 RemDet)": {
            "Medium mAP": "≥23.0% (baseline: 14.28%, +8.7%)",
            "Overall mAP": "≥47.0% (baseline: 44.03%, +2.97%)",
            "Medium Recall": "≥25.0% (baseline: 11.7%, +13.3%)",
        },
    }
    
    for level, metrics in criteria.items():
        print(f"{level}:")
        for metric, target in metrics.items():
            print(f"  - {metric}: {target}")
        print()


def print_deployment_steps():
    """打印部署步骤"""
    print_header("部署步骤 (Step-by-Step)")
    
    steps = [
        {
            "num": "1️⃣",
            "title": "本地文件检查",
            "actions": [
                "运行: python deployment_tracker.py",
                "确认所有文件存在 (13 个)",
            ],
            "status": "✅ (当前步骤)",
        },
        {
            "num": "2️⃣",
            "title": "上传文件到服务器",
            "actions": [
                "复制 PowerShell 上传命令",
                "在本地终端执行 scp 命令",
                "等待所有文件上传完成",
            ],
            "status": "⏸️ (待执行)",
        },
        {
            "num": "3️⃣",
            "title": "服务器环境验证",
            "actions": [
                "SSH 登录: ssh ubuntu@10.16.62.111",
                "切换目录: cd /data2/user/2024/lzy/yolo12-bimodal",
                "激活环境: conda activate lzy-yolo12",
                "运行验证: python verify_phase3.py",
                "预期: All 8 checks passed ✅",
            ],
            "status": "⏸️ (待执行)",
        },
        {
            "num": "4️⃣",
            "title": "模型构建测试",
            "actions": [
                "运行测试: python test_phase3.py",
                "预期: All tests passed ✅",
                "检查参数量: ~9.52M (+1.4%)",
            ],
            "status": "⏸️ (待执行)",
        },
        {
            "num": "5️⃣",
            "title": "启动 YOLO12n 训练",
            "actions": [
                "添加执行权限: chmod +x train_all_scales.sh",
                "创建 tmux 会话: tmux new -s phase3",
                "启动训练: ./train_all_scales.sh",
                "或单独训练 n: CUDA_VISIBLE_DEVICES=6 python train_phase3.py ...",
                "分离会话: Ctrl+B, D",
            ],
            "status": "⏸️ (待执行)",
        },
        {
            "num": "6️⃣",
            "title": "监控训练进度",
            "actions": [
                "查看日志: tail -f logs/phase3_n.log",
                "检查 mAP: grep 'mAP50-95' logs/phase3_n.log",
                "定期验证: CUDA_VISIBLE_DEVICES=6 python val_depeth.py ...",
            ],
            "status": "⏸️ (待执行)",
        },
        {
            "num": "7️⃣",
            "title": "验证 YOLO12n 结果",
            "actions": [
                "等待训练完成 (~2 天)",
                "运行验证: python val_depeth.py --model runs/train/phase3_channelc2f_n/weights/best.pt",
                "检查 Medium mAP: 目标 ≥18%",
                "检查 Overall mAP: 目标 ≥45%",
            ],
            "status": "⏸️ (待执行)",
        },
        {
            "num": "8️⃣",
            "title": "决定下一步",
            "actions": [
                "如果成功 (Medium mAP ≥18%) → 继续训练其他尺度",
                "如果失败 (Medium mAP <18%) → 分析原因，调整方案",
            ],
            "status": "⏸️ (待执行)",
        },
    ]
    
    for step in steps:
        print(f"{step['num']} {step['title']} - {step['status']}")
        for action in step["actions"]:
            print(f"     {action}")
        print()


def print_quick_reference():
    """打印快速参考"""
    print_header("快速参考命令")
    
    commands = {
        "本地上传": [
            "cd f:\\CV\\Paper\\yoloDepth\\yoloDepth",
            "# 复制 PowerShell 上传命令 (见上方输出)",
        ],
        "服务器登录": [
            "ssh ubuntu@10.16.62.111",
            "cd /data2/user/2024/lzy/yolo12-bimodal",
            "conda activate lzy-yolo12",
        ],
        "验证部署": [
            "python verify_phase3.py",
            "python test_phase3.py",
        ],
        "启动训练": [
            "chmod +x train_all_scales.sh validate_all_phase3.sh",
            "tmux new -s phase3",
            "./train_all_scales.sh",
            "# Ctrl+B, D (分离会话)",
        ],
        "监控训练": [
            "tail -f logs/phase3_n.log",
            "grep 'mAP50-95' logs/phase3_n.log",
            "watch -n 1 nvidia-smi",
        ],
        "验证结果": [
            "CUDA_VISIBLE_DEVICES=6 python val_depeth.py --model runs/train/phase3_channelc2f_n/weights/best.pt",
            "./validate_all_phase3.sh",
        ],
    }
    
    for category, cmds in commands.items():
        print(f"{category}:")
        for cmd in cmds:
            print(f"  {cmd}")
        print()


# ================================================================================================
# 主函数
# ================================================================================================

def main():
    """主函数"""
    # 1. 检查本地文件
    files_ready = check_local_files()
    
    # 2. 打印上传命令
    if files_ready:
        print_upload_commands()
    
    # 3. 打印训练计划
    print_training_plan()
    
    # 4. 打印成功标准
    print_success_criteria()
    
    # 5. 打印部署步骤
    print_deployment_steps()
    
    # 6. 打印快速参考
    print_quick_reference()
    
    # 7. 总结
    print_header("总结")
    
    if files_ready:
        print("✅ 所有文件准备完毕！")
        print()
        print("下一步:")
        print("  1. 复制上述 PowerShell 上传命令")
        print("  2. 在本地终端执行上传")
        print("  3. SSH 到服务器运行验证")
        print("  4. 启动 YOLO12n 训练 (优先级最高)")
        print()
        print("预计时间线:")
        print("  - Day 0:     上传 + 验证 + 启动训练")
        print("  - Day 0-2:   YOLO12n 训练")
        print("  - Day 2:     验证结果，决定是否继续")
        print("  - Day 2-27:  其他尺度训练 (如果 n 成功)")
        print()
        print("🚀 准备就绪！Good luck!")
    else:
        print("⚠️  部分文件缺失，请先检查并生成缺失文件！")
    
    print()


if __name__ == "__main__":
    main()
