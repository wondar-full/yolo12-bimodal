#!/usr/bin/env python3
"""
生成RemDet对比表格 (论文Table格式)
用法: python generate_comparison_table.py
"""

import json
from pathlib import Path
from typing import Dict, List

# RemDet基线 (从论文Table 1)
REMDET_BASELINES = {
    "RemDet-Tiny": {
        "AP@val": 20.3,
        "AP@val_50": 33.5,
        "AP@val_s": 10.2,
        "Params(M)": 3.1,
        "FLOPs(G)": 5.1,
        "Latency(ms)": 13.2,
    },
    "RemDet-S": {
        "AP@val": 25.5,
        "AP@val_50": 42.3,
        "AP@val_s": 15.9,
        "Params(M)": 5.7,
        "FLOPs(G)": 10.2,
        "Latency(ms)": 4.8,
    },
    "RemDet-M": {
        "AP@val": 27.8,
        "AP@val_50": 45.0,
        "AP@val_s": 17.6,
        "Params(M)": 8.2,
        "FLOPs(G)": 13.7,
        "Latency(ms)": 6.5,
    },
    "RemDet-L": {
        "AP@val": 29.3,
        "AP@val_50": 47.4,
        "AP@val_s": 18.7,
        "Params(M)": 8.9,
        "FLOPs(G)": 67.4,
        "Latency(ms)": 7.1,
    },
    "RemDet-X": {
        "AP@val": 29.9,
        "AP@val_50": 48.3,
        "AP@val_s": 19.5,
        "Params(M)": 9.8,
        "FLOPs(G)": 114,
        "Latency(ms)": 8.9,
    },
}

# 我们的模型验证结果路径
OUR_RESULTS_PATHS = {
    "YOLO12-N-RGBD": "runs/val/visdrone_n_val/metrics/val_summary.json",
    "YOLO12-S-RGBD": "runs/val/visdrone_s_val/metrics/val_summary.json",
    "YOLO12-M-RGBD": "runs/val/visdrone_m_val/metrics/val_summary.json",
    "YOLO12-L-RGBD": "runs/val/visdrone_l_val/metrics/val_summary.json",
    "YOLO12-X-RGBD": "runs/val/visdrone_x_val/metrics/val_summary.json",
}


def load_our_results() -> Dict:
    """加载我们的验证结果"""
    results = {}
    for name, path in OUR_RESULTS_PATHS.items():
        p = Path(path)
        if p.exists():
            with open(p) as f:
                data = json.load(f)
                results[name] = {
                    "AP@val": data["metrics"]["AP@[0.5:0.95]"],
                    "AP@val_50": data["metrics"]["AP@0.50"],
                    "AP@val_s": data["metrics"]["AP_small"],
                    "Params(M)": data["efficiency"]["params_m"],
                    "FLOPs(G)": data["efficiency"]["flops_g"],
                    "Latency(ms)": data["efficiency"]["latency_ms"],
                }
        else:
            print(f"⚠️  警告: {name} 结果文件未找到: {path}")
    return results


def print_markdown_table(our_results: Dict):
    """打印Markdown格式表格"""
    print("\n## 性能对比表 (VisDrone val set)")
    print("\n| Model | AP@val | AP@val_50 | AP@val_s | Params(M) | FLOPs(G) | Latency(ms) |")
    print("|-------|--------|-----------|----------|-----------|----------|-------------|")
    
    # RemDet基线
    for name, metrics in REMDET_BASELINES.items():
        print(f"| {name:<18} | {metrics['AP@val']:6.1f} | {metrics['AP@val_50']:9.1f} | "
              f"{metrics['AP@val_s']:8.1f} | {metrics['Params(M)']:9.1f} | "
              f"{metrics['FLOPs(G)']:8.1f} | {metrics['Latency(ms)']:11.1f} |")
    
    # 分隔线
    print("|-------|--------|-----------|----------|-----------|----------|-------------|")
    
    # 我们的结果
    for name, metrics in our_results.items():
        # 计算相对RemDet的提升
        remdet_name = name.replace("YOLO12", "RemDet").replace("-RGBD", "")
        if remdet_name in REMDET_BASELINES:
            delta_ap50 = metrics['AP@val_50'] - REMDET_BASELINES[remdet_name]['AP@val_50']
            delta_str = f" ({delta_ap50:+.1f})"
        else:
            delta_str = ""
        
        print(f"| {name:<18} | {metrics['AP@val']:6.1f} | "
              f"{metrics['AP@val_50']:9.1f}{delta_str} | "
              f"{metrics['AP@val_s']:8.1f} | {metrics['Params(M)']:9.1f} | "
              f"{metrics['FLOPs(G)']:8.1f} | {metrics['Latency(ms)']:11.1f} |")
    
    print("")


def print_latex_table(our_results: Dict):
    """打印LaTeX格式表格 (用于论文)"""
    print("\n## LaTeX 表格代码")
    print("```latex")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Comparison with RemDet on VisDrone validation set.}")
    print("\\label{tab:visdrone_comparison}")
    print("\\begin{tabular}{l|ccc|cc}")
    print("\\hline")
    print("Model & AP$^{val}$ & AP$^{val}_{50}$ & AP$^{val}_s$ & Params(M) & FLOPs(G) \\\\")
    print("\\hline")
    
    # RemDet基线
    for name, metrics in REMDET_BASELINES.items():
        print(f"{name} & {metrics['AP@val']:.1f} & {metrics['AP@val_50']:.1f} & "
              f"{metrics['AP@val_s']:.1f} & {metrics['Params(M)']:.1f} & "
              f"{metrics['FLOPs(G)']:.1f} \\\\")
    
    print("\\hline")
    
    # 我们的结果
    for name, metrics in our_results.items():
        # 加粗优于RemDet的结果
        remdet_name = name.replace("YOLO12", "RemDet").replace("-RGBD", "")
        if remdet_name in REMDET_BASELINES:
            if metrics['AP@val_50'] > REMDET_BASELINES[remdet_name]['AP@val_50']:
                ap50_str = f"\\textbf{{{metrics['AP@val_50']:.1f}}}"
            else:
                ap50_str = f"{metrics['AP@val_50']:.1f}"
        else:
            ap50_str = f"{metrics['AP@val_50']:.1f}"
        
        print(f"{name} & {metrics['AP@val']:.1f} & {ap50_str} & "
              f"{metrics['AP@val_s']:.1f} & {metrics['Params(M)']:.1f} & "
              f"{metrics['FLOPs(G)']:.1f} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    print("```\n")


def print_summary(our_results: Dict):
    """打印改进总结"""
    print("\n## 改进总结")
    print("\n| 尺寸 | 我们的AP@val_50 | RemDet AP@val_50 | 提升 | 状态 |")
    print("|------|----------------|-----------------|------|------|")
    
    model_pairs = [
        ("YOLO12-N-RGBD", "RemDet-Tiny"),
        ("YOLO12-S-RGBD", "RemDet-S"),
        ("YOLO12-M-RGBD", "RemDet-M"),
        ("YOLO12-L-RGBD", "RemDet-L"),
        ("YOLO12-X-RGBD", "RemDet-X"),
    ]
    
    wins = 0
    total = 0
    
    for our_name, remdet_name in model_pairs:
        if our_name in our_results:
            ours = our_results[our_name]['AP@val_50']
            theirs = REMDET_BASELINES[remdet_name]['AP@val_50']
            delta = ours - theirs
            status = "✅ 超越" if delta > 0 else ("⚠️  接近" if delta > -1 else "❌ 不及")
            
            if delta > 0:
                wins += 1
            total += 1
            
            print(f"| {our_name.split('-')[1]:<4} | {ours:14.1f} | {theirs:15.1f} | "
                  f"{delta:+5.1f} | {status:<6} |")
    
    print(f"\n胜率: {wins}/{total} ({wins/total*100:.0f}%)")
    
    if wins == total:
        print("🎉 完美! 所有尺寸都超越了RemDet!")
    elif wins >= total / 2:
        print("👍 不错! 多数尺寸超越了RemDet!")
    else:
        print("💪 继续努力! 可以考虑:")
        print("   - 启用SOLR损失")
        print("   - 延长训练到400 epochs")
        print("   - 调整depth融合权重")


def main():
    """主函数"""
    print("=" * 70)
    print("  RemDet性能对比表生成器")
    print("=" * 70)
    
    # 加载结果
    our_results = load_our_results()
    
    if not our_results:
        print("\n❌ 错误: 未找到任何验证结果!")
        print("\n请先运行验证脚本:")
        print("  python val_uav_joint.py --weights runs/train/.../weights/best.pt")
        return
    
    print(f"\n✅ 加载了 {len(our_results)} 个模型的结果\n")
    
    # 生成表格
    print_markdown_table(our_results)
    print_latex_table(our_results)
    print_summary(our_results)
    
    # 保存到文件
    output_file = Path("performance_comparison.md")
    with open(output_file, "w", encoding="utf-8") as f:
        import sys
        from io import StringIO
        
        # 重定向输出到字符串
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        print_markdown_table(our_results)
        print_latex_table(our_results)
        print_summary(our_results)
        
        content = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        f.write(content)
    
    print(f"\n✅ 对比表已保存到: {output_file}")
    print("\n可以直接复制到论文或GitHub README中!")


if __name__ == "__main__":
    main()
