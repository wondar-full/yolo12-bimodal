#!/usr/bin/env python3
"""
Training Progress Monitor
实时监控训练进度和RGB-D融合效果

Usage:
    python monitor_training.py --results runs/train/rgbd_v1_full/results.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_curves(csv_path):
    """绘制训练曲线"""
    df = pd.read_csv(csv_path)
    
    # 去除空格
    df.columns = df.columns.str.strip()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('YOLOv12-RGBD Training Progress', fontsize=16, fontweight='bold')
    
    epochs = df['epoch']
    
    # 1. mAP curves
    axes[0, 0].plot(epochs, df['metrics/mAP50(B)'] * 100, label='mAP@0.5', linewidth=2, color='blue')
    axes[0, 0].plot(epochs, df['metrics/mAP50-95(B)'] * 100, label='mAP@0.5:0.95', linewidth=2, color='orange')
    axes[0, 0].axhline(y=45.2, color='red', linestyle='--', label='RemDet-X Target (45.2%)')
    axes[0, 0].axhline(y=41, color='green', linestyle='--', label='Phase 1 Target (41%)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('mAP (%)')
    axes[0, 0].set_title('Mean Average Precision')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Precision & Recall
    axes[0, 1].plot(epochs, df['metrics/precision(B)'] * 100, label='Precision', linewidth=2, color='purple')
    axes[0, 1].plot(epochs, df['metrics/recall(B)'] * 100, label='Recall', linewidth=2, color='brown')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Percentage (%)')
    axes[0, 1].set_title('Precision & Recall')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Loss curves
    axes[0, 2].plot(epochs, df['train/box_loss'], label='Box Loss', linewidth=2)
    axes[0, 2].plot(epochs, df['train/cls_loss'], label='Cls Loss', linewidth=2)
    axes[0, 2].plot(epochs, df['train/dfl_loss'], label='DFL Loss', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].set_title('Training Losses')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Validation losses
    axes[1, 0].plot(epochs, df['val/box_loss'], label='Val Box Loss', linewidth=2, color='green')
    axes[1, 0].plot(epochs, df['val/cls_loss'], label='Val Cls Loss', linewidth=2, color='red')
    axes[1, 0].plot(epochs, df['val/dfl_loss'], label='Val DFL Loss', linewidth=2, color='blue')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Validation Losses')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Learning rate
    axes[1, 1].plot(epochs, df['lr/pg0'], label='LR pg0', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    # 6. Performance summary
    latest = df.iloc[-1]
    summary_text = f"""
    Latest Metrics (Epoch {int(latest['epoch'])}):
    
    mAP@0.5:      {latest['metrics/mAP50(B)']*100:.2f}%
    mAP@0.5:0.95: {latest['metrics/mAP50-95(B)']*100:.2f}%
    Precision:    {latest['metrics/precision(B)']*100:.2f}%
    Recall:       {latest['metrics/recall(B)']*100:.2f}%
    
    Gap to RemDet-X:
    mAP@0.5: {45.2 - latest['metrics/mAP50(B)']*100:.2f}%
    
    Train Time: {latest['time']/60:.1f} min
    """
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                     verticalalignment='center', transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # 保存图片
    save_path = Path(csv_path).parent / 'training_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved training curves to: {save_path}")
    
    plt.show()


def print_summary(csv_path):
    """打印训练摘要"""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    latest = df.iloc[-1]
    best_map50 = df['metrics/mAP50(B)'].max()
    best_epoch = df.loc[df['metrics/mAP50(B)'].idxmax(), 'epoch']
    
    print("=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"Total Epochs:     {int(latest['epoch'])}")
    print(f"Training Time:    {latest['time']/3600:.2f} hours")
    print()
    print(f"Latest mAP@0.5:      {latest['metrics/mAP50(B)']*100:.2f}%")
    print(f"Latest mAP@0.5:0.95: {latest['metrics/mAP50-95(B)']*100:.2f}%")
    print(f"Latest Precision:    {latest['metrics/precision(B)']*100:.2f}%")
    print(f"Latest Recall:       {latest['metrics/recall(B)']*100:.2f}%")
    print()
    print(f"Best mAP@0.5:        {best_map50*100:.2f}% (Epoch {int(best_epoch)})")
    print()
    print("Comparison with Targets:")
    print(f"  Phase 1 Target (41%):    {latest['metrics/mAP50(B)']*100 - 41:+.2f}%")
    print(f"  RemDet-X Target (45.2%): {latest['metrics/mAP50(B)']*100 - 45.2:+.2f}%")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, required=True, help='Path to results.csv')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    
    args = parser.parse_args()
    
    csv_path = Path(args.results)
    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        return
    
    print_summary(csv_path)
    
    if not args.no_plot:
        plot_training_curves(csv_path)


if __name__ == "__main__":
    main()
