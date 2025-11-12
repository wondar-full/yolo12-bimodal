#!/usr/bin/env python3
"""
Generate Phase 3 ChannelC2f YAML configs for all model scales (n, s, m, l, x)

对标 RemDet 的 tiny, m, l, x 模型
"""

from pathlib import Path

# Base template (yolo12s-rgbd-channelc2f.yaml 的前半部分)
HEADER_TEMPLATE = """# Ultralytics YOLO 🚀, AGPL-3.0 License
# YOLO12-{scale_upper} RGB-D ChannelC2f - Phase 3: Enhanced Medium-Scale Detection
# Created: 2025-10-28 for yoloDepth Phase 3
# Objective: Solve Medium mAP crisis (14.28% → target 20%+)
# Scale: {scale_desc}

# ================================================================================================
# Phase 3 Background
# ================================================================================================
# Problem Discovery:
#   - Medium objects占比: 45.5% (17,647个) - 最多！
#   - Medium mAP: 14.28% - 最低！
#   - Medium Recall: 11.7% - 严重漏检！
#   - Small mAP (18.13%) 和 Large mAP (26.88%) 都正常
#
# Root Cause:
#   - P4层 (stride=16, medium detection layer) 特征表达不足
#   - 模型偏向Small和Large，忽略Medium
#
# Solution:
#   - 在P4层使用ChannelC2f (C2f + Channel Attention)
#   - 增强中等尺度特征的通道表达能力
#
# Expected Improvement:
#   - Medium mAP: 14.28% → 20-25% (+6-11%)
#   - Overall mAP: 44.03% → 46-48% (+2-4%)
#
# ================================================================================================

# Model Configuration
nc: 10  # VisDrone classes

# Model compound scaling constants
scales:
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024]  # YOLO12-N (Nano)  - 对标 RemDet-Tiny
  s: [0.50, 0.50, 1024]  # YOLO12-S (Small) - 对标 RemDet-S
  m: [0.50, 1.00, 512]   # YOLO12-M (Medium) - 对标 RemDet-M
  l: [1.00, 1.00, 512]   # YOLO12-L (Large) - 对标 RemDet-L
  x: [1.00, 1.50, 512]   # YOLO12-X (XLarge) - 对标 RemDet-X

# ================================================================================================
# Backbone (Phase 3 Modified)
# ================================================================================================
# Key Modification:
#   Layer 6 (P4/16): A2C2f → ChannelC2f ⭐
#   - Only P4 layer uses ChannelC2f (medium-scale detection)
#   - P3 and P5 use standard modules

backbone:
  # [from, repeats, module, args]

  # Layer 0-1: RGB-D Input Processing
  - [-1, 1, RGBDStem, [4, 128, 3, 2, 1, 64, "gated_add", 16, True]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4

  # Layer 2-4: P3 Path (Small detection - unchanged)
  - [-1, 2, C3k2, [256, False, 0.25]]  # 2
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]  # 4

  # Layer 5-6: P4 Path (Medium detection - ⭐ Phase 3 MODIFICATION)
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  
  # ⭐ Phase 3: Replace A2C2f with ChannelC2f
  # Args: [c2, shortcut, g, e, reduction]
  #   c1 and n are auto-inserted by parse_model()
  #   c2=512: Output channels
  #   shortcut=True: Use residual connections
  #   g=1: Groups (standard convolution)
  #   e=0.5: Expansion ratio
  #   reduction=16: Channel attention bottleneck ratio
  - [-1, 4, ChannelC2f, [512, True, 1, 0.5, 16]]  # 6-P4/16 ⭐ Phase 3

  # Layer 7-8: P5 Path (Large detection - unchanged)
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 4, A2C2f, [1024, True, 1]]  # 8

# ================================================================================================
# Head (Unchanged from Phase 1)
# ================================================================================================
head:
  # Upsample path
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 9
  - [[-1, 6], 1, Concat, [1]]  # 10 - cat backbone P4 (ChannelC2f output)
  - [-1, 2, A2C2f, [512, False, -1]]  # 11

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 12
  - [[-1, 4], 1, Concat, [1]]  # 13 - cat backbone P3
  - [-1, 2, A2C2f, [256, False, -1]]  # 14

  # Downsample path
  - [-1, 1, Conv, [256, 3, 2]]  # 15
  - [[-1, 11], 1, Concat, [1]]  # 16 - cat head P4
  - [-1, 2, A2C2f, [512, False, -1]]  # 17

  - [-1, 1, Conv, [512, 3, 2]]  # 18
  - [[-1, 8], 1, Concat, [1]]  # 19 - cat head P5
  - [-1, 2, C3k2, [1024, True]]  # 20

  # Detection heads
  - [[14, 17, 20], 1, Detect, [nc]]  # 21 - Detect(P3, P4, P5)
"""

# Model scale specifications
SCALES = {
    'n': {
        'scale_upper': 'N',
        'scale_desc': 'Nano (最轻量, 对标 RemDet-Tiny)',
        'depth': 0.50,
        'width': 0.25,
        'params': '~2.5M',
        'flops': '~5G',
        'pretrained': 'yolo12n.pt'
    },
    's': {
        'scale_upper': 'S',
        'scale_desc': 'Small (轻量, 对标 RemDet-S)',
        'depth': 0.50,
        'width': 0.50,
        'params': '~9.5M',
        'flops': '~20G',
        'pretrained': 'yolo12s.pt'
    },
    'm': {
        'scale_upper': 'M',
        'scale_desc': 'Medium (平衡, 对标 RemDet-M)',
        'depth': 0.50,
        'width': 1.00,
        'params': '~20M',
        'flops': '~40G',
        'pretrained': 'yolo12m.pt'
    },
    'l': {
        'scale_upper': 'L',
        'scale_desc': 'Large (高精度, 对标 RemDet-L)',
        'depth': 1.00,
        'width': 1.00,
        'params': '~40M',
        'flops': '~80G',
        'pretrained': 'yolo12l.pt'
    },
    'x': {
        'scale_upper': 'X',
        'scale_desc': 'XLarge (最高精度, 对标 RemDet-X)',
        'depth': 1.00,
        'width': 1.50,
        'params': '~60M',
        'flops': '~120G',
        'pretrained': 'yolo12x.pt'
    }
}

def generate_yaml(scale: str):
    """Generate YAML config for a specific scale."""
    
    spec = SCALES[scale]
    
    # Generate content
    content = HEADER_TEMPLATE.format(
        scale_upper=spec['scale_upper'],
        scale_desc=spec['scale_desc']
    )
    
    # Add footer with scale-specific info
    footer = f"""
# ================================================================================================
# Model Summary (Expected - YOLO12-{spec['scale_upper']} Phase 3)
# ================================================================================================
# Parameters: {spec['params']} (+1.4% vs Phase 1)
# FLOPs: {spec['flops']} (+0.5% vs Phase 1)
# Pretrained: {spec['pretrained']} (optional)
#
# RemDet Comparison Target:
#   - RemDet-{spec['scale_upper']}: See AAAI2025 paper Table 1
#   - Goal: Match or exceed RemDet performance on VisDrone
#
# Training Command:
#   python train_phase3.py \\
#     --model ultralytics/cfg/models/12/yolo12{scale}-rgbd-channelc2f.yaml \\
#     --data data/visdrone-rgbd.yaml \\
#     --epochs 150 \\
#     --batch 16 \\
#     --name phase3_channelc2f_{scale}
#
# ================================================================================================
"""
    
    content += footer
    
    return content

def main():
    """Generate all YAML configs."""
    
    output_dir = Path("ultralytics/cfg/models/12")
    
    print("=" * 80)
    print("Generating Phase 3 ChannelC2f YAML Configs for All Scales")
    print("=" * 80)
    print()
    
    for scale in ['n', 's', 'm', 'l', 'x']:
        spec = SCALES[scale]
        
        filename = f"yolo12{scale}-rgbd-channelc2f.yaml"
        filepath = output_dir / filename
        
        print(f"[{scale.upper()}] Generating {filename}...")
        print(f"  - Scale: {spec['scale_desc']}")
        print(f"  - Params: {spec['params']}")
        print(f"  - Output: {filepath}")
        
        content = generate_yaml(scale)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  ✅ Created successfully\n")
    
    print("=" * 80)
    print("✅ All YAML configs generated!")
    print("=" * 80)
    print()
    print("Generated files:")
    for scale in ['n', 's', 'm', 'l', 'x']:
        filename = f"yolo12{scale}-rgbd-channelc2f.yaml"
        print(f"  - {filename}")
    print()
    print("Next steps:")
    print("  1. Review generated configs")
    print("  2. Upload to server")
    print("  3. Train all scales:")
    print("     - yolo12n: ~2 days (150 epochs)")
    print("     - yolo12s: ~3 days")
    print("     - yolo12m: ~5 days")
    print("     - yolo12l: ~7 days")
    print("     - yolo12x: ~10 days")
    print()
    print("  4. Compare with RemDet:")
    print("     - RemDet-Tiny vs YOLO12n-ChannelC2f")
    print("     - RemDet-S vs YOLO12s-ChannelC2f")
    print("     - RemDet-M vs YOLO12m-ChannelC2f")
    print("     - RemDet-L vs YOLO12l-ChannelC2f")
    print("     - RemDet-X vs YOLO12x-ChannelC2f")
    print()

if __name__ == "__main__":
    main()
