#!/bin/bash
# 检查RGBDGGFEFusion是否存在于服务器上

echo "========================================"
echo "检查1: rgbd_ggfe_fusion.py 文件是否存在"
echo "========================================"
ls -lh ultralytics/nn/modules/rgbd_ggfe_fusion.py
echo ""

echo "========================================"
echo "检查2: __init__.py 是否导入 RGBDGGFEFusion"
echo "========================================"
grep -n "RGBDGGFEFusion" ultralytics/nn/modules/__init__.py
echo ""

echo "========================================"
echo "检查3: 尝试直接导入"
echo "========================================"
python -c "from ultralytics.nn.modules import RGBDGGFEFusion; print('✅ Import successful'); print(RGBDGGFEFusion)"
echo ""

echo "========================================"
echo "检查4: ggfe.py 文件是否存在"
echo "========================================"
ls -lh ultralytics/nn/modules/ggfe.py
echo ""

echo "========================================"
echo "检查5: 尝试导入GGFE"
echo "========================================"
python -c "from ultralytics.nn.modules import GGFE; print('✅ GGFE import successful'); print(GGFE)"
