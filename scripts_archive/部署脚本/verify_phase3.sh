#!/bin/bash
# Phase 3 Server Verification Script
# Usage: bash verify_phase3.sh

echo "================================================================================"
echo "Phase 3: Server Code Verification"
echo "================================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check 1: ChannelAttention in block.py
echo -e "${YELLOW}[1/5] Checking ChannelAttention class...${NC}"
if grep -q "class ChannelAttention(nn.Module):" ultralytics/nn/modules/block.py; then
    echo -e "${GREEN}  ✅ ChannelAttention class found${NC}"
else
    echo -e "${RED}  ❌ ChannelAttention class NOT found!${NC}"
    exit 1
fi

# Check 2: ChannelC2f in block.py
echo -e "\n${YELLOW}[2/5] Checking ChannelC2f class...${NC}"
if grep -q "class ChannelC2f(nn.Module):" ultralytics/nn/modules/block.py; then
    echo -e "${GREEN}  ✅ ChannelC2f class found${NC}"
else
    echo -e "${RED}  ❌ ChannelC2f class NOT found!${NC}"
    exit 1
fi

# Check 3: Forward method implementation
echo -e "\n${YELLOW}[3/5] Checking ChannelC2f forward method...${NC}"

# Extract the entire ChannelC2f class (from class definition to next class)
channelc2f_class=$(sed -n '/^class ChannelC2f/,/^class /p' ultralytics/nn/modules/block.py)

# Check if forward method exists
if echo "$channelc2f_class" | grep -q "def forward"; then
    echo -e "${GREEN}  ✅ Forward method found${NC}"
    
    # Check if forward method contains self.ca(x) - the key Phase 3 addition
    if echo "$channelc2f_class" | grep -q "self.ca(x)"; then
        echo -e "${GREEN}  ✅ Forward method contains self.ca(x) - implementation complete${NC}"
    else
        echo -e "${RED}  ❌ Forward method is empty or incomplete!${NC}"
        echo -e "${RED}     Missing: self.ca(x) call${NC}"
        exit 1
    fi
else
    echo -e "${RED}  ❌ Forward method NOT found!${NC}"
    exit 1
fi

# Check 4: __all__ exports
echo -e "\n${YELLOW}[4/5] Checking __all__ exports...${NC}"
if grep -q '"ChannelAttention"' ultralytics/nn/modules/block.py; then
    echo -e "${GREEN}  ✅ ChannelAttention in __all__${NC}"
else
    echo -e "${RED}  ❌ ChannelAttention NOT in __all__!${NC}"
    exit 1
fi

if grep -q '"ChannelC2f"' ultralytics/nn/modules/block.py; then
    echo -e "${GREEN}  ✅ ChannelC2f in __all__${NC}"
else
    echo -e "${RED}  ❌ ChannelC2f NOT in __all__!${NC}"
    exit 1
fi

# Check 5: YAML config
echo -e "\n${YELLOW}[5/5] Checking YAML configuration...${NC}"
if [ -f "ultralytics/cfg/models/12/yolo12s-rgbd-channelc2f.yaml" ]; then
    echo -e "${GREEN}  ✅ YAML config file exists${NC}"
    
    if grep -q "ChannelC2f" ultralytics/cfg/models/12/yolo12s-rgbd-channelc2f.yaml; then
        echo -e "${GREEN}  ✅ YAML contains ChannelC2f module${NC}"
    else
        echo -e "${RED}  ❌ YAML does NOT contain ChannelC2f!${NC}"
        exit 1
    fi
else
    echo -e "${RED}  ❌ YAML config file NOT found!${NC}"
    exit 1
fi

echo ""
echo "================================================================================"
echo -e "${GREEN}✅ All verification checks passed!${NC}"
echo "================================================================================"
echo ""
echo -e "${YELLOW}📋 Code statistics:${NC}"
echo -n "  - ChannelAttention lines: "
sed -n '/^class ChannelAttention/,/^class /p' ultralytics/nn/modules/block.py | wc -l
echo -n "  - ChannelC2f lines: "
sed -n '/^class ChannelC2f/,/^class /p' ultralytics/nn/modules/block.py | wc -l
echo ""

echo -e "${YELLOW}🚀 Ready to proceed with:${NC}"
echo "  python test_phase3.py"
echo "  # If test passes:"
echo "  CUDA_VISIBLE_DEVICES=6 nohup python train_phase3.py > train_phase3.log 2>&1 &"
echo ""
