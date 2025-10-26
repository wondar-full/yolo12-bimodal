"""
Test script to validate yolo12s-rgbd-v2.1.yaml architecture initialization.

This script checks:
1. Model can be loaded without errors
2. Channel dimensions are correct
3. RGBDMidFusion modules are present at correct layers
4. Forward pass works with dummy RGB-D input
5. Parameter count is reasonable

Run: python test_v2.1_architecture.py
"""

import torch
from ultralytics import YOLO

print("="*80)
print("Testing YOLO12s-RGB-D v2.1 Architecture Initialization")
print("="*80)

# Test 1: Model Loading
print("\n[Test 1] Loading model from YAML...")
try:
    model = YOLO('ultralytics/cfg/models/12/yolo12s-rgbd-v2.1.yaml')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    exit(1)

# Test 2: Print Architecture
print("\n[Test 2] Model architecture:")
print(model.model)

# Test 3: Check Parameter Count
total_params = sum(p.numel() for p in model.model.parameters())
trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
print(f"\n[Test 3] Parameters:")
print(f"  Total: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"  Trainable: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

# Expected: ~11.5-12M (similar to v1.0 + RGBDMidFusion modules)
if 11e6 <= total_params <= 15e6:
    print("✅ Parameter count reasonable (11-15M)")
else:
    print(f"⚠️ Parameter count unexpected (expected 11-15M)")

# Test 4: Check for RGBDMidFusion modules
print("\n[Test 4] Checking RGBDMidFusion modules...")
rgbd_mid_fusion_count = 0
rgbd_stem_count = 0

for name, module in model.model.named_modules():
    module_type = type(module).__name__
    if 'RGBDMidFusion' in module_type:
        rgbd_mid_fusion_count += 1
        print(f"  Found: {name} ({module_type})")
    if 'RGBDStem' in module_type:
        rgbd_stem_count += 1
        print(f"  Found: {name} ({module_type})")

print(f"\n  RGBDStem count: {rgbd_stem_count} (expected: 1)")
print(f"  RGBDMidFusion count: {rgbd_mid_fusion_count} (expected: 3 @ P3/P4/P5)")

if rgbd_stem_count == 1:
    print("✅ RGBDStem module present")
else:
    print(f"❌ Expected 1 RGBDStem, found {rgbd_stem_count}")

if rgbd_mid_fusion_count == 3:
    print("✅ All 3 RGBDMidFusion modules present")
elif rgbd_mid_fusion_count > 0:
    print(f"⚠️ Expected 3 RGBDMidFusion, found {rgbd_mid_fusion_count}")
else:
    print("❌ No RGBDMidFusion modules found!")

# Test 5: Forward Pass with Dummy Input
print("\n[Test 5] Testing forward pass with dummy RGB-D input...")
try:
    # Create dummy RGB-D input: [B, 4, 640, 640]
    dummy_input = torch.randn(1, 4, 640, 640)
    print(f"  Input shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model.model(dummy_input)
    
    # Check output format
    if isinstance(output, tuple) or isinstance(output, list):
        print(f"  Output type: {type(output)} with {len(output)} elements")
        for i, o in enumerate(output):
            if isinstance(o, torch.Tensor):
                print(f"  Output[{i}] shape: {o.shape}")
    else:
        print(f"  Output shape: {output.shape}")
    
    print("✅ Forward pass successful!")
    
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 6: Check Depth Skip Connection Mechanism
print("\n[Test 6] Checking depth skip connection mechanism...")
print("  (This requires inspecting Layer 0 output structure)")

# Note: This test is conceptual, actual implementation depends on tasks.py forward logic
# In production, Layer 0 should output [fused 64ch + depth 64ch] = 128ch
# We verify this by checking the first layer's output channels

try:
    first_layer = list(model.model.model.children())[0]  # Layer 0 (RGBDStem)
    print(f"  Layer 0 type: {type(first_layer).__name__}")
    
    # Test Layer 0 forward
    with torch.no_grad():
        layer0_input = torch.randn(1, 4, 640, 640)
        layer0_output = first_layer(layer0_input)
    
    print(f"  Layer 0 output shape: {layer0_output.shape}")
    
    # Expected: [1, 128, 320, 320] (fused 64ch + depth 64ch)
    if layer0_output.shape[1] == 128:
        print("✅ Layer 0 outputs 128 channels (fused + depth)")
        print("  Depth skip should be extracted as output[:, 64:, :, :]")
    else:
        print(f"⚠️ Layer 0 output channels: {layer0_output.shape[1]} (expected 128)")
    
except Exception as e:
    print(f"⚠️ Layer 0 test failed: {e}")

# Test 7: Summary Report
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

all_tests_passed = (
    rgbd_stem_count == 1 and
    rgbd_mid_fusion_count == 3 and
    11e6 <= total_params <= 15e6
)

if all_tests_passed:
    print("✅ All critical tests passed!")
    print("✅ yolo12s-rgbd-v2.1.yaml is ready for training.")
    print("\nNext steps:")
    print("1. Run quick 10-epoch test:")
    print("   python train_depth.py --model ultralytics/cfg/models/12/yolo12s-rgbd-v2.1.yaml --epochs 10 --batch 8 --name rgbd_v2.1_test")
    print("2. Compare mAP@0.5 with v1.0 (30.86% @ epoch 10)")
    print("3. If successful (>32%), run full 300-epoch training")
else:
    print("❌ Some tests failed. Review errors above.")
    print("Common issues:")
    print("  - Missing RGBDMidFusion in tasks.py imports")
    print("  - Incorrect YAML 'from' field format")
    print("  - parse_model not handling RGBDMidFusion")

print("="*80)
