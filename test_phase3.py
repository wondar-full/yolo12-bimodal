#!/usr/bin/env python3
"""
Phase 3 Local Testing Script

Tests ChannelC2f implementation before server training:
1. Model construction
2. Forward pass
3. Parameter count
4. ChannelAttention module existence

Usage:
    python test_phase3.py
"""

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info
from pathlib import Path

def test_phase3():
    """Test ChannelC2f implementation locally."""
    
    print("=" * 80)
    print("Phase 3: ChannelC2f Local Testing")
    print("=" * 80)
    print()
    
    # =================================================================
    # Test 1: Model Construction
    # =================================================================
    
    print("Test 1: Model Construction")
    print("-" * 80)
    
    model_cfg = "ultralytics/cfg/models/12/yolo12s-rgbd-channelc2f.yaml"
    
    if not Path(model_cfg).exists():
        print(f"❌ Model config not found: {model_cfg}")
        return False
    
    try:
        model = YOLO(model_cfg)
        print(f"✅ Model built successfully from {model_cfg}")
    except Exception as e:
        print(f"❌ Model construction failed: {e}")
        return False
    
    print()
    
    # =================================================================
    # Test 2: Forward Pass
    # =================================================================
    
    print("Test 2: Forward Pass")
    print("-" * 80)
    
    try:
        # Create dummy input [B, C, H, W] - 4 channels (RGB + Depth)
        x = torch.randn(1, 4, 640, 640)
        print(f"Input shape: {x.shape}")
        
        # Forward pass
        with torch.no_grad():
            y = model(x)
        
        print(f"✅ Forward pass successful")
        print(f"Output type: {type(y)}")
        
        if isinstance(y, (list, tuple)):
            print(f"Output shapes: {[yi.shape for yi in y]}")
        else:
            print(f"Output shape: {y.shape}")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # =================================================================
    # Test 3: Parameter Count & FLOPs
    # =================================================================
    
    print("Test 3: Parameter Count & FLOPs")
    print("-" * 80)
    
    try:
        # Get model info
        model_info(model.model, imgsz=640)
        
        # Extract key metrics
        n_p = sum(x.numel() for x in model.model.parameters())
        n_p_m = n_p / 1e6
        
        print()
        print(f"Total Parameters: {n_p:,} ({n_p_m:.2f}M)")
        
        # Expected for Phase 3
        expected_params = 9.52e6  # 9.52M
        params_diff = abs(n_p - expected_params) / expected_params
        
        if params_diff < 0.05:  # Within 5%
            print(f"✅ Parameter count close to expected ({expected_params/1e6:.2f}M)")
        else:
            print(f"⚠️  Parameter count differs from expected:")
            print(f"   Expected: {expected_params/1e6:.2f}M")
            print(f"   Actual:   {n_p_m:.2f}M")
            print(f"   Diff:     {params_diff*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Parameter count failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # =================================================================
    # Test 4: ChannelAttention Verification
    # =================================================================
    
    print("Test 4: ChannelAttention Module Verification")
    print("-" * 80)
    
    found_ca = False
    found_channelc2f = False
    ca_locations = []
    
    for name, module in model.model.named_modules():
        module_type = type(module).__name__
        
        if 'ChannelAttention' in module_type:
            found_ca = True
            ca_locations.append(name)
        
        if 'ChannelC2f' in module_type:
            found_channelc2f = True
            print(f"✅ Found ChannelC2f at: {name}")
    
    if found_ca:
        print(f"✅ Found ChannelAttention modules:")
        for loc in ca_locations:
            print(f"   - {loc}")
    else:
        print(f"❌ ChannelAttention not found in model!")
        return False
    
    if not found_channelc2f:
        print(f"❌ ChannelC2f not found in model!")
        return False
    
    print()
    
    # =================================================================
    # Test 5: Compare with Phase 1
    # =================================================================
    
    print("Test 5: Comparison with Phase 1 Baseline")
    print("-" * 80)
    
    try:
        # Load Phase 1 model for comparison
        phase1_cfg = "ultralytics/cfg/models/12/yolo12s-rgbd-v1.yaml"
        
        if Path(phase1_cfg).exists():
            model_phase1 = YOLO(phase1_cfg)
            n_p1 = sum(x.numel() for x in model_phase1.model.parameters())
            
            param_increase = (n_p - n_p1) / n_p1 * 100
            
            print(f"Phase 1 Parameters: {n_p1/1e6:.2f}M")
            print(f"Phase 3 Parameters: {n_p/1e6:.2f}M")
            print(f"Parameter Increase: {param_increase:+.2f}%")
            
            if param_increase < 2.0:
                print(f"✅ Parameter increase within acceptable range (<2%)")
            else:
                print(f"⚠️  Parameter increase higher than expected (target: +1.4%)")
        else:
            print(f"⚠️  Phase 1 config not found, skipping comparison")
    
    except Exception as e:
        print(f"⚠️  Comparison failed: {e}")
    
    print()
    
    # =================================================================
    # Summary
    # =================================================================
    
    print("=" * 80)
    print("✅ All Tests Passed!")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("  1. Upload code to server:")
    print("     scp -r ultralytics/ ubuntu@server:/data2/user/2024/lzy/yolo12-bimodal/")
    print("     scp train_phase3.py ubuntu@server:/data2/user/2024/lzy/yolo12-bimodal/")
    print()
    print("  2. Start training:")
    print("     CUDA_VISIBLE_DEVICES=6 nohup python train_phase3.py > train_phase3.log 2>&1 &")
    print()
    print("  3. Monitor progress:")
    print("     tail -f train_phase3.log")
    print()
    
    return True


if __name__ == "__main__":
    success = test_phase3()
    exit(0 if success else 1)
