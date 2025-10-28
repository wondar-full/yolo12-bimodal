#!/usr/bin/env python3
"""
Phase 3 Code Verification Script

Verifies that ChannelC2f implementation is complete and correct.
Usage: python verify_phase3.py
"""

import os
import sys
from pathlib import Path

# ANSI color codes
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
CYAN = '\033[0;36m'
NC = '\033[0m'  # No Color

def print_header(text):
    """Print colored header."""
    print(f"\n{CYAN}{'='*80}{NC}")
    print(f"{CYAN}{text}{NC}")
    print(f"{CYAN}{'='*80}{NC}\n")

def print_check(num, total, text):
    """Print check header."""
    print(f"{YELLOW}[{num}/{total}] {text}{NC}")

def print_success(text):
    """Print success message."""
    print(f"{GREEN}  ✅ {text}{NC}")

def print_error(text):
    """Print error message."""
    print(f"{RED}  ❌ {text}{NC}")

def check_file_exists(filepath):
    """Check if file exists."""
    return Path(filepath).exists()

def check_class_exists(filepath, class_name):
    """Check if class exists in file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    return f"class {class_name}(nn.Module):" in content

def check_method_exists(filepath, class_name, method_name):
    """Check if method exists in class."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract class content
    class_start = content.find(f"class {class_name}(nn.Module):")
    if class_start == -1:
        return False
    
    # Find next class definition
    next_class = content.find("\nclass ", class_start + 1)
    if next_class == -1:
        class_content = content[class_start:]
    else:
        class_content = content[class_start:next_class]
    
    return f"def {method_name}(" in class_content

def check_code_contains(filepath, class_name, code_snippet):
    """Check if class contains specific code snippet."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract class content
    class_start = content.find(f"class {class_name}(nn.Module):")
    if class_start == -1:
        return False
    
    # Find next class definition
    next_class = content.find("\nclass ", class_start + 1)
    if next_class == -1:
        class_content = content[class_start:]
    else:
        class_content = content[class_start:next_class]
    
    return code_snippet in class_content

def check_in_all_exports(filepath, export_name):
    """Check if name is in __all__ exports."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find __all__ definition
    all_start = content.find("__all__ = (")
    if all_start == -1:
        return False
    
    all_end = content.find(")", all_start)
    all_content = content[all_start:all_end]
    
    return f'"{export_name}"' in all_content

def get_class_line_count(filepath, class_name):
    """Count lines in a class definition."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    class_start = content.find(f"class {class_name}(nn.Module):")
    if class_start == -1:
        return 0
    
    next_class = content.find("\nclass ", class_start + 1)
    if next_class == -1:
        class_content = content[class_start:]
    else:
        class_content = content[class_start:next_class]
    
    return len(class_content.split('\n'))

def main():
    """Main verification function."""
    print_header("Phase 3: Code Verification")
    
    # File paths
    block_py = "ultralytics/nn/modules/block.py"
    yaml_cfg = "ultralytics/cfg/models/12/yolo12s-rgbd-channelc2f.yaml"
    
    all_passed = True
    
    # ================================================================
    # Check 1: block.py exists
    # ================================================================
    
    print_check(1, 6, "Checking block.py file...")
    
    if check_file_exists(block_py):
        print_success(f"{block_py} exists")
    else:
        print_error(f"{block_py} NOT found!")
        return False
    
    # ================================================================
    # Check 2: ChannelAttention class
    # ================================================================
    
    print_check(2, 6, "Checking ChannelAttention class...")
    
    if check_class_exists(block_py, "ChannelAttention"):
        print_success("ChannelAttention class found")
        
        # Check forward method
        if check_method_exists(block_py, "ChannelAttention", "forward"):
            print_success("ChannelAttention.forward() method found")
            
            # Check key implementation details
            if check_code_contains(block_py, "ChannelAttention", "self.avg_pool"):
                print_success("Contains self.avg_pool (Squeeze)")
            
            if check_code_contains(block_py, "ChannelAttention", "self.fc"):
                print_success("Contains self.fc (Excitation)")
        else:
            print_error("ChannelAttention.forward() NOT found!")
            all_passed = False
    else:
        print_error("ChannelAttention class NOT found!")
        all_passed = False
    
    # ================================================================
    # Check 3: ChannelC2f class
    # ================================================================
    
    print_check(3, 6, "Checking ChannelC2f class...")
    
    if check_class_exists(block_py, "ChannelC2f"):
        print_success("ChannelC2f class found")
        
        # Check __init__ method
        if check_method_exists(block_py, "ChannelC2f", "__init__"):
            print_success("ChannelC2f.__init__() method found")
            
            # Check key components
            if check_code_contains(block_py, "ChannelC2f", "self.cv1"):
                print_success("Contains self.cv1 (Input conv)")
            
            if check_code_contains(block_py, "ChannelC2f", "self.cv2"):
                print_success("Contains self.cv2 (Output conv)")
            
            if check_code_contains(block_py, "ChannelC2f", "self.m = nn.ModuleList"):
                print_success("Contains self.m (Bottleneck stack)")
            
            if check_code_contains(block_py, "ChannelC2f", "self.ca = ChannelAttention"):
                print_success("Contains self.ca (Channel Attention) ⭐")
            else:
                print_error("Missing self.ca = ChannelAttention!")
                all_passed = False
        
        # Check forward method
        if check_method_exists(block_py, "ChannelC2f", "forward"):
            print_success("ChannelC2f.forward() method found")
            
            # Check critical line: self.ca(x)
            if check_code_contains(block_py, "ChannelC2f", "self.ca(x)"):
                print_success("forward() calls self.ca(x) - Phase 3 implementation complete! ⭐")
            else:
                print_error("forward() does NOT call self.ca(x)!")
                print_error("Implementation is incomplete!")
                all_passed = False
        else:
            print_error("ChannelC2f.forward() NOT found!")
            all_passed = False
    else:
        print_error("ChannelC2f class NOT found!")
        all_passed = False
    
    # ================================================================
    # Check 4: __all__ exports
    # ================================================================
    
    print_check(4, 6, "Checking __all__ exports...")
    
    if check_in_all_exports(block_py, "ChannelAttention"):
        print_success("ChannelAttention in __all__")
    else:
        print_error("ChannelAttention NOT in __all__!")
        all_passed = False
    
    if check_in_all_exports(block_py, "ChannelC2f"):
        print_success("ChannelC2f in __all__")
    else:
        print_error("ChannelC2f NOT in __all__!")
        all_passed = False
    
    # ================================================================
    # Check 5: YAML configuration
    # ================================================================
    
    print_check(5, 6, "Checking YAML configuration...")
    
    if check_file_exists(yaml_cfg):
        print_success(f"{yaml_cfg} exists")
        
        with open(yaml_cfg, 'r', encoding='utf-8') as f:
            yaml_content = f.read()
        
        if "ChannelC2f" in yaml_content:
            print_success("YAML contains ChannelC2f module")
            
            # Count occurrences (should be 1 - only P4 layer)
            count = yaml_content.count("ChannelC2f")
            print_success(f"ChannelC2f used {count} time(s) in YAML")
            
            if count == 1:
                print_success("Correct: Only P4 layer uses ChannelC2f ✓")
            elif count > 1:
                print_error(f"Warning: ChannelC2f used {count} times (expected 1)")
        else:
            print_error("YAML does NOT contain ChannelC2f!")
            all_passed = False
    else:
        print_error(f"{yaml_cfg} NOT found!")
        all_passed = False
    
    # ================================================================
    # Check 6: Import test
    # ================================================================
    
    print_check(6, 6, "Testing Python import...")
    
    try:
        # Add parent directory to path
        sys.path.insert(0, str(Path.cwd()))
        
        # Try importing
        from ultralytics.nn.modules.block import ChannelAttention, ChannelC2f
        
        print_success("Successfully imported ChannelAttention")
        print_success("Successfully imported ChannelC2f")
        
        # Check if classes are callable
        if callable(ChannelAttention):
            print_success("ChannelAttention is callable")
        
        if callable(ChannelC2f):
            print_success("ChannelC2f is callable")
        
    except ImportError as e:
        print_error(f"Import failed: {e}")
        all_passed = False
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        all_passed = False
    
    # ================================================================
    # Summary
    # ================================================================
    
    print_header("Verification Summary")
    
    if all_passed:
        print(f"{GREEN}✅ All verification checks passed!{NC}\n")
        
        print(f"{YELLOW}📋 Code statistics:{NC}")
        ca_lines = get_class_line_count(block_py, "ChannelAttention")
        cc2f_lines = get_class_line_count(block_py, "ChannelC2f")
        print(f"  - ChannelAttention: {ca_lines} lines")
        print(f"  - ChannelC2f: {cc2f_lines} lines")
        print()
        
        print(f"{YELLOW}🚀 Next steps:{NC}")
        print("  1. Test model construction:")
        print("     python test_phase3.py")
        print()
        print("  2. If test passes, upload to server:")
        print("     scp ultralytics/nn/modules/block.py ubuntu@server:...")
        print("     scp ultralytics/cfg/models/12/yolo12s-rgbd-channelc2f.yaml ubuntu@server:...")
        print("     scp train_phase3.py ubuntu@server:...")
        print()
        print("  3. Start training on server:")
        print("     CUDA_VISIBLE_DEVICES=6 nohup python train_phase3.py > train_phase3.log 2>&1 &")
        print()
        
        return True
    else:
        print(f"{RED}❌ Verification failed! Please fix the issues above.{NC}\n")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
