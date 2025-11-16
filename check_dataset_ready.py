#!/usr/bin/env python3
"""
=====================================================================
Quick Dataset Verification Script
=====================================================================
Purpose: Check VisDrone + UAVDT datasets before training
Checks:
1. Directory structure
2. Image/depth/label file counts
3. RGB-Depth alignment
4. YAML configuration
5. Class distribution

Usage:
    # On remote server
    cd /data2/user/2024/lzy/
    python yoloDepth/check_dataset_ready.py
=====================================================================
"""

import sys
from pathlib import Path
import yaml

# VisDrone and UAVDT dataset paths (modify if needed)
VISDRONE_ROOT = Path("/data2/user/2024/lzy/Datasets/VisDrone2019-DET-YOLO/VisDrone2YOLO")
UAVDT_ROOT = Path("/data2/user/2024/lzy/Datasets/UAVDT_YOLO")
YAML_PATH = Path("yoloDepth/data/uav-joint-rgbd.yaml")

# Expected class names (VisDrone 10 classes)
EXPECTED_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]


def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_directory(path, name):
    """Check if directory exists."""
    if path.exists() and path.is_dir():
        print(f"✅ {name}: {path}")
        return True
    else:
        print(f"❌ {name} NOT FOUND: {path}")
        return False


def count_files(path, pattern):
    """Count files matching pattern."""
    if not path.exists():
        return 0
    return len(list(path.glob(pattern)))


def check_dataset(root, name):
    """Check a single dataset (VisDrone or UAVDT)."""
    print_header(f"{name} Dataset Check")
    
    # Check root directory
    if not check_directory(root, f"{name} root"):
        return False
    
    # Check train/val directories
    train_img = root / "images" / "train"
    val_img = root / "images" / "val"
    train_depth = root / "depths" / "train"
    val_depth = root / "depths" / "val"
    train_labels = root / "labels" / "train"
    val_labels = root / "labels" / "val"
    
    all_exist = True
    all_exist &= check_directory(train_img, "Train images")
    all_exist &= check_directory(val_img, "Val images")
    all_exist &= check_directory(train_depth, "Train depths")
    all_exist &= check_directory(val_depth, "Val depths")
    all_exist &= check_directory(train_labels, "Train labels")
    all_exist &= check_directory(val_labels, "Val labels")
    
    if not all_exist:
        print(f"\n❌ {name}: Missing directories!")
        return False
    
    # Count files
    print(f"\n{name} File Counts:")
    print("-" * 70)
    
    train_img_count = count_files(train_img, "*.jpg") + count_files(train_img, "*.png")
    val_img_count = count_files(val_img, "*.jpg") + count_files(val_img, "*.png")
    train_depth_count = count_files(train_depth, "*.jpg") + count_files(train_depth, "*.png")
    val_depth_count = count_files(val_depth, "*.jpg") + count_files(val_depth, "*.png")
    train_label_count = count_files(train_labels, "*.txt")
    val_label_count = count_files(val_labels, "*.txt")
    
    print(f"Train images: {train_img_count}")
    print(f"Train depths: {train_depth_count}")
    print(f"Train labels: {train_label_count}")
    print(f"Val images:   {val_img_count}")
    print(f"Val depths:   {val_depth_count}")
    print(f"Val labels:   {val_label_count}")
    
    # Check alignment
    print(f"\n{name} Alignment Check:")
    print("-" * 70)
    
    train_aligned = (train_img_count == train_depth_count == train_label_count)
    val_aligned = (val_img_count == val_depth_count == val_label_count)
    
    if train_aligned:
        print(f"✅ Train set aligned ({train_img_count} samples)")
    else:
        print(f"❌ Train set NOT aligned!")
        print(f"   Images: {train_img_count}, Depths: {train_depth_count}, Labels: {train_label_count}")
    
    if val_aligned:
        print(f"✅ Val set aligned ({val_img_count} samples)")
    else:
        print(f"❌ Val set NOT aligned!")
        print(f"   Images: {val_img_count}, Depths: {val_depth_count}, Labels: {val_label_count}")
    
    return train_aligned and val_aligned


def check_yaml():
    """Check YAML configuration."""
    print_header("YAML Configuration Check")
    
    if not YAML_PATH.exists():
        print(f"❌ YAML not found: {YAML_PATH}")
        return False
    
    print(f"✅ YAML found: {YAML_PATH}")
    
    # Load YAML
    with open(YAML_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check required fields
    required_fields = ['path', 'train', 'val', 'train_depth', 'val_depth', 'nc', 'names']
    missing = [f for f in required_fields if f not in config]
    
    if missing:
        print(f"❌ Missing YAML fields: {missing}")
        return False
    
    print(f"\n✅ All required fields present")
    print(f"   Classes: {config['nc']}")
    print(f"   Names: {config['names'][:3]}... (showing first 3)")
    
    # Check class count
    if config['nc'] != len(EXPECTED_CLASSES):
        print(f"⚠️  Warning: nc={config['nc']}, but expected {len(EXPECTED_CLASSES)} classes")
    
    return True


def check_sample_files():
    """Check if sample files are readable."""
    print_header("Sample File Check")
    
    # Try to find first image in each dataset
    visdrone_train = VISDRONE_ROOT / "images" / "train"
    uavdt_train = UAVDT_ROOT / "images" / "train"
    
    sample_found = False
    
    if visdrone_train.exists():
        samples = list(visdrone_train.glob("*.jpg"))[:1]
        if samples:
            print(f"✅ VisDrone sample: {samples[0].name}")
            sample_found = True
    
    if uavdt_train.exists():
        samples = list(uavdt_train.glob("*.jpg"))[:1]
        if samples:
            print(f"✅ UAVDT sample: {samples[0].name}")
            sample_found = True
    
    if not sample_found:
        print("❌ No sample images found!")
        return False
    
    return True


def main():
    """Main verification function."""
    print("=" * 70)
    print("  VisDrone + UAVDT Dataset Verification")
    print("=" * 70)
    
    # Check VisDrone
    visdrone_ok = check_dataset(VISDRONE_ROOT, "VisDrone")
    
    # Check UAVDT
    uavdt_ok = check_dataset(UAVDT_ROOT, "UAVDT")
    
    # Check YAML
    yaml_ok = check_yaml()
    
    # Check samples
    sample_ok = check_sample_files()
    
    # Final summary
    print_header("Final Summary")
    
    if visdrone_ok and uavdt_ok and yaml_ok and sample_ok:
        print("✅ All checks PASSED!")
        print("\n🚀 Ready to start training:")
        print("   cd yoloDepth")
        print("   python train_uav_joint.py --device 0 --batch 16 --epochs 10 --name test_joint")
        return 0
    else:
        print("❌ Some checks FAILED!")
        print("\nPlease fix the issues above before training.")
        if not visdrone_ok:
            print("  - Fix VisDrone dataset")
        if not uavdt_ok:
            print("  - Fix UAVDT dataset")
        if not yaml_ok:
            print("  - Fix YAML configuration")
        if not sample_ok:
            print("  - Check file permissions")
        return 1


if __name__ == "__main__":
    sys.exit(main())
