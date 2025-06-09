#!/usr/bin/env python3
"""Debug script to check data paths"""
import os

# Check what directories actually exist
print("Checking data directory structure...")
data_dir = "./data"
if os.path.exists(data_dir):
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            print(f"\n📁 {item}/")
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                if os.path.isdir(subitem_path):
                    file_count = len(
                        [
                            f
                            for f in os.listdir(subitem_path)
                            if f.endswith((".png", ".jpg", ".tif"))
                        ]
                    )
                    print(f"    📁 {subitem}/ ({file_count} files)")
else:
    print("❌ ./data directory not found!")

# Test specific paths
print("\n" + "=" * 50)
print("Testing specific paths:")

test_paths = {
    "human kidney images": "./data/human kidney/tissue images",
    "human kidney labels": "./data/human kidney/label masks",
    "mouse kidney images": "./data/mouse kidney/tissue images",
    "mouse kidney labels": "./data/mouse kidney/label masks",
}

for name, path in test_paths.items():
    if os.path.exists(path):
        files = [f for f in os.listdir(path) if f.endswith((".png", ".jpg", ".tif"))]
        print(f"✓ {name}: {len(files)} files")
        if files:
            print(f"    Example: {files[0]}")
    else:
        print(f"❌ {name}: Path not found")

print("\n" + "=" * 50)
print("If paths are wrong, update multi_organ_config.py DATASET_PATHS and LABEL_PATHS")
