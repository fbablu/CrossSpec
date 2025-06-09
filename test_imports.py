#!/usr/bin/env python3
"""
Test script to verify model imports work correctly
"""

try:
    from simple_models import SimpleUNet as Unet

    print("✓ SimpleUNet import successful")
except ImportError as e:
    print(f"✗ SimpleUNet import failed: {e}")

try:
    from simple_models import SimplePSPNet as PSPNet

    print("✓ SimplePSPNet import successful")
except ImportError as e:
    print(f"✗ SimplePSPNet import failed: {e}")

try:
    from simple_models import SimpleDeepLab as DeepLab

    print("✓ SimpleDeepLab import successful")
except ImportError as e:
    print(f"✗ SimpleDeepLab import failed: {e}")

# Test model instantiation
try:
    model = Unet(num_classes=7)
    print("✓ Model instantiation successful")
except Exception as e:
    print(f"✗ Model instantiation failed: {e}")

print("\nIf all imports successful, you can run the training scripts.")
