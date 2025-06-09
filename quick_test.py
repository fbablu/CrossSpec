"""Quick test to verify everything works"""

import torch
from simple_models import SimpleUNet
from multi_organ_dataloader import MultiOrganDataset

# Test model
model = SimpleUNet(num_classes=7)
print("✓ Model created successfully")

# Test dataloader
dataset = MultiOrganDataset(mode="homologous", organs=["kidney"])
print(f"✓ Dataset loaded: {len(dataset)} samples")

if len(dataset) > 0:
    sample = dataset[0]
    print(f"✓ Sample shape: {sample['image'].shape}, {sample['label'].shape}")
else:
    print("⚠ No data found - check your data paths")
