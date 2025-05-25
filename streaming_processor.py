#!/usr/bin/env python3
import os
import subprocess
import shutil
from pathlib import Path
import openslide
import numpy as np
from PIL import Image


def download_single_wsi(file_id, output_dir='temp_wsi'):
    """Download single WSI by file ID"""
    Path(output_dir).mkdir(exist_ok=True)

    cmd = ['./gdc-client', 'download', file_id, '-d', output_dir]
    result = subprocess.run(cmd, capture_output=True)

    if result.returncode == 0:
        # Find downloaded .svs file
        for file_path in Path(output_dir).rglob('*.svs'):
            return str(file_path)
    return None


def extract_patches(wsi_path, patch_size=1024, overlap=0.5, output_dir='patches'):
    """Extract patches from WSI"""
    Path(output_dir).mkdir(exist_ok=True)

    slide = openslide.OpenSlide(wsi_path)
    width, height = slide.dimensions
    step_size = int(patch_size * (1 - overlap))

    patches_saved = 0
    wsi_name = Path(wsi_path).stem

    for x in range(0, width - patch_size, step_size):
        for y in range(0, height - patch_size, step_size):
            # Extract patch
            patch = slide.read_region((x, y), 0, (patch_size, patch_size))
            patch = patch.convert('RGB')

            # Skip mostly white/background patches
            if is_tissue_patch(patch):
                patch_name = f"{wsi_name}_x{x}_y{y}.png"
                patch.save(f"{output_dir}/{patch_name}")
                patches_saved += 1

    slide.close()
    return patches_saved


def process_patches_with_ai(patch_dir, model):
    """Process patches through your AI model, then delete them"""
    patch_features = []

    for patch_file in Path(patch_dir).glob('*.png'):
        # Load patch
        patch = Image.open(patch_file)

        ########################################################################################
        ################################### INPUT INTO AI MODEL ################################
        ########################################################################################
        # Run through your AI model (replace with your model)
        # features = model.extract_features(patch)
        # patch_features.append(features)
        ########################################################################################
        ################################### INPUT INTO AI MODEL ################################
        ########################################################################################

        # Delete patch to save space
        os.remove(patch_file)

    return patch_features


def is_tissue_patch(patch, threshold=0.8):
    """Check if patch contains tissue (not mostly background)"""
    patch_array = np.array(patch)

    # Convert to grayscale and check if mostly white
    gray = np.mean(patch_array, axis=2)
    white_ratio = np.sum(gray > 200) / gray.size

    return white_ratio < threshold
    """Check if patch contains tissue (not mostly background)"""
    patch_array = np.array(patch)

    # Convert to grayscale and check if mostly white
    gray = np.mean(patch_array, axis=2)
    white_ratio = np.sum(gray > 200) / gray.size

    return white_ratio < threshold


def process_manifest_streaming(manifest_file='manifest.txt', max_files=100):
    """Process WSIs one at a time to save disk space"""

    with open(manifest_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip header

    total_patches = 0
    processed_files = 0

    for line in lines[:max_files]:
        file_id = line.split('\t')[0]
        filename = line.split('\t')[1]

        print(
            f"\n=== Processing {processed_files+1}/{min(max_files, len(lines))} ===")
        print(f"File: {filename}")

        # Download WSI
        wsi_path = download_single_wsi(file_id)
        if not wsi_path:
            print("❌ Download failed")
            continue

        try:
            # Extract patches
            patches = extract_patches(wsi_path)
            print(f"Extracted {patches} patches")

            ########################################################################################
            ################################### INPUT INTO AI MODEL ################################
            ########################################################################################
            # Process patches through AI model
            # features = process_patches_with_ai('patches', your_model)
            # Save features to file instead of keeping patches
            ########################################################################################
            ################################### INPUT INTO AI MODEL ################################
            ########################################################################################

            # Clean up patches after processing
            shutil.rmtree('patches', ignore_errors=True)
            print(f"✅ Processed and cleaned {patches} patches")

        except Exception as e:
            print(f"❌ Processing failed: {e}")

        finally:
            # Clean up WSI to save space
            shutil.rmtree('temp_wsi', ignore_errors=True)

        processed_files += 1

    print(f"\n=== Summary ===")
    print(f"Processed: {processed_files} WSIs")
    print(f"Total patches: {total_patches}")


if __name__ == "__main__":
    max_files = int(
        input("How many WSIs to process? (recommended: 50-200): ") or 100)

    confirm = input(
        f"\nProcess {max_files} WSIs into patches? This saves disk space. (y/n): ")
    if confirm.lower() == 'y':
        process_manifest_streaming('manifest.txt', max_files)
    else:
        print("Cancelled")
