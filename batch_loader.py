#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

def split_and_download_manifest(manifest_file='manifest.txt', batch_size=50):
    """Split manifest and download in batches"""
    
    with open(manifest_file, 'r') as f:
        lines = f.readlines()
    
    header = lines[0]
    data_lines = lines[1:]
    
    total_files = len(data_lines)
    total_batches = (total_files + batch_size - 1) // batch_size
    
    print(f"Total files: {total_files}")
    print(f"Creating {total_batches} batches of {batch_size} files each...")
    
    for batch_num in range(1, total_batches + 1):
        start_idx = (batch_num - 1) * batch_size
        end_idx = min(start_idx + batch_size, total_files)
        batch_lines = data_lines[start_idx:end_idx]
        
        # Create batch manifest
        batch_manifest = f"batch_{batch_num}.txt"
        with open(batch_manifest, 'w') as f:
            f.write(header)
            f.writelines(batch_lines)
        
        # Group batches into fewer directories (200 batches per group)
        batches_per_group = 200
        group_num = (batch_num - 1) // batches_per_group + 1
        batch_dir = f"wsi_group_{group_num}"
        Path(batch_dir).mkdir(exist_ok=True)
        
        print(f"\n=== Downloading batch {batch_num}/{total_batches} ===")
        print(f"Files: {len(batch_lines)}")
        
        # Download batch
        cmd = ['./gdc-client', 'download', '-m', batch_manifest, '-d', f'./{batch_dir}/']
        
        try:
            result = subprocess.run(cmd)
            if result.returncode == 0:
                print(f"✅ Batch {batch_num} completed")
            else:
                print(f"❌ Batch {batch_num} failed")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Cleanup batch manifest
        os.remove(batch_manifest)

if __name__ == "__main__":
    batch_size = int(input("Enter batch size (recommended: 50): ") or 50)
    
    confirm = input(f"\nThis will download ALL files in batches of {batch_size}. Continue? (y/n): ")
    if confirm.lower() == 'y':
        split_and_download_manifest('manifest.txt', batch_size)
    else:
        print("Cancelled")