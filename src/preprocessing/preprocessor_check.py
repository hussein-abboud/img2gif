# !filepath: src/preprocessing/check_processed.py

# Purpose:
# This script verifies the integrity and consistency of all processed PyTorch tensor files in the dataset.
# It checks whether each `.pt` file can be loaded and conforms to the expected shape. Any mismatches or
# loading errors are reported, helping identify corrupted or improperly preprocessed files.

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
from src.utils.project_paths import PROCESSED_DATA_DIR as PROCESSED_PATH

EXPECTED_SHAPE = (15, 3, 64, 64)

def check_all(verbose: bool = False):
    if not PROCESSED_PATH.exists():
        print(f"[ERROR] {PROCESSED_PATH} not found.")
        return

    files = list(PROCESSED_PATH.glob("*.pt"))
    if not files:
        print("[ERROR] No processed .pt files found.")
        return

    total = len(files)
    failed = 0

    for file in files:
        try:
            tensor = torch.load(file, weights_only=True)
            if tensor.shape != EXPECTED_SHAPE:
                if verbose:
                    print(f"[❌] {file.name} wrong shape: {tensor.shape}")
                failed += 1
        except Exception as e:
            if verbose:
                print(f"[❌] {file.name} load failed: {e}")
            failed += 1

    print("\n--- Preprocessing Check Summary ---")
    print(f"Checked: {total} | Successful: {total - failed} | Failed: {failed}")
