# !filepath: src/preprocessing/check_processed.py

import torch
from pathlib import Path

PROCESSED_PATH = Path(__file__).resolve().parents[2] / "data" / "processed"
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
