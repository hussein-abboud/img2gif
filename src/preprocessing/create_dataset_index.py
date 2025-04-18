# !filepath: src/preprocessing/create_dataset_index.py

# Purpose:
# This script scans through raw simulation data directories, extracts relevant metadata from JSON files,
# checks for corresponding processed PyTorch tensor files, and compiles an index CSV file. The output
# CSV serves as a centralized reference to locate data tensors and their labels for model training or evaluation.


import json
import csv
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
INDEX_FILE = Path(__file__).resolve().parents[2] / "data" / "dataset_index.csv"


def create_index():
    rows = []
    for run_dir in sorted(RAW_DATA_DIR.glob("run_*")):
        if not run_dir.is_dir():
            continue

        json_path = run_dir / "outcome.json"
        pt_path = PROCESSED_DATA_DIR / f"{run_dir.name}.pt"  # you saved tensors here in preprocessing

        if not json_path.exists() or not pt_path.exists():
            print(f"[SKIP] Missing files in {run_dir.name}")
            continue

        try:
            with open(json_path, "r") as f:
                meta = json.load(f)

            rows.append({
                "id": run_dir.name,
                "pt_path": str(pt_path),
                "label": int(meta["hit"]),  # 1 for hit, 0 for miss
                "color": meta["plane_color"],
                "spawn_side": meta["spawn_side"],
                "spawn_y": meta["spawn_y"]
            })

        except Exception as e:
            print(f"[ERROR] {run_dir.name}: {e}")

    # Save to CSV
    with open(INDEX_FILE, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Dataset index saved to {INDEX_FILE} with {len(rows)} entries.")

if __name__ == "__main__":
    create_index()
