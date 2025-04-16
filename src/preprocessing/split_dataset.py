# !filepath: src/preprocessing/split_dataset.py

import csv
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
INDEX_FILE = ROOT / "data" / "dataset_index.csv"
SPLIT_DIR = ROOT / "data"

def read_csv_dicts(csv_path: Path) -> list[dict]:
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)

def write_csv_dicts(csv_path: Path, rows: list[dict]):
    if not rows:
        print(f"[WARN] No data to write for {csv_path.name}")
        return

    with open(csv_path, mode="w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

def split_dataset():
    print("[INFO] Reading dataset index...")
    all_rows = read_csv_dicts(INDEX_FILE)

    # Convert label field to int for splitting
    for row in all_rows:
        row["label"] = int(row["label"])

    # Stratified split
    labels = [row["label"] for row in all_rows]
    train_rows, temp_rows = train_test_split(all_rows, test_size=0.3, stratify=labels, random_state=42)

    val_labels = [row["label"] for row in temp_rows]
    val_rows, test_rows = train_test_split(temp_rows, test_size=0.5, stratify=val_labels, random_state=42)

    # Save CSVs
    write_csv_dicts(SPLIT_DIR / "train.csv", train_rows)
    write_csv_dicts(SPLIT_DIR / "val.csv", val_rows)
    write_csv_dicts(SPLIT_DIR / "test.csv", test_rows)

    print("\n[DONE] Stratified splits created:")
    print(f" → Train: {len(train_rows)}")
    print(f" → Val:   {len(val_rows)}")
    print(f" → Test:  {len(test_rows)}")

