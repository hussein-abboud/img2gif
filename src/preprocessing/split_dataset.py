# !filepath: src/preprocessing/split_dataset.py

# Purpose:
# This script reads the centralized dataset index CSV and performs a stratified split into training, validation,
# and test sets. It supports using only a fraction of the available data and ensures balanced label distribution
# across the splits. The output CSV files are saved for downstream training and evaluation workflows.

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import csv
import logging
from sklearn.model_selection import train_test_split

from src.utils.project_paths import INDEX_FILE, DATA_DIR as SPLIT_DIR

logging.basicConfig(level=logging.INFO)


def read_csv_dicts(csv_path: Path) -> list[dict]:
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv_dicts(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        logging.warning(f"No data to write for {csv_path.name}")
        return

    with open(csv_path, mode="w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def split_dataset(fraction_of_data_to_be_used: float = 1.0) -> None:
    logging.info("Reading dataset index...")
    all_rows = read_csv_dicts(INDEX_FILE)

    for row in all_rows:
        row["label"] = int(row["label"])

    if not (0 < fraction_of_data_to_be_used <= 1.0):
        raise ValueError("fraction_of_data_to_be_used must be between 0 and 1")

    if fraction_of_data_to_be_used < 1.0:
        labels = [row["label"] for row in all_rows]
        all_rows, _ = train_test_split(
            all_rows,
            train_size=fraction_of_data_to_be_used,
            stratify=labels,
            random_state=42
        )

    labels = [row["label"] for row in all_rows]
    train_rows, temp_rows = train_test_split(
        all_rows,
        test_size=0.3,
        stratify=labels,
        random_state=42
    )

    val_labels = [row["label"] for row in temp_rows]
    val_rows, test_rows = train_test_split(
        temp_rows,
        test_size=0.5,
        stratify=val_labels,
        random_state=42
    )

    write_csv_dicts(SPLIT_DIR / "train.csv", train_rows)
    write_csv_dicts(SPLIT_DIR / "val.csv", val_rows)
    write_csv_dicts(SPLIT_DIR / "test.csv", test_rows)

    logging.info("\n[DONE] Stratified splits created:")
    logging.info(f" → Train: {len(train_rows)}")
    logging.info(f" → Val:   {len(val_rows)}")
    logging.info(f" → Test:  {len(test_rows)}")


if __name__ == "__main__":
    split_dataset(fraction_of_data_to_be_used=0.1)
