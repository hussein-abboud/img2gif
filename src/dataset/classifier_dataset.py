# !filepath: src/dataset/classifier_dataset.py

import torch
from torch.utils.data import Dataset
from pathlib import Path
import csv

class ClassifierDataset(Dataset):
    """
    PyTorch Dataset for loading only the first frame of each tensor
    and its associated binary label (hit=1, miss=0).
    """
    def __init__(self, csv_file: Path):
        """
        Args:
            csv_file (Path): Path to a CSV file (train.csv, val.csv, test.csv)
        """
        self.samples = []
        self._load_csv(csv_file)

    def _load_csv(self, csv_file: Path):
        """Load the CSV and store path-label pairs."""
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pt_path = Path(row.get("pt_path", "")).resolve()
                label = int(row.get("label", 0))
                if pt_path.exists():
                    self.samples.append({
                        "pt_path": pt_path,
                        "label": label
                    })
                else:
                    print(f"[WARNING] File does not exist: {pt_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        tensor = torch.load(sample["pt_path"], weights_only=True)  # shape: (15, 3, 64, 64)
        frame_0 = tensor[0]  # shape: (3, 64, 64)
        label = sample["label"]
        return frame_0, label
