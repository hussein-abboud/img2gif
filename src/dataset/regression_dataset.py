# !filepath: src/dataset/regression_dataset.py

# Purpose:
# This dataset class is tailored for frame-to-frame regression tasks, where the model predicts the final
# frame (frame_14) from the initial frame (frame_0) in a simulation sequence. It loads preprocessed tensor
# files and provides paired tensors for supervised learning.


import torch
from torch.utils.data import Dataset
from pathlib import Path
import csv

class RegressionDataset(Dataset):
    """
    PyTorch Dataset for frame regression.
    Returns: (frame_0_tensor, frame_14_tensor)
    """
    def __init__(self, csv_file: Path):
        """
        Args:
            csv_file (Path): path to a CSV file (train.csv, val.csv, etc.)
        """
        self.samples = []
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(Path(row["pt_path"]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pt_path = self.samples[idx]
        tensor = torch.load(pt_path, weights_only=True)  # shape: (15, 3, 64, 64)
        frame_0 = tensor[0]     # (3, 64, 64)
        frame_14 = tensor[14]   # (3, 64, 64)
        return frame_0, frame_14
